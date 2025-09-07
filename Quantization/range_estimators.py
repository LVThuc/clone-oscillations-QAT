import copy

import numpy as np
import torch
from torch import nn

from utils.utils import to_numpy, MethodMap, ClassEnumOptions


class RangeEstimatorBase(nn.Module):
	def __init__(self, per_channel=False, quantizer=None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.register_buffer('current_xmin', None)
		self.register_buffer('current_xmax', None)
		self.per_channel = per_channel
		self.quantizer = quantizer

	def forward(self, x):
		raise NotImplementedError()

	def reset(self):
		self.current_xmin = None
		self.current_xmax = None

	def __repr__(self):
		lines = self.extra_repr().split('\n')
		extra_str = lines[0] if len(lines) == 1 else '\n  ' + '\n  '.join(lines) + '\n'

		return self._get_name() + '(' + extra_str + ')'


class CurrentMinMaxEstimator(RangeEstimatorBase):
	"""
	Estimates the current minimum and maximum values of the input tensor.
	"""

	def __init__(self, percentile=None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.percentile = percentile

	def forward(self, x):
		if self.per_channel:
			x = x.view(
				x.shape[0], -1
			)  # Cái code này nhìn hơi lạ, per_channel nma nó lại gộp cả tensor vào?
		if self.percentile:
			axis = -1 if self.per_channel else None
			data_np = to_numpy(x)
			x_min, x_max = np.percentile(
				data_np, (self.percentile, 100 - self.percentile), axis=axis
			)
			self.current_xmin = torch.tensor(x_min).to(x.device)
			self.current_xmax = torch.tensor(x_max).to(x.device)
		else:
			self.current_xmin = x.min(-1)[0].detach() if self.per_channel else x.min().detach()
			self.current_xmax = x.max(-1)[0].detach() if self.per_channel else x.max().detach()
		return self.current_xmin, self.current_xmax


class RunningMinMaxEstimator(RangeEstimatorBase):
	"""
	Estimates the running minimum and maximum values of the input tensor using EMA.

	Parameters
	----------
	momentum: float
	    The momentum factor for the running estimates.
	"""

	def __init__(self, momentum=0.9, *args, **kwargs):
		self.momentum = momentum
		super().__init__(*args, **kwargs)

	def forward(self, x):
		if self.per_channel:
			x_flattened = x.view(x.shape[0], -1)
			x_min = x_flattened.min(-1)[0].detach()
			x_max = x_flattened.max(-1)[0].detach()
		else:
			x_min = torch.min(x).detach()
			x_max = torch.max(x).detach()

		if self.current_xmin is None:
			self.current_xmin = x_min
			self.current_xmax = x_max
		else:
			self.current_xmin = self.momentum * self.current_xmin + (1 - self.momentum) * x_min
			self.current_xmax = self.momentum * self.current_xmax + (1 - self.momentum) * x_max

		return self.current_xmin, self.current_xmax


class MSE_Estimator(RangeEstimatorBase):
	""" """

	def __init__(self, num_candidates=100, range_margin=0.5, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.num_candidates = num_candidates
		self.loss_array = None
		self.max_pos_thr = None
		self.max_neg_thr = None
		self.max_search_range = None
		self.one_sided_dist = None
		self.range_margin = range_margin
		if self.quantizer is None:
			raise NotImplementedError('Quantizer must be provided for MSE_Estimator')
		self.max_int_skew = (2**self.quantizer.n_bits) // 4  # for asymmetric quantization

	def quantize(self, x_float, x_min=None, x_max=None):
		temp_q = copy.deepcopy(self.quantizer)
		temp_q.per_channel = False
		if x_min or x_max:
			temp_q.set_quant_range(x_min, x_max)
		return temp_q(x_float)

	def loss_fx(self, data, neg_thr, pos_thr, per_channel_loss=False):
		"""
		Compute the mean squared error (MSE)

		Parameters
		----------
		data : torch.Tensor
		neg_thr : float or torch.Tensor
		pos_thr : float or torch.Tensor
		per_channel_loss : bool, optional (default=False)

		Returns
		-------
		numpy.ndarray or float
		    - If per_channel_loss=True: 1D numpy array of per-channel MSE losses.
		    - If per_channel_loss=False: scalar total MSE loss over the tensor.
		"""
		y = self.quantize(data, x_min=neg_thr, x_max=pos_thr)
		temp_sum = torch.sum(((data - y) ** 2).view(len(data), -1), dim=-1)
		if per_channel_loss:
			return to_numpy(temp_sum)
		else:
			return to_numpy(torch.sum(temp_sum))

	@property
	def step_size(self):
		if self.one_sided_dist is None:
			raise NotImplementedError('NoDataPassedError')

		return self.max_search_range / self.num_candidates

	@property
	def optimization_method(self):
		if self.one_sided_dist is None:
			raise NotImplementedError('NoDataPassedError')

		if self.one_sided_dist or self.quantizer.symmetric:
			# 1-D grid search
			return self._perform_1D_search
		else:
			# 2-D grid_search
			return self._perform_2D_search

	def _define_search_range(self, data):
		self.channel_groups = len(data) if self.per_channel else 1
		self.current_xmax = torch.zeros(self.channel_groups, device=data.device)
		self.current_xmin = torch.zeros(self.channel_groups, device=data.device)

		if self.one_sided_dist or self.quantizer.symmetric:
			# 1D search space
			self.loss_array = np.zeros(
				(self.channel_groups, self.num_candidates + 1)
			)  # 1D search space
			self.loss_array[:, 0] = np.inf  # exclude interval_start=interval_finish
			# Defining the search range for clipping thresholds
			self.max_pos_thr = max(abs(float(data.min())), float(data.max())) + self.range_margin
			self.max_neg_thr = -self.max_pos_thr
			self.max_search_range = self.max_pos_thr
		else:
			# 2D search space (3rd and 4th index correspond to asymmetry where fourth
			# index represents whether the skew is positive (0) or negative (1))
			self.loss_array = np.zeros(
				[self.channel_groups, self.num_candidates + 1, self.max_int_skew, 2]
			)  # 2D search space
			self.loss_array[:, 0, :, :] = np.inf  # exclude interval_start=interval_finish
			# Define the search range for clipping thresholds in asymmetric case
			self.max_pos_thr = float(data.max()) + self.range_margin
			self.max_neg_thr = float(data.min()) - self.range_margin
			self.max_search_range = max(abs(self.max_pos_thr), abs(self.max_neg_thr))

	def _perform_1D_search(self, data):
		"""
		Grid search through all candidate quantizers in 1D to find the best
		The loss is accumulated over all batches without any momentum
		:param data: input tensor
		"""
		for cand_index in range(1, self.num_candidates + 1):
			neg_thr = 0 if self.one_sided_dist else -self.step_size * cand_index
			pos_thr = self.step_size * cand_index

			self.loss_array[:, cand_index] += self.loss_fx(
				data, neg_thr, pos_thr, per_channel_loss=self.per_channel
			)
			# find the best clipping thresholds
		min_cand = self.loss_array.argmin(axis=1)
		xmin = (
			np.zeros(self.channel_groups) if self.one_sided_dist else -self.step_size * min_cand
		).astype(np.single)
		xmax = (self.step_size * min_cand).astype(np.single)
		self.current_xmax = torch.tensor(xmax).to(device=data.device)
		self.current_xmin = torch.tensor(xmin).to(device=data.device)

	def _perform_2D_search(self, data):
		"""
		Grid search through all candidate quantizers in 1D to find the best
		The loss is accumulated over all batches without any momentum
		Parameters
		----------
		data:   PyTorch Tensor
		Returns
		-------

		"""
		for cand_index in range(1, self.num_candidates + 1):
			# defining the symmetric quantization range
			temp_start = -self.step_size * cand_index
			temp_finish = self.step_size * cand_index
			temp_delta = float(temp_finish - temp_start) / (2**self.quantizer.n_bits - 1)
			for shift in range(self.max_int_skew):
				for reverse in range(2):
					# introducing asymmetry in the quantization range
					skew = ((-1) ** reverse) * shift * temp_delta
					neg_thr = max(temp_start + skew, self.max_neg_thr)
					pos_thr = min(temp_finish + skew, self.max_pos_thr)

					self.loss_array[:, cand_index, shift, reverse] += self.loss_fx(
						data, neg_thr, pos_thr, per_channel_loss=self.per_channel
					)

		for channel_index in range(self.channel_groups):
			min_cand, min_shift, min_reverse = np.unravel_index(
				np.argmin(self.loss_array[channel_index], axis=None),
				self.loss_array[channel_index].shape,
			)
			min_interval_start = -self.step_size * min_cand
			min_interval_finish = self.step_size * min_cand
			min_delta = float(min_interval_finish - min_interval_start) / (
				2**self.quantizer.n_bits - 1
			)
			min_skew = ((-1) ** min_reverse) * min_shift * min_delta
			xmin = max(min_interval_start + min_skew, self.max_neg_thr)
			xmax = min(min_interval_finish + min_skew, self.max_pos_thr)

			self.current_xmin[channel_index] = torch.tensor(xmin).to(device=data.device)
			self.current_xmax[channel_index] = torch.tensor(xmax).to(device=data.device)

	def forward(self, data):
		if self.loss_array is None:
			if self.one_sided_dist is None:
				self.one_sided_dist = bool((data.min() >= 0).item())

			self._define_search_range(data)

		self.optimization_method(data)

		return self.current_xmin, self.current_xmax

	def reset(self):
		super().reset()
		self.loss_array = None

	def extra_repr(self):
		repr = ' ,num_candidates={}'.format(self.num_candidates)
		return repr


def range_estimator(method: str, *args, **kwargs) -> RangeEstimatorBase:
	"""
	Factory function to create a range estimator instance.

	Parameters
	----------
	method : str
	    The range estimation method to use. Options:
	    - "current_minmax": Uses CurrentMinMaxEstimator
	    - "running_minmax": Uses RunningMinMaxEstimator (EMA)
	    - "mse": Uses MSE_Estimator (grid search with MSE minimization)
	*args, **kwargs
	    Additional arguments passed to the estimator constructors.

	Returns
	-------
	RangeEstimatorBase
	    An instance of the requested range estimator.

	Raises
	------
	ValueError
	    If the method is not recognized.
	"""
	method = method.lower()
	if method == 'current_minmax':
		return CurrentMinMaxEstimator(*args, **kwargs)
	elif method == 'running_minmax':
		return RunningMinMaxEstimator(*args, **kwargs)
	elif method == 'mse':
		return MSE_Estimator(*args, **kwargs)
	else:
		raise ValueError(f'Unknown range estimator method: {method}')


class RangeEstimator(ClassEnumOptions):
	"""
	Base class for range estimators.
	options: [CurrentMinMax, RunningMinMax, MSE]
	"""

	CurrentMinMax = MethodMap(CurrentMinMaxEstimator)
	RunningMinMax = MethodMap(RunningMinMaxEstimator)
	MSE = MethodMap(MSE_Estimator)
