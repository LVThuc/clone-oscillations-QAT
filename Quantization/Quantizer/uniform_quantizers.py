from ast import Not
import torch
from Quantization.Quantizer.rounding import grad_estimator
from Quantization.Quantizer.quantizer_base import QuantizerBase
from utils.utils import ClassEnumOptions, MethodMap


class AsymmetricUniformQuantizer(QuantizerBase):
	"""
	Asymmetric uniform quantizer
	Zero-point != 0
	Quantization khong doi xung quanh 0

	Parameters
	----------
	n_bits: int
	    Number of bits for quantization.
	scale_domain: str ('log', 'linear) with default='linear'
	    Domain of scale factor
	per_channel: bool
	    If True: allows for per-channel quantization
	discretizer: callable rounding function


	"""

	def __init__(
		self,
		n_bits,
		scale_domain='linear',
		discretizer='STE',  # STE hoặc EWGS
		discretizer_args=(),
		# grad_scaling = False,
		eps=1e-8,
		**kwargs,
	):
		super().__init__(n_bits, **kwargs)

		assert scale_domain in ['linear', 'log'], "scale_domain must be 'linear' or 'log'"
		self.scale_domain = scale_domain
		self.eps = eps
		self.register_buffer('_delta', None)
		self.register_buffer('_zero_float', None)
		if isinstance(discretizer, str):
			self.discretizer = grad_estimator(discretizer, *discretizer_args)
		elif len(discretizer_args) > 0:
			self.discretizer = discretizer(*discretizer_args)
		else:
			self.discretizer = discretizer

	# Debugging property
	@property
	def delta(self):
		if self._delta is not None:
			return self._delta
		else:
			raise NotImplementedError('Quantizer not initialized')

	@property
	def is_initialized(self):
		return self._delta is not None

	@property
	def zero_float(self):
		if self._zero_float is not None:
			return self._zero_float
		else:
			raise NotImplementedError('Quantizer not initialized')

	@property
	def symmetric(self):
		return False

	@property
	def int_min(self):
		# integer grid minimum
		return 0.0

	@property
	def int_max(self):
		# integer grid maximum
		return 2.0**self.n_bits - 1

	@property
	def scale(self):
		if self.scale_domain == 'linear':
			return torch.clamp(self.delta, min=self.eps)
		elif self.scale_domain == 'log':
			# return torch.exp(self.delta)
			raise NotImplementedError('Log scale domain not implemented yet')

	@property
	def zero_point(self):
		zero_point = self.discretizer(self.zero_float)
		zero_point = torch.clamp(zero_point, self.int_min, self.int_max)
		return zero_point

	@property
	def x_max(self):
		return self.scale * (self.int_max - self.zero_point)

	@property
	def x_min(self):
		return self.scale * (self.int_min - self.zero_point)

	def to_integer_forward(self, x_float, *args, **kwargs):
		"""
		Quantized input to integer

		Parameters
		----------
		x_float: torch.Tensor
		    fp32 tensor


		Returns
		-------
		x_int: torch.Tensor
		    Quantized output tensor in integer format.
		"""
		scale = self.scale
		zero_point = self.zero_point
		x_int = self.discretizer(x_float / scale + zero_point)
		x_int = torch.clamp(x_int, self.int_min, self.int_max)
		return x_int

	def forward(self, x_float, *args, **kwargs):
		"""
		Quantizes (quantized to integer and the scales back to original domain)

		Parameters
		----------
		x_float: torch.Tensor
		    fp32 tensor

		Returns
		-------
		x_quant: torch.Tensor
		    Quantized-Dequantized Tensor in fp32
		"""
		if self.per_channel:
			self._adjust_params_per_channel(x_float)

		scale = self.scale
		zero_point = self.zero_point
		x_int = self.to_integer_forward(x_float, *args, **kwargs)
		x_quant = scale * (x_int - zero_point)

		return x_quant

	def _adjust_params_per_channel(self, x):
		if x.ndim != self.delta.ndim:
			new_shape = [-1] + [1] * (len(x.shape) - 1)
			if isinstance(self._delta, torch.nn.Parameter):
				self._delta.data = self._delta.data.view(new_shape)
				if self._zero_float is not None:
					self._zero_float.data = self._zero_float.data.view(new_shape)
			else:
				self._delta = self._delta.view(new_shape)
				if self._zero_float is not None:
					self._zero_float = self._zero_float.view(new_shape)

	def _tensorize_min_max(self, x_min, x_max):
		"""
		Tensorize the min and max values to match the quantization parameters shape.

		Parameters
		----------
		x_min: torch.Tensor
		    Minimum values tensor(1D)
		x_max: torch.Tensor
		    Maximum values tensor(1D)

		Returns
		-------
		x_min: torch.Tensor
		    Tensorized minimum values
		x_max: torch.Tensor
		    Tensorized maximum values
		"""
		if not torch.is_tensor(x_min):
			x_min = torch.tensor(x_min).float()
			x_max = torch.tensor(x_max).float()

		if x_max.ndim > 0 and len(x_max) > 1 and not self.per_channel:
			print(x_max)
			print(self.per_channel)
			raise ValueError(
				'For per-tensor quantization, x_min and x_max must be scalars or 1D tensor'
			)

		x_min = torch.min(x_min, torch.zeros_like(x_min))
		x_max = torch.max(x_max, torch.zeros_like(x_max) * self.eps)
		return x_min, x_max

	def set_quant_range(self, x_min, x_max):
		self.x_min_fp32 = x_min
		self.x_max_fp32 = x_max
		x_min, x_max = self._tensorize_min_max(x_min, x_max)
		delta = (x_max - x_min) / (self.int_max)
		zero_float = (-x_min / delta).detach()

		if isinstance(self._delta, torch.nn.Parameter):
			self._delta.data = delta.detach()
			self._zero_float.data = zero_float
		else:
			self._delta = delta.detach()
			self._zero_float = zero_float

	def make_range_trainable(self):
		# Converts trainable parameters to nn.Parameters
		if self.delta not in self.parameters():
			self._delta = torch.nn.Parameter(self._delta)
			self._zero_float = torch.nn.Parameter(self._zero_float)

	def fix_ranges(self):
		# Removes trainable quantization params from nn.Parameters
		if self.delta in self.parameters():
			_delta = self._delta.data
			_zero_float = self._zero_float.data
			del self._delta  # delete the parameter
			del self._zero_float
			self.register_buffer('_delta', _delta)
			self.register_buffer('_zero_float', _zero_float)


class SymmetricUniformQuantizer(QuantizerBase):
	"""
	Symmetric uniform quantizer
	Zero-point = 0
	Quantization doi xung quanh 0

	Parameters
	----------
	n_bits: int
	    Number of bits for quantization.
	scale_domain: str ('log', 'linear) with default='linear'
	    Domain of scale factor
	per_channel: bool
	    If True: allows for per-channel quantization
	discretizer: callable rounding function
	"""

	def __init__(
		self,
		n_bits,
		scale_domain='linear',
		discretizer='STE',  # STE hoặc EWGS
		discretizer_args=(),
		# grad_scaling = False,
		eps=1e-8,
		**kwargs,
	):
		raise NotImplementedError('SymmetricUniformQuantizer is not implemented yet')


class QuantizationMethod(ClassEnumOptions):
	"""
	Quantization methods for neural network weights.
	Options : [Asymmetric, Symmetric]
	"""

	Asymmetric = MethodMap(AsymmetricUniformQuantizer)
	Symmetric = MethodMap(SymmetricUniformQuantizer)
