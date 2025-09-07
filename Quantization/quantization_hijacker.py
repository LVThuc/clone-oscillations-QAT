import copy

import torch.nn as nn


from Quantization.quantization_manager import QuantizationManager
from Quantization.range_estimators import RangeEstimator
from Quantization.base_quantized_classes import QuantizedModule

from timm.models.layers.activations import Swish, HardSwish, HardSigmoid
from timm.models.layers.activations_me import SwishMe, HardSwishMe, HardSigmoidMe

activations_set = [
	nn.ReLU,
	nn.ReLU6,
	nn.Hardtanh,
	nn.Sigmoid,
	nn.Tanh,
	nn.GELU,
	nn.PReLU,
	Swish,
	SwishMe,
	HardSwish,
	HardSwishMe,
	HardSigmoid,
	HardSigmoidMe,
]


class QuantizationHijacker(QuantizedModule):
	"""
	Hijack các layer ở forward pass để chèn các hàm lượng tử hóa vào weight và activation.
	"""

	def __init__(self, *args, activation: nn.Module = None, **kwargs):
		super().__init__(*args, **kwargs)
		if activation:
			assert isinstance(activation, tuple(activations_set)), str(activation)

		self.activation_function = copy.deepcopy(activation) if activation else None

		self.activation_quantizer = QuantizationManager(
			qmethod=self.act_method,
			init=self.act_range_method,
			qparams=self.act_qparams,
			range_estim_params=self.act_range_options,
		)

		if self.weight_range_method == RangeEstimator.CurrentMinMax:
			weight_init_params = dict(percentile=self.percentile)
		else:
			weight_init_params = self.weight_range_options

		self.weight_quantizer = QuantizationManager(
			qmethod=self.method,
			init=self.weight_range_method,
			qparams=self.weight_qparams,
			per_channel=self.per_channel_weights,
			range_estim_params=weight_init_params,
		)

	# Ham này quantize weight nè .
	def get_params(self):
		weight, bias = self.get_weight_bias()

		if self._quant_w:
			weight = self.quantize_weights(weight)

		return weight, bias

	def quantize_weights(self, weights):
		return self.weight_quantizer(weights)

	def run_forward(self, x, weight, bias, offsets=None):
		# Performs the actual linear operation of the layer
		raise NotImplementedError()

	def get_weight_bias(self):
		bias = None
		if hasattr(self, 'bias'):
			bias = self.bias
		return self.weight, bias

	def forward(self, x, offset=None):
		# Truong hop quantize input
		if self.quantize_input and self._quant_a:
			x = self.activation_quantizer(x)
		weight, bias = self.get_params()

		res = self.run_forward(x, weight, bias, offsets=offset)

		if self.activation_function is not None:
			res = self.activation_function(res)
		# Truong hop ko quantize input
		if not self.quantize_input and self._quant_a:
			res = self.activation_quantizer(res)
		return res
