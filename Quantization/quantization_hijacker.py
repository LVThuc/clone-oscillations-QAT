import copy

import torch
from torch import nn

from Quantization.Quantizer import QuantizerBase, QuantizationMethod

from Quantization.quantization_manager import QuantizationManager
from Quantization.range_estimators import RangeEstimator, RangeEstimatorBase
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
        
        if self.weight_range_method == RangeEstimator.current
