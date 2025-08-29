import copy
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.pooling import _AdaptiveAvgPoolNd, _AvgPoolNd

from Quantization.base_quantized_classes import QuantizedActivation, QuantizedModule
from Quantization.quantization_hijacker import QuantizationHijacker, activations_set
from Quantization.quantization_manager import QuantizationManager
from Quantization.quantized_folded_bn import BNFusedHijacker
# May cai nay init theo MRO, ko can rewrite init cung duoc
class QuantConv1d(QuantizationHijacker, nn.Conv1d):
    def run_forward(self, x, weight, bias, offsets = None):
        return F.conv1d(
            x.contiguous(),
            weight.contiguous(),
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

class QuantConv(QuantizationHijacker, nn.Conv2d):
    def run_forward(self, x, weight, bias, offsets = None):
        return F.conv2d(
            x.contiguous(),
            weight.contiguous(),
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        
# Chi overwrite lai mot so ham can thiet
class QuantConvTransposeBase(QuantizationHijacker):
    def quantize_weight(self, weight):
        # vi spaghetti code nen phai swap vi tri cua in va out(thay vi out va in) de fit voi logic trong range_estimator voi quantizers
        if self.per_channel_weights:
            weights = weights.transpose(1,0).contiguous()
        weight = self.weight_quantizer(weights)
        if self.per_channel_weights:
            weight = weight.transpose(1,0).contiguous()
        return weight

class QuantLinear(QuantizationHijacker, nn.Linear):
    def run_forward(self, x, weight, bias, offsets=None):
        return F.linear(x.contiguous(), weight.contiguous(), bias=bias)
    
#Batch-Normalized-Quantized-Conv1d
class BNQConv1d(BNFusedHijacker, nn.Conv1d):
    def run_forward()