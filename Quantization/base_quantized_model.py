from typing import Union, Dict

import torch
from torch import nn, Tensor

from Quantization.base_quantized_classes import (
    QuantizedModule,
    _set_layer_estimate_ranges,
    _set_layer_estimate_ranges_train,
    _set_layer_learn_ranges,
    _set_layer_fix_ranges,
)
from Quantization.Quantizer import QuantizerBase

class QuantizedModel(nn.Module):
    """
    class represent 1 quantized model voi cac ham can thiet
    """
    def __init__(self, input_size=(1,3,96,96)):
        super().__init__()
        self.input_size = input_size

    def load_state_dict(
        self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]], strict: bool = True
    ):
        """
        This function overwrites the load_state_dict of nn.Module to ensure that quantization
        parameters are loaded correctly for quantized model.

        Copy ca cac quantization_params !!!
        """
        quant_state_dict = {
            k: v for k, v in state_dict.items() if k.endswith("_quant_a") or k.endswith("_quant_w")
        }

        if quant_state_dict:
            super().load_state_dict(quant_state_dict, strict=False)
        else:
            raise ValueError(
                "The quantization states of activations or weights should be "
                "included in the state dict "
            )
        # Pass dummy data qua quantized model de khoi tao shape !!
        device = next(self.parameters()).device
        dummy_input = torch.rand(*self.input_size, device=device)
        with torch.no_grad():
            self.forward(dummy_input)

        # Load state dict
        super().load_state_dict(state_dict, strict)
        
        
# Các hàm duyệt qua từng module của model để gọi hàm tương ứng(hầu hết là bật tắt quantization flag)
    def quantized_weights(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.quantized_weights()

        self.apply(_fn)

    def full_precision_weights(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.full_precision_weights()

        self.apply(_fn)

    def quantized_acts(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.quantized_acts()

        self.apply(_fn)

    def full_precision_acts(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.full_precision_acts()

        self.apply(_fn)

    def quantized(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.quantized()

        self.apply(_fn)

    def full_precision(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.full_precision()

        self.apply(_fn)

    def estimate_ranges(self):
        self.apply(_set_layer_estimate_ranges)

    def estimate_ranges_train(self):
        self.apply(_set_layer_estimate_ranges_train)

# để điều chỉnh quant options 
    def set_quant_state(self, weight_quant, act_quant):
        if act_quant:
            self.quantized_acts()
        else:
            self.full_precision_acts()

        if weight_quant:
            self.quantized_weights()
        else:
            self.full_precision_weights()
    def grad_scaling(self, grad_scaling=True):
        def _fn(module):
            if isinstance(module, QuantizerBase):
                module.grad_scaling = grad_scaling

        self.apply(_fn)
        # Methods for switching quantizer quantization states

    def learn_ranges(self):
        self.apply(_set_layer_learn_ranges)

    def fix_ranges(self):
        self.apply(_set_layer_fix_ranges)