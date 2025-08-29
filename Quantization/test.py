from sympy import Range
from Quantization.base_quantized_classes import QuantizedModule, QuantizedActivation

import torch
import torch.nn as nn

x = torch.randn(4, 8)

N = QuantizedActivation(n_bits = 2)
print(N)
y_fp32 = N(x)
print("Output (FP32):", y_fp32.shape)
print(x)
assert(torch.equal(y_fp32, x))

N.quantized_acts()
print("Quantizer status sau khi bật:", N.get_quantizer_status())
y_quant = N(x)
print(y_quant)
print("Output (Quantized):", y_quant.shape)
N.full_precision_acts()
print("Quantizer status sau khi tắt:", N.get_quantizer_status())
print(N(x))
N.learn_ranges()
N.fix_ranges()
N.estimate_ranges()
N.estimate_ranges_train()
