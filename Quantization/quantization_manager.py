from enum import auto 

from torch import nn
from Quantization.Quantizer.quantizer_base import QuantizerBase
from Quantization.range_estimators import range_estimator, CurrentMinMaxEstimator, RunningMinMaxEstimator, MSE_Estimator

class QuantizationManager(nn.Module):
    