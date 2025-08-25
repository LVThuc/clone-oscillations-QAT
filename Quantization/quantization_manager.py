from enum import auto 

from torch import nn
from Quantization.Quantizer.quantizer_base import QuantizerBase
from Quantization.Quantizer.uniform_quantizers import QuantizationMethod
from Quantization.range_estimators import RangeEstimatorBase, range_estimator, CurrentMinMaxEstimator, RunningMinMaxEstimator, MSE_Estimator, RangeEstimator
from utils.utils import BaseEnumOptions

class Qstates(BaseEnumOptions):
    estimate_ranges = auto()  # ranges are updated in eval and train mode
    fix_ranges = auto()  # quantization ranges are fixed for train and eval
    learn_ranges = auto()  # quantization params are nn.Parameters

class QuantizationManager(nn.Module):
    def __init__(
        self,
        qmethod: QuantizerBase = QuantizationMethod.Asymmetric.cls,
        init: RangeEstimatorBase = RangeEstimator.CurrentMinMax.cls,
        per_channel: bool = False,
        x_min = None,
        x_max = None,
        qparams = None,
        range_estim_params = None
    ):
        super().__init__()
        self.qmethod = qmethod
        self.init = init