from enum import auto 

from torch import nn
from Quantization.Quantizer.quantizer_base import QuantizerBase
from Quantization.Quantizer.uniform_quantizers import QuantizationMethod
from Quantization.range_estimators import RangeEstimatorBase, range_estimator, CurrentMinMaxEstimator, RunningMinMaxEstimator, MSE_Estimator, RangeEstimator
from utils.utils import BaseEnumOptions

class Qstates(BaseEnumOptions):
    """
    Tập hợp flag cho quá trình lượng tử hóa
    """
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
        # bat flag estimate_ranges.
        self.state = Qstates.estimate_ranges
        self.per_channel = per_channel
        self.qparams = {} if qparams is None else qparams
        self.range_estim_params = {} if range_estim_params is None else range_estim_params
        self.range_estimator = None
        #Init quantizer
        self.quantizer = self.qmethod(per_channel=self.per_channel, **qparams)
        self.quantizer.state = self.state

        if x_min is not None and x_max is not None:
            self.quantizer.set_quant_range(x_min, x_max)
            self.state = Qstates.fix_ranges
            self.quantizer.state = self.state
        else: 
            self.range_estimator = self.init(
                per_channel = self.per_channel, quantizer = self.quantizer, **self.range_estim_params 
            )
    @property
    def n_bits(self)
        return self.quantizer.n_bits
    
    def estimate_ranges(self, x):
        self.state = Qstates.estimate_ranges
        self.quantizer.state = self.state

    def learn_ranges(self):
        self.quantizer.make_range_trainable()
        self.state = Qstates.learn_ranges
        self.quantizer.state = self.state
    
