from enum import auto 

from sympy import Q
from torch import nn
from Quantization.Quantizer.quantizer_base import QuantizerBase
from Quantization.Quantizer.uniform_quantizers import QuantizationMethod
from Quantization.range_estimators import (
    RangeEstimatorBase, 
    range_estimator, 
    CurrentMinMaxEstimator, 
    RunningMinMaxEstimator, 
    MSE_Estimator, 
    RangeEstimator
)
from utils.utils import BaseEnumOptions


class Qstates(BaseEnumOptions):
    """
    Tập hợp các trạng thái (flags) cho quá trình lượng tử hóa.

    Attributes
    ----------
    estimate_ranges : auto
        Ranges được cập nhật trong cả chế độ train và eval.
    fix_ranges : auto
        Ranges cố định
    learn_ranges : auto
        Learnable Quantization parameters
    """
    estimate_ranges = auto()
    fix_ranges = auto()
    learn_ranges = auto()


class QuantizationManager(nn.Module):
    """
    Quản lý toàn bộ pipeline lượng tử hóa: bao gồm khởi tạo quantizer,
    thiết lập/ước lượng range, và chuyển trạng thái giữa estimate, fix, và learn.

    Parameters
    ----------
    qmethod : QuantizerBase, optional
        Lớp Quantizer sử dụng [Asymmetric/Symmetric]. 
        Mặc định là `QuantizationMethod.Asymmetric.cls`.
    init : RangeEstimatorBase, optional
        Lớp Range Estimator để khởi tạo quá trình lượng tử hóa 
        [CurrentMinMax, RunningMinMax, MSE]. 
        Mặc định là `RangeEstimator.CurrentMinMax.cls`.
    per_channel : bool
    x_min : torch.Tensor or None
    x_max : torch.Tensor or None
    qparams : dict or None, optional
        Các tham số cho quantizer 
    range_estim_params : dict or None, optional
        Các tham số cho range estimator
    Attributes
    ----------
    state : Qstates
        Trạng thái hiện tại của quá trình lượng tử hóa.
    quantizer : QuantizerBase
        Đối tượng quantizer thực hiện lượng tử hóa.
    range_estimator : RangeEstimatorBase or None
        Đối tượng range estimator, dùng để cập nhật giá trị min/max nếu cần.
    """

    def __init__(
        self,
        qmethod: QuantizerBase = QuantizationMethod.Asymmetric.cls,
        init: RangeEstimatorBase = RangeEstimator.CurrentMinMax.cls,
        per_channel: bool = False,
        x_min=None,
        x_max=None,
        qparams=None,
        range_estim_params=None
    ):
        super().__init__()
        self.qmethod = qmethod
        self.init = init
        # Mặc định ở trạng thái estimate_ranges.
        self.state = Qstates.estimate_ranges
        self.per_channel = per_channel
        self.qparams = {} if qparams is None else qparams
        self.range_estim_params = {} if range_estim_params is None else range_estim_params
        self.range_estimator = None

        # Khởi tạo quantizer.
        self.quantizer = self.qmethod(per_channel=self.per_channel, **self.qparams)
        self.quantizer.state = self.state

        # Nếu đã cho sẵn range thì dùng fix_ranges, ngược lại dùng estimator.
        if x_min is not None and x_max is not None:
            self.quantizer.set_quant_range(x_min, x_max)
            self.state = Qstates.fix_ranges
            self.quantizer.state = self.state
        else:
            self.range_estimator = self.init(
                per_channel=self.per_channel, 
                quantizer=self.quantizer, 
                **self.range_estim_params
            )

    @property
    def n_bits(self):

        return self.quantizer.n_bits

    def estimate_ranges(self):
        """
        Chuyển sang chế độ estimate_ranges:
        - Ranges được cập nhật trong cả train và eval.
        """
        self.state = Qstates.estimate_ranges
        self.quantizer.state = self.state

    def learn_ranges(self):
        """
        Chuyển sang chế độ learn_ranges:
        - Các tham số lượng tử hóa (min, max) trở thành tham số học được.
        """
        self.quantizer.make_range_trainable()
        self.state = Qstates.learn_ranges
        self.quantizer.state = self.state

    def forward(self, x):
        """
        Thực hiện lượng tử hóa tensor đầu vào.

        Nếu ở trạng thái estimate_ranges hoặc learn_ranges (và training=True),
        thì range estimator sẽ được gọi để tính min/max mới, sau đó cập nhật cho quantizer.

        Parameters
        ----------
        x : torch.Tensor
            Tensor đầu vào.

        Returns
        -------
        torch.Tensor
            Tensor sau khi đã lượng tử hóa.
        """
        if (self.state == Qstates.estimate_ranges) or (self.state == Qstates.learn_ranges and self.training):
            cur_xmin, cur_xmax = self.range_estimator(x)
            self.quantizer.set_quant_range(cur_xmin, cur_xmax)
        return self.quantizer(x)
