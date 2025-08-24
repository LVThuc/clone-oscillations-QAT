import copy
from enum import auto

import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize_scalar

from utils import to_numpy

class RangeEstimatorBase(nn.Module):
    def __init__(self, per_channel=False, quantizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("current_xmin", None)
        self.register_buffer("current_xmax", None)
        self.per_channel = per_channel
        self.quantizer = quantizer

    def forward(self, x):
        raise NotImplementedError()

    def reset(self):
        self.current_xmin = None
        self.current_xmax = None

    def __repr__(self):
        lines = self.extra_repr().split("\n")
        extra_str = lines[0] if len(lines) == 1 else "\n  " + "\n  ".join(lines) + "\n"

        return self._get_name() + "(" + extra_str + ")"
    
class CurrentMinMaxEstimator(RangeEstimatorBase):
    """
    Estimates the current minimum and maximum values of the input tensor.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        if self.per_channel:
            x = x.view(x.shape[0],-1) # Cái code này nhìn hơi lạ, per_channel nma nó lại gộp cả tensor vào?
        self.current_xmin = x.min(-1)[0].detach() if self.per_channel else x.min().detach()
        self.current_xmax = x.max(-1)[0].detach() if self.per_channel else x.max().detach()
        return self.current_xmin, self.current_xmax