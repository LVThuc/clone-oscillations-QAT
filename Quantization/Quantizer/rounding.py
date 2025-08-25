from torch import nn 
import torch
from torch.autograd import Function
from functools import partial
# 2 Method of estimating gradients in QAT


class STERounding(Function):
    """
    Straight-Through Estimator (STE).
    - Forward: làm tròn giá trị.
    - Backward: cho gradient đi qua y như cũ.
    """
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)
    
    @staticmethod
    def backward(ctx, output_grad):
        # Trả về gradient đầu vào không thay đổi
        return output_grad


class EWGSRounding(Function):
    """
    Element-wise Gradient Scaling (EWGS).
    - Forward: làm tròn giá trị và lưu lại sai số làm tròn (input - round(input)).
    - Backward: scale gradient dựa trên sai số và một siêu tham số (hyperparameter).
    """
    @staticmethod
    def forward(ctx, input, scaling_factor):
        input_int = torch.round(input)
        # Lưu sai số làm tròn và hệ số scale cho backward pass
        ctx.save_for_backward(input - input_int)
        ctx._scaling_factor = scaling_factor
        return input_int

    @staticmethod
    def backward(ctx, g):
        diff = ctx.saved_tensors[0]
        delta = ctx._scaling_factor 
        # Scale gradient gốc 'g'
        scale = 1 + delta * torch.sign(g) * diff
        # Trả về gradient đã scale. `None` cho các đối số không cần gradient (scaling_factor)
        return g * scale, None

def grad_estimator(method: str, *args, **kwargs):
    """
    Hàm Factory: Trả về một hàm làm tròn (discretizer) có thể gọi được.
    - method: "STE" hoặc "EWGS"
    - args/kwargs: các tham số phụ cho discretizer (ví dụ: scaling_factor cho EWGS)
    """
    if method == "STE":
        return STERounding.apply
    elif method == "EWGS":
        # Sử dụng partial để tạo một hàm mới với scaling_factor đã được cung cấp sẵn
        # Nếu không có args, mặc định scaling_factor = 0.2
        scaling_factor = args[0] if args else 0.2
        return  lambda x: EWGSRounding.apply(x, scaling_factor)
    else:
        raise ValueError(f"Unknown method: {method}")