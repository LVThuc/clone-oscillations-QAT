import numpy as np
from enum import Flag, auto
from functools import partial
from collections import namedtuple
def to_numpy(tensor):
    """
    Helper function that turns the given tensor into a numpy array

    Parameters
    ----------
    tensor : torch.Tensor

    Returns
    -------
    tensor : float or np.array

    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "is_cuda"):
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()
    if hasattr(tensor, "detach"):
        return tensor.detach().numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()

    return np.array(tensor)

class BaseEnumOptions(Flag):
    def __str__(self):
        return self.name

    @classmethod
    def list_names(cls):
        return [m.name for m in cls]


class ClassEnumOptions(BaseEnumOptions):
    @property
    def cls(self):
        return self.value.cls

    def __call__(self, *args, **kwargs):
        return self.value.cls(*args, **kwargs)

MethodMap = partial(namedtuple("MethodMap", ["value", "cls"]), auto())
"""
Hướng dẫn sử dụng
-----------------

Mục đích:
    Cung cấp một cách tiện lợi để định nghĩa Enum mà mỗi phần tử
    vừa có ID riêng (tự động sinh bởi auto()), vừa ánh xạ đến một class cụ thể.
    Nhờ đó, ta có thể tạo instance từ enum trực tiếp thay vì phải dùng if/else.

Thành phần:
    - BaseEnumOptions: mở rộng từ enum.Flag, thêm __str__ để in tên Enum,
      và list_names() để lấy toàn bộ tên enum dưới dạng list.
    - ClassEnumOptions: kế thừa BaseEnumOptions, cho phép:
        * .cls → trả về class được gán trong MethodMap.
        * gọi trực tiếp enum như một factory để tạo instance.
    - MethodMap: một namedtuple(value, cls) với 'value' sinh tự động (auto()).
      Thường được dùng làm value trong Enum.

Ví dụ sử dụng:
    >>> class Models(ClassEnumOptions):
    ...     LINEAR = MethodMap(int)
    ...     CONV   = MethodMap(list)

    # Truy cập class được gắn
    >>> Models.LINEAR.cls
    <class 'int'>

    # Tạo instance trực tiếp từ enum
    >>> Models.CONV()
    === Models.CONV.cls()

    # Lấy danh sách tên enum
    >>> Models.list_names()
    ['LINEAR', 'CONV']
"""