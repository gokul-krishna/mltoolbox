from basic import *
from torch import Tensor
from cv2 import filter2D


kernel_filter = 1 / 12. * np.array([[-1, 2, -2, 2, -1],
                                    [2, -6, 8, -6, 2],
                                    [-2, 8, -12, 8, -2],
                                    [2, -6, 8, -6, 2],
                                    [-1, 2, -2, 2, -1]])


def conv_2d_transform(x):
    """
    Takes torch Tensor as input
    Reference: /reference/kernel.jpg
    """
    assert type(x) == Tensor
    return filter2D(im.numpy().astype(np.float32),
                    -1, kernel_filter)
