import PIL
import torch
from .basic import *
from torchvision.transforms import Lambda
from cv2 import filter2D


kernel_filter = 1 / 12. * np.array([[-1, 2, -2, 2, -1],
                                    [2, -6, 8, -6, 2],
                                    [-2, 8, -12, 8, -2],
                                    [2, -6, 8, -6, 2],
                                    [-1, 2, -2, 2, -1]])


def conv_2d_filter(x):
    """
    input: PIL image
    output: torch Tensor
    Reference: /reference/kernel.jpg
    transpose operations because pytorch has 3xHxW
    whereas cv2 used HxWx3
    Can't normalize this
    """
    assert type(x) == PIL.Image.Image
    x = filter2D(np.array(x).astype(np.float32), -1, kernel_filter)
    return torch.from_numpy(x.transpose((2, 0, 1)))


Conv2dFilter = Lambda(lambda x: conv_2d_filter(x))
