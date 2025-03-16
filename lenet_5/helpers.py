import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2DSpecialGrouping(nn.Module):
    """
    Grouping the feature maps, like in the paper described.
    
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=4)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=4)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=4)


    def forward(self):
        pass


class Subsampling(nn.Module):
    """
    Subsampling like in the paper described.
    The inputs are added together, multiplied by a learnable coeficient and a bias is added. 

    Assuming subsampling from a 2D convolution with n_out channels, this results in:
    n_out + n_out = 2*n_out learnable parameters
    """ 

    def __init__(self, in_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()

        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride

        self.kernel = torch.ones((in_channels, in_channels, self.kernel_size, self.kernel_size))
        self.coef = nn.Parameter(torch.rand((in_channels, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((in_channels, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.conv2d(input=x, weight=self.kernel, stride=self.stride)
        out = out * self.coef + self.bias
        return out