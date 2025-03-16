import torch
import torch.nn as nn
import torch.nn.functional as F

from lenet_5.helpers import Subsampling


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        """
        Layer C1
        Trainable parameters: 6 * 5 * 5 + 6 = 156

        Layer S2
        Trainable parameters: 6 + 6 = 12

        Layer C3
        Trainable parameters: 16 * 5 * 5

        Layer S4 
        
        """
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        
        self.S2 = Subsampling(in_channels=6, kernel_size=2, stride=2)

        self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        self.S4 = Subsampling(in_channels=16, kernel_size=2, stride=2)
        
        self.C5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        
        self.F6 = nn.Linear(120, 84)
        
        self.F7 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input                                         # [N, 1, 28, 28]
        out = F.pad(x, pad=(2, 2, 2, 2), value=0.0)     # [N, 1, 32, 32]
        out = self.C1(out)                              # [N, 6, 28, 28]
        out = F.tanh(self.S2(out))                      # [N, 6, 14, 14]
        out = self.C3(out)                              # [N, 16, 10, 10]
        out = F.tanh(self.S4(out))                      # [N, 16, 5, 5]
        out = F.tanh(self.C5(out))                      # [N, 120, 1, 1]
        out = out.flatten(start_dim=1)                  # [N, 120]
        out = F.tanh(self.F6(out))                      # [N, 84]
        out = self.F7(out)                              # [N, 10] :)
        return out

