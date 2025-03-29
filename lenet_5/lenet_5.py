"""
Author: Raphael Senn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        """
        
        Layer C1
        -----------------------------------------------------------------------
        Trainable params:   (6 * 5 * 5) + 6                         = 156
        Connections:        (28 * 28 * 6 * 5 * 5) + (6 * 28 * 28)   = 122'304
        
        Layer S2
        -----------------------------------------------------------------------
        Trainable params:   0                                       = 0 
        Connections:        (2 * 2 * 6 * 14 * 14)                   = 4'704

        Layer C3
        -----------------------------------------------------------------------
        Trainable parameters: (6 * 16 * 5 * 5) + 16                 = 2'416

        Connections:      (10 * 10 * 16 * 5 * 5) + 16 * 10 * 10     = 41'600

        Layer S4
        -----------------------------------------------------------------------
        Trainable params:   0                                       = 0 
        Connections:        (2 * 2 * 16 * 5 * 5)                    = 1'600

        Layer C5
        -----------------------------------------------------------------------
        Trainable params:   (120 * 16 * 5 * 5) + 120                = 48'120
        Connections:        (120 * 16 * 5 * 5) + 120                = 48'120

        Layer F6
        -----------------------------------------------------------------------
        Trainable params:   (120 * 84) + 84                         = 10'164
        Connections:        (120 * 84) + 84                         = 10'164

        Output layer
        -----------------------------------------------------------------------
        Trainable params:   (84 * 10) + 10                          = 850
        Connections:        (84 * 10) + 10                          = 850

        -----------------------------------------------------------------------
        (+)                                                        
        Trainable params:                                           = 61'706
        Connections:                                                = 229'342
        """

        self.C1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            bias=True)

        self.S2 = nn.AvgPool2d(
            kernel_size=2,
            stride=2)

        self.C3 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            bias=True,
            kernel_size=5)

        self.S4 = nn.AvgPool2d(
            kernel_size=2,
            stride=2)

        self.C5 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5,
            bias=True)
        
        self.F6 = nn.Linear(
            in_features=120,
            out_features=84,
            bias=True)
        
        self.F7 = nn.Linear(
            in_features=84,
            out_features=10,
            bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.pad(x, pad=(2, 2, 2, 2), value=0.0)     # [N, 1, 32, 32]
        out = self.C1(out)                              # [N, 6, 28, 28]
        out = F.tanh(out)                               # activation
        
        out = self.S2(out)                              # [N, 6, 14, 14]
        out = F.tanh(out)                               # activation
        
        out = self.C3(out)                              # [N, 16, 10, 10]
        out = F.tanh(out)                               # activation

        out = self.S4(out)                              # [N, 16, 5, 5]
        out = F.tanh(out)                               # activation
        
        out = self.C5(out)                              # [N, 120, 1, 1]
        out = F.tanh(out)                               # activation

        out = out.flatten(start_dim=1)                  # [N, 120]
        
        out = self.F6(out)                              # [N, 84]
        out = F.tanh(out)                               # activation
        
        out = self.F7(out)                              # [N, 10] :)
        return out