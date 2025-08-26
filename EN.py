import torch
import torch.nn as nn
from torchvision.ops.stochastic_depth import StochasticDepth

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, act=True, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
       x = self.silu(self.bn(self.c(x)))
       return x

# Squeeze-and-Excitation Block - Basically reduces parameters.
class SeBlock(nn.Module):
    def __init__(self, in_channels, r):
        super().__init__()
        C = in_channels
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(C, C//r, bias=False)
        self.fc2 = nn.Linear(C//r, C, bias=False)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        f = self.globpool(x)
        f = torch.flatten(f,1)
        f = self.silu(self.fc1(f))
        f = self.sigmoid(self.fc2(f))
        f = f[:,:,None,None]

        scale = x * f
        return scale

# MBConv - Uses ConvBlock but is more efficient - explain in lit review. Explain why you chose this model?
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, exp, r):
        super().__init__()

        # self, in_channels, out_channels,  kernel_size=3, stride=1, model=0, groups=1, act=True, bias=False, downsample=None
        
        exp_channels = in_channels * exp
        self.add = in_channels == out_channels and stride == 1
        self.c1 = ConvBlock(in_channels, exp_channels, 1, 1) if exp > 1 else nn.Identity()
        self.c2 = ConvBlock(exp_channels, exp_channels, kernel_size, stride, groups = exp_channels)
        self.se = SeBlock(exp_channels, r)
        self.c3 = ConvBlock(exp_channels, out_channels, 1, 1, act = False)
        self.sd = StochasticDepth(0.8, "batch")


    def forward(self, x):
        f = self.c1(x)
        f = self.c2(f)
        f = self.se(f)
        f = self.c3(f)

        if self.add:
            f = x + f

        f = self.sd(f)

        return f

# Classfier - Fully connected layer. Typical ANN.
class Classifier(nn.Module):
    def __init__(self, in_channels, classes, p):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_channels, classes)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.dropout(self.pool(x))
        x = torch.flatten(x, 1)
        return self.fc(x)

# EfficientNet
class EfficientNet(nn.Module):
    def __init__(self, model_config : str, in_channels : int = 3, classes : int = 1000, show : str = False):
        super().__init__()
        self.show = show # if True print the output dim between layers 
        config = Config()
        stages = config.stages
        phis = config.phis[model_config]

        phi, res, p = phis
        self._calculate_coef(phi)

        self.net = nn.ModuleList([])
        self.channels = []
        f, c, l, k, s, exp = stages[0]
        self._add_layer(3, f, c, l, k, s)

        for i in range(1, len(stages)-1):
            if i == 1:
                r = 4
            else:
                r = 24

            f, c, l, k, s, exp = stages[i]
            self._add_layer(self.channels[-1], f, c, l, k, s, exp, r)

        f, c, l, k, s, exp = stages[-1]
        self._add_layer(self.channels[-1], f, c, l, k, s)
        self.net.append(Classifier(self.channels[-1], classes, p))

    def forward(self, x):
        i = 1
        for F in self.net:
            x = F(x)
        return x

    def _add_layer(self, in_channels, f, c, l, k, s, *args):
        c, l = self._update_feat(c, l)
        if l == 1:
            self.net.append(f(in_channels, c, k, s, *args))
        else:
            self.net.append(f(in_channels, c, k, 1, *args))
            for _ in range(l-2):
                self.net.append(f(c, c, k, 1, *args))                
        
            self.net.append(f(c, c, k, s, *args))

        self.channels.append(c)
                
    def _calculate_coef(self, phi, alpha=1.2, beta=1.1):
        self.d = alpha**phi
        self.w = beta**phi

    def _update_feat(self, c, l):
        return int(c * self.w), int(l * self.d)

class Config:
    stages = [
            # [Operator(f), Channels(c), Layers(l), Kernel(k), Stride(s), Expansion(exp)]
            [ConvBlock, 32, 1, 3, 2, 1], 
            [MBConv, 16, 1, 3, 1, 1],
            [MBConv, 24, 2, 3, 2, 6],
            [MBConv, 40, 2, 5, 2, 6],
            [MBConv, 80, 3, 3, 2, 6],
            [MBConv, 112, 3, 5, 1, 6],
            [MBConv, 192, 4, 5, 2, 6],
            [MBConv, 320, 1, 3, 1, 6],
            [ConvBlock, 1280, 1, 1, 1, 0]
    ]

    phis = {"B0" : (0, 224, 0.2)}


    