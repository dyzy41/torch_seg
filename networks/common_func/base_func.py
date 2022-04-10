"""Basic Module for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['_ConvBNPReLU', '_ConvBN', '_BNPReLU', '_ConvBNReLU', '_DepthwiseConv', 'InvertedResidual']


class _ConvBNReLU(nn.Module):
    def __init__(self, in_cannels, num_classannels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_cannels, num_classannels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(num_classannels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _ConvBNPReLU(nn.Module):
    def __init__(self, in_cannels, num_classannels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNPReLU, self).__init__()
        self.conv = nn.Conv2d(in_cannels, num_classannels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(num_classannels)
        self.prelu = nn.PReLU(num_classannels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class _ConvBN(nn.Module):
    def __init__(self, in_cannels, num_classannels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_cannels, num_classannels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(num_classannels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class _BNPReLU(nn.Module):
    def __init__(self, num_classannels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_BNPReLU, self).__init__()
        self.bn = norm_layer(num_classannels)
        self.prelu = nn.PReLU(num_classannels)

    def forward(self, x):
        x = self.bn(x)
        x = self.prelu(x)
        return x


# -----------------------------------------------------------------
#                      For PSPNet
# -----------------------------------------------------------------
class _PSPModule(nn.Module):
    def __init__(self, in_cannels, sizes=(1, 2, 3, 6), **kwargs):
        super(_PSPModule, self).__init__()
        num_classannels = int(in_cannels / 4)
        self.avgpools = nn.ModuleList()
        self.convs = nn.ModuleList()
        for size in sizes:
            self.avgpool.append(nn.AdaptiveAvgPool2d(size))
            self.convs.append(_ConvBNReLU(in_cannels, num_classannels, 1, **kwargs))

    def forward(self, x):
        size = x.size()[2:]
        feats = [x]
        for (avgpool, conv) in enumerate(zip(self.avgpools, self.convs)):
            feats.append(F.interpolate(conv(avgpool(x)), size, mode='bilinear', align_corners=True))
        return torch.cat(feats, dim=1)


# -----------------------------------------------------------------
#                      For MobileNet
# -----------------------------------------------------------------
class _DepthwiseConv(nn.Module):
    """conv_dw in MobileNet"""

    def __init__(self, in_cannels, num_classannels, stride, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_DepthwiseConv, self).__init__()
        self.conv = nn.Sequential(
            _ConvBNReLU(in_cannels, in_cannels, 3, stride, 1, groups=in_cannels, norm_layer=norm_layer),
            _ConvBNReLU(in_cannels, num_classannels, 1, norm_layer=norm_layer))

    def forward(self, x):
        return self.conv(x)


# -----------------------------------------------------------------
#                      For MobileNetV2
# -----------------------------------------------------------------
class InvertedResidual(nn.Module):
    def __init__(self, in_cannels, num_classannels, stride, expand_ratio, norm_layer=nn.BatchNorm2d, **kwargs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_cannels == num_classannels

        layers = list()
        inter_channels = int(round(in_cannels * expand_ratio))
        if expand_ratio != 1:
            # pw
            layers.append(_ConvBNReLU(in_cannels, inter_channels, 1, relu6=True, norm_layer=norm_layer))
        layers.extend([
            # dw
            _ConvBNReLU(inter_channels, inter_channels, 3, stride, 1,
                        groups=inter_channels, relu6=True, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(inter_channels, num_classannels, 1, bias=False),
            norm_layer(num_classannels)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


if __name__ == '__main__':
    x = torch.randn(1, 32, 64, 64)
    model = InvertedResidual(32, 64, 2, 1)
    out = model(x)
