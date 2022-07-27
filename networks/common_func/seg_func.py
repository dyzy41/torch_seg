import torch
import torch.nn as nn
import torch.nn.functional as F

class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


class MultiHead(nn.Module):
    def __init__(self, in_channels, num_class, size=(512, 512), norm_layer=nn.BatchNorm2d):
        super(MultiHead, self).__init__()
        self.size = size
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            norm_layer(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_class, 1, 1, 0)
        )

    def forward(self, x):
        x = self.block(x)
        x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=True)
        return x


class SingleHead(nn.Module):
    def __init__(self, in_channels, num_class, size=(512, 512), norm_layer=nn.BatchNorm2d):
        super(SingleHead, self).__init__()
        self.size = size
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, num_class, 1, 1, 0)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block(x)
        x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=True)
        x = self.sigmoid(x)
        return x