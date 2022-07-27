""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.common_func.get_backbone import get_model

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_cannels, num_classannels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_cannels, num_classannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classannels, num_classannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_cannels, num_classannels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_cannels, num_classannels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_cannels, num_classannels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_cannels // 2, in_cannels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_cannels, num_classannels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_cannels, num_classannels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_cannels, num_classannels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class FusionBlock(nn.Module):
    def __init__(self):
        super(FusionBlock, self).__init__()
        self.conv_block = DoubleConv(256, 256)

    def forward(self, hx, lx):
        size = lx.size()
        hx2l = F.interpolate(hx, size=(size[2], size[3]), mode='bilinear')
        out = hx2l+lx
        return out


class MFNet50(nn.Module):
    def __init__(self, in_c, num_class, pretrained_path='download', bilinear=True):
        super(MFNet50, self).__init__()
        self.in_c = 3
        self.num_class = num_class
        self.bilinear = bilinear
        self.backbone = get_model('resnet50', checkpoint_path=pretrained_path)
        self.inc = DoubleConv(in_c, 64)

        self.conv_block2 = DoubleConv(256, 256)
        self.conv_block3 = DoubleConv(512, 256)
        self.conv_block4 = DoubleConv(1024, 256)
        self.conv_block5 = DoubleConv(2048, 256)

        self.fb = FusionBlock()
        self.outc = OutConv(256, num_class)
        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

    def forward(self, x):
        size = x.size()
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4]   #64->2048

        x2 = self.conv_block2(x2)
        x3 = self.conv_block3(x3)
        x4 = self.conv_block4(x4)
        x5 = self.conv_block5(x5)

        f4 = self.fb(x5, x4)
        f3 = self.fb(x4, x3)
        f2 = self.fb(x3, x2)

        p3 = self.fb(f4, f3)
        p2 = self.fb(f3, f2)

        m2 = self.fb(p3, p2)
        out = F.interpolate(m2, size=(size[2], size[3]), mode='bilinear')
        out = self.outc(out)
        out = self.activate(out)
        return out


if __name__ == '__main__':
    x = torch.rand(2, 3, 512, 512)
    model = MFNet50(3, 2)
    y = model(x)
    print(y.shape)

