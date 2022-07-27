""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.common_func.get_backbone import get_model
from networks.common_func.non_local import NonLocalBlock
from einops import rearrange


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


class BANet(nn.Module):
    def __init__(self, in_c, num_class, pretrained_path=None, bilinear=True):
        super(BANet, self).__init__()
        self.in_c = 3
        self.num_class = num_class
        self.bilinear = bilinear
        self.backbone = get_model('resnet50', checkpoint_path=pretrained_path)

        self.inc = DoubleConv(in_c, 64)

        self.c5 = DoubleConv(2048, 512)
        self.c4 = DoubleConv(1024, 512)
        self.c3 = DoubleConv(512, 512)
        self.c2 = DoubleConv(256, 512)
        self.c11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.c33 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.c33_block = nn.Sequential(self.c33,
                                       nn.BatchNorm2d(512),
                                       nn.ReLU())
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.nlb = NonLocalBlock(512 * 4)

        self.outc = OutConv(512, num_class)
        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

    def AttFusion(self, xh, xl):
        xh = F.interpolate(xh, scale_factor=2, mode='bilinear', align_corners=True)
        xh = self.c11(xh)
        xl_bk = self.c11(xl)
        x_sum = xl_bk + xh
        x_sum = self.relu(x_sum)
        x_sum = self.c11(x_sum)
        x_sum = self.sigmoid(x_sum)
        x_fusion = torch.mul(x_sum, xl)
        return x_fusion

    def BlockAtt(self, x, factor):
        x_div = rearrange(x, 'b c (h k1) (w k2) -> b (c k1 k2) h w', k1=factor, k2=factor)
        x_div = self.nlb(x_div)
        x_total = rearrange(x_div, 'b (c k1 k2) h w -> b c (h k1) (w k2)', k1=factor, k2=factor)
        x_out = x + x_total
        x_out = self.c33_block(x_out)
        return x_out

    def forward(self, x):
        size = x.size()
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4]

        f2 = self.c2(x2)
        f3 = self.c3(x3)
        f4 = self.c4(x4)
        f5 = self.c5(x5)
        # f5 = self.BlockAtt(f5, 2)
        f4 = F.interpolate(f5, scale_factor=2, mode='bilinear', align_corners=True) + f4
        f3 = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=True) + f3
        f2 = F.interpolate(f3, scale_factor=2, mode='bilinear', align_corners=True) + f2

        af4 = self.AttFusion(f5, f4)
        af3 = self.AttFusion(f4, f3)
        af2 = self.AttFusion(f3, f2)

        af_sum = F.interpolate(af4, scale_factor=4, mode='bilinear', align_corners=True) + \
                 F.interpolate(af3, scale_factor=2, mode='bilinear', align_corners=True) + af2

        af_sum = self.c33_block(af_sum)

        out = F.interpolate(af_sum, size=(size[2], size[3]), mode='bilinear', align_corners=True)
        out = self.outc(out)
        out = self.activate(out)
        return out


class Res_UNet_34(nn.Module):
    def __init__(self, in_c, num_class, bilinear=True):
        super(Res_UNet_34, self).__init__()
        self.in_c = 3
        self.num_class = num_class
        self.bilinear = bilinear
        self.backbone = get_model('resnet34')

        self.inc = DoubleConv(in_c, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(512 + 256, 256, bilinear)
        self.up2 = Up(256 + 128, 128, bilinear)
        self.up3 = Up(128 + 64, 64, bilinear)
        self.up4 = Up(64 + 64, 64, bilinear)

        self.outc = OutConv(64, num_class)
        self.Softmax = nn.Softmax()

    def forward(self, x):
        size = x.size()
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4]

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = F.interpolate(x, size=(size[2], size[3]), mode='bilinear')
        out = self.outc(out)
        out = self.Softmax(out)
        return out


class Res_UNet_101(nn.Module):
    def __init__(self, in_c, num_class, bilinear=True):
        super(Res_UNet_101, self).__init__()
        self.in_c = 3
        self.num_class = num_class
        self.bilinear = bilinear
        self.backbone = get_model('resnet101')

        self.inc = DoubleConv(in_c, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(2048 + 1024, 256, bilinear)
        self.up2 = Up(512 + 256, 128, bilinear)
        self.up3 = Up(256 + 128, 64, bilinear)
        self.up4 = Up(64 + 64, 64, bilinear)

        self.outc = OutConv(64, num_class)
        self.Softmax = nn.Softmax()

    def forward(self, x):
        size = x.size()
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4]

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = F.interpolate(x, size=(size[2], size[3]), mode='bilinear')
        out = self.outc(out)
        out = self.Softmax(out)
        return out


class Res_UNet_152(nn.Module):
    def __init__(self, in_c, num_class, bilinear=True):
        super(Res_UNet_152, self).__init__()
        self.in_c = 3
        self.num_class = num_class
        self.bilinear = bilinear
        self.backbone = get_model('resnet152')

        self.inc = DoubleConv(in_c, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(2048 + 1024, 256, bilinear)
        self.up2 = Up(512 + 256, 128, bilinear)
        self.up3 = Up(256 + 128, 64, bilinear)
        self.up4 = Up(64 + 64, 64, bilinear)

        self.outc = OutConv(64, num_class)
        self.Softmax = nn.Softmax()

    def forward(self, x):
        size = x.size()
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4]

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = F.interpolate(x, size=(size[2], size[3]), mode='bilinear')
        out = self.outc(out)
        out = self.Softmax(out)
        return out
