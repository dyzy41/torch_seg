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


class _ASPPConv(nn.Module):
    def __init__(self, in_cannels, num_classannels, atrous_rate, norm_layer):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_cannels, num_classannels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(num_classannels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_cannels, num_classannels, norm_layer):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_cannels, num_classannels, 1, bias=False),
            norm_layer(num_classannels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class _ASPP(nn.Module):
    def __init__(self, in_cannels, atrous_rates, norm_layer):
        super(_ASPP, self).__init__()
        num_classannels = 256
        self.b0 = nn.Sequential(
            nn.Conv2d(in_cannels, num_classannels, 1, bias=False),
            norm_layer(num_classannels),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_cannels, num_classannels, rate1, norm_layer)
        self.b2 = _ASPPConv(in_cannels, num_classannels, rate2, norm_layer)
        self.b3 = _ASPPConv(in_cannels, num_classannels, rate3, norm_layer)
        self.b4 = _AsppPooling(in_cannels, num_classannels, norm_layer=norm_layer)

        self.project = nn.Sequential(
            nn.Conv2d(5 * num_classannels, num_classannels, 1, bias=False),
            norm_layer(num_classannels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x


class TwinNet50(nn.Module):
    def __init__(self, in_c, num_class, pretrained_path='download', bilinear=True):
        super(TwinNet50, self).__init__()
        self.in_c = 3
        self.num_class = num_class
        self.bilinear = bilinear
        self.backbone = get_model('resnet50', checkpoint_path=pretrained_path)
        self.inc = DoubleConv(in_c, 64)

        self.conv_block2 = DoubleConv(256, 256)
        self.conv_block3 = DoubleConv(512, 256)
        self.conv_block4 = DoubleConv(1024, 256)
        self.conv_block5 = DoubleConv(2048, 256)

        self.ps = nn.PixelShuffle(32)
        self.conv_b = DoubleConv(2048, 2048)
        self.ps1 = nn.PixelShuffle(32)
        self.conv_f = DoubleConv(2048, 2048)

        self.fb = FusionBlock()
        self.aspp = _ASPP(256, [12, 24, 36], nn.BatchNorm2d)
        self.outc = OutConv(256, num_class)
        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

    def forward(self, x):
        size = x.size()
        layers = self.backbone(x)
        layers1 = self.backbone(x)

        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4]   #64->2048
        y1, y2, y3, y4, y5 = layers1[0], layers1[1], layers1[2], layers1[3], layers1[4]  # 64->2048


        out_b = self.ps(self.conv_b(x5))
        out_f = self.ps1(self.conv_f(y5))

        sum2 = x2+y2
        sum3 = x3 + y3
        sum4 = x4 + y4
        sum5 = x5 + y5

        sum2 = self.conv_block2(sum2)
        sum3 = self.conv_block3(sum3)
        sum4 = self.conv_block4(sum4)
        sum5 = self.conv_block5(sum5)

        f4 = self.fb(sum5, sum4)
        f3 = self.fb(sum4, sum3)
        f2 = self.fb(sum3, sum2)

        p3 = self.fb(f4, f3)
        p2 = self.fb(f3, f2)

        m2 = self.fb(p3, p2)
        # m2 = self.aspp(m2)
        out = F.interpolate(m2, size=(size[2], size[3]), mode='bilinear')
        out = self.outc(out)
        out = self.activate(out)
        return out, out_b, out_f


if __name__ == '__main__':
    x = torch.rand(2, 3, 512, 512)
    model = TwinNet50(3, 2)
    y = model(x)
    print(y.shape)

