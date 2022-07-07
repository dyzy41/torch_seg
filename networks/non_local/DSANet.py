import torch.nn as nn
from torch.nn import functional as F
from networks.common_func.get_backbone import get_model
import torch
from networks.common_func.block_attention import BlockAtt
from networks.common_func.non_local import NonLocalBlock
from networks.common_func.base_func import DUpsampling


class OutConv(nn.Module):
    def __init__(self, in_cannels, num_classannels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_cannels, num_classannels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


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


class DSANet(nn.Module):
    def __init__(self, in_c=3, num_class=2, sub_factor=4, pretrained_path=None, bilinear=True):
        super(DSANet, self).__init__()
        self.in_c = in_c
        self.num_class = num_class
        self.sub_factor = sub_factor
        self.size_arr = [256, 128, 64, 32, 16, 8]

        self.bilinear = bilinear
        self.backbone = get_model('inception_resnet_v2', checkpoint_path=pretrained_path)

        self.inc = DoubleConv(in_c, 64)

        self.c5 = DoubleConv(1536, 256)
        self.c4 = DoubleConv(1088, 256)
        self.c3 = DoubleConv(320, 256)
        self.c2 = DoubleConv(192, 256)

        self.base_conv = DoubleConv(256, 256)

        self.blockatt2 = BlockAtt(256, 4, 64)
        self.blockatt3 = BlockAtt(256, 4, 32)
        # self.blockatt4 = BlockAtt(256, 2, 16)
        # self.blockatt5 = BlockAtt(256, 2, 8)
        self.blockatt4 = NonLocalBlock(256)
        self.blockatt5 = NonLocalBlock(256)

        self.Dup0 = DUpsampling(256, 2, 256)
        self.Dup1 = DUpsampling(256, 2, 256)
        self.Dup2 = DUpsampling(256, 2, 256)

        self.sigmoid = nn.Sigmoid()
        self.outc = OutConv(256, num_class)
        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()
        self.att_avgp = nn.AvgPool2d((self.sub_factor, self.sub_factor))

    def forward(self, x):
        size = x.size()
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4]

        x2 = F.interpolate(x2, size=(self.size_arr[2], self.size_arr[2]), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=(self.size_arr[3], self.size_arr[3]), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(self.size_arr[4], self.size_arr[4]), mode='bilinear', align_corners=True)
        x5 = F.interpolate(x5, size=(self.size_arr[5], self.size_arr[5]), mode='bilinear', align_corners=True)


        f2 = self.c2(x2)
        f3 = self.c3(x3)
        f4 = self.c4(x4)
        f5 = self.c5(x5)

        f2 = self.blockatt2(f2)
        f3 = self.blockatt3(f3)
        f4 = self.blockatt4(f4)
        f5 = self.blockatt5(f5)

        t4 = self.base_conv(self.Dup0(f5)+f4)
        t3 = self.base_conv(self.Dup1(t4) + f3)
        t2 = self.base_conv(self.Dup2(t3) + f2)

        out = self.outc(t2)
        out = F.interpolate(out, size=(size[2], size[3]), mode='bilinear', align_corners=True)
        out = self.activate(out)
        return out


if __name__ == '__main__':
    x = torch.rand(2, 3, 225, 225)
    model = DSANet(pretrained_path='download')

    y = model(x)
    print(y.shape)