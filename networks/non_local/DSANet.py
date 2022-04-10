import torch.nn as nn
from torch.nn import functional as F
from networks.common_func.get_backbone import get_model
from einops import rearrange
import torch
from networks.common_func.non_local import NonLocalBlock


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

        self.bilinear = bilinear
        self.backbone = get_model('resnet50', pretrained=False)

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
        self.att_avgp = nn.AvgPool2d((self.sub_factor, self.sub_factor))

    def BlockAtt(self, x):
        x_avgp = self.att_avgp(x)
        x_div = rearrange(x, 'b c (h k1) (w k2) -> b (c k1 k2) h w', k1=self.sub_factor, k2=self.sub_factor)
        x_div = self.nlb(x_div)
        x_total = rearrange(x_div, 'b (c k1 k2) h w -> b c (h k1) (w k2)', k1=self.sub_factor, k2=self.sub_factor)

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
        f5 = self.BlockAtt(f3)

        out = F.interpolate(f5, size=(size[2], size[3]), mode='bilinear', align_corners=True)
        out = self.outc(out)
        out = self.activate(out)
        return out


if __name__ == '__main__':
    x = torch.rand(2, 3, 256, 256)
    model = DSANet()

    y = model(x)
    print(y.shape)