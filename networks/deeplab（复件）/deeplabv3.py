import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common_func.get_backbone import get_model
from ..common_func.base_func import _ConvBNReLU


class _FCNHead(nn.Module):
    def __init__(self, in_cannels, channels, norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        inter_channels = in_cannels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_cannels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


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


class DeepLabV3(nn.Module):

    def __init__(self, in_c, num_class, pretrained_path=None, **kwargs):
        super(DeepLabV3, self).__init__()
        aux = True
        self.aux = aux
        self.num_class = num_class
        self.in_c = in_c

        self.pretrained = get_model('resnet50', checkpoint_path=pretrained_path)

        # deeplabv3 plus
        self.head = _DeepLabHead(num_class, c1_channels=2048, **kwargs)
        if aux:
            self.auxlayer = _FCNHead(728, num_class)
        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

    def base_forward(self, x):
        # Entry flow
        features = self.pretrained(x)
        return features[4]

    def forward(self, x):
        size = x.size()[2:]
        out_feature = self.base_forward(x)
        x = self.head(out_feature)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        x = self.activate(x)
        return x


class _DeepLabHead(nn.Module):
    def __init__(self, num_class, c1_channels=128, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(c1_channels, [12, 24, 36], norm_layer)
        self.c1_block = _ConvBNReLU(c1_channels, 48, 3, padding=1, norm_layer=norm_layer)
        self.block = nn.Sequential(
            _ConvBNReLU(256, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.5),
            _ConvBNReLU(256, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_class, 1))

    def forward(self, x):
        x = self.aspp(x)
        x = self.block(x)
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        return x

if __name__ == '__main__':
    net = DeepLabV3(num_class=6)
    # nrte = meca(64, 3)

    x = torch.randn(2, 3, 256, 256)
    y = net(x)
    print(y.shape)