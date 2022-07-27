import torch
import torch.nn as nn


class ConvCC3(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvCC3, self).__init__()
        self.conv13 = nn.Conv2d(in_c, out_c, (1, 3), 1, (0, 1))
        self.conv31 = nn.Conv2d(in_c, out_c, (3, 1), 1, (1, 0))
        self.conv = nn.Conv2d(in_c, out_c, 1)
        # self.outc = nn.Sequential(nn.Conv2d(out_c, out_c, 1),
        #                           nn.BatchNorm2d(out_c),
        #                           nn.ReLU())
        self.outc = nn.Sequential(nn.Conv2d(out_c, out_c, 1))

    def forward(self, x):
        x13 = self.conv13(x)
        x31 = self.conv31(x)
        x = self.conv(x)
        out = x13+x31+x
        out = self.outc(out)
        return out


class ConvCC5(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvCC5, self).__init__()
        self.conv15 = nn.Conv2d(in_c, out_c, (1, 5), 1, (0, 2))
        self.conv51 = nn.Conv2d(in_c, out_c, (5, 1), 1, (2, 0))
        self.conv = nn.Conv2d(in_c, out_c, 1)
        self.conv3 = ConvCC3(in_c, out_c)
        # self.outc = nn.Sequential(nn.Conv2d(out_c, out_c, 1),
        #                           nn.BatchNorm2d(out_c),
        #                           nn.ReLU())
        self.outc = nn.Sequential(nn.Conv2d(out_c, out_c, 1))


    def forward(self, x):
        x15 = self.conv15(x)
        x51 = self.conv51(x)
        x3 = self.conv3(x)
        x = self.conv(x)

        out = x15+x51+x3+x
        out = self.outc(out)
        return out


class ConvCC7(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvCC7, self).__init__()
        self.conv17 = nn.Conv2d(in_c, out_c, (1, 7), 1, (0, 3))
        self.conv71 = nn.Conv2d(in_c, out_c, (7, 1), 1, (3, 0))
        self.conv = nn.Conv2d(in_c, out_c, 1)
        self.conv3 = ConvCC3(in_c, out_c)
        self.conv5 = ConvCC5(in_c, out_c)
        # self.outc = nn.Sequential(nn.Conv2d(out_c, out_c, 1),
        #                           nn.BatchNorm2d(out_c),
        #                           nn.ReLU())
        self.outc = nn.Sequential(nn.Conv2d(out_c, out_c, 1))

    def forward(self, x):
        x17 = self.conv17(x)
        x71 = self.conv71(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x = self.conv(x)

        out = x17+x71+x3+x5+x
        out = self.outc(out)
        return out





if __name__ == '__main__':
    x = torch.rand(2, 128, 64, 64)
    model = ConvCC7(128, 128)

    y = model(x)
    print(y.shape)