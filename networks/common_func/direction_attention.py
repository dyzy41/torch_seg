import torch.nn as nn
from einops import rearrange
import torch


class DirAttention(nn.Module):
    def __init__(self, in_c, length=1):
        super(DirAttention, self).__init__()
        self.in_c = in_c
        self.conv_Q = nn.Conv2d(in_channels=self.in_c, out_channels=self.in_c, kernel_size=1, stride=1,
                                padding=0)
        self.conv_K = nn.Conv2d(in_channels=self.in_c, out_channels=self.in_c, kernel_size=1, stride=1,
                                padding=0)
        self.conv_V = nn.Conv2d(in_channels=self.in_c, out_channels=self.in_c, kernel_size=1, stride=1,
                                padding=0)
        self.length = length
        self.softmax = nn.Softmax(dim=1)
        self.conv_C = nn.Conv2d(in_channels=self.in_c*self.length, out_channels=self.in_c, kernel_size=1, stride=1, padding=0)
        self.outc = nn.Sequential(nn.Conv2d(self.in_c*3, self.in_c, 3, 1, 1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(self.in_c))

    def h_direction(self, x):
        x_Q = self.conv_Q(x)
        x_Q = rearrange(x_Q, 'b c h w -> b (c w) h 1')

        x_K = self.conv_K(x)
        x_K = x_K.transpose(3, 2)
        x_K = rearrange(x_K, 'b c h w -> b (c h) 1 w')

        att_map = torch.matmul(x_Q, x_K)
        att_map = self.conv_C(att_map)
        att_map = self.softmax(att_map)
        return x*att_map

    def w_direction(self, x):
        x_Q = self.conv_Q(x)
        x_Q = x_Q.transpose(3, 2)
        x_Q = rearrange(x_Q, 'b c h w -> b (c w) h 1')

        x_K = self.conv_K(x)
        x_K = rearrange(x_K, 'b c h w -> b (c h) 1 w')

        att_map = torch.matmul(x_Q, x_K)
        att_map = self.conv_C(att_map)
        att_map = self.softmax(att_map)
        return x*att_map

    def forward(self, x):
        h_att = self.h_direction(x)
        w_att = self.w_direction(x)
        out = torch.cat([x, h_att, w_att], dim=1)
        out = self.outc(out)
        return out


if __name__ == '__main__':

    x = torch.rand(2, 64, 128, 128)
    model = ConnectAttention(in_c=64, length=128)
    y = model(x)

    print(y.shape)
