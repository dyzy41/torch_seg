import torch
import torch.nn as nn
from einops import rearrange
from networks.common_func.non_local import NonLocalBlock
import torch.nn.functional as F


class BlockAtt(nn.Module):
    def __init__(self, channel, sub_factor, size):
        super(BlockAtt, self).__init__()
        self.sub_factor = sub_factor
        self.channel = channel
        self.size = size
        self.down_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.channel * self.sub_factor * self.sub_factor, out_channels=self.channel,
                      kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.channel),
            nn.ReLU())
        self.up_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.channel // self.sub_factor // self.sub_factor, out_channels=self.channel,
                      kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.channel),
            nn.ReLU())
        self.nlb_sub = NonLocalBlock(self.channel)
        self.att_avgp = nn.AvgPool2d((self.size // self.sub_factor, self.size // self.sub_factor))
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=1, stride=1,
                                    padding=0),
                                        nn.BatchNorm2d(self.channel),
                                        nn.ReLU()
                                        )
        self.nlb_block = NonLocalBlock(self.channel)

    def forward(self, x):
        x_avgp = self.att_avgp(x)
        x_div = rearrange(x, 'b c (h k1) (w k2) -> b (c k1 k2) h w', k1=self.sub_factor, k2=self.sub_factor)
        x_div = self.down_channel(x_div)
        x_div = self.nlb_sub(x_div)
        x_total = rearrange(x_div, 'b (c k1 k2) h w -> b c (h k1) (w k2)', k1=self.sub_factor, k2=self.sub_factor)
        x_total = self.up_channel(x_total)

        x_avgp = self.nlb_block(x_avgp)
        x_avgp = F.interpolate(x_avgp, size=(self.size, self.size), mode='nearest')
        x_total = x_avgp * x_total
        x_out = x + x_total
        x_out = self.conv_block(x_out)
        return x_out


if __name__ == '__main__':
    model = BlockAtt(channel=256, sub_factor=4, size=128)
    print(model)

    input = torch.randn(2, 256, 128, 128)
    out = model(input)
    print(out.shape)
