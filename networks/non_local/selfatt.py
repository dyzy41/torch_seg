import torch.nn as nn
from torch.nn import functional as F
import torch
from einops import rearrange


class SAtt(nn.Module):
    def __init__(self, channel=256):
        super(SAtt, self).__init__()
        self.channel = channel
        self.q_block = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU())
        self.k_block = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU())
        self.v_block = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU())

        self.out_block = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU())

    def forward(self, x):
        batch_size = x.size(0)
        query = self.q_block(x)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.k_block(x)
        value = self.v_block(x)

        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *x.shape[2:])

        context = self.out_block(context)
        return context


class SAtt_sq(nn.Module):
    def __init__(self, channel=256, sub_factor=4):
        super(SAtt_sq, self).__init__()
        self.channel = channel
        self.sub_factor = sub_factor
        self.q_block = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU())
        self.k_block = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU())
        self.v_block = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU())

        self.out_block = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU())
        self.att_avgp = nn.AvgPool2d((self.sub_factor, self.sub_factor))

    def gen_attmap(self, x):
        query = self.q_block(x)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.k_block(x)
        value = self.v_block(x)

        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        sim_map = F.softmax(sim_map, dim=-1)
        return sim_map, value

    def forward(self, x):
        x_avg = self.att_avgp(x)
        batch_size = x.size(0)
        x_div = rearrange(x, 'b c (h k1) (w k2) -> b (c k1 k2) h w', k1=self.sub_factor, k2=self.sub_factor)
        block_map, block_value = self.gen_attmap(x_div)
        block_context = torch.matmul(block_map, block_value)
        block_context = block_context.permute(0, 2, 1).contiguous()
        block_context = block_context.reshape(batch_size, -1, *x_div.shape[2:])
        x_div = rearrange(block_context, 'b (c k1 k2) h w -> b c (h k1) (w k2)', k1=self.sub_factor, k2=self.sub_factor)

        avg_map, avg_value = self.gen_attmap(x_avg)
        avg_context = torch.matmul(avg_map, avg_value)
        avg_context = avg_context.permute(0, 2, 1).contiguous()
        avg_context = avg_context.reshape(batch_size, -1, *x_div.shape[2:])
        x_avg = rearrange(avg_context, 'b (c k1 k2) h w -> b c (h k1) (w k2)', k1=self.sub_factor, k2=self.sub_factor)

        x_avg = F.interpolate(x_avg, scale_factor=self.sub_factor, mode='bilinear', align_corners=True)

        x_out = x_div+x_avg
        x_out = self.out_block(x_out)
        return x_out


if __name__ == '__main__':
    x = torch.rand(2, 256, 64, 64)
    model = SAtt(256)

    y = model(x)
    print(y.shape)
