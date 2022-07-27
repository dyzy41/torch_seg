import torch.nn as nn
import torch


class ConvLSTM(nn.Module):
    def __init__(self, series_length=256, lstm_hidden_size=256, num_lstm_layers=2, bidirectional=False):
        super(ConvLSTM, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(series_length, series_length, 3, 1, 1),
                                        nn.BatchNorm2d(series_length),
                                        nn.ReLU(),
                                        nn.Conv2d(series_length, series_length, 3, 1, 1),
                                        nn.BatchNorm2d(series_length)
                                        )
        self.lstm_hidden_size = lstm_hidden_size
        self.series_length = series_length
        self.num_lstm_layers = num_lstm_layers
        self.lstm1 = nn.LSTM(input_size=self.series_length,
                             hidden_size=self.lstm_hidden_size,
                             num_layers=self.num_lstm_layers,
                             dropout=0.5,
                             bidirectional=bidirectional)
    def init_hidden(self, x):
        batch_size = x.size(1)
        h = x.data.new(self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        c = x.data.new(self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        return h.cuda(), c.cuda()

    def forward(self, x):
        size = x.size()
        x = self.conv_block(x)
        x = x.view(size[0], size[1], -1).contiguous()
        x = x.permute(2, 0, 1)

        h, c = self.init_hidden(x)
        output, (h, c) = self.lstm1(x, (h, c))  # h: (num_layers * num_directions, batch, lstm_hidden_size)
        output = output.permute(1, 2, 0).contiguous()
        output = output.view(size[0], self.lstm_hidden_size, size[2], -1)
        return output


# x = torch.rand(4, 256, 64, 64).cuda()
# model = ConvLSTM().cuda()
# y = model(x)
# print(y.shape)