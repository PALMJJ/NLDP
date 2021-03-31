import torch
import torch.nn as nn
from torch.nn import init

class RFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4 * out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x0, x1, x2, x3), dim=1)
        x_cat = self.conv_cat(x_cat)

        x = self.relu(x_cat + self.conv_res(x))
        return x