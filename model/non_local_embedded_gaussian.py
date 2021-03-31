import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(NonLocalBlock2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.out_channels is None:
            self.out_channels = in_channels // 2
            if self.out_channels == 0:
                self.out_channels = 1

        self.g = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)

        self.W = nn.Sequential(nn.Conv2d(self.out_channels, self.in_channels, 1, 1, 0), nn.BatchNorm2d(self.in_channels))
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)

        self.phi = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)

        self.g = nn.Sequential(self.g, nn.MaxPool2d(2, 2))

        self.phi = nn.Sequential(self.phi, nn.MaxPool2d(2, 2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x): # x -> (b, c, h, w)
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.out_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.out_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.out_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.out_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z