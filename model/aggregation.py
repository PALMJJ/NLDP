import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Aggregation1(nn.Module):
    def __init__(self, channel=32):
        super(Aggregation1, self).__init__()
        self.conv0_1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv0_2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv0_3 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv1_1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv2_1 = nn.Conv2d(channel, channel, 3, 1, 1)

        self.conv0_4 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv_cat10 = nn.Conv2d(2 * channel, 2 * channel, 3, 1, 1)
        self.conv1_3 = nn.Conv2d(2 * channel, 2 * channel, 3, 1, 1)
        self.conv_cat21 = nn.Conv2d(3 * channel, 3 * channel, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(3 * channel, 3 * channel, 3, 1, 1)
        self.conv_cat32 = nn.Conv2d(4 * channel, 4 * channel, 3, 1, 1)

        self.conv3_1 = nn.Conv2d(4 * channel, 4 * channel, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(4 * channel, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x0, x1, x2, x3): # 参数分别为vgg16的最后一个元素往前数, x0 : 1/32, x1: 1/16, x2 : 1/8, x3: 1/4
        out0 = x0
        out10 = self.conv0_1(F.interpolate(x0, scale_factor=2, mode='bilinear', align_corners=True)) * x1
        out210 = self.conv0_2(F.interpolate(F.interpolate(x0, scale_factor=2, mode='bilinear', align_corners=True), scale_factor=2, mode='bilinear', align_corners=True)) * self.conv1_1(F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)) * x2
        out3210 = self.conv0_3(F.interpolate(F.interpolate(F.interpolate(x0, scale_factor=2, mode='bilinear', align_corners=True), scale_factor=2, mode='bilinear', align_corners=True), scale_factor=2, mode='bilinear', align_corners=True)) * self.conv1_2(F.interpolate(F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True), scale_factor=2, mode='bilinear', align_corners=True)) * self.conv2_1(F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)) * x3

        cat10 = torch.cat((out10, self.conv0_4(F.interpolate(out0, scale_factor=2, mode='bilinear', align_corners=True))), 1)
        cat10 = self.conv_cat10(cat10)
        cat21 = torch.cat((out210, self.conv1_3(F.interpolate(cat10, scale_factor=2, mode='bilinear', align_corners=True))), 1)
        cat21 = self.conv_cat21(cat21)
        cat32 = torch.cat((out3210, self.conv2_2(F.interpolate(cat21, scale_factor=2, mode='bilinear', align_corners=True))), 1)
        cat32 = self.conv_cat32(cat32)

        x = self.conv3_1(cat32)
        x = self.conv3_2(x)

        return x

class Aggregation2(nn.Module):
    def __init__(self, channel=32):
        super(Aggregation2, self).__init__()
        self.conv0_1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv0_2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv1_1 = nn.Conv2d(channel, channel, 3, 1, 1)

        self.conv0_3 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv_cat10 = nn.Conv2d(2 * channel, 2 * channel, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(2 * channel, 2 * channel, 3, 1, 1)
        self.conv_cat21 = nn.Conv2d(3 * channel, 3 * channel, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(3 * channel, 3 * channel, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(3 * channel, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x0, x1, x2): # 参数分别为vgg16的最后一个元素往前数, x0 : 1/32, x1: 1/16, x2 : 1/8
        out0 = x0
        out10 = self.conv0_1(F.interpolate(x0, scale_factor=2, mode='bilinear', align_corners=True)) * x1
        out210 = self.conv0_2(F.interpolate(F.interpolate(x0, scale_factor=2, mode='bilinear', align_corners=True), scale_factor=2, mode='bilinear', align_corners=True)) * self.conv1_1(F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)) * x2

        cat10 = torch.cat((out10, self.conv0_3(F.interpolate(out0, scale_factor=2, mode='bilinear', align_corners=True))), 1)
        cat10 = self.conv_cat10(cat10)
        cat21 = torch.cat((out210, self.conv1_2(F.interpolate(cat10, scale_factor=2, mode='bilinear', align_corners=True))), 1)
        cat21 = self.conv_cat21(cat21)

        x = self.conv2_1(cat21)
        x = self.conv2_2(x)

        return x