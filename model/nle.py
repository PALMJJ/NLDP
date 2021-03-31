import torch
from .rfb import RFB
import torch.nn as nn
from .vgg import VGG16
import torch.nn.functional as F
from .aggregation import Aggregation1, Aggregation2
from .non_local_embedded_gaussian import NonLocalBlock2D

class NLE(nn.Module):
    def __init__(self):
        super(NLE, self).__init__()
        self.vgg16 = VGG16()
        self.rfb2 = RFB(256, 32)
        self.rfb3 = RFB(512, 32)
        self.rfb4 = RFB(512, 32)
        self.rfb5 = RFB(512, 32)
        self.aggregation1 = Aggregation1(32)
        self.aggregation2 = Aggregation2(32)
        self.non_local1 = NonLocalBlock2D(1)
        self.non_local2 = NonLocalBlock2D(1)

        self.conv1 = nn.Sequential(self.vgg16.base[16], self.vgg16.base[17], self.vgg16.base[18], self.vgg16.base[19], self.vgg16.base[20], self.vgg16.base[21], self.vgg16.base[22])
        self.conv2 = nn.Sequential(self.vgg16.base[23], self.vgg16.base[24], self.vgg16.base[25], self.vgg16.base[26], self.vgg16.base[27], self.vgg16.base[28], self.vgg16.base[29])
        self.conv3 = nn.Sequential(self.vgg16.base[30], self.vgg16.extr)

        self.conv4 = nn.Sequential(self.vgg16.base[23], self.vgg16.base[24], self.vgg16.base[25], self.vgg16.base[26], self.vgg16.base[27], self.vgg16.base[28], self.vgg16.base[29])
        self.conv5 = nn.Sequential(self.vgg16.base[30], self.vgg16.extr)

        self.rfb6 = RFB(256, 32)
        self.rfb7 = RFB(512, 32)
        self.rfb8 = RFB(512, 32)
        self.rfb9 = RFB(512, 32)

        self.rfb10 = RFB(512, 32)
        self.rfb11 = RFB(512, 32)
        self.rfb12 = RFB(512, 32)

    def forward(self, x):
        vgg_out = self.vgg16(x)

        rfb_out2 = self.rfb2(vgg_out[2])
        rfb_out3 = self.rfb3(vgg_out[3])
        rfb_out4 = self.rfb4(vgg_out[4])
        rfb_out5 = self.rfb5(vgg_out[5])

        init_saliency1 = self.aggregation1(rfb_out5, rfb_out4, rfb_out3, rfb_out2)
        init_saliency2 = self.aggregation2(rfb_out5, rfb_out4, rfb_out3)

        out2 = self.non_local1(init_saliency1.sigmoid()) * vgg_out[2]
        out3 = self.conv1(out2)
        out4 = self.conv2(out3)
        out5 = self.conv3(out4)
        out2 = self.rfb6(out2)
        out3 = self.rfb7(out3)
        out4 = self.rfb8(out4)
        out5 = self.rfb9(out5)
        final_saliency1 = self.aggregation1(out5, out4, out3, out2)
        out3 = self.non_local2(init_saliency2.sigmoid()) * vgg_out[3]
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out3 = self.rfb10(out3)
        out4 = self.rfb11(out4)
        out5 = self.rfb12(out5)
        final_saliency2 = self.aggregation2(out5, out4, out3)

        init_saliency1 = F.interpolate(init_saliency1, scale_factor=4, mode='bilinear', align_corners=True)
        init_saliency2 = F.interpolate(init_saliency2, scale_factor=8, mode='bilinear', align_corners=True)
        final_saliency1 = F.interpolate(final_saliency1, scale_factor=4, mode='bilinear', align_corners=True)
        final_saliency2 = F.interpolate(final_saliency2, scale_factor=8, mode='bilinear', align_corners=True)

        return init_saliency1, init_saliency2, final_saliency1, final_saliency2

if __name__ == '__main__':
    x = torch.randn(1, 3, 352, 352)
    model = NLE()
    init_saliency1, init_saliency2, final_saliency1, final_saliency2 = model(x)
    print(init_saliency1.shape)
    print(init_saliency2.shape)
    print(final_saliency1.shape)
    print(final_saliency2.shape)