import torch
import torch.nn as nn
from torch.nn import init

def vgg(cfg, i=3, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

class VGG_extra(nn.Module):
    def __init__(self, cfg, inplanes=512):
        super(VGG_extra, self).__init__()
        layers = []
        self.cfg = cfg
        for v in self.cfg:
            conv2d = nn.Conv2d(inplanes, v, kernel_size=3, padding=1, dilation=1, bias=False)
            layers += [conv2d, nn.ReLU(inplace=True)]
            inplanes = v
        self.extra = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.extra(x)
        return x

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.cfg = {'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 'extr': [512, 512, 512]}
        self.extract = [3, 8, 15, 22, 29]
        self.base = nn.ModuleList(vgg(self.cfg['base']))
        self.extr = VGG_extra(self.cfg['extr'], 512)

        pre_train = torch.load('/home/hengyuli/NLENet10/model/vgg16-397923af.pth')
        self._initialize_weights(pre_train)

    def forward(self, x):
        out = []
        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.extract:
                out.append(x)
        x = self.extr(x)
        out.append(x)
        return out

    def _initialize_weights(self, pre_train):
        keys = list(pre_train.keys())
        self.base[0].weight.data.copy_(pre_train[keys[0]])
        self.base[2].weight.data.copy_(pre_train[keys[2]])
        self.base[5].weight.data.copy_(pre_train[keys[4]])
        self.base[7].weight.data.copy_(pre_train[keys[6]])
        self.base[10].weight.data.copy_(pre_train[keys[8]])
        self.base[12].weight.data.copy_(pre_train[keys[10]])
        self.base[14].weight.data.copy_(pre_train[keys[12]])
        self.base[17].weight.data.copy_(pre_train[keys[14]])
        self.base[19].weight.data.copy_(pre_train[keys[16]])
        self.base[21].weight.data.copy_(pre_train[keys[18]])
        self.base[24].weight.data.copy_(pre_train[keys[20]])
        self.base[26].weight.data.copy_(pre_train[keys[22]])
        self.base[28].weight.data.copy_(pre_train[keys[24]])
        self.base[0].bias.data.copy_(pre_train[keys[1]])
        self.base[2].bias.data.copy_(pre_train[keys[3]])
        self.base[5].bias.data.copy_(pre_train[keys[5]])
        self.base[7].bias.data.copy_(pre_train[keys[7]])
        self.base[10].bias.data.copy_(pre_train[keys[9]])
        self.base[12].bias.data.copy_(pre_train[keys[11]])
        self.base[14].bias.data.copy_(pre_train[keys[13]])
        self.base[17].bias.data.copy_(pre_train[keys[15]])
        self.base[19].bias.data.copy_(pre_train[keys[17]])
        self.base[21].bias.data.copy_(pre_train[keys[19]])
        self.base[24].bias.data.copy_(pre_train[keys[21]])
        self.base[26].bias.data.copy_(pre_train[keys[23]])
        self.base[28].bias.data.copy_(pre_train[keys[25]])


if __name__ == '__main__':
    x = torch.randn(1, 3, 352, 352)
    model = VGG16()
    print(model)
    out = model(x)
    for i in range(len(out)):
        print(out[i].shape)