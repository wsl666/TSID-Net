import torch
import torch.nn as nn
import torch.nn.functional as F
from .deconv import FastDeconv
from .deform import DeformConv2d

# Dynamic Feature Enhancement Module
class DFEM(nn.Module):
  def __init__(self, nChannels):
    super(DFEM, self).__init__()

    self.conv_x = DeformConv2d(nChannels, nChannels)
    self.conv_m = nn.Conv2d(nChannels, nChannels, kernel_size=3,stride=1,padding=1)
    self.conv_y = DeformConv2d(nChannels, nChannels)
    self.conv_o = nn.Conv2d(nChannels*3, nChannels, kernel_size=1)

  def forward(self, x):

    conv_x = self.conv_x(x)
    conv_m = self.conv_m(x)
    conv_y = self.conv_y(x)
    xmy = torch.cat([conv_x,conv_m,conv_y],dim=1)
    out = self.conv_o(xmy) + x

    return out

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class PDU(nn.Module):  # physical block
    def __init__(self, channel):
        super(PDU, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ka = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.td = nn.Sequential(
            default_conv(channel, channel, 3),
            default_conv(channel, channel // 8, 3),
            nn.ReLU(inplace=True),
            default_conv(channel // 8, channel, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.avg_pool(x)
        a = self.ka(a)
        t = self.td(x)
        j = torch.mul((1 - t), a) + torch.mul(t, x)
        return j


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y


class Dehazeblock(nn.Module):  # origin
    def __init__(self, conv,dim):
        super(Dehazeblock, self).__init__()
        self.conv1 = conv(dim,dim,3)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = conv(dim,dim,3)
        self.calayer = CALayer(dim)
        self.pdu = PDU(dim)

    def forward(self, x):
        res = self.act(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.pdu(res)
        res += x
        return res

class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Dehazeblock(conv,dim) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res


class SRNet(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(SRNet, self).__init__()
        self.gps = gps
        self.dim = 128
        kernel_size = 3
        pre_process = [
            FastDeconv(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1),
            conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

        self.pa = PALayer(self.dim)
        self.dfem = DFEM(self.dim)
        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)
        ]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x):

        x_pre = self.pre(x)

        res_g1 = self.g1(x_pre)
        res_g2 = self.g2(res_g1)
        res_g3 = self.g3(res_g2)

        w = self.ca(torch.cat([res_g1, res_g2, res_g3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]

        # res_g = w[:, 0, ::] * res_g1 + w[:, 1, ::] * res_g2 + w[:, 2, ::] * res_g3

        res_g1w = w[:, 0, ::] * res_g1
        res_g2w = w[:, 1, ::] * res_g2
        res_g3w = w[:, 2, ::] * res_g3

        res_g = res_g1w + res_g2w + res_g3w

        res_pa = self.pa(res_g)

        res_dfem = self.dfem(res_pa)

        out = self.post(res_dfem)
        # out = self.post(res_dfem) + x

        return out


if __name__ == "__main__":
    x=torch.randn(1,3,256,256).cuda()
    net = SRNet(gps=3, blocks=3).cuda()
    print(net(x).shape)
