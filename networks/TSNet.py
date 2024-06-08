import torch
import torch.nn as nn
import numpy as np
import time
from .deform import DeformConv2d


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

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


class ChannelAttention(nn.Module):
    def __init__(self, nc,number, norm_layer = nn.BatchNorm2d):
        super(ChannelAttention, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(nc, nc, 3, stride=1, padding=1, bias=True),
                                   norm_layer(nc),
                                   nn.PReLU(nc),
                                   nn.Conv2d(nc, nc, 3, stride=1, padding=1, bias=True),
                                   norm_layer(nc)
                                   )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv2 = nn.Sequential(nn.Conv2d(nc, number, 1, stride=1, padding=0, bias=False),
                                    nn.ReLU(),
                                    nn.Conv2d(number, nc, 1, stride=1, padding=0, bias=False),
                                    nn.Sigmoid())
        self.conv3 = nn.Conv2d(nc*2,nc,1)

    def forward(self, x):

        ca_avg = self.conv2(self.avg_pool(self.conv1(x))) * x
        ca_max = self.conv2(self.max_pool(self.conv1(x))) * x

        avg_max = torch.cat([ca_avg,ca_max],dim=1)

        ca_map = self.conv3(avg_max)

        return ca_map

#Multiscale SpatialAttention
class MSSpatialAttention(nn.Module):
    def __init__(self, nc, number, norm_layer = nn.BatchNorm2d):
        super(MSSpatialAttention, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(nc,nc,kernel_size=3,stride=1,padding=1,bias=False),
                                   norm_layer(nc),
                                   nn.PReLU(nc),
                                   nn.Conv2d(nc,number,kernel_size=3,stride=1,padding=1,bias=False),
                                   norm_layer(number)
                                   )

        self.conv2 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=3,dilation=3,bias=False)
        self.conv3 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=5,dilation=5,bias=False)
        self.conv4 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=7,dilation=7,bias=False)

        self.conv5 = nn.Sequential(nn.Conv2d(number*4,1,kernel_size=3,stride=1,padding=1,bias=False),
                                   nn.ReLU(),
                                   nn.Conv2d(1, 1, 1, stride=1, padding=0, bias=False),
                                   nn.Sigmoid())

    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        
        x1 = x
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        
        se = torch.cat([x1, x2, x3, x4], dim=1)
        
        out = self.conv5(se) * x0
        
        return out

#Block
class Block(nn.Module):
    def __init__(self, conv,dim):
        super(Block, self).__init__()
        self.conv1 = conv(dim,dim,3)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = conv(dim,dim,3)
        self.ca = ChannelAttention(dim,dim)
        self.mspa = MSSpatialAttention(dim,dim)

    def forward(self, x):
        res = self.act(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.ca(res)
        res = self.mspa(res)
        res += x
        return res


class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv,dim) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

class TSNet(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(TSNet, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
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

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x):

        x_pre = self.pre(x)
        # print(x_pre.shape)
        res_g1 = self.g1(x_pre)
        res_g2 = self.g2(res_g1)
        res_g3 = self.g3(res_g2)

        w = self.ca(torch.cat([res_g1, res_g2, res_g3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]

        res_g1w = w[:, 0, ::] * res_g1
        res_g2w = w[:, 1, ::] * res_g2
        res_g3w = w[:, 2, ::] * res_g3

        res_g = res_g1w + res_g2w + res_g3w

        res_pa = self.pa(res_g)

        out = self.post(res_pa)

        # return out,res_g1w,res_g2w,res_g3w
        return out,res_g3w


class CKTTeacher(nn.Module):
    def __init__(self, goodt_path=None, badt_path=None,requires_grad=False):
        super(CKTTeacher, self).__init__()
        self.GoodT = TSNet(gps=3,blocks=6).cuda()
        self.BadT  = TSNet(gps=3,blocks=6).cuda()

        self.GoodT.load_state_dict(torch.load(goodt_path))
        self.BadT.load_state_dict(torch.load(badt_path))
        print("Load teacher models weights successfully")
        self.l1 = nn.L1Loss()
        if not requires_grad:
            print("Teacher models requires_grad = False")

            for param in self.GoodT.parameters():
                param.requires_grad = requires_grad

            for param in self.BadT.parameters():
                param.requires_grad = requires_grad

    def forward(self,x,y,z=None):

        good_feats_y = self.GoodT(y)
        good_feats_y = good_feats_y[1:]

        bad_feats_z = self.BadT(z)
        bad_feats_z = bad_feats_z[1:]

        loss1 = 0
        loss2 = 0
        d_ap = 0
        d_an = 0
        #
        for i in range(len(good_feats_y)):

            d_ap = self.l1(x[i], good_feats_y[i].detach())
            loss1 += d_ap

            d_an = self.l1(x[i], bad_feats_z[i].detach())
            contrastive = d_ap / (d_an + 1e-7)

            loss2 += contrastive

        return loss1 + loss2 * 0.1
        # return loss2



        
if __name__ == '__main__':
    path= "../checkpoints/GoodT/NH-haze_T.pth"
    net = CKTTeacher(goodt_path=path).cuda()
    input_tensor = torch.Tensor(np.random.random((1,3,256,256))).cuda()
    start = time.time()
    out = net(input_tensor,input_tensor)
    end = time.time()
    print('Process Time: %f'%(end-start))
    print(out.shape)


