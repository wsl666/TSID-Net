import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1) 
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4) 
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class ContrastLoss(nn.Module):
    def __init__(self):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, a, p, inp,candidate_list,weight):
        a_vgg, p_vgg = self.vgg(a), self.vgg(p)
        inp_vgg = self.vgg(inp)
        n1_vgg,n2_vgg,n3_vgg,n4_vgg,n5_vgg =  self.vgg(candidate_list[0].cuda()),self.vgg(candidate_list[1].cuda()),self.vgg(candidate_list[2].cuda()),\
                                              self.vgg(candidate_list[3].cuda()),self.vgg(candidate_list[4].cuda())
        loss = 0
        n1_weight, n2_weight, n3_weight, n4_weight, n5_weight, inp_weight = weight
        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):

            d_ap  = self.l1(a_vgg[i], p_vgg[i].detach())
            d_inp = self.l1(a_vgg[i], inp_vgg[i].detach())
            d_an1 = self.l1(a_vgg[i], n1_vgg[i].detach())
            d_an2 = self.l1(a_vgg[i], n2_vgg[i].detach())
            d_an3 = self.l1(a_vgg[i], n3_vgg[i].detach())
            d_an4 = self.l1(a_vgg[i], n4_vgg[i].detach())
            d_an5 = self.l1(a_vgg[i], n5_vgg[i].detach())

            # contrastive = d_ap / (d_an0 + 1e-7)
            contrastive = d_ap / (d_an1 * n1_weight + d_an2 * n2_weight + d_an3 * n3_weight + d_an4 * n4_weight + d_an5 * n5_weight + d_inp * inp_weight+ 1e-7)

            loss += self.weights[i] * contrastive

        return loss

if __name__=="__main__":
    x=torch.ones(1,3,256,256).cuda()
    y=torch.zeros(1,3,256,256).cuda()
    z=torch.zeros(1,3,256,256).cuda()
    l=ContrastLoss().cuda()
    loss=l(x,y,z).cuda()
    print(loss)