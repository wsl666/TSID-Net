from networks.utils import mean_variance_norm
from torch import nn
import torch

class AttentionBlock(nn.Module):
    def __init__(self, in_planes):
        super(AttentionBlock, self).__init__()

        self.query_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.key_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.value_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

        self.softmax = nn.Softmax(dim=-1)

        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self, content, style):

        proj_query = self.query_conv(mean_variance_norm(content))
        proj_key = self.key_conv(mean_variance_norm(style))
        proj_value = self.value_conv(style)

        b, c, h, w = proj_query.size()
        proj_query = proj_query.view(b, -1, w * h).permute(0, 2, 1)

        b, c, h, w = proj_key.size()
        proj_key = proj_key.view(b, -1, w * h)

        b, c, h, w = proj_value.size()
        proj_value = proj_value.view(b, -1, w * h)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        out = self.out_conv(out)
        out += content

        return out

class TransformNet(nn.Module):
    def __init__(self, in_planes=512):
        super(TransformNet, self).__init__()

        self.ab4_1 = AttentionBlock(in_planes=in_planes)
        self.ab5_1 = AttentionBlock(in_planes=in_planes)
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1):

        self.upsample5_1 = nn.Upsample(size=(content4_1.size()[2], content4_1.size()[3]), mode='nearest')

        out = self.merge_conv(self.merge_conv_pad(self.ab4_1(content4_1, style4_1) + self.upsample5_1(self.ab5_1(content5_1, style5_1))))

        return out
