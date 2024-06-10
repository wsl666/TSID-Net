from networks import VGG19
from networks import Transform
from networks import Decoder
from utils import *
from torch import nn

class RSNet(nn.Module):
    def __init__(self):
        super(RSNet, self).__init__()

        self.encoder = VGG19.Encoder()
        self.transform = Transform.TransformNet()
        self.decoder = Decoder.Decoder()

    def forward(self, content, style):

        content_feats = self.encoder(content)
        style_feats = self.encoder(style)
        stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        out = self.decoder(stylized,content_feats[1],content_feats[0])
        out_feats = self.encoder(out)

        """FOR IDENTITY LOSSES"""
        c_c = self.decoder((self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4])),content_feats[1],content_feats[0])
        s_s = self.decoder((self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4])),style_feats[1],style_feats[0])
        c_c_feats = self.encoder(c_c)
        s_s_feats = self.encoder(s_s)

        return out, out_feats, content_feats, style_feats, c_c, s_s, c_c_feats, s_s_feats

if __name__ == "__main__":
    net=RSNet().cuda()
    x=torch.randn(1,3,256,256).cuda()
    y=torch.randn(1,3,256,256).cuda()
    res=net(x,y)
    print(res.shape)