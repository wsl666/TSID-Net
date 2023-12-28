import torch
import torch.nn as nn

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.scale4 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU())
        self.scale3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU()
            )
        self.scale2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU()
            )
        self.scale1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU())
        self.out=nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)))



    def forward(self, x,y,z):

        out4 = self.scale4(x)
        out3 = self.scale3(out4)
        out2 = self.scale2(out3)
        out2 = out2 + y
        out1 = self.scale1(out2)
        out1 = out1 + z
        out  = self.out(out1)

        return out


if __name__ == "__main__":
    net=Decoder()
    x=torch.randn(1,512,32,32)
    y=torch.randn(1,128,128,128)
    z=torch.randn(1,64,256,256)
    res=net(x,y,z)
    print(res.shape)