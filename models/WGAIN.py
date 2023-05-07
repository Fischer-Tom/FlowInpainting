import torch
import torch.nn as nn


class WGAIN(nn.Module):

    def __init__(self,**kwargs):
        super().__init__()

        self.G = Generator(8)
        self.C = Critic(3)

    def forward(self,I,M,u):

        return self.G(I,M,u)

class Generator(nn.Module):

    def __init__(self, in_ch):
        super().__init__()

        self.elu  = nn.ELU()
        self.mp = nn.MaxPool2d(2)
        self.us = nn.UpsamplingNearest2d(scale_factor=2)

        self.x11 = nn.Conv2d(in_channels=in_ch,out_channels=32, kernel_size=5, dilation=2, padding='same')
        self.x12 = nn.Conv2d(in_channels=in_ch,out_channels=32, kernel_size=5, dilation=5, padding='same')
        self.x13 = nn.Conv2d(in_channels=in_ch,out_channels=64, kernel_size=5, padding='same')

        self.x21 = nn.Conv2d(in_channels=128,out_channels=32, kernel_size=5, dilation=2, padding='same')
        self.x22 = nn.Conv2d(in_channels=128,out_channels=32, kernel_size=5, dilation=5, padding='same')
        self.x23 = nn.Conv2d(in_channels=128,out_channels=64, kernel_size=5, padding='same')

        self.x31 = nn.Conv2d(in_channels=128,out_channels=64, kernel_size=5, dilation=2, padding='same')
        self.x32 = nn.Conv2d(in_channels=128,out_channels=64, kernel_size=5, dilation=5, padding='same')
        self.x33 = nn.Conv2d(in_channels=128,out_channels=128, kernel_size=5, padding='same')


        self.y11 = nn.Conv2d(in_channels=256,out_channels=128, kernel_size=5, dilation=2, padding='same')
        self.y12 = nn.Conv2d(in_channels=256,out_channels=128, kernel_size=5, dilation=5, padding='same')
        self.y13 = nn.Conv2d(in_channels=256,out_channels=256, kernel_size=5, padding='same')

        self.y21 = nn.ConvTranspose2d(in_channels=768,out_channels=64, kernel_size=5, dilation=2, padding=4)
        self.y22 = nn.ConvTranspose2d(in_channels=768,out_channels=64, kernel_size=5, dilation=5, padding=10)
        self.y23 = nn.ConvTranspose2d(in_channels=768,out_channels=128, kernel_size=5, padding=2)

        self.y31 = nn.ConvTranspose2d(in_channels=384,out_channels=32, kernel_size=5, dilation=2, padding=4)
        self.y32 = nn.ConvTranspose2d(in_channels=384,out_channels=32, kernel_size=5, dilation=5, padding=10)
        self.y33 = nn.ConvTranspose2d(in_channels=384,out_channels=64, kernel_size=5, padding=2)

        self.y41 = nn.ConvTranspose2d(in_channels=256,out_channels=32, kernel_size=5, dilation=2, padding=4)
        self.y42 = nn.ConvTranspose2d(in_channels=256,out_channels=32, kernel_size=5, dilation=5, padding=10)
        self.y43 = nn.ConvTranspose2d(in_channels=256,out_channels=64, kernel_size=5, padding=2)

        self.y5 = nn.ConvTranspose2d(in_channels=128+in_ch,out_channels=8, kernel_size=3, padding=1)
        self.y = nn.ConvTranspose2d(in_channels=8,out_channels=2, kernel_size=3, padding=1)

    def forward(self, I,M,u,r):

        x = torch.cat((I,u,r,M),dim=1)

        x11 = self.elu(self.x11(x))
        x12 = self.elu(self.x12(x))
        x13 = self.elu(self.x13(x))

        x1o = torch.cat((x11,x12,x13), dim=1)
        x1  = self.mp(x1o)

        x21 = self.elu(self.x21(x1))
        x22 = self.elu(self.x22(x1))
        x23 = self.elu(self.x23(x1))

        x2o = torch.cat((x21, x22, x23), dim=1)
        x2 = self.mp(x2o)

        x31 = self.elu(self.x31(x2))
        x32 = self.elu(self.x32(x2))
        x33 = self.elu(self.x33(x2))

        x3o = torch.cat((x31, x32, x33), dim=1)
        x3 = self.mp(x3o)

        y11 = self.elu(self.y11(x3))
        y12 = self.elu(self.y12(x3))
        y13 = self.elu(self.y13(x3))

        y1o = torch.cat((y11,y12,y13), dim=1)
        y1 = self.us(y1o)
        y1 = torch.cat((y1,x3o),dim=1)

        y21 = self.elu(self.y21(y1))
        y22 = self.elu(self.y22(y1))
        y23 = self.elu(self.y23(y1))

        y2o = torch.cat((y21,y22,y23), dim=1)
        y2 = self.us(y2o)
        y2 = torch.cat((y2,x2o),dim=1)

        y31 = self.elu(self.y31(y2))
        y32 = self.elu(self.y32(y2))
        y33 = self.elu(self.y33(y2))

        y3o = torch.cat((y31,y32,y33), dim=1)
        y3 = self.us(y3o)
        y3 = torch.cat((y3,x1o),dim=1)

        y41 = self.elu(self.y41(y3))
        y42 = self.elu(self.y42(y3))
        y43 = self.elu(self.y43(y3))

        y4o = torch.cat((y41,y42,y43,x), dim=1)
        y5 = self.elu(self.y5(y4o))
        y = self.y(y5)

        return (1-M)*y + M*u


class Critic(nn.Module):

    def __init__(self,in_ch):
        super().__init__()
        self.lReLU = nn.LeakyReLU(0.3)
        self.x1 = nn.Conv2d(in_channels=in_ch,out_channels=64, kernel_size=5, stride=2)
        self.x2 = nn.Conv2d(in_channels=64,out_channels=128, kernel_size=5, stride=2)
        self.x3 = nn.Conv2d(in_channels=128,out_channels=256, kernel_size=5, stride=2)
        self.x4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2)
        self.x5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2)

        self.linear = nn.Linear(41472,1)
        #self.linear = nn.Linear(86528,1)
        #self.linear = nn.Linear(12800, 1)

    def forward(self,imp, M):
        x = torch.cat((imp,M),dim=1)
        x = self.lReLU(self.x1(x))
        x = self.lReLU(self.x2(x))
        x = self.lReLU(self.x3(x))
        x = self.lReLU(self.x4(x))
        x = self.lReLU(self.x5(x))

        x = torch.flatten(x,start_dim=1)
        x = self.linear(x)

        return x

