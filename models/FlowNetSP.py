import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss_functions import EPE_Loss, MultiScale_EPE_Loss
from utils.basic_blocks import SimpleConv, SimpleUpConv

class FlowNetSP(nn.Module):

    def __init__(self, dim=64, **kwargs):
        super().__init__()
        self.encoder = Encoder(dim=dim)
        self.decoder = Decoder(in_ch=dim * 16)

    def forward(self, I1, Mask, Masked_Flow):
        stacked_images = torch.cat((I1, Mask, Masked_Flow), 1)
        encoder_out = self.encoder(stacked_images)
        out = self.decoder(encoder_out)
        return out

    def get_loss(self, pred, gt):
        if self.training:
            return MultiScale_EPE_Loss(pred, gt)
        return EPE_Loss(pred, gt)

    def get_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=100_000, gamma=0.5)

    def update_lr(self, scheduler, iter):
        if iter > 300_000:
            scheduler.step()

    def constrain_weight(self):
        pass

class Encoder(nn.Module):

    def __init__(self, dim, in_ch=6):
        super().__init__()
        dim = dim
        self.conv1 = SimpleConv(in_ch, dim, 7, 2, 3)
        self.conv2 = SimpleConv(dim, dim * 2, 5, 2, 2)
        self.conv3 = SimpleConv(dim * 2, dim * 4, 5, 2, 2)
        self.conv3_1 = SimpleConv(dim * 4, dim * 4, 3, 1, 1)
        self.conv4 = SimpleConv(dim * 4, dim * 8, 3, 2, 1)
        self.conv4_1 = SimpleConv(dim * 8, dim * 8, 3, 1, 1)
        self.conv5 = SimpleConv(dim * 8, dim * 8, 3, 2, 1)
        self.conv5_1 = SimpleConv(dim * 8, dim * 8, 3, 1, 1)
        self.conv6 = SimpleConv(dim * 8, dim * 16, 3, 2, 1)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2(x0)
        x2 = self.conv3_1(self.conv3(x1))
        x3 = self.conv4_1(self.conv4(x2))
        x4 = self.conv5_1(self.conv5(x3))
        x5 = self.conv6(x4)

        return [x0, x1, x2, x3, x4, x5]


class Decoder(nn.Module):

    def __init__(self, in_ch=1024):
        super().__init__()
        dim = in_ch
        self.deconv5 = SimpleUpConv(dim, dim // 2, 1, 2, 0, 1)
        self.deconv4 = SimpleUpConv(dim, dim // 4, 1, 2, 0, 1)
        self.deconv3 = SimpleUpConv(dim // 2 + dim // 4 + 2, dim // 8, 1, 2, 0, 1)
        self.deconv2 = SimpleUpConv(dim // 8 + dim // 4 + 2, dim // 16, 1, 2, 0, 1)
        self.deconv1 = SimpleUpConv(dim // 16 + dim // 8 + 2, dim // 32, 1, 2, 0, 1)
        self.deconv0 = SimpleUpConv(dim // 32 + dim // 16 + 2, dim // 32, 1, 2, 0, 1)

        self.flow5 = nn.Conv2d(in_ch, 2, 5, 1, 2, bias=True)
        self.flow4 = nn.Conv2d(dim // 4 + dim // 2 + 2, 2, 5, 1, 2, bias=True)
        self.flow3 = nn.Conv2d(dim // 8 + dim // 4 + 2, 2, 5, 1, 2, bias=True)
        self.flow2 = nn.Conv2d(dim // 16 + dim // 8 + 2, 2, 5, 1, 2, bias=True)
        self.flow1 = nn.Conv2d(dim // 32 + dim // 16 + 2, 2, 5, 1, 2, bias=True)
        self.flow0 = nn.Conv2d(dim // 32 + 2, 2, 5, 1, 2, bias=True)

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)


    def forward(self, x):
        [x0, x1, x2, x3, x4, x] = x

        x = torch.cat((self.deconv5(x), x4), dim=1)
        flow5 = self.flow5(x)

        x = torch.cat((self.deconv4(x), x3, self.upsample2(flow5)), dim=1)
        flow4 = self.flow4(x)

        x = torch.cat((self.deconv3(x), x2, self.upsample2(flow4)), dim=1)
        flow3 = self.flow3(x)

        x = torch.cat((self.deconv2(x), x1, self.upsample2(flow3)), dim=1)
        flow2 = self.flow2(x)

        x = torch.cat((self.deconv1(x), x0, self.upsample2(flow2)), dim=1)
        flow1 = self.flow1(x)

        x = torch.cat((self.deconv0(x), self.upsample2(flow1)), dim=1)
        flow0 = self.flow0(x)

        if self.training:
            return [flow0, flow1, flow2, flow3, flow4]
        return flow0
