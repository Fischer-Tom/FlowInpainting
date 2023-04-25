import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss_functions import EPE_Loss
from utils.basic_blocks import PeronaMalikDiffusivity

class InpaintingNetwork(nn.Module):

    def __init__(self, super_steps, steps, disc, **kwargs):
        super().__init__()
        assert len(steps) == len(super_steps)
        self.max_pool = nn.MaxPool2d(2)
        self.sum_pool = nn.AvgPool2d(2, divisor_override=1)
        self.av_pool = nn.AvgPool2d(2)
        self.resolutions = len(self.super_steps)
        self.blocks = nn.ModuleList([FSI_Block(s, **kwargs) for s in steps])
        self.output_resolution = self.resolutions - 1
        self.disc = disc

    def get_loss(self, pred, gt):
        return EPE_Loss(pred, gt)

    def get_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10_000, gamma=0.5)

    def update_lr(self, scheduler, iter):
        if iter > 50_000:
            scheduler.step()

    def get_masks(self, I, m, u):
        masks = [m]
        flows = [u]
        images = [I]

        for i in range(0, self.resolutions - 1):
            u = self.get_flow(u, m)
            m = self.max_pool(m)
            I = self.av_pool(I)
            flows.append(u)
            masks.append(m)
            images.append(I)
        return reversed(flows[self.train_resolutions:]), reversed(masks[self.train_resolutions:]), reversed(
            images[self.train_resolutions:])

    def get_flow(self, u, mask):
        divisor = self.sum_pool(mask)
        flow = self.sum_pool(u)
        return flow / torch.clamp(divisor, min=1)

    def forward(self, I, m, u, pre_guess=None):
        flows, masks, images = self.get_masks(I, m, u)
        u = pre_guess
        for j, (f,c,i,block) in enumerate(zip(flows, masks, images, self.blocks)):
            if u is None: u = f
            u = (1. - c) * u + c * f
            u = block(u, f, c, i)
            if self.training and j == self.output_resolution:
                break
            if j < self.resolutions - 1:
                u = F.interpolate(u, scale_factor=2, mode='bilinear')

        return u
    def constrain_weight(self):
        if self.disc == 'WWW':
            pass
        with torch.no_grad():
            for res_block in self.blocks:
                for fsi_block in res_block.blocks:
                    for block in fsi_block.blocks:
                        block.grad_x /= 1.4142*block.grad_x.abs().sum((2,3), keepdims=True)
                        block.grad_y /= 1.4142*block.grad_y.abs().sum((2,3), keepdims=True)
class FSI_Block(nn.Module):

    def __init__(self, step, disc, **kwargs):
        super().__init__()
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor((4 * i + 2) / (2 * i + 3)),
                                                     requires_grad=kwargs['grads']['alphas']) for i in range(step)])
        self.blocks = nn.ModuleList([DiffusionBlock(**kwargs) for _ in range(step)])
        self.g = Diffusivity(mul=2)
        self.zero_pad = nn.ZeroPad2d(1)

    def forward(self, u, f, c, i):
        v11,v12,lam1,lam2 = self.g(i)

        D_a = v11 * v11 * lam1 + v12 * v12 * lam2
        D_c = lam1 + lam2 - D_a
        D_b = v11 * v12 * (lam1 - lam2)
        D_a = self.zero_pad(D_a)
        D_b = self.zero_pad(D_b)
        D_c = self.zero_pad(D_c)

        u_prev = u
        for alpha, block in zip(self.alphas, self.blocks):
            diffused = alpha * block(u, f, c, i, D_a, D_b, D_c) + (1 - alpha) * u_prev
            u_new = (1. - c) * diffused + c * f
            u_prev = u
            u = u_new

        return u
class Diffusivity(nn.Module):

    def __init__(self, mul):
        super().__init__()

        self.conv_d1 = nn.Conv2d(in_channels=3, out_channels=3 * mul, kernel_size=3, bias=True, padding=1, stride=2)
        self.conv_1 = nn.Conv2d(in_channels=3 * mul, out_channels=3 * mul, kernel_size=3, bias=True, padding=1,
                                stride=1)
        self.conv_d2 = nn.Conv2d(in_channels=3 * mul, out_channels=3 * mul * mul, kernel_size=3, bias=True, padding=1,
                                 stride=2)
        self.conv_2 = nn.Conv2d(in_channels=3 * mul * mul, out_channels=3 * mul * mul, kernel_size=3, bias=True,
                                padding=1, stride=1)
        self.conv_d3 = nn.Conv2d(in_channels=3 * mul * mul, out_channels=3 * mul * mul * mul, kernel_size=3, bias=True,
                                 padding=1, stride=2)

        self.conv_u3 = nn.ConvTranspose2d(in_channels=3 * mul * mul * mul, out_channels=3 * mul * mul, kernel_size=3,
                                          stride=2, padding=1, output_padding=1, bias=True)
        self.conv_u2 = nn.ConvTranspose2d(in_channels=3 * mul * mul + 3 * mul * mul, out_channels=3 * mul,
                                          kernel_size=3,
                                          stride=2, padding=1, output_padding=1, bias=True)
        self.conv_u1 = nn.ConvTranspose2d(in_channels=3 * mul + 3 * mul, out_channels=3 * mul, kernel_size=3,
                                          stride=2, padding=1, output_padding=1, bias=True)

        self.out_conv = nn.Conv2d(in_channels=3 * mul, out_channels=4, kernel_size=3, bias=True, padding=1, stride=1)
        self.diff = PeronaMalikDiffusivity()
        self.relu = nn.ReLU()

    def forward(self, I):
        b, _, w, h = I.shape

        x = self.relu(self.conv_d1(I))
        x1 = self.relu(self.conv_1(x))
        x = self.relu(self.conv_d2(x1))
        x2 = self.relu(self.conv_2(x))
        x = self.relu(self.conv_d3(x2))

        x = self.relu(self.conv_u3(x))
        x = torch.cat((x2, x), dim=1)
        x = self.relu(self.conv_u2(x))
        x = torch.cat((x1, x), dim=1)
        x = self.relu(self.conv_u1(x))
        x = self.diff(self.out_conv(x))

        V = F.normalize(x[:, 0:2, :, :], dim=1, p=2.)
        mu1 = self.diff(x[:, 2:3, :, :])
        mu2 = self.diff(x[:, 3:4, :, :])
        return V[:, 0:1, :, :], V[:, 1:2, :, :], mu1, mu2

