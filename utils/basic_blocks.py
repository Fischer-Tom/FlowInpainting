import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionBlock(nn.Module):

    def __init__(self, tau, grads,alpha, **kwargs):
        super().__init__()

        self.pad = nn.ReplicationPad2d((0,1,0,1))
        grad_x1,grad_x2, grad_y1,grad_y2 = self.get_weight(2)
        self.grad_x1 = nn.Parameter(grad_x1, requires_grad=True)
        self.grad_x2 = nn.Parameter(grad_x2, requires_grad=True)
        self.grad_y1 = nn.Parameter(grad_y1, requires_grad=True)
        self.grad_y2 = nn.Parameter(grad_y2, requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=grads.get('alpha'))
        if self.alpha > 0:
            tau = 1.0 / (4.0 * (1.0 - self.alpha))
        self.tau = nn.Parameter(tau, requires_grad=grads.get('tau'))


    def forward(self, u, a,b,c):

        a = a[:,:,1:,1:]
        b = b[:, :, 1:, 1:]
        c = c[:, :, 1:, 1:]
        sign = torch.sign(b)
        alpha = self.alpha
        beta = (1. - 2. * alpha) * sign

        w1a = a * (1-alpha) / 2
        w2a = a * alpha / 2
        w1b = b * (1-beta) / 4
        w2b = b * (1+beta) / 4
        w1c = c * (1-alpha) / 2
        w2c = c * alpha / 2


        ux1 = self.pad(F.conv2d(self.pad(u), self.grad_x1, groups = u.size(1)))
        ux2 = self.pad(F.conv2d(self.pad(u), self.grad_x2, groups = u.size(1)))
        uy1 = self.pad(F.conv2d(self.pad(u), self.grad_y1, groups = u.size(1)))
        uy2 = self.pad(F.conv2d(self.pad(u), self.grad_y2, groups = u.size(1)))


        uxx1 = F.conv2d(w1a*ux1+w2a*ux2+w1b*uy1+w2b*uy2, -self.grad_x1.flip(2).flip(3), groups=ux1.size(1))
        uxx2 = F.conv2d(w2a*ux1+w1a*ux2+w2b*uy1+w1b*uy2, -self.grad_x2.flip(2).flip(3), groups=ux2.size(1))

        uyy1 = F.conv2d(w1b*ux1 + w2b*ux2 + w1c*uy1 + w2c*uy2, -self.grad_y1.flip(2).flip(3), groups=uy1.size(1))
        uyy2 = F.conv2d(w2b*ux1 + w1b*ux2 + w2c*uy1 + w1c*uy2, -self.grad_y2.flip(2).flip(3), groups=uy2.size(1))


        Au = 0.5*(uxx1 + uxx2 + uyy1 + uyy2)
        u = (u + self.tau * Au)
        if Au.isnan().any():
            print("Happened")
        return u
    def get_weight(self,c, h1=1, h2=1):
        hx = 1 / (h1)
        hy = 1 / (h2)
        weightx1 = torch.zeros((1, 1, 2, 2))
        weightx2 = torch.zeros((1, 1, 2, 2))
        weighty1 = torch.zeros((1, 1, 2, 2))
        weighty2 = torch.zeros((1, 1, 2, 2))
        weightx1[0][0][0][0] = -hx
        weightx1[0][0][0][1] = hx
        weightx2[0][0][1][0] = -hx
        weightx2[0][0][1][1] = hx
        weighty1[0][0][0][0] = -hy
        weighty1[0][0][1][0] = hy
        weighty2[0][0][0][1] = -hx
        weighty2[0][0][1][1] = hx

        image_weight_x1 = weightx1.repeat(c, 1, 1, 1)
        image_weight_x2 = weightx2.repeat(c, 1, 1, 1)
        image_weight_y1 = weighty1.repeat(c, 1, 1, 1)
        image_weight_y2 = weighty2.repeat(c, 1, 1, 1)


        return image_weight_x1, image_weight_x2, image_weight_y1, image_weight_y2

class WWWDiffusion(nn.Module):

    def __init__(self, tau, alpha, grads, **kwargs):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=grads.get('alpha'))
        if self.alpha > 0:
            tau = 1.0 / (4.0 * (1.0 - self.alpha))
        self.tau = nn.Parameter(tau, requires_grad=grads.get('tau'))
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, u, a, b, c):
        Au = self.get_delta(u, a, b, c)

        u = (u + self.tau * Au)

        return u

    def get_delta(self, u, a, b, c):
        _, _, h, w = u.shape
        h = h + 1
        w = w + 1
        u = self.pad(u)

        sign = torch.sign(b)
        alpha = self.alpha
        beta = (1. - 2. * alpha) * sign
        delta = alpha * (a + c) + beta * b
        wpo = 0.5 * (a[:, :, 1:w, 1:h] - delta[:, :, 1:w, 1:h] + a[:, :, 1:w, 0:h - 1] - delta[:, :, 1:w, 0:h - 1])
        wmo = 0.5 * (
                a[:, :, 0:w - 1, 1:h] - delta[:, :, 0:w - 1, 1:h] + a[:, :, 0:w - 1, 0:h - 1] - delta[:, :, 0:w - 1,
                                                                                                0:h - 1])
        wop = 0.5 * (c[:, :, 1:w, 1:h] - delta[:, :, 1:w, 1:h] + c[:, :, 0:w - 1, 1:h] - delta[:, :, 0:w - 1, 1:h])
        wom = 0.5 * (
                c[:, :, 1:w, 0:h - 1] - delta[:, :, 1:w, 0:h - 1] + c[:, :, 0:w - 1, 0:h - 1] - delta[:, :, 0:w - 1,
                                                                                                0:h - 1])
        wpp = 0.5 * (b[:, :, 1:w, 1:h] + delta[:, :, 1:w, 1:h])
        wmm = 0.5 * (b[:, :, 0:w - 1, 0:h - 1] + delta[:, :, 0:w - 1, 0:h - 1])
        wmp = 0.5 * (delta[:, :, 0:w - 1, 1:h] - b[:, :, 0:w - 1, 1:h])
        wpm = 0.5 * (delta[:, :, 1:w, 0:h - 1] - b[:, :, 1:w, 0:h - 1])
        woo = - wpo - wmo - wop - wom - wpp - wmm - wmp - wpm
        Au = (woo * u[:, :, 1:w, 1:h]
              + wpo * u[:, :, 2:, 1:h]
              + wmo * u[:, :, 0:w - 1, 1:h]
              + wop * u[:, :, 1:w, 2:]
              + wom * u[:, :, 1:w, 0:h - 1]
              + wpp * u[:, :, 2:, 2:]
              + wmm * u[:, :, 0:w - 1, 0:h - 1]
              + wpm * u[:, :, 2:, 0:h - 1]
              + wmp * u[:, :, 0:w - 1, 2:])

        return Au
class PeronaMalikDiffusivity(nn.Module):
    def __init__(self, contrast = 1.):
        super().__init__()
        self.contrast = nn.Parameter(torch.tensor(contrast), requires_grad=True)

    def forward(self, x):
        # Adapted to enforce contrast parameter >0
        divisor = (x * x) / (self.contrast * self.contrast + 1e-8)

        return 1 / (1 + divisor)

class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ks, stride=1, pad=1, pad_mode='zeros', bias=True, act=nn.ReLU()):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=ks,
                              stride=stride, padding=pad, padding_mode=pad_mode, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=ks,
                               stride=stride, padding=pad, padding_mode=pad_mode, bias=bias)
        self.act = act

    def forward(self, x):
        identity = x
        x = self.conv2(self.act(self.conv1(x)))
        return self.act(identity + x)
class SimpleConv(nn.Module):

    def __init__(self, in_ch, out_ch, ks, stride=1, pad=1, pad_mode='zeros', bias=True, act=nn.ReLU()):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks,
                              stride=stride, padding=pad, padding_mode=pad_mode, bias=bias)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class SimpleUpConv(nn.Module):

    def __init__(self, in_ch, out_ch, ks, stride=1, pad=1, output_padding=1, bias=True, act=nn.ReLU()):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks,
                                       stride=stride, padding=pad, output_padding=output_padding, bias=bias)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

