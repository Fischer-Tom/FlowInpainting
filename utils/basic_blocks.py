import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionBlock(nn.Module):

    def __init__(self, tau, grads, **kwargs):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(tau), requires_grad=grads.get('tau'))
        self.pad = nn.ReplicationPad2d(1)
        grad_x, grad_y = self.get_weight(2)
        self.grad_x = nn.Parameter(grad_x, requires_grad=True)
        self.grad_y = nn.Parameter(grad_y, requires_grad=True)

    def forward(self, u, a,b,c):


        ux = F.conv2d(self.pad(u), self.grad_x, groups = u.size(1))
        uy = F.conv2d(self.pad(u), self.grad_y, groups= u.size(1))

        uxx = F.conv2d(a*self.pad(ux)+b*self.pad(uy), -self.grad_x.flip(2).flip(3), groups=ux.size(1))
        uyy = F.conv2d(b*self.pad(ux)+c*self.pad(uy), -self.grad_y.flip(2).flip(3), groups=uy.size(1))

        Au = uxx + uyy
        u = (u + self.tau * Au)

        return u
    def get_weight(self,c, h1=1, h2=1):
        hx = 1 / (1.4142*h1)
        hy = 1 / (1.4142*h2)
        weightx = torch.zeros((1, 1, 3, 3))
        weighty = torch.zeros((1, 1, 3, 3))
        weightx[0][0][1][1] = -hx
        weightx[0][0][1][2] = hx
        weighty[0][0][1][1] = -hy
        weighty[0][0][2][1] = hy

        image_weight_x = weightx.repeat(c, 1, 1, 1)
        image_weight_y = weighty.repeat(c, 1, 1, 1)

        return image_weight_x, image_weight_y

class WWW_DiffusionBlock(nn.Module):

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

