import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss_functions import EPE_Loss, MultiScale_EPE_Loss
from utils.basic_blocks import WWWDiffusion, DiffusionBlock

class InpaintingFlowNet(nn.Module):

    def __init__(self, diffusion_position, disc, dim=64, **kwargs):
        super().__init__()
        self.disc = disc
        self.flow_encoder = FlowEncoder(diffusion_position,disc, dim=dim, **kwargs)
        self.image_encoder = ImageEncoder(dim=dim, **kwargs)
        self.decoder = Decoder(diffusion_position,disc, dim, **kwargs)
        with torch.no_grad():
            self.constrain_weight()

    def forward(self, I1, Mask, Masked_Flow):
        stacked = torch.cat((Mask, Masked_Flow), 1)
        image_features = self.image_encoder(I1)
        encoder_out = self.flow_encoder(stacked, image_features)
        out = self.decoder(encoder_out, image_features)

        if self.training:
            out[0] = (1 - Mask) * out[0] + Mask * Masked_Flow
        else:
            out = (1 - Mask) * out + Mask * Masked_Flow
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
        if self.disc == 'resnet' or self.disc == 'WWW':
            return
        if 'encoder' in self.flow_encoder.mode :
            for name, module_block in self.flow_encoder.named_children():
                if name.startswith('dif'):
                    for fsi_block in module_block:
                        for block in fsi_block.blocks:
                            sqrtc = torch.sqrt(torch.tensor(block.grad_x1.shape[0]))
                            block.grad_x1 /= sqrtc*block.grad_x1.abs().sum((2,3), keepdims=True)
                            block.grad_y1 /= sqrtc*block.grad_y1.abs().sum((2,3), keepdims=True)
                            block.grad_x2 /= sqrtc * block.grad_x2.abs().sum((2, 3), keepdims=True)
                            block.grad_y2 /= sqrtc * block.grad_y2.abs().sum((2, 3), keepdims=True)
                            #block.grad_xx.weight /= sqrtc*block.grad_xx.weight.abs().sum((2,3), keepdims=True)
                            #block.grad_yy.weight /= sqrtc*block.grad_yy.weight.abs().sum((2,3), keepdims=True)
        else:
            for name, module_block in self.decoder.named_children():
                if name.startswith('dif'):
                    for fsi_block in module_block:
                        for block in fsi_block.blocks:
                            sqrtc = torch.sqrt(torch.tensor(block.grad_x1.shape[0]))
                            block.grad_x1 /= sqrtc*block.grad_x1.abs().sum((2,3), keepdims=True)
                            block.grad_y1 /= sqrtc*block.grad_y1.abs().sum((2,3), keepdims=True)
                            block.grad_x2 /= sqrtc * block.grad_x2.abs().sum((2, 3), keepdims=True)
                            block.grad_y2 /= sqrtc * block.grad_y2.abs().sum((2, 3), keepdims=True)
                            #block.grad_xx.weight /= sqrtc*block.grad_xx.weight.abs().sum((2,3), keepdims=True)
                            #block.grad_yy.weight /= sqrtc*block.grad_yy.weight.abs().sum((2,3), keepdims=True)


class FlowEncoder(nn.Module):

    def __init__(self, diffusion, disc,dim, in_ch=3, **kwargs):
        super().__init__()
        self.mode = diffusion
        self.step = kwargs['step']
        self.disc = disc
        self.conv1 = SimpleConv(in_ch, dim, 7, 2, 3)
        self.conv2 = SimpleConv(dim, dim * 2, 5, 2, 2)
        self.conv3 = SimpleConv(dim * 2, dim * 4, 5, 2, 2)
        self.conv4 = SimpleConv(dim * 4, dim * 8, 3, 2, 1)
        self.conv5 = SimpleConv(dim * 8, dim * 8, 3, 2, 1)
        self.conv6 = SimpleConv(dim * 8, dim * 8, 3, 2, 1)

        if 'encoder' in self.mode:
            if self.disc == 'resnet':
                self.dif0 = nn.Sequential(End_ResidualBlock(dim*2,dim,3,1,1),*[ResidualBlock(dim,dim,3,1,1) for _ in range(self.step)])#FSI_Block(dim, dim, **kwargs)
                self.dif1 = nn.Sequential(End_ResidualBlock(dim*2*2,dim*2,3,1,1),*[ResidualBlock(dim*2,dim*2,3,1,1) for _ in range(self.step)])#FSI_Block(dim*2, dim*2, **kwargs)
                self.dif2 = nn.Sequential(End_ResidualBlock(dim*4*2,dim*4,3,1,1),*[ResidualBlock(dim*4,dim*4,3,1,1) for _ in range(self.step)])#FSI_Block(dim*4, dim*4, **kwargs)
                self.dif3 = nn.Sequential(End_ResidualBlock(dim*(8+4),dim*8,3,1,1),*[ResidualBlock(dim*8,dim*8,3,1,1) for _ in range(self.step)])#FSI_Block(dim * 8, dim*4, **kwargs)
                self.dif4 = nn.Sequential(End_ResidualBlock(dim*(8+4),dim*8,3,1,1),*[ResidualBlock(dim*8,dim*8,3,1,1) for _ in range(self.step)])#FSI_Block(dim*8, dim*4, **kwargs)
            else:
                self.dif0 = nn.ModuleList([torch.jit.script(FSI_Block(dim, dim,disc, **kwargs))])
                self.dif1 = nn.ModuleList([torch.jit.script(FSI_Block(dim*2, dim*2,disc, **kwargs))])
                self.dif2 = nn.ModuleList([torch.jit.script(FSI_Block(dim*4, dim*4,disc, **kwargs))])
                self.dif3 = nn.ModuleList([torch.jit.script(FSI_Block(dim * 8, dim*4,disc, **kwargs))])
                self.dif4 = nn.ModuleList([torch.jit.script(FSI_Block(dim*8, dim*4,disc, **kwargs))])

    def forward(self, x, image_features):
        [i0, i1, i2, i3, i4] = image_features

        x0 = self.conv1(x)
        if 'encoder' in self.mode:
            if self.disc == 'resnet':
                xin = torch.cat((x0,i0),dim=1)
                x0 = self.dif0(xin)
            else:
                for block in self.dif0:
                    x0 = block(x0,i0)




        x1 = self.conv2(x0)
        if 'encoder' in self.mode:
            if self.disc == 'resnet':
                xin = torch.cat((x1,i1),dim=1)
                x1 = self.dif1(xin)
            else:
                for block in self.dif1:
                    x1 = block(x1,i1)



        x2 = self.conv3(x1)
        if 'encoder' in self.mode:
            if self.disc == 'resnet':
                xin = torch.cat((x2,i2),dim=1)
                x2 = self.dif2(xin)
            else:
                for block in self.dif2:
                    x2 = block(x2,i2)



        x3 = self.conv4(x2)
        if 'encoder' in self.mode:
            if self.disc == 'resnet':
                xin = torch.cat((x3,i3),dim=1)
                x3 = self.dif3(xin)
            else:
                for block in self.dif3:
                    x3 = block(x3,i3)



        x4 = self.conv5(x3)
        if 'encoder' in self.mode:
            if self.disc == 'resnet':
                xin = torch.cat((x4,i4),dim=1)
                x4 = self.dif4(xin)
            else:
                for block in self.dif4:
                    x4 = block(x4,i4)



        x5 = self.conv6(x4)
        return [x0, x1, x2, x3, x4, x5]

class ImageEncoder(nn.Module):

    def __init__(self, dim, in_ch=3, **kwargs):
        super().__init__()
        dim = dim
        self.conv1 = SimpleConv(in_ch, dim, 7, 2, 3)
        self.conv2 = SimpleConv(dim, dim * 2, 5, 2, 2)
        self.conv3 = SimpleConv(dim * 2, dim * 4, 5, 2, 2)
        self.conv3_1 = SimpleConv(dim * 4, dim * 4, 3, 1, 1)
        self.conv4 = SimpleConv(dim * 4, dim * 4, 3, 2, 1)
        self.conv4_1 = SimpleConv(dim * 4, dim * 4, 3, 1, 1)
        self.conv5 = SimpleConv(dim * 4, dim * 4, 3, 2, 1)
        self.conv5_1 = SimpleConv(dim * 4, dim * 4, 3, 1, 1)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2(x0)
        x2 = self.conv3_1(self.conv3(x1))
        x3 = self.conv4_1(self.conv4(x2))
        x4 = self.conv5_1(self.conv5(x3))

        return [x0, x1, x2, x3, x4]
class Decoder(nn.Module):

    def __init__(self, diffusion,disc, dim, **kwargs):
        super().__init__()
        self.mode = diffusion
        self.disc = disc
        self.step = kwargs['step']
        self.deconv5 = SimpleUpConv(dim*8, dim * 8, 1, 2, 0, 1)
        self.deconv4 = SimpleUpConv(dim*20, dim * 8, 1, 2, 0, 1)
        self.deconv3 = SimpleUpConv(dim*20+2, dim * 8, 1, 2, 0, 1)
        self.deconv2 = SimpleUpConv(dim*16+2, dim * 8, 1, 2, 0, 1)
        self.deconv1 = SimpleUpConv(dim*12+2, dim * 4, 1, 2, 0, 1)
        self.deconv0 = SimpleUpConv(dim*6+2, dim * 4, 1, 2, 0, 1)

        self.flow4 = nn.Conv2d(dim*20, 2, 5, 1, 2, bias=True)
        self.flow3 = nn.Conv2d(dim*20 + 2, 2, 5, 1, 2, bias=True)
        self.flow2 = nn.Conv2d(dim*16 + 2, 2, 5, 1, 2, bias=True)
        self.flow1 = nn.Conv2d(dim*12 + 2, 2, 5, 1, 2, bias=True)
        self.flow0 = nn.Conv2d(dim*6 + 2, 2, 5, 1, 2, bias=True)
        self.out   = nn.Conv2d(dim*4+2, 2, 5, 1, 2, bias=True)

        if 'decoder' in self.mode:
            if self.disc == 'resnet':
                self.dif4 = nn.Sequential(End_ResidualBlock(dim*12,dim*8,3,1,1), *[ResidualBlock(dim * 8, dim*8, 3, 1, 1) for _ in range(self.step)]) # torch.jit.script(FSI_Block(dim, dim, **kwargs)
                self.dif3 = nn.Sequential(End_ResidualBlock(dim*12,dim*8,3,1,1),*[ResidualBlock(dim * 8, dim*8, 3, 1, 1) for _ in range(self.step)]) # torch.jit.script(FSI_Block(dim*2, dim*2, **kwargs)
                self.dif2 = nn.Sequential(End_ResidualBlock(dim*12,dim*8,3,1,1),*[ResidualBlock(dim * 8, dim*8, 3, 1, 1) for _ in range(self.step)])  # torch.jit.script(FSI_Block(dim*4, dim*4, **kwargs)
                self.dif1 = nn.Sequential(End_ResidualBlock(dim*10,dim*8,3,1,1),*[ResidualBlock(dim * 8, dim*8, 3, 1, 1) for _ in range(self.step)])  # torch.jit.script(FSI_Block(dim * 8, dim*4, **kwargs)
                self.dif0 = nn.Sequential(End_ResidualBlock(dim*5,dim*4,3,1,1),*[ResidualBlock(dim * 4, dim*4, 3, 1, 1) for _ in range(self.step)])  # torch.jit.script(FSI_Block(dim*8, dim*4, **kwargs)
            else:
                self.dif4 = nn.ModuleList([torch.jit.script(FSI_Block(dim*8, dim*4,disc, **kwargs))])
                self.dif3 = nn.ModuleList([torch.jit.script(FSI_Block(dim*8, dim*4,disc, **kwargs))])
                self.dif2 = nn.ModuleList([torch.jit.script(FSI_Block(dim*8, dim*4,disc, **kwargs))])
                self.dif1 = nn.ModuleList([torch.jit.script(FSI_Block(dim*8, dim*2,disc, **kwargs))])
                self.dif0 = nn.ModuleList([torch.jit.script(FSI_Block(dim*4, dim,disc, **kwargs))])


        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, flow_features, image_features):
        [x0, x1, x2, x3, x4, x] = flow_features
        [i0, i1, i2, i3, i4] = image_features


        conv = self.deconv5(x)
        if 'decoder' in self.mode:
            if self.disc == 'resnet':
                xin = torch.cat((conv,i4),dim=1)
                conv = self.dif4(xin)
            else:
                for block in self.dif4:
                    conv = block(conv,i4)




        x = torch.cat((conv, i4, x4), dim=1)
        flow4 = self.flow4(x)

        conv = self.deconv4(x)
        if 'decoder' in self.mode:
            if self.disc == 'resnet':
                xin = torch.cat((conv,i3),dim=1)
                conv = self.dif3(xin)
            else:
                for block in self.dif3:
                    conv = block(conv,i3)



        x = torch.cat((conv, i3, x3, self.upsample2(flow4)), dim=1)
        flow3 = self.flow3(x)

        conv = self.deconv3(x)
        if 'decoder' in self.mode:
            if self.disc == 'resnet':
                xin = torch.cat((conv,i2),dim=1)
                conv = self.dif2(xin)
            else:
                for block in self.dif2:
                    conv = block(conv,i2)



        x = torch.cat((conv, i2, x2, self.upsample2(flow3)), dim=1)
        flow2 = self.flow2(x)

        conv = self.deconv2(x)
        if 'decoder' in self.mode:
            if self.disc == 'resnet':
                xin = torch.cat((conv, i1), dim=1)
                conv = self.dif1(xin)
            else:
                for block in self.dif1:
                    conv = block(conv, i1)


        x = torch.cat((conv, i1, x1, self.upsample2(flow2)), dim=1)
        flow1 = self.flow1(x)

        conv = self.deconv1(x)
        if 'decoder' in self.mode:
            if self.disc == 'resnet':
                xin = torch.cat((conv, i0), dim=1)
                conv = self.dif0(xin)
            else:
                for block in self.dif0:
                    conv = block(conv, i0)



        x = torch.cat((conv, i0, x0, self.upsample2(flow1)), dim=1)
        flow0 = self.flow0(x)

        x = torch.cat((self.deconv0(x), self.upsample2(flow0)), dim=1)
        out = self.out(x)

        if self.training:
            return [out, flow0, flow1, flow2, flow3, flow4]
        return out

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



class FSI_Block(nn.Module):

    def __init__(self, flow_c,i_c,disc,alpha, **kwargs):
        super().__init__()
        step = kwargs['step']
        self.disc = disc
        self.alpha = torch.tensor(alpha)
        self.learned_mode = kwargs['learned_mode']
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor((4 * i + 2) / (2 * i + 3)),
                                                     requires_grad=kwargs['grads']['alphas']) for i in range(step)])
        self.blocks = nn.ModuleList([DiffusionBlock(flow_c,**kwargs) for _ in range(step)]) if disc == "DB" else \
            nn.ModuleList([WWWDiffusion(**kwargs) for _ in range(step)])
        self.tensor = DepthwiseSeparableConvolution(in_ch=flow_c + i_c, out_ch=self.learned_mode,disc=self.disc, ks=3)


    def forward(self, u, f):
        dt_in = torch.cat((u,f), dim=1)
        (D_a,D_b,D_c), alpha = self.tensor(dt_in,self.alpha)
        u_prev = u
        for a, block in zip(self.alphas, self.blocks):
            u_new = a * block(u, D_a,D_b,D_c,alpha) + (1 - a) * u_prev
            u_prev = u
            u = u_new

        return u


class DepthwiseSeparableConvolution(nn.Module):

    def __init__(self, in_ch, out_ch,disc, ks):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks, padding='same', groups=1, bias=True)
        self.zero_pad = nn.ZeroPad2d(1) if disc=='WWW' else nn.ZeroPad2d((1,0,1,0))
        self.pad = nn.ReplicationPad2d(1) if disc=='WWW' else nn.ReplicationPad2d((1,0,1,0))
        self.diffusivity = PeronaMalikDiffusivity()
        self.out_mode = out_ch


    def get_diffusion_tensor(self, v11, v12, act1, act2):
        lam1 = act1
        lam2 = act2

        a = v11 * v11 * lam1 + v12 * v12 * lam2
        c = lam1 + lam2 - a
        b = v11 * v12 * (lam1 - lam2)

        a = self.zero_pad(a)
        b = self.zero_pad(b)
        c = self.zero_pad(c)
        return a, b, c
    def forward(self, x,alpha):
        x = self.conv(x)

        V = F.normalize(x[:, 0:2, :, :], dim=1, p=2.)
        mu = self.diffusivity(x[:, 2:4, :, :])
        DT = self.get_diffusion_tensor(V[:, 0:1, :, :], V[:, 1:2, :, :], mu[:,0:1,:,:], mu[:,1:2,:,:])

        if self.out_mode == 5:
            alpha = self.pad(torch.sigmoid(x[:,4:5,:,:])/2)

        return DT, alpha
class PeronaMalikDiffusivity(nn.Module):
    def __init__(self, contrast = 1.):
        super().__init__()
        self.contrast = nn.Parameter(torch.tensor(contrast), requires_grad=True)

    def forward(self, x):

        # Adapted to enforce contrast parameter >0
        divisor = (x * x) / (self.contrast * self.contrast+1e-8)

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

class End_ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ks, stride=1, pad=1, pad_mode='zeros', bias=True, act=nn.ReLU()):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=ks,
                              stride=stride, padding=pad, padding_mode=pad_mode, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks,
                               stride=stride, padding=pad, padding_mode=pad_mode, bias=bias)
        self.act = act

    def forward(self, x):
        x = self.act(self.conv2(self.act(self.conv1(x))))
        return x