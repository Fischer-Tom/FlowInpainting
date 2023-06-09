import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss_functions import EPE_Loss, MultiScale_EPE_Loss
from utils.basic_blocks import WWWDiffusion, DiffusionBlock, DiffusivityModule, PeronaMalikDiffusivity

class InpaintingFlowNet(nn.Module):

    def __init__(self, diffusion_position, disc,dim=64, **kwargs):
        super().__init__()
        self.disc = disc
        self.flow_encoder = FlowEncoder(diffusion_position,disc, dim=dim,in_ch=2, **kwargs)
        self.image_encoder = DiffusivityModule(dim=44,learned_mode=kwargs['learned_mode'])
        self.decoder = Decoder(diffusion_position,disc, dim, **kwargs)
        self.mask_down1 = nn.Conv2d(1, 1, 3, 2, 1)
        self.mask_down2 = nn.Conv2d(1, 1, 3, 2, 1)
        self.mask_down3 = nn.Conv2d(1, 1, 3, 2, 1)
        self.sig = nn.Sigmoid()
        with torch.no_grad():
            self.constrain_weight()

    def forward(self, I1, Mask, Masked_Flow):

        image_features = self.image_encoder(I1)
        encoder_out = self.flow_encoder(Masked_Flow)

        dM1 = self.sig(self.mask_down1(Mask))
        dM2 = self.sig(self.mask_down2(dM1))
        dM3 = self.sig(self.mask_down3(dM2))

        out = self.decoder(encoder_out, image_features,[Mask,dM1,dM2,dM3])


        return out

    def get_loss(self, pred, gt):
        #if self.training:
        #    return MultiScale_EPE_Loss(pred, gt)
        return EPE_Loss(pred, gt)

    def get_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=100_000, gamma=0.5)

    def update_lr(self, scheduler, iter):
        if iter > 300_000:
            scheduler.step()
    def constrain_weight(self):
        if self.disc == 'resnet' or self.disc == 'WWW' or True:
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
        self.conv1 = SimpleConv(in_ch, dim, 3, 2, 1)
        self.conv2 = SimpleConv(dim, dim * 2, 3, 2, 1)
        self.conv3 = SimpleConv(dim * 2, dim * 4, 3, 2, 1)
        self.conv4 = SimpleConv(dim * 4, dim * 8, 3, 2, 1)

    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)
        return x

class ImageEncoder(nn.Module):

    def __init__(self, dim, in_ch=3, **kwargs):
        super().__init__()
        dim = dim
        self.conv1 = SimpleConv(in_ch, dim, 5, 2, 2)
        self.conv2 = SimpleConv(dim, dim, 3, 1, 1)
        self.conv2_1 = SimpleConv(dim, dim * 2, 3, 2, 1)

        self.conv3 = SimpleConv(dim * 2, dim * 4, 3, 2, 1)
        self.conv3_1 = SimpleConv(dim * 4, dim * 4, 3, 1, 1)
        self.conv4 = SimpleConv(dim * 4, dim * 4, 3, 2, 1)
        self.conv4_1 = SimpleConv(dim * 4, dim * 4, 3, 1, 1)
        self.conv5 = SimpleConv(dim * 4, dim * 4, 3, 2, 1)
        self.conv5_1 = SimpleConv(dim * 4, dim * 4, 3, 1, 1)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2_1(self.conv2(x0))
        x2 = self.conv3_1(self.conv3(x1))
        x3 = self.conv4_1(self.conv4(x2))
        x4 = self.conv5_1(self.conv5(x3))

        return [x0, x1, x2, x3, x4]
class Decoder(nn.Module):

    def __init__(self, diffusion,disc, dim, **kwargs):
        super().__init__()
        self.mode = diffusion
        self.disc = disc
        self.steps = kwargs["steps"]
        self.deconv3 = SimpleUpConv(dim*8, dim * 8, 1, 2, 0, 1)
        self.deconv2 = SimpleUpConv(dim*8, dim * 8, 1, 2, 0, 1)
        self.deconv1 = SimpleUpConv(dim*8, dim * 4, 1, 2, 0, 1)
        self.deconv0 = SimpleUpConv(dim*4, dim * 4, 1, 2, 0, 1)


        self.out   = nn.Conv2d(dim*4, 2, 5, 1, 2, bias=True)


        kwargs['step'] = self.steps[0]
        self.dif3 = nn.ModuleList([FSI_Block(dim*8, dim*4,disc, **kwargs)])
        kwargs['step'] = self.steps[1]
        self.dif2 = nn.ModuleList([FSI_Block(dim*8, dim*4,disc, **kwargs)])
        kwargs['step'] = self.steps[2]
        self.dif1 = nn.ModuleList([FSI_Block(dim*4, dim*2,disc, **kwargs)])
        kwargs['step'] = self.steps[3]
        self.dif0 = nn.ModuleList([FSI_Block(2, dim,disc, **kwargs)])


    def forward(self, x, image_features,masks):
        [i3,i2, i1, i0] = image_features
        [dM0, dM1, dM2, dM3] = masks

        x = self.deconv3(x)
        for block in self.dif3:
            x = block(x,i3,dM3)
        x = self.deconv2(x)
        for block in self.dif2:
            x = block(x, i2,dM2)
        x = self.deconv1(x)
        for block in self.dif1:
            x = block(x, i1,dM1)
        x = self.deconv0(x)
        out = self.out(x)
        for block in self.dif0:
            out = block(out, i0,dM0)
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
        self.zero_pad = nn.ZeroPad2d(1) if self.learned_mode == 'WWW' else nn.ZeroPad2d((1,0,1,0))
        self.pad = nn.ReplicationPad2d(1) if self.learned_mode == 'WWW' else nn.ReplicationPad2d((1,0,1,0))
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor((4 * i + 2) / (2 * i + 3)),
                                                     requires_grad=kwargs['grads']['alphas']) for i in range(step)])
        self.blocks = nn.ModuleList([DiffusionBlock(flow_c,**kwargs) for _ in range(step)]) if disc == "DB" else \
            nn.ModuleList([WWWDiffusion(**kwargs) for _ in range(step)])
        self.g = PeronaMalikDiffusivity()


    def forward(self, u, D,c):
        D_a,D_b,D_c, alpha = self.get_DiffusionTensor(D)
        u_prev = u
        f = u
        for a, block in zip(self.alphas, self.blocks):
            diffused = a * block(u, D_a, D_b, D_c,alpha) + (1 - a) * u_prev
            u_new = (1. - c) * diffused + c * f
            u_prev = u
            u = u_new
        return u
    def get_DiffusionTensor(self, x):
        V = F.normalize(x[:, 0:2, :, :], dim=1, p=2.)
        v11 = V[:,0:1,:,:]
        v12 = V[:,1:2,:,:]

        mu1 = self.g(x[:, 2:3, :, :])
        mu2 = self.g(x[:, 3:4, :, :])

        a = v11 * v11 * mu1 + v12 * v12 * mu2
        c = mu1 + mu2 - a
        b = v11 * v12 * (mu1 - mu2)
        a = self.zero_pad(a)
        b = self.zero_pad(b)
        c = self.zero_pad(c)

        if self.learned_mode == 5:
            alpha = self.pad(torch.sigmoid(x[:,4:5,:,:])/2)
        else:
            alpha = self.alpha
        return a,b,c, alpha

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
