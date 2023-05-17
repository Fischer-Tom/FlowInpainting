import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss_functions import EPE_Loss
from utils.basic_blocks import PeronaMalikDiffusivity, WWWDiffusion, DiffusionBlock, DiffusivityModule



class InpaintingNetwork(nn.Module):

    def __init__(self, dim,steps,disc,alpha, learned_mode,**kwargs):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.sum_pool = nn.AvgPool2d(2, divisor_override=1)
        self.av_pool = nn.AvgPool2d(2)
        self.output_resolution = len(steps) - 1
        self.alpha = torch.tensor(alpha)
        self.learned_mode = learned_mode
        self.blocks = nn.ModuleList([FSI_Block(s, disc, **kwargs) for s in steps])
        self.disc = disc
        self.DM = DiffusivityModule(dim,self.learned_mode)
        self.g = PeronaMalikDiffusivity()
        self.zero_pad = nn.ZeroPad2d(1)
        with torch.no_grad():
            self.constrain_weight()
        self.pad = nn.ReplicationPad2d(1)




    def get_loss(self, pred, gt):
        return EPE_Loss(pred, gt)

    def get_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=100_000, gamma=0.5)

    def update_lr(self, scheduler, iter):
        if iter > 300_000:
            scheduler.step()

    def get_masks(self, I, m, u):
        masks = [m]
        flows = [u]
        image_features = self.DM(I)
        for i in range(0, self.output_resolution):
            u = self.get_flow(u, m)
            m = self.max_pool(m)
            flows.append(u)
            masks.append(m)
        return reversed(flows), reversed(masks), image_features

    def get_flow(self, u, mask):
        divisor = self.sum_pool(mask)
        flow = self.sum_pool(u)
        return flow / torch.clamp(divisor, min=1)

    def forward(self, I, m, u, pre_guess=None):
        flows, masks, image_features = self.get_masks(I, m, u)
        u = pre_guess
        for j, (f,c,i,block) in enumerate(zip(flows, masks, image_features, self.blocks)):
            if u is None: u = f
            Da,Db,Dc,alpha = self.get_DiffusionTensor(i)

            u = (1. - c) * u + c * f
            u = block(u, f, c, Da,Db,Dc,alpha)
            if self.training and j == self.output_resolution:
                break
            if j < self.output_resolution:
                u = F.interpolate(u, scale_factor=2, mode='bilinear')

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

    def constrain_weight(self):
        if self.disc == 'DB':
            with torch.no_grad():
                for fsi_block in self.blocks:
                    for block in fsi_block.blocks:
                        block.grad_x1 /= 1.4142*block.grad_x1.abs().sum((2,3), keepdims=True)
                        block.grad_y1 /= 1.4142*block.grad_y1.abs().sum((2,3), keepdims=True)
                        block.grad_x2 /= 1.4142*block.grad_x2.abs().sum((2,3), keepdims=True)
                        block.grad_y2 /= 1.4142*block.grad_y2.abs().sum((2,3), keepdims=True)
class FSI_Block(nn.Module):

    def __init__(self, timesteps, disc, **kwargs):
        super().__init__()
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor((4 * i + 2) / (2 * i + 3)),
                                                     requires_grad=kwargs['grads']['alphas']) for i in range(timesteps)])
        self.blocks = nn.ModuleList([torch.jit.script(DiffusionBlock(2,**kwargs)) for _ in range(timesteps)]) if disc == "DB" else \
            nn.ModuleList([torch.jit.script(WWWDiffusion(**kwargs)) for _ in range(timesteps)])


    def forward(self, u, f, c,Da,Db,Dc,alpha):

        u_prev = u
        for a, block in zip(self.alphas, self.blocks):
            diffused = a * block(u, Da, Db, Dc,alpha) + (1 - a) * u_prev
            u_new = (1. - c) * diffused + c * f
            u_prev = u
            u = u_new

        return u

