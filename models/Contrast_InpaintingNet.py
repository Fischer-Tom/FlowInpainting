import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss_functions import EPE_Loss
from utils.basic_blocks import PeronaMalikDiffusivity, WWWDiffusion, DiffusionBlock, SimpleConv, SimpleUpConv



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
        self.zero_pad = nn.ZeroPad2d(1) if self.disc == 'WWW' else nn.ZeroPad2d((1,0,1,0))
        self.pad = nn.ReplicationPad2d(1) if self.disc == 'WWW' else nn.ReplicationPad2d((1,0,1,0))
        self.pad_2 = nn.ReplicationPad2d(1)

        with torch.no_grad():
            self.constrain_weight()




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
        images = [I]
        image_features = self.DM(I)
        for i in range(0, self.output_resolution):
            u = self.get_flow(u, m)
            m = self.max_pool(m)
            I = self.av_pool(I)
            flows.append(u)
            masks.append(m)
            images.append(I)
        return reversed(flows), reversed(masks), reversed(images), image_features

    def get_flow(self, u, mask):
        divisor = self.sum_pool(mask)
        flow = self.sum_pool(u*mask)
        return flow / torch.clamp(divisor, min=1)

    def forward(self, I, m, u, pre_guess=None):
        flows, masks, images, image_features = self.get_masks(I, m, u)
        u = pre_guess
        for j, (f,c,i,i_f,block) in enumerate(zip(flows, masks, images, image_features, self.blocks)):
            if u is None: u = f
            Da,Db,Dc,alpha = self.get_DiffusionTensor(i,i_f)

            u = (1. - c) * u + c * f
            u = block(u, f, c, Da,Db,Dc,alpha)
            if self.training and j == self.output_resolution:
                break
            if j < self.output_resolution:
                u = F.interpolate(u, scale_factor=2, mode='bilinear')

        return u
    def get_DiffusionTensor(self, i, i_f):

        ixx, iyy, ixy = self.get_structure_tensor(i)
        v11, v12, mu1 = self.get_eigenvectors(ixx, iyy, ixy)


        mu1 = self.g(i_f[:, 0:1, :, :])
        mu2 = self.g(i_f[:, 1:2, :, :])

        a = v11 * v11 * mu1 + v12 * v12 * mu2
        c = mu1 + mu2 - a
        b = v11 * v12 * (mu1 - mu2)
        a = self.zero_pad(a)
        b = self.zero_pad(b)
        c = self.zero_pad(c)

        if self.learned_mode == 5:
            alpha = self.pad(torch.sigmoid(i_f[:,2:3,:,:])/2)
        else:
            alpha = self.alpha
        return a,b,c, alpha

    def get_structure_tensor(self, I1):

        # TODO: Add (1-alpha) component
        _, _, h, w = I1.shape
        h = h+1
        w = w+1
        I1 = self.pad_2(I1)
        Ixo = I1[:, :, 2:, 1:h] - I1[:, :, 1:w, 1:h]
        Ixp = I1[:, :, 2:, 2:] - I1[:, :, 1:w, 2:]
        Iyo = I1[:, :, 1:w, 2:] - I1[:, :, 1:w, 1:h]
        Iyp = I1[:, :, 2:, 2:] - I1[:, :, 2:, 1:h]

        Ixx = 0.5 * (Ixo * Ixo + Ixp * Ixp)
        Iyy = 0.5 * (Iyo * Iyo + Iyp * Iyp)
        Ixy = 0.25 * (Ixo + Ixp) * (Iyo + Iyp)

        return Ixx, Iyy, Ixy

    def get_eigenvectors(self, Ixx, Iyy, Ixy):
        """
        Computes Eigenvectors of the Structure Tensor with the principal axis transformation
        :param ux2: squared x derivative
        :param uy2: squared y derivaitve
        :param uxy: x derivative x y derivative
        :return: Difussion tensor a,b,c
        """
        Ixx = torch.sum(Ixx, dim=1, keepdim=True)
        Iyy = torch.sum(Iyy, dim=1, keepdim=True)
        Ixy = torch.sum(Ixy, dim=1, keepdim=True)
        aux = torch.sqrt((Iyy - Ixx) * (Iyy - Ixx) + 4. * Ixy * Ixy)

        iso = aux == 0.0

        mu1 = 0.5 * (Ixx + Iyy + aux)
        mu2 = 0.5 * (Ixx + Iyy - aux)

        order = Ixx > Iyy

        v11 = torch.zeros_like(Ixx)
        v12 = torch.zeros_like(Ixx)

        temp1 = Ixx - Iyy + aux
        temp2 = 2. * Ixy

        v11[order] = temp1[order]
        v12[order] = temp2[order]

        temp1 = Iyy - Ixx + aux
        order = torch.logical_not(order)
        v12[order] = temp1[order]
        v11[order] = temp2[order]

        v11[iso] = 1.
        v12[iso] = 0.
        mu1[iso] = Ixx[iso]
        mu2[iso] = Ixx[iso]

        norm = torch.sqrt(v11 * v11 + v12 * v12)
        low_norm = norm < 1e-6

        v11[low_norm] = 1.
        v12[low_norm] = 0.

        high_norm = torch.logical_not(low_norm)
        v11[high_norm] = v11[high_norm] / norm[high_norm]
        v12[high_norm] = v12[high_norm] / norm[high_norm]

        return v11, v12, mu1
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

class DiffusivityModule(nn.Module):

    def __init__(self, dim, learned_mode):
        super().__init__()
        self.conv0 = SimpleConv(3, dim, 3, 1, 1)
        self.conv1 = SimpleConv(dim, dim, 3, 2, 1)
        self.conv1_1 = SimpleConv(dim, dim, 3, 1, 1)
        self.conv2 = SimpleConv(dim, dim * 2, 3, 2, 1)
        self.conv2_1 = SimpleConv(dim*2, dim * 2, 3, 1, 1)
        self.conv3 = SimpleConv(dim * 2, dim * 4, 3, 2, 1)
        self.conv3_1 = SimpleConv(dim * 4, dim * 4, 3, 1, 1)
        self.conv4 = SimpleConv(dim * 4, dim * 8, 3, 2, 1)

        self.deconv4 = SimpleUpConv(dim*8, dim * 4, 1, 2, 0, 1)
        self.deconv3 = SimpleUpConv(dim * 4 + dim * 4, dim * 4, 1, 2, 0, 1)
        self.deconv2 = SimpleUpConv(dim * 4 + dim * 2, dim * 2, 1, 2, 0, 1)
        self.deconv1 = SimpleUpConv(dim * 2 + dim*1, dim, 1, 2, 0, 1)

        self.DT0 = nn.Conv2d(in_channels=dim+dim,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.DT1 = nn.Conv2d(in_channels=dim*2+dim,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.DT2 = nn.Conv2d(in_channels=dim*4+dim*2,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.DT3 = nn.Conv2d(in_channels=dim*4+dim*4,out_channels=3,kernel_size=3,stride=1,padding=1)


        self.relu = nn.ReLU()

    def forward(self, I):
        b, _, w, h = I.shape
        x0 = self.conv0(I)
        x1 = self.conv1_1(self.conv1(x0))
        x2 = self.conv2_1(self.conv2(x1))
        x3 = self.conv3_1(self.conv3(x2))
        x4 = self.conv4(x3)

        x3 = torch.cat((self.deconv4(x4), x3), dim=1)
        x2 = torch.cat((self.deconv3(x3), x2), dim=1)
        x1 = torch.cat((self.deconv2(x2), x1 ), dim=1)
        x0 = torch.cat((self.deconv1(x1), x0 ), dim=1)

        x3 = self.DT3(x3)
        x2 = self.DT2(x2)
        x1 = self.DT1(x1)
        x0 = self.DT0(x0)




        return [x3,x2,x1,x0]