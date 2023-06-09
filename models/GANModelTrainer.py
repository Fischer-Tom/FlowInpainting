from os.path import join

import matplotlib.pyplot as plt
import torch
import torch.optim
import flow_vis
from imagelib.core import inverse_normalize
from utils.loss_functions import EPE_Loss
from tqdm import tqdm
class GANModelTrainer:


    def __init__(self, G,C, optimizer, gpu, train_iter, **kwargs):

        self.G = G
        self.C = C
        self.train_iters = 0
        self.total_iters = train_iter
        self.gpu = gpu
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=1e-4,betas=(0.5,0.999),weight_decay=4e-4)
        self.optimizer_C = torch.optim.Adam(self.C.parameters(), lr=1e-4,betas=(0.5,0.999),weight_decay=4e-4)
        self.lambda_GP = 10.
        self.critic_iters = 1


    def get_optimizer(self, type, lr, weight_decay):

        if type == "Adam":
            return torch.optim.Adam(self.net.parameters(), lr = lr, weight_decay=weight_decay)
        raise ValueError("Invalid Optimizer. Choices are: Adam")



    def load_parameters(self, path, **kwargs):
        self.G.load_state_dict(torch.load(join(path,"model.pt")))

    def save_parameters(self, path):
        torch.save(self.G.state_dict(), join(path, f"model{self.train_iters}.pt"))



    def train(self, loader):
        self.G.train()
        self.C.train()

        running_loss = 0.0
        iterations = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        I1, Mask, Flow, predict_flow = None, None, None, None
        with tqdm(loader, unit="batch") as tepoch:

            for i, sample in enumerate(tepoch):
                tepoch.set_description(f"Iterations {i}")
                sample = [samp.cuda(self.gpu) for samp in sample]

                I1, I2 = sample[0:2]
                Mask = sample[2]
                real= sample[3] / 100.0
                Masked_Flow = sample[-1] / 100.0

                # Time Iteration duration
                #indices = torch.cat((Mask,Mask), dim=1)
                start.record()
                r = (1 - Mask) * torch.randn_like(Masked_Flow)
                for _ in range(self.critic_iters):
                    fake = self.G(I1, Mask, Masked_Flow,r)
                    # Query Model
                    fake_guess = self.C(fake,Mask).reshape(-1)
                    real_guess = self.C(real,Mask).reshape(-1)
                    gp = self.get_gradient_penalty(real, fake, Mask)
                    loss_C = -(torch.mean(real_guess) - torch.mean(fake_guess)) + self.lambda_GP * gp

                    # Update Weights and learning rate
                    self.optimizer_C.zero_grad()
                    loss_C.backward(retain_graph=True)
                    self.optimizer_C.step()
                    #torch.nn.utils.clip_grad_norm_(self.C.parameters(),1.0)

                fake_guess = self.C(fake,Mask).reshape(-1)
                mae = EPE_Loss(100*fake,100*real)#torch.mean(torch.abs(fake-real))
                loss_gen = -torch.mean(fake_guess) + mae
                self.optimizer_G.zero_grad()
                loss_gen.backward()
                self.optimizer_G.step()


                end.record()
                torch.cuda.synchronize()
                # Update running loss
                fake = (1-Mask)*fake + Mask*Masked_Flow
                running_loss += EPE_Loss(100.0*real, 100.0*fake).item()
                iterations += 1
                self.train_iters += 1
                tepoch.set_postfix(critic_loss=-loss_C.item(), loss=running_loss / iterations)
                if self.train_iters > self.total_iters:
                    break
                """
                if not (i % 50):
                    plt.imsave(f'./sample-{i // 2000}.png',
                               flow_vis.flow_to_color(100.0*fake[0].detach().cpu().permute(1, 2, 0).numpy()))
                    plt.imsave(f'./real-{i // 2000}.png',
                               flow_vis.flow_to_color(100.0*real[0].detach().cpu().permute(1, 2, 0).numpy()))
                    #print(running_loss / iterations)
                """

        Flow_vis = flow_vis.flow_to_color(100.0*real[0].detach().cpu().permute(1,2,0).numpy())
        Pred_vis = flow_vis.flow_to_color(100.0*fake[0].detach().cpu().permute(1, 2, 0).numpy())
        I1_vis = inverse_normalize(I1[0].detach().cpu())
        Masked_vis = flow_vis.flow_to_color(100.0*Masked_Flow[0].detach().cpu().permute(1, 2, 0).numpy())
        Mask_vis = torch.cat((Mask[0],Mask[0],Mask[0]),dim=0).detach().cpu()
        images = torch.stack((I1_vis,Mask_vis,torch.tensor(Flow_vis).permute(2,0,1),torch.tensor(Masked_vis).permute(2,0,1),torch.tensor(Pred_vis).permute(2,0,1)))

        self.critic_iters = 1
        return running_loss / iterations, start.elapsed_time(end), images

    def validate(self, loader):
        self.G.eval()
        self.C.eval()

        running_loss = 0.0
        iterations = 0
        I1, Mask, Flow, predict_flow = None, None, None, None
        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with tqdm(loader, unit="batch") as tepoch:
                for sample in enumerate(tepoch):

                    tepoch.set_description(f"Iterations {iterations}")
                    sample = [samp.cuda(self.gpu) for samp in sample]

                    I1, I2 = sample[0:2]
                    Mask = sample[2]
                    Flow = sample[3] / 100.0
                    Masked_Flow = sample[-1] / 100.0
                    r = (1 - Mask) * torch.randn_like(Masked_Flow)

                    # Query Model
                    start.record()

                    predict_flow = self.G(I1, Mask, Masked_Flow, r)
                    batch_risk = EPE_Loss(100.0*predict_flow,100.0*Flow)
                    end.record()
                    torch.cuda.synchronize()
                    # Update running loss
                    running_loss += batch_risk.item()
                    iterations += 1

        Flow_vis = flow_vis.flow_to_color(100.0*Flow[0].detach().cpu().permute(1,2,0).numpy())
        Pred_vis = flow_vis.flow_to_color(100.0*predict_flow[0].detach().cpu().permute(1, 2, 0).numpy())
        Masked_vis = flow_vis.flow_to_color(100.0*Masked_Flow[0].detach().cpu().permute(1, 2, 0).numpy())
        I1_vis = inverse_normalize(I1[0].detach().cpu())
        Mask_vis = torch.cat((Mask[0],Mask[0],Mask[0]),dim=0).detach().cpu()
        images = torch.stack((I1_vis,Mask_vis,torch.tensor(Flow_vis).permute(2,0,1),torch.tensor(Masked_vis).permute(2,0,1),torch.tensor(Pred_vis).permute(2,0,1)))
        return running_loss / iterations , start.elapsed_time(end), images



    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def get_gradient_penalty(self,real_guess,fake_guess, M):
        b,c,_,_ = real_guess.shape
        eps = torch.rand((b,1,1,1), device=real_guess.device).repeat(1,c,1,1)
        difference = fake_guess - real_guess
        interpolate = real_guess + (eps*difference)
        int_score = self.C(interpolate,M)
        grad = torch.autograd.grad(inputs=interpolate,
                                outputs=int_score,
                                grad_outputs=torch.ones_like(int_score),
                                create_graph=True,
                                retain_graph=True, )[0]
        grad = grad.view(grad.shape[0], -1)
        grad_norm = grad.norm(2, dim=1)
        gp = torch.mean((grad_norm - 1.) ** 2)
        return gp