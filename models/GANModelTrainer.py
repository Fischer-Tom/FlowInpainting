from os.path import join
import torch
import torch.optim
import flow_vis
from imagelib.core import inverse_normalize
from utils.loss_functions import EPE_Loss
class GANModelTrainer:


    def __init__(self, G,C, optimizer, gpu, train_iter, **kwargs):

        self.G = G
        self.C = C
        self.train_iters = 0
        self.total_iters = train_iter
        self.gpu = gpu
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=5e-5)
        self.optimizer_C = torch.optim.Adam(self.C.parameters(), lr=5e-5)


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
        for i, sample in enumerate(loader):


            sample = [samp.cuda(self.gpu) for samp in sample]

            I1, I2 = sample[0:2]
            Mask = sample[2]
            Flow = sample[-1]
            Masked_Flow = torch.zeros_like(Flow).cuda(self.gpu)
            indices = torch.cat((Mask, Mask),1) == 1.
            Masked_Flow[indices] = Flow[indices]
            # Time Iteration duration

            start.record()

            self.set_requires_grad(self.C, True)
            # Query Model
            fake = self.G(I1, Mask, Masked_Flow)
            real = Flow
            fake_guess = self.C(fake,Mask)
            real_guess = self.C(real,Mask)
            loss_C = -torch.mean(real_guess) + torch.mean(fake_guess)

            # Update Weights and learning rate
            self.optimizer_C.zero_grad()
            loss_C.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.C.parameters(), 1.)
            self.optimizer_C.step()

            self.set_requires_grad(self.C, False)
            self.optimizer_G.zero_grad()
            fake_guess = self.C(fake,Mask)
            epe = EPE_Loss(fake,real)
            loss_gen = -0.005*torch.mean(fake_guess) + epe
            loss_gen.backward()
            self.optimizer_G.step()



            end.record()
            torch.cuda.synchronize()
            # Update running loss
            running_loss += epe.item()
            iterations += 1
            self.train_iters += 1
            if self.train_iters > self.total_iters:
                break
        Flow_vis = flow_vis.flow_to_color(Flow[0].detach().cpu().permute(1,2,0).numpy())
        Pred_vis = flow_vis.flow_to_color(predict_flow[0][0].detach().cpu().permute(1, 2, 0).numpy())
        I1_vis = inverse_normalize(I1[0].detach().cpu())
        Masked_vis = flow_vis.flow_to_color(Masked_Flow[0].detach().cpu().permute(1, 2, 0).numpy())
        Mask_vis = torch.cat((Mask[0],Mask[0],Mask[0]),dim=0).detach().cpu()
        images = torch.stack((I1_vis,Mask_vis,torch.tensor(Flow_vis).permute(2,0,1),torch.tensor(Masked_vis).permute(2,0,1),torch.tensor(Pred_vis).permute(2,0,1)))
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
            for i, sample in enumerate(loader):
                sample = [samp.cuda(self.gpu) for samp in sample]

                I1, I2 = sample[0:2]
                Mask = sample[2]
                Flow = sample[-1]
                Masked_Flow = torch.zeros_like(Flow).cuda(self.gpu)
                indices = torch.cat((Mask, Mask), 1) == 1.
                Masked_Flow[indices] = Flow[indices]
                # Query Model

                start.record()

                predict_flow = self.G(I1, Mask, Masked_Flow)
                batch_risk = EPE_Loss(predict_flow,Flow)
                end.record()
                torch.cuda.synchronize()
                # Update running loss
                running_loss += batch_risk.item()
                iterations += 1
        Flow_vis = flow_vis.flow_to_color(Flow[0].detach().cpu().permute(1,2,0).numpy())
        Pred_vis = flow_vis.flow_to_color(predict_flow[0].detach().cpu().permute(1, 2, 0).numpy())
        Masked_vis = flow_vis.flow_to_color(Masked_Flow[0].detach().cpu().permute(1, 2, 0).numpy())
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