from os.path import join

import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.optim
import flow_vis
from imagelib.core import inverse_normalize
from tqdm import tqdm
class ModelTrainer:

    def __init__(self, net, optimizer, gpu, train_iter, **kwargs):

        self.net = net
        self.train_iters = 0
        self.total_iters = train_iter
        self.gpu = gpu
        self.optimizer = self.get_optimizer(optimizer, kwargs['lr'], kwargs['weight_decay'])
        self.scheduler = self.net.get_scheduler(self.optimizer)


    def get_optimizer(self, type, lr, weight_decay):

        if type == "Adam":
            return torch.optim.Adam(self.net.parameters(), lr = lr, weight_decay=weight_decay)
        raise ValueError("Invalid Optimizer. Choices are: Adam")



    def load_parameters(self, path, **kwargs):
        self.net.load_state_dict(torch.load(path))

    def save_parameters(self, path):
        torch.save(self.net.state_dict(), join(path, f"model{self.train_iters}.pt"))
        torch.save(self.optimizer.state_dict(), join(path, f"optimizer{self.train_iters}.pt"))
        torch.save(self.scheduler.state_dict(), join(path, f"scheduler{self.train_iters}.pt"))



    def train(self, loader):
        self.net.train()

        running_loss = 0.0
        iterations = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        I1, Mask, Flow, predict_flow = None, None, None, None
        #with tqdm(loader, unit="batch") as tepoch:
        for i, sample in enumerate(loader):
            sample = [samp.cuda() for samp in sample]

            I1, I2 = sample[0:2]
            Mask = sample[2]
            Flow = sample[3]
            Masked_Flow = sample[-1]

            # Time Iteration duration
            start.record()
            self.optimizer.zero_grad(set_to_none=True)
            # Query Model
            predict_flow = self.net(I1, Mask, Masked_Flow)
            batch_risk = self.net.get_loss(predict_flow, Flow)

            # Update Weights and learning rate
            batch_risk.backward()
            self.optimizer.step()
            self.net.update_lr(self.scheduler, self.train_iters)
            with torch.no_grad():
                self.net.constrain_weight()
            end.record()
            #torch.cuda.synchronize()
            # Update running loss
            running_loss += batch_risk.item()
            iterations += 1
            print(running_loss/iterations)
            self.train_iters += 1
            if self.train_iters > self.total_iters:
                break
            if not torch.is_tensor(predict_flow):
                predict_flow = predict_flow[0]
        #    tepoch.set_postfix(loss=running_loss / iterations)
        #print(f"Loss: {running_loss/iterations}, timing: {start.elapsed_time(end)}")

        Flow_vis = flow_vis.flow_to_color(Flow[0].detach().cpu().permute(1,2,0).numpy())
        Pred_vis = flow_vis.flow_to_color(predict_flow[0].detach().cpu().permute(1, 2, 0).numpy())
        I1_vis = inverse_normalize(I1[0].cpu())
        Masked_vis = flow_vis.flow_to_color(Masked_Flow[0].detach().cpu().permute(1, 2, 0).numpy())
        Mask_vis = torch.cat((Mask[0],Mask[0],Mask[0]),dim=0).detach().cpu()
        images = torch.stack((I1_vis,Mask_vis,torch.tensor(Flow_vis).permute(2,0,1),torch.tensor(Masked_vis).permute(2,0,1),torch.tensor(Pred_vis).permute(2,0,1)))
        return running_loss / iterations, start.elapsed_time(end), images

    def validate(self, loader):
        self.net.eval()
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
                Flow = sample[3]
                Masked_Flow = sample[-1]
                # Query Model

                start.record()

                predict_flow = self.net(I1, Mask, Masked_Flow)
                batch_risk = self.net.get_loss(predict_flow, Flow)
                end.record()
                torch.cuda.synchronize()
                # Update running loss
                running_loss += batch_risk.item()
                iterations += 1
                Flow_vis = flow_vis.flow_to_color(Flow[0].detach().cpu().permute(1,2,0).numpy())
                Pred_vis = flow_vis.flow_to_color(torch.nan_to_num_(predict_flow[0]).detach().cpu().permute(1, 2, 0).numpy())
                Masked_vis = flow_vis.flow_to_color(Masked_Flow[0].detach().cpu().permute(1, 2, 0).numpy())
                I1_vis = inverse_normalize(I1[0].detach().cpu())
                Mask_vis = torch.cat((Mask[0],Mask[0],Mask[0]),dim=0).detach().cpu()
                images = torch.stack((I1_vis,Mask_vis,torch.tensor(Flow_vis).permute(2,0,1),torch.tensor(Masked_vis).permute(2,0,1),torch.tensor(Pred_vis).permute(2,0,1)))
                I1 = images[0, ::]
                Flow = images[2, ::]
                Masked = images[3, ::]
                Pred = images[4, ::]
                plt.imsave(f"sampleImages/Image_FlowNet_{i}.png", I1.permute(1, 2, 0).numpy())
                plt.imsave(f"sampleImages/Flow_FlowNet_{i}.png", Flow.permute(1, 2, 0).numpy().astype(np.uint8))
                plt.imsave(f"sampleImages/Masked_FlowNet_{i}.png",
                           Masked.permute(1, 2, 0).numpy().astype(np.uint8))
                plt.imsave(f"sampleImages/Pred_FlowNet_{i}.png", Pred.permute(1, 2, 0).numpy().astype(np.uint8))

        return running_loss / iterations , start.elapsed_time(end), images