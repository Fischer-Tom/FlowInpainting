from os.path import join
import torch
import torch.optim
import flow_vis
from imagelib.core import inverse_normalize
from imagen_pytorch import ImagenTrainer
import matplotlib.pyplot as plt
from utils.loss_functions import Scaled_EPE_Loss_mean
import einops
class PD_Trainer:

    def __init__(self, net, optimizer, gpu, train_iter, **kwargs):

        self.net = net
        self.trainer = ImagenTrainer(self.net)
        self.train_iters = 0
        self.total_iters = train_iter
        self.gpu = gpu



    def get_optimizer(self, type, lr, weight_decay):

        if type == "Adam":
            return torch.optim.Adam(self.net.parameters(), lr = lr, weight_decay=weight_decay)
        raise ValueError("Invalid Optimizer. Choices are: Adam")



    def load_parameters(self, path, **kwargs):
        self.net.load_state_dict(torch.load(join(path,"model.pt")))

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
        """
        I1, Mask, Flow, predict_flow = None, None, None, None
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
            self.train_iters += 1
            if self.train_iters > self.total_iters:
                break
            if not torch.is_tensor(predict_flow):
                predict_flow = predict_flow[0]
            #print(f"Loss: {running_loss/iterations}, timing: {start.elapsed_time(end)}")
        
        Flow_vis = flow_vis.flow_to_color(Flow[0].detach().cpu().permute(1,2,0).numpy())
        Pred_vis = flow_vis.flow_to_color(predict_flow[0].detach().cpu().permute(1, 2, 0).numpy())
        I1_vis = inverse_normalize(I1[0].cpu())
        Masked_vis = flow_vis.flow_to_color(Masked_Flow[0].detach().cpu().permute(1, 2, 0).numpy())
        Mask_vis = torch.cat((Mask[0],Mask[0],Mask[0]),dim=0).detach().cpu()
        images = torch.stack((I1_vis,Mask_vis,torch.tensor(Flow_vis).permute(2,0,1),torch.tensor(Masked_vis).permute(2,0,1),torch.tensor(Pred_vis).permute(2,0,1)))
        """
        #self.trainer.load('/home/fischer/PD_samples/imagen_FullCond.pt')
        for j in range(250):
            for i, sample in enumerate(loader):
                sample = [samp.cuda() for samp in sample]
                I1, I2 = sample[0:2]
                Mask = sample[2]
                Flow = sample[3]
                Masked_Flow = sample[-1]
                Condition = I1#torch.cat((I1, Masked_Flow, Mask), dim=1)
                m = torch.nn.functional.interpolate(einops.rearrange(Mask.bool(), 'b ... -> b 1 ...').float()[:,0,::], mode='nearest-exact',size=64)
                MaskSave = torch.cat((m,m,m), dim=1)
                plt.imsave(f'./Mask-{i // 2000}.png',MaskSave[0].cpu().permute(1, 2, 0).numpy())
                self.trainer.load('./imagen.pt')
                
                images = self.trainer.sample(batch_size=1,stop_at_unet_number=1,cond_images=Condition[0:1,::], inpaint_images=Flow[0:1,::], inpaint_masks=Mask[0:1,0,::].bool(), cond_scale=5.)
                images = images * 100.0# 1353.2810
                plt.imsave(f'./sample-{i // 2000}.png',
                           flow_vis.flow_to_color(images[0].cpu().permute(1, 2, 0).numpy()))
                plt.imsave(f'./flow-{i // 2000}.png', flow_vis.flow_to_color(Flow[0].cpu().permute(1, 2, 0).numpy()))
                plt.imsave(f'./image-{i // 2000}.png',inverse_normalize(I1[0].cpu()).permute(1, 2, 0).numpy())



                #images = self.trainer.sample(batch_size=1,stop_at_unet_number=2,cond_images=Condition[0:1,::], inpaint_images=Flow[0:1,::], inpaint_masks=Mask[0:1,0,::].bool())
                #images = images * 100.0# 1353.2810
                #plt.imsave(f'./sample_full-{i // 2000}.png',
                           #flow_vis.flow_to_color(images[0].cpu().permute(1, 2, 0).numpy()))
                images = self.trainer.sample(batch_size=1, stop_at_unet_number=1, cond_images=Condition[0:1, ::],
                                             inpaint_images=Flow[0:1, ::], inpaint_masks=Mask[0:1, 0, ::].bool(),
                                            )
                images = images * 100.0  # 1353.2810
                plt.imsave(f'./sample2-{i // 2000}.png',
                           flow_vis.flow_to_color(images[0].cpu().permute(1, 2, 0).numpy()))
                images = self.trainer.sample(batch_size=1, stop_at_unet_number=1, cond_images=Condition[0:1, ::],
                                             inpaint_images=Flow[0:1, ::], inpaint_masks=Mask[0:1, 0, ::].bool(),
                                             cond_scale=25.)
                images = images * 100.0  # 1353.2810
                plt.imsave(f'./sample-HighCond-{i // 2000}.png',
                           flow_vis.flow_to_color(images[0].cpu().permute(1, 2, 0).numpy()))
                exit()

                loss = self.trainer(Flow, unet_number=2,cond_images=Condition, max_batch_size=4)
                self.trainer.update(unet_number=2)
                """
                loss = self.net(Flow,cond_images=Condition, unet_number=2)
                print(f'loss: {loss}')
                loss.backward()
                #self.trainer.update(unet_number=1)
                """
                if not (i % 2000):  # is_main makes sure this can run in distributed
                    images = self.trainer.sample(batch_size=1,stop_at_unet_number=2,cond_images=Condition[0:1,::], inpaint_images=Flow[0:1,::], inpaint_masks=Mask[0:1,0,::].bool(),cond_scale=5.)  # returns List[Image]
                    #print(Scaled_EPE_Loss_mean(images*1353.2810,Flow*1353.2810).item())
                    images = images * 100.0#1353.2810
                    iterations += 1
                    #images[0].save(f'./sample-{i // 500}.png')
                    print(f'Iter: {j}, loss: {loss}')
                    plt.imsave(f'./sample-{i // 2000}.png',flow_vis.flow_to_color(images[0].cpu().permute(1, 2, 0).numpy()))
                    self.trainer.save('./imagen.pt')
        return running_loss / iterations, start.elapsed_time(end)

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
        return running_loss / iterations , start.elapsed_time(end), images