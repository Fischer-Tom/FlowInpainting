import argparse
import os
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import init_process_group
import os
from models.ModelTrainer import ModelTrainer
from models.GANModelTrainer import GANModelTrainer
from models.PD_Trainer import PD_Trainer
from imagen_pytorch import Unet, SRUnet256, ElucidatedImagen
from torch.utils.data import Subset
from dataset.FlyingThings import FlyingThingsDataset
from dataset.Sintel import SintelDataset

from torchvision import transforms
parser = argparse.ArgumentParser(description="OFNet")

# Data Loader Details
parser.add_argument('--batch_size', type=int, default=16, help="Number of Image pairs per batch")
parser.add_argument('--augment', type=bool, default=False, help="Use Data Augmentation")
parser.add_argument('--seed', type=int, default=42, help="Seed for the Random Number Generator")
parser.add_argument('--data_path', type=str, default='dataset/Sintel',
                    help="Relative or Absolute Path to the training Data")
parser.add_argument('--dl_workers', type=int, default=4, help="Workers for the Dataloader")
parser.add_argument('--train_split', type=float, default=0.9, help="Fraction of the Dataset to use for Training")
parser.add_argument('--test_split', type=float, default=0.05, help="Fraction of the Dataset to use for Testing")
parser.add_argument('--dataset', type=str, default="FlyingThings",
                    help="Dataset to use. Supports: FlyingThings, Sintel")

# Training Details
parser.add_argument('--train_iter', type=int, default=900_000, help="Number of Epochs to train")
parser.add_argument('--test_interval', type=int, default=10_000,
                    help="After how many Epochs the model parses the Test set")
parser.add_argument('--distributed', type=bool, default=False, help="Use Multiple GPUs for training")
parser.add_argument('--model', type=str, default="FlowNetS",
                    help="Model to train. Supports: FlowNetS, LCONVFlowNetS, FullFlowNetS, LCONCFullFlowNetS")
parser.add_argument('--model_mode', type=str, default="Single",
                    help="Model to train. Supports: Single, GAN")

# Load and Save Paths
parser.add_argument('--pretrained', type=str, default="", help="Pretrained Model")
parser.add_argument('--save_path', type=str, default="Train/", help="Where to save Model State")

# Optimization
parser.add_argument('--optimizer', type=str, default="Adam", help="Which optimizer to use. Supports: adam")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate")
parser.add_argument('--scheduler', type=str, default="StepLR", help="Learning Rate Scheduler. Supports: StepLR")
parser.add_argument('--weight_decay', type=float, default=4e-4, help="Weight decay parameter")
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help="Beta Values to use in case of Adam Optimizer")

# Miscellaneous
parser.add_argument('--mode', type=str, default='train', help="Mode. Supports: train, test")
parser.add_argument('--dim', type=int, default=44, help="Model Dimension Multiplicator")
parser.add_argument('--mask', type=float, default=0.9, help="Mask Density for Sintel")

# Diffusion arguments
parser.add_argument('--tau', type=float, default=0.41, help='timestep size')
parser.add_argument('--diffusion_position', type=str, default='decoder',
                    help='Which diffusion type. Supports: encoder, decoder, none')
parser.add_argument('--alpha', type=float, default=0.41, help="Free parameter for the WWW stencil")
parser.add_argument('--grads', nargs='+', type=bool, default=[False, False, False, False, False, True],
                    help="Which parameters to learn in dict form")
parser.add_argument('--lam', type=float, default=1., help="Diffusivity parameter")
parser.add_argument('--steps', nargs='+', type=int, default=[5,15,30,45], help="How many steps per resolution")
parser.add_argument('--step', type=int, default=5, help="How many steps per resolution")
parser.add_argument('--disc', type=str, default="DB", help="Discretization")
parser.add_argument('--learned_mode', type=int, default=5, help="How many parameters to learn")
parser.add_argument('--subset_size', type=int, default=100, help="How many parameters to learn")
parser.add_argument('--presmooth', type=bool, default=False, help="Gaussian Pre-Smoothing")


parser.add_argument('--use_dt', type=bool, default=False, help="Whether or not we use DT in Res_InpaintingNet")
parser.add_argument('--split_mode', type=str, default='diff', help="Type of diffusion. Supports: diff, id, resnet")



def make_dict(grad_list):
    return {'lam': grad_list[0], 'tau': grad_list[1], 'conv': grad_list[2], 'alphas': grad_list[3],
            'gamma': grad_list[4], 'alpha':grad_list[5]}


args = parser.parse_args()
args.grads = make_dict(args.grads)
dataset = FlyingThingsDataset
sintel_dataset = SintelDataset
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 16,
          'pin_memory': True}
# Datasets and Loaders
val_dataset = dataset(args.data_path, args.mask, mode='test', type='IP', presmooth=args.presmooth)
sintel_val_dataset = sintel_dataset(args.data_path, args.mask, mode='test', type='IP', presmooth=args.presmooth)
validation_loader = torch.utils.data.DataLoader(val_dataset, **params)
sintel_validation_loader = torch.utils.data.DataLoader(sintel_val_dataset, **params)


if "C_InpaintingNet" in args.model:
    from models.Contrast_InpaintingNet import InpaintingNetwork as model
elif "InpaintingNet" in args.model:
    args.dim=44
    from models.InpaintingNet import InpaintingNetwork as model
elif "DTM" in args.model:
    args.dim = 16
    from models.DTM import InpaintingFlowNet as model
elif 'InpaintingFlowNet' in args.model:
    from models.InpaintingFlowNet import InpaintingFlowNet as model
elif 'FlowNetS+' in args.model:
    args.dim = 48
    from models.FlowNetSP import FlowNetSP as model
elif 'WGAIN' in args.model:
    from models.WGAIN import WGAIN as model
else:
    raise ImportError()


Densities = ['N1', 'N5', 'N10']
nets = ['InpaintingNet', 'DTM', 'FlowNetS+', 'WGAIN']

for net_name in nets:
    net = load_net(net_name)
    #weights.append(sum(p.numel() for p in net.parameters()))

    running_loss = 0.0
    iterations = 0
    running_timings = 0.0
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        if ds == 'Geo':
            for i, sample in enumerate(test_loader):
                sample = [samp.cuda() for samp in sample]

                I1, I2 = sample[0:2]
                Mask = sample[2]
                Flow = sample[3]# / 100.0
                Masked_Flow = sample[-1]# / 100.0
                # Query Model
                #r = (1 - Mask) * torch.randn_like(Masked_Flow)
                #Condition = torch.cat((I1, Masked_Flow, Mask), dim=1)

                start.record()
                #predict_flow = net.sample(batch_size=1, stop_at_unet_number=2, cond_images=Condition[0:1, ::],
                #       inpaint_images=Flow[0:1, ::], inpaint_masks=Mask[0:1, 0, ::].bool(), cond_scale=5.)
                #predict_flow = net.G(I1, Mask, Masked_Flow, r)
                predict_flow = net(I1, Mask, Masked_Flow)
                batch_risk = EPE_Loss(predict_flow, Flow)#net.get_loss(predict_flow, Flow)
                end.record()
                torch.cuda.synchronize()
                # Update running loss
                running_loss += batch_risk.item()
                iterations += 1
                running_timings += start.elapsed_time(end)
                if i > 1:
                    break
