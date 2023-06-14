import random
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms
from imagelib.core import tensor_to_image
from imagelib.inout import read, write_flow
from imageio.v3 import imwrite
import numpy as np
import os, sys, argparse
import torch
from imagen_pytorch import Unet, SRUnet256, Imagen, ElucidatedImagen
sys.path.insert(0, '~/projects/ModelEvaluation')

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
parser.add_argument('--test_interval', type=int, default=15_000,
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
parser.add_argument('--mask', type=float, default=0.95, help="Mask Density for Sintel")

# Diffusion arguments
parser.add_argument('--tau', type=float, default=0.41, help='timestep size')
parser.add_argument('--diffusion_position', type=str, default='encoderdecoder',
                    help='Which diffusion type. Supports: encoder, decoder, none')
parser.add_argument('--alpha', type=float, default=0.41, help="Free parameter for the WWW stencil")
parser.add_argument('--grads', nargs='+', type=bool, default=[False, False, False, False, False, True],
                    help="Which parameters to learn in dict form")
parser.add_argument('--lam', type=float, default=1., help="Diffusivity parameter")
parser.add_argument('--steps', nargs='+', type=int, default=[5, 15, 30, 45], help="How many steps per resolution")
parser.add_argument('--step', type=int, default=5, help="How many steps per resolution")
parser.add_argument('--disc', type=str, default="DB", help="Discretization")
parser.add_argument('--learned_mode', type=int, default=5, help="How many parameters to learn")

parser.add_argument('--use_dt', type=bool, default=False, help="Whether or not we use DT in Res_InpaintingNet")
parser.add_argument('--split_mode', type=str, default='diff', help="Type of diffusion. Supports: diff, id, resnet")


def make_dict(grad_list):
    return {'lam': grad_list[0], 'tau': grad_list[0], 'conv': grad_list[0], 'alphas': grad_list[0],
            'gamma': grad_list[0]}


args = parser.parse_args()
args.grads = make_dict(args.grads)
plt.style.use('classic')


def load_net(name):
    device = 'cuda'
    net = None
    path = ''
    if name == 'PD_Inpainting':
        unet1 = Unet(
            dim=128,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=3,
            layer_attns=(False, False, True, True),
            layer_cross_attns=(False, False, False, False),
            channels=2,
            channels_out=2,
            cond_images_channels=6
        )

        unet2 = SRUnet256(
            dim=128,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=(2, 4, 8, 8),
            layer_attns=(False, False, True, True),
            layer_cross_attns=(False, False, False, False),
            cond_images_channels=6
        )

        # imagen, which contains the unets above (base unet and super resoluting ones)

        net = ElucidatedImagen(
            unets=(unet1, unet2),
            image_sizes=(64, 384),
            cond_drop_prob=0.1,
            num_sample_steps=(64, 32),
            # number of sample steps - 64 for base unet, 32 for upsampler (just an example, have no clue what the optimal values are)
            sigma_min=0.002,  # min noise level
            sigma_max=(80, 160),  # max noise level, @crowsonkb recommends double the max noise level for upsampler
            sigma_data=0.5,  # standard deviation of data distribution
            rho=7,  # controls the sampling schedule
            P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
            P_std=1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
            S_churn=80,  # parameters for stochastic sampling - depends on dataset, Table 5 in apper
            S_tmin=0.05,
            S_tmax=50,
            S_noise=1.003,
            condition_on_text=False,
            channels=2,
            auto_normalize_img=False
        )
    else:
        if "InpaintingNet" in name:
            from models.InpaintingNet import InpaintingNetwork as model
            args.dim=44
        elif 'InpaintingFlowNet' in name:
            from models.InpaintingFlowNet import InpaintingFlowNet as model
            args.dim=34
        elif 'FlowNetS+' in name:
            from models.FlowNetSP import FlowNetSP as model
            args.dim=48
        elif 'WGAIN' in name:
            from models.WGAIN import WGAIN as model
        else:
            print("wtf")

        net = model(**vars(args))
    #net.load_state_dict(torch.load(f'/home/fischer/projects/checkpoints/checkpoints/{path}{name}.pt'))
    net.eval()
    net.to(device)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(f"Created Model {name} with {pytorch_total_params} total Parameters")
    return net


def inverse_normalize(tensor, mean=torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32),
                      std=torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)):
    inverse_norm = torchvision.transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    tensor = inverse_norm(tensor)
    return tensor


def collage_models_OF(nets, I1s, I2s, Flows, names=None, prefix=None):
    device = 'cuda'
    Preds = []
    for I1, I2, Flow in zip(I1s, I2s, Flows):
        current_preds = []
        I1 = I1.to(device)
        I2 = I2.to(device)
        Flow = Flow.to(device)
        for net_name in nets:
            with torch.no_grad():
                net = load_net(net_name)
                if prefix is None:
                    pred = net(I1, I2)
                else:
                    prefix_current = prefix + f'/{net_name}'
                    pred = net(I1, I2)
                    # pred = net.eigenvalues(I1, Mask, Masked_Flow, prefix_current)
                current_preds.append(pred.cpu())

        Preds.append(current_preds)
    I1s = [inverse_normalize(I1) for I1 in I1s]
    I2s = [inverse_normalize(I2) for I2 in I2s]

    fig, ax = plt.subplots(nrows=len(I1s), ncols=3 + len(nets))

    joint_list = flatten_list([[I1s[i], I2s[i], Flows[i]] + Preds[i] for i in range(len(I1s))])
    labels = ["Image 1", "Image 2", "Flow"]
    labels += [f"{net}" for net in nets] if names is None else names

    for i, axi in enumerate(ax.flat):
        image = tensor_to_image(joint_list[i])
        if image.shape[-1] == 1:
            axi.imshow(image, cmap='gray')
        else:
            axi.imshow(image)
        axi.axis('off')
    for ax, col in zip(ax[0], labels):
        ax.set_title(col, fontsize=6)
    plt.tight_layout()
    plt.axis('off')
    plt.show()


def collage_models(nets, I1s, I2s, Masks, Flows, names=None, prefix=None):
    device = 'cuda'
    Preds = []
    Masked_Flows = []
    for I1, I2, Mask, Flow in zip(I1s, I2s, Masks, Flows):
        current_preds = []
        I1 = I1.to(device)
        I2 = I2.to(device)
        Mask = Mask.to(device)
        Flow = Flow.to(device)
        Masked_Flow = torch.zeros_like(Flow)
        indices = torch.cat((Mask, Mask), 1) == 1.
        Masked_Flow[indices] = Flow[indices]
        Masked_Flows.append(Masked_Flow.cpu())
        Masked_Flow = Masked_Flow.to(device)
        for net_name in nets:
            with torch.no_grad():
                net = load_net(net_name)
                if prefix is None:
                    pred = net(I1, Mask, Masked_Flow)
                else:
                    prefix_current = prefix + f'/{net_name}'
                    pred = net(I1, Mask, Masked_Flow)
                    # pred = net.eigenvalues(I1, Mask, Masked_Flow, prefix_current)
                current_preds.append(pred.cpu())

        Preds.append(current_preds)

    fig, ax = plt.subplots(nrows=len(I1s), ncols=5 + len(nets))

    joint_list = flatten_list(
        [[I1s[i], I2s[i], Masks[i], Masked_Flows[i], Flows[i]] + Preds[i] for i in range(len(I1s))])
    labels = ["Image 1", "Image 2", "Mask", "Init", "Flow"]
    labels += [f"{net}" for net in nets] if names is None else names

    for i, axi in enumerate(ax.flat):
        image = tensor_to_image(joint_list[i])
        if image.shape[-1] == 1:
            axi.imshow(image, cmap='gray')
        else:
            axi.imshow(image)
        axi.axis('off')
    for ax, col in zip(ax[0], labels):
        ax.set_title(col, fontsize=6)
    plt.tight_layout()
    plt.axis('off')
    plt.show()


def collage(I1, I2, Mask, Masked_Flow, Flow, Pred):
    assert len(I1) == len(I2) == len(Mask) == len(Masked_Flow) == len(Flow) == len(Pred)

    fig, ax = plt.subplots(nrows=len(I1), ncols=6)

    joint_list = flatten_list([[I1[i], I2[i], Mask[i], Masked_Flow[i], Flow[i], Pred[i]] for i in range(len(I1))])
    labels = ["Image 1", "Image 2", "Mask", "Masked Flow", "Flow", "Prediction"]

    for i, axi in enumerate(ax.flat):
        image = tensor_to_image(joint_list[i])
        if image.shape[-1] == 1:
            axi.imshow(image, cmap='gray')
        else:
            axi.imshow(image)
        axi.axis('off')
    for ax, col in zip(ax[0], labels):
        ax.set_title(col)
    plt.tight_layout()
    plt.axis('off')
    plt.show()


def saveImages(I1, I2, Mask, Masked_Flow, Flow, Pred, Path=''):
    assert len(I1) == len(I2) == len(Mask) == len(Masked_Flow) == len(Flow) == len(Pred)
    i = 0
    for i1, i2, mask, mflow, flow, pred in zip(I1, I2, Mask, Masked_Flow, Flow, Pred):
        Prefix = f'{i}'
        imwrite(os.path.join(Path, Prefix + 'img1.png'), tensor_to_image(inverse_normalize(i1)))
        imwrite(os.path.join(Path, Prefix + 'img2.png'), tensor_to_image(inverse_normalize(i2)))
        imwrite(os.path.join(Path, Prefix + 'mask.png'), tensor_to_image(mask))
        imwrite(os.path.join(Path, Prefix + 'gt.png'), tensor_to_image(flow))
        # imwrite(os.path.join(Path,Prefix+'mflow.png'),tensor_to_image(mflow))
        imwrite(os.path.join(Path, Prefix + 'pred.png'), tensor_to_image(pred))
        i += 1


def show_eigenvalues(net, I1):
    pass


def flatten_list(l):
    return [item for sublist in l for item in sublist]

