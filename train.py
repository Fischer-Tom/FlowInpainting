#!/usr/bin/env python3
import argparse
import os
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import init_process_group
import os
from models.ModelTrainer import ModelTrainer
from models.GANModelTrainer import GANModelTrainer

parser = argparse.ArgumentParser(description="OFNet")

# Data Loader Details
parser.add_argument('--batch_size', type=int, default=4, help="Number of Image pairs per batch")
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
parser.add_argument('--dim', type=int, default=24, help="Model Dimension Multiplicator")
parser.add_argument('--mask', type=float, default=0.95, help="Mask Density for Sintel")

# Diffusion arguments
parser.add_argument('--tau', type=float, default=2., help='timestep size')
parser.add_argument('--diffusion', type=str, default='encoder',
                    help='Which diffusion type. Supports: encoder, decoder, none')
parser.add_argument('--alpha', type=float, default=0.41, help="Free parameter for the WWW stencil")
parser.add_argument('--grads', nargs='+', type=bool, default=[False, False, False, False, False, True],
                    help="Which parameters to learn in dict form")
parser.add_argument('--lam', type=float, default=1., help="Diffusivity parameter")
parser.add_argument('--steps', nargs='+', type=int, default=[5,15,35,45], help="How many steps per resolution")
parser.add_argument('--step', type=int, default=5, help="How many steps per resolution")
parser.add_argument('--disc', type=str, default="DB", help="Discretization")


parser.add_argument('--use_dt', type=bool, default=False, help="Whether or not we use DT in Res_InpaintingNet")
parser.add_argument('--split_mode', type=str, default='diff', help="Type of diffusion. Supports: diff, id, resnet")



def make_dict(grad_list):
    return {'lam': grad_list[0], 'tau': grad_list[1], 'conv': grad_list[2], 'alphas': grad_list[3],
            'gamma': grad_list[4], 'alpha':grad_list[5]}


args = parser.parse_args()
args.grads = make_dict(args.grads)
train_writer = SummaryWriter(args.save_path + 'logs/train/' + args.model)
validation_writer = SummaryWriter(args.save_path + 'logs/test/' + args.model)


def main_worker(gpu, ngpus, args):
    args.gpu = gpu
    # Load Model here
    net = None
    ds = 'IP'

    try:
        if "Res_InpaintingFlowNetNet" in args.model:
            from models.Res_InpaintingFlowNet import Res_InpaintingFlowNet as model
        elif "InpaintingNet" in args.model:
            from models.InpaintingNet import InpaintingNetwork as model
        elif 'InpaintingFlowNet' in args.model:
            from models.InpaintingNet import InpaintingFlowNet as model
        elif 'FlowNetS+' in args.model:
            from models.FlowNetSP import FlowNetSP as model
        elif 'WGAIN' in args.model:
            from models.WGAIN import WGAIN as model
        else:
            raise ImportError()

    except ImportError:
        print("Invalid Model choice. Supported are: FlowNetS, FlowNetC, UNet, Dummy")
        exit(1)

    net = model(**vars(args))
    if args.distributed:

        init_process_group(backend='nccl', world_size=ngpus, rank=args.gpu)
        torch.cuda.set_device(args.gpu)
        net.cuda(args.gpu)

        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=True)

        print(f"Model has been loaded on GPU {args.gpu}")

    else:
        net = net.cuda(args.gpu)

    # Datasets and Loaders

    try:
        if args.dataset == 'Sintel':
            from dataset.Sintel import SintelDataset
            from torchvision import transforms
            dataset = SintelDataset
            params = {'batch_size': 4,
                      'shuffle': True,
                      'num_workers': 4}
            # Datasets and Loaders
            val_dataset = dataset(args.data_path, args.mask, mode='test')
            train_dataset = dataset(args.data_path, args.mask, mode='train')
            train_loader = torch.utils.data.DataLoader(train_dataset, **params)
            validation_loader = torch.utils.data.DataLoader(val_dataset, **params)
        elif args.dataset == 'FlyingThings':
            from dataset.FlyingThings import FlyingThingsDataset
            from dataset.Sintel import SintelDataset

            from torchvision import transforms
            dataset = FlyingThingsDataset
            sintel_dataset = SintelDataset
            params = {'batch_size': 4,
                      'shuffle': True,
                      'num_workers': 4}
            # Datasets and Loaders
            val_dataset = dataset(args.data_path, args.mask, mode='test', type=ds)
            sintel_val_dataset = sintel_dataset(args.data_path, args.mask, mode='test', type=ds)
            train_dataset = dataset(args.data_path, args.mask, mode='train', type=ds)
            train_loader = torch.utils.data.DataLoader(train_dataset, **params)
            validation_loader = torch.utils.data.DataLoader(val_dataset, **params)
            sintel_validation_loader = torch.utils.data.DataLoader(sintel_val_dataset, **params)

        else:

            raise ImportError()
    except ImportError:
        print("Invalid Dataset Choice. Supported are: FlyingGeometry, FlyingChairs")
        exit(1)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(f"Created Model {args.model} with {pytorch_total_params} total Parameters")
    # Load ModelTrainer and Potentialy saved state
    trainer = ModelTrainer(net, **vars(args)) if args.model_mode == 'Single' else GANModelTrainer(net.G,net.C,**vars(args))

    if args.mode == 'test':
        #test_dataset = dataset(os.path.join(args.data_path, f'test'))
        #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
        #                                          shuffle=True, num_workers=args.dl_workers)
        test_risk, inf_speed, _ = trainer.validate(sintel_validation_loader)
        print(f"Test Risk is {test_risk:.5f} with inference time {inf_speed:.3f}")

        return
    # trainer.save_parameters(args.save_path + f'checkpoints/{args.model}')


    # trainer.load_parameters(args.save_path + f'checkpoints/{args.model}')
    # print(f"Model loaded from {args.save_path}checkpoints/{args.model}")

    test_epochs = 1
    while trainer.train_iters < args.train_iter:
        risk, train_speed, samples = trainer.train(train_loader)
        if args.gpu == 0:
            train_writer.add_scalar('Train Risk', risk, trainer.train_iters)
            print(f'[Training Iterations|Risk | Train time]: {trainer.train_iters} | {risk:.5f} | {train_speed: .3f}')
            train_writer.add_images('Train Set Samples', samples, test_epochs)

            if trainer.train_iters > test_epochs * args.test_interval:
                validation_risk, inf_speed, samples = trainer.validate(validation_loader)
                validation_writer.add_scalar('Test Risk', validation_risk, test_epochs)
                validation_writer.add_images('Validation Set Samples', samples, test_epochs)
                print(f'[Test Epochs| Test Risk | Inference Time]: {test_epochs} | {validation_risk:.5f} | {inf_speed:.3f}')
                if args.dataset == 'FlyingThings':
                    validation_risk, inf_speed, samples = trainer.validate(sintel_validation_loader)
                    validation_writer.add_scalar('Sintel Test Risk', validation_risk, test_epochs)
                    validation_writer.add_images('Sintel Validation Set Samples', samples, test_epochs)
                    print(f'[Test Epochs| Test Risk | Inference Time]: {test_epochs} | {validation_risk:.5f} | {inf_speed:.3f}')
                trainer.save_parameters(args.save_path + f'checkpoints/{args.model}')
                test_epochs += 1


def main():
    os.makedirs(args.save_path + f'checkpoints/{args.model}/', exist_ok=True)
    os.makedirs(args.save_path + 'logs', exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'logs', 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'logs', 'test'), exist_ok=True)
    n_gpus = torch.cuda.device_count()

    if args.distributed:
        torch.multiprocessing.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()
