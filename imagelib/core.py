import torch
import flow_vis
import numpy as np
import matplotlib.pyplot as plt

import torchvision.utils
from PIL import Image


def image_to_tensor(path: str, mode='RGB'):
    if mode == 'GS':
        im = Image.open(path).convert('L')
        im_np = np.expand_dims(np.asarray(im), axis=2)
    else:
        im = Image.open(path)
        im_np = np.asarray(im)
    return torch.from_numpy(im_np).permute(2, 0, 1)

def image_to_numpy(path: str, mode='RGB'):
    if mode == 'GS':
        im = Image.open(path).convert('L')
        im_np = np.expand_dims(np.asarray(im), axis=2)
    else:
        im = Image.open(path)
        im_np = np.asarray(im)
    return im_np


def tensor_to_image(tensor):
    pass


def display_numpy_image(array):
    if len(array.shape) == 2:
        plt.imshow(array, cmap='gray')
        plt.axis('off')
        plt.show()
        return
    plt.imshow(array)
    plt.axis('off')
    plt.show()

def display_flow_tensor(tensor, name=""):
    if len(tensor.shape) == 4:
        display_batch_flow_tensor(tensor)
    else:
        np_flow = tensor.permute((1, 2, 0)).numpy()
        flow_color = flow_vis.flow_to_color(np_flow, convert_to_bgr=False)
        plt.imshow(flow_color)
        plt.axis('off')
        plt.imsave(name,flow_color)
        plt.show()

def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
def display_flow_numpy(np_array):
    flow_color = flow_vis.flow_to_color(np_array, convert_to_bgr=False)
    plt.imshow(flow_color)
    plt.axis('off')
    plt.show()

def display_batch_flow_tensor(tensor):
    raise Exception("not implemented!")


def display_tensor(tensor, name=""):
    plt.axis('off')
    if tensor.shape[0] == 3:
        plt.imshow(tensor.permute(1, 2, 0))
        plt.imsave(name, tensor.permute(1, 2, 0).numpy())

    else:
        plt.imshow(tensor.permute(1, 2, 0), cmap='gray')
        plt.imsave(name, tensor.numpy().squeeze(), cmap='gray')

    plt.show()

def draw_kernels(kernelTensor):
    kss,h,w = kernelTensor.shape
    print(kernelTensor.shape)
    ks = int(np.sqrt(kss))
    grid = torchvision.utils.make_grid(kernelTensor.reshape(ks,ks,-1,h*w).permute(3,2,0,1),nrow=h)
    plt.figure(figsize=(h, w))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.ioff()
    plt.show()






