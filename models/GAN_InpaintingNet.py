import torch.nn as nn
from models.WGAIN import Critic
from models.InpaintingNet import InpaintingNetwork
class GAN_InpaintingNetwork(nn.Module):

    def __init__(self,**kwargs):
        super().__init__()

        self.G = InpaintingNetwork(**kwargs)
        self.C = Critic(3)

    def forward(self,I,M,u):

        return self.G(I,M,u)