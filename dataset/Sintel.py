from os.path import join
import os
import torch
from torch.utils.data import Dataset
from imagelib.inout import read
from torchvision import transforms
import numpy as np

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


class SintelDataset(Dataset):
    def __init__(self, img_dir, density, presmooth, transform=transforms.Compose([
        transforms.ToTensor(),transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229, 0.224, 0.225)),
        transforms.CenterCrop(384)]),
                 mode = 'train',
                 type = 'IP'
):
        if presmooth:
            transform = transforms.Compose([
            transforms.ToTensor(),transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229, 0.224, 0.225)),
            transforms.CenterCrop(384), transforms.GaussianBlur(5,sigma=(0.25,2))])
        self.mode = mode
        self.transform = transform
        self.flow_transform = transforms.CenterCrop(384)
        self.img0_list = []
        self.img1_list = []
        self.flow_list = []
        self.get_IDs(img_dir)
        self.density = density
        self.type = type


    def __len__(self):
        return len(self.img1_list)

    def __getitem__(self, idx):
        im0 = read(self.img0_list[idx])
        im1 = read(self.img1_list[idx])

        flow = read(self.flow_list[idx])
        if self.transform:
            im0 = self.transform(np.array(im0))
            im1 = self.transform(np.array(im1))
            flow = self.flow_transform(torch.Tensor(np.array(flow)).permute(2,0,1))

        c,h,w = im1.shape

        mask = (torch.FloatTensor(1, h, w).uniform_() > self.density).float()
        m_flow = torch.zeros_like(flow)
        indices = torch.cat((mask,mask), dim=0).bool()
        m_flow[indices] = flow[indices]
        """
        indices_1 = torch.cat((mask, torch.zeros_like(mask)), dim=0).bool()
        mean_1 = flow[indices_1].mean()
        indices_2 = torch.cat((torch.zeros_like(mask), mask), dim=0).bool()
        mean_2 = flow[indices_2].mean()

        m_flow = flow
        m_flow[indices_1] = mean_1
        m_flow[indices_2] = mean_2
        """
        return im0, im1, mask, flow, m_flow

    def get_IDs(self, img_dir):
        data = ['final']
        for type in data:
            folder_map = os.listdir(os.path.join(img_dir,type))

            for folder in folder_map:
                for file in os.listdir(os.path.join(img_dir,type,folder)):
                    filename = file[:-3]
                    index = ''.join(x for x in filename if x.isdigit())
                    second_image = int(index) + 1
                    filename2 = f'frame_' + f'{second_image}'.zfill(4) + '.png'
                    if os.path.exists(os.path.join(img_dir,'flow',folder,filename)+'flo'):
                        self.img0_list.append(os.path.join(img_dir,type,folder,file))
                        self.img1_list.append(os.path.join(img_dir,type,folder,filename2))

                        self.flow_list.append(os.path.join(img_dir,'flow',folder,filename)+'flo')
