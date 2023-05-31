from os.path import join
import os
import torch
from imagelib.inout import read
from torchvision import transforms
import numpy as np
from itypes import Dataset


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


class FlyingThingsDataset(Dataset):
    def __init__(self, img_dir, density, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.CenterCrop(384)]),
                 mode='train',
                 type='IP'
                 ):
        self.mode = mode
        self.type = type
        self.transform = transform
        self.flow_transform = transforms.CenterCrop(384)
        self.density = density
        path = '/share_chairilg/data/FlyingThings3d_full/out_json/optical_flow_files/optical_flow_train_finalpass_subset.json' if self.mode == 'train' else '/share_chairilg/data/FlyingThings3d_full/out_json/optical_flow_files/optical_flow_test_finalpass_subset.json'
        self.ds = Dataset(path).read()

    def __len__(self):
        return self.ds.__len__()

    def __getitem__(self, idx):
        item = self.ds.__getitem__(idx)
        im0 = item['left0'].data()
        im1 = item['left1'].data()
        flow = item['left_forward'].data()
        if self.transform:
            im0 = self.transform(np.array(im0[:,:,:3]))
            im1 = self.transform(np.array(im1[:,:,:3]))
            flow = self.flow_transform(torch.Tensor(np.nan_to_num(flow[:,:,:2])).permute(2, 0, 1))
        c, h, w = im1.shape

        mask = (torch.FloatTensor(1, h, w).uniform_() > self.density).float()

        indices_1 = torch.cat((mask, torch.zeros_like(mask)), dim=0).bool()
        mean_1 = flow[indices_1].mean()
        indices_2 = torch.cat((torch.zeros_like(mask),mask), dim=0).bool()
        mean_2 = flow[indices_2].mean()

        m_flow = flow
        m_flow[indices_1] = mean_1
        m_flow[indices_2] = mean_2

        return im0, im1, mask, flow, m_flow