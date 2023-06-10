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


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
        
        Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class FlyingThingsDataset(Dataset):
    def __init__(self, img_dir, density, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.CenterCrop(384)]),
                 mode='train',
                 type='IP'
                 ):
        super().__init__()
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
            flow = self.flow_transform(torch.Tensor(np.nan_to_num(flow[:,:,:2])).permute(2, 0, 1)) / 100.0#1353.2810
        c, h, w = im1.shape

        mask = (torch.FloatTensor(1, h, w).uniform_() > self.density).float()
        m_flow = torch.zeros_like(flow)
        indices = torch.cat((mask,mask), dim=0).bool()
        m_flow[indices] = flow[indices]
        """
        indices_1 = torch.cat((mask, torch.zeros_like(mask)), dim=0).bool()
        mean_1 = flow[indices_1].mean()
        indices_2 = torch.cat((torch.zeros_like(mask),mask), dim=0).bool()
        mean_2 = flow[indices_2].mean()

        m_flow = flow
        m_flow[indices_1] = mean_1
        m_flow[indices_2] = mean_2
        """
        return im0, im1, mask, flow, m_flow

class SintelDataset(Dataset):
    def __init__(self, img_dir, density, transform=transforms.Compose([
        transforms.ToTensor(),transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229, 0.224, 0.225)),
        transforms.CenterCrop(384)]),
                 mode = 'train',
                 type = 'IP'
):
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
            flow = self.flow_transform(torch.Tensor(np.array(flow)).permute(2,0,1)) / 100.0

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
