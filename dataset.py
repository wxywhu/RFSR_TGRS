#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
import torch
import torch.utils.data as  data
from torch.utils.data import DataLoader
import lmdb
import os
from tqdm import tqdm
import scipy.io as sio

def get_data_paths(dataroot):
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_paths_from_images(dataroot))
    return paths

def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_mat_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images
    
   
def is_mat_file(filename): 
    extension ='.mat'    
    return filename.endswith(extension)

def uint2single(img):
    return np.float32(img/65535.0)
    

    
def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))
    
# convert single (HxWxC) to 3-dimensional torch tensor
def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img).copy()).permute(2, 0, 1).float()

# convert single (HxWxC) to 4-dimensional torch tensor
def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img).copy()).permute(2, 0, 1).float().unsqueeze(0)
    

def tensor2single(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    return img
    
    
def get_patch(ilr, hr, patch_size, s, ix=-1, iy=-1):
    (ih, iw, c) = ilr.shape
    ip = patch_size
    if ix == -1:
        ix = random.randrange(0, iw - patch_size + 1)
    if iy == -1:
        iy = random.randrange(0, ih - patch_size + 1)
    
    ilr = ilr[iy:iy + ip, ix:ix + ip,:]
    hr = hr[iy:iy + ip, ix:ix + ip,:]
    
    return ilr, hr

# In[5]:

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_input
        label = self.next_target
        self.preload()
        return data, label
        

class HSIlmdbDataset(data.Dataset):
    def __init__(self, dataset_dir, scale, patch_size, channels, fcnn=False):
        super(HSIlmdbDataset, self).__init__()
        #self.opt = opt
        self.sf = 4
        #self.H_s = 504
        self.n_channels = channels
        self.patch_size = patch_size
        self.lp =self.patch_size //self.sf
        #self.L_s = self.H_s //self.sf
        p = os.path.join(dataset_dir,'x'+str(scale))
        self.env = lmdb.open(p)
        self.txn = self.env.begin(write=False) 
        self.count=0
        
        self.fcnn = fcnn
        
    def __len__(self):
        num = self.txn.stat()['entries'] //4
        return num
        
    def __getitem__(self, index):
        img_HR_bin = self.txn.get((str(index)+'HR').encode())
        img_HR_buf = np.frombuffer(img_HR_bin, dtype="uint16")
        img_HR =  img_HR_buf.reshape(self.patch_size, self.patch_size, self.n_channels)
        img_HR = uint2single(img_HR)
        img_LR_bin = self.txn.get((str(index)+'LR').encode())
        img_LR_buf = np.frombuffer(img_LR_bin, dtype="uint16")
        img_LR =  img_LR_buf.reshape(self.lp, self.lp, self.n_channels)
        img_LR = uint2single(img_LR)
        #VLR, HR = get_patch(img_VLR, img_HR, self.patch_size, self.sf)
        
        LR = single2tensor3(img_LR) 
        HR = single2tensor3(img_HR)
        return LR, HR
