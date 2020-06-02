# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:55:40 2020

@author: yyhhlancelot
"""
from scipy.io import loadmat
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def minmax(mat, min_, max_):
    return (mat - min_) / (max_ - min_)

def change_val_format(config, x_paths_list, y_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_batch_tensor = None
    for path in x_paths_list:
        img = loadmat(path)['tmp_patch']
        img_norm = minmax(img, config.train_min, config.train_max)
        img_norm_tensor = torch.from_numpy(img_norm).float().view(1, 
                                                                  config.img_channels,
                                                                  img_norm.shape[0],
                                                                  img_norm.shape[1])
        if val_batch_tensor is None:
            val_batch_tensor = img_norm_tensor
        else:
            val_batch_tensor = torch.cat((val_batch_tensor, img_norm_tensor), 0)
    label_tensor = torch.tensor(y_list)
    return val_batch_tensor.to(device), label_tensor.to(device)

class SeisMatLoader(Dataset):
    def __init__(self, config, x_paths_list, y_list):
        super(SeisMatLoader, self).__init__()
        self.config = config
        self.min = self.config.train_min
        self.max = self.config.train_max
        self.x_paths_list = x_paths_list
        self.y_list = y_list
         
    def __getitem__(self, index):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        img_path = self.x_paths_list[index]
        img_label = self.y_list[index]
        img = loadmat(img_path)['tmp_patch']
        img_norm = minmax(img, self.min, self.max)
        img_norm_tensor = torch.from_numpy(img_norm).float().view(self.config.img_channels,
                                                                  img_norm.shape[0],
                                                                  img_norm.shape[1])
        label_tensor = torch.tensor([img_label])

        return img_norm_tensor.to(device), label_tensor.to(device)
    def __len__(self):
        return len(self.x_paths_list)
        