# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 21:47:29 2020

@author: yyhhlancelot
"""

from model_temp import *
import torch
from scipy.io import loadmat

def minmax(mat, min_, max_):
    return (mat - min_) / (max_ - min_)

tmp_model = HorizonClassifyNet()
path = 'J:/desktop_material/master_dissertation/code/point2/model_save/model_conv3.pt'
tmp_model.load_state_dict(torch.load(path))
tmp_model.eval()
ori_mat = loadmat('J:/desktop_material/master_dissertation/code/point2/data_prepare/train_data/slice_inline_1.mat')['slice_inline_1']
# ori_mat = loadmat('J:/desktop_material/master_dissertation/code/point2/data_prepare/val_data/slice_inline_2.mat')['slice_inline_2']



max_ = ori_mat.max()
min_ = ori_mat.min()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# img_path = 'J:/desktop_material/master_dissertation/code/point2/data_prepare/val_data/label_1/421_350.mat'
img_path = 'J:/desktop_material/master_dissertation/code/point2/data_prepare/val_data/label_0/421_458.mat'
img = loadmat(img_path)['tmp_patch']
img_norm = minmax(img, min_, max_)
img_norm_tensor = torch.from_numpy(img_norm).float().view(1, 1,
                                                          img_norm.shape[0],
                                                          img_norm.shape[1])#.to(device)
print(tmp_model(img_norm_tensor))