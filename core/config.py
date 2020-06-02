# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:30:44 2020

@author: yyhhlancelot
"""
import numpy as np
from scipy.io import loadmat

class Config(object):
    def __init__(self):
        # self.train_label_0_dir = 'J:/desktop_material/master_dissertation/code/point2/data_prepare/train_data/inline_1_label_0'
        self.train_label_0_dir = 'J:/desktop_material/master_dissertation/code/point2/data_prepare/train_data/inline_3_label_0'
        # self.train_label_1_dir = 'J:/desktop_material/master_dissertation/code/point2/data_prepare/train_data/inline_1_label_1'
        self.train_label_1_dir = 'J:/desktop_material/master_dissertation/code/point2/data_prepare/train_data/inline_3_label_1'
        
        # self.train_path = 'J:/desktop_material/master_dissertation/code/point2/data_prepare/train_data/data_index_1.txt'
        self.train_path = 'J:/desktop_material/master_dissertation/code/point2/data_prepare/train_data/data_index_3.txt'
        # self.train_ori_mat = loadmat('J:/desktop_material/master_dissertation/code/point2/data_prepare/train_data/slice_inline_1.mat')['slice_inline_1']
        self.train_ori_mat = loadmat('J:/desktop_material/master_dissertation/code/point2/data_prepare/train_data/slice_inline_3.mat')['slice_inline_3']
        
        self.val_path = 'J:/desktop_material/master_dissertation/code/point2/data_prepare/val_data/data_index_5.txt'
        # self.val_ori_mat = loadmat('J:/desktop_material/master_dissertation/code/point2/data_prepare/val_data/slice_inline_2.mat')['slice_inline_2']
        self.val_ori_mat = loadmat('J:/desktop_material/master_dissertation/code/point2/data_prepare/val_data/slice_inline_5.mat')['slice_inline_5']
        self.class_num = 2
        self.train_max = self.train_ori_mat.max()
        self.train_min = self.train_ori_mat.min()
        
        self.batch_size = 32
        self.img_channels = 1
        self.epochs = 1
        self.learning_rate = 0.001
        self.weight_decay = 0.0001