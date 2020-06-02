# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:21:48 2020

@author: yyhhlancelot
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HorizonClassifyNet(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = 3),
            nn.BatchNorm2d(num_features = 96),
            nn.ReLU(inplace = True)
            # nn.MaxPool2d(kernel_size = 3, stride = 2)
            )
        self.feature_extractor = nn.Sequential(self.layer1,
                                               self.layer2,
                                                self.layer3)
        self.classifier = nn.Sequential(
            nn.Linear(96 * 3 * 3, 512),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
            )
    def forward(self, x):
        b = self.feature_extractor(x)
        # print(b.size())
        b = b.view(b.size(0), 96 * 3 * 3)
        # print(b.size())
        c = self.classifier(b)
        return c
        # return self.classifier(self.feature_extractor(x))
