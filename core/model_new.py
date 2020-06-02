# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:03:58 2020

@author: yyhhlancelot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels_, out_channels_, stride_ = 1):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels = in_channels_, out_channels = out_channels_, padding = 1, stride = stride_, kernel_size = 3),
            nn.BatchNorm2d(num_features = out_channels_),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = out_channels_, out_channels = out_channels_, padding = 1, stride = 1, kernel_size = 3),
            nn.BatchNorm2d(num_features = out_channels_))
        self.shortcut = nn.Sequential()
        if stride_ != 1 or in_channels_ != out_channels_:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels = in_channels_, out_channels = out_channels_, stride = stride_, kernel_size = 1),
                nn.BatchNorm2d(num_features = out_channels_))
            
    def forward(self, x):
        op = self.left(x)
        # print('add : ',op.size(), self.shortcut(x).size())
        op += self.shortcut(x)
        op = F.relu(op)
        return op
        
# class ResidualBlock(nn.Module):
#     def __init__(self, inchannel, outchannel, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel)
#         )
#         self.shortcut = nn.Sequential()
#         # if stride != 1 or inchannel != outchannel:
#         #     self.shortcut = nn.Sequential(
#         #         nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
#         #         nn.BatchNorm2d(outchannel)
#         #     )

#     def forward(self, x):
#         out = self.left(x)
#         print('add : ',out.size(), self.shortcut(x).size())
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

class IsglNet(nn.Module):
    def __init__(self, basic_block, num_classes = 2):
        super(IsglNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 64, padding = 1, kernel_size = 3),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU())
        self.layer1 = self.make_layer(basic_block, out_channels = 64, num_blocks = 2, stride = 1)
        self.layer2 = self.make_layer(basic_block, out_channels = 128, num_blocks = 2, stride = 2)
        self.layer3 = self.make_layer(basic_block, out_channels = 256, num_blocks = 2, stride = 2)
        self.layer4 = self.make_layer(basic_block, out_channels = 512, num_blocks = 2, stride = 2)
        self.fc = nn.Linear(512, num_classes)
    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        inp = self.conv1(x)

        bp1 = self.layer1(inp) # block_pair1

        bp2 = self.layer2(bp1)

        bp3 = self.layer3(bp2)

        bp4 = self.layer4(bp3)

        av1 = F.avg_pool2d(bp4, 4)

        v1 = av1.view(av1.size(0), -1)

        op = self.fc(v1)
        return op
    
