# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:29:48 2020

@author: yyhhlancelot
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data_loader import *
from config import *
from model_temp import *
from model_new import *
import numpy as np
import random
import os

def read2list(config):
    
    train_paths_list = []
    train_labels_list = []
    val_paths_list = []
    val_labels_list = []
    with open(config.train_path) as f:
        for line in f.readlines():
            line = line.split(' ')
            train_paths_list.append(line[0])
            train_labels_list.append(int(line[1][0]))
    with open(config.val_path) as f:
        for line in f.readlines():
            line = line.split(' ')
            if int(line[1][0]) == 1:
                val_paths_list.append(line[0])
                val_labels_list.append(int(line[1][0]))
            
    return train_paths_list, train_labels_list, val_paths_list, val_labels_list

def shuffle_train(paths_list, labels_list):
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(paths_list)
    random.seed(randnum)
    random.shuffle(labels_list)

def label_counts(dir_path):
    return len(os.listdir(dir_path))

def train():
    
    config = Config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_paths_list, train_labels_list, val_paths_list, val_labels_list = read2list(config)
    
    shuffle_train(train_paths_list, train_labels_list)
    
    dataset = SeisMatLoader(config, train_paths_list, train_labels_list)
    
    train_loader = DataLoader(dataset = dataset, 
                              batch_size = config.batch_size,
                              shuffle = False)
    val_batch_tensor, val_label_tensor = change_val_format(config, val_paths_list, val_labels_list)
    
    # print(val_batch_tensor.size(), val_label_tensor.size())
    
    # model = HorizonClassifyNet().to(device)
    

    
    len_0 = label_counts(config.train_label_0_dir)
    len_1 = label_counts(config.train_label_1_dir)
    weights = [1/len_0, 1/len_1]
    class_weights = torch.FloatTensor(weights).cuda()
    
    criterion = torch.nn.CrossEntropyLoss(weight = class_weights)

    
        
    models = [HorizonClassifyNet().to(device), Res18(BasicBlock).to(device)]
    
    for model_index, model in enumerate(models):
        optimizer = torch.optim.Adam(model.parameters(), 
                                     weight_decay = config.weight_decay,
                                     lr = config.learning_rate)
        train_loss = [np.inf]
        val_loss = [np.inf]
        for epoch in range(10):
            iter = 0
            for x, y in train_loader:
                model.train()
                optimizer.zero_grad()
                y_pred = model(x)
    
                # print(y_pred)
                # print(y)
                y = y.view(y.size()[0])
                loss = criterion(y_pred, y)
                # print(loss)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                if iter % 10 == 0:    
                    print('epoch : {} - Training loss : {:0.4f}'.format(epoch, train_loss[-1]))
                
                if iter % 50 == 0:
                    model.eval()
                    test_list = np.random.choice(range(val_batch_tensor.size(0)), config.batch_size)
                    y_pred = model(val_batch_tensor[test_list])
                    loss = criterion(y_pred, val_label_tensor[test_list])
                    val_loss.append(loss.item())
                    print('epoch : {} - Val loss : {:0.4f}'.format(epoch, val_loss[-1]))
                iter += 1
        if model_index == 0:
            path = 'J:/desktop_material/master_dissertation/code/point2/model_save/inline3_conv_inline5.pt'
        elif model_index == 1:
            path = 'J:/desktop_material/master_dissertation/code/point2/model_save/inline3_res_inline5.pt'
        
        torch.save(model.state_dict(), path)
# import torch
# # input_ = torch.randn(2, 2, requires_grad = True)

# input_ = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
# # input_1 = torch.tensor([[0.4, 0.6]])
# # print(input_)
# # # target = torch.randint(high = 2, size = (2,), dtype = torch.int64)
# target1 = torch.tensor([0, 1])
# # target2 = torch.tensor([[1, 0]]).to(torch.float)
# # target3 = torch.tensor([[1, 0]]).to(torch.float)
# # print(target1)
# # print(target2)
# # print(target3)

# loss1 = torch.nn.functional.cross_entropy(input_, target1, reduction = 'none')
# # loss2 = torch.nn.functional.binary_cross_entropy(input_, target2, reduction = 'none')
# # loss3 = torch.nn.functional.binary_cross_entropy_with_logits(input_, target3, reduction = 'none')
# # loss4 = torch.nn.functional.binary_cross_entropy_with_logits(input_1, target3, reduction = 'none')

# print(loss1)
# print(loss2)
# print(loss3)
# print(loss4)

# from torch.autograd import Variable
# from torch import nn
# net_out = Variable(torch.Tensor([[1,2,3]]))
# target = Variable( torch.LongTensor([0]))

# criterion = nn.CrossEntropyLoss()
# criterion(net_out,target)

if __name__ == '__main__':
    
    train()