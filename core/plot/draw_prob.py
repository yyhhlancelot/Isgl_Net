# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:53:24 2020

@author: yyhhlancelot
"""
import matplotlib.pyplot as plt
import seaborn as sns
from model_temp import *
from model_new import *
import torch
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
from matplotlib.pyplot import MultipleLocator
def minmax(mat, min_, max_):
    return (mat - min_) / (max_ - min_)

def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

def softmax(tensor):
    length = tensor.size()[1]
    sum_exp = 0
    prob = []
    for index in range(length):
        item_val = tensor[0][index].item()
        sum_exp += np.exp(item_val)
    for index in range(length):
        item_val = tensor[0][index].item()
        prob.append(np.exp(item_val) / sum_exp)
    return prob

if __name__ == '__main__':
    # tmp_model = HorizonClassifyNet()
    tmp_model = IsglNet(BasicBlock)
    # path = 'J:/desktop_material/master_dissertation/code/point2/model_save/inline3_conv_inline5.pt'
    path = 'J:/desktop_material/master_dissertation/code/point2/model_save/inline3_res_inline5.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp_model.load_state_dict(torch.load(path))
    tmp_model.to(device)
    tmp_model.eval()
    
    train_mat = loadmat('J:/desktop_material/master_dissertation/code/point2/data_prepare/train_data/slice_inline_3.mat')['slice_inline_3']
    max_ = train_mat.max()
    min_ = train_mat.min()
    test_mat = loadmat('J:/desktop_material/master_dissertation/code/point2/data_prepare/val_data/slice_inline_20.mat')['slice_inline_20']
    prob_mat = np.zeros(shape = (test_mat.shape[0], test_mat.shape[1]))
    val_path = 'J:/desktop_material/master_dissertation/code/point2/data_prepare/val_data/data_index_20.txt'
    
    try:
        with open(val_path) as f:
            for line in tqdm(f.readlines()):
                line = line.split(' ')
                img_path = line[0]
                img = loadmat(img_path)['tmp_patch']
                img_norm = minmax(img, min_, max_)
                img_norm_tensor = torch.from_numpy(img_norm).float().view(1, 1,
                                                              img_norm.shape[0],
                                                              img_norm.shape[1]).to(device)
                prob_1 = softmax(tmp_model(img_norm_tensor))[1]
                index_x = int(img_path.split('/')[-1].split('.')[0].split('_')[0]) - 1
                index_y = int(img_path.split('/')[-1].split('.')[0].split('_')[1]) - 1
                prob_mat[index_y][index_x] = prob_1
            f.close()
    except KeyboardInterrupt:
        f.close()
        raise
    f.close()
    # img_path = 'J:/desktop_material/master_dissertation/code/point2/data_prepare/val_data/label_0/421_458.mat'
    # img = loadmat(img_path)['tmp_patch']
    # img_norm = minmax(img, min_, max_)
    # img_norm_tensor = torch.from_numpy(img_norm).float().view(1, 1,
    #                                                           img_norm.shape[0],
    #                                                           img_norm.shape[1])#.to(device)

    # x=np.random.normal(500, 100, size=1000)
    # y=np.random.normal(100, 50, size=1000)
    # heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # plt.imshow(heatmap, extent=extent,alpha=.5)
    
    ## Draw
    
    # def mycolorbar(cbar):
    #     cax = cbar.ax
    #     caxY = cax.yaxis
    #     ylab = caxY.get_label()
    #     # ylab.set_verticalalignment('top')
    #     ylab.set_horizontalalignment('center')
    #     ylab.set_rotation(90)
        
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
    # ax = sns.heatmap(test_mat,cmap = sns.diverging_palette(h_neg = 240, h_pos = 20, s = 0, l = 50, n=100))
    # random_sampling = np.random.randint(2, size = (651, 441))
    x_major_locator = MultipleLocator(100)
    y_major_locator = MultipleLocator(100)
    ax.imshow(test_mat, cmap ='gray', aspect='auto')

    # ax.plot(3, 100, 'k.', markersize = 5)
    prob = ax.imshow(prob_mat[:, :481], cmap = 'bwr', alpha = 0.8, aspect='auto')

    cb = fig.colorbar(prob, ax = ax)
    cb.set_label('Probability Of Horizon')
    # mycolorbar(cb)
    # ax = sns.heatmap(prob_mat * random_sampling,cmap = sns.diverging_palette(h_neg = 240, h_pos = 20, s = 80, l = 50, n=100))
    # ax.scatter(horizon_2[:, 0] - 1, (horizon_2[:, 1] - 1), s = (10,))
    ax.set_xlabel('Crossline')
    ax.set_ylabel('Time(ms)')
    ax.set_xticks(np.arange(0,481,100))
    ax.set_yticks(np.arange(0,651,200))
    ax.set_yticklabels(['0', '400', '800', '1200'])
    # ax.set_xticklabels(['1800', '1900', '2000', '2100', '2200'])
    # plt.savefig('J:/desktop_material/master_dissertation/code/point2/results/inline3_conv3_inline20.pdf')
    # np.save('J:/desktop_material/master_dissertation/code/point2/results/inline3_conv3_inline20.npy', prob_mat)
    plt.savefig('J:/desktop_material/master_dissertation/code/point2/results/inline3_res18_inline20.pdf')
    np.save('J:/desktop_material/master_dissertation/code/point2/results/inline3_res18_inline20.npy', prob_mat)
    plt.show()
