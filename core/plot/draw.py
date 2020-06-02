# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:35:11 2020

@author: yyhhlancelot
"""
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
from scipy.io import loadmat
slice_1 = loadmat('J:/desktop_material/master_dissertation/code/point2/data_prepare/train_data/slice_inline_1.mat')['slice_inline_1']
slice_2 = loadmat('J:/desktop_material/master_dissertation/code/point2/data_prepare/val_data/slice_inline_2.mat')['slice_inline_2']
horizon_1 = loadmat('J:/desktop_material/master_dissertation/code/point2/data_prepare/train_horizons/horizon_inline_1.mat')['horizon_inline_1']
horizon_2 = loadmat('J:/desktop_material/master_dissertation/code/point2/data_prepare/val_horizons/horizon_inline_2.mat')['horizon_inline_2']

x1 = horizon_1[:, 0] - 1
y1 = horizon_1[:, 1] - 1
x2 = horizon_2[:, 0] - 1
y2 = horizon_2[:, 1] - 1

x_major_locator = MultipleLocator(100)
y_major_locator = MultipleLocator(100)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
# ax1 = sns.heatmap(slice_1,cmap = sns.diverging_palette(h_neg = 240, h_pos = 20, s = 0, l = 50, n=100))
# ax1.scatter(horizon_1[:, 0] - 1, (horizon_1[:, 1] - 1), s = (10,))
# ax1.set_xlabel('crossline')
# ax1.set_ylabel('time(ms)')

ax = sns.heatmap(slice_2,cmap = sns.diverging_palette(h_neg = 240, h_pos = 20, s = 0, l = 50, n=100))
ax.scatter(horizon_2[:, 0] - 1, (horizon_2[:, 1] - 1), s = (10,))
ax.set_xlabel('crossline')
ax.set_ylabel('time(ms)')
# set x_ticks
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.savefig('C:/Users/yyhhlancelot/Desktop/1.jpg')
plt.show()





