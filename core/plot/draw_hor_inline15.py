# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:22:59 2020

@author: yyhhlancelot
"""
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
from matplotlib.pyplot import MultipleLocator

test_mat = loadmat('J:/desktop_material/master_dissertation/code/point2/data_prepare/val_data/slice_inline_5.mat')['slice_inline_5']
prob_mat_inline3_conv3_inline5 = np.load('J:/desktop_material/master_dissertation/code/point2/results/inline3_conv3_inline5.npy')
prob_mat_inline3_res18_inline5 = np.load('J:/desktop_material/master_dissertation/code/point2/results/inline3_res18_inline5.npy')
val_hor = loadmat('J:/desktop_material/master_dissertation/code/point2/data_prepare/val_horizons/horizon_inline_5.mat')['horizon_inline_5']
conv_hor = loadmat('J:/desktop_material/master_dissertation/code/point2/data_prepare/val_horizons/horizon_inline_5_conventional.mat')['horizon_inline_5_conventional']
x1 = []
y1 = []
tmp_y_start = float("nan")
tmp_y_end = float("nan")
for x in range(100,500):
    for y in range(400,650):
        
        if prob_mat_inline3_conv3_inline5[y,x] > 0.9 and math.isnan(tmp_y_start):
            tmp_y_start = y
            continue
        elif math.isnan(tmp_y_end) and prob_mat_inline3_conv3_inline5[y,x] > 0.9:
            continue
        elif math.isnan(tmp_y_start) is False and prob_mat_inline3_conv3_inline5[y,x] < 0.9:
            tmp_y_end = y-1
            y1.append(int((tmp_y_start + tmp_y_end)/2))
            tmp_y_start = float("nan")
            tmp_y_end = float("nan")
            x1.append(x)
            break
        else:
            continue

x2 = []
y2 = []
tmp_y_start = float("nan")
tmp_y_end = float("nan")
for x in range(100,500):
    for y in range(400,650):
        
        if prob_mat_inline3_res18_inline5[y,x] > 0.9 and math.isnan(tmp_y_start):
            tmp_y_start = y
            continue
        elif math.isnan(tmp_y_end) and prob_mat_inline3_res18_inline5[y,x] > 0.9:
            continue
        elif math.isnan(tmp_y_start) is False and prob_mat_inline3_res18_inline5[y,x] < 0.9:
            tmp_y_end = y-1
            y2.append(int((tmp_y_start + tmp_y_end)/2))
            tmp_y_start = float("nan")
            tmp_y_end = float("nan")
            x2.append(x)
            break
        else:
            continue

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
# ax = sns.heatmap(test_mat,cmap = sns.diverging_palette(h_neg = 240, h_pos = 20, s = 0, l = 50, n=100))
# random_sampling = np.random.randint(2, size = (651, 441))
x_major_locator = MultipleLocator(100)
y_major_locator = MultipleLocator(200)

# plt.scatter(x1, y1, s=10, c = u'#ff7f0e')
# plt.plot(x1, y1,  c = u'#ff7f0e', linewidth = 4)
# plt.scatter(x2[1:], y2[1:], s=4)
# plt.plot(x2[:53], y2[:53],  c = u'#1f77b4', linewidth = 4)
# plt.plot(x2[53:], y2[53:],  c = u'#1f77b4', linewidth = 4)
# plt.plot(x2[1:], y2[1:],c = u'#1f77b4',linewidth = 4)
# plt.scatter(val_hor[:,0]-1, val_hor[:,1]-1, c = u'#2ca02c', s=2)
# plt.plot(val_hor[:,0]-1, val_hor[:,1]-1, c = u'#2ca02c', linewidth = 4)
# plt.plot(np.concatenate((val_hor[109:,:],val_hor[:109,:]), axis = 0)[:,0]-1, np.concatenate((val_hor[109:,:],val_hor[:109,:]), axis = 0)[:,1]-1, u'#2ca02c', linewidth = 4)
# plt.scatter(conv_hor[:,0]-1, conv_hor[:,1]-1, s=1)
plt.plot(conv_hor[:,0]-1, conv_hor[:,1]-1, u'#d62728', linewidth = 4)
# plt.legend(["CNN","ISGL-CNN","Manual", "Conventional"])
# plt.legend(["CNN","ISGL-CNN","Manual", "Conventional"])
# plt.legend(["ISGL-net"])
# plt.legend(["CNN"])
# plt.legend(["Manual"])
plt.legend(["Conventional"])
ax.imshow(test_mat[:, :481], cmap ='gray', aspect='auto')

ax.set_xlabel('Crossline')
ax.set_ylabel('Time(ms)')
# ax.set_xticks(np.arange(20,421,100))
ax.set_yticks(np.arange(0,651,200))
ax.set_yticklabels(['0', '400', '800', '1200'])
# ax.set_xticklabels(['1800', '1900', '2000', '2100', '2200'])

plt.savefig("J:/desktop_material/master_dissertation/code/point2/results/inline_5_hor_conventional.pdf")
plt.show()


cnn = np.array(x1)
cnn = np.column_stack((cnn, np.array(y1)))

isgl_cnn = np.array(x2)
isgl_cnn = np.column_stack((isgl_cnn, np.array(y2)))

cnn_ae = 0
for (x, y) in cnn[2:]:
    index = np.argwhere(val_hor[:,0]==x).tolist()
    if index:
        cnn_ae += np.abs(np.mean(val_hor[:,1][np.argwhere(val_hor[:,0]==x)]) - y)
    else:
        continue
    
isgl_cnn_ae = 0
for (x, y) in isgl_cnn[2:]:
    index = np.argwhere(val_hor[:,0]==x).tolist()
    if index:
        isgl_cnn_ae += np.abs(np.mean(val_hor[:,1][np.argwhere(val_hor[:,0]==x)]) - y)
    else:
        continue
    
conv_ae = 0
for (x, y) in conv_hor:
    index = np.argwhere(val_hor[:,0]==x).tolist()
    if index:
        conv_ae += np.abs(np.mean(val_hor[:,1][np.argwhere(val_hor[:,0]==x)]) - y)
    else:
        continue