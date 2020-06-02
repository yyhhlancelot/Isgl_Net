# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:01:05 2020

@author: yyhhlancelot
"""
from tqdm import tqdm
if __name__ == '__main__':
    ori_path = 'J:/desktop_material/master_dissertation/code/point2/data_prepare/val_data/data_index_1_ori.txt'
    wt_path = 'J:/desktop_material/master_dissertation/code/point2/data_prepare/val_data/data_index_1.txt'
    f2 = open(wt_path, 'w')
    with open(ori_path) as f:
        for line in tqdm(f.readlines()):
            line = line.split('/')
            line[6] = 'val_data'
            if line[7].split('_')[1] == '1':
                line[7] = 'inline_1_label_1'
            else:
                line[7] = 'inline_1_label_0'
            line_change = line[0]
            for i in range(1, len(line)):
                line_change += '/'+line[i]
            f2.write(line_change)
    f.close()
    f2.close()