from __future__ import print_function
import matplotlib.pyplot as plt
# %matplotlib inline

import numpy as np

from utils.measure_utils import *
from utils.common_utils import *
from utils.visualize_utils import *

# hyperparameter
img_name = 'F16_GT'

num_iter = 20000

file_names = [
    'F16_GT_SwinUnet2_20000_noise_mlp_skip_nopixel_noprenoise_noqkscale',
    'F16_GT_SwinUnet2_20000_noise_skip_nopixel_noprenoise_noqkscale',
    ]

f1 = []
f2 = []

for file_name in file_names:
    log_path = 'outputs/' + file_name + '/logs_%s.txt'%img_name
    log2_path = 'outputs/' + file_name + '/logs2_%s.txt'%img_name
    loss_list, frequency_lists, psnr_list, ratio_list = get_log_data(log_path)
    frequency_lists2 = get_log2_data(log2_path)
    
    for i in range(num_iter):
        frequency_lists[i] = np.var(frequency_lists[i])
        frequency_lists2[i] = np.var(frequency_lists2[i])
    
    f1.append(frequency_lists)
    f2.append(frequency_lists2)


plot_fbc_variance(range(num_iter), f1, lim=0.15, labels=['SwinMLP', 'SwinTransformer'], save_path='outputs/noise_fbc_variance.png')
plot_fbc_variance(range(num_iter), f2, lim=0.15, labels=['SwinMLP', 'SwinTransformer'], save_path='outputs/origin_fbc_variance.png')