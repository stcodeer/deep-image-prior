from __future__ import print_function
import matplotlib.pyplot as plt
# %matplotlib inline

import numpy as np
import scipy.stats as sc

from utils.measure_utils import *
from utils.common_utils import *
from utils.visualize_utils import *

# hyperparameter
img_name = 'F16_GT'

num_iter = 20000

file_names = [
    'F16_GT_SwinUnet2_20000_noise_skip_nopixel_noprenoise_noqkscale',
    'skip/F16_GT_skip_1000_noise_skip_noprenoise',
    'F16_GT_SwinUnet2_20000_noise_mhmlp_skip_nopixel_noprenoise_noqkscale_pad',
    'F16_GT_SwinUnet2_20000_noise_none_skip_nopixel_noprenoise_noqkscale',
    'F16_GT_SwinUnet2_20000_noise_mlp_skip_nopixel_noprenoise_noqkscale_position_wise',
    'F16_GT_SwinUnet2_20000_noise_mhmlp_skip_nopixel_noprenoise_noqkscale_roll',
    'F16_GT_SwinUnet2_20000_noise_cnn_skip_nopixel_noprenoise_noqkscale_1d',
    'F16_GT_SwinUnet2_20000_noise_cnn_skip_nopixel_noprenoise_noqkscale_2d',
    ]

labels=['Transformer', 'skip', 'MHMLP_pad', 'None', 'position wise MLP', 'MHMLP_roll', 'CNN1d', 'CNN2d']

# file_names = [
#     'testape+embedpadding/F16_GT_SwinUnet2_20000_noise_skip_nopixel_noprenoise_noqkscale',
#     'testape+embedpadding/F16_GT_SwinUnet2_20000_noise_skip_nopixel_noprenoise_noqkscale_ape',
#     ]

f_variance = []
f_variance2 = []

f_skewness = []
f_skewness2 = []

f_kurtosis = []
f_kurtosis2 = []

for i, file_name in enumerate(file_names):
    log_path = 'outputs/' + file_name + '/logs_%s.txt'%img_name
    log2_path = 'outputs/' + file_name + '/logs2_%s.txt'%img_name
    loss_list, frequency_lists, psnr_list, ratio_list = get_log_data(log_path)
    frequency_lists2 = get_log2_data(log2_path)
    
    frequency_lists = frequency_lists + (num_iter - len(frequency_lists)) * [[-100]]
    frequency_lists2 = frequency_lists2 + (num_iter - len(frequency_lists2)) * [[-100]]
    
    variance = np.zeros(num_iter)
    variance2 = np.zeros(num_iter)
    
    skewness = np.zeros(num_iter)
    skewness2 = np.zeros(num_iter)
    
    kurtosis = np.zeros(num_iter)
    kurtosis2 = np.zeros(num_iter)
    
    for j in range(num_iter):
        if j % 1000 == 0:
            print(i, j)
        variance[j] = np.var(frequency_lists[j])
        variance2[j] = np.var(frequency_lists2[j])
        
        skewness[j] = sc.skew(frequency_lists[j])
        skewness2[j] = sc.skew(frequency_lists2[j])
        
        kurtosis[j] = sc.kurtosis(frequency_lists[j])
        kurtosis2[j] = sc.kurtosis(frequency_lists2[j])
    
    f_variance.append(variance)
    f_variance2.append(variance2)
    
    f_skewness.append(skewness)
    f_skewness2.append(skewness2)
    
    f_kurtosis.append(kurtosis)
    f_kurtosis2.append(kurtosis2)

plot_fbc_stats(range(num_iter), f_variance, lim=0.16, title='variance', labels=labels, plotlim=True, save_path='outputs/core_noise_fbc_variance.png')
plot_fbc_stats(range(num_iter), f_variance2, lim=0.16, title='variance', labels=labels, plotlim=True, save_path='outputs/core_origin_fbc_variance.png')

plot_fbc_stats(range(num_iter), f_skewness, lim=(-0.5, 1.5), title='skewness', labels=labels, save_path='outputs/core_noise_fbc_skewness.png')
plot_fbc_stats(range(num_iter), f_skewness2, lim=(-0.5, 1.5), title='skewness', labels=labels, save_path='outputs/core_origin_fbc_skewness.png')

plot_fbc_stats(range(num_iter), f_kurtosis, lim=(-2, 0), title='kurtosis', labels=labels, save_path='outputs/core_noise_fbc_kurtosis.png')
plot_fbc_stats(range(num_iter), f_kurtosis2, lim=(-2, 0), title='kurtosis', labels=labels, save_path='outputs/core_origin_fbc_kurtosis.png')