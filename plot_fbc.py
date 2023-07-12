from __future__ import print_function
import matplotlib.pyplot as plt
# %matplotlib inline

from utils.measure_utils import *
from utils.common_utils import *
from utils.visualize_utils import *

# hyperparameter
img_name = 'F16_GT'

num_iter = 7500
best_iter = 9149

file_name = 'outputs/ttestinformer/F16_GT_iformer_small_10000_noise_skip'

log_path = file_name + '/logs_%s.txt'%img_name
log2_path = file_name + '/logs2_%s.txt'%img_name

# loss_list, frequency_lists, psnr_list, ratio_list = get_log_data(log_path)
frequency_lists = get_frequency_data(log_path, size=0.2)
frequency_lists2 = get_frequency_data(log2_path, size=0.2)


# get_loss_fig(loss_list, num_iter, ylim=0.05, save_path=file_name+'/%s_loss.png'%img_name) # save the loss figure
        
# data_lists =[]
# data_lists.append(psnr_list)
# data_lists.append(ratio_list)
# get_psnr_ratio_fig(data_lists, num_iter, ylim=35, ylabel='PSNR', save_path=file_name+'/%s_psnr_ratio_10000.png'%img_name) # save the psnr_ratio figure


get_fbc_fig(frequency_lists, num_iter, best_iter=best_iter, ylim=1, save_path=file_name+'/%s_fbc_7500.png'%img_name) # save the fbc figure
# get_fbc_fig(frequency_lists2, num_iter, best_iter=best_iter, ylim=1, save_path=file_name+'/%s_fbc_pure.png'%img_name) # save the fbc figure

cnt = len(frequency_lists[0])

for i in range(num_iter):
    for j in range(cnt):
        frequency_lists[i][j] = frequency_lists[i][j] / frequency_lists2[i][j]
        
# for i in range(0, cnt, 5):
#     list = []
get_fbc_fig(frequency_lists, num_iter, ylim=1.1, save_path=file_name+'/%s_fbc_ratio_7500_%s.png'%(img_name, str(0)+'_'+str(5)))
