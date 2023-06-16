# !git clone https://github.com/DmitryUlyanov/deep-image-prior
# !mv deep-image-prior/* ./

from __future__ import print_function
import matplotlib.pyplot as plt
# %matplotlib inline

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # run on GPU
import numpy as np
from models import *

import torch
import torch.optim

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.measure_utils import *
from utils.common_utils import *
from utils.visualize_utils import *

# hyperparameter
img_name = 'F16_GT'

num_iter = 20000

file_name = 'outputs/F16_GT_SwinUnet2_20000_noise_skip_nopixel_noprenoise'

log_path = file_name + '/logs_%s.txt'%img_name
log2_path = file_name + '/logs2_%s.txt'%img_name

loss_list, frequency_lists, psnr_list, ratio_list = get_log_data(log_path)
frequency_lists2 = get_log2_data(log2_path)


get_loss_fig(loss_list, num_iter, ylim=0.05, save_path=file_name+'/%s_loss.png'%img_name) # save the loss figure


get_fbc_fig(frequency_lists, num_iter, ylim=1, save_path=file_name+'/%s_fbc.png'%img_name) # save the fbc figure
get_fbc_fig(frequency_lists2, num_iter, ylim=1, save_path=file_name+'/%s_fbc_pure.png'%img_name) # save the fbc figure

data_lists =[]
data_lists.append(psnr_list)
data_lists.append(ratio_list)
get_psnr_ratio_fig(data_lists, num_iter, ylim=35, ylabel='PSNR', save_path=file_name+'/%s_psnr_ratio.png'%img_name) # save the psnr_ratio figure
