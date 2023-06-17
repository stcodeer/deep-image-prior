
from __future__ import print_function
import matplotlib.pyplot as plt
# %matplotlib inline

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # run on GPU
import numpy as np
from pywt import dwt2, idwt2
 
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.measure_utils import *
from utils.common_utils import *
from utils.visualize_utils import *

# read data
fname = 'data/denoising/F16_GT.png'
img_pil = crop_image(get_image(fname, -1)[0], d=32)
img_np = pil_to_np(img_pil)

# add noise 
sigma = 25
sigma_ = sigma/255.
img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

# 2D harr wavelet transform
# cA, (cH, cV, cD) = dwt2(img_noisy_np, 'haar')
cA, (cH, cV, cD) = dwt2(np.array([img_noisy_np[0]]), 'haar', mode='constant')

# x_before
plot_image_grid([img_noisy_np], factor=13, save_path='outputs/F16_GT_before_cpu.png')

# average
plot_image_grid([(cA-np.min(cA))/(np.max(cA)-np.min(cA))], factor=13, save_path='outputs/F16_GT_average_cpu.png')
# horizontal
plot_image_grid([(cH-np.min(cH))/(np.max(cH)-np.min(cH))], factor=13, save_path='outputs/F16_GT_horizontal_cpu.png')
# vertical
plot_image_grid([(cV-np.min(cV))/(np.max(cV)-np.min(cV))], factor=13, save_path='outputs/F16_GT_vertical_cpu.png')
# diagonal
plot_image_grid([(cD-np.min(cD))/(np.max(cD)-np.min(cD))], factor=13, save_path='outputs/F16_GT_diagonal_cpu.png')
 
# 2D harr inverse wavelet transform
rimg_noisy_np = idwt2((cA,(cH,cV,cD)), 'haar')
plot_image_grid([rimg_noisy_np], factor=13, save_path='outputs/F16_GT_after_cpu.png')