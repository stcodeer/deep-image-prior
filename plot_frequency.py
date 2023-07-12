from __future__ import print_function

import numpy as np
from models import *

from utils.measure_utils import *
from utils.common_utils import *
from utils.visualize_utils import *

from pywt import dwt2, idwt2

# hyperparameter

img_name = 'F16_GT'

fname = 'data/denoising/%s.png'%img_name

out_file = 'outputs/figs_frequency_db3'

if not os.path.exists('outputs'):
        os.mkdir('outputs')
        
if not os.path.exists(out_file):
        os.mkdir(out_file)

imsize = -1
sigma = 25
sigma_ = sigma/255.

img_pil = crop_image(get_image(fname, imsize)[0], d=32)
img_np = pil_to_np(img_pil)

img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

img = img_np.transpose(1, 2, 0)
noisy_img = img_noisy_np.transpose(1, 2, 0)

plot_frequency_figure(img, save_path='%s/F16_GT_clean_log'%out_file)
plot_frequency_figure(noisy_img, save_path='%s/F16_GT_noisy_log'%out_file)

plot_frequency_figure(img, scale='linear', lim=100, save_path='%s/F16_GT_clean_linear'%out_file)
plot_frequency_figure(noisy_img, scale='linear', lim=100, save_path='%s/F16_GT_noisy_linear'%out_file)

plot_frequency_distribution(img, save_path='%s/F16_GT_clean_log'%out_file)
plot_frequency_distribution(noisy_img, save_path='%s/F16_GT_noisy_log'%out_file)

plot_frequency_distribution(img, scale='linear', lim=1000, save_path='%s/F16_GT_clean_linear'%out_file)
plot_frequency_distribution(noisy_img, scale='linear', lim=1000, save_path='%s/F16_GT_noisy_linear'%out_file)

plot_filtered_figure(img, img, save_path='%s/F16_GT_clean'%out_file)
plot_filtered_figure(noisy_img, img, save_path='%s/F16_GT_noisy'%out_file)

plot_filtered_figure(img, img, hollow=True, save_path='%s/F16_GT_clean_hollow'%out_file)
plot_filtered_figure(noisy_img, img, hollow=True, save_path='%s/F16_GT_noisy_hollow'%out_file)

cA, (cH, cV, cD) = dwt2(img_noisy_np, 'db3')

cA = (cA - np.min(cA)) / (np.max(cA) - np.min(cA))
cH = (cH - np.min(cH)) / (np.max(cH) - np.min(cH))
cV = (cV - np.min(cV)) / (np.max(cV) - np.min(cV))
cD = (cD - np.min(cD)) / (np.max(cD) - np.min(cD))

cA = cA.transpose(1, 2, 0)
cH = cH.transpose(1, 2, 0)
cV = cV.transpose(1, 2, 0)
cD = cD.transpose(1, 2, 0)

plot_frequency_figure(cA, save_path='%s/F16_GT_noisy_average_log'%out_file)
plot_frequency_figure(cH, save_path='%s/F16_GT_noisy_horizontal_log'%out_file)
plot_frequency_figure(cV, save_path='%s/F16_GT_noisy_vertical_log'%out_file)
plot_frequency_figure(cD, save_path='%s/F16_GT_noisy_diagonal_log'%out_file)

plot_frequency_figure(cA, scale='linear', lim=100, save_path='%s/F16_GT_noisy_average_linear'%out_file)
plot_frequency_figure(cH, scale='linear', lim=100, save_path='%s/F16_GT_noisy_horizontal_linear'%out_file)
plot_frequency_figure(cV, scale='linear', lim=100, save_path='%s/F16_GT_noisy_vertical_linear'%out_file)
plot_frequency_figure(cD, scale='linear', lim=100, save_path='%s/F16_GT_noisy_diagonal_linear'%out_file)

plot_frequency_distribution(cA, save_path='%s/F16_GT_noisy_average_log'%out_file)
plot_frequency_distribution(cH, save_path='%s/F16_GT_noisy_horizontal_log'%out_file)
plot_frequency_distribution(cV, save_path='%s/F16_GT_noisy_vertical_log'%out_file)
plot_frequency_distribution(cD, save_path='%s/F16_GT_noisy_diagonal_log'%out_file)

plot_frequency_distribution(cA, scale='linear', lim=1000, save_path='%s/F16_GT_noisy_average_linear'%out_file)
plot_frequency_distribution(cH, scale='linear', lim=1000, save_path='%s/F16_GT_noisy_horizontal_linear'%out_file)
plot_frequency_distribution(cV, scale='linear', lim=1000, save_path='%s/F16_GT_noisy_vertical_linear'%out_file)
plot_frequency_distribution(cD, scale='linear', lim=1000, save_path='%s/F16_GT_noisy_diagonal_linear'%out_file)

