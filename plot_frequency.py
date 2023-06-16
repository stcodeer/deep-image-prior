"""
*Uncomment if running on colab* 
Set Runtime -> Change runtime type -> Under Hardware Accelerator select GPU in Google Colab 
"""
# !git clone https://github.com/DmitryUlyanov/deep-image-prior
# !mv deep-image-prior/* ./

from __future__ import print_function
import matplotlib.pyplot as plt
# %matplotlib inline

import os
import numpy as np
from models import *

import torch
import torch.optim

from utils.measure_utils import *
from utils.common_utils import *
from utils.visualize_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # run on CPU
# dtype = torch.FloatTensor # run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # run on GPU
dtype = torch.cuda.FloatTensor # run on GPU

# hyperparameter

img_name = 'F16_GT'

fname = 'data/denoising/%s.png'%img_name

imsize = -1
sigma = 25
sigma_ = sigma/255.


# load data

# Add synthetic noise
img_pil = crop_image(get_image(fname, imsize)[0], d=32)
# img_pil = img_pil.crop((0, 0, 256, 256))
img_np = pil_to_np(img_pil)

img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

img = img_np.transpose(1, 2, 0)
noisy_img = img_noisy_np.transpose(1, 2, 0)

plot_frequency_figure(img, save_path='outputs/figs/F16_GT_clean')
plot_frequency_figure(noisy_img, save_path='outputs/figs/F16_GT_noisy')

plot_filtered_figure(img, img, save_path='outputs/figs/F16_GT_clean')
plot_filtered_figure(noisy_img, img, save_path='outputs/figs/F16_GT_noisy')
