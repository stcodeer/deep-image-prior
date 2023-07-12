from __future__ import print_function

import numpy as np
from models import *

from utils.measure_utils import *
from utils.common_utils import *
from utils.visualize_utils import *

from pywt import dwt2, wavelist

# hyperparameter

img_name = 'F16_GT'
img_name_256 = 'F16_GT_256'

fname = 'data/denoising/%s.png'%img_name
fname_256 = 'data/denoising/%s.png'%img_name_256

imsize = -1
sigma = 25
sigma_ = sigma/255.

img_pil = crop_image(get_image(fname, imsize)[0], d=32)
img_np = pil_to_np(img_pil)
img_pil_256 = crop_image(get_image(fname_256, imsize)[0], d=32)
img_np_256 = pil_to_np(img_pil_256)

img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

img = img_np.transpose(1, 2, 0)
img_256 = img_np_256.transpose(1, 2, 0)
noisy_img = img_noisy_np.transpose(1, 2, 0)


log_path = 'plot_wavelet_distribution.log'
log_file = open(log_path, "w")

for wave in wavelist(kind='discrete'):
    
    log_file.write(wave + '\n')

    cA, (cH, cV, cD) = dwt2(img_noisy_np, wave, mode='periodization')

    cA = (cA - np.min(cA)) / (np.max(cA) - np.min(cA))
    cH = (cH - np.min(cH)) / (np.max(cH) - np.min(cH))
    cV = (cV - np.min(cV)) / (np.max(cV) - np.min(cV))
    cD = (cD - np.min(cD)) / (np.max(cD) - np.min(cD))

    cA = cA.transpose(1, 2, 0)
    cH = cH.transpose(1, 2, 0)
    cV = cV.transpose(1, 2, 0)
    cD = cD.transpose(1, 2, 0)

    avg_mask_cA = get_circular_statastic(cA, cA, size=0.2)
    avg_mask_cH = get_circular_statastic(cH, cA, size=0.2)
    avg_mask_cV = get_circular_statastic(cV, cA, size=0.2)
    avg_mask_cD = get_circular_statastic(cD, cA, size=0.2)
    
    log_file.write('fbc(cA/cA): ' + str([round(avg_mask_cA[i], 3) for i in range(5)]) + '\n')
    log_file.write('fbc(cH/cA): ' + str([round(avg_mask_cH[i], 3) for i in range(5)]) + '\n')
    log_file.write('fbc(cV/cA): ' + str([round(avg_mask_cV[i], 3) for i in range(5)]) + '\n')
    log_file.write('fbc(cD/cA): ' + str([round(avg_mask_cD[i], 3) for i in range(5)]) + '\n')

    avg_mask_cA = get_circular_statastic(cA, img_256, size=0.2)
    avg_mask_cH = get_circular_statastic(cH, img_256, size=0.2)
    avg_mask_cV = get_circular_statastic(cV, img_256, size=0.2)
    avg_mask_cD = get_circular_statastic(cD, img_256, size=0.2)

    log_file.write('fbc(cA/256GT): ' + str([round(avg_mask_cA[i], 3) for i in range(5)]) + '\n')
    log_file.write('fbc(cH/256GT): ' + str([round(avg_mask_cH[i], 3) for i in range(5)]) + '\n')
    log_file.write('fbc(cV/256GT): ' + str([round(avg_mask_cV[i], 3) for i in range(5)]) + '\n')
    log_file.write('fbc(cD/256GT): ' + str([round(avg_mask_cD[i], 3) for i in range(5)]) + '\n\n')
    
log_file.close()