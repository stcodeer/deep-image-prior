from __future__ import print_function

import numpy as np
from models import *

from utils.measure_utils import *
from utils.common_utils import *
from utils.visualize_utils import *
from matplotlib import pyplot as plt

# hyperparameter

img_name = 'F16_GT'

fname = 'data/denoising/%s.png'%img_name

out_file = 'outputs/figs_random_distribution'


if not os.path.exists('outputs'):
        os.mkdir('outputs')
        
if not os.path.exists(out_file):
        os.mkdir(out_file)

file_name = 'outputs/F16_GT_skip_1000_noise_skip_gray_50fbc'
iter = 0

imsize = -1
sigma = 25
sigma_ = sigma/255.
size = 0.02
h, w = 512, 512

# img_pil = crop_image(get_image(fname, imsize)[0], d=32)
# img_np = pil_to_np(img_pil)
# img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

file1 = open('outputs/F16_GT_skip_1000_noise_skip_gray_50fbc/logs_gray.txt', 'r')
lines = file1.readlines()
file1.close()
img_np = np.array(eval(lines[0]))
img_noisy_np = np.array(eval(lines[1]))
print(img_np.shape)
print(img_noisy_np.shape)

img = img_np.transpose(1, 2, 0)
noisy_img = img_noisy_np.transpose(1, 2, 0)

# print('FBC_compare: ', get_circular_statastic(img, noisy_img, size=size))
# print('FBC_compare: ', get_circular_statastic(img[:,:,0], noisy_img[:,:,0], size=size))
# print('FBC_compare: ', get_circular_statastic(img[:,:,1], noisy_img[:,:,1], size=size))
# print('FBC_compare: ', get_circular_statastic(img[:,:,2], noisy_img[:,:,2], size=size))

img_partition = plot_filtered_figure(img, img, size=size, hollow=True)
noisy_img_partition = plot_filtered_figure(noisy_img, noisy_img, size=size, hollow=True)


img = np.expand_dims(rgb2gray(img), 0) # resize

img = img.astype(np.float64)

log_path = file_name + '/logs_%s.txt'%img_name
log2_path = file_name + '/logs2_%s.txt'%img_name

loss_list, frequency_lists, psnr_list, ratio_list = get_log_data(log_path)
frequency_lists = get_frequency_data(log_path, size=size)
frequency_lists2 = get_frequency_data(log2_path, size=size)

# g 混合 f 信号
# r = [1.00, 0.89, 0.62, 0.37, 0.21]
g = frequency_lists[iter]
f = frequency_lists2[iter]

point = 4

can = []

ftimage = np.zeros((1, h, w))

for i in range(len(g)):
    
    # tmp = noisy_img_partition[i] * g[i]
    
    ftimage_partition = np.fft.fftshift(np.fft.fft2(tmp))
    
    ftimage_partition = ftimage_partition * g[i]
    
    ftimage = ftimage + ftimage_partition
    
new_img = np.fft.ifft2(np.fft.ifftshift(ftimage)).real

new_img = np.expand_dims(np.squeeze(new_img), 0) # resize

psnr_gt = compare_psnr(img, new_img)

print('PSNR(skip): ', psnr_list[iter])
print('PSNR(random): ', psnr_gt)

plot_image_grid([new_img, np.expand_dims(rgb2gray(noisy_img), 0)], factor=13, save_path=out_file+'/%s_new_img.png'%img_name)

new_img = np.squeeze(new_img)

noisy_img = rgb2gray(noisy_img)

# print('----------theoretical----------')
# print('FBC_noise: ', g)
# print('FBC_clean: ', f)
# print('----------practical----------')
# print('FBC_noise: ', get_circular_statastic(new_img, noisy_img, size=size))
# print('FBC_clean: ', get_circular_statastic(new_img, np.squeeze(img), size=size))


fig, ax = plt.subplots(figsize=(7,6))
ax.set_xlim(0, 50)
ax.set_ylim(0, 1.1)

plt.xlabel("Optimization Iteration")
plt.ylabel("FBC ($\\bar{H}$)")

plt.plot(range(0, 50), g, linewidth=4, color='k', label='noise')
plt.plot(range(0, 50), f, linewidth=4, color='r', label='clean/skip')
plt.plot(range(0, 50), get_circular_statastic(new_img, np.squeeze(img), size=size), linewidth=4, color='b', label='clean/random')

plt.legend(loc=4,)
plt.grid()
plt.savefig(out_file+'/%s_compared_to_random.png'%img_name)
plt.close()