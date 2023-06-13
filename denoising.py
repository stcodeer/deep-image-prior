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
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # run on GPU
import numpy as np
from models import *
from models.ViT import ViT
from models.swin_transformer_2 import SwinUnet2
from models.swin_transformer_2_decoder import Swin2Decoder
from models.inception_transformer import iformer_small, iformer_base, iformer_large

import torch
import torch.optim

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.denoising_utils import *
from utils.measure_utils import *
from utils.common_utils import *
from utils.visualize_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
# dtype = torch.FloatTensor # run on CPU
dtype = torch.cuda.FloatTensor # run on GPU

# hyperparameter
img_name = 'F16_GT'
fname = 'data/denoising/%s.png'%img_name

NET_TYPE = 'SwinUnet2' # one of skip|ViT|SwinUnet2|Swin2Decoder|iformer_small|iformer_base|iformer_large

# exp_name = '_skip_nopixel_halfprenoise'
exp_name = '_skip_nopixel_noprenoise_noqkscale_embedpadding'

num_iter = 20000

INPUT = 'noise' # 'noise', meshgrid', 'fourier'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

wavelet_method = 'None' # 'None', 'haar' (only enable when SwinUnet2)
imsize = -1
PLOT = False
sigma = 25 # 25, 50
sigma_ = sigma/255.
reg_noise_std = 0 # 1./30., 1./20.
OPTIMIZER='adam' # 'adam', 'LBFGS', 'adam_gradual_warmup'
show_every = 500
exp_weight = 0.99

# output path

if not os.path.exists('outputs'):
        os.mkdir('outputs')

file_name = 'outputs/' + img_name + '_' + NET_TYPE + '_' + str(num_iter) + '_' + INPUT + exp_name

print(file_name)

if not os.path.exists(file_name):
        os.mkdir(file_name)

log_path = file_name + '/logs_%s.txt'%img_name
log_file = open(log_path, "w")

log2_path = file_name + '/logs2_%s.txt'%img_name
log2_file = open(log2_path, "w")

result_path = file_name + '/result_%s.txt'%img_name
result_file = open(result_path, "w")

# load data

if fname == 'data/denoising/snail.jpg':
    img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_noisy_np = pil_to_np(img_noisy_pil)
    
    # As we don't have ground truth
    img_pil = img_noisy_pil
    img_np = img_noisy_np
    
    plot_image_grid([img_np], 4, 5, plot=PLOT)
        
elif fname == 'data/denoising/F16_GT.png':
    # Add synthetic noise
    img_pil = crop_image(get_image(fname, imsize)[0], d=32)
    # img_pil = img_pil.crop((0, 0, 256, 256))
    img_np = pil_to_np(img_pil)
    
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    
    plot_image_grid([img_np, img_noisy_np], 4, 6, plot=PLOT)
else:
    assert False
print(img_np.shape)



# if fname == 'data/denoising/snail.jpg':
#     num_iter = 2400
#     input_depth = 3
#     figsize = 5 
    
#     net = skip(
#                 input_depth, 3, 
#                 num_channels_down = [8, 16, 32, 64, 128], 
#                 num_channels_up   = [8, 16, 32, 64, 128],
#                 num_channels_skip = [0, 0, 0, 4, 4], 
#                 upsample_mode='bilinear',
#                 need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

#     net = net.type(dtype)
    
#     LR = 0.01

if fname == 'data/denoising/F16_GT.png':
    
    figsize = 4
    LR_min = 0
    
    if NET_TYPE == 'skip':
        input_depth = 32 # should be 32
        
        net = get_net(input_depth, 'skip', pad,
                    skip_n33d=128, 
                    skip_n33u=128, 
                    skip_n11=4, 
                    num_scales=5,
                    upsample_mode='bilinear').type(dtype)
        
        LR = 1e-2
        
    elif NET_TYPE == 'ViT':
        input_depth = 3
    
        net = ViT(depth=6, img_hsize=img_np.shape[1], img_wsize=img_np.shape[2]).type(dtype)
        
        LR = 1e-3
    
    elif NET_TYPE[0:4] == 'Swin':
        if 'Decoder' in NET_TYPE:
            input_depth = 768
        else:
            input_depth = 32

        net = locals()[NET_TYPE](img_size=img_np.shape[1],
                        in_chans=input_depth,
                        out_chans=3,
                        window_size=16,
                        wavelet_method=wavelet_method,
                        ).type(dtype)
        
        if OPTIMIZER == 'adam_gradual_warmup':
            LR = 5e-5
            LR_min = 1e-6
        else:
            LR = 5e-5
        
    elif NET_TYPE[0:7] == 'iformer':
        input_depth = 32
        
        net = locals()[NET_TYPE](in_chans=input_depth, img_size=(img_np.shape[1], img_np.shape[2])).type(dtype)
        
        LR = 5e-5
    
    else:
        assert False

else:
    assert False
    
if 'Decoder' in NET_TYPE:
    net_input = get_noise(input_depth, INPUT, (img_pil.size[1]//32, img_pil.size[0]//32)).type(dtype).detach()
else:
    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

net_input_np = torch_to_np(net_input)
# if input_depth == 1 or input_depth == 3:
#     plot_image_grid([np.clip(net_input_np, 0, 1)], factor=figsize, nrow=1, plot=PLOT, save_path=file_name+'/noise.png')

# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_noisy_last = 0
max_psnr = 0
best_iteration = 0
max_psnr_sm = 0
best_iteration_sm = 0

i = 0
def closure():
    
    global i, out, out_avg, psrn_noisy_last, last_net, net_input
    global max_psnr, best_iteration
    global max_psnr_sm, best_iteration_sm
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    
    out = net(net_input)
    
    if NET_TYPE == 'ViT':
        out = torch.reshape(out, (1, input_depth, img_np.shape[1], img_np.shape[2])) # 放到 ViT 里
    
    # print(out.shape)
    
    # print("out.shape: ", out.shape)
    
    # print("img_noisy_torch.shape: ", img_noisy_torch.shape)
    
    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
            
    total_loss = mse(out, img_noisy_torch)
    total_loss.backward()
    
    psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
    psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
    psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 
    
    pre_img = out.detach().cpu().numpy()[0]
    pre_img = pre_img.transpose(1, 2, 0)
    noisy_img = img_noisy_np.transpose(1, 2, 0)
    img = img_np.transpose(1, 2, 0)
    
    #frequency-band correspondence metric
    avg_mask_it = get_circular_statastic(pre_img, noisy_img,  size=0.2)
    avg_mask_it2 = get_circular_statastic(pre_img, img, size=0.2)
    
    #automatic stopping
    blur_it = PerceptualBlurMetric(pre_img)#the blurriness of the output image
    sharp_it = MLVSharpnessMeasure(pre_img)#the sharpness of the output image
    ratio_it = blur_it/sharp_it#the ratio
    
    # Note that we do not have GT for the "snail" example
    # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
    log_file.write('Iteration: %05d, Loss: %f, PSRN_gt: %f, mask: %s, ratio: %f\n' % (i, total_loss.item(), psrn_gt, avg_mask_it, ratio_it))
    log2_file.write('%s\n' % (avg_mask_it2))
    log_file.flush()
    log2_file.flush()
    # if  (i <= 5000 and i % show_every == 0) or (psrn_gt > 30 and psrn_gt > max_psnr + 0.02):
    if i % show_every == 0:
        out_np = torch_to_np(out)
        plot_image_grid([np.clip(out_np, 0, 1), np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1, plot=PLOT, save_path=file_name+'/denoising_%s.png'%str(i))
        
    
    if psrn_gt > max_psnr:
        max_psnr = psrn_gt
        best_iteration = i
        
    if psrn_gt_sm > max_psnr_sm:
        max_psnr_sm = psrn_gt_sm
        best_iteration_sm = i
        
    
    # Backtracking
    if i % show_every:
        if psrn_noisy - psrn_noisy_last < -5: 
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return total_loss*0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_noisy_last = psrn_noisy
            
    i += 1

    return total_loss

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter, LR_min)

log_file.close()
log2_file.close()
result_file.write('Max PSNR: '+str(max_psnr)+'\n')
result_file.write('Best Iteration: '+str(best_iteration)+'\n')
result_file.write('Max PSNR sm: '+str(max_psnr_sm)+'\n')
result_file.write('Best Iteration sm: '+str(best_iteration_sm)+'\n')
result_file.close()

out_np = torch_to_np(out)
plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13, plot=PLOT, save_path=file_name+'/%s_denoised.png'%img_name) # save the denoised image

loss_list, frequency_lists, psnr_list, ratio_list = get_log_data(log_path)
frequency_lists2 = get_log2_data(log2_path)

get_loss_fig(loss_list, num_iter, ylim=0.05, save_path=file_name+'/%s_loss.png'%img_name) # save the loss figure

get_fbc_fig(frequency_lists, num_iter, ylim=1, save_path=file_name+'/%s_fbc.png'%img_name) # save the fbc figure
get_fbc_fig(frequency_lists2, num_iter, ylim=1, save_path=file_name+'/%s_fbc_pure.png'%img_name) # save the fbc figure

data_lists =[]
data_lists.append(psnr_list)
data_lists.append(ratio_list)
get_psnr_ratio_fig(data_lists, num_iter, ylim=35, ylabel='PSNR', save_path=file_name+'/%s_psnr_ratio.png'%img_name) # save the psnr_ratio figure
