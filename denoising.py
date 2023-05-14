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
from models.swin_transformer import SwinTransformer
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

imsize =-1
PLOT = True
sigma = 25
sigma_ = sigma/255.

img_name = 'F16_GT'

# denoising
fname = 'data/denoising/%s.png'%img_name

NET_TYPE = 'skip' # one of skip|ViT|Swin|iformer_small|iformer_base|iformer_large

exp_num = 0

if not os.path.exists('outputs'):
        os.mkdir('outputs')

file_name = 'outputs/' + img_name + '_' + NET_TYPE + '_' + str(exp_num)

if not os.path.exists(file_name):
        os.mkdir(file_name)

log_path = file_name + '/logs_%s.txt'%img_name
log_file = open(log_path, "w")

# load data

if fname == 'data/denoising/snail.jpg':
    img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_noisy_np = pil_to_np(img_noisy_pil)
    
    # As we don't have ground truth
    img_pil = img_noisy_pil
    img_np = img_noisy_np
    
    if PLOT:
        plot_image_grid([img_np], 4, 5)
        
elif fname == 'data/denoising/F16_GT.png':
    # Add synthetic noise
    img_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_np = pil_to_np(img_pil)
    
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    
    if PLOT:
        plot_image_grid([img_np, img_noisy_np], 4, 6)
else:
    assert False
print(img_np.shape)

INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

reg_noise_std = 1./30. # set to 1./20. for sigma=50

OPTIMIZER='adam' # 'LBFGS'
show_every = 100
exp_weight=0.99

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
    num_iter = 5000
    figsize = 4
    if NET_TYPE == 'skip':
        input_depth = 32 # should be 32
        
        net = get_net(input_depth, 'skip', pad,
                    skip_n33d=128, 
                    skip_n33u=128, 
                    skip_n11=4, 
                    num_scales=5,
                    upsample_mode='bilinear').type(dtype)
        
        LR = 0.01
        
    elif NET_TYPE == 'ViT':
        input_depth = 3
    
        net = ViT(depth=6, img_hsize=img_np.shape[1], img_wsize=img_np.shape[2]).type(dtype)
        
        LR = 0.001
    
    elif NET_TYPE == 'Swin':
        input_depth = 3
        
        net = SwinTransformer(img_size=(img_np.shape[1], img_np.shape[2]), 
                            window_size=16, 
                            embed_dim=96, 
                            drop_rate=0.0, 
                            attn_drop_rate=0.0).type(dtype)
        
        LR = 0.001
        
    elif NET_TYPE[0:7] == 'iformer':
        input_depth = 32
        
        net = locals()[NET_TYPE](in_chans=input_depth, img_size=(img_np.shape[1], img_np.shape[2])).type(dtype)
        
        LR = 0.002
    
    else:
        assert False

else:
    assert False
    
net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach() # 这个地方会不会在长宽不等的时候弄反？

# net_input_np = torch_to_np(net_input)
# plot_image_grid([np.clip(net_input_np, 0, 1)], factor=figsize, nrow=1)

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

i = 0
def closure():
    
    global i, out, out_avg, psrn_noisy_last, last_net, net_input
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    
    out = net(net_input)
    
    if NET_TYPE == 'ViT':
        out = torch.reshape(out, (1, input_depth, img_np.shape[1], img_np.shape[2]))
    
    # if NET_TYPE == 'Swin':
    #     dosomething
    
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
    
    #frequency-band correspondence metric
    avg_mask_it = get_circular_statastic(pre_img, noisy_img,  size=0.2)
    
    #automatic stopping
    blur_it = PerceptualBlurMetric (pre_img)#the blurriness of the output image
    sharp_it = MLVSharpnessMeasure(pre_img)#the sharpness of the output image
    ratio_it = blur_it/sharp_it#the ratio
    
    # Note that we do not have GT for the "snail" example
    # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
    log_file.write('Iteration: %05d, Loss: %f, PSRN_gt: %f, mask: %s, ratio: %f\n' % (i, total_loss.item(), psrn_gt, avg_mask_it, ratio_it))
    log_file.flush()
    if  PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        plt = plot_image_grid([np.clip(out_np, 0, 1),
                         np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)
        plt.savefig(file_name+'/denoising_%s.png'%str(i))
        
        
    
    # Backtracking
    # if i % show_every:
    #     if psrn_noisy - psrn_noisy_last < -5: 
    #         print('Falling back to previous checkpoint.')

    #         for new_param, net_param in zip(last_net, net.parameters()):
    #             net_param.data.copy_(new_param.cuda())

    #         return total_loss*0
    #     else:
    #         last_net = [x.detach().cpu() for x in net.parameters()]
    #         psrn_noisy_last = psrn_noisy
            
    i += 1

    return total_loss

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)
log_file.close()

out_np = torch_to_np(out)
plt = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13)
plt.savefig(file_name+'/%s_denoised.png'%img_name) # save the denoised image

frequency_lists, psnr_list, ratio_list = get_log_data(log_path)
get_fbc_fig(frequency_lists, num_iter, ylim=1, save_path=file_name+'/%s_fbc.png'%img_name) # save the fbc figure

data_lists =[]
data_lists.append(psnr_list)
data_lists.append(ratio_list)
get_psnr_ratio_fig(data_lists, num_iter, ylim=35, ylabel='PSNR', save_path=file_name+'/%s_psnr_ratio.png'%img_name) # save the psnr_ratio figure