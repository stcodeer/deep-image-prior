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
from models import get_net
from models.ViT import ViT
from models.swin_transformer_2 import SwinUnet2
from models.swin_transformer_2_decoder import Swin2Decoder
from models.inception_transformer import iformer_small, iformer_base, iformer_large
from models.Generator_TransGAN_church import Generator as TransGAN_church
from models.Generator_TransGAN_celeba import Generator as TransGAN_celeba
from models.Generator_ViTGAN import get_architecture as ViTGAN
from models.Uformer import get_arch as Uformer

import torch
import torch.optim

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.measure_utils import *
from utils.common_utils import *
from utils.visualize_utils import *
from distutils.util import strtobool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # run on GPU
if os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
    dtype = torch.FloatTensor
else:
    dtype = torch.cuda.FloatTensor

# hyperparameter

img_name = sys.argv[1]
num_iter = int(sys.argv[2])
sigma = int(sys.argv[3])
reg_noise_std = float(sys.argv[4])
show_every = int(sys.argv[5])
PLOT = bool(strtobool(sys.argv[6]))
NET_TYPE = sys.argv[7]
exp_name = sys.argv[8]
wavelet_method = sys.argv[9]
INPUT = sys.argv[10]
OPTIMIZER = sys.argv[11]
exp_weight = float(sys.argv[12])
depths = [int(sys.argv[13][i]) for i in range(4)]
num_heads = [int(sys.argv[14][i]) for i in range(4)]
core = sys.argv[15]
fbc_size = float(sys.argv[16])
feat_scale = bool(strtobool(sys.argv[17]))
attn_scale = bool(strtobool(sys.argv[18]))

pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

fname = 'data/denoising/%s.png'%img_name

imsize = -1
sigma_ = sigma/255.

# output path

if not os.path.exists('outputs'):
        os.mkdir('outputs')

file_name = 'outputs/' + img_name + '_' + NET_TYPE + '_' + str(num_iter) + '_' + INPUT + exp_name

if not wavelet_method == 'None':
    file_name = file_name + '_' + wavelet_method + 'wavelet'

print('output file name: ', file_name)

if not os.path.exists(file_name):
        os.mkdir(file_name)

log_path = file_name + '/logs_%s.txt'%img_name
log_file = open(log_path, "w")

log2_path = file_name + '/logs2_%s.txt'%img_name
log2_file = open(log2_path, "w")

# log3_path = file_name + '/logs3_%s.txt'%img_name
# log3_file = open(log3_path, "w")

# log_gray_path = file_name + '/logs_gray.txt'
# log_gray_file = open(log_gray_path, "w")

result_path = file_name + '/result_%s.txt'%img_name
result_file = open(result_path, "w")

# load data

# if fname == 'data/denoising/snail.jpg':
#     img_noisy_pil = crop_image(get_image(fname, imsize)[0], d=32)
#     img_noisy_np = pil_to_np(img_noisy_pil)
    
#     # As we don't have ground truth
#     img_pil = img_noisy_pil
#     img_np = img_noisy_np
    
#     plot_image_grid([img_np], 4, 5, plot=PLOT)
        
if fname == 'data/denoising/F16_GT.png':
    # Add synthetic noise
    img_pil = crop_image(get_image(fname, imsize)[0], d=32)
    # img_pil = img_pil.crop((0, 0, 256, 256))
    img_np = pil_to_np(img_pil)
    
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    
    # noisy_np = img_noisy_np - img_np
    
    plot_image_grid([img_np, img_noisy_np], 4, 6, plot=PLOT)
    
else:
    assert False


# rgb to gray

# img_np = img_np.transpose(1, 2, 0)
# img_noisy_np = img_noisy_np.transpose(1, 2, 0)

# img_np = np.expand_dims(rgb2gray(img_np), 0)
# img_noisy_np = np.expand_dims(rgb2gray(img_noisy_np), 0)

# log_gray_file.write(str(img_np.tolist())+'\n')
# log_gray_file.write(str(img_noisy_np.tolist()))

# log_gray_file.close()


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
                    n_channels=img_np.shape[0],
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
    
    elif 'Swin' in NET_TYPE:
        if 'Decoder' in NET_TYPE:
            input_depth = 768
        else:
            input_depth = 32

        net = locals()[NET_TYPE](
                        core=core,
                        img_size=img_np.shape[1],
                        in_chans=input_depth,
                        out_chans=img_np.shape[0],
                        window_size=16,
                        wavelet_method=wavelet_method,
                        depths=depths,
                        num_heads=num_heads,
                        feat_scale=feat_scale,
                        attn_scale=attn_scale,
                        ).type(dtype)
        
        if OPTIMIZER == 'adam_gradual_warmup':
            LR = 5e-5
            LR_min = 1e-6
        else:
            LR = 5e-5
        
    elif 'iformer' in NET_TYPE:
        input_depth = 32
        
        net = locals()[NET_TYPE](in_chans=input_depth, img_size=(img_np.shape[1], img_np.shape[2])).type(dtype)
        
        LR = 5e-5
        
    elif 'Uformer' in NET_TYPE:
        input_depth = 32
        
        net = Uformer(NET_TYPE,
                      img_size=img_np.shape[1],
                      in_chans=input_depth,
                      out_chans=img_np.shape[0],
                      ).type(dtype)
        
        LR = 5e-5
    
    elif 'TransGAN' in NET_TYPE:
        batch_size = 1 # 100
        latent_dim = 1024 # 512
        embed_dim = 256 # 1024
        window_size = 8 # 16
        
        net = locals()[NET_TYPE](
                        img_size=img_np.shape[1],
                        in_chans=batch_size,
                        window_size=window_size,
                        latent_dim=latent_dim,
                        embed_dim=embed_dim,
                        depth=[2, 2, 2, 2, 2, 2],
                        ).type(dtype)
        
        LR = 5e-5

    elif NET_TYPE == 'ViTGAN':
        num_samples = 1 # img_np.shape[1]
        token_width = 8 # 8
        num_layers = 4 # 4
        embed_dim = 384 # 384
        cips_dim = 256 # 512
        use_nerf_proj = True
        
        net = ViTGAN(image_size=img_np.shape[1],
                     token_width=token_width,
                     num_layers=num_layers,
                     embed_dim=embed_dim,
                     cips_dim=cips_dim,
                     use_nerf_proj=use_nerf_proj,
                     ).type(dtype)
        
        LR = 1e-4

    else:
        assert False

else:
    assert False


# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)


# Random Input z
if 'Decoder' in NET_TYPE:
    net_input = get_noise(input_depth, INPUT, (img_pil.size[1]//32, img_pil.size[0]//32)).type(dtype).detach()
elif 'TransGAN' in NET_TYPE:
    net_input = torch.normal(0, 1, (batch_size, latent_dim)).type(dtype).detach()
elif NET_TYPE == 'ViTGAN':
    net_input = torch.randn(num_samples, embed_dim).type(dtype).detach()
else:
    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

net_input_np = torch_to_np(net_input)
# if input_depth == 1 or input_depth == 3:
#     plot_image_grid([np.clip(net_input_np, 0, 1)], factor=figsize, nrow=1, plot=PLOT, save_path=file_name+'/noise.png')


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
    # out = torch.utils.checkpoint.checkpoint_sequential(net, 5, net_input)
    
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
    # noisy = noisy_np.transpose(1, 2, 0)
    
    #frequency-band correspondence metric
    avg_mask_it = get_circular_statastic(pre_img, noisy_img,  size=fbc_size)
    avg_mask_it2 = get_circular_statastic(pre_img, img, size=fbc_size)
    # avg_mask_it3 = get_circular_statastic(pre_img, noisy, size=fbc_size)
    
    #automatic stopping
    blur_it = PerceptualBlurMetric(pre_img)#the blurriness of the output image
    sharp_it = MLVSharpnessMeasure(pre_img)#the sharpness of the output image
    ratio_it = blur_it/sharp_it#the ratio
    
    # Note that we do not have GT for the "snail" example
    # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
    log_file.write('Iteration: %05d, Loss: %f, PSRN_gt: %f, mask: %s, ratio: %f\n' % (i, total_loss.item(), psrn_gt, avg_mask_it, ratio_it))
    log2_file.write('%s\n' % (avg_mask_it2))
    # log3_file.write('%s\n' % (avg_mask_it3))
    log_file.flush()
    log2_file.flush()
    # log3_file.flush()
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
        
    
    # # Backtracking
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
optimize(OPTIMIZER, p, closure, LR, num_iter, LR_min)

log_file.close()
log2_file.close()
# log3_file.close()
result_file.write('Max PSNR: '+str(max_psnr)+'\n')
result_file.write('Best Iteration: '+str(best_iteration)+'\n')
result_file.write('Max PSNR sm: '+str(max_psnr_sm)+'\n')
result_file.write('Best Iteration sm: '+str(best_iteration_sm)+'\n')
result_file.close()

out_np = torch_to_np(out)
plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13, plot=PLOT, save_path=file_name+'/%s_denoised.png'%img_name) # save the denoised image

loss_list, frequency_lists, psnr_list, ratio_list = get_log_data(log_path)
frequency_lists2 = get_frequency_data(log2_path)

get_loss_fig(loss_list, num_iter, ylim=0.05, save_path=file_name+'/%s_loss.png'%img_name) # save the loss figure

get_fbc_fig(frequency_lists, num_iter, ylim=1, save_path=file_name+'/%s_fbc.png'%img_name) # save the fbc figure
get_fbc_fig(frequency_lists2, num_iter, ylim=1, save_path=file_name+'/%s_fbc_pure.png'%img_name) # save the fbc figure

data_lists =[]
data_lists.append(psnr_list)
data_lists.append(ratio_list)
get_psnr_ratio_fig(data_lists, num_iter, ylim=35, ylabel='PSNR', save_path=file_name+'/%s_psnr_ratio.png'%img_name) # save the psnr_ratio figure
