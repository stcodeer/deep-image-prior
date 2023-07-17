import os

log_path = 'test_denoising.log'
log_file = open(log_path, "w")


def denoising(hyperparameters):
    try:
        print('%s' % ('hyperparameters: ' + str(hyperparameters)))
        os.system(r'python ./denoising.py %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s' %hyperparameters)
        
    except Exception as e:
        log_file.write('[Error] %s\n' % e)
        
    else:
        log_file.write('[Done]\n')
        
    log_file.write('%s\n' % ('hyperparameters: ' + str(hyperparameters)))

    log_file.flush()
    

img_name = 'F16_GT'

num_iter = 1000 # 20000

sigma = 25 # 25, 50

reg_noise_std = 0 # 1./30., 1./20.

show_every = 50

PLOT = False

# skip
# ViT
# SwinUnet2 | Swin2Decoder
# iformer_small | iformer_base | iformer_large
# Uformer_T | Uformer_S | Uformer_B | Uformer_S_noshift | Uformer_B_fastleff
# TransGAN_church | TransGAN_celeba
# ViTGAN
NET_TYPE = 'Uformer_T'

exp_name = '_' # '_skip_nopixel_noprenoise_noqkscale'

INPUT = 'noise' # 'noise', meshgrid', 'fourier'

OPTIMIZER = 'adam' # 'adam', 'LBFGS', 'adam_gradual_warmup'

exp_weight = 0.99 # 0.99

fbc_size = 0.2

# following hyperparameters only available for SwinUnet2

depths = 2222 # 2222

num_heads = 8888 # 8888

wavelet_method = 'None' # 'None', 'haar', 'db5', 'db9', 'db13', 'db25', 'db37'

core = 'transformer' # 'transformer', 'None', 'multi_head_mlp', 'mlp', 'cnn', 'position-wise mlp'(tmp unavailable), 'cnn1d'(tmp unavailable)

feat_scale = False # 'False', 'True'

attn_scale = False # 'False', 'True'

denoising((
    img_name,
    num_iter,
    sigma,
    reg_noise_std,
    show_every,
    PLOT,
    NET_TYPE,
    exp_name,
    wavelet_method,
    INPUT,
    OPTIMIZER,
    exp_weight,
    depths,
    num_heads,
    core,
    fbc_size,
    feat_scale,
    attn_scale,
))

log_file.close()