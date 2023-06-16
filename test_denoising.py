import os

log_path = 'test_denoising.log'
log_file = open(log_path, "w")


def denoising(hyperparameters):
    try:
        print('%s' % ('hyperparameters: ' + str(hyperparameters)))
        os.system(r'python ./denoising.py %s %s %s %s %s %s %s %s %s %s %s %s' %hyperparameters)
        
    except Exception as e:
        log_file.write('[Error] %s\n' % e)
        
    else:
        log_file.write('[Done]\n')
        
    log_file.write('%s\n' % ('hyperparameters: ' + str(hyperparameters)))

    log_file.flush()
    

img_name = 'F16_GT'

num_iter = 20000

sigma = 25 # 25, 50

reg_noise_std = 0. # 1./30., 1./20.

show_every = 1000

PLOT = False

NET_TYPE = 'SwinUnet2' # one of skip|ViT|SwinUnet2|Swin2Decoder|iformer_small|iformer_base|iformer_large

exp_name = '_skip_nopixel_noprenoise_noqkscale'

wavelet_method = 'None' # 'None', 'haar', 'db5', 'db9', 'db13', 'db25', 'db37' (only enable when SwinUnet2)

INPUT = 'noise' # 'noise', meshgrid', 'fourier'

OPTIMIZER = 'adam' # 'adam', 'LBFGS', 'adam_gradual_warmup'

exp_weight = 0.99


for sigma in (75, 100):
    for NET_TYPE in ('SwinUnet2', 'skip'):
        
        exp_name = '__skip_nopixel_noprenoise_noqkscale' + '_sgima' + str(sigma)
        
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
        ))

log_file.close()