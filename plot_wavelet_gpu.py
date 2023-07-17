import torch
from pytorch_wavelets import DWTForward, DWTInverse
from utils.common_utils import *


def plot(a, save_path):
    x = a.numpy()
    plot_image_grid([(x-np.min(x))/(np.max(x)-np.min(x))], factor=13, save_path=save_path)
    

# read data
fname = 'data/denoising/F16_GT.png'
img_pil = crop_image(get_image(fname, -1)[0], d=32)
img_np = pil_to_np(img_pil)

# add noise 
sigma = 25
sigma_ = sigma/255.
img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

X = torch.tensor(img_noisy_np)
print(X.shape)

# x_before
plot(X, 'outputs/figs_wavelet/F16_GT_before_gpu.png')

# 2D harr wavelet transform
wave = 'db1'
mode = 'periodization'
xfm = DWTForward(J=1, wave=wave, mode=mode)
X = torch.unsqueeze(X, 0)
cA, cD = xfm(X)
cA = torch.squeeze(cA)
cD[0] = torch.squeeze(cD[0])
print(cD[0].shape)

cH, cV, cD = torch.chunk(cD[0], 3, 1)
cH = torch.squeeze(cH)
cV = torch.squeeze(cV)
cD = torch.squeeze(cD)


# average 
plot(cA, 'outputs/figs_wavelet/F16_GT_average_gpu.png')
# horizontal
plot(cH, 'outputs/figs_wavelet/F16_GT_horizontal_gpu.png')
# vertical
plot(cV, 'outputs/figs_wavelet/F16_GT_vertical_gpu.png')
# diagonal
plot(cD, 'outputs/figs_wavelet/F16_GT_diagonal_gpu.png')

# 2D harr inverse wavelet transform
cH = torch.unsqueeze(cH, 1)
cV = torch.unsqueeze(cV, 1)
cD = torch.unsqueeze(cD, 1)
cD = torch.cat((cH, cV, cD), 1)

cA = torch.unsqueeze(cA, 0)
cD = torch.unsqueeze(cD, 0)

# cD = torch.zeros(cD.shape)

ifm = DWTInverse(wave=wave, mode=mode)
Y = ifm((cA, [cD]))
Y = torch.squeeze(Y)
print(Y.shape)

# x_before
plot(Y, 'outputs/figs_wavelet/F16_GT_after_gpu.png')