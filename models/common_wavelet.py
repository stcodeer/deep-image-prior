import torch
import pywt
from utils.common_utils import plot_image_grid
from pytorch_wavelets import DWTForward, DWTInverse


def dwt_np(x, wavelet='haar'):
    """
    2D discrete wavelet transform
    input: B C H W
    output: B C*4 H//2 W//2
    """
    cA, (cH, cV, cD) = pywt.dwt2(x, wavelet, mode='periodization')
    
    # # x_before
    # plot_image_grid([x], factor=13, save_path='x_before.png')
    # # average 
    # plot_image_grid([(cA-np.min(cA))/(np.max(cA)-np.min(cA))], factor=13, save_path='A.png')
    # # horizontal
    # plot_image_grid([(cH-np.min(cH))/(np.max(cH)-np.min(cH))], factor=13, save_path='H.png')
    # # vertical
    # plot_image_grid([(cV-np.min(cV))/(np.max(cV)-np.min(cV))], factor=13, save_path='V.png')
    # # diagonal
    # plot_image_grid([(cD-np.min(cD))/(np.max(cD)-np.min(cD))], factor=13, save_path='D.png')
    
    x = torch.cat([cA, cH, cV, cD], dim=1)
    
    return x

def idwt_np(x, wavelet='haar'):
    """
    2D inverse discrete wavelet transform
    input: B C H W
    output: B C//4 H*2 W*2
    """
    assert x.shape[1] % 4 == 0, 'x should be divided by 4'
    
    cA, cH, cV, cD = torch.chunk(x, 4, 1)
        
    x = pywt.idwt2((cA, (cH, cV, cD)), wavelet, mode='periodization')
    
    # # x_after
    # plot_image_grid([x], factor=13, save_path='x_after.png')
    
    return x

def dwt(x, wavelet='haar'):
    """
    2D discrete wavelet transform
    input: B C H W
    output: B C*4 H//2 W//2
    """
    xfm = DWTForward(J=1, wave=wavelet, mode='periodization').cuda()
    cA, cD = xfm(x)

    cH, cV, cD = torch.chunk(cD[0], 3, 2)
    cH = torch.squeeze(cH, 2)
    cV = torch.squeeze(cV, 2)
    cD = torch.squeeze(cD, 2)
    
    x = torch.cat([cA, cH, cV, cD], dim=1)
    
    return x
    
def idwt(x, wavelet='haar'):
    """
    2D inverse discrete wavelet transform
    input: B C H W
    output: B C//4 H*2 W*2
    """
    assert x.shape[1] % 4 == 0, 'x should be divided by 4'
    
    cA, cH, cV, cD = torch.chunk(x, 4, 1)
    
    cH = torch.unsqueeze(cH, 2)
    cV = torch.unsqueeze(cV, 2)
    cD = torch.unsqueeze(cD, 2)
    cD = torch.cat((cH, cV, cD), 2)

    ifm = DWTInverse(wave=wavelet, mode='periodization').cuda()
    x = ifm((cA, [cD]))
    
    return x