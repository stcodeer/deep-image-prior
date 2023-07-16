# The code is based on code publicly available at
#   https://github.com/rosinality/stylegan2-pytorch
# written by Seonghyeon Kim.

import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from .common_ViTGAN.op import FusedLeakyReLU, conv2d_gradfix
from .common_ViTGAN.layers import PixelNorm, Upsample, Blur, EqualConv2d
from .common_ViTGAN.layers import EqualLinear, LFF
from .common_ViTGAN.vit_common import SpectralNorm, Attention, FeedForward
from .common_ViTGAN.vit_cips import CIPSGenerator
from .common_ViTGAN.op import FusedLeakyReLU
from .common_ViTGAN.generator import StyleLayer, ToRGB

def convert_to_coord_format(b, h, w, device='cpu', integer_values=False):
    if integer_values:
        x_channel = torch.arange(w, dtype=torch.float, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.arange(h, dtype=torch.float, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
    else:
        x_channel = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
    return torch.cat((x_channel, y_channel), dim=1)


class SelfModulatedLayerNorm(nn.Module):
    def __init__(self, dim, spectral_norm=False):
        super().__init__()
        self.param_free_norm = nn.LayerNorm(dim, eps=0.001, elementwise_affine=False)
        if spectral_norm:
            self.mlp_gamma = SpectralNorm(EqualLinear(dim, dim, activation='linear'))
            self.mlp_beta = SpectralNorm(EqualLinear(dim, dim, activation='linear'))
        else:
            self.mlp_gamma = EqualLinear(dim, dim, activation='linear')
            self.mlp_beta = EqualLinear(dim, dim, activation='linear')

    def forward(self, inputs):
        x, cond_input = inputs
        bs = x.shape[0]
        cond_input = cond_input.reshape((bs, -1))

        gamma = self.mlp_gamma(cond_input)
        gamma = gamma.reshape((bs, 1, -1))
        beta = self.mlp_beta(cond_input)
        beta = beta.reshape((bs, 1, -1))

        out = self.param_free_norm(x)
        out = out * (1.0 + gamma) + beta

        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim = 384, heads = 6, dim_head = 64, \
        mlp_dim = 1536, l2_attn = True, spectral_norm = True, dropout = 0.):

        super().__init__()
        self.layernorm1 = SelfModulatedLayerNorm(dim, spectral_norm = False)
        self.attn = Attention(dim, heads = heads, dim_head = dim_head, \
                    l2_attn = l2_attn, spectral_norm = spectral_norm, dropout = dropout)
        self.layernorm2 = SelfModulatedLayerNorm(dim, spectral_norm = False)
        self.ff = FeedForward(dim, mlp_dim, spectral_norm = spectral_norm, dropout = dropout)

        
    def forward(self, inputs):
        x, latent = inputs
        x = self.layernorm1([x, latent])
        x = self.attn(x) + x
        x = self.layernorm2([x, latent])
        x = self.ff(x) + x
        return x


class Generator(nn.Module):
    def __init__(self, size=32, token_width=8, num_layers=4,
                 embed_dim=384, n_mlp=8, channel_multiplier=2, small32=False,
                 blur_kernel=[1, 3, 3, 1], lr_mlp=0.01, use_nerf_proj=False):
        super().__init__()
        self.size = size
        self.token_width = token_width
        self.style_dim = embed_dim
        self.num_layers = num_layers
        self.use_nerf_proj = use_nerf_proj

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(self.style_dim, self.style_dim,
                                      lr_mul=lr_mlp,
                                      activation='fused_lrelu'))
                                      

        self.style = nn.Sequential(*layers)

        self.coords = convert_to_coord_format(1, token_width, token_width, integer_values=False, device='cpu')#self.device)
        self.lff = LFF(self.style_dim)

        self.feat_dim = {
            0: embed_dim,
            1: embed_dim,
            2: embed_dim,
            3: embed_dim,
            4: embed_dim,#int(256 * channel_multiplier),
            5: embed_dim,#int(128 * channel_multiplier),
            6: embed_dim,#int(64 * channel_multiplier),
            7: embed_dim,#int(32 * channel_multiplier),
            8: embed_dim,#int(16 * channel_multiplier),
        }

        self.cnn_channels = {
            8: embed_dim,
            16: embed_dim,
            32: embed_dim,
            64: int(embed_dim // 2 * channel_multiplier),
            128: int(embed_dim // 4 * channel_multiplier),
            256: int(embed_dim // 8 * channel_multiplier),
            512: int(embed_dim // 16 * channel_multiplier),
            1024: int(embed_dim // 32 * channel_multiplier),
        }

        #self.input = ConstantInput(self.channels[4])

        self.log_size = int(math.log(size, 2))

        self.layers = nn.ModuleList()
        self.convs = nn.ModuleList()
        #self.to_rgbs = nn.ModuleList()
        #self.noises = nn.Module()

        #in_channel = self.channels[4]
        for i in range(self.num_layers):
            this_dim = self.feat_dim[i]
            self.layers.append(TransformerBlock(dim = this_dim, heads = 6, dim_head = this_dim // 6, \
                                mlp_dim = this_dim*4, l2_attn = True, spectral_norm = True, dropout = 0.))

        self.layernorm = SelfModulatedLayerNorm(self.feat_dim[self.num_layers-1], spectral_norm=False)

        in_channel = self.cnn_channels[8]

        if self.use_nerf_proj == False:
            for i in range(4, self.log_size + 1):
                in_channel = self.cnn_channels[2 ** (i - 1)]
                out_channel = self.cnn_channels[2 ** i]
                self.convs.append(
                    StyleLayer(in_channel, out_channel, 3, self.style_dim,
                            upsample=True, blur_kernel=blur_kernel)
                )
                self.convs.append(
                    StyleLayer(out_channel, out_channel, 3, self.style_dim,
                            blur_kernel=blur_kernel)
                )
            self.to_rgb = ToRGB(out_channel, self.style_dim, upsample=False)
        else:
            self.cips = CIPSGenerator(size=size//token_width, style_dim=self.style_dim, n_mlp=4)


    @property
    def device(self):
        return self.lff.ffm.conv.weight.device

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(self, input,
                return_latents=False,
                input_is_latent=False,
                noise=None):

        latent = self.style(input) if not input_is_latent else input

        bs = latent.shape[0]
        coords = self.coords.repeat(bs, 1, 1, 1).to(self.device)#("cuda")
        pe = self.lff(coords)
        x = torch.permute(pe, (0, 2, 3, 1)).reshape((bs, -1, self.style_dim))

        for layer in self.layers:
            x = layer([x, latent])
        x = self.layernorm([x, latent])
        
        if self.use_nerf_proj == False:
            x = x.reshape((bs, self.token_width, self.token_width, x.shape[-1]))
            x = x.permute((0, 3, 1, 2))
            for conv_layer in self.convs:
                x = conv_layer(x, latent)

            x = self.to_rgb(x, latent)
        else:
            x = x.reshape((-1, x.shape[-1]))
            x = self.cips(x)
            mul = self.size // self.token_width
            #torch.Size([128, 3, 4, 4])

            x = x.reshape((bs, self.token_width, self.token_width, 3, mul, mul))
            x = x.permute((0, 3, 1, 4, 2, 5))
            x = x.reshape([bs, 3, self.size, self.size])

        return x
    
def get_architecture(image_size, token_width=8, num_layers=4, embed_dim=384, use_nerf_proj=False):
    generator = Generator(size=image_size, token_width=token_width, num_layers=num_layers, embed_dim=embed_dim, n_mlp=8, small32=True, use_nerf_proj=use_nerf_proj)

    return generator