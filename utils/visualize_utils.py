from matplotlib import pyplot as plt
import numpy as np
import re
from .common_utils import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

plt.rcParams.update({'font.size': 18})


def get_log2_data(file_name):
    file1 = open(file_name, 'r')
    Lines = file1.readlines()
    file1.close()
    
    frequency_lists = []
    
    for line in Lines:
        strings = re.split('\[|\]|,\s', line.strip())
        
        fre_list = []
        
        for i in range(1, 6):
            fre_list.append(float(strings[i]))
            
        frequency_lists.append(np.array(fre_list))
        
    return frequency_lists


def get_log_data(file_name):
    file1 = open(file_name, 'r')
    Lines = file1.readlines()
    file1.close()

    loss_list=[]
    frequency_lists=[]
    psnr_list=[]
    ratio_list=[]

    for line in Lines:
        strings = re.split(':|,', line.strip())
        #print (strings)
        
        loss_list.append(float(strings[3]))
        
        fre_list = []
        fre_list.append(float(strings[-7][2:]))
        fre_list.append(float(strings[-6]))
        fre_list.append(float(strings[-5]))
        fre_list.append(float(strings[-4]))
        fre_list.append(float(strings[-3][:-1]))
        frequency_lists.append(np.array(fre_list))

        psnr_list.append(float(strings[5]))
        
        ratio_list.append(float(strings[-1]))

    return loss_list, frequency_lists, np.array(psnr_list), np.array(ratio_list)


def get_fbc_fig(all_norms,num_iter,ylim=1,save_path='',img_name=''):
    fig, ax = plt.subplots(figsize=(7,6))
    ax.set_xlim(0,num_iter)
    ax.set_ylim(0, ylim)

    norms=np.array(all_norms)

    # label_list = ['Frequency band (1,lowest)','Frequency band (2)','Frequency band (3)','Frequency band (4)','Frequency band (5, highest)']

    plt.xlabel("Optimization Iteration")
    plt.ylabel("FBC ($\\bar{H}$)")
    #plt.title('FBC (%s)'%img_name)

    color_list = ['#331900', '#994C00', '#CC6600',  '#FF8000', '#FF9933']
    rate = 1
    for i in range(norms.shape[1]):
        # plt.plot(range(0,num_iter,rate), norms[:num_iter:rate,i], linewidth=4, color=color_list[i], label=label_list[i]) 
        plt.plot(range(0,num_iter,rate), norms[:num_iter:rate,i], linewidth=4, color=color_list[i]) 

    # plt.legend(loc=4,)
    plt.grid()
    plt.savefig(save_path)
    # plt.show()
    plt.close()


def get_psnr_ratio_fig(all_datas,num_iter,ylim=35, ylabel='',save_path='',img_name=''):
    fig, ax = plt.subplots(figsize=(7,6))
    ax.set_xlim(0,num_iter)
    ax.set_ylim(0, ylim)

    plt.xlabel("Optimization Iteration")
    #plt.ylabel(ylabel)
    #plt.title(img_name)

    label_list = ['PSNR','Ratio']
    color_list = ['#d94a31','#4b43db']

    rate = 1
    for i in range(len(all_datas)):
        if i == 0:
            y_max = np.max(all_datas[i])
            y_max_idx = np.argmax(all_datas[i])
            
            plt.axhline(y_max, c=color_list[i], ls='-.')
            plt.axvline(y_max_idx, c=color_list[i], ls='-.')
            
        plt.plot(range(0,num_iter,rate), all_datas[i][0:num_iter:rate], linewidth=4, color=color_list[i], label=label_list[i])

    plt.legend(loc=0,)
    plt.grid()
    plt.savefig(save_path)
    # plt.show()
    plt.close()
    
    
def get_loss_fig(all_datas,num_iter,ylim=0.05, ylabel='',save_path='',img_name=''):
    fig, ax = plt.subplots(figsize=(7,6))
    ax.set_xlim(0, num_iter)
    ax.set_ylim(0, ylim)

    plt.xlabel('Optimization Iteration')
    plt.ylabel('Loss')

    rate = 1
    plt.plot(range(0,num_iter,rate), all_datas[0:num_iter:rate], linewidth=4)
    
    plt.grid()
    plt.savefig(save_path)
    # plt.show()
    plt.close()
    
    
def plot_frequency_distribution(img, size=0.2, scale='log', lim=-1, plot=False, save_path='default'):
    if len(img.shape)==3:
        img = rgb2gray(img)

    ftimage = np.fft.fft2(img)
    ftimage = abs(np.fft.fftshift(ftimage))

    h,w = ftimage.shape

    center = (int(w/2), int(h/2))
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    distribution = {}
    
    for y in range(h):
        for x in range(w):
            if dist_from_center[y][x] not in distribution:
                distribution[dist_from_center[y][x]] = ftimage[y][x]
            else:
                distribution[dist_from_center[y][x]] = distribution[dist_from_center[y][x]] + ftimage[y][x]
    
    distribution = sorted(distribution.items())
    
    index = [distribution[i][0] for i in range(len(distribution))]
    value = [distribution[i][1] for i in range(len(distribution))]
    
    fig, ax = plt.subplots(figsize=(7,6))
    
    plt.yscale(scale)
    
    if not lim == -1:
        ax.set_ylim(0, lim)
    
    plt.plot(index, value, '.')
    
    for sz in np.linspace(size, 1, int(1/size)):
        plt.axvline(center[0]*sz, c='r', ls='-.')
    
    if not save_path == 'default': 
        plt.savefig(save_path+'_distribution.png')
    if plot:
        print("plotting...")
        plt.show()
    plt.close()
    
    
def plot_frequency_figure(img, scale='log', lim=-1, plot=False, save_path='default'):
    if len(img.shape)==3:
        img = rgb2gray(img)

    ftimage = np.fft.fft2(img)
    ftimage = abs(np.fft.fftshift(ftimage))
    
    if scale == 'log':
        ftimage = np.log10(ftimage)
    
    if not lim == -1:
        ftimage = np.clip(ftimage, 0, lim)
    
    ftimage = np.expand_dims(ftimage, 0)
    
    plot_image_grid([ftimage], 7, 6, plot=plot, save_path=save_path+'_frequency.png' if not save_path=='default' else 'default')
    
    
def plot_filtered_figure(img, img_gt, size=0.2, plot=False, save_path='default'):
    if len(img.shape)==3:
        img = rgb2gray(img)
        img_gt = rgb2gray(img_gt)
    
    img_gt=img_gt.astype(np.float64)

    assert(size>0 and size<1)

    ftimage = np.fft.fft2(img)
    ftimage = np.fft.fftshift(ftimage) # no abs

    h,w = ftimage.shape

    center = (int(w/2), int(h/2))
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    for sz in np.linspace(size, 1, int(1/size)):
        radius = center[0] * sz # pow(center[0]**2+center[1]**2,0.5)
        mask = dist_from_center <= radius
        mask = mask.astype(np.int32)
        
        filtered_ftimage = mask * ftimage
        
        filtered_img = abs(np.fft.ifft2(np.fft.ifftshift(filtered_ftimage)))
        
        psnr_gt = compare_psnr(img_gt, filtered_img)
        
        filtered_img = np.expand_dims(filtered_img, 0)
        
        plot_image_grid([filtered_img], 7, 6, plot=plot, title='PSNR_GT: '+str(psnr_gt), save_path=save_path+'_filtered_'+str(sz)[0:3]+'.png' if not save_path=='default' else 'default')
        

def plot_fbc_stats(x, ys, lim=-1, title='default', labels='default', plot=False, plotlim=False, save_path='default'):
    
    fig, ax = plt.subplots(figsize=(30,15))
    
    if not title == 'default':
        ax.set_title(title, loc='left')
        
    c = ['r', 'g', 'b', 'k', 'y', 'c', 'm']
    # ls = ['-.', '-', '.', '--']
    
    for i, y in enumerate(ys):
        ax = plt.subplot(2, 4, i+1)
        
        if isinstance(lim, list) or isinstance(lim, tuple):
            ax.set_ylim(lim[0], lim[1])
        elif not lim == -1:
                ax.set_ylim(0, lim)
        
        if plotlim:
            y_max = np.max(y)
            y_max_idx = np.argmax(y)
            plt.axhline(y_max, c='r', ls='-.')
            plt.axvline(y_max_idx, c='r', ls='-.')
            
        plt.plot(x, y, '.', label=labels[i], c=c[0])
        
        plt.legend(loc='lower right')
        
    
    if not save_path == 'default': 
        plt.savefig(save_path)
    if plot:
        print("plotting...")
        plt.show()
    plt.close()
        