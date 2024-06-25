# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:53:32 2023

@author: mlocarno
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from cmcrameri import cm
from scipy.signal import argrelextrema
import matplotlib.ticker as ticker

# path='M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data/Marco/phd/2023-09-15 Before after NSs - Q1/'
path='M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data/Marco/phd/2023-03-21 Before after NSs - Q6a - 100ms exposure - telescope/'

plt.rcParams.update({'font.size': 16})

def convertTIF(file_path_analyze):  #converts .TIF to numpy array
    dataset = Image.open(file_path_analyze)
    h,w = np.shape(dataset)
    frames=dataset.n_frames #maximum (for 256 pixel image) 16000 
    tiffarray = np.zeros((h,w,frames))
    for i in range(frames):
       dataset.seek(i)
       try:
           tiffarray[:,:,i] = np.array(dataset)
       except:
           print("Error at frame number "+format(i))      
           break
    return tiffarray.astype(np.double),frames

def map2D(img):
    plt.figure()
    plt.imshow(img,cmap=cm.batlow)
    cbar=plt.colorbar()
    cbar.set_label(label='Signal',rotation=270, labelpad=20)
    plt.axis('off')
    # plt.savefig(folder+"/plots/"+file_name+'map.png',dpi=300)
    plt.show()
    return True

def extract_fluorescence(img):
    std=np.std(img,axis=2)
    # map2D(std)

    hist_std=np.histogram(std.flatten(),bins=100)
    # plt.hist(std.flatten(),bins=hist_std[1])
    # plt.show()

    #Lmin=argrelextrema(hist_std[0], np.less)[0][0] #position of first local minima
    #threshold_cell=(hist_std[1][Lmin]+hist_std[1][Lmin+1])/2
    threshold_cell=25

    # print('Suggested threshold cell: '+format(threshold_cell))
    idx=std>threshold_cell
    # map2D(idx)

    Lmax=argrelextrema(hist_std[0], np.greater)[0][0] #position of first local maxima
    threshold_bg=(hist_std[1][Lmax]+hist_std[1][Lmax+1])/2

    idx_bg=std<=threshold_bg
    # map2D(idx_bg)
    
    # while input('If you want to change it, input "C": ')=='C':
    #     threshold_cell=float(input('Input a different threshold: '))
        
    #     idx=std>threshold_cell
    #     map2D(idx)
    
    #     Lmax=argrelextrema(hist_std[0], np.greater)[0][0] #position of first local maxima
    #     threshold_bg=(hist_std[1][Lmax]+hist_std[1][Lmax+1])/2
    
    #     idx_bg=std<=threshold_bg
    #     # map2D(idx_bg)

    maskbg=np.ma.masked_array(np.mean(img,axis=2),~idx_bg)
    bg=maskbg.mean()

    signal=np.tensordot(idx,img-bg)/np.sum(idx)
    signal=signal/signal[0]
        
    return signal


if __name__ == '__main__':
    folders = ['control', 'nss - 400uL']
    # nFOV = {'control': 5, 'nss - 400uL': 4}
    nFOV = {'control': 10, 'nss - 400uL': 8}
    # C_time = np.array([0, 5, 10, 20, 30])
    # N_time = np.array([0, 1, 5, 10, 20, 30])
    C_time = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    N_time = np.array([0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    
    plt.figure()
    for folder in folders:
        if folder == 'control':
            time = C_time
            color=[0.35, 0.7, 0.9, 0.5]
        else:
            time = N_time
            color=[0.9, 0.6, 0, 0.5]
        mean_signal = np.zeros([len(time)])
        M2 = np.zeros([len(time)])

        for i in range(nFOV[folder]):
            name = '/fov' + format(i) + '.tif'
            img, frames = convertTIF(path + folder + name)
            # map2D(np.mean(img,axis=2))
            signal = extract_fluorescence(img)
            # Add scatter plot for individual data points
            plt.scatter(time, signal, color=color, marker='x')
            # Add thin lines connecting data points
            plt.plot(time, signal, color=color, linestyle='-', linewidth=0.5)
            delta = signal - mean_signal
            mean_signal += delta / (i + 1)
            delta2 = signal - mean_signal
            M2 += delta * delta2

        variance = M2 / i
        STD_signal = np.sqrt(variance)
        if folder == 'control':
            C_mean = mean_signal
            C_sem = STD_signal / np.sqrt(nFOV[folder])
            data_tosave = np.column_stack((C_time, C_mean, C_sem))
            # Combine time_points and signal[206:217]
            np.savetxt(path+'control.txt', data_tosave, delimiter=', ', fmt='%.3f')
            # Save the data to a text file
            C=C_mean[-1]
            dC=C_sem[-1]
        else:
            N_mean = mean_signal
            N_sem = STD_signal / np.sqrt(nFOV[folder])
            data_tosave = np.column_stack((N_time, N_mean, N_sem))
            # Combine time_points and signal[206:217]
            np.savetxt(path+'NSs.txt', data_tosave, delimiter=', ', fmt='%.3f')
            # Save the data to a text file
            N=N_mean[-1]
            dN=N_sem[-1]

    # Add error bar plots
    plt.errorbar(C_time, C_mean, yerr=C_sem, fmt='--o', capsize=5, capthick=2, elinewidth=2, linewidth=2,
                 label='Control', color=[0.35, 0.7, 0.9])
    plt.errorbar(N_time, N_mean, yerr=N_sem, fmt='-o', capsize=5, capthick=2, elinewidth=2, linewidth=2,
                 label='NSs', color=[0.9, 0.6, 0])

    # Add legend, labels, and formatting
    plt.legend(loc='best')
    plt.xlabel('Time after addition [min]')
    plt.ylabel('F / F(t=0) [a.u.]')
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(5))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    fig = plt.gcf()
    fig.set_size_inches(12 / 2.54, 12 / 2.54)  # 12cmx12cm
    plt.tight_layout(pad=0.5)
    plt.savefig(path+'Comparison.png',dpi=300)
    plt.show()
    
    FE=(N-C)/C*100
    dFE=np.sqrt((dN/C)**2+(N*dC/(C**2))**2)*100
    print('Fluorescence enhancement: ('+format(FE)+' +/- '+format(dFE)+') %')