# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 17:20:48 2023

@author: mlocarno
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from cmcrameri import cm
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit

path_Control='M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data/GR project/2022-03-17 GR new single+combo mutants and arch versions/QuasAr6a/patchclamp/'
path_NSs='M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data/Marco/phd/2023-01-24 Patch Clamp QuasAr6a on NSs/'
path_Analysis='M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/Data/Marco/phd/2024-01-30 Sensitivity speed analysis Q6 NSs (new)/'
folder_Control=np.array(['cell1/more focused/','cell2/square 5hz/','cell3/square 5hz/','cell4/square 5hz/'])
folder_NSs=np.array(['cell1/','cell2/','cell3/','cell4/','cell5/'])
name='1000fps.tif'

fps=1000
freq=5

# Set the default line width and font size
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['font.size'] = 14

#CVD-friendly qualitatively color cycle
OkabeIto=np.array([[0,0,0],
                   [0.9,0.6,0],
                   [0.35,0.7,0.9],
                   [0,0.6,0.5],
                   [0.95,0.9,0.25],
                   [0,0.45,0.7],
                   [0.8,0.4,0],
                   [0.8,0.6,0.7]])

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
    map2D(std)

    hist_std=np.histogram(std.flatten(),bins=100)
    plt.hist(std.flatten(),bins=hist_std[1])
    plt.show()

    Lmin=argrelextrema(hist_std[0], np.less)[0][0] #position of first local minima
    threshold_cell=(hist_std[1][Lmin]+hist_std[1][Lmin+1])/2

    print('Suggested threshold cell: '+format(threshold_cell))
    answer=input('If you want to change it, input "C": ')
    if answer=='C':
        threshold_cell=float(input('Input a different threshold: '))

    idx=std>threshold_cell
    map2D(idx)

    Lmax=argrelextrema(hist_std[0], np.greater)[0][0] #position of first local maxima
    threshold_bg=(hist_std[1][Lmax]+hist_std[1][Lmax+1])/2

    idx_bg=std<=threshold_bg
    map2D(idx_bg)

    maskbg=np.ma.masked_array(np.mean(img,axis=2),~idx_bg)
    bg=maskbg.mean()

    signal=np.tensordot(idx,img-bg)/np.sum(idx)
    plt.plot(signal)
    plt.show()
    
    return signal,threshold_cell,threshold_bg

def avg_period(signal,fps,freq,figname='single period.png'):
    period=int(fps/freq)
    stop=(len(signal)+1)//period
    d=stop*period-len(signal)
    # print(d)
    # d=0
    time=signal[:,0]
    avg=np.zeros([period-d])
    
    for i in range(1,stop):
        if i<stop-1:
            avg=(avg*(i-1)+signal[int(i*period)-d:int((i+1)*period-d-1),1])/i
        else:
            avg=(avg*(i-1)+signal[int(i*period):int((i+1)*period),1])/i
    
    avg=np.roll(avg,1)
    
    S=extract_sensitivity(avg)
    
    popt=fit_single_exp(time[0:period//2-d], avg[0:period//2-d])
    poptdown=fit_single_exp(time[0:period//2-d], avg[period//2:period-d])
    fit_up=single_exp(time[0:period//2-d],*popt)
    fit_down=single_exp(time[0:period//2-d],*poptdown)
    
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=OkabeIto)
    line, =plt.plot(time[0:period-d]*1e3,avg,'-',linewidth=1,label='Data')
    plt.plot(time[0:period//2-d]*1e3,fit_up,'--',label='Fit upswing')
    plt.plot((time[0:period//2-d]+time[period//2-d])*1e3,fit_down,':', label='Fit downswing')
    plt.xlabel('Time [ms]')
    plt.ylabel('Average fluorescence\n[counts/px]')
 
    tau_up=popt[1]
    tau_down=poptdown[1]
    leg=plt.legend(title='\u0394F/F ='+'{:.0f}'.format(S*100)+'%\n\u03C4$_{up}$ = '+'{:.1f}'.format(tau_up*1000)+' ms \n\u03C4$_{down}$ = '+'{:.1f}'.format(tau_down*1000)+' ms',loc='upper right')
    
    leg._legend_box.align = "left"
    fig = plt.gcf()
    fig.set_size_inches(16/2.54, 10/2.54)
    plt.tight_layout(pad=1)
    plt.savefig(figname,dpi=300)
    plt.show()
    
    return avg,tau_up,tau_down,S

def extract_sensitivity(avg):
    h=np.mean(avg[np.where(avg>=0.8*(avg[99]-avg[0])+avg[0])[0]])
    l=np.mean(avg[np.where(avg<=0.2*(avg[100]-avg[198])+avg[198])[0]])
    S=(h-l)/l
    return S

def single_exp(t, A, tau, C):
    return A*np.exp(-t/tau) + C

def fit_single_exp(time_axis, avg_signals):
    # initial guesses for fit parameters
    A0 = avg_signals[0] - avg_signals[-1]
    tau0 = (time_axis[-1] - time_axis[0]) / 2
    C0 = avg_signals[-1]

    # perform curve fitting
    p0 = [A0, tau0, C0]
    popt, pcov = curve_fit(single_exp, time_axis, avg_signals, p0=p0)
    return popt

def extract_traces():
    for folder in folder_Control:
        n=np.where(folder_Control==folder)[0][0]
        img,frames=convertTIF(path_Control+folder+name)
        signal,threshold_cell,threshold_bg=extract_fluorescence(img)
        data_tosave = np.column_stack((np.linspace(0,4.998,len(signal)), signal))
        header_str = f"Threshold cell = {threshold_cell:.3f}\nThreshold background = {threshold_bg:.3f}\n"
        np.savetxt(path_Control + 'raw_' + format(n + 1) + '.txt', data_tosave, delimiter=', ', fmt='%.3f', header=header_str)

    for folder in folder_NSs:
        n=np.where(folder_NSs==folder)[0][0]
        img,frames=convertTIF(path_NSs+folder+name)
        signal,threshold_cell,threshold_bg=extract_fluorescence(img)
        data_tosave = np.column_stack((np.linspace(0,4.998,len(signal)), signal)) # Combine time_points and signal[206:217]
        data_tosave = np.column_stack((np.linspace(0,4.998,len(signal)), signal))
        header_str = f"Threshold cell = {threshold_cell:.3f}\nThreshold background = {threshold_bg:.3f}\n"
        np.savetxt(path_NSs + 'raw_' + format(n + 1) + '.txt', data_tosave, delimiter=', ', fmt='%.3f', header=header_str)

def extract_dFF_speed():
    #Control
    C_up=np.array([])
    C_down=np.array([])
    C_S=np.array([])
    
    for cell in range(1,len(folder_Control)+1):
        signal=np.loadtxt(path_Control + 'raw_' + format(cell) + '.txt', delimiter=', ')
        avg,tau_up,tau_down,S=avg_period(signal,fps,freq,figname=path_Analysis+'avg_control_'+format(cell)+'.png') 
        C_up=np.append(C_up,tau_up)
        C_down=np.append(C_down,tau_down)
        C_S=np.append(C_S,S)

    np.savetxt(path_Analysis + 'Tau_UP_Control.txt', C_up, delimiter=', ', fmt='%.4f')
    np.savetxt(path_Analysis + 'Tau_DOWN_Control.txt', C_down, delimiter=', ', fmt='%.4f')
    np.savetxt(path_Analysis + 'Sensitivity_Control.txt', C_S, delimiter=', ', fmt='%.3f')

    #Nanostars
    N_up=np.array([])
    N_down=np.array([])
    N_S=np.array([])
    
    for cell in range(1,len(folder_NSs)+1):
        signal=np.loadtxt(path_NSs + 'raw_' + format(cell) + '.txt', delimiter=', ')
        avg,tau_up,tau_down,S=avg_period(signal,fps,freq,figname=path_Analysis+'avg_nss_'+format(cell)+'.png') 
        N_up=np.append(N_up,tau_up)
        N_down=np.append(N_down,tau_down)
        N_S=np.append(N_S,S)

    np.savetxt(path_Analysis + 'Tau_UP_NSs.txt', N_up, delimiter=', ', fmt='%.4f')
    np.savetxt(path_Analysis + 'Tau_DOWN_NSs.txt', N_down, delimiter=', ', fmt='%.4f')
    np.savetxt(path_Analysis + 'Sensitivity_NSs.txt', N_S, delimiter=', ', fmt='%.3f')

if __name__=='__main__':
    extract_dFF_speed()
    
    # C=36
    # N=61
    # dC=32/np.sqrt(73)
    # dN=33/np.sqrt(180)
    # FE=(N-C)/C*100
    # dFE=np.sqrt((dN/C)**2+(N*dC/(C**2))**2)*100
    # print('Fluorescence enhancement: ('+format(FE)+' +/- '+format(dFE)+') %')