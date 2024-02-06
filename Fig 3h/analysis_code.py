# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:52:31 2024

@author: mlocarno
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as ticker

folder=""
file_name="SIM_lambda650nm_XY"
file_path=folder+file_name+".txt"
pixels=561

plt.rcParams.update({'font.size': 18})

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

rc('font', **font)

data=np.loadtxt(file_path,dtype=str,delimiter="\n")

x=np.array(data[2:pixels+2],dtype=float)
y=np.array(data[pixels+3:2*pixels+3],dtype=float)

E=np.zeros((0,pixels))
temp=np.array(data[2*pixels+4:][0:])
for line in temp:
    add_line=np.fromstring(line,dtype=float,sep=" ")
    E=np.append(E,np.expand_dims(add_line,0),axis=0)

E=E.transpose()

# Create a mask to select the pixels within the annular region
x_int = np.tile((x/(2.5e-10)).astype(int),(pixels,1))
y_int = np.transpose(np.tile((y/(2.5e-10)).astype(int),(pixels,1)))
x_center = np.where(np.isclose(x_int,0,atol=1e-5))[0][0]
y_center = np.where(np.isclose(y_int,-280,atol=1e-5))[0][0]

# Create a 2D map for the radius
r_map = np.sqrt((x_int - x_center)**2 + (y_int - y_center)**2)

r_array=np.arange(0,31,0.5)
avg_array=np.array([])
median_array=np.array([])
std_array=np.array([])
sem_array=np.array([])
q1_array=np.array([])
q3_array=np.array([])

for dr in r_array:
    # Define the radii of the inner and outer circles
    r1 = int((35+dr)*1e-9/(2.5e-10))  # Inner radius in pixels
    r2 = int((35+dr+0.5)*1e-9/(2.5e-10))  # Outer radius in pixels
    
    # Create a mask to select the pixels within the annular region
    mask = np.logical_and(r_map >= r1, r_map <= r2)
    
    # Use the mask to extract the values within the annular region
    values_in_annulus = E[mask]
    MEAN=np.mean(values_in_annulus)
    STD=np.std(values_in_annulus)
    MEDIAN=np.median(values_in_annulus)
    q1 = np.percentile(values_in_annulus, 25)
    q3 = np.percentile(values_in_annulus, 75)
    
    # Calculate the average of the selected values
    avg_array=np.append(avg_array,MEAN)
    median_array=np.append(median_array,MEDIAN)
    std_array=np.append(std_array,STD)
    sem_array=np.append(sem_array,STD/np.sqrt(len(values_in_annulus)))
    q1_array=np.append(q1_array,q1)
    q3_array=np.append(q3_array,q3)

gamma=0.129
gamma_nr=0.3
Y=(avg_array*gamma)/(avg_array*gamma+gamma_nr)
median_Y=(median_array*gamma)/(median_array*gamma+gamma_nr)
std_Y=(gamma*gamma_nr)/((avg_array*gamma+gamma_nr)**2)*std_array
dq1_Y=(gamma*gamma_nr)/((median_array*gamma+gamma_nr)**2)*(median_array-q1_array)
dq3_Y=(gamma*gamma_nr)/((median_array*gamma+gamma_nr)**2)*(q3_array-median_array)

dY_Y=(Y-0.3)/0.3*100
median_dY_Y=(median_Y-0.3)/0.3*100
dq1_dY_Y=dq1_Y/0.3*100
dq3_dY_Y=dq3_Y/0.3*100
std_dY_Y=std_Y/0.3*100

plt.axhline(1, color=[0.35,0.7,0.9], linestyle='--',linewidth=2,label='Cy5')
plt.plot(r_array,avg_array, color=[0.9,0.6,0],linewidth=2,label='Cy5+NSs')
plt.fill_between(r_array, avg_array - std_array, avg_array + std_array, color=[0.9, 0.6, 0, 0.3])
plt.axvline(13, color='black', linestyle=':',linewidth=2,label='BSA+sAv')
plt.axvspan(0, 5, facecolor='gray', alpha=0.3)
plt.axvspan(20, 30, facecolor='gray', alpha=0.3)
plt.xlabel('d [nm]')
plt.ylabel('F.E. [a.u.]')
plt.legend(loc='best', framealpha=1.0)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))  # Adjust the interval as needed
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(40))  # Adjust the interval as needed
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(5))
plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(20))
plt.gca().set_xlim([0, 30])  # Replace min_value_x and max_value_x with your desired values
plt.gca().set_ylim([-40, 120])  # Replace min_value_y and max_value_y with your desired values
fig = plt.gcf()
fig.set_size_inches(12/2.54, 12/2.54) #12cmx12cm
plt.tight_layout(pad=0.5)
plt.show()

plt.axhline(0.3, color=[0.35,0.7,0.9], linestyle='--',linewidth=2,label='Cy5')
plt.plot(r_array,Y,color=[0.9,0.6,0],linewidth=2,label='Cy5+NSs')
plt.fill_between(r_array, Y - std_Y, Y + std_Y, color=[0.9, 0.6, 0, 0.3])
plt.axvline(13, color='black', linestyle=':',linewidth=2,label='BSA+sAv')
plt.axvspan(0, 5, facecolor='gray', alpha=0.3)
plt.axvspan(20, 30, facecolor='gray', alpha=0.3)
plt.xlabel('d [nm]')
plt.ylabel('Y [a.u.]')
plt.legend(loc='best', framealpha=1.0)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))  # Adjust the interval as needed
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.3))  # Adjust the interval as needed
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(5))
plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.15))
plt.gca().set_xlim([0, 30])  # Replace min_value_x and max_value_x with your desired values
plt.gca().set_ylim([0, 1.2])  # Replace min_value_y and max_value_y with your desired values
fig = plt.gcf()
fig.set_size_inches(12/2.54, 12/2.54) #12cmx12cm
plt.tight_layout(pad=0.5)
plt.show()

plt.axhline(0, color=[0.35,0.7,0.9], linestyle='--',linewidth=2,label='Cy5')
plt.plot(r_array,dY_Y,color=[0.9,0.6,0],linewidth=2,label='Cy5+NSs')
plt.fill_between(r_array, dY_Y - std_dY_Y, dY_Y + std_dY_Y, color=[0.9, 0.6, 0, 0.3])
plt.axvline(13, color='black', linestyle=':',linewidth=2,label='BSA+sAv')
plt.axvspan(0, 5, facecolor='gray', alpha=0.3)
plt.axvspan(20, 30, facecolor='gray', alpha=0.3)
plt.errorbar(13,42,yerr=6,fmt='^', capsize=5, capthick=2, elinewidth=2, linewidth=2, color='black')
plt.errorbar(8.5,90,yerr=50,fmt='o', capsize=5, capthick=2, elinewidth=2, linewidth=2, color='gray')
plt.errorbar(14.5,30,yerr=50,fmt='o', capsize=5, capthick=2, elinewidth=2, linewidth=2, color='gray')
plt.xlabel('d [nm]')
plt.ylabel('dY/Y [%]')
plt.legend(loc='best', framealpha=1.0)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))  # Adjust the interval as needed
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(50))  # Adjust the interval as needed
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(5))
plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(25))
plt.gca().set_xlim([0, 30])  # Replace min_value_x and max_value_x with your desired values
plt.gca().set_ylim([-75, 250])  # Replace min_value_y and max_value_y with your desired values
fig = plt.gcf()
fig.set_size_inches(12/2.54, 12/2.54) #12cmx12cm
plt.tight_layout(pad=0.5)
# plt.savefig(folder+'fig3_h.png',dpi=300)
plt.show()

