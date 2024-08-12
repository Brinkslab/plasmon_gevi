# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:09:11 2021

@author: LocalAdmin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import rc
from cmcrameri import cm
import matplotlib.ticker as ticker

folder="M:/tnw/ist/do/projects/Neurophotonics/Brinkslab/People/Marco Locarno/PhD/Archive previous years/2022/FDTD nanoparticles/results/Field Enhancement nanostar/Results - Field enhancement maps/"
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

# Define the radii of the inner and outer circles
r1 = int((35+13)*1e-9/(2.5e-10))  # Inner radius in pixels
r2 = int((35+13+0.5)*1e-9/(2.5e-10))  # Outer radius in pixels

# Create a 2D map for the radius
r_map = np.sqrt((x_int - x_center)**2 + (y_int - y_center)**2)

# Create a mask to select the pixels within the annular region
mask = np.logical_and(r_map >= r1, r_map <= r2)

# Use the mask to extract the values within the annular region
values_in_annulus = E[mask]

print("Maximum value: "+format(np.max(values_in_annulus)))

E[mask]=0 #only to visualize the ring in white

# Calculate the average of the selected values
average_value_in_annulus = np.mean(values_in_annulus)

print(f"Average value in the annular region: {average_value_in_annulus}")

plt.figure()
plt.imshow(E, interpolation='antialiased',cmap=cm.batlow,extent=[x[0]*1e9,x[-1]*1e9,y[0]*1e9,y[-1]*1e9],norm=colors.LogNorm())
cbar=plt.colorbar()
cbar.set_label(label='Field enhancement [ |E|\u00b2 / |E\u2070|\u00b2 ]',rotation=270, labelpad=20)
plt.axis('off')
plt.savefig(folder+file_name+'map2.png',dpi=300)
plt.show()
