import os
import sys
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../Python modules/") # necessary for the tex fonts
from numpy import genfromtxt 
import matplotlib.pyplot as plt
import numpy as np
from MakePdf import *

### read all data

### Region IIb

IIbgranared = genfromtxt('IIgranared.xls', skip_header = 1) #, delimiter=' ')
IIbgranablue = genfromtxt('IIgranablue.xls', skip_header = 1)
IIbbgred = genfromtxt('IIbgred.xls', skip_header = 1) #, delimiter=' ')
IIbbgblue = genfromtxt('IIbgblue.xls', skip_header = 1)

### Region III

IIIgranared = genfromtxt('IIIgranared.xls', skip_header = 1) #, delimiter=' ')
IIIgranablue = genfromtxt('IIIgranablue.xls', skip_header = 1)
IIIbgred = genfromtxt('IIIbgred.xls', skip_header = 1) #, delimiter=' ')
IIIbgblue = genfromtxt('IIIbgblue.xls', skip_header = 1)

### Region V

Vgranared = genfromtxt('Vgranared.xls', skip_header = 1) #, delimiter=' ')
Vgranablue = genfromtxt('Vgranablue.xls', skip_header = 1)
Vbgred = genfromtxt('Vbgred.xls', skip_header = 1) #, delimiter=' ')
Vbgblue = genfromtxt('Vbgblue.xls', skip_header = 1)

fig4 = plt.figure(figsize=(8, 6), dpi=80)
ax1 = plt.subplot2grid((1,3), (0, 0), colspan=1)
ax1.set_title('Region II: \n no correlation between channels', fontsize = 22)
plt.plot(np.arange(1,51)*50, IIbgranared[:,1]/np.max(IIbgranared),color = 'g',lw=5,label='location of grana, grana channel')
plt.plot(np.arange(1,51)*50,IIbgranablue[:,1]/np.max(IIbgranablue),color = 'b',lw=5,label='location of grana, scintillator channel')
plt.plot(np.arange(1,51)*50,IIbbgred[:,1]/np.max(IIbbgred),color = 'g',lw=2,label='background, grana channel')
plt.plot(np.arange(1,51)*50,IIbbgblue[:,1]/np.max(IIbbgblue),color = 'b',lw=2,label='background, scintillator channel')
ax1.set_ylim([0.85,1.01])
ax1.set_xlabel('Cumulative e-beam exposure, per pixel ($\mu$s)', fontsize = 18)
ax1.set_ylabel('Intensity, normalized', fontsize = 18)
plt.legend(loc='best')

ax1 = plt.subplot2grid((1,3), (0, 1), colspan=1)
ax1.set_title('Region III: \n anti-correlation between channels', fontsize = 22)
plt.plot(np.arange(1,51)*50, IIIgranared[:,1]/np.max(IIIgranared),color = 'g',lw=5,label='location of grana, grana channel')
plt.plot(np.arange(1,51)*50,IIIgranablue[:,1]/np.max(IIIgranablue),color = 'b',lw=5,label='location of grana, scintillator channel')
plt.plot(np.arange(1,51)*50,IIIbgred[:,1]/np.max(IIIbgred),color = 'g',lw=2,label='background, grana channel')
plt.plot(np.arange(1,51)*50,IIIbgblue[:,1]/np.max(IIIbgblue),color = 'b',lw=2,label='background, scintillator channel')
ax1.set_ylim([0.85,1.01])
ax1.set_xlabel('Cumulative e-beam exposure, per pixel ($\mu$s)', fontsize = 18)
ax1.set_ylabel('Intensity, normalized', fontsize = 18)
#plt.legend(loc='best')

ax1 = plt.subplot2grid((1,3), (0, 2), colspan=1)
ax1.set_title('Region V: \n positive correlation between channels', fontsize = 22)
plt.plot(np.arange(1,51)*50, Vgranared[0:50]/np.max(Vgranared),color = 'g',lw=5,label='location of grana, grana channel')
plt.plot(np.arange(1,51)*50,Vgranablue[0:50]/np.max(Vgranablue),color = 'b',lw=5,label='location of grana, scintillator channel')
plt.plot(np.arange(1,51)*50,Vbgred[0:50]/np.max(Vbgred),color = 'g',lw=2,label='background, grana channel')
plt.plot(np.arange(1,51)*50,Vbgblue[0:50]/np.max(Vbgblue),color = 'b',lw=2,label='background, scintillator channel')
ax1.set_ylim([0.85,1.01])
ax1.set_xlabel('Cumulative e-beam exposure, per pixel ($\mu$s)', fontsize = 18)
ax1.set_ylabel('Intensity, normalized', fontsize = 18)
#plt.legend(loc='best')

multipage_longer('Bleaching.pdf',dpi=80)