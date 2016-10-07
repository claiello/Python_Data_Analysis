import os
import sys
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../Python modules/") # necessary for the tex fonts
from numpy import genfromtxt 
import matplotlib.pyplot as plt
import numpy as np
from MakePdf import *

### read all data

### Region V

Vndred = genfromtxt('Vndred.xls', skip_header = 1) #, delimiter=' ')
Vndblue = genfromtxt('Vndblue.xls', skip_header = 1)
Vbgred = genfromtxt('Vbgred.xls', skip_header = 1) #, delimiter=' ')
Vbgblue = genfromtxt('Vbgblue.xls', skip_header = 1)

### Region VI

VIndred = genfromtxt('VIndred.xls', skip_header = 1) #, delimiter=' ')
VIndblue = genfromtxt('VIndblue.xls', skip_header = 1)
VIbgred = genfromtxt('VIbgred.xls', skip_header = 1) #, delimiter=' ')
VIbgblue = genfromtxt('VIbgblue.xls', skip_header = 1)


fig4 = plt.figure(figsize=(8, 6), dpi=80)
ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
ax1.set_title('Scan V: anti-correlation between channels, 2.5kV', fontsize = 20)
plt.plot(np.arange(1,21)*5000,Vndred[:,1]/np.max(Vndred),color = 'r',lw=5,label='location of ND, ND channel')
plt.plot(np.arange(1,21)*5000,Vndblue[:,1]/np.max(Vndblue),color = 'b',lw=5,label='location of ND, scintillator channel')
plt.plot(np.arange(1,21)*5000,Vbgred[:,1]/np.max(Vbgred),color = 'r',lw=2,label='background, ND channel')
plt.plot(np.arange(1,21)*5000,Vbgblue[:,1]/np.max(Vbgblue),color = 'b',lw=2,label='background, scintillator channel')
ax1.set_ylim([0.68,1.005])
ax1.set_xlabel('Cumulative e-beam exposure, per pixel ($\mu$s)', fontsize = 18)
ax1.set_ylabel('Intensity, normalized', fontsize = 18)
plt.legend(loc='best')
ax1.set_xlim([5000,50000])
ax1.set_yticks([0.7,0.8,0.9, 1.0])

ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
ax1.set_title('Scan VI: anti-correlation between channels, 2kV', fontsize = 20)
plt.plot(np.arange(1,21)*5000,VIndred[:,1]/np.max(VIndred),color = 'r',lw=5,label='location of ND, grana channel')
plt.plot(np.arange(1,21)*5000,VIndblue[:,1]/np.max(VIndblue),color = 'b',lw=5,label='location of ND, scintillator channel')
plt.plot(np.arange(1,21)*5000,VIbgred[:,1]/np.max(VIbgred),color = 'r',lw=2,label='background, ND channel')
plt.plot(np.arange(1,21)*5000,VIbgblue[:,1]/np.max(VIbgblue),color = 'b',lw=2,label='background, scintillator channel')
ax1.set_ylim([0.68,1.005])
ax1.set_xlabel('Cumulative e-beam exposure, per pixel ($\mu$s)', fontsize = 18)
ax1.set_ylabel('Intensity, normalized', fontsize = 18)
#plt.legend(loc='best')
ax1.set_xlim([5000,50000])
ax1.set_yticks([0.7,0.8, 0.9, 1.0])

multipage_longer('Bleaching.pdf',dpi=80)