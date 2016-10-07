import os
import sys
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../Python modules/") # necessary for the tex fonts
from numpy import genfromtxt 
import matplotlib.pyplot as plt
import numpy as np
from MakePdf import *

### read all data

### Region III TR
####This is the time-resolved data, 40ns time bin, 6mus per pixel
IIIallframered = genfromtxt('IIIallframered.xls', skip_header = 1) #, delimiter=' ')
IIIallframeblue = genfromtxt('IIIallframeblue.xls', skip_header = 1)

### Region III TR
####This is the time-resolved data, 40ns time bin, 6mus per pixel
TRIIIbeadred = genfromtxt('TRIIIbeadred.xls', skip_header = 1) #, delimiter=' ')
TRIIIbeadblue = genfromtxt('TRIIIbeadblue.xls', skip_header = 1)
TRIIIbgred = genfromtxt('TRIIIbgred.xls', skip_header = 1) #, delimiter=' ')
TRIIIbgblue = genfromtxt('TRIIIbgblue.xls', skip_header = 1)

### Region III 
####This is the time-resolved data, 40ns time bin, 6mus per pixel
IIIbeadred = genfromtxt('IIIbeadred.xls', skip_header = 1) #, delimiter=' ')
IIIbeadblue = genfromtxt('IIIbeadblue.xls', skip_header = 1)
IIIbgred = genfromtxt('IIIbgred.xls', skip_header = 1) #, delimiter=' ')
IIIbgblue = genfromtxt('IIIbgblue.xls', skip_header = 1)

fig4 = plt.figure(figsize=(8, 6), dpi=80)
ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
ax1.set_title('Region III TR: positive correlation between channels')
init = 80
plt.plot(np.arange(1,151-init)*0.04,TRIIIbeadred[init:,1]/np.max(TRIIIbeadred),color = 'r',lw=5,label='location of bead, bead channel')
plt.plot(np.arange(1,151-init)*0.04,TRIIIbeadblue[init:,1]/np.max(TRIIIbeadblue),color = 'b',lw=5,label='location of bead, scintillator channel')
plt.plot(np.arange(1,151-init)*0.04,TRIIIbgred[init:,1]/np.max(TRIIIbgred),color = 'r',lw=2,label='background, bead channel')
plt.plot(np.arange(1,151-init)*0.04,TRIIIbgblue[init:,1]/np.max(TRIIIbgblue),color = 'b',lw=2,label='background, scintillator channel')
#ax1.set_ylim([0.68,1.005])
ax1.set_xlabel('Time ($\mu$s)')
ax1.set_ylabel('Intensity, normalized')
#plt.legend(loc='best')
ax1.set_xlim([0.0,2.75])
ax1.set_yticks([0.0,0.5,1.0])

#ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
#ax1.set_title('Region III TR: positive correlation between channels')
#plt.plot(np.arange(1,151)*0.04,IIIallframered/np.max(IIIallframered),color = 'r',lw=5,label='location of ND, ND channel')
#plt.plot(np.arange(1,151)*0.04,IIIallframeblue/np.max(IIIallframeblue),color = 'b',lw=5,label='location of ND, scintillator channel')
##plt.plot(np.arange(1,6)*100,Vbgred/np.max(Vbgred),color = 'r',lw=2,label='background, ND channel')
##plt.plot(np.arange(1,6)*100,Vbgblue/np.max(Vbgblue),color = 'b',lw=2,label='background, scintillator channel')
##ax1.set_ylim([0.68,1.005])
#ax1.set_xlabel('Time ($\mu$s)')
#ax1.set_ylabel('Intensity, normalized')
#plt.legend(loc='best')
##ax1.set_xlim([5000,50000])
##ax1.set_yticks([0.7,0.8,0.9, 1.0])

ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
ax1.set_title('Region III: positive correlation between channels')
plt.plot(np.arange(1,6)*100,IIIbeadred/np.max(IIIbeadred),color = 'r',lw=5,label='location of bead, bead channel')
plt.plot(np.arange(1,6)*100,IIIbeadblue/np.max(IIIbeadblue),color = 'b',lw=5,label='location of bead, scintillator channel')
plt.plot(np.arange(1,6)*100,IIIbgred/np.max(IIIbgred),color = 'r',lw=2,label='background, bead channel')
plt.plot(np.arange(1,6)*100,IIIbgblue/np.max(IIIbgblue),color = 'b',lw=2,label='background, scintillator channel')
ax1.set_ylim([0.9,1.005])
ax1.set_xlabel('Cumulative e-beam exposure, per pixel ($\mu$s)')
ax1.set_ylabel('Intensity, normalized')
plt.legend(loc='best')
#ax1.set_xlim([5000,50000])
ax1.set_yticks([0.9, 0.95,1.0])

multipage_longer('Bleaching.pdf',dpi=80)