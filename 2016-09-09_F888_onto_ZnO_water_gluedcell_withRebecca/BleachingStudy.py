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

Tribeadred = genfromtxt('Tribeadred.xls', skip_header = 1) #, delimiter=' ')
Tribeadblue = genfromtxt('Tribeadblue.xls', skip_header = 1)
Tribgred = genfromtxt('Tribgred.xls', skip_header = 1) #, delimiter=' ')
Tribgblue = genfromtxt('Tribgblue.xls', skip_header = 1)

fig4 = plt.figure(figsize=(8, 6), dpi=80)
ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
ax1.set_title('Region IV: positive correlation between channels')
plt.plot(np.arange(1,21)*100,Tribeadred[:,1]/np.max(Tribeadred),color = 'r',lw=5,label='location of bead, bead channel')
plt.plot(np.arange(1,21)*100,Tribeadblue[:,1]/np.max(Tribeadblue),color = 'b',lw=5,label='location of bead, scintillator channel')
plt.plot(np.arange(1,21)*100,Tribgred[:,1]/np.max(Tribgred),color = 'r',lw=2,label='background, bead channel')
plt.plot(np.arange(1,21)*100,Tribgblue[:,1]/np.max(Tribgblue),color = 'b',lw=2,label='background, scintillator channel')
#ax1.set_ylim([0.68,1.005])
ax1.set_xlabel('Cumulative e-beam exposure, per pixel ($\mu$s)')
ax1.set_ylabel('Intensity, normalized')
plt.legend(loc='best')
ax1.set_xlim([100,2000])
#ax1.set_yticks([0.7,0.8,0.9, 1.0])

#ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
#ax1.set_title('Region VI: anti-correlation between channels')
#plt.plot(np.arange(1,21)*5000,VIndred/np.max(VIndred),color = 'r',lw=5,label='location of ND, grana channel')
#plt.plot(np.arange(1,21)*5000,VIndblue/np.max(VIndblue),color = 'b',lw=5,label='location of ND, scintillator channel')
#plt.plot(np.arange(1,21)*5000,VIbgred/np.max(VIbgred),color = 'r',lw=2,label='background, ND channel')
#plt.plot(np.arange(1,21)*5000,VIbgblue/np.max(VIbgblue),color = 'b',lw=2,label='background, scintillator channel')
#ax1.set_ylim([0.68,1.005])
#ax1.set_xlabel('Cumulative e-beam exposure, per pixel ($\mu$s)')
#ax1.set_ylabel('Intensity, normalized')
##plt.legend(loc='best')
#ax1.set_xlim([5000,50000])
#ax1.set_yticks([0.7,0.8, 0.9, 1.0])

multipage_longer('Bleaching.pdf',dpi=80)