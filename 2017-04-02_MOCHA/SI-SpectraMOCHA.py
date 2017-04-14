import os
import sys
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../Python modules/") # necessary for the tex fonts
import scipy as sp
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
import h5py
import numpy as np
import matplotlib.cm as cm
import scipy.ndimage as ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from MakePdf import *
from matplotlib.pyplot import cm #to plot following colors of rainbow
from matplotlib import rc
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = DeprecationWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.simplefilter(action = "ignore", category = PendingDeprecationWarning)
from tifffile import *
from sklearn.mixture import GMM 
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.misc
import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile
from uncertainties import unumpy

import skimage
from skimage import exposure
from my_fits import *

from numpy import genfromtxt
import matplotlib.cm as cm
from my_fits import *

from subtract_background import subtract_background
from matplotlib import colors as mcolors

### aux functions
def moving_average(a,n=3):
    vec = np.cumsum(a)
    vec[n:] = vec[n:] - vec[:-n]
    return (1/n)*vec[n-1:]

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx #array[idx]
    
def fft_cut_noise(spec,wl):
    total_wl = wl[-1]
    t = wl
    x = np.linspace(0, total_wl, len(t))
    fft_y = np.fft.fft(t)
    n = t.size
    timestep = wl[1] - wl[0]
    freq = np.fft.fftfreq(n, d = timestep)
    ind = np.abs(freq) > 0.1
    fft_y_cut = np.copy(fft_y)
    fft_y_cut[ind] = 0.0
    new_y = np.abs(np.fft.ifft(fft_y_cut))
    
    return new_y
    
### settings
fsizepl = 24
fsizenb = 20
###
sizex=8
sizey=6
dpi_no=80

fig1= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig1.set_size_inches(1200./fig1.dpi,900./fig1.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    

noplots = 2
nolines = 2

ax0 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.xaxis.set_ticks_position('bottom')
ax0.yaxis.set_ticks_position('left')

ax1 = plt.subplot2grid((nolines,noplots), (1,0), colspan=1, rowspan=1)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax2 = plt.subplot2grid((nolines,noplots), (1,1), colspan=1, rowspan=1)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
 
specBG1min = genfromtxt('BGMith1min.txt', delimiter='')
specBG15s = genfromtxt('BGMith15s.txt', delimiter='')
specBG30s = genfromtxt('BGMith30s.txt', delimiter='')

specMith1min = genfromtxt('Mith1min.txt', delimiter='')
specMith15s = genfromtxt('Mith15s.txt', delimiter='')
specMith30sss1 = genfromtxt('Mith30sScanSpeed1.txt', delimiter='')
specMith30sss5 = genfromtxt('Mith30sScanSpeed5.txt', delimiter='')

###get background, taken at RT
specbg1min = np.array(specBG1min[:,1])
specbg15s = np.array(specBG15s[:,1])
specbg30s = np.array(specBG30s[:,1])

vector =  [specMith1min, specMith15s, specMith30sss1, specMith30sss5]
specbg =  [specbg1min,   specbg15s,   specbg30s,      specbg30s]

indexmov = 5
colors = iter(cm.rainbow(np.linspace(0, 1, len(vector)))) 

indice = 0

labelu = ['1 min', '15 s', '30 s scan speed 1', '30 s scan speed 5']

time = [60, 15, 30, 30] #integration time in seconds

indice = 0
for spec in vector:
    # x, y vectors
    wavel = spec[:,0] 
    spec_to_plot = spec[:,1] - specbg[indice]
    
    # cut vector only between 300 and 720nm
    aa = find_nearest(wavel,300)
    bb = find_nearest(wavel,720)
    wavel = wavel[aa:bb]
    spec_to_plot = spec_to_plot[aa:bb]
    
    ### take out constant background
    #spec_to_plot = spec_to_plot - np.average(spec_to_plot)
    # put min to zero
    spec_to_plot = spec_to_plot + np.abs(np.min(spec_to_plot))
      
    mov_avg_index = indexmov
    spec_to_plot = moving_average(spec_to_plot,n=mov_avg_index)
    wavel = moving_average(wavel,n=mov_avg_index)
    
    colorful = next(colors)
    
    ax0.plot(wavel,spec_to_plot/time[indice], lw=2, color=colorful,label=labelu[indice])
    
    a = find_nearest(wavel,380)
    b = find_nearest(wavel,420)
    
    c = find_nearest(wavel,457)
    d = find_nearest(wavel,487)
    
    f = find_nearest(wavel,483)
    g = find_nearest(wavel,523)
    
    spec_to_plot_filter1 = spec_to_plot[a:b]
    spec_to_plot_filter5 = spec_to_plot[c:d]
    spec_to_plot_filter6 = spec_to_plot[f:g]
    
    print(indice)
    clnormno = [1.0, np.sum(spec_to_plot_filter5)/np.sum(spec_to_plot), np.sum(spec_to_plot_filter6)/np.sum(spec_to_plot), np.sum(spec_to_plot_filter1)/np.sum(spec_to_plot)]
    clnormrightfilter = [1.0, np.sum(spec_to_plot_filter6)/np.sum(spec_to_plot_filter5), np.sum(spec_to_plot_filter1)/np.sum(spec_to_plot_filter5)]

    ax1.plot(np.arange(1,5),clnormno, linestyle='None', marker = 'o', color=colorful,label=labelu[indice])
    ax2.plot(np.arange(1,4),clnormrightfilter,linestyle='None', marker = 'o', color=colorful,label=labelu[indice])
    
    indice = indice + 1
    
ax0.legend(loc = 'best',frameon=False, fontsize=fsizenb)
handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles[::-1], labels[::-1],loc = 'best',frameon=False, fontsize=fsizenb)
ax0.set_ylabel('Bulk cathodoluminescence \n emission spectrum (Hz)',fontsize=fsizepl)
ax0.set_xlabel('Wavelength (nm)',fontsize=fsizepl)
ax0.tick_params(labelsize=fsizenb)
ax0.set_xticks([200,300,400,500,600,700,800,900])
ax0.set_xlim([300,720])

ax1.set_ylabel('Avg CL, norm. (a.u.)',fontsize=fsizepl)
ax1.set_xlabel('Filters',fontsize=fsizepl)
ax1.set_xticks([1,2,3,4])
ax1.set_xlim([0.5, 4.5])
ax1.set_xticklabels(['None', '472/30nm', '503/40nm', '400/40nm'])
ax2.set_ylabel('Avg CL, norm. (a.u.)',fontsize=fsizepl)
ax2.set_xlabel('Filters',fontsize=fsizepl)
ax2.set_xticks([1,2,3])
ax2.set_xticklabels(['472/30nm', '503/40nm', '400/40nm'])
ax2.set_xlim([0.5, 3.5])
ax1.tick_params(labelsize=fsizenb)
ax2.tick_params(labelsize=fsizenb)
ax2.set_ylim([0, 1.05])
ax1.set_ylim([0, 1.05])

for label in ax1.get_xmajorticklabels() + ax2.get_xmajorticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")

plt.tight_layout()
multipage_longer_desired_aspect_ratio('SI-spectraApril.pdf',1600,1200,dpi=80)



lklklk
