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
#from BackgroundCorrection import *
#from TConversionThermocoupler import *
import matplotlib.cm as cm
import scipy.ndimage as ndimage
#from matplotlib_scalebar.scalebar import ScaleBar #### Has issue with plotting using latex font. only import when needed, then unimport
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from MakePdf import *
from matplotlib.pyplot import cm #to plot following colors of rainbow
from matplotlib import rc
#from CreateDatasets import *
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = DeprecationWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.simplefilter(action = "ignore", category = PendingDeprecationWarning)
#from Registration import * 
from tifffile import *
from sklearn.mixture import GMM 
import matplotlib.cm as cm
#from FluoDecay import *
#from PlottingFcts import *
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
    
def tauestimate(counts_red, error_red, counts_blue, error_blue):
    
        print(counts_red.shape[1])
        
        ucounts_red = unumpy.uarray(counts_red, error_red)
        ucounts_blue = unumpy.uarray(counts_blue, error_blue)
        
        def helper(arrayx):
             return np.cumsum(arrayx*np.arange(1,counts_red.shape[1]+1), axis = 1)/np.cumsum(arrayx, axis = 1)
        
        return helper(ucounts_red),helper(ucounts_blue)
    
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

noplots = 8
nolines = 8


################################################################################
index=0
Time_bin = 1000#in ns; 
let = ['ns40']
sys.path.append("../2017-04-05_Andrea_NPs_Different_excitation_times/") # necessary 
#se = np.load('../2017-04-05_Andrea_NPs_single/'+ str(let[index]) +'SEchannel.npz') 
#red = np.load('../2017-04-05_Andrea_NPs_single/'+ str(let[index]) +'Redbright.npz') 
#blue = np.load('../2017-04-05_Andrea_NPs_single/' + str(let[index]) +'Bluebright.npz') 

bluedecay = np.load('../2017-04-05_Andrea_NPs_Different_excitation_times/'+'Blue_decay_arrayPixel.npz',mmap_mode='r')     
reddecay = np.load('../2017-04-05_Andrea_NPs_Different_excitation_times/'+'Red_decay_arrayPixel.npz',mmap_mode='r')
bgbluedecay = np.load('../2017-04-05_Andrea_NPs_Different_excitation_times/'+'bgBlue_decay_arrayPixel.npz',mmap_mode='r')     
bgreddecay = np.load('../2017-04-05_Andrea_NPs_Different_excitation_times/'+'bgRed_decay_arrayPixel.npz',mmap_mode='r') 

######### DECAY PLOT 
ax3 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
ax3.semilogy(np.arange(0,498),reddecay['data'].reshape([498])/np.max(reddecay['data'].reshape([498])),'o',color='r',markersize=6, markeredgewidth=0.0,label='Red band signal')  
ax3.semilogy(np.arange(0,498),bluedecay['data'].reshape([498])/np.max(bluedecay['data'].reshape([498])),'o',color='g',markersize=6, markeredgewidth=0.0,label='Green band signal')  
ax3.semilogy(np.arange(0,498),bgreddecay['data'].reshape([498])/np.max(bgreddecay['data'].reshape([498])),'o',color='DarkRed',markersize=4,markeredgewidth=0.0,label='Red band background')  
ax3.semilogy(np.arange(0,498),bgbluedecay['data'].reshape([498])/np.max(bgbluedecay['data'].reshape([498])),'o',color='DarkGreen',markersize=4, markeredgewidth=0.0,label='Green band background')  
ax3.legend(loc='best',fontsize=fsizenb,frameon=False)
ax3.set_ylabel("Norm. cathodoluminescence (a.u.)",fontsize=fsizepl)
ax3.set_xlabel('Transient cathodoluminescence \n acquisition time ($\mu$s)',  fontsize=fsizepl)
ax3.text(-1.1, 1.0, 'a', transform=ax3.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})
#ax3.set_xticks([250, 500,750]) 
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')
#ax3.set_ylim([0.0095,1.05])
#ax3.set_yticks([0.01,0.1,1])
#ax3.set_yticklabels(['0.01','0.1','1'])
ax3.tick_params(labelsize=fsizenb)
ax3.set_xlim([0,498])

###### TAU PLOT

ax112 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
ax112.text(1.1, 1.0, 'b', transform=ax112.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})
ax112.spines['left'].set_visible(False)
ax112.spines['top'].set_visible(False)
ax112.xaxis.set_ticks_position('bottom')
ax112.yaxis.set_ticks_position('right')
ax112.yaxis.set_label_position('right')
#ax112.set_ylim([0,510])
#ax112.set_yticks([100,200,300,400,500])

redtau, bluetau = tauestimate(reddecay['data'],np.std(reddecay['data'])/2000,bluedecay['data'],np.std(bluedecay['data'])/2000)
redtaubg, bluetaubg = tauestimate(bgreddecay['data'],np.std(bgreddecay['data'])/2000,bgbluedecay['data'],np.std(bgbluedecay['data'])/2000)

redtau = redtau.reshape([498])
bluetau= bluetau.reshape([498])
redtaubg = redtaubg.reshape([498])
bluetaubg =bluetaubg.reshape([498])

#this shows that first bin is poisson
#vecforhist = (datared[initbin,:,:]*hlp)*2.0e-6*1000.0 ####IN COUNTS, IE, COUNT RATE * DELTA T; *1000 IS FROM KHZ TO HZ #was initbin:
#vecforhist = vecforhist.flatten()
#plt.figure()
#xxx, xx = plt.hist(vecforhist[~np.isnan(vecforhist)],5) #, 50, normed=1, facecolor='green', alpha=0.75)
#plt.show()
#oioioi

ax112.plot(np.arange(0,498),unumpy.nominal_values(redtau),color='r',ls='--',lw=2)  
ax112.plot(np.arange(0,498),unumpy.nominal_values(bluetau),color='g',ls='--',lw=2)  
ax112.set_ylabel(r"$\tau$ ($\mu$s)",fontsize=fsizepl)
ax112.set_xlabel('Transient cathodoluminescence \n acquisition time ($\mu$s)',  fontsize=fsizepl)
#ax112.set_xlim(xmax=1000.0) #2000
#Plot whole of background decay
ax112.plot(np.arange(0,498),unumpy.nominal_values(redtaubg),color='DarkRed',ls='--',lw=2)   
ax112.plot(np.arange(0,498),unumpy.nominal_values(bluetaubg),color='DarkGreen',ls='--',lw=2) 
ax112.plot(np.arange(0,498),np.arange(0,498)/2.0,color='k',ls='--',lw=2)  

print(unumpy.std_devs(redtau))

#my_edgecolor='#ff3232'
#my_facecolor='#ff6666'
#ax112.fill_between(np.arange(0,bluedecay['data'].shape[0]),unumpy.nominal_values(redtau)-unumpy.std_devs(redtau),unumpy.nominal_values(redtau)+unumpy.std_devs(redtau),
#                 edgecolor=my_edgecolor,
#                 facecolor=my_facecolor,
#                 alpha=0.5,
#                 linewidth=1.0)
#                 
#my_edgecolor='#74C365'
#my_facecolor='#74C365'
#ax112.fill_between(np.arange(0,bluedecay['data'].shape[0]),unumpy.nominal_values(bluetau)-unumpy.std_devs(bluetau),unumpy.nominal_values(bluetau)+unumpy.std_devs(bluetau),
#                 edgecolor=my_edgecolor,
#                 facecolor=my_facecolor,
#                 alpha=0.5,
#                 linewidth=1.0)
#                 
#my_edgecolor='#801515'
#my_facecolor='#801515'
#ax112.fill_between(np.arange(0,bluedecay['data'].shape[0]),unumpy.nominal_values(redtaubg)-unumpy.std_devs(redtaubg),unumpy.nominal_values(redtaubg)+unumpy.std_devs(redtaubg),
#                 edgecolor=my_edgecolor,
#                 facecolor=my_facecolor,
#                 alpha=0.5,
#                 linewidth=1.0)
#                 
#my_edgecolor='#003D1B'
#my_facecolor='#003D1B'
#ax112.fill_between(np.arange(0,bluedecay['data'].shape[0]),unumpy.nominal_values(bluetaubg)-unumpy.std_devs(bluetaubg),unumpy.nominal_values(bluetaubg)+unumpy.std_devs(bluetaubg),
#                 edgecolor=my_edgecolor,
#                 facecolor=my_facecolor,
#                 alpha=0.5,
#                 linewidth=1.0)



#ax112.set_xlim([6,1000])
#ax112.set_ylim([0.02,15])
ax112.tick_params(labelsize=fsizenb)
ax112.set_xticks([250, 400]) 
#ax112.set_yticks([0.1,1,10]) 
#ax112.set_yticklabels(['0.1','1','10'])

plt.tight_layout()


multipage_longer('Fig1OnlyDecays.pdf',dpi=900)


## works for errorbars but doesnt look nice
#hlp_red_200 = np.nanmean(datared*hlp,axis=(1,2))
#hlp_red_200_err = np.sqrt(hlp_red_200)
#ax112.fill_between(xx_array/1e-6,hlp_red_200 + hlp_red_200_err, hlp_red_200 - 0*hlp_red_200_err,alpha=0.5, edgecolor=my_edgecolor, facecolor= my_facecolor)   
#ax112.set_yscale('log')

