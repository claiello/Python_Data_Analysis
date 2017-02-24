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

plot_spec= True
if plot_spec:

    ax0 = plt.subplot2grid((nolines,noplots), (5,0), colspan=4, rowspan=3)
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.xaxis.set_ticks_position('bottom')
    ax0.yaxis.set_ticks_position('left')
         
    spec = genfromtxt('spectrumCSV.csv', delimiter=',')
    # x, y vectors
    wavel = spec[:,0]
    spec_to_plot = spec[:,1]
    # cut vector only between 300 and 720nm
    a = find_nearest(wavel,300)
    b = find_nearest(wavel,720)
    wavel = wavel[a:b]
    spec_to_plot = spec_to_plot[a:b]
    # moving avg to cut noise
    mov_avg_index = 5
    spec_to_plot = moving_average(spec_to_plot,n=mov_avg_index)
    wavel = moving_average(wavel,n=mov_avg_index) #wavel + mov_avg_index/2.0 * (wavel[1] - wavel[0])
    
    # take out cst background in interval 300/720
    spec_to_plot = spec_to_plot - np.average(spec_to_plot)
    # make all go to zero min
    min_val = np.amin(spec_to_plot)
    spec_to_plot = spec_to_plot + np.abs(min_val)
    #normalize taking into account 300/720 interval
    spec_to_plot = spec_to_plot/np.max(spec_to_plot)
    
    #find index of red/green transition
    e = find_nearest(wavel,593)
    
    # Find center of mass in interval 300/720, for two bands
    #green
    aux_y = spec_to_plot[:e]
    aux_x = wavel[:e]
    f = find_nearest(aux_y, np.max(aux_y))
    print(aux_x[f])
    ax0.axvline(x=aux_x[f] , lw=2, color='g', ls='--')
    
    #red
    aux_y = spec_to_plot[e:]
    aux_xx = wavel[e:]
    g = find_nearest(aux_y, np.max(aux_y))
    print(aux_xx[g])
    ax0.axvline(x=aux_xx[g] , lw=2, color='r', ls='--', ymax = 0.8)
    
    
    # plot
    ax0.plot(wavel[:e],spec_to_plot[:e], lw=2, color='g')
    ax0.plot(wavel[e:],spec_to_plot[e:], lw=2, color='r')
    # plot vertical line at 593nm (dichroic)
    ax0.axvline(x=593, lw=2, color='k', ls='--', ymax = 0.8)
    
    #labels
    xmin = 500
    xmax = 700
    ax0.set_xlim([xmin, xmax])
    ax0.set_ylabel('Bulk cathodoluminescence \n emission spectrum (a.u.)',fontsize=fsizepl)
    ax0.set_xlabel('Wavelength (nm)',fontsize=fsizepl)
    ax0.tick_params(labelsize=fsizenb)
    ax0.set_yticks([0.5,1.0])
    ax0.set_ylim([0,1.05])
    #ax0.set_xticks([500,544.76,593+ mov_avg_index/2.0 * (wavel[1] - wavel[0]),662.93,700])
    #ax0.set_xticklabels(['500','544.76','593','662.93','700'])
    ax0.set_xticks([aux_x[f],593,aux_xx[g]])
    ax0.set_xticklabels([str("{0:.1f}".format(aux_x[f])),'593.0',str("{0:.1f}".format(aux_xx[g]))])


    # adding dE with arrow
    ax0.annotate('',  xy=(541.2+1.5, 0.875), xytext=(659.4-1.5, 0.875),
            arrowprops=dict(arrowstyle='<|-|>', facecolor='k', edgecolor='k'),zorder=100)
            #shrink=0.05,
    ax0.text(600.3,0.925,r'$\Delta$E $\sim$ 0.41\,eV', fontsize=fsizenb, va='center',ha='center')

sys.path.append("../2016-12-19_Andrea_BigNPs_5DiffTemps/") # necessary 
###############################################################################
index = 0
Pixel_size = [2.2] #nm
Ps = str("{0:.2f}".format(Pixel_size[index])) 
let = ['Beautiful']
import boe_bar as sb
import matplotlib.colors as colors

se = np.load("../2016-12-19_Andrea_BigNPs_5DiffTemps/" + str(let[index]) +'SEchannel.npz',mmap_mode='r') 
segmm = np.load("../2016-12-19_Andrea_BigNPs_5DiffTemps/" +str(let[index]) +'SEchannelGMM.npz',mmap_mode='r') 
red = np.load("../2016-12-19_Andrea_BigNPs_5DiffTemps/" +str(let[index]) +'Redbright.npz',mmap_mode='r') 
blue = np.load("../2016-12-19_Andrea_BigNPs_5DiffTemps/" +str(let[index]) +'Bluebright.npz',mmap_mode='r') 

fsizepl = 24
fsizenb = 20 
sizex = 8 
sizey = 6
dpi_no = 80
lw = 2

length_scalebar = 100.0 #in nm 
scalebar_legend = '100 nm'
length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[index]))

sizex = 8
sizey = 6
dpi_no = 80


import matplotlib.gridspec as gridspec
import matplotlib as mpl

axA = plt.subplot2grid((nolines,noplots), (0,0), colspan=2, rowspan=3)
axA.text(-0.4, 1.0, 'a', transform=axA.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})

import matplotlib.image as mpimg
#img=mpimg.imread('boe.png')
#axA.imshow(img)
#axA.spines['right'].set_visible(False)
#axA.spines['top'].set_visible(False)
#axA.spines['left'].set_visible(False)
#axA.spines['bottom'].set_visible(False)
#axA.set_yticks([])
#axA.set_xticks([])


axB = plt.subplot2grid((nolines,noplots), (0,3), colspan=2, rowspan=3)
axB.text(-0.5, 1.0, 'b', transform=axB.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})

#img2=mpimg.imread('em.png')
#axB.imshow(img2)
#axB.spines['right'].set_visible(False)
#axB.spines['top'].set_visible(False)
#axB.spines['left'].set_visible(False)
#axB.spines['bottom'].set_visible(False)
#axB.set_yticks([])
#axB.set_xticks([])

inset200 = plt.subplot2grid((nolines,noplots), (0,6), colspan=2, rowspan=3)
inset200.text(-0.3, 1.0, 'd', transform=inset200.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})

yap = np.load('ZZZYAP3.npz')
blueyap = np.average(yap['datablue'],axis=(0))/1.0e3 
indu = 83
Time_bin = 40
my_timo = np.arange(1,blueyap[indu:,:,:].shape[0]+1)*Time_bin/1000
inset200.plot(my_timo, np.average(blueyap[indu:,:,:]/np.max(np.average(blueyap[indu:,:,:],axis=(1,2))),axis=(1,2)), 'o',color='b',markersize=6, markeredgewidth=0.0)
inset200.set_xlabel("Experiment time ($\mu$s)",fontsize=fsizepl)
inset200.set_ylabel("Approx. instrument \n response function (a.u.)",fontsize=fsizepl)
inset200.set_xlim([0,1.1]) #2.5
inset200.set_xticks([1]) #1,2
inset200.set_ylim([0,1.1])
inset200.set_yticks([0.5,1])
inset200.xaxis.set_ticks_position('bottom')
inset200.yaxis.set_ticks_position('left')
inset200.tick_params(labelsize=fsizenb) 
inset200.spines['right'].set_visible(False)
inset200.spines['top'].set_visible(False)
inset200.axvline(my_timo[15], lw=2, color='k', ls='--',ymax = 0.19)
inset200.axvline(my_timo[16] , lw=2, color='k', ls='--',ymax = 0.19)
inset200.text(0.55, 0.23, '40 ns', fontsize=fsizenb) 


###############################################################################

gs = mpl.gridspec.GridSpec(8, 3)
gs.update(wspace=0.1, hspace=-0.5, left=0.115, right=0.5, bottom=0.1, top=0.99) 

ax11 = plt.subplot(gs[4,0])

ax11.text(-0.7, 1.20, 'c', transform=ax11.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5}) #was 1.0 not 1.2


ax11.set_title('Electron signal',fontsize=fsizepl) #as per accompanying txt files
plt.imshow(se['data'],cmap=cm.Greys_r)
#plt.pcolor(se['data'],cmap=cm.Greys_r)


import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=fsizenb)
sbar = sb.AnchoredScaleBar(ax11.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4, my_fontsize = fsizenb)
ax11.add_artist(sbar)
plt.axis('off')


ax1 = plt.subplot(gs[4,1])
ax1.set_title('Green band',fontsize=fsizepl)
plotb = np.average(blue['data'],axis = (0,1))
plotb = plotb
im = plt.imshow(plotb,cmap=cm.Greens,norm=colors.PowerNorm(0.5))#, vmin=plotb.min(), vmax=plotb.max())) #or 'OrRd'
plt.axis('off')

ax2 = plt.subplot(gs[4,2])
#ax2 = plt.subplot2grid((nolines,noplots), (4, 2), colspan=1, rowspan=1)
ax2.set_title('Red band',fontsize=fsizepl)
plotr = np.average(red['data'],axis = (0,1))#hlpse = np.copy(segmm['bright'])
#hlpse[~np.isnan(hlpse)] = 0.0 #inside
#hlpse[np.isnan(hlpse)] = 1.0 #outside

#hlpse = np.ones([arr_img60.shape[0], arr_img60.shape[1]])
#hlpse[hlpse > 0.5*np.average(se['data'], axis=(0,1))] = 0.0
plotr = plotr
imb = plt.imshow(plotr,cmap=cm.Reds,norm=colors.PowerNorm(0.5))#, vmin=plotr.min(), vmax=plotr.max())) #or 'OrRd'
plt.axis('off') 

#sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4, my_fontsize = fsizenb)
#ax1.add_artist(sbar)
unit = '(kHz)'
box = ax1.get_position()
ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
axColor = plt.axes([box.x0, box.y0*1.01, box.width,0.01 ])    #original 1.1
cb2 = plt.colorbar(im, cax = axColor, orientation="horizontal")

cb2.set_label("photon cts. (kHz)", fontsize = fsizenb)
cb2.set_ticks([7, 14])
cb2.ax.tick_params(labelsize=fsizenb) 

#sbar = sb.AnchoredScaleBar(ax2.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4, my_fontsize = fsizenb)
#ax2.add_artist(sbar)  
box = ax2.get_position()
ax2.set_position([box.x0, box.y0*1.00, box.width, box.height])
axColor = plt.axes([box.x0, box.y0*1.01, box.width,0.01 ])    
cb1 = plt.colorbar(imb, cax = axColor, orientation="horizontal")
cb1.set_label('photon cts. (kHz)', fontsize = fsizenb) 
cb1.set_ticks([3.5,7])

cb1.ax.tick_params(labelsize=fsizenb) 



#fig1.subplots_adjust(wspace=-0.1, hspace=-0.1)


#
#
################################################################################
Time_bin = 2000#in ns; 
let = ['I']
sys.path.append("../2016-11-07_Andrea_small_long_and_short_LTs/") # necessary 
se = np.load('../2016-11-07_Andrea_small_long_and_short_LTs/'+ str(let[index]) +'SEchannel.npz') 
segmm = np.load('../2016-11-07_Andrea_small_long_and_short_LTs/'+ str(let[index]) +'SEchannelGMM.npz') 
red = np.load('../2016-11-07_Andrea_small_long_and_short_LTs/'+ str(let[index]) +'Redbright.npz') 
blue = np.load('../2016-11-07_Andrea_small_long_and_short_LTs/' + str(let[index]) +'Bluebright.npz') 

hlp = np.copy(segmm['bright'])
hlp[~np.isnan(hlp)] = 1.0  #inside
hlpd = np.copy(segmm['bright'])
hlpd[~np.isnan(hlpd)] = 0.0 
hlpd[np.isnan(hlpd)] = 1.0 

ax112 = plt.subplot2grid((nolines,noplots), (3,4), colspan=4, rowspan=5)
ax112.text(-0.1, 1.0, 'e', transform=ax112.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})
ax112.spines['right'].set_visible(False)
ax112.spines['top'].set_visible(False)
ax112.xaxis.set_ticks_position('bottom')
ax112.yaxis.set_ticks_position('left')
datared = np.average(red['data'], axis = (0))
datablue = np.average(blue['data'], axis = (0))
initbin = (102)-1 #init bin for decay
backgdinit = 1
### 700ns /40ns = 7. ....
datared_init = datared[0:backgdinit,:,:]
#datared = datared[initbin:,:,:]
datablue_init = datablue[0:backgdinit,:,:]
#datablue = datablue[initbin:,:,:]
aaa = datared*hlp
xx_array = np.arange(0,aaa.shape[0])*Time_bin*1e-9

ax112.semilogy(xx_array/1e-6,np.nanmean(datared*hlp,axis=(1,2)),'o',color='r',markersize=6, markeredgewidth=0.0)  
ax112.semilogy(xx_array/1e-6,np.nanmean(datablue*hlp,axis=(1,2)),'o',color='g',markersize=6, markeredgewidth=0.0)  
ax112.set_ylabel("Average cathodoluminescence \n per pixel (kHz)",fontsize=fsizepl)
ax112.set_xlabel('Experiment time ($\mu$s)',  fontsize=fsizepl)
ax112.set_xlim(xmax=1000.0) #2000
#Plot whole of background decay
ax112.semilogy(xx_array/1e-6,np.nanmean(datared*hlp,axis=(1,2)),'o',color='r',markersize=6,markeredgewidth=0.0)   
ax112.semilogy(xx_array/1e-6,np.nanmean(datablue*hlp,axis=(1,2)),'o',color='g',markersize=6, markeredgewidth=0.0) 
ax112.semilogy(xx_array/1e-6,np.nanmean(datared*hlpd,axis=(1,2)),'o',color='DarkRed',markersize=4, markeredgewidth=0.0)   
ax112.semilogy(xx_array/1e-6,np.nanmean(datablue*hlpd,axis=(1,2)),'o',color='DarkGreen',markersize=4, markeredgewidth=0.0) 

#Plot mean signal before e-beam on
ax112.semilogy(xx_array/1e-6,np.nanmean(datared_init*hlp,axis=(0,1,2))*np.ones(len(xx_array)),'r--',lw=2)
#Plot mean background
ax112.semilogy(xx_array/1e-6,np.nanmean(datared_init*hlpd,axis=(0,1,2))*np.ones(len(xx_array)),'--',color='DarkRed',lw=2)
#Plot mean signal before e-beam on
ax112.semilogy(xx_array/1e-6,np.nanmean(datablue_init*hlp,axis=(0,1,2))*np.ones(len(xx_array)),'g--',lw=2)
#Plot mean background
ax112.semilogy(xx_array/1e-6,np.nanmean(datablue_init*hlpd,axis=(0,1,2))*np.ones(len(xx_array)),'--',color='DarkGreen',lw=2)
ax112.axvspan(2.0,200.0, alpha=0.25, color='yellow')
ax112.text(25,10.6, 'e-beam on', fontsize=fsizenb) #was 11.5 not 10.6

#to show time detail
ajuda=xx_array/1e-6
ajuda2=np.nanmean(datablue*hlp,axis=(1,2))
ax112.vlines(ajuda[140], ajuda2[140], 1.0, colors='k', linestyles='dashed',lw=2,zorder=5000)
ax112.vlines(ajuda[144], ajuda2[141], 1.0, colors='k', linestyles='dashed',lw=2,zorder=6000)
ax112.text(ajuda[130], 1.125, '1 $\mu$s', fontsize=fsizenb)

ax112.set_xlim([6,1000])
ax112.set_ylim([0.02,15])
ax112.tick_params(labelsize=fsizenb)
ax112.set_xticks([250, 500,750]) 
ax112.set_yticks([0.1,1,10]) 
ax112.set_yticklabels(['0.1','1','10'])

se = np.load('../2016-11-07_Andrea_small_long_and_short_LTs/'+'ISEchannel.npz') 
segmm = np.load('../2016-11-07_Andrea_small_long_and_short_LTs/'+'ISEchannelGMM.npz') 

arr_img60 = np.array(se['data'])
se_data = np.array(se['data'])
for k in range(se_data.shape[0]):
            if k < 200:
                for j in range(se_data.shape[1]):
                    if j < 250:
                        if se_data[k, j] < -0.1885:
                            se_data[k, j] = 0.0
                    else:
                        if se_data[k, j] < -0.1735:
                            se_data[k, j] = 0.0
            if 200 <= k and k < 375:
                for j in range(se_data.shape[1]):
                    if se_data[k, j] < -0.1755:
                        se_data[k, j] = 0.0
            if k >= 375:
                for j in range(se_data.shape[1]):
                    if se_data[k, j] < -0.185:
                        se_data[k, j] = 0.0
hlpse = se_data 
hlpse[hlpse < -0.1] = 1.0

arr_img50 = hlpse


length_scalebar = 100.0 #in nm 
scalebar_legend = '100 nm'
length_scalebar_in_pixels = np.ceil(length_scalebar/(1.4))        
import boe_bar as sb

ypos = 0.455
inset2 = fig1.add_axes([0.81, ypos, .12, .12],zorder=1) #was 0.55
inset2.imshow(arr_img50,cmap = cm.Greys_r, zorder=1)
#sbar = sb.AnchoredScaleBar(inset2.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4, my_fontsize = fsizenb)
#inset2.add_artist(sbar)    
plt.setp(inset2, xticks=[], yticks=[],zorder=1)

inset3 = fig1.add_axes([0.7075, ypos, .12, .12],zorder=1) #was 0.55
inset3.imshow(arr_img60,cmap = cm.Greys_r, zorder=1)
sbar = sb.AnchoredScaleBar(inset3.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4, my_fontsize = fsizenb)
inset3.add_artist(sbar)    
plt.setp(inset3, xticks=[], yticks=[],zorder=1)

ax112.annotate('', xy=(500, 0.45), xytext=(700, 3.6),
            arrowprops=dict(facecolor='g', shrink=0.05,edgecolor='None'),zorder=100)
ax112.annotate('', xy=(500, 0.12), xytext=(700, 3.6),
            arrowprops=dict(facecolor='r', shrink=0.05,edgecolor='None'),zorder=101)
ax112.annotate('', xy=(780,0.18), xytext=(800,6),
            arrowprops=dict(facecolor='DarkGreen', shrink=0.05,edgecolor='None'),zorder=102)
ax112.annotate('', xy=(900, 0.02), xytext=(800,6),
            arrowprops=dict(facecolor='DarkRed', shrink=0.05,edgecolor='None'),zorder=103)
ax112.text(635, 10.6, 'segmentation', fontsize=fsizenb) 


ax112.zorder = 10
ax112.patch.set_facecolor('None') 
################################################################################



plt.tight_layout()

multipage_longer('Fig1.pdf',dpi=900)


## works for errorbars but doesnt look nice
#hlp_red_200 = np.nanmean(datared*hlp,axis=(1,2))
#hlp_red_200_err = np.sqrt(hlp_red_200)
#ax112.fill_between(xx_array/1e-6,hlp_red_200 + hlp_red_200_err, hlp_red_200 - 0*hlp_red_200_err,alpha=0.5, edgecolor=my_edgecolor, facecolor= my_facecolor)   
#ax112.set_yscale('log')

