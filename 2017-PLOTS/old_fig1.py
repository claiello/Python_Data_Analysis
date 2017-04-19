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
from matplotlib import colors as mcolors

from numpy import genfromtxt
import skimage.morphology
from skimage.morphology import watershed
####office
from skimage.feature import peak_local_max,canny #UNCOMMENT AND RUN IN BOES COMPI
from skimage import filters 
from skimage.filters import threshold_otsu, threshold_adaptive, rank #, threshold_local
####laptop
#from skimage.feature import peak_local_max#,canny
#from skimage import filter
#from skimage.filter import threshold_otsu, threshold_adaptive, rank, canny #, threshold_local
######

from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.morphology import disk
from scipy import ndimage
import sklearn

import matplotlib.patches as patches
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.io import imread, imsave, imshow
from skimage.morphology import watershed



### aux functions
def moving_average(a,n=3):
    vec = np.cumsum(a)
    vec[n:] = vec[n:] - vec[:-n]
    return (1.0/n)*vec[n-1:]

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
    
def togs(image):
    
    return  (((image - image.min()) / (image.max() - image.min())) * 255.9).astype(np.uint8)
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

    ax0 = plt.subplot2grid((nolines,noplots), (4,0), colspan=4, rowspan=3)
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.xaxis.set_ticks_position('bottom')
    ax0.yaxis.set_ticks_position('left')
     
    #new spectrum
    specBG = genfromtxt('BGApril.txt', delimiter='')
    specRT = genfromtxt('RTApril.txt', delimiter='')
    
    # x, y vectors
    wavel = specRT[:,0] 
    spec_to_plot = specRT[:,1] -  np.array(specBG[:,1])
    # cut vector only between 300 and 720nm
    a = find_nearest(wavel,195)
    b = find_nearest(wavel,905)
    wavel = wavel[a:b]
    spec_to_plot = spec_to_plot[a:b]
    # moving avg to cut noise
    indexmov = 5
    mov_avg_index = indexmov
    spec_to_plot = moving_average(spec_to_plot,n=mov_avg_index)
    wavel = moving_average(wavel,n=mov_avg_index)
  
    
    a = find_nearest(wavel,534)
    b = find_nearest(wavel,566)
    c = find_nearest(wavel,623)
    d = find_nearest(wavel,677)
    wl = 654.8
    wlt = '654.8'
    d = find_nearest(wavel,wl)
    
    ax0.axvspan(534,566, alpha=0.25, color='green')
    ax0.axvspan(623,677, alpha=0.25, color='red')
    
    #labels
    xmin = 500
    xmax = 700
    ax0.set_xlim([xmin, xmax])
    ax0.set_ylabel('Cathodoluminescence \n emission spectrum (a.u.),\n 1kX magn., $\sim$ 25 $^{\circ}$C',fontsize=fsizepl)
    ax0.set_xlabel('Wavelength (nm)',fontsize=fsizepl)
    ax0.tick_params(labelsize=fsizenb)
    ax0.set_yticks([0.5,1.0])
    ax0.set_ylim([0,1.05])
#    ax0.set_xticks([aux_x[f],aux_xx[g]])
#    ax0.set_xticklabels([str("{0:.1f}".format(aux_x[f])),str("{0:.1f}".format(aux_xx[g]))])
    
    ax0.set_xticks([534,566,540.5 ,623,677,wl])
    ax0.set_xticklabels(['534','566','540.5','623.0','677',wlt])
    ax0.get_xaxis().majorTicks[0].label1.set_horizontalalignment('right')
    ax0.get_xaxis().majorTicks[2].label1.set_horizontalalignment('left')
    #ax0.plot(wavel,spec_to_plot/np.max(spec_to_plot), lw=2, color='k')    
    ax0.plot(np.array(wavel),np.array(spec_to_plot/np.max(spec_to_plot)), lw=2, color='k')
    
    ax0.vlines(x=wl , ymin = 0, ymax = 1.05, linewidth=2, color='r', linestyle='--')
    ax0.vlines(x=540.5 , ymin = 0, ymax = 1.05, linewidth=2, color='g', linestyle='--')


    # adding dE with arrow
    yposarr = 1.02 #0.875
    ax0.annotate('',  xy=(540.5+1.5, yposarr), xytext=(654.1-1.5, yposarr),
            arrowprops=dict(arrowstyle='<|-|>', facecolor='k', edgecolor='k'),zorder=100)
            #shrink=0.05,
    ax0.text(600.3,0.95,r'$\Delta$E $\sim$ 0.40\,eV', fontsize=fsizenb, va='center',ha='center')
    # Exact DeltaE = 0.39839eV

sys.path.append("../2016-12-19_Andrea_BigNPs_5DiffTemps/") # necessary 
###############################################################################
index = 0
Pixel_size = [2.2] #nmsys.path.append('/usr/lib/python2.7/dist-packages')
#import cv2
#Ps = str("{0:.2f}".format(Pixel_size[index])) 
#let = ['Beautiful']
import boe_bar as sb
import matplotlib.colors as colors

#se = np.load("../2016-12-19_Andrea_BigNPs_5DiffTemps/" + str(let[index]) +'SEchannel.npz',mmap_mode='r') 
#segmm = np.load("../2016-12-19_Andrea_BigNPs_5DiffTemps/" +str(let[index]) +'SEchannelGMM.npz',mmap_mode='r') 
#red = np.load("../2016-12-19_Andrea_BigNPs_5DiffTemps/" +str(let[index]) +'Redbright.npz',mmap_mode='r') 
#blue = np.load("../2016-12-19_Andrea_BigNPs_5DiffTemps/" +str(let[index]) +'Bluebright.npz',mmap_mode='r') 

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
axA.text(-0.625, 1.0, 'a', transform=axA.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})
axA.axis('off')
import matplotlib.image as mpimg

axB = plt.subplot2grid((nolines,noplots), (0,3), colspan=2, rowspan=3)
axB.text(-1.5, 1.0, 'b', transform=axB.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})
axB.axis('off')

inset200 = plt.subplot2grid((nolines,noplots), (0,6), colspan=2, rowspan=3)
inset200.text(-0.55, 1.0, 'd', transform=inset200.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})

yap = np.load('ZZZYAP3.npz')
blueyap = np.average(yap['datablue'],axis=(0))/1.0e3 
indu = 83
Time_bin = 40
my_timo = np.arange(1,blueyap[indu:,:,:].shape[0]+1)*Time_bin/1000.0
inset200.semilogy(my_timo, np.average(blueyap[indu:,:,:]/np.max(np.average(blueyap[indu:,:,:],axis=(1,2))),axis=(1,2)), 'o',color='purple',markersize=6, markeredgewidth=0.0)
inset200.set_xlabel("Experiment time ($\mu$s)",fontsize=fsizepl)
inset200.set_ylabel("Approx. instrument \n response function (a.u.)",fontsize=fsizepl)
inset200.set_xlim([0,1.1]) #2.5
inset200.set_xticks([1]) #1,2

inset200.xaxis.set_ticks_position('bottom')
inset200.yaxis.set_ticks_position('left')
inset200.tick_params(labelsize=fsizenb) 
inset200.spines['right'].set_visible(False)
inset200.spines['top'].set_visible(False)
inset200.axvline(my_timo[15], lw=2, color='k', ls='--',ymin = 0.45, ymax = 0.75)
inset200.axvline(my_timo[16] , lw=2, color='k', ls='--',ymin = 0.45, ymax = 0.75)
inset200.text(0.55, 0.23, '40 ns', fontsize=fsizenb) 
mith = np.load('Decay-Mithrene.npz')
inset200.semilogy(np.arange(0,len(mith['data'][0,158:]))*Time_bin/1000.0, mith['data'][0,158:]/np.max(mith['data'][0,158:]),'o',color='b',markersize=6, markeredgewidth=0.0)

inset200.set_ylim([0.0009,1.2])
inset200.set_yticks([0.001, 0.01, 0.1, 1]) #, minslopered,minslopegreen])
inset200.set_yticklabels(['0.001','0.01','0.1','1']) #,'0.021','0.048'])
inset200.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

###############################################################################

gs = mpl.gridspec.GridSpec(8, 3)
gs.update(wspace=0.1, hspace=-0.5, left=0.115, right=0.5, bottom=0.1, top=0.99) 

ax11 = plt.subplot(gs[7,0])

ax11.text(-0.7, 1.20, 'c', transform=ax11.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5}) #was 1.0 not 1.2


ax11.set_title('Electron signal',fontsize=fsizepl) #as per accompanying txt files
#plt.imshow(se['data'],cmap=cm.Greys_r)


import matplotlib.font_manager as fm
#fontprops = fm.FontProperties(size=fsizenb)
#sbar = sb.AnchoredScaleBar(ax11.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4, my_fontsize = fsizenb)
#ax11.add_artist(sbar)
#plt.axis('off')


ax1 = plt.subplot(gs[7,1])
ax1.set_title('Gradient',fontsize=fsizepl)
#plotb = np.average(blue['data'],axis = (0,1))
#plotb = plotb
#im = plt.imshow(plotb,cmap=cm.Greens,norm=colors.PowerNorm(0.5))#, vmin=plotb.min(), vmax=plotb.max())) #or 'OrRd'
#plt.axis('off')

ax2 = plt.subplot(gs[7,2])
#ax2 = plt.subplot2grid((nolines,noplots), (4, 2), colspan=1, rowspan=1)
ax2.set_title('Segmentation',fontsize=fsizepl)
#plotr = np.average(red['data'],axis = (0,1))#hlpse = np.copy(segmm['bright'])

#imb = plt.imshow(plotr,cmap=cm.Reds,norm=colors.PowerNorm(0.5))#, vmin=plotr.min(), vmax=plotr.max())) #or 'OrRd'
#plt.axis('off') 
#unit = '(kHz)'
#box = ax1.get_position()
#ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
#axColor = plt.axes([box.x0, box.y0*1.01, box.width,0.01 ])    #original 1.1
#cb2 = plt.colorbar(im, cax = axColor, orientation="horizontal")
#
#cb2.set_label("photon cts. (kHz)", fontsize = fsizenb)
#cb2.set_ticks([7, 14])
#cb2.ax.tick_params(labelsize=fsizenb) 
#
##sbar = sb.AnchoredScaleBar(ax2.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4, my_fontsize = fsizenb)
##ax2.add_artist(sbar)  
#box = ax2.get_position()
#ax2.set_position([box.x0, box.y0*1.00, box.width, box.height])
#axColor = plt.axes([box.x0, box.y0*1.01, box.width,0.01 ])    
#cb1 = plt.colorbar(imb, cax = axColor, orientation="horizontal")
#cb1.set_label('photon cts. (kHz)', fontsize = fsizenb) 
#cb1.set_ticks([3.5,7])
#
#cb1.ax.tick_params(labelsize=fsizenb) 



#fig1.subplots_adjust(wspace=-0.1, hspace=-0.1)


#
#
#==============================================================================
# ################################################################################
sys.path.append("../2017-04-05_Andrea_NPs_Different_excitation_times/")
blue = np.load('../2017-04-05_Andrea_NPs_Different_excitation_times/'+'ns500Bluebright.npz',mmap_mode='r')     
red = np.load('../2017-04-05_Andrea_NPs_Different_excitation_times/'+'ns500Redbright.npz',mmap_mode='r')

red = np.array(red['data'])
blue = np.array(blue['data'])

red = np.average(red,axis=(0,2,3))
blue = np.average(blue,axis=(0,2,3))

ax112 = plt.subplot2grid((nolines,noplots), (3,4), colspan=4, rowspan=5)
ax112.text(-0.2, 1.0, 'e', transform=ax112.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
      bbox={'facecolor':'None', 'pad':5})
ax112.spines['right'].set_visible(False)
ax112.spines['top'].set_visible(False)
ax112.xaxis.set_ticks_position('bottom')
ax112.yaxis.set_ticks_position('left')
Time_bin = 1000.0
xx_array = np.arange(12,1010)*Time_bin*1e-9
ax112.semilogy(xx_array/1e-6,red[12:]/np.max(red[12:]),'o',color='r',markersize=6, markeredgewidth=0.0)  
ax112.semilogy(xx_array/1e-6,blue[12:]/np.max(blue[12:]),'o',color='g',markersize=6, markeredgewidth=0.0)  
ax112.set_ylabel("Average cathodoluminescence \n per pixel (kHz)",fontsize=fsizepl)
ax112.set_xlabel('Experiment time ($\mu$s)',  fontsize=fsizepl)
ax112.set_xlim(xmax=1010.0) #2000

# #to show time detail
ajuda=xx_array/1e-6
ajuda2=np.nanmean(blue)
ax112.vlines(750, 0.3, 0.4, colors='k', linestyles='dashed',lw=2,zorder=5000)
ax112.vlines(760, 0.3, 0.4, colors='k', linestyles='dashed',lw=2,zorder=6000)
ax112.text(755, 0.5, '1 $\mu$s', fontsize=fsizenb)
# 
ax112.set_xlim([10,1010])
ax112.set_ylim([0.0009,1.2])
ax112.tick_params(labelsize=fsizenb)
ax112.set_xticks([500,1000]) 
ax112.set_yticks([0.001, 0.01, 0.1,1]) 
ax112.set_yticklabels(['0.001', '0.01', '0.1','1'])
ax112.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax112.axvspan(12.0,500.0,alpha=0.25,color='yellow')
ax112.text(250, 0, 'e-beam on', fontsize=fsizenb, va='center',ha='center') 
# 
# se = np.load('../2016-11-07_Andrea_small_long_and_short_LTs/'+'ISEchannel.npz') 
# segmm = np.load('../2016-11-07_Andrea_small_long_and_short_LTs/'+'ISEchannelGMM.npz') 
# 
# arr_img60 = np.array(se['data'])
# se_data = np.array(se['data'])
# for k in range(se_data.shape[0]):
#             if k < 200:
#                 for j in range(se_data.shape[1]):
#                     if j < 250:
#                         if se_data[k, j] < -0.1885:
#                             se_data[k, j] = 0.0
#                     else:
#                         if se_data[k, j] < -0.1735:
#                             se_data[k, j] = 0.0
#             if 200 <= k and k < 375:
#                 for j in range(se_data.shape[1]):
#                     if se_data[k, j] < -0.1755:
#                         se_data[k, j] = 0.0
#             if k >= 375:
#                 for j in range(se_data.shape[1]):
#                     if se_data[k, j] < -0.185:
#                         se_data[k, j] = 0.0
# hlpse = se_data 
# hlpse[hlpse < -0.1] = 1.0
# 
# arr_img50 = hlpse
# 
# 
# length_scalebar = 100.0 #in nm 
# scalebar_legend = '100 nm'
# length_scalebar_in_pixels = np.ceil(length_scalebar/(1.4))        
# import boe_bar as sb
# 
# ypos = 0.455
# inset2 = fig1.add_axes([0.81, ypos, .12, .12],zorder=1) #was 0.55
# inset2.imshow(arr_img50,cmap = cm.Greys_r, zorder=1)
# #sbar = sb.AnchoredScaleBar(inset2.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4, my_fontsize = fsizenb)
# #inset2.add_artist(sbar)    
# plt.setp(inset2, xticks=[], yticks=[],zorder=1)
# 
# inset3 = fig1.add_axes([0.7075, ypos, .12, .12],zorder=1) #was 0.55
# inset3.imshow(arr_img60,cmap = cm.Greys_r, zorder=1)
# sbar = sb.AnchoredScaleBar(inset3.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4, my_fontsize = fsizenb)
# inset3.add_artist(sbar)    
# plt.setp(inset3, xticks=[], yticks=[],zorder=1)
# 
# ax112.annotate('', xy=(500, 0.45), xytext=(700, 3.6),
#             arrowprops=dict(facecolor='g', shrink=0.05,edgecolor='None'),zorder=100)
# ax112.annotate('', xy=(500, 0.12), xytext=(700, 3.6),
#             arrowprops=dict(facecolor='r', shrink=0.05,edgecolor='None'),zorder=101)
# ax112.annotate('', xy=(780,0.18), xytext=(800,6),
#             arrowprops=dict(facecolor='DarkGreen', shrink=0.05,edgecolor='None'),zorder=102)
# ax112.annotate('', xy=(900, 0.02), xytext=(800,6),
#             arrowprops=dict(facecolor='DarkRed', shrink=0.05,edgecolor='None'),zorder=103)
#==============================================================================
#ax112.text(635, 10.6, 'segmentation', fontsize=fsizenb) 


ax112.zorder = 10
ax112.patch.set_facecolor('None') 
################################################################################


####
do_B = True #already ran, can just open files and read
if do_B:
    sys.path.append('/usr/lib/python2.7/dist-packages')
    import cv2
    print('dob')
    
    Pixel_size = np.array([0.89]) 
    
    SEA= np.load('../2017-02-13_SingleAnalysis/V0SEchannelB.npz')
    xlen = SEA['data'].shape[0]
    ylen = SEA['data'].shape[1]
    delx = 0#+28
    dely = 0 #+26
    xval = 96
    yval = 106
    cutx = 0 #32
    cutxtop = 0 #10
    se = SEA['data'][int(np.floor(xlen/2.0)-xval+delx+cutxtop):int(np.floor(xlen/2.0)+xval+delx-cutx),int(np.floor(ylen/2.0)-yval+dely):int(np.floor(ylen/2.0)+yval+dely)]
       
    ax11.imshow(se,cmap=cm.Greys_r)
    ax11.axis('off') 
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[0]))
    sbar = sb.AnchoredScaleBar(ax11.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4, my_fontsize = fsizenb, my_linewidth=2)
    ax11.add_artist(sbar)       
       
    # fourier transform
    
    myfft = np.fft.fft2(se)
    mycut = 15 # less means more cut here, i.e. lower frequencies only
    myfft[0+mycut:192-mycut, 0:] = 0
    myfft[0:, 0+mycut:212-mycut] = 0
    
    new_pic = np.abs(np.fft.ifft2(myfft))
    se = new_pic
    #binary threshold
    se_data2 = np.copy(se)

    tv_filt = denoise_tv_chambolle(se_data2, weight=0.0025)
    grad_x, grad_y = np.gradient(tv_filt)
    new_pic_grad = np.sqrt(grad_x**2 + grad_y**2)
    new_pic_grad = scipy.ndimage.filters.gaussian_filter(new_pic_grad,1)

    ########
    I8 = (((new_pic_grad - new_pic_grad.min()) / (new_pic_grad.max() - new_pic_grad.min())) * 255.9).astype(np.uint8)
    bw = cv2.adaptiveThreshold(I8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)  #101,1

    from scipy import ndimage
    hlpse2 = ndimage.imread('../2017-02-13_SingleAnalysis/bw_minimal.png', mode='L')

    hlpse2 = togs(1.-hlpse2)
    distance = ndimage.distance_transform_edt(hlpse2)
    local_maxi = peak_local_max(
            distance, 
            num_peaks = 15, 
            indices = False, 
            footprint = np.ones((50,50)),
            labels = hlpse2) #footprint = min dist between maxima to find #footprint was 25,25
    markers = skimage.morphology.label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=hlpse2)
    lab = np.unique(labels_ws)
    
    #relatively new code
#    labels_ws[labels_ws == 2] = 0 # delete this label
#    my_max = np.max(labels_ws)
#    labels_ws[labels_ws == my_max] = 2 # replace highest labels with labels 2
#    lab = np.unique(labels_ws)

   #old code, was working
    #Make random colors, not degrade
#    rand_ind = np.random.permutation(lab)
#    new_labels_ws = np.copy(labels_ws)
#    for k in range(new_labels_ws.shape[0]):
#        for j in range(new_labels_ws.shape[1]):
#            new_labels_ws[k, j] = rand_ind[labels_ws[k, j]]
#    labels_ws = new_labels_ws
        
    areas = np.array([])
    for k in lab:
        areas = np.append(areas, len( labels_ws[labels_ws == k] ))
    cut_k = []
    cut_labels_ws = np.copy(labels_ws)
    non_cut_k = []  ###### change cut_k
    for k in range(len(lab)):
        if (areas[k] < 1200) or (areas[k] > 4000):
            cut_labels_ws[cut_labels_ws == lab[k]] = 0
            cut_k.append(k)
        else:
            non_cut_k.append(k)  ###change cut_k
    
    #new code to make black white
    #cut_labels_ws[cut_labels_ws > 1] = 1.0
    #or 'OrRd'
#    
#    ax011.axis('off')
#    ax0111.imshow(bw,cmap=cm.Greys_r) #or 'OrRd'
#    ax0111.axis('off')
#    ax0112.imshow(new_pic_grad,cmap=cm.Greys_r) #or 'OrRd'
#    ax0112.axis('off')

   
#    ax2.imshow(bw,cmap=cm.Greys_r) #or 'OrRd'
    #####UNCOMMEN 4 below
    ax2.imshow(cut_labels_ws, cmap = cm.Greys_r)
    ax1.imshow(new_pic_grad,cmap=cm.Greys_r) #or 'OrRd'
    ax1.axis('off')
    ax2.axis('off')

    del SEA, se
    gc.collect()
 




#####3



plt.tight_layout()

multipage_longer('Fig1.pdf',dpi=900)


## works for errorbars but doesnt look nice
#hlp_red_200 = np.nanmean(datared*hlp,axis=(1,2))
#hlp_red_200_err = np.sqrt(hlp_red_200)
#ax112.fill_between(xx_array/1e-6,hlp_red_200 + hlp_red_200_err, hlp_red_200 - 0*hlp_red_200_err,alpha=0.5, edgecolor=my_edgecolor, facecolor= my_facecolor)   
#ax112.set_yscale('log')

