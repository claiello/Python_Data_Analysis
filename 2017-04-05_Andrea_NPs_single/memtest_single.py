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
import matplotlib.cm as cm
import scipy.misc
import gc
import tempfile
from tempfile import TemporaryFile
import skimage
from skimage import exposure
from my_fits import *
import pickle
import my_fits
from uncertainties import unumpy
from numpy import genfromtxt
import boe_bar as sb
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

from uncertainties import unumpy

from boe_segment import give_bolinha

from calc_exp_hist import *


#from skimage.morphology import black_tophat, skeletonize, convex_hull_image
#from skimage import segmentation
#from skimage.morphology import erosion, dilation, opening, closing, white_tophat
#from skimage.filters import roberts, sobel, scharr, prewitt

sys.path.append('/usr/lib/python2.7/dist-packages')
import cv2

### settings
fsizepl = 24
fsizenb = 20
###
def tauestimate(counts_red, error_red):
    
    print(counts_red.shape[1])
    
    ucounts_red = unumpy.uarray(counts_red, error_red)
    
    def helper(arrayx):
         arrayx[arrayx < 1e-12] = 1e-12   #so that no division by zero     
         return np.cumsum(arrayx*np.arange(1,counts_red.shape[1]+1), axis = 1)/np.cumsum(arrayx, axis = 1)
    
    return helper(ucounts_red)
    
def tauestimate2(counts_red, error_red, counts_blue, error_blue):
    
        print(counts_red.shape[1])
        
        ucounts_red = unumpy.uarray(counts_red, error_red)
        ucounts_blue = unumpy.uarray(counts_blue, error_blue)
        
        def helper(arrayx):
             return np.cumsum(arrayx*np.arange(1,counts_red.shape[1]+1), axis = 1)/np.cumsum(arrayx, axis = 1)
        
        return helper(ucounts_red),helper(ucounts_blue)
    
def markers_from_distance_local_max(image, distance=None, min_distance=10):
    if distance is None:
        distance = ndi.distance_transform_edt(image)
    return

def togs(image):
    
    return  (((image - image.min()) / (image.max() - image.min())) * 255.9).astype(np.uint8)

# eventually, should be range(len(lab)); right now, getting some areas only to see if works
def pickvec(lab):
    
    return range(len(lab))  #np.arange(0,4)
    
def tor(area):
    
    return 2*np.sqrt(2.*area/(3.*np.sqrt(3)))
    
def moving_average(a,n=3):
        ret = np.cumsum(a)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n-1:]/n
        
def plotinho(ax0, dset,my_color,my_edgecolor, my_facecolor):   
        
        movav = 1
           
        ax0.plot(moving_average(np.arange(1,taured.shape[1]+1),movav),moving_average(unumpy.nominal_values(dset[0,:]),movav),color=my_color,ls='--',lw=2)
       
        ax0.fill_between(moving_average(np.arange(1,taured.shape[1]+1),movav),
                                                 moving_average(unumpy.nominal_values(dset[0,:]),movav)-moving_average(unumpy.std_devs(dset[0,:]),movav),
                                                 moving_average(unumpy.nominal_values(dset[0,:]),movav)+moving_average(unumpy.std_devs(dset[0,:]),movav),
                                                 edgecolor=my_edgecolor,
                                                 facecolor=my_facecolor,
                                                 alpha=0.5,
                                                 linewidth=0.0)
         
        ax0.plot(moving_average(np.arange(1,taured.shape[1]+1),movav),moving_average(np.arange(1,taured.shape[1]+1)/2,movav),color='k',ls='--',lw=2)        
        
        my_x= moving_average(np.arange(1,taured.shape[1]+1),movav)
      
        ax0.set_xlim(xmin= my_x[0],xmax = my_x[-1])       
        
        ax0.set_xticks([500,1000])
        ax0.set_xticklabels([])
        
        ax0.set_yticks([100,200,300,400,500,600])#,200])#
        ax0.set_ylim([0,610])#
        ax0.tick_params(labelsize=fsizenb)



@profile
def my_test():
         
   
   sizex = 8
   sizey=6
   dpi_no = 80
   
   fig1= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
   fig1.set_size_inches(1200./fig1.dpi,900./fig1.dpi)
   plt.rc('text', usetex=True)
   plt.rc('font', family='serif')
   plt.rc('font', serif='Palatino')    
   
   nolines = 20
   noplots = 21
   
   ax00 = plt.subplot2grid((nolines,noplots), (0,0), colspan=2, rowspan=2)
   ax03 = plt.subplot2grid((nolines,noplots), (0,2), colspan=2, rowspan=2)
   ax00.text(-0.125, 1.0, 'a', transform=ax00.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
            bbox={'facecolor':'None', 'pad':5})
   
   ax001 = plt.subplot2grid((nolines,noplots), (3,0), colspan=2, rowspan=2)
   ax011 = plt.subplot2grid((nolines,noplots), (3,2), colspan=2, rowspan=2)
   ax001.text(-0.1, 1.0, 'b', transform=ax001.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
            bbox={'facecolor':'None', 'pad':5})
   
   ax0022 = plt.subplot2grid((nolines,noplots), (6,0), colspan=2, rowspan=2)
   axpic = plt.subplot2grid((nolines,noplots), (6,2), colspan=2, rowspan=2)
   ax0022.text(-0.1, 1.0, 'c', transform=ax0022.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
            bbox={'facecolor':'None', 'pad':5})
   ax0022.axis('off')
   axpic.axis('off')
            
   ax0022b = plt.subplot2grid((nolines,noplots), (9,0), colspan=2, rowspan=2)
   axpicb = plt.subplot2grid((nolines,noplots), (9,2), colspan=2, rowspan=2)
   ax0022b.text(-0.1, 1.0, 'd', transform=ax0022b.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
            bbox={'facecolor':'None', 'pad':5})
            
   axratiomag = plt.subplot2grid((nolines,noplots), (0,5), colspan=4, rowspan=3)
   axtaumag = plt.subplot2grid((nolines,noplots), (4,5), colspan=4, rowspan=3)
   axstdmag = plt.subplot2grid((nolines,noplots), (8,5), colspan=4, rowspan=3)
   
   axratiomag.spines['right'].set_visible(False)
   axratiomag.spines['top'].set_visible(False)
   axratiomag.xaxis.set_ticks_position('bottom')
   axratiomag.yaxis.set_ticks_position('left')
   axratiomag.set_ylabel(r'Ratio of int. (a.u.)',fontsize=fsizepl)
   axratiomag.set_xlabel('Avg. pixels per NP (a.u.)',fontsize=fsizepl)
   axratiomag.tick_params(labelsize=fsizenb)
   
   axtaumag.spines['right'].set_visible(False)
   axtaumag.spines['top'].set_visible(False)
   axtaumag.xaxis.set_ticks_position('bottom')
   axtaumag.yaxis.set_ticks_position('left')
   axtaumag.set_ylabel(r'$\tau$ ($\mu$s)',fontsize=fsizepl)
   axtaumag.set_xlabel(r'Acquisition time ($\mu$s)',fontsize=fsizepl)
   axtaumag.set_xticks([500,1000])
   axtaumag.set_xticklabels([500,1000])
   axtaumag.tick_params(labelsize=fsizenb)
   
   
   axstdmag.spines['right'].set_visible(False)
   axstdmag.spines['top'].set_visible(False)
   axstdmag.xaxis.set_ticks_position('bottom')
   axstdmag.yaxis.set_ticks_position('left')
   axstdmag.tick_params(labelsize=fsizenb)
   
   axhist150ratio = plt.subplot2grid((nolines,noplots), (0,11), colspan=4, rowspan=3)
   axexp150ratio = plt.subplot2grid((nolines,noplots), (4,11), colspan=4, rowspan=3)
   axhist150tau_blue = plt.subplot2grid((nolines,noplots), (8,11), colspan=4, rowspan=3)
   
   axhist300ratio = plt.subplot2grid((nolines,noplots), (0,17), colspan=4, rowspan=3)
   axexp300ratio = plt.subplot2grid((nolines,noplots), (4,17), colspan=4, rowspan=3)
   axhist300tau_blue = plt.subplot2grid((nolines,noplots), (8,17), colspan=4, rowspan=3)
   
   axhist150ratio.tick_params(labelsize=fsizenb)
   axexp150ratio.tick_params(labelsize=fsizenb)
   axhist150tau_blue.tick_params(labelsize=fsizenb)
   axhist300ratio.tick_params(labelsize=fsizenb)
   axexp300ratio.tick_params(labelsize=fsizenb)
   axhist300tau_blue.tick_params(labelsize=fsizenb)
   
   #salebar
   length_scalebar = 100.0 #in nm 
   scalebar_legend = '100 nm'
   #when beam is on and off
   backgroundinit = 50
   initbin = 202
   #to plot time evolution with rainbow, initial index and steps
   step = 50
   initind = 10
      
   #### A
   do_150 = True
   if do_150:
       
       print('do 150')
       
       SEA= np.load('x150SEchannel.npz') #init shape (342, 315)
       xlen = SEA['data'].shape[0]
       ylen = SEA['data'].shape[1]
       xinit = 42
       xfinal = -42
       yinit = 15
       yfinal = -15
       se = SEA['data'][xinit:xfinal,yinit:yfinal]
       
       new_pic = give_bolinha('x150SEchannel.npz', xinit, yinit, xfinal, yfinal, corr_threshold = 0.35, n = 30, r = 5, save_file = False, do_plot = False)
       cutx = 35 # was 50
       cuty = 1
       se = se[cutx:-cutx, cuty:-cuty]
   
       ax00.imshow(se,cmap=cm.Greys_r)
       ax00.axis('off')
       
       new_pic = new_pic[cutx:-cutx, cuty:-cuty]
       setr = new_pic
       #binary threshold
       se_data2 = np.copy(setr)
       
       new_hlp = new_pic
       I8 = (new_hlp * 255.9).astype(np.uint8)
       bw = cv2.adaptiveThreshold(I8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 0)
       from scipy import ndimage
       hlpse2 = bw 
       hlpse2[hlpse2 > 1] = 1.0
       hlpse2[hlpse2 < 1] = 0.0
       distance = ndimage.distance_transform_edt(hlpse2)
       
       local_maxi = peak_local_max(
           distance, 
           num_peaks = 50, 
           indices = False, 
           footprint = np.ones((27,27)),
           labels = hlpse2) #footprint = min dist between maxima to find #footprint was 25,25
       markers = skimage.morphology.label(local_maxi)
       labels_ws = watershed(-distance, markers, mask=hlpse2)
       lab = np.unique(labels_ws)
       
       # Make random colors, not degrade
   #    rand_ind = np.random.permutation(lab)
   #    new_labels_ws = np.copy(labels_ws)
   #    for k in range(new_labels_ws.shape[0]):
   #        for j in range(new_labels_ws.shape[1]):
   #            new_labels_ws[k, j] = rand_ind[labels_ws[k, j]]
   #    labels_ws =  new_labels_ws
       
       Pixel_size = np.array([2.5]) 
       length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[0]))
       sbar = sb.AnchoredScaleBar(ax00.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 9, my_fontsize = fsizenb, my_linewidth= 2)
       ax00.add_artist(sbar)
       
       areas = np.array([])
       for k in lab:
           areas = np.append(areas, len( labels_ws[labels_ws == k] ))
       cut_k = []
       cut_labels_ws = np.copy(labels_ws)
       non_cut_k = []  ###### change cut_k
       for k in range(len(lab)):
           if (areas[k] < 10) or (areas[k] > 4000):
               cut_labels_ws[cut_labels_ws == lab[k]] = 0
               cut_k.append(k)
           else:
               non_cut_k.append(k)  ###change cut_k
       
       #print(len(lab))
       #print(len(non_cut_k))
       #ax01.imshow(new_pic_grad,cmap=cm.Greys_r) #or 'OrRd'
       #ax01.axis('off')
       #ax02.imshow(masklarge,cmap=cm.Greys_r) #or 'OrRd'
       #ax02.axis('off')
      
       #code to make all black - this will crash if running to save data
       #cut_labels_ws[cut_labels_ws > 1] = 1.0
       
       ax03.imshow(cut_labels_ws, cmap = cm.Greys_r) #or 'OrRd'
       ax03.axis('off')
       
       del SEA, se
       gc.collect()
       
           ####### load file that exists
       C = pickle.load( open( "150.p", "rb" ) )
       areas = C['areas']
       taured = C['taured']
       tauredstd = C['tauredstd']
       taublue = C['taublue']
       taubluestd = C['taubluestd']
       intens = C['intens']
       stdintens = C['intensstd']
       intensr = C['intensr']
       stdintensr = C['intensstdr']
       intensb = C['intensb']
       stdintensb = C['intensstdb']
       non_cut_k = C['non_cut_k']  
       
       
   #    REDA = np.load('x150Redbright.npz')
   #    reda = REDA['data'][:,:,xinit+cutx:xfinal-cutx,yinit+cuty:yfinal-cuty] #same no pixels than C, single
   #    del REDA
   #    gc.collect()
   #    BLUEA = np.load('x150Bluebright.npz')
   #    bluea = BLUEA['data'][:,:,xinit+cutx:xfinal-cutx,yinit+cuty:yfinal-cuty]#same no pixels than C, single
   #    del BLUEA
   #    gc.collect() 
   #    
   #    no_avg = reda.shape[0]
   #    intens = np.empty([len(lab),no_avg])
   #    stdintens = np.empty([len(lab),no_avg])
   #    intensr = np.empty([len(lab),no_avg])
   #    stdintensr = np.empty([len(lab),no_avg])
   #    intensb = np.empty([len(lab),no_avg])
   #    stdintensb = np.empty([len(lab),no_avg])
   #    
   #    taured = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
   #    taublue = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
   #    tauredstd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
   #    taubluestd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
       
       for k in non_cut_k:
   #    import random
   #    for k in random.sample(non_cut_k, 5):
        
   #       print('indice150')
   #       print(k)
   #       
   #       #Ratio
   #       print('Ratio')
   #       hlp = np.zeros(cut_labels_ws.shape)
   #       hlp[cut_labels_ws == lab[k]] = 1.0
   #       hlp[cut_labels_ws != lab[k]] = np.nan
   #       ureda  =  reda[:,backgroundinit:initbin,:,:]
   #       ubluea =  bluea[:,backgroundinit:initbin,:,:]
   #                     
   #       vecr = np.nanmean(ureda * hlp, axis = (1,2,3))
   #       vecb = np.nanmean(ubluea * hlp, axis = (1,2,3))
   #       del ureda, ubluea
   #       gc.collect()
   #       
   #       vec = vecr/vecb
   #       
   #       No = np.nansum(hlp.astype(np.float64))*reda[:,backgroundinit:initbin,:,:].shape[1]
   #              
   #       #import IPython
   #       #IPython.embed()
   #       
   #       vecstd = vec * np.sqrt( 1.0/vecr/No + 1.0/vecb/No )
   #       
   #       vecstdr = 1.0/np.sqrt(No) * np.sqrt(vecr) 
   #       vecstdb = 1.0/np.sqrt(No) * np.sqrt(vecb) 
   #       
   #       intens[k,:] = vec
   #       stdintens[k,:] = vecstd
   #       intensr[k,:] = vecr
   #       stdintensr[k,:] = vecstdr
   #       intensb[k,:] = vecb
   #       stdintensb[k,:] = vecstdb
   #       print(vec)
   #       print(vecstd)
   #       del vec, vecstd
   #       gc.collect()
   #       
   #     
   #       
   #       print('Taus')
   #       #Taus as a function of time
   #       hlp = np.zeros(cut_labels_ws.shape)
   #       hlp[cut_labels_ws == lab[k]] = 1.0
   #       Notr = np.sum(hlp.astype(np.float64))
   #       redd = np.sum(reda[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
   #       hr = tauestimate(redd,np.sqrt(redd)/np.sqrt(Notr))
   #       taured[k,:,:] = unumpy.nominal_values(hr)
   #       tauredstd[k,:,:] = unumpy.std_devs(hr)
   #       del hr, redd
   #       gc.collect()
   #       blued = np.sum(bluea[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
   #       hb = tauestimate(blued,np.sqrt(blued)/np.sqrt(Notr))
   #       taublue[k,:,:] = unumpy.nominal_values(hb)
   #       taubluestd[k,:,:] = unumpy.std_devs(hb)
   #       del hb, blued
   #       gc.collect()
            pass
        
   #    print('no pixels 150kX')
   #    print(np.nanmean(C['areas'][C['non_cut_k']])/Pixel_size**2)
   #    klklklk
                
       axratiomag.errorbar(172,np.nanmean(C['intens'][C['non_cut_k']],axis=(0,1)),yerr=np.nanstd(C['intens'][C['non_cut_k']],axis=(0,1)),marker='o', color='k', markersize=12) 
   
       plot_ratio_hist(
               C['intens'][C['non_cut_k']],
               axhist150ratio,
               axexp150ratio,
               no_of_bins = 30,
               index = 0)
    
       plot_tau_hist(
               C['taured'][C['non_cut_k']],
               axhist150tau_blue,
               "",#axexp150tau_blue,
               which_taus = [250, 500, 1000],
               no_of_bins = 30,
               my_color = 'r',
               my_title= 'Red band ')
   
       plot_tau_hist(
               C['taublue'][C['non_cut_k']],
               axhist150tau_blue,
               "", #axexp150tau_blue,
               which_taus = [250, 500, 1000],
               no_of_bins = 30,
               my_color = 'g',
               my_title= 'Green band ')
   
   #    del reda, bluea
   #    gc.collect()
   #    
   #    save_data = {}
   #    save_data['areas'] = areas*Pixel_size**2 #in nm^2
   #    save_data['taured'] = taured
   #    save_data['tauredstd'] = tauredstd
   #    save_data['taublue'] = taublue
   #    save_data['taubluestd'] = taubluestd
   #    save_data['intens'] = intens
   #    save_data['intensstd'] = stdintens
   #    save_data['intensr'] = intensr
   #    save_data['intensstdr'] = stdintensr
   #    save_data['intensb'] = intensb
   #    save_data['intensstdb'] = stdintensb
   #    save_data['non_cut_k'] = non_cut_k
   #    
   #    del taured, taublue
   #    gc.collect()
   #    
   #    pickle.dump(save_data, open("150.p", "wb"))   
   #    
   #    print('radius found for C') #to check which of the found areas is background, which is signal
   #    print(tor(areas*Pixel_size**2))
   #    print(non_cut_k)
   
   do_PENTA = True
   if do_PENTA:
       
       print('do PENTA')
       
       SEA= np.load('Try1SEchannel.npz') #init shape (342, 315)
       xlen = SEA['data'].shape[0]
       ylen = SEA['data'].shape[1]
   #    print(xlen)
   #    print(ylen)
   #    klklklk
       xinit = 70
       xfinal = -70
       yinit = 53
       yfinal = -53
       se = SEA['data'][xinit:xfinal,yinit:yfinal]
   
       #PLAY A LITTLE BIT WITH THOSE     
       new_pic = give_bolinha('Try1SEchannel.npz', xinit, yinit, xfinal, yfinal, corr_threshold = 0.35, n = 30, r = 5, save_file = False, do_plot = False)
       cutx = 1 #
       cutyinit = 1 #20  #### A lot of background to the left - keep as much as possible as I need to get good bg statistics
       cuty = 1
       se = se[cutx:-cutx, cutyinit:-cuty]
   
       ax0022.imshow(se,cmap=cm.Greys_r)
       ax0022.axis('off')
       
       new_pic = new_pic[cutx:-cutx, cutyinit:-cuty]
       setr = new_pic
       #binary threshold
       se_data2 = np.copy(setr)
       
       new_hlp = new_pic
       I8 = (new_hlp * 255.9).astype(np.uint8)
       bw = cv2.adaptiveThreshold(I8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 0)
       from scipy import ndimage
       hlpse2 = bw 
       hlpse2[hlpse2 > 1] = 1.0
       hlpse2[hlpse2 < 1] = 0.0
       distance = ndimage.distance_transform_edt(hlpse2)
       
       local_maxi = peak_local_max(
           distance, 
           num_peaks = 6, 
           indices = False, 
           footprint = np.ones((50,50)),
           labels = hlpse2) #footprint = min dist between maxima to find #footprint was 25,25
       markers = skimage.morphology.label(local_maxi)
       labels_ws = watershed(-distance, markers, mask=hlpse2)
       lab = np.unique(labels_ws)
       
       #Make random colors, not degrade
   #    rand_ind = np.random.permutation(lab)
   #    new_labels_ws = np.copy(labels_ws)
   #    for k in range(new_labels_ws.shape[0]):
   #        for j in range(new_labels_ws.shape[1]):
   #            new_labels_ws[k, j] = rand_ind[labels_ws[k, j]]
   #    labels_ws =  new_labels_ws
       
       Pixel_size = np.array([0.74]) 
       length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[0]))
       sbar = sb.AnchoredScaleBar(ax00.transData, length_scalebar_in_pixels, "", style = 'dark', loc = 8, my_fontsize = fsizenb, my_linewidth= 2)
       ax0022.add_artist(sbar)
       
       areas = np.array([])
       for k in lab:
           areas = np.append(areas, len( labels_ws[labels_ws == k] ))
       cut_k = []
       cut_labels_ws = np.copy(labels_ws)
       non_cut_k = []  ###### change cut_k
       for k in range(len(lab)):
           if (areas[k] < 100) or (areas[k] > 4000):
               cut_labels_ws[cut_labels_ws == lab[k]] = 0
               cut_k.append(k)
           else:
               non_cut_k.append(k)  ###change cut_k
       
       #print(len(lab))
       #print(len(non_cut_k))
       #ax01.imshow(new_pic_grad,cmap=cm.Greys_r) #or 'OrRd'
       #ax01.axis('off')
       #ax02.imshow(masklarge,cmap=cm.Greys_r) #or 'OrRd'
       #ax02.axis('off')
      
       #code to make all black - this will crash if running to save data
       #cut_labels_ws[cut_labels_ws > 1] = 1.0
    
       axpic.imshow(cut_labels_ws, cmap = cm.Greys_r) #or 'OrRd'
       axpic.axis('off')
       
       del SEA, se
       gc.collect()
       
       #plt.show()    
       #end of registration code
       
            ####### load file that exists
   #    C = pickle.load( open( "PENTA.p", "rb" ) )
   #    areas = C['areas']
   #    taured = C['taured']
   #    tauredstd = C['tauredstd']
   #    taublue = C['taublue']
   #    taubluestd = C['taubluestd']
   #    intens = C['intens']
   #    stdintens = C['intensstd']
   #    intensr = C['intensr']
   #    stdintensr = C['intensstdr']
   #    intensb = C['intensb']
   #    stdintensb = C['intensstdb']
   #    non_cut_k = C['non_cut_k']  
       
       from guppy import hpy

       h = hpy()

       print h.heap()  

       pass

       #REDA = np.load('Try1Redbright.npz')
       REDA = np.load('Try1Redbright.npz', mmap_mode = 'r')
       reda = REDA['data'][:,:,xinit+cutx:xfinal-cutx,yinit+cutyinit:yfinal-cuty] #same no pixels than C, single
    
       #import IPython
       #IPython.embed()
   
       del REDA
       gc.collect()
       #BLUEA = np.load('Try1Bluebright.npz')
       BLUEA = np.load('Try1Bluebright.npz', mmap_mode = 'r')
       bluea = BLUEA['data'][:,:,xinit+cutx:xfinal-cutx,yinit+cutyinit:yfinal-cuty]#same no pixels than C, single
       del BLUEA
       gc.collect() 
       
       no_avg = reda.shape[0]
       intens = np.empty([len(lab),no_avg])
       stdintens = np.empty([len(lab),no_avg])
       intensr = np.empty([len(lab),no_avg])
       stdintensr = np.empty([len(lab),no_avg])
       intensb = np.empty([len(lab),no_avg])
       stdintensb = np.empty([len(lab),no_avg])
       
       taured = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
       taublue = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
       tauredstd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
       taubluestd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
       
       for k in non_cut_k:
   #    import random
   #    for k in random.sample(non_cut_k, 5):
        
          print('indicePENTA')
          print(k)
          
          #Ratio
          print('Ratio')
          hlp = np.zeros(cut_labels_ws.shape)
          hlp[cut_labels_ws == lab[k]] = 1.0
          hlp[cut_labels_ws != lab[k]] = np.nan
          ureda  =  reda[:,backgroundinit:initbin,:,:]
          ubluea =  bluea[:,backgroundinit:initbin,:,:]
                        
          vecr = np.nanmean(ureda * hlp, axis = (1,2,3))
          vecb = np.nanmean(ubluea * hlp, axis = (1,2,3))
          del ureda, ubluea
          gc.collect()
          
          vec = vecr/vecb
          
          No = np.nansum(hlp.astype(np.float64))*reda[:,backgroundinit:initbin,:,:].shape[1]
                 
          #import IPython
          #IPython.embed()
          
          vecstd = vec * np.sqrt( 1.0/vecr/No + 1.0/vecb/No )
          
          vecstdr = 1.0/np.sqrt(No) * np.sqrt(vecr) 
          vecstdb = 1.0/np.sqrt(No) * np.sqrt(vecb) 
          
          intens[k,:] = vec
          stdintens[k,:] = vecstd
          intensr[k,:] = vecr
          stdintensr[k,:] = vecstdr
          intensb[k,:] = vecb
          stdintensb[k,:] = vecstdb
          print(vec)
          print(vecstd)
          del vec, vecstd
          gc.collect()
          
        
          
          print('Taus')
          #Taus as a function of time
          hlp = np.zeros(cut_labels_ws.shape)
          hlp[cut_labels_ws == lab[k]] = 1.0
          Notr = np.sum(hlp.astype(np.float64))
          redd = np.sum(reda[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
          hr = tauestimate(redd,np.sqrt(redd)/np.sqrt(Notr))
          taured[k,:,:] = unumpy.nominal_values(hr)
          tauredstd[k,:,:] = unumpy.std_devs(hr)
          del hr, redd
          gc.collect()
          blued = np.sum(bluea[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
          hb = tauestimate(blued,np.sqrt(blued)/np.sqrt(Notr))
          taublue[k,:,:] = unumpy.nominal_values(hb)
          taubluestd[k,:,:] = unumpy.std_devs(hb)
          del hb, blued
          gc.collect()
            #pass
          
          
       del reda, bluea
       gc.collect()
       
       save_data = {}
       save_data['areas'] = areas*Pixel_size**2 #in nm^2
       save_data['taured'] = taured
       save_data['tauredstd'] = tauredstd
       save_data['taublue'] = taublue
       save_data['taubluestd'] = taubluestd
       save_data['intens'] = intens
       save_data['intensstd'] = stdintens
       save_data['intensr'] = intensr
       save_data['intensstdr'] = stdintensr
       save_data['intensb'] = intensb
       save_data['intensstdb'] = stdintensb
       save_data['non_cut_k'] = non_cut_k
       
       del taured, taublue
       gc.collect()
       
       pickle.dump(save_data, open("PENTA.p", "wb"))   
       
       print('radius found for C') #to check which of the found areas is background, which is signal
       print(tor(areas*Pixel_size**2))
       print(non_cut_k)
      # print('mean area of NP')
       #print(np.nanmean(C['areas'][C['non_cut_k']])/Pixel_size**2)
       
       #run until here
       klklklk
   
   do_300 = True #already ran, can just open files and read
   if do_300:
       
       print('do300')
       
       length_scalebar = 100.0
       Pixel_size = np.array([1.25]) 
       
       SEA= np.load('x300SEchannel.npz') #init shape (342, 315)
       xlen = SEA['data'].shape[0]
       ylen = SEA['data'].shape[1]
       xinit = 27
       xfinal = -27
       yinit = 29
       yfinal = -29
       se = SEA['data'][xinit:xfinal,yinit:yfinal]
       
       
       new_pic = give_bolinha('x300SEchannel.npz', xinit, yinit, xfinal, yfinal, corr_threshold = 0.35, n = 30, r = 8, save_file = False, do_plot = False)
       cutx = 50
       cuty = 1
       se = se[0:-cutx, cuty:-cuty]
   
       ax001.imshow(se,cmap=cm.Greys_r)
       ax001.axis('off')
       
       new_pic = new_pic[0:-cutx, cuty:-cuty]
       setr = new_pic
       #binary threshold
       se_data2 = np.copy(setr)
       
       new_hlp = new_pic
       I8 = (new_hlp * 255.9).astype(np.uint8)
       bw = cv2.adaptiveThreshold(I8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 0)
       from scipy import ndimage
       hlpse2 = bw 
       hlpse2[hlpse2 > 1] = 1.0
       hlpse2[hlpse2 < 1] = 0.0
       distance = ndimage.distance_transform_edt(hlpse2)
       
       local_maxi = peak_local_max(
           distance, 
           num_peaks = 50, 
           indices = False, 
           footprint = np.ones((50,50)),
           labels = hlpse2) #footprint = min dist between maxima to find #footprint was 25,25
       markers = skimage.morphology.label(local_maxi)
       labels_ws = watershed(-distance, markers, mask=hlpse2)
       lab = np.unique(labels_ws)
       
       # Make random colors, not degrade
   #    rand_ind = np.random.permutation(lab)
   #    new_labels_ws = np.copy(labels_ws)
   #    for k in range(new_labels_ws.shape[0]):
   #        for j in range(new_labels_ws.shape[1]):
   #            new_labels_ws[k, j] = rand_ind[labels_ws[k, j]]
   #    labels_ws =  new_labels_ws
       
       length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[0]))
       sbar = sb.AnchoredScaleBar(ax001.transData, length_scalebar_in_pixels, '', style = 'dark', loc = 4, my_fontsize = fsizenb, my_linewidth= 2)
       ax001.add_artist(sbar)
       
       areas = np.array([])
       for k in lab:
           areas = np.append(areas, len( labels_ws[labels_ws == k] ))
       cut_k = []
       cut_labels_ws = np.copy(labels_ws)
       non_cut_k = []  ###### change cut_k
       for k in range(len(lab)):
           if (areas[k] < 10) or (areas[k] > 4000):
               cut_labels_ws[cut_labels_ws == lab[k]] = 0
               cut_k.append(k)
           else:
               non_cut_k.append(k)  ###change cut_k
       
       #print(len(lab))
       #print(len(non_cut_k))
       #ax01.imshow(new_pic_grad,cmap=cm.Greys_r) #or 'OrRd'
       #ax01.axis('off')
       #ax02.imshow(masklarge,cmap=cm.Greys_r) #or 'OrRd'
       #ax02.axis('off')
      
       #code to make all black - this will crash if running to save data
       #cut_labels_ws[cut_labels_ws > 1] = 1.0
       
       ax011.imshow(cut_labels_ws, cmap = cm.Greys_r) #or 'OrRd'
       ax011.axis('off')
   
       del SEA, se
       gc.collect()
       
           ###### load file that exists
       C = pickle.load( open( "300.p", "rb" ) )
       areas = C['areas']
       taured = C['taured']
       tauredstd = C['tauredstd']
       taublue = C['taublue']
       taubluestd = C['taubluestd']
       intens = C['intens']
       stdintens = C['intensstd']
       intensr = C['intensr']
       stdintensr = C['intensstdr']
       intensb = C['intensb']
       stdintensb = C['intensstdb']
       non_cut_k = C['non_cut_k']  
       
       
   #    REDA = np.load('x300Redbright.npz')
   #    reda = REDA['data'][:,:,xinit+0:xfinal-cutx,yinit+cuty:yfinal-cuty] #same no pixels than C, single
   #    del REDA
   #    gc.collect()
   #    BLUEA = np.load('x300Bluebright.npz')
   #    bluea = BLUEA['data'][:,:,xinit+0:xfinal-cutx,yinit+cu def tauestimate(counts_red, error_red, counts_blue, error_blue):
   #    del BLUEA
   #    gc.collect() 
   #    
   #    no_avg = reda.shape[0]
   #    intens = np.empty([len(lab),no_avg])
   #    stdintens = np.empty([len(lab),no_avg])
   #    intensr = np.empty([len(lab),no_avg])
   #    stdintensr = np.empty([len(lab),no_avg])
   #    intensb = np.empty([len(lab),no_avg])
   #    stdintensb = np.empty([len(lab),no_avg])
   #    
   #    taured = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
   #    taublue = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
   #    tauredstd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
   #    taubluestd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
       
       for k in non_cut_k:
        
   #       print('indice300')
   #       print(k)
   #       
   #       #Ratio
   #       print('Ratio')
   #       hlp = np.zeros(cut_labels_ws.shape)
   #       hlp[cut_labels_ws == lab[k]] = 1.0
   #       hlp[cut_labels_ws != lab[k]] = np.nan
   #       ureda  =  reda[:,backgroundinit:initbin,:,:]
   #       ubluea =  bluea[:,backgroundinit:initbin,:,:]
   #                     
   #       vecr = np.nanmean(ureda * hlp, axis = (1,2,3))
   #       vecb = np.nanmean(ubluea * hlp, axis = (1,2,3))
   #       
   #       vec = vecr/vecb
   #       
   #       No = np.nansum(hlp.astype(np.float64))*reda[:,backgroundinit:initbin,:,:].shape[1]
   #              
   #       #import IPython
   #       #IPython.embed()
   #       
   #       vecstd = vec * np.sqrt( 1.0/vecr/No + 1.0/vecb/No )
   #       
   #       vecstdr = 1.0/np.sqrt(No) * np.sqrt(vecr) 
   #       vecstdb = 1.0/np.sqrt(No) * np.sqrt(vecb) 
   #       del ureda, ubluea
   #       gc.collect()
   #       intens[k,:] = vec
   #       stdintens[k,:] = vecstd
   #       intensr[k,:] = vecr
   #       stdintensr[k,:] = vecstdr
   #       intensb[k,:] = vecb
   #       stdintensb[k,:] = vecstdb
   #       print(vec)
   #       print(vecstd)
   #       del vec, vecstd, vecstdr, vecstdb, vecr, vecb
   #       gc.collect()
   #       
   #     
   #       
   #       print('Taus')
   #       #Taus as a function of time
   #       hlp = np.zeros(cut_labels_ws.shape)
   #       hlp[cut_labels_ws == lab[k]] = 1.0
   #       Notr = np.sum(hlp.astype(np.float64))
   #       redd = np.zeros([reda.shape[0],1398])
   #       blued = np.zeros([reda.shape[0],1398])
   #       for jj in np.arange(0,reda.shape[0]):
   #           redd[jj,:] = np.sum(reda[jj,initbin:,:,:] * hlp, axis = (1,2))/Notr
   #           blued[jj,:] = np.sum(bluea[jj,initbin:,:,:] * hlp, axis = (1,2))/Notr
   #       #redd = np.sum(reda[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
   #       #blued = np.sum(bluea[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
   #       hr = tauestimate(redd,np.sqrt(redd)/np.sqrt(Notr))
   #       taured[k,:,:] = unumpy.nominal_values(hr)
   #       tauredstd[k,:,:] = unumpy.std_devs(hr)
   #       hb = tauestimate(blued,np.sqrt(blued)/np.sqrt(Notr))
   #       taublue[k,:,:] = unumpy.nominal_values(hb)
   #       taubluestd[k,:,:] = unumpy.std_devs(hb)
   #       sizered = 1398 
   #       del hr, hb, redd, blued
   #       gc.collect()
          
   #           CONSIDER TAKING AVG OF NON ZERO ONES
               pass
   #           aux = np.copy(intens[k,:])
   #           aux[aux < 1.0e-12] = np.nan
   #           axratiomag.errorbar(4,np.nanmean(aux),yerr=np.nanstd(aux),marker='o', color='k', markersize=12) 
   #           axratiomag.set_xlim([0,10])
   #           movav = 50
   #           axtaumag.plot(moving_average(np.arange(0,1398),movav),moving_average(np.average(taured[k,:,:],axis=0),movav),lw=1,color='r')
   #           axtaumag.plot(moving_average(np.arange(0,1398),movav),moving_average(np.average(taublue[k,:,:],axis=0),movav),lw=1,color='g')
   #           axstdmag.plot(moving_average(np.arange(0,1398),movav),moving_average(np.std(taured[k,:,:],axis=0)/np.average(taured[k,:,:],axis=0),movav),lw=1,color='r')
   #           axstdmag.plot(moving_average(np.arange(0,1398),movav),moving_average(np.std(taublue[k,:,:],axis=0)/np.average(taublue[k,:,:],axis=0),movav),lw=1,color='g')
       
         
       axratiomag.errorbar(631,np.nanmean(C['intens'][C['non_cut_k']],axis=(0,1)),yerr=np.nanstd(C['intens'][C['non_cut_k']],axis=(0,1)),marker='o', color='k', markersize=12) 
   #    axtaumag.plot(np.arange(0,1398),np.average(C['taured'][C['non_cut_k']],axis=(0,1)),lw=2,color='r')
   #    axtaumag.plot(np.arange(0,1398),np.average(C['taublue'][C['non_cut_k']],axis=(0,1)),lw=2,color='g')
   #    axstdmag.plot(np.arange(0,1398),np.std(C['taured'][C['non_cut_k']],axis=(0,1))/np.average(C['taured'][C['non_cut_k']],axis=(0,1))*100.0,lw=2,color='r')
   #    axstdmag.plot(np.arange(0,1398),np.std(C['taublue'][C['non_cut_k']],axis=(0,1))/np.average(C['taublue'][C['non_cut_k']],axis=(0,1))*100.0,lw=2,color='g')
   #    
       
   #    print('no pixels 300kX')
   #    print(np.nanmean(C['areas'][C['non_cut_k']])/Pixel_size**2)
   #    klklklk
   
       # plotting histograms
       plot_ratio_hist(
               C['intens'][C['non_cut_k']],
               axhist300ratio,
               axexp300ratio,
               no_of_bins = 30,
               index = 1)
    
       plot_tau_hist(
               C['taured'][C['non_cut_k']],
               axhist300tau_blue,
               "", #axexp300tau_blue,
               which_taus = [250, 500, 1000],
               no_of_bins = 30,
               my_color = 'r',
               my_title= 'Red band ')
   
       plot_tau_hist(
               C['taublue'][C['non_cut_k']],
               axhist300tau_blue,
               "",#axexp300tau_blue,
               which_taus = [250, 500, 1000],
               no_of_bins = 30,
               my_color = 'g',
               my_title= 'Green band ')
   
   
   #    del reda, bluea
   #    gc.collect()
   #    
   #    save_data = {}
   #    save_data['areas'] = areas*Pixel_size**2 #in nm^2
   #    save_data['taured'] = taured
   #    save_data['tauredstd'] = tauredstd
   #    save_data['taublue'] = taublue
   #    save_data['taubluestd'] = taubluestd
   #    save_data['intens'] = intens
   #    save_data['intensstd'] = stdintens
   #    save_data['intensr'] = intensr
   #    save_data['intensstdr'] = stdintensr
   #    save_data['intensb'] = intensb
   #    save_data['intensstdb'] = stdintensb
   #    save_data['non_cut_k'] = non_cut_k
   #    
   #    del taured, taublue
   #    gc.collect()
   #    
   #    pickle.dump(save_data, open("300.p", "wb"))   
   #    
   #    print('radius found for C') #to check which of the found areas is background, which is signal
   ##    print(tor(areas[non_cut_k])) #if used saved areas
   #    print(tor(areas*Pixel_size**2))
   #    print(non_cut_k)
        
   #### C
   do_C = False #ran already - can just load data
   if do_C:
       
       print('do 450')
       
       length_scalebar = 100.0
       Pixel_size = np.array([0.83]) 
       
       SEA= np.load('x450SEchannel.npz') #init shape (342, 315)
       xlen = SEA['data'].shape[0]
       ylen = SEA['data'].shape[1]
       xinit = 200
       xfinal = -200
       yinit = 41
       yfinal = -41
       se = SEA['data'][xinit:xfinal,yinit:yfinal]
       
       new_pic = give_bolinha('x450SEchannel.npz', xinit, yinit, xfinal, yfinal, corr_threshold = 0.35, n = 30, r = 8, save_file = False, do_plot = False)
   #    cutx = 50
   #    cuty = 1
   #    se = se[0:-cutx, cuty:-cuty]
   
       ax0022.imshow(se,cmap=cm.Greys_r)
       ax0022.axis('off')
       
       #new_pic = new_pic[cutx:-cutx, cuty:-cuty]
       setr = new_pic
       #binary threshold
       se_data2 = np.copy(setr)
       
       new_hlp = new_pic
       I8 = (new_hlp * 255.9).astype(np.uint8)
       bw = cv2.adaptiveThreshold(I8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 0)
       from scipy import ndimage
       hlpse2 = bw 
       hlpse2[hlpse2 > 1] = 1.0
       hlpse2[hlpse2 < 1] = 0.0
       distance = ndimage.distance_transform_edt(hlpse2)
       
       local_maxi = peak_local_max(
           distance, 
           num_peaks = 50, 
           indices = False, 
           footprint = np.ones((50,50)),
           labels = hlpse2) #footprint = min dist between maxima to find #footprint was 25,25
       markers = skimage.morphology.label(local_maxi)
       labels_ws = watershed(-distance, markers, mask=hlpse2)
       lab = np.unique(labels_ws)
       
       # Make random colors, not degrade
       rand_ind = np.random.permutation(lab)
       new_labels_ws = np.copy(labels_ws)
       for k in range(new_labels_ws.shape[0]):
           for j in range(new_labels_ws.shape[1]):
               new_labels_ws[k, j] = rand_ind[labels_ws[k, j]]
       labels_ws =  new_labels_ws
       
       length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[0]))
       sbar = sb.AnchoredScaleBar(ax0022.transData, length_scalebar_in_pixels, '', style = 'bright', loc = 4, my_fontsize = fsizenb)
       ax0022.add_artist(sbar)
       
       areas = np.array([])
       for k in lab:
           areas = np.append(areas, len( labels_ws[labels_ws == k] ))
       cut_k = []
       cut_labels_ws = np.copy(labels_ws)
       non_cut_k = []  ###### change cut_k
       for k in range(len(lab)):
           if (areas[k] < 10) or (areas[k] > 20000): # or (k == 0): #k==o is modif!!! k==0 point is WEIRD
               cut_labels_ws[cut_labels_ws == lab[k]] = 0
               cut_k.append(k)
           else:
               non_cut_k.append(k)  ###change cut_k
       
       #print(len(lab))
       #print(len(non_cut_k))
       #ax01.imshow(new_pic_grad,cmap=cm.Greys_r) #or 'OrRd'
       #ax01.axis('off')
       #ax02.imshow(masklarge,cmap=cm.Greys_r) #or 'OrRd'
       #ax02.axis('off')
      
       #code to make all black - this will crash if running to save data
       #cut_labels_ws[cut_labels_ws > 1] = 1.0
       
       axpic.imshow(cut_labels_ws, cmap = cm.Greys_r) #or 'OrRd'
       axpic.axis('off')
   
       del SEA, se
       gc.collect()
    
   #### C
   
       
   #### C
   do_750= True #ran already - can just load data
   if do_750:
       
       print('do750')
       
       length_scalebar = 100.0
       Pixel_size = np.array([1.48]) 
       
       SEA= np.load('x750SEchannel.npz') #init shape (342, 315)
       xlen = SEA['data'].shape[0]
       ylen = SEA['data'].shape[1]
       xinit = 17
       xfinal = -17
       yinit = 18
       yfinal = -18
       se = SEA['data'][xinit:xfinal,yinit:yfinal]
       
       new_pic = give_bolinha('x750SEchannel.npz', xinit, yinit, xfinal, yfinal, corr_threshold = 0.35, n = 30, r = 8, save_file = False, do_plot = False)
   #    cutx = 20
   #    cuty = 1
       #se = se[0:-cutx, cuty:-cuty]
   
       ax0022b.imshow(se,cmap=cm.Greys_r)
       ax0022b.axis('off')
       
       #new_pic = new_pic[0:-cutx, cuty:-cuty]
       setr = new_pic
       #binary threshold
       se_data2 = np.copy(setr)
       
       new_hlp = new_pic
       I8 = (new_hlp * 255.9).astype(np.uint8)
       bw = cv2.adaptiveThreshold(I8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 0)
       from scipy import ndimage
       hlpse2 = bw 
       hlpse2[hlpse2 > 1] = 1.0
       hlpse2[hlpse2 < 1] = 0.0
       distance = ndimage.distance_transform_edt(hlpse2)
       
       local_maxi = peak_local_max(
           distance, 
           num_peaks = 50, 
           indices = False, 
           footprint = np.ones((50,50)),
           labels = hlpse2) #footprint = min dist between maxima to find #footprint was 25,25
       markers = skimage.morphology.label(local_maxi)
       labels_ws = watershed(-distance, markers, mask=hlpse2)
       lab = np.unique(labels_ws)
       
       # Make random colors, not degrade
       rand_ind = np.random.permutation(lab)
       new_labels_ws = np.copy(labels_ws)
       for k in range(new_labels_ws.shape[0]):
           for j in range(new_labels_ws.shape[1]):
               new_labels_ws[k, j] = rand_ind[labels_ws[k, j]]
       labels_ws =  new_labels_ws
       
       length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[0]))
       sbar = sb.AnchoredScaleBar(ax0022b.transData, length_scalebar_in_pixels, "", style = 'bright', loc = 8, my_fontsize = fsizenb, my_linewidth= 2)
       ax0022b.add_artist(sbar)
       
       areas = np.array([])
       for k in lab:
           areas = np.append(areas, len( labels_ws[labels_ws == k] ))
       cut_k = []
       cut_labels_ws = np.copy(labels_ws)
       non_cut_k = []  ###### change cut_k
       for k in range(len(lab)):
           if (areas[k] < 10) or (areas[k] > 4000):# or (k == 0): 
               cut_labels_ws[cut_labels_ws == lab[k]] = 0
               cut_k.append(k)
           else:
               non_cut_k.append(k)  ###change cut_k
       
       axpicb.imshow(cut_labels_ws, cmap = cm.Greys_r) #or 'OrRd'
       axpicb.axis('off')
       
   #    plt.show()
   #    
   #    import IPython
   #    IPython.embed()
       
       del SEA, se
       gc.collect()
       
   #==============================================================================
   #     ####### load file that exists
   #     C = pickle.load( open( "750.p", "rb" ) )
   #     areas = C['areas']
   #     taured = C['taured']
   #     tauredstd = C['tauredstd']
   #     taublue = C['taublue']
   #     taubluestd = C['taubluestd']
   #     intens = C['intens']
   #     stdintens = C['intensstd']
   #     intensr = C['intensr']
   #     stdintensr = C['intensstdr']
   #     intensb = C['intensb']
   #     stdintensb = C['intensstdb']
   #     non_cut_k = C['non_cut_k']  
   #     
   #     
   # #    REDA = np.load('x750Redbright.npz')
   # #    reda = REDA['data'][:,:,xinit:xfinal,yinit:yfinal] #same no pixels than C, single
   # #    del REDA
   # #    gc.collect()
   # #    BLUEA = np.load('x750Bluebright.npz')
   # #    bluea = BLUEA['data'][:,:,xinit:xfinal,yinit:yfinal]#same no pixels than C, single
   # #    del BLUEA
   # #    gc.collect() 
   # #    
   # #    no_avg = reda.shape[0]
   # #    intens = np.empty([len(lab),no_avg])
   # #    stdintens = np.empty([len(lab),no_avg])
   # #    intensr = np.empty([len(lab),no_avg])
   # #    stdintensr = np.empty([len(lab),no_avg])
   # #    intensb = np.empty([len(lab),no_avg])
   # #    stdintensb = np.empty([len(lab),no_avg])
   # #    
   # #    taured = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
   # #    taublue = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
   # #    tauredstd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
   # #    taubluestd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
   #     
   #     for k in non_cut_k:
   #      
   # #       print('indice750')
   # #       print(k)
   # #       
   # #       #Ratio
   # #       print('Ratio')
   # #       hlp = np.zeros(cut_labels_ws.shape)
   # #       hlp[cut_labels_ws == lab[k]] = 1.0
   # #       hlp[cut_labels_ws != lab[k]] = np.nan
   # #       ureda  =  reda[:,backgroundinit:initbin,:,:]
   # #       ubluea =  bluea[:,backgroundinit:initbin,:,:]
   # #                     
   # #       vecr = np.nanmean(ureda * hlp, axis = (1,2,3))
   # #       vecb = np.nanmean(ubluea * hlp, axis = (1,2,3))
   # #       
   # #       vec = vecr/vecb
   # #       
   # #       No = np.nansum(hlp.astype(np.float64))*reda[:,backgroundinit:initbin,:,:].shape[1]
   # #              
   # #       #import IPython
   # #       #IPython.embed()
   # #       
   # #       vecstd = vec * np.sqrt( 1.0/vecr/No + 1.0/vecb/No )
   # #       
   # #       vecstdr = 1.0/np.sqrt(No) * np.sqrt(vecr) 
   # #       vecstdb = 1.0/np.sqrt(No) * np.sqrt(vecb) 
   # #       del ureda, ubluea
   # #       gc.collect()
   # #       intens[k,:] = vec
   # #       stdintens[k,:] = vecstd
   # #       intensr[k,:] = vecr
   # #       stdintensr[k,:] = vecstdr
   # #       intensb[k,:] = vecb
   # #       stdintensb[k,:] = vecstdb
   # #       print('vec,std')
   # #       print(vec)
   # #       print(vecstd)
   # #       del vec, vecstd
   # #       gc.collect()
   # #       
   # #     
   # #       
   # #       print('Taus')
   # #       #Taus as a function of time
   # #       hlp = np.zeros(cut_labels_ws.shape)
   # #       hlp[cut_labels_ws == lab[k]] = 1.0
   # #       Notr = np.sum(hlp.astype(np.float64))
   # #       redd = np.sum(reda[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
   # #       blued = np.sum(bluea[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
   # #       hr = tauestimate(redd,np.sqrt(redd)/np.sqrt(Notr))
   # #       taured[k,:,:] = unumpy.nominal_values(hr)
   # #       tauredstd[k,:,:] = unumpy.std_devs(hr)
   # #       hb = tauestimate(blued,np.sqrt(blued)/np.sqrt(Notr))
   # #       taublue[k,:,:] = unumpy.nominal_values(hb)
   # #       taubluestd[k,:,:] = unumpy.std_devs(hb)
   # #       sizered = 1398 
   # #       del hr, hb, redd, blued
   # #       gc.collect()
   #        
   #            pass
   #     
   #     print(np.nanmean(C['intens'][C['non_cut_k']],axis=(0,1)))
   # #    axratiomag.plot(4,4.89,marker='o', color='k', markersize=12) 
   #     axratiomag.plot(4,np.nanmean(C['intens'][C['non_cut_k']],axis=(0,1)),marker='o', color='k', markersize=12) 
   #     axstdmag.plot(np.arange(0,1398),np.std(C['taured'][C['non_cut_k']],axis=(0,1))/np.average(C['taured'][C['non_cut_k']],axis=(0,1))*100.0,lw=1,color='r')
   #==============================================================================
   #    axstdmag.plot(np.arange(0,1398),np.std(C['taublue'][C['non_cut_k']],axis=(0,1))/np.average(C['taublue'][C['non_cut_k']],axis=(0,1))*100.0,lw=1,color='g')      
             
   #    del reda, bluea
   #    gc.collect()
   #    
   #    save_data = {}
   #    save_data['areas'] = areas*Pixel_size**2 #in nm^2
   #    save_data['taured'] = taured
   #    save_data['tauredstd'] = tauredstd
   #    save_data['taublue'] = taublue
   #    save_data['taubluestd'] = taubluestd
   #    save_data['intens'] = intens
   #    save_data['intensstd'] = stdintens
   #    save_data['intensr'] = intensr
   #    save_data['intensstdr'] = stdintensr
   #    save_data['intensb'] = intensb
   #    save_data['intensstdb'] = stdintensb
   #    save_data['non_cut_k'] = non_cut_k
   #    
   #    del taured, taublue
   #    gc.collect()
   #    
   #    pickle.dump(save_data, open("750.p", "wb"))   
   #    
   #    print('radius found for C') #to check which of the found areas is background, which is signal
   #    print(tor(areas*Pixel_size**2))
   #    print(non_cut_k)
   
             
   do_750BG= True#ran already - can just load data
   if do_750BG:
       
       print('do750BG')
       
       length_scalebar = 100.0
       Pixel_size = np.array([1.48]) 
       
       SEA= np.load('x750SEchannel.npz') #init shape (342, 315)
       xlen = SEA['data'].shape[0]
       ylen = SEA['data'].shape[1]
       xinit = 17
       xfinal = -17
       yinit = 18
       yfinal = -18
       se = SEA['data'][xinit:xfinal,yinit:yfinal]
       
       new_pic = give_bolinha('x750SEchannel.npz', xinit, yinit, xfinal, yfinal, corr_threshold = 0.35, n = 30, r = 8, save_file = False, do_plot = False)
   #    cutx = 20
   #    cuty = 1
       #se = se[0:-cutx, cuty:-cuty]
   
       #new_pic = new_pic[0:-cutx, cuty:-cuty]
       setr = new_pic
       #binary threshold
       se_data2 = np.copy(setr)
       
       new_hlp = new_pic
       I8 = (new_hlp * 255.9).astype(np.uint8)
       bw = cv2.adaptiveThreshold(I8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 0)
       from scipy import ndimage
       hlpse2 = bw 
       hlpse2[hlpse2 > 1] = 1.0
       hlpse2[hlpse2 < 1] = 0.0
       distance = ndimage.distance_transform_edt(hlpse2)
       
       local_maxi = peak_local_max(
           distance, 
           num_peaks = 50, 
           indices = False, 
           footprint = np.ones((50,50)),
           labels = hlpse2) #footprint = min dist between maxima to find #footprint was 25,25
       markers = skimage.morphology.label(local_maxi)
       labels_ws = watershed(-distance, markers, mask=hlpse2)
       lab = np.unique(labels_ws)
       
       # Make random colors, not degrade
   #    rand_ind = np.random.permutation(lab)
   #    new_labels_ws = np.copy(labels_ws)
   #    for k in range(new_labels_ws.shape[0]):
   #        for j in range(new_labels_ws.shape[1]):
   #            new_labels_ws[k, j] = rand_ind[labels_ws[k, j]]
   #    labels_ws =  new_labels_ws
       
       
       areas = np.array([])
       for k in lab:
           areas = np.append(areas, len( labels_ws[labels_ws == k] ))
       cut_k = []
       cut_labels_ws = np.copy(labels_ws)
       non_cut_k = []  ###### change cut_k
       for k in range(len(lab)):
           if (areas[k] < 4000):# or (k == 0): 
               cut_labels_ws[cut_labels_ws == lab[k]] = 0
               cut_k.append(k)
           else:
               non_cut_k.append(k)  ###change cut_k
   
       del SEA, se
       gc.collect()
       
   #==============================================================================
   #     ####### load file that exists
   #     C = pickle.load( open( "750BG.p", "rb" ) )
   #     areas = C['areas']
   #     taured = C['taured']
   #     tauredstd = C['tauredstd']
   #     taublue = C['taublue']
   #     taubluestd = C['taubluestd']
   #     intens = C['intens']
   #     stdintens = C['intensstd']
   #     intensr = C['intensr']
   #     stdintensr = C['intensstdr']
   #     intensb = C['intensb']
   #     stdintensb = C['intensstdb']
   #     non_cut_k = C['non_cut_k']  
   #     
   #     
   # #    REDA = np.load('x750Redbright.npz')
   # #    reda = REDA['data'][:,:,xinit:xfinal,yinit:yfinal] #same no pixels than C, single
   # #    del REDA
   # #    gc.collect()
   # #    BLUEA = np.load('x750Bluebright.npz')
   # #    bluea = BLUEA['data'][:,:,xinit:xfinal,yinit:yfinal]#same no pixels than C, single
   # #    del BLUEA
   # #    gc.collect() 
   # #    
   # #    no_avg = reda.shape[0]
   # #    intens = np.empty([len(lab),no_avg])
   # #    stdintens = np.empty([len(lab),no_avg])
   # #    intensr = np.empty([len(lab),no_avg])
   # #    stdintensr = np.empty([len(lab),no_avg])
   # #    intensb = np.empty([len(lab),no_avg])
   # #    stdintensb = np.empty([len(lab),no_avg])
   # #    
   # #    taured = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
   # #    taublue = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
   # #    tauredstd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
   # #    taubluestd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
   # #    
   #     for k in non_cut_k:
   # #     
   # #       print('indice750')
   # #       print(k)
   # #       
   # ##Ratio
   # #       print('Ratio')
   # #       hlp = np.zeros(cut_labels_ws.shape)
   # #       hlp[cut_labels_ws == lab[k]] = 1.0
   # #       hlp[cut_labels_ws != lab[k]] = np.nan
   # #       ureda  =  reda[:,backgroundinit:initbin,:,:]
   # #       ubluea =  bluea[:,backgroundinit:initbin,:,:]
   # #                     
   # #       vecr = np.nanmean(ureda * hlp, axis = (1,2,3))
   # #       vecb = np.nanmean(ubluea * hlp, axis = (1,2,3))
   # #       
   # #       vec = vecr/vecb
   # #       
   # #       No = np.nansum(hlp.astype(np.float64))*reda[:,backgroundinit:initbin,:,:].shape[1]
   # #              
   # #       #import IPython
   # #       #IPython.embed()
   # #       
   # #       vecstd = vec * np.sqrt( 1.0/vecr/No + 1.0/vecb/No )
   # #       
   # #       vecstdr = 1.0/np.sqrt(No) * np.sqrt(vecr) 
   # #       vecstdb = 1.0/np.sqrt(No) * np.sqrt(vecb) 
   # #       del ureda, ubluea
   # #       gc.collect()
   # #       intens[k,:] = vec
   # #       stdintens[k,:] = vecstd
   # #       intensr[k,:] = vecr
   # #       stdintensr[k,:] = vecstdr
   # #       intensb[k,:] = vecb
   # #       stdintensb[k,:] = vecstdb
   # #       print(vec)
   # #       print(vecstd)
   # #       del vec, vecstd
   # #       gc.collect()
   # #       
   # #       print('Taus')
   # #       #Taus as a function of time
   # #       hlp = np.zeros(cut_labels_ws.shape)
   # #       hlp[cut_labels_ws == lab[k]] = 1.0
   # #       Notr = np.sum(hlp.astype(np.float64))
   # #       redd = np.sum(reda[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
   # #       blued = np.sum(bluea[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
   # #       hr = tauestimate(redd,np.sqrt(redd)/np.sqrt(Notr))
   # #       taured[k,:,:] = unumpy.nominal_values(hr)
   # #       tauredstd[k,:,:] = unumpy.std_devs(hr)
   # #       hb = tauestimate(blued,np.sqrt(blued)/np.sqrt(Notr))
   # #       taublue[k,:,:] = unumpy.nominal_values(hb)
   # #       taubluestd[k,:,:] = unumpy.std_devs(hb)
   # #       sizered = 1398 
   # #       del hr, hb, redd, blued
   # #       gc.collect()
   #        
   #      pass
   #           
   #     print(np.nanmean(C['intens'][C['non_cut_k']],axis=(0,1)))
   #==============================================================================
   #    axratiomag.plot(4,6.06,marker='x', color='k', markersize=12) 
   #    axratiomag.plot(4,np.nanmean(C['intens'][C['non_cut_k']],axis=(0,1)),marker='x', color='k', markersize=12) 
   
   #    del reda, bluea
   #    gc.collect()
   #    
   #    save_data = {}
   #    save_data['areas'] = areas*Pixel_size**2 #in nm^2
   #    save_data['taured'] = taured
   #    save_data['tauredstd'] = tauredstd
   #    save_data['taublue'] = taublue
   #    save_data['taubluestd'] = taubluestd
   #    save_data['intens'] = intens
   #    save_data['intensstd'] = stdintens
   #    save_data['intensr'] = intensr
   #    save_data['intensstdr'] = stdintensr
   #    save_data['intensb'] = intensb
   #    save_data['intensstdb'] = stdintensb
   #    save_data['non_cut_k'] = non_cut_k
   #    
   #    del taured, taublue
   #    gc.collect()
   #    
   #    pickle.dump(save_data, open("750BG.p", "wb"))   
   #    
   #    print('radius found for C') #to check which of the found areas is background, which is signal
   #    print(tor(areas*Pixel_size**2))
   #    print(non_cut_k)
   
   
       
   ##### Plot single and hexagon
   #   print('radius found for C')
   #   print(tor(areas[non_cut_k]))
   #    se_data2 = np.copy(se)    
   #    tv_filt = denoise_tv_chambolle(se_data2, weight=0.1) #WAS0.5
   #    grad_x, grad_y = np.gradient(tv_filt)
   #    new_pic_grad = np.sqrt(grad_x**2 + grad_y**2)
   #    new_pic_grad = canny(togs(new_pic_grad), sigma = 3) #WAS 2 #uncomment and run in Boes's compi
   #    hex_angle = 18
   #    hex_angle2 = 28 #18#35
   #    axpic.imshow(new_pic_grad,cmap=cm.Greys)
   #    axpic2.imshow(1-cut_labels_ws,cmap=cm.Greys) #WAS IMSHOWING new_pic_grad
   #    #axpic2.add_patch(patches.RegularPolygon((80, 70), 6, 60, fill = True, facecolor = '#ff6666', edgecolor='None', orientation = hex_angle * np.pi/180.0, alpha=0.5))
   #    print(tor(areas[non_cut_k]))
   #    #TO PLOT HEXAGON NEEDS TO muLTIPLY BY PIXEL AREA!!! IS IT?
   #    axpic2.add_patch(patches.RegularPolygon((80, 70), 6, tor(areas[non_cut_k])*0.74, fill = True, facecolor = '#ff6666', edgecolor='None', orientation = hex_angle2 * np.pi/180.0, alpha=0.5))
   #    axpic.axis('off')
   #    axpic2.axis('off')
       
   ###### 750kx SINGLE
       
   ##### TAUS
       
   blue = np.load('../2017-04-05_Andrea_NPs_single/Blue_decay_arrayPixel.npz',mmap_mode='r')     
   red = np.load('../2017-04-05_Andrea_NPs_single/Red_decay_arrayPixel.npz',mmap_mode='r')
   bgblue = np.load('../2017-04-05_Andrea_NPs_single/bgBlue_decay_arrayPixel.npz',mmap_mode='r')     
   bgred = np.load('../2017-04-05_Andrea_NPs_single/bgRed_decay_arrayPixel.npz',mmap_mode='r') 
   
   red = red['data']
   blue = blue['data']
   bgred = bgred['data']
   bgblue = bgblue['data']  
   
   nopix = 1693.0
   nopixbg = 4469.0 
   
   error_red = np.sqrt(red)/np.sqrt(50*nopix)
   error_blue = np.sqrt(blue)/np.sqrt(50*nopix)
   
   error_redbg = np.sqrt(bgred)/np.sqrt(50*nopixbg)
   error_bluebg = np.sqrt(bgblue)/np.sqrt(50*nopixbg)
   
   taured, taublue = tauestimate2(red,error_red,blue,error_blue)
   bgtaured, bgtaublue = tauestimate2(bgred,error_redbg,bgblue,error_bluebg)
   
   plotinho(axtaumag, taured,'r',my_edgecolor='#ff3232', my_facecolor='#ff6666')
   plotinho(axtaumag, taublue,'g',my_edgecolor='#74C365', my_facecolor='#74C365')
   
   plotinho(axtaumag, bgtaured,'DarkRed',my_edgecolor='#801515', my_facecolor='#801515')
   plotinho(axtaumag, bgtaublue,'#003100',my_edgecolor='#003D1B', my_facecolor='#003D1B')
   
   hlpplot_1 = unumpy.std_devs(taured)/unumpy.nominal_values(taured)
   hlpplot_2 = unumpy.std_devs(taublue)/unumpy.nominal_values(taublue)
   
   hlpplot_1 = hlpplot_1[0]
   hlpplot_2 = hlpplot_2[0]
   
   axstdmag.plot(hlpplot_1*100.0,'r',linewidth=2)
   axstdmag.plot(hlpplot_2*100.0,'g',linewidth=2)
   axstdmag.set_ylabel(r'$\sigma_{\tau}$/$\tau$ ($\%$)',fontsize=fsizepl)
   axstdmag.set_xlabel('Acquisition time ($\mu$s)',fontsize=fsizepl)
   axstdmag.set_xticks([500,1000])
   axstdmag.set_yticks([0.1, 0.2])
   axstdmag.set_ylim([0,0.22])
   
   ####### INTENSITY
   
   blue = np.load('../2017-04-05_Andrea_NPs_single/Blue_int_arrayPixel.npz',mmap_mode='r')     
   red = np.load('../2017-04-05_Andrea_NPs_single/Red_int_arrayPixel.npz',mmap_mode='r')
   bgblue = np.load('../2017-04-05_Andrea_NPs_single/bgBlue_int_arrayPixel.npz',mmap_mode='r')     
   bgred = np.load('../2017-04-05_Andrea_NPs_single/bgRed_int_arrayPixel.npz',mmap_mode='r') 
   
   axratiomag.plot(1693,red['data']/blue['data'],marker='o', color='k', markersize=12) 
   axratiomag.plot(1693,bgred['data']/bgblue['data'],marker='x', color='k', markersize=12) 
   
   error_red = np.sqrt(red['data'])/np.sqrt(50*nopix)
   error_blue = np.sqrt(blue['data'])/np.sqrt(50*nopix)
   
   axratiomag.set_xlim([0,2000])
   axratiomag.set_xticks([500,1000,1500])
   axratiomag.set_ylim([1.7,6.3])
   axratiomag.set_yticks([2,4,6])
   
   axhist150tau_blue.set_ylim([0,75])
   axhist150tau_blue.set_yticks([10,30,50,70])
   axhist300tau_blue.set_ylim([0,75])
   axhist300tau_blue.set_yticks([10,30,50,70])
   
   axtaumag.set_yticks([250,500])
   axtaumag.set_xticks([500,1000])
   
   ###### 750kx SINGLE END
   #plt.tight_layout()
   multipage_longer('Single.pdf',dpi=80)  
   
     
   plt.show()
   
   lklklk
   
   
   multipage_longer('Single.pdf',dpi=80)  


if __name__ == '__main__':
    my_test()

