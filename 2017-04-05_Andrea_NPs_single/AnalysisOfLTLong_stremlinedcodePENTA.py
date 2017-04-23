
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
from CreateDatasets import *
#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################
#######

#Time_bin = 1000#in ns; 
#nominal_time_on = 150.0 #time during which e-beam nominally on, in mus
#totaltrpoints = 1400 #total number of time-resolved points

#No_experiments = 1*np.ones([10]) #50 avgs
                  
nametr = ['test']

description = ['Andrea small NaYF4:Er'] 
               
let = ['Try1'] #no of pixels

######################################## Plot with dose for different apertures
##files below exist 

blue_int_array = np.zeros(len(nametr))
red_int_array = np.zeros(len(nametr))

blue_std_array = np.zeros(len(nametr))
red_std_array = np.zeros(len(nametr))

red_decay_array = np.zeros([len(nametr),1398])
blue_decay_array = np.zeros([len(nametr),1398])

bgblue_int_array = np.zeros(len(nametr))
bgred_int_array = np.zeros(len(nametr))

bgblue_std_array = np.zeros(len(nametr))
bgred_std_array = np.zeros(len(nametr))

bgred_decay_array = np.zeros([len(nametr),1398])
bgblue_decay_array = np.zeros([len(nametr),1398])

#pisize =Pixel_size

listofindex =np.arange(0,len(nametr))

consider_whole_light = [] 

ax0 = plt.subplot2grid((2,4), (0,0), colspan=1, rowspan=1)
ax1 = plt.subplot2grid((2,4), (0,1), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((2,4), (0,2), colspan=1, rowspan=1)
ax2b = plt.subplot2grid((2,4), (0,3), colspan=1, rowspan=1)
ax3= plt.subplot2grid((2,4), (1,0), colspan=1, rowspan=1)
ax4 = plt.subplot2grid((2,4), (1,1), colspan=1, rowspan=1)
ax5 = plt.subplot2grid((2,4), (1,2), colspan=1, rowspan=1)

axvec = [ax0, ax1,ax2,ax2b, ax3,ax4,ax5]

#those cuts are ok
#original was 100x100 pixels
#current size (10, 1600, 370,353)
xinit = [70 ] 
xend =  [-70 ]
yinit = [53] 
yend =  [-53 ]

listofindex =np.arange(0,len(nametr))

##done
#do_signal = True
#do_red = True

#do_signal = True
#do_red = False

#
#do_signal = False
#do_red = True

do_signal = False
do_red = False

for index in listofindex:
    
    print(index)
    
    print('before loading')
    
    if do_red == True:
        red0 = np.load(str(let[index]) +'Redbright.npz',mmap_mode='r')  
    se = np.load(str(let[index]) +'SEchannel.npz',mmap_mode='r') 
    
    if do_red == True:
        red = red0['data']
    
#    print(red.shape)
#    klll
    if do_red == True:
        del red0
        gc.collect()
    
    segmm, means, covars, weights = gmmone_tr_in_masked_channel_modif_memory_issue(se['data'][xinit[index]:xend[index],yinit[index]:yend[index]]) #, red[:,:,xinit[index]:xend[index],yinit[index]:yend[index]]) 
    #############################################################
    if do_red == True:
        red = red[:,:,xinit[index]:xend[index],yinit[index]:yend[index]]
   
    del means, covars, weights
    gc.collect()
    
    del se
    gc.collect()
    
    backgdinit = 50
    initbin = (150+50+3)-1

    print('after skimage')
    
    #################
    
    #to plot the pics, uncomment 5 next lines
#    if True:
       # axvec[index].imshow(se['data'][xinit[index]:xend[index],yinit[index]:yend[index]],cmap=cm.Greys) #or 'OrRd'
#        axvec[index].imshow(segmm,cmap=cm.Greys)
#        print('after imshow')   
# #       del segmm, red, blue,se
#  #      gc.collect()
#multipage_longer('CheckcutsPixelPENTA.pdf',dpi=80)
#multipage_longer('ChecksegmmPixelPENTA.pdf',dpi=80)
#klklkk  
#if True:
     
     #INSIDE
    if index in consider_whole_light:
         hlp = 1.0 #outside, consider all light
    else:
         hlp = np.copy(segmm)
         hlp[~np.isnan(hlp)] = 1.0  #inside
         #outside continues nan
     
     # OUTSIDE
    if index in consider_whole_light:
         hlpd  = 0.0 #consider all light
    else:
         hlpd = np.copy(segmm)
         hlpd[~np.isnan(hlpd)] = -5000.0 
         hlpd[np.isnan(hlpd)] = 1.0
         #added line
         hlpd[(hlpd == -5000.0)] = np.nan 
         
    del segmm
    gc.collect()
    
#    print('stats')
#    print(np.sum(np.isfinite(hlp))) #23719 SIGNAL pixels , counting all 5 NPS
#    print(np.sum(np.isfinite(hlpd))) #33091 BG  , counting all 5 NPS
#    print(red.shape)
#    kkkkkc
    
    #1
#    red_int_array[index] = np.nansum(red[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) / np.sum(np.isfinite(hlp))
#    gc.collect()
#    #2
#    red_decay_array[index,:] = np.sum(red[:,initbin:,:,:]*hlp,axis=(0,2,3))/ np.sum(np.isfinite(hlp))
#    gc.collect()
    #3
#    bgred_int_array[index] = np.nansum(red[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))  / np.sum(np.isfinite(hlpd))
#    gc.collect()
    #4
#    soma = np.sum(np.isfinite(hlpd))
#    hlpd[np.isnan(hlpd)] = 0.0
#    fonction = red[:,initbin:,:,:]*hlpd
#    del hlp, hlpd
#    a = np.sum(fonction, axis = 3)
#    del fonction
#    gc.collect()
#    b = np.sum(a, axis = 2)
#    del a
#    gc.collect()
#    c = np.sum(b, axis = 0)
#    del b
#    gc.collect()
#    bgred_decay_array[index,:] = c  / soma
#    gc.collect()
#    
    
#    print('red')
#    if (do_signal == True) & (do_red == True):
#        #red_int_array[index] = np.nanmean(red[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) 
#        red_int_array[index] = np.nansum(red[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) / np.sum(np.isfinite(hlp))
#        gc.collect()
#        #red_decay_array[index,:] = np.nanmean(red[:,initbin:,:,:]*hlp,axis=(0,2,3))
#        red_decay_array[index,:] = np.sum(red[:,initbin:,:,:]*hlp,axis=(0,2,3))/ np.sum(np.isfinite(hlp))
#        gc.collect()
#        #red_std_array[index] = np.nanstd(red[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) 
#        #gc.collect()
#        #bgred_int_array[index] = np.nanmean(red[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3)) 
#    
#    if (do_signal == False) & (do_red == True):
#        bgred_int_array[index] = np.nansum(red[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))  / np.sum(np.isfinite(hlpd))
#        gc.collect()
#        #bgred_decay_array[index,:] = np.nanmean(red[:,initbin:,:,:]*hlpd,axis=(0,2,3))
#        bgred_decay_array[index,:] = np.nansum(red[:,initbin:,:,:]*hlpd,axis=(0,2,3))  / np.sum(np.isfinite(hlpd))
#        gc.collect()
#        #bgred_std_array[index] = np.nanstd(red[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))
#        #gc.collect()
    
    if do_red == True:
        del red
        gc.collect()
    
    if do_red == False:
        blue0 = np.load(str(let[index]) +'Bluebright.npz',mmap_mode='r')  
        blue = blue0['data']
        del blue0
        gc.collect()
        
        blue = blue[:,:,xinit[index]:xend[index],yinit[index]:yend[index]]
    
    print('blue')
    
    #1
#    blue_int_array[index] = np.nansum(blue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) / np.sum(np.isfinite(hlp))
#    gc.collect()
    #2
#    soma = np.sum(np.isfinite(hlp))
#    hlp[np.isnan(hlp)] = 0.0
#    blue_decay_array[index,:] = np.sum(blue[:,initbin:,:,:]*hlp,axis=(0,2,3)) / soma
#    gc.collect()
    #3
#    bgblue_int_array[index] = np.nansum(blue[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3)) / np.sum(np.isfinite(hlpd))
#    gc.collect()
    #4
    soma =  np.sum(np.isfinite(hlpd))
    hlpd[np.isnan(hlpd)] = 0.0
    bgblue_decay_array[index,:] = np.sum(blue[:,initbin:,:,:]*hlpd,axis=(0,2,3))/ soma
    gc.collect()
    
#    if  (do_signal == True) & (do_red == False):
#        #blue_int_array[index] = np.nanmean(blue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) 
#        blue_int_array[index] = np.nansum(blue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) / np.sum(np.isfinite(hlp))
#        gc.collect()
#        #blue_decay_array[index,:] = np.nanmean(blue[:,initbin:,:,:]*hlp,axis=(0,2,3))
#        blue_decay_array[index,:] = np.nansum(blue[:,initbin:,:,:]*hlp,axis=(0,2,3)) / np.sum(np.isfinite(hlp))
#        gc.collect()
#    #    blue_std_array[index] = np.nanstd(blue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) 
#    #    gc.collect()
#        #bgblue_int_array[index] = np.nanmean(blue[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3)) 
#        
#    if (do_signal == False) & (do_red == False):
#        bgblue_int_array[index] = np.nansum(blue[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3)) / np.sum(np.isfinite(hlpd))
#        gc.collect()
#        #bgblue_decay_array[index,:] = np.nanmean(blue[:,initbin:,:,:]*hlpd,axis=(0,2,3))
#        bgblue_decay_array[index,:] = np.nansum(blue[:,initbin:,:,:]*hlpd,axis=(0,2,3))/ np.sum(np.isfinite(hlpd))
#        gc.collect()
#    #    bgblue_std_array[index] = np.nanstd(blue[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3)) 
#    #    gc.collect()

    if do_red == False:
        del blue
        gc.collect()
 
#foreground
  
#mycode = 'Red_std_arrayPENTA = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Red_std_arrayPENTA', data = red_std_array)
# 
#mycode = 'Blue_std_arrayPENTA = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Blue_std_arrayPENTA', data = blue_std_array)
  
if do_signal == True:
    pass
 
#    if do_red == True:
        #1
#        mycode = 'Red_int_arrayPENTA = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez('Red_int_arrayPENTA', data = red_int_array)
         
        #2
#        mycode = 'Red_decay_arrayPENTA = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez('Red_decay_arrayPENTA', data = red_decay_array)
    
#    if do_red == False:
    
        #3
#        mycode = 'Blue_int_arrayPENTA = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez('Blue_int_arrayPENTA', data = blue_int_array)
        
        #4
#        mycode = 'Blue_decay_arrayPENTA = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez('Blue_decay_arrayPENTA', data = blue_decay_array)

###background

#mycode = 'bgRed_std_arrayPENTA = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('bgRed_std_arrayPENTA', data = bgred_std_array)
# 
#mycode = 'bgBlue_std_arrayPENTA = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('bgBlue_std_arrayPENTA', data = bgblue_std_array)

if do_signal == False:
#    pass
    
#    if do_red == True:
 
        #1
#        mycode = 'bgRed_int_arrayPENTA = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez('bgRed_int_arrayPENTA', data = bgred_int_array)
        
        #2
#        mycode = 'bgRed_decay_arrayPENTA = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez('bgRed_decay_arrayPENTA', data =bgred_decay_array)
#    
#    if do_red == False:
        #3
#        mycode = 'bgBlue_int_arrayPENTA = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez('bgBlue_int_arrayPENTA', data = bgblue_int_array)
        
        #4
        mycode = 'bgBlue_decay_arrayPENTA = tempfile.NamedTemporaryFile(delete=False)'
        exec(mycode)
        np.savez('bgBlue_decay_arrayPENTA', data = bgblue_decay_array)
    
kjjhjh
