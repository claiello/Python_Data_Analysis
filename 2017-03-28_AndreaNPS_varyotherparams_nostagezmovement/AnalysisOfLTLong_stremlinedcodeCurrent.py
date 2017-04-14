#==============================================================================
# current scan
# all 10kV, current confirmed with Optiprobe
#stage z NOT changed in between frames, NOT optimized for counts
#focus manually readjusted between kvs
#150kX, 1MHz clock, 300x300pixels, 5 avgs
#WITH FILTERS: 550/32 IN BLUE, 650/49 (STANDARD) IN RED
#==============================================================================

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
#from mpl_toolkits.axes_grid1 import make_axes_locatable
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
#from tifffile import *
#from sklearn.mixture import GMM 
import matplotlib.cm as cm
#from FluoDecay import *
#from PlottingFcts import *
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.misc
#import matplotlib.animation as animation
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

Time_bin = 1000#in ns; 
nominal_time_on = 150.0 #time during which e-beam nominally on, in mus
totaltrpoints = 1400 #total number of time-resolved points

description = ['Andrea small NaYF4:Er'] # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
No_experiments = 5*np.ones([8])
                  
nametr = ['2017-03-28-1037_ImageSequence__150.000kX_10.000kV_30mu_4',
          '2017-03-28-1059_ImageSequence__150.000kX_10.000kV_10mu_5',
          '2017-03-28-1118_ImageSequence__150.000kX_10.000kV_12mu_6',
          '2017-03-28-1137_ImageSequence__150.000kX_10.000kV_30mu_7',
          '2017-03-28-1217_ImageSequence__150.000kX_10.000kV_30mu_9',
          '2017-03-28-1258_ImageSequence__150.000kX_10.000kV_12mu_11',
          '2017-03-28-1337_ImageSequence__150.000kX_10.000kV_12mu_12',
          '2017-03-28-1358_ImageSequence__150.000kX_10.000kV_30mu_13']

description = ['Andrea small NaYF4:Er'] # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
kv = [379,48,9800,662,379,9800,6000,267] #current in pA

#nominal Temps
let = ['pA379','pA48','pA9800','pA662','pA379','pA9800','pA6000','pA267']

description = ['Andrea small NaYF4:Er'] # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
Pixel_size =  2.48#nm
Ps = Pixel_size #pixel size in nm, the numbers above with round nm precision

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

pisize =Pixel_size

listofindex =np.arange(0,len(nametr))#,11]


consider_whole_light = [] 

ax0 = plt.subplot2grid((2,4), (0,0), colspan=1, rowspan=1)
ax1 = plt.subplot2grid((2,4), (0,1), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((2,4), (0,2), colspan=1, rowspan=1)
ax2b = plt.subplot2grid((2,4), (0,3), colspan=1, rowspan=1)
ax3= plt.subplot2grid((2,4), (1,0), colspan=1, rowspan=1)
ax4 = plt.subplot2grid((2,4), (1,1), colspan=1, rowspan=1)
ax5 = plt.subplot2grid((2,4), (1,2), colspan=1, rowspan=1)
ax5b = plt.subplot2grid((2,4), (1,3), colspan=1, rowspan=1)

axvec = [ax0, ax1,ax2,ax2b, ax3,ax4,ax5, ax5b]

#those cuts are ok, any more and contrast falls down due to registration
xinit = [ 17,  0,  27,   0,  18,  27,   3,   6] 
xend =  [-17, -1, -27,  -1, -17, -27,  -3,  -6]
yinit = [ 11,  0,  18,   9,  11,  18,  21,  13] 
yend =  [-11, -1, -18,  -8, -10, -18, -21, -13]

listofindex =np.arange(0,len(nametr))
for index in listofindex:
    
    print(index)
    
    print('before loading')
    
    #
    red0 = np.load(str(let[index]) +'Redbright.npz',mmap_mode='r')  
    blue0 = np.load(str(let[index]) +'Bluebright.npz',mmap_mode='r')  
    se = np.load(str(let[index]) +'SEchannel.npz',mmap_mode='r') 
    
    red = red0['data']
    blue = blue0['data']
    del red0, blue0
    #
    gc.collect()
    
    segmm, means, covars, weights = gmmone_tr_in_masked_channel_modif_memory_issue(se['data'][xinit[index]:xend[index],yinit[index]:yend[index]], red[:,:,xinit[index]:xend[index],yinit[index]:yend[index]]) 
    #############################################################
    red = red[:,:,xinit[index]:xend[index],yinit[index]:yend[index]]
    blue = blue[:,:,xinit[index]:xend[index],yinit[index]:yend[index]]
    
    del means, covars, weights
    gc.collect()
    
    backgdinit = 50
    initbin = (150+50+3)-1

    print('after skimage')
    
    #################
    
#    #to plot the pics, uncomment 5 next lines
#    if True:
#        #axvec[index].imshow(se['data'][xinit[index]:xend[index],yinit[index]:yend[index]],cmap=cm.Greys) #or 'OrRd'
#        axvec[index].imshow(segmm,cmap=cm.Greys)
#        print('after imshow')   
#        del segmm, red, blue,se
#        gc.collect()
##multipage_longer('CheckcutsCurrent.pdf',dpi=80)
#multipage_longer('ChecksegmmCurrent.pdf',dpi=80)
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
#         hlpd = np.copy(segmm)
#         hlpd[~np.isnan(hlpd)] = 0.0 
#         hlpd[np.isnan(hlpd)] = 1.0
         #NEW VERSION
         hlpd = np.copy(segmm)
         hlpd[~np.isnan(hlpd)] = -5000.0 
         hlpd[np.isnan(hlpd)] = 1.0
         #added line
         hlpd[(hlpd == -5000.0)] = np.nan 
    
    dataALLred = red[:,:,:,:]
    dataALLblue = blue[:,:,:,:]
    
    nominal_time_on = 150.0
    
    print('bef nanmean')
     
    red_int_array[index] = np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) 
    gc.collect()
    print('1')
    blue_int_array[index] = np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) 
    gc.collect()
    print('2')
     
    red_decay_array[index,:] = np.nanmean(dataALLred[:,initbin:,:,:]*hlp,axis=(0,2,3))
    gc.collect()
    print('3')
    blue_decay_array[index,:] = np.nanmean(dataALLblue[:,initbin:,:,:]*hlp,axis=(0,2,3))
    gc.collect()
    print('4')
     
    red_std_array[index] = np.nanstd(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) 
    gc.collect()
    print('5')
    blue_std_array[index] = np.nanstd(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) 
    gc.collect()
    print('6')
    
    bgred_int_array[index] = np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3)) 
    gc.collect()
    print('7')
    bgblue_int_array[index] = np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3)) 
    gc.collect()
    print('8')
     
    bgred_decay_array[index,:] = np.nanmean(dataALLred[:,initbin:,:,:]*hlpd,axis=(0,2,3))
    gc.collect()
    print('9')
    bgblue_decay_array[index,:] = np.nanmean(dataALLblue[:,initbin:,:,:]*hlpd,axis=(0,2,3))
    gc.collect()
    print('10')
     
    bgred_std_array[index] = np.nanstd(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))
    gc.collect()
    print('11')
    bgblue_std_array[index] = np.nanstd(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3)) 
    gc.collect()
    
    print('after nanmean')
    
    del dataALLred, dataALLblue, segmm
    gc.collect()
 
##### ONCE ALL FITS WORK, 
###### NEED TO RUN THESE LINES BELOW SO THAT ALL NPZ FILES ARE CREATED 
 
#foreground
  
mycode = 'Red_std_arrayCurrent = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Red_std_arrayCurrent', data = red_std_array)
 
mycode = 'Blue_std_arrayCurrent = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Blue_std_arrayCurrent', data = blue_std_array)
 
mycode = 'Red_int_arrayCurrent = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Red_int_arrayCurrent', data = red_int_array)
 
mycode = 'Blue_int_arrayCurrent = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Blue_int_arrayCurrent', data = blue_int_array)
 
mycode = 'Red_decay_arrayCurrent = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Red_decay_arrayCurrent', data = red_decay_array)

mycode = 'Blue_decay_arrayCurrent = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Blue_decay_arrayCurrent', data = blue_decay_array)

###background

mycode = 'bgRed_std_arrayCurrent = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgRed_std_arrayCurrent', data = bgred_std_array)
 
mycode = 'bgBlue_std_arrayCurrent = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgBlue_std_arrayCurrent', data = bgblue_std_array)
 
mycode = 'bgRed_int_arrayCurrent = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgRed_int_arrayCurrent', data = bgred_int_array)
 
mycode = 'bgBlue_int_arrayCurrent = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgBlue_int_arrayCurrent', data = bgblue_int_array)

mycode = 'bgRed_decay_arrayCurrent = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgRed_decay_arrayCurrent', data =bgred_decay_array)

mycode = 'bgBlue_decay_arrayCurrent = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgBlue_decay_arrayCurrent', data = bgblue_decay_array)
    
kjjhjh
