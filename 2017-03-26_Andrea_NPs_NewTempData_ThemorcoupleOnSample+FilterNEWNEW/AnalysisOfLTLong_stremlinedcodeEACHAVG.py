#==============================================================================
# 1.6ms, 50 move, 150 excite, 1400 transient
# 1MHz clock rate (1mus timebins)
# 5 frames
# 150kX mag
# standard: 300pixels
# 10kV
# 30mum == 379pA
#with filter: 592 dicrhoic + 550/***32***nm in blue pmt + 650/54nm in red pmt, semrock brightline
#IL chanel config: using preamp from Supra room
#doing filter than amplifying
#preamplifier set to: channel A, floating input, filter at 5kHz, LP, slope 12, gain 1 x100, DC coupling
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
               
kv = [10]

#nominal Temps
nametr = ['2017-03-26-1237_ImageSequence__150.000kX_10.000kV_30mu_5',
          '2017-03-26-1344_ImageSequence__150.000kX_10.000kV_30mu_6',
          '2017-03-26-1455_ImageSequence__150.000kX_10.000kV_30mu_7',
          '2017-03-26-1555_ImageSequence__150.000kX_10.000kV_30mu_8',
          '2017-03-26-1707_ImageSequence__150.000kX_10.000kV_30mu_9',
          '2017-03-26-1802_ImageSequence__150.000kX_10.000kV_30mu_10']

Pixel_size =  2.48#nm
Ps = Pixel_size #pixel size in nm, the numbers above with round nm precision
No_experiments = 5*np.ones([6])

let = ['RT','N30','N40','N50','N60', 'N70']
temp = [24.9, 30.4, 39.75, 51.05, 60.05, 70.4]
tempstd = [0.1, 0.3, 0.25, 0.95, 0.35, 0.7]
######################################## Plot with dose for different apertures
##files below exist 

no_avg = 5

il_data = np.zeros([no_avg,len(nametr)])
il_data_std = np.zeros([no_avg,len(nametr)])

blue_int_array = np.zeros([no_avg,len(nametr)])
red_int_array = np.zeros([no_avg,len(nametr)])

blue_std_array = np.zeros([no_avg,len(nametr)])
red_std_array = np.zeros([no_avg,len(nametr)])

red_decay_array = np.zeros([no_avg,len(nametr),1398])
blue_decay_array = np.zeros([no_avg,len(nametr),1398])

bgblue_int_array = np.zeros([no_avg,len(nametr)])
bgred_int_array = np.zeros([no_avg,len(nametr)])

bgblue_std_array = np.zeros([no_avg,len(nametr)])
bgred_std_array = np.zeros([no_avg,len(nametr)])

bgred_decay_array = np.zeros([no_avg,len(nametr),1398])
bgblue_decay_array = np.zeros([no_avg,len(nametr),1398])

pisize =Pixel_size

listofindex =np.arange(0,len(nametr))#,11]


consider_whole_light = []; #[0,1,2,3,4,5] #0,1,2,3,4,5,6]

ax0 = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=1)
ax1 = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((2,3), (0,2), colspan=1, rowspan=1)
ax3= plt.subplot2grid((2,3), (1,0), colspan=1, rowspan=1)
ax4 = plt.subplot2grid((2,3), (1,1), colspan=1, rowspan=1)
ax5 = plt.subplot2grid((2,3), (1,2), colspan=1, rowspan=1)

axvec = [ax0, ax1,ax2,ax3,ax4,ax5]

#all cuts adjusted so that if one less pixel, loses contrast
xinit = [ 16,   5, 12,  22,  4,  13] 
xend =  [-16,  -5,-12, -22, -4, -13]
yinit = [ 12,   6,  1,  31,  2,  27] 
yend =  [-12,  -6, -1, -31, -2, -27]

il_data[:,0] = np.array([24.9,24.9,24.9,24.9,24.9])
il_data[:,1] = np.array([ 30.24231529 , 30.43283653 , 30.54502869 , 30.61672211 , 30.67356873])
il_data[:,2] = np.array([ 39.62433624 , 39.77347946 , 39.86793518 , 39.93108368 , 39.97883606])
il_data[:,3] = np.array([ 50.28258896 , 50.85370255 , 51.2881012 ,  51.61391068  , 51.85993195])
il_data[:,4] = np.array([ 59.79606628 , 59.98893356 , 60.12455368 , 60.27425385 , 60.38433838])
il_data[:,5] = np.array([ 69.90750122 , 70.30992889 , 70.62423706 , 70.83724213 , 71.00095367])

listofindex =np.arange(0,len(nametr))
for index in listofindex:
    
    print(index)
   # il_data[:,index] = temp[index]
    il_data_std[:,index] = tempstd[index]
    
    print('after il data')   
    
    print('before loading')
    
    #
    
    se = np.load(str(let[index]) +'SEchannel.npz',mmap_mode='r') 
    
    
    
    
    
    
    backgdinit = 50
    initbin = (150+50+3)-1

    print('after skimage')
    
    #################
    
    #to plot the pics, uncomment 5 next lines
    if True:
        #axvec[index].imshow(se['data'][xinit[index]:xend[index],yinit[index]:yend[index]],cmap=cm.Greys) #or 'OrRd'
        
        boe_hlp = se['data'][xinit[index]:xend[index],yinit[index]:yend[index]] 
       
        fft_hlp = np.fft.fft2(boe_hlp)

        #axvec[index].pcolor(np.log(np.abs(fft_hlp)))#, vmin = 3, vmax = 4)
        
        #cla test
        fft_hlp[225:240, 0:70] = 0
        fft_hlp[190:260, 25:30] = 0
        fft_hlp[10:90, 250:270] = 0
        fft_hlp[45:60, 230:280] = 0

#        
#        fft_hlp[220:246, 20:35] = 0
#        fft_hlp[220:246, 250:270] = 0
#                
#        fft_hlp[45:60, 250:270] = 0
#        fft_hlp[45:60, 20:35] = 0

        new_pic = np.fft.ifft2(fft_hlp)
        

        #axvec[index+1].pcolor(boe_hlp)#, vmin = 3, vmax = 4)
        #axvec[index].pcolor(new_pic)#, vmin = 3, vmax = 4)
        
        #plt.show()

        #axvec[index].pcolor(new_pic,cmap=cm.Greys) 
        
        
        print('after imshow')   
        #del segmm, red, blue,se
        gc.collect()
        
        red0 = np.load(str(let[index]) +'Redbright.npz',mmap_mode='r')  
        blue0 = np.load(str(let[index]) +'Bluebright.npz',mmap_mode='r') 
        
        red = red0['data'] 
        blue = blue0['data']
        del red0, blue0
        gc.collect()
        
        segmm, means, covars, weights = gmmone_tr_in_masked_channel_modif_memory_issue(np.abs(new_pic)) 
    ##############################################################
        red = red[:,:,xinit[index]:xend[index],yinit[index]:yend[index]]
        blue = blue[:,:,xinit[index]:xend[index],yinit[index]:yend[index]]
            
        del means, covars, weights
        gc.collect()
        
#        axvec[index].imshow(segmm,cmap=cm.Greys)
#        print(segmm)
#        print(np.nanmax(segmm))
#        print(np.nanmin(segmm))
#        del segmm#, red
#        gc.collect()
                    
#multipage_longer('Checkcuts.pdf',dpi=80)
#plt.show()
#multipage_longer('Checkfft.pdf',dpi=80)
#multipage_longer('Checksegmm.pdf',dpi=80)
#klklkk  
#if True:
     
     #INSIDE
    if index in consider_whole_light:
         hlp = 1.0 #outside, consider all light
    else:
         hlp = np.copy(segmm)
         hlp[~np.isnan(hlp)] = 1.0  #inside
     
     # OUTSIDE
    if index in consider_whole_light:
         hlpd  = 0.0 #consider all light
    else:
         hlpd = np.copy(segmm)
         hlpd[~np.isnan(hlpd)] = 0.0 
         hlpd[np.isnan(hlpd)] = 1.0
     
    datared = np.average(red, axis = (0))
    datablue = np.average(blue, axis = (0))
     
    if True is False:
         pass
    else:
         initbin = (150+50+3)-1 #init bin for decay
         backgdinit = 50
         ### 700ns /40ns = 7. ....
         datared_init = datared[0:backgdinit,:,:]
         datared = datared[initbin:,:,:]
         datablue_init = datablue[0:backgdinit,:,:]
         datablue = datablue[initbin:,:,:]

     
    del datared, datablue, datablue_init, datared_init
    gc.collect()
    
    dataALLred = red[:,:,:,:]
    dataALLblue = blue[:,:,:,:]
    
    #nominal_time_on = 150.0
    
    print('bef nanmean')
     
    red_int_array[:,index] = np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)) 
    gc.collect()
    print('1')
    blue_int_array[:,index] = np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)) 
    gc.collect()
    print('2')
     
    red_decay_array[:,index,:] = np.nanmean(dataALLred[:,initbin:,:,:]*hlp,axis=(2,3))
    gc.collect()
    print('3')
    blue_decay_array[:,index,:] = np.nanmean(dataALLblue[:,initbin:,:,:]*hlp,axis=(2,3))
    gc.collect()
    print('4')
     
    red_std_array[:,index] = np.nanstd(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)) 
    gc.collect()
    print('5')
    blue_std_array[:,index] = np.nanstd(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)) 
    gc.collect()
    print('6')
    
    bgred_int_array[:,index] = np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3)) 
    gc.collect()
    print('7')
    bgblue_int_array[:,index] = np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3)) 
    gc.collect()
    print('8')
     
    bgred_decay_array[:,index,:] = np.nanmean(dataALLred[:,initbin:,:,:]*hlpd,axis=(2,3))
    gc.collect()
    print('9')
    bgblue_decay_array[:,index,:] = np.nanmean(dataALLblue[:,initbin:,:,:]*hlpd,axis=(2,3))
    gc.collect()
    print('10')
     
    bgred_std_array[:,index] = np.nanstd(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3))
    gc.collect()
    print('11')
    bgblue_std_array[:,index] = np.nanstd(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3)) 
    gc.collect()
    
    print('after nanmean')
    
    del dataALLred, dataALLblue, red, blue, se, segmm
    gc.collect()
 
##### ONCE ALL FITS WORK, 
###### NEED TO RUN THESE LINES BELOW SO THAT ALL NPZ FILES ARE CREATED 
 
mycode = 'Il_dataEACHAVG = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Il_dataEACHAVG ', data = il_data)

mycode = 'Il_data_stdEACHAVG  = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Il_data_stdEACHAVG ', data = il_data_std)

#foreground
  
mycode = 'Red_std_arrayEACHAVG  = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Red_std_arrayEACHAVG ', data = red_std_array)
 
mycode = 'Blue_std_arrayEACHAVG  = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Blue_std_arrayEACHAVG ', data = blue_std_array)
 
mycode = 'Red_int_arrayEACHAVG  = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Red_int_arrayEACHAVG ', data = red_int_array)
 
mycode = 'Blue_int_arrayEACHAVG  = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Blue_int_arrayEACHAVG ', data = blue_int_array)
 
mycode = 'Red_decay_arrayEACHAVG  = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Red_decay_arrayEACHAVG ', data = red_decay_array)

mycode = 'Blue_decay_arrayEACHAVG  = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Blue_decay_arrayEACHAVG ', data = blue_decay_array)

###background

mycode = 'bgRed_std_arrayEACHAVG  = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgRed_std_arrayEACHAVG ', data = bgred_std_array)
 
mycode = 'bgBlue_std_arrayEACHAVG  = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgBlue_std_arrayEACHAVG ', data = bgblue_std_array)
 
mycode = 'bgRed_int_arrayEACHAVG  = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgRed_int_arrayEACHAVG ', data = bgred_int_array)
 
mycode = 'bgBlue_int_arrayEACHAVG  = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgBlue_int_arrayEACHAVG ', data = bgblue_int_array)

mycode = 'bgRed_decay_arrayEACHAVG  = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgRed_decay_arrayEACHAVG ', data =bgred_decay_array)

mycode = 'bgBlue_decay_arrayEACHAVG  = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgBlue_decay_arrayEACHAVG ', data = bgblue_decay_array)
    
kjjhjh
