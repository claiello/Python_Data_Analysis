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
nametr = ['2017-03-26-1924_ImageSequence__150.000kX_10.000kV_30mu_1',
          '2017-03-26-1958_ImageSequence__150.000kX_10.000kV_30mu_2',
          '2017-03-26-2035_ImageSequence__150.000kX_10.000kV_30mu_3',
          '2017-03-26-2117_ImageSequence__150.000kX_10.000kV_30mu_4',
          '2017-03-26-2153_ImageSequence__150.000kX_10.000kV_30mu_5',
          '2017-03-26-2219_ImageSequence__150.000kX_10.000kV_30mu_6']

Pixel_size =  2.48#nm
Ps = Pixel_size #pixel size in nm, the numbers above with round nm precision
No_experiments = 5*np.ones([6])

let = ['N70D','N60D','N50D', 'N40D','N30D','RTD']
temp = [70.5, 58.8, 49.75,39.9, 30.5, 25.0 ]
tempstd = [0.7, 1.7, 0.55, 0.1, 0.0, 0.0]
######################################## Plot with dose for different apertures
##files below exist 
il_data = np.zeros([len(nametr)])
il_data_std = np.zeros([len(nametr)])

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


consider_whole_light = [0,1,2,3,4,5] #0,1,2,3,4,5,6]

ax0 = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=1)
ax1 = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((2,3), (0,2), colspan=1, rowspan=1)
ax3= plt.subplot2grid((2,3), (1,0), colspan=1, rowspan=1)
ax4 = plt.subplot2grid((2,3), (1,1), colspan=1, rowspan=1)
ax5 = plt.subplot2grid((2,3), (1,2), colspan=1, rowspan=1)

axvec = [ax0, ax1,ax2,ax3,ax4,ax5]

listofindex =np.arange(0,len(nametr))
for index in listofindex:
    
    print(index)
    il_data[index] = temp[index]
    il_data_std[index] = tempstd[index]
    
    
    
    
mycode = 'Il_dataGOINGDOWN = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Il_dataGOINGDOWN', data = il_data)

mycode = 'Il_data_stdGOINGDOWN = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Il_data_stdGOINGDOWN', data = il_data_std)


klklklklklk
    
    #ANTIGO 
#==============================================================================
#     #IL
#     il = np.load(str(let[index]) +'ILchannel.npz') 
# #    ###IL
# #    if index == 7:
# #        il_data[index] = 31.2
# #    else:
# #        aparam = 0.233801
# #        bparam = 0.000144
# #        delta = aparam*np.average(il['data'], axis = (0,1,2)) + bparam
# #        il_data[index] = KThermocouplerconversion(np.average(il['data'], axis = (0,1,2)) + 1.0e-3 + delta)
#         
#     
# #    if index == 0:
# #       deltav2 = +0.240e-3  # KThermocouplerconversion(1.0e-3) #take average per frame 
# #    if index == 1:
# #        deltav2 = +0.229e-3
# #    if index == 2:
# #        deltav2 = +0.100e-3
# #    if index == 3:
# #        deltav2 = +0.225e-3
# #    if index == 4:
# #        deltav2 = +0.359e-3
# #    if index == 5:
# #        deltav2 = +0.505e-3
# #    if index == 6:
# #       deltav2 = +0.660e-3
# #    
# #    
# #    hulper = np.average(il['data'], axis = (1,2))   
# #    il_data[index] = KThermocouplerconversion(hulper[0] + 1.0e-3 + deltav2) #take average per frame   
#      
#     il_data[index] = temp[index] 
#     #print(hulper[0])
#     #print(np.average(il['data'], axis = (0,1,2)))
#     #print(il_data)
#     
#     
#     
#     
#     ############################## FFT to cut noise in Temperature/IL data
#     #result = []
#     #print(dataALLred.shape[1])
#     total_time = totaltrpoints*Time_bin/1000.0 # in us
#     #print(total_time)
#     se1 = np.array(il['data'])
#     #se = se[0, :, :]
#     t = se1.flatten()
#     x = np.linspace(0, total_time, len(t))
#     #plt.figure(1)
#     #plt.plot( x * 1e-6/1e-3, t)
#     #plt.figure(2)
#     fft_y = np.fft.fft(t)
#     n = t.size
#     timestep = Time_bin
#     freq = np.fft.fftfreq(n, d = timestep)
#    #plt.semilogy(freq, np.abs(fft_y))
#     ind = np.abs(freq) > 1000
#     fft_y_cut = np.copy(fft_y)
#     fft_y_cut[ind] = 0.0
#    #plt.semilogy(freq, np.abs(fft_y_cut), 'r')
#     new_y = np.abs(np.fft.ifft(fft_y_cut))
#    #plt.figure(1)
#    #plt.plot( x * 1e-6/1e-3, new_y, label = str(k))
#    #plt.legend()
#     #result.append(np.mean(new_y[:-10]))
#     #result = np.array(result)
#     #deltav =0# -0.190e-3
#     #il_data[index] = KThermocouplerconversion(np.array(np.mean(new_y[:-10])) + 1.0e-3)
#     il_data_std[index] = KThermocouplerconversion(np.std(new_y[:-10]) )
#     #print(il_data)
#     #print(il_data_std)
# #    
#    
#     #il_data_std[index] = KThermocouplerconversion(np.std(il['data'], axis = (1,2)) )
#==============================================================================
    
    #print(il_data_std) 
    #del il, se1, t, fft_y, fft_y_cut, freq, ind, x, new_y
    #gc.collect()
    
    print('after il data') 
    
    print('before loading')



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

    
    import hlp_func
    # filename = Redbright.npz or Bluebright.npz
    hlp_func.calc_red_blue(let, index, 'Redbright.npz', red_int_array, red_decay_array, red_std_array, bgred_int_array, bgred_decay_array, bgred_std_array, hlp, hlpd)
    
    hlp_func.calc_red_blue(let, index, 'Bluebright.npz', blue_int_array, blue_decay_array, blue_std_array, bgblue_int_array, bgblue_decay_array, bgblue_std_array, hlp, hlpd)

    print("now done")


##### ONCE ALL FITS WORK, 
###### NEED TO RUN THESE LINES BELOW SO THAT ALL NPZ FILES ARE CREATED 
 
mycode = 'Il_dataGOINGDOWN = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Il_dataGOINGDOWN', data = il_data)

mycode = 'Il_data_stdGOINGDOWN = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Il_data_stdGOINGDOWN', data = il_data_std)

#foreground
  
mycode = 'Red_std_arrayGOINGDOWN = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Red_std_arrayGOINGDOWN', data = red_std_array)
 
mycode = 'Blue_std_arrayGOINGDOWN = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Blue_std_arrayGOINGDOWN', data = blue_std_array)
 
mycode = 'Red_int_arrayGOINGDOWN = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Red_int_arrayGOINGDOWN', data = red_int_array)
 
mycode = 'Blue_int_arrayGOINGDOWN = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Blue_int_arrayGOINGDOWN', data = blue_int_array)
 
mycode = 'Red_decay_arrayGOINGDOWN = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Red_decay_arrayGOINGDOWN', data = red_decay_array)

mycode = 'Blue_decay_arrayGOINGDOWN = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Blue_decay_arrayGOINGDOWN', data = blue_decay_array)

###background

mycode = 'bgRed_std_arrayGOINGDOWN = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgRed_std_arrayGOINGDOWN', data = bgred_std_array)
 
mycode = 'bgBlue_std_arrayGOINGDOWN = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgBlue_std_arrayGOINGDOWN', data = bgblue_std_array)
 
mycode = 'bgRed_int_arrayGOINGDOWN = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgRed_int_arrayGOINGDOWN', data = bgred_int_array)
 
mycode = 'bgBlue_int_arrayGOINGDOWN = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgBlue_int_arrayGOINGDOWN', data = bgblue_int_array)

mycode = 'bgRed_decay_arrayGOINGDOWN = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgRed_decay_arrayGOINGDOWN', data =bgred_decay_array)

mycode = 'bgBlue_decay_arrayGOINGDOWN = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('bgBlue_decay_arrayGOINGDOWN', data = bgblue_decay_array)
    
kjjhjh
