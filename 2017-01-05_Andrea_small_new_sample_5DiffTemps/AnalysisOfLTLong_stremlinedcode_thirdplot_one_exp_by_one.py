

#### THIS FILE CONSIDERS ONLY THE SE IMAGE TAKEN PRIOR TO TR

import os
import sys
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../Python modules/") # necessary for the tex fonts
import scipy as sp
import scipy.misc
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
from BackgroundCorrection import *
from TConversionThermocoupler import *
import matplotlib.cm as cm
import scipy.ndimage as ndimage
#from matplotlib_scalebar.scalebar import ScaleBar #### Has issue with plotting using latex font. only import when needed, then unimport
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from MakePdf import *
from matplotlib.pyplot import cm #to plot following colors of rainbow
from matplotlib import rc
from CreateDatasets import *
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = DeprecationWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.simplefilter(action = "ignore", category = PendingDeprecationWarning)
from Registration import * 
from tifffile import *
from sklearn.mixture import GMM 
import matplotlib.cm as cm
from FluoDecay import *
from PlottingFcts import *

from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.misc
import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile
#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################
Time_bin = 1000#in ns; 
nominal_time_on = 150.0 #time during which e-beam nominally on, in mus
totalpoints = 1500 #total number of time-resolved points

nametr = ['2017-01-05-1557_ImageSequence__250.000kX_10.000kV_30mu_15',
          '2017-01-05-1634_ImageSequence__250.000kX_10.000kV_30mu_20',
          '2017-01-05-1709_ImageSequence__250.000kX_10.000kV_30mu_23',
          '2017-01-05-1745_ImageSequence__250.000kX_10.000kV_30mu_26',
          '2017-01-05-1831_ImageSequence__250.000kX_10.000kV_30mu_30',
          '2017-01-05-1906_ImageSequence__250.000kX_10.000kV_30mu_32']


Pixel_size = 0.89*np.ones(len(nametr)) #nm
Ps = Pixel_size #pixel size in nm, the numbers above with round nm precision
No_experiments = 5*np.ones([6])

description = 'Andrea small NaYF4:Er'    # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
kv = [10]

let = ['V0','V0pt25' ,'V0pt5', 'V0pt5b','V0pt75','V1']
######################################## Plot with dose for different apertures
##files below exist 


pisize =Pixel_size

listofindex =np.arange(0,6)#,11]

index = 0


#####FIG WITH SUMMARY   
##### WHEN FILES ABOVE ARE CREATED, CREATE FIGURE BELOW WITH REGION SIZES VS LIFETIMES AND REGION SIZES VS S/N OR BRIGHTNESS OR <SIGNAL INT>/<BACKGROUND INT>
fastfactor = 1

fsizepl = 24
fsizenb = 20

red_int_array = np.zeros([5,6])
blue_int_array = np.zeros([5,6])
red_std_array = np.zeros([5,6])
blue_std_array = np.zeros([5,6])
b_array_red = np.zeros([5,6])
be_array_red = np.zeros([5,6])
e_array_red = np.zeros([5,6])
ee_array_red = np.zeros([5,6])
b_array_blue = np.zeros([5,6])
be_array_blue = np.zeros([5,6])
e_array_blue = np.zeros([5,6])
ee_array_blue = np.zeros([5,6])
il_data =  np.zeros([5,6])
il_data_std =  np.zeros([5,6])

RTred_int_array = np.zeros([5,1])
RTblue_int_array = np.zeros([5,1])
RTred_std_array = np.zeros([5,1])
RTblue_std_array = np.zeros([5,1])
RTb_array_red = np.zeros([5,1])
RTbe_array_red = np.zeros([5,1])
RTe_array_red = np.zeros([5,1])
RTee_array_red = np.zeros([5,1])
RTb_array_blue = np.zeros([5,1])
RTbe_array_blue = np.zeros([5,1])
RTe_array_blue = np.zeros([5,1])
RTee_array_blue = np.zeros([5,1])
RTil_data =  np.zeros([5,1])
RTil_data_std =  np.zeros([5,1])

for chosenexp in np.arange(0,5):
    
    print(chosenexp)
    Red_int_array = np.load('Red_int_array'+  'ChosenExp' + str(chosenexp) + '.npz') 
    red_int_array[chosenexp, :] = Red_int_array['data']
    Blue_int_array = np.load('Blue_int_array'+  'ChosenExp' + str(chosenexp) + '.npz') 
    blue_int_array[chosenexp, :] = Blue_int_array['data']
    
    Red_std_array = np.load('Red_std_array'+  'ChosenExp' + str(chosenexp) + '.npz') 
    red_std_array[chosenexp, :] = Red_int_array['data']
    Blue_std_array = np.load('Blue_std_array'+  'ChosenExp' + str(chosenexp) + '.npz') 
    blue_std_array[chosenexp, :] = Blue_int_array['data']
    
    B_array_red= np.load('B_array_red'+  'ChosenExp' + str(chosenexp) + '.npz')
    b_array_red[chosenexp, :] = B_array_red['data']  
    Be_array_red = np.load('Be_array_red'+  'ChosenExp' + str(chosenexp) + '.npz')
    be_array_red[chosenexp, :] = Be_array_red['data'] 
    E_array_red = np.load('E_array_red'+  'ChosenExp' + str(chosenexp) + '.npz')
    e_array_red[chosenexp, :] = E_array_red['data']   
    Ee_array_red = np.load('Ee_array_red'+  'ChosenExp' + str(chosenexp) + '.npz')
    ee_array_red[chosenexp, :] = Ee_array_red['data']   
    
    B_array_blue= np.load('B_array_blue'+  'ChosenExp' + str(chosenexp) + '.npz')
    b_array_blue[chosenexp, :] = B_array_blue['data']  
    Be_array_blue = np.load('Be_array_blue'+  'ChosenExp' + str(chosenexp) + '.npz')
    be_array_blue[chosenexp, :] = Be_array_blue['data'] 
    E_array_blue = np.load('E_array_blue'+  'ChosenExp' + str(chosenexp) + '.npz')
    e_array_blue[chosenexp, :] = E_array_blue['data']   
    Ee_array_blue = np.load('Ee_array_blue'+  'ChosenExp' + str(chosenexp) + '.npz')
    ee_array_blue[chosenexp, :] = Ee_array_blue['data'] 
    
    Il_data = np.load('Il_data'+  'ChosenExp' + str(chosenexp) + '.npz')
    il_data[chosenexp, :] = Il_data['data'][:,0]
    Il_data_std = np.load('Il_data_std'+  'ChosenExp' + str(chosenexp) + '.npz')
    il_data_std[chosenexp, :] = Il_data_std['data'][:,0] 

    RTRed_int_array = np.load('RTRed_int_array'+  'ChosenExp' + str(chosenexp) + '.npz') 
    RTred_int_array[chosenexp, :] = RTRed_int_array['data']
    RTBlue_int_array = np.load('RTBlue_int_array'+  'ChosenExp' + str(chosenexp) + '.npz') 
    RTblue_int_array[chosenexp, :] = RTBlue_int_array['data']
    
    RTRed_std_array = np.load('RTRed_std_array'+  'ChosenExp' + str(chosenexp) + '.npz') 
    RTred_std_array[chosenexp, :] = RTRed_int_array['data']
    RTBlue_std_array = np.load('RTBlue_std_array'+  'ChosenExp' + str(chosenexp) + '.npz') 
    RTblue_std_array[chosenexp, :] = RTBlue_int_array['data']
    
    RTB_array_red= np.load('RTB_array_red'+  'ChosenExp' + str(chosenexp) + '.npz')
    RTb_array_red[chosenexp, :] = RTB_array_red['data']  
    RTBe_array_red = np.load('RTBe_array_red'+  'ChosenExp' + str(chosenexp) + '.npz')
    RTbe_array_red[chosenexp, :] = RTBe_array_red['data'] 
    RTE_array_red = np.load('RTE_array_red'+  'ChosenExp' + str(chosenexp) + '.npz')
    RTe_array_red[chosenexp, :] = RTE_array_red['data']   
    RTEe_array_red = np.load('RTEe_array_red'+  'ChosenExp' + str(chosenexp) + '.npz')
    RTee_array_red[chosenexp, :] = RTEe_array_red['data']   
    
    RTB_array_blue= np.load('RTB_array_blue'+  'ChosenExp' + str(chosenexp) + '.npz')
    RTb_array_blue[chosenexp, :] = RTB_array_blue['data']  
    RTBe_array_blue = np.load('RTBe_array_blue'+  'ChosenExp' + str(chosenexp) + '.npz')
    RTbe_array_blue[chosenexp, :] = RTBe_array_blue['data'] 
    RTE_array_blue = np.load('RTE_array_blue'+  'ChosenExp' + str(chosenexp) + '.npz')
    RTe_array_blue[chosenexp, :] = RTE_array_blue['data']   
    RTEe_array_blue = np.load('RTEe_array_blue'+  'ChosenExp' + str(chosenexp) + '.npz')
    RTee_array_blue[chosenexp, :] = RTEe_array_blue['data'] 
    
    RTIl_data = np.load('RTIl_data'+  'ChosenExp' + str(chosenexp) + '.npz')
    RTil_data[chosenexp, :] = RTIl_data['data'][:,0]
    RTIl_data_std = np.load('RTIl_data_std'+  'ChosenExp' + str(chosenexp) + '.npz')
    RTil_data_std[chosenexp, :] = RTIl_data_std['data'][:,0] 

## All RT have models with "yes" error (ie, error visible)

## Take out points whose errors are too large
frame = [0,0,2,0,0,2,4,2,3,1,2,4]
temp = [0,1,1,3,4,4,4,5,5,5,2,2]

####LAST 3 PTS, ERROR ON MODEL NOT THAT BAD, BUT ERROR ON TAU SUPER BAD SO IM CUTTING THOSE TOO

for index_failed_model in np.arange(0,len(frame)):
    red_int_array[frame[index_failed_model], temp[index_failed_model]] = np.nan
    blue_int_array[frame[index_failed_model], temp[index_failed_model]] = np.nan
    red_std_array[frame[index_failed_model], temp[index_failed_model]] =np.nan
    blue_std_array[frame[index_failed_model], temp[index_failed_model] ]=np.nan
    b_array_red[frame[index_failed_model], temp[index_failed_model]] = np.nan
    be_array_red[frame[index_failed_model], temp[index_failed_model]] = np.nan
    e_array_red[frame[index_failed_model], temp[index_failed_model] ]=np.nan
    ee_array_red[frame[index_failed_model], temp[index_failed_model]] = np.nan
    b_array_blue[frame[index_failed_model], temp[index_failed_model]] = np.nan
    be_array_blue[frame[index_failed_model], temp[index_failed_model]] = np.nan
    e_array_blue[frame[index_failed_model], temp[index_failed_model]] =np.nan
    ee_array_blue[frame[index_failed_model], temp[index_failed_model]] = np.nan
   
####
length_scalebar = 50.0 #in nm 
scalebar_legend = '50nm'
length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[index]))
###let = ['V0','V0pt25' ,'V0pt5', 'V0pt75','V1']
fig42= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig42.set_size_inches(1200./fig42.dpi,900./fig42.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    
#fig42.suptitle('Lifetimes fitted after 1300$\mu$s decay, small NaYF$_4$ CS from Andrea \n (10kV, 30$\mu$m aperture, 40ns time bins)',fontsize=fsizetit)     
#ax0 = plt.subplot2grid((3,6), (0,0), colspan=1, rowspan=1)
#se = np.load('V0' +'SEchannel.npz',mmap_mode='r') 
#plt.imshow(se['data'],cmap=cm.Greys_r)
#plt.axis('off')
#sbar = sb.AnchoredScaleBar(ax0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#ax0.add_artist(sbar)
#
#ax00 = plt.subplot2grid((3,6), (0,1), colspan=1, rowspan=1)
#se = np.load('V0pt25' +'SEchannel.npz',mmap_mode='r') 
#plt.imshow(se['data'],cmap=cm.Greys_r)
#plt.axis('off')
#sbar = sb.AnchoredScaleBar(ax00.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#ax00.add_artist(sbar)
#
#ax000 = plt.subplot2grid((3,6), (0,2), colspan=1, rowspan=1)
#se = np.load('V0pt5' +'SEchannel.npz',mmap_mode='r') 
#plt.imshow(se['data'],cmap=cm.Greys_r)
#plt.axis('off')
#sbar = sb.AnchoredScaleBar(ax000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#ax000.add_artist(sbar)
#
#ax0000 = plt.subplot2grid((3,6), (0,3), colspan=1, rowspan=1)
#se = np.load('V0pt5b' +'SEchannel.npz',mmap_mode='r') 
#plt.imshow(se['data'],cmap=cm.Greys_r)
#plt.axis('off')
#sbar = sb.AnchoredScaleBar(ax0000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#ax0000.add_artist(sbar)
#
#ax00000 = plt.subplot2grid((3,6), (0,4), colspan=1, rowspan=1)
#se = np.load('V0pt75' +'SEchannel.npz',mmap_mode='r') 
#plt.imshow(se['data'],cmap=cm.Greys_r)
#plt.axis('off')
#sbar = sb.AnchoredScaleBar(ax00000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#ax00000.add_artist(sbar)
#
#ax000000 = plt.subplot2grid((3,6), (0,5), colspan=1, rowspan=1)
#se = np.load('V1' +'SEchannel.npz',mmap_mode='r') 
#plt.imshow(se['data'],cmap=cm.Greys_r)
#plt.axis('off')
#sbar = sb.AnchoredScaleBar(ax000000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#ax000000.add_artist(sbar)

######## just to test cut
#
#xval = 80
#yval = 80
#
#ax0 = plt.subplot2grid((3,6), (1,0), colspan=1, rowspan=1)
#se = np.load('V0' +'SEchannel.npz',mmap_mode='r') 
#xlen = se['data'].shape[0]
#ylen = se['data'].shape[1]
#delx = 0#+28
#dely = 0
#plt.imshow(se['data'][np.floor(xlen/2)-xval+delx:np.floor(xlen/2)+xval+delx,np.floor(ylen/2)-yval+dely:np.floor(ylen/2)+yval+dely],cmap=cm.Greys_r)
#plt.axis('off')
#sbar = sb.AnchoredScaleBar(ax0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#ax0.add_artist(sbar)
#
#ax00 = plt.subplot2grid((3,6), (1,1), colspan=1, rowspan=1)
#se = np.load('V0pt25' +'SEchannel.npz',mmap_mode='r') 
#xlen = se['data'].shape[0]
#ylen = se['data'].shape[1]
#delx = 0#+28
#dely = +26
#plt.imshow(se['data'][np.floor(xlen/2)-xval+delx:np.floor(xlen/2)+xval+delx,np.floor(ylen/2)-yval+dely:np.floor(ylen/2)+yval+dely],cmap=cm.Greys_r)
#plt.axis('off')
#sbar = sb.AnchoredScaleBar(ax00.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#ax00.add_artist(sbar)
#
#ax000 = plt.subplot2grid((3,6), (1,2), colspan=1, rowspan=1)
#se = np.load('V0pt5' +'SEchannel.npz',mmap_mode='r') 
#xlen = se['data'].shape[0]
#ylen = se['data'].shape[1]
#delx = 0#+28
#dely = -4
#plt.imshow(se['data'][np.floor(xlen/2)-xval+delx:np.floor(xlen/2)+xval+delx,np.floor(ylen/2)-yval+dely:np.floor(ylen/2)+yval+dely],cmap=cm.Greys_r)
#plt.axis('off')
#sbar = sb.AnchoredScaleBar(ax000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#ax000.add_artist(sbar)
#
#ax0000 = plt.subplot2grid((3,6), (1,3), colspan=1, rowspan=1)
#se = np.load('V0pt5b' +'SEchannel.npz',mmap_mode='r')
#xlen = se['data'].shape[0]
#ylen = se['data'].shape[1] 
#delx = 0#+28
#dely = 0
#plt.imshow(se['data'][np.floor(xlen/2)-xval+delx:np.floor(xlen/2)+xval+delx,np.floor(ylen/2)-yval+dely:np.floor(ylen/2)+yval+dely],cmap=cm.Greys_r)
#plt.axis('off')
#sbar = sb.AnchoredScaleBar(ax0000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#ax0000.add_artist(sbar)
#
#ax00000 = plt.subplot2grid((3,6), (1,4), colspan=1, rowspan=1)
#se = np.load('V0pt75' +'SEchannel.npz',mmap_mode='r') 
#xlen = se['data'].shape[0]
#ylen = se['data'].shape[1]
#delx = 0#+28
#dely = 0
#plt.imshow(se['data'][np.floor(xlen/2)-xval+delx:np.floor(xlen/2)+xval+delx,np.floor(ylen/2)-yval+dely:np.floor(ylen/2)+yval+dely],cmap=cm.Greys_r)
#plt.axis('off')
#sbar = sb.AnchoredScaleBar(ax00000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#ax00000.add_artist(sbar)
#
#ax000000 = plt.subplot2grid((3,6), (1,5), colspan=1, rowspan=1)
#se = np.load('V1' +'SEchannel.npz',mmap_mode='r') 
#xlen = se['data'].shape[0]
#ylen = se['data'].shape[1]
#delx = 0#+28
#dely = 0
#plt.imshow(se['data'][np.floor(xlen/2)-xval+delx:np.floor(xlen/2)+xval+delx,np.floor(ylen/2)-yval+dely:np.floor(ylen/2)+yval+dely],cmap=cm.Greys_r)
#plt.axis('off')
#sbar = sb.AnchoredScaleBar(ax000000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#ax000000.add_artist(sbar)
#
######## just to test cut

ax2 = plt.subplot2grid((2,5), (0,0), colspan=5, rowspan=1)
ax1 = plt.subplot2grid((2,5), (1,0), colspan=5, sharex=ax2)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

ma_size = [3,4,5,6,7]

#EACH FIT
for chosenexp in np.arange(0,5):

    print(chosenexp)
    
    #other Ts
    x_vec = il_data[chosenexp,:]
    #ax2.errorbar(x_vec, b_array_red[chosenexp,:], yerr=be_array_red[chosenexp,:], fmt='rs',markersize=ma_size[chosenexp])
    ax2.errorbar(x_vec, e_array_red[chosenexp,:], yerr=ee_array_red[chosenexp,:], fmt='ro', markersize=ma_size[chosenexp])
    #ax2.errorbar(x_vec, b_array_blue[chosenexp,:], yerr=be_array_blue[chosenexp,:], fmt='gs',markersize=ma_size[chosenexp])
    ax2.errorbar(x_vec, e_array_blue[chosenexp,:], yerr=ee_array_blue[chosenexp,:], fmt='go', markersize=ma_size[chosenexp])
    
    #RT
    x_vec = RTil_data[chosenexp,:]
    #ax2.errorbar(x_vec, RTb_array_red[chosenexp,:], yerr=RTbe_array_red[chosenexp,:], fmt='rs',markersize=ma_size[chosenexp])
    ax2.errorbar(x_vec, RTe_array_red[chosenexp,:], yerr=RTee_array_red[chosenexp,:], fmt='ro', markersize=ma_size[chosenexp])
    #ax2.errorbar(x_vec, RTb_array_blue[chosenexp,:], yerr=RTbe_array_blue[chosenexp,:], fmt='gs',markersize=ma_size[chosenexp])
    ax2.errorbar(x_vec, RTe_array_blue[chosenexp,:], yerr=RTee_array_blue[chosenexp,:], fmt='go', markersize=ma_size[chosenexp])
  
#MEAN OF FITS  
#RT + other Ts
x_vec = np.concatenate((np.average(RTil_data,axis = 0),np.average(il_data,axis = 0)),axis=0)
#ax2.errorbar(x_vec, np.average(b_array_red, axis = 0),  fmt='rs',markersize=10)
ax2.errorbar(x_vec, np.concatenate((np.average(RTe_array_red, axis = 0),np.nanmean(e_array_red, axis = 0)),axis=0), color='r', marker = 'o', markersize=10)
#ax2.errorbar(x_vec, np.average(b_array_blue,axis = 0),  fmt='gs',markersize=10)
ax2.errorbar(x_vec, np.concatenate((np.average(RTe_array_blue,axis = 0),np.nanmean(e_array_blue,axis = 0)),axis=0), color='g', marker = 'o', markersize=10)

ax2.set_ylabel('Time constants ($\mu$s)',fontsize=fsizepl)
#ax2.set_ylim([0,500])
#plt.xlim([0,500])
#ax2.legend(fontsize=fsizepl)
ax2.tick_params(labelsize=fsizenb)
ax1.set_xlabel('Temperature ($^{\circ}$C)',fontsize=fsizepl)
plt.sca(ax1) 
#ax2.set_xlabel('Temperature ($^{\circ}$C)',fontsize=fsizepl)
#plt.sca(ax2) 
plt.setp(ax2.get_xticklabels(), visible=False)
plt.xlim([20,100])
ax2.set_ylim([0,1000])
###

### error
import uncertainties as unc  
import uncertainties.unumpy as unumpy  

ratio2 = []
x_vec = RTil_data
u_red_int_array=unumpy.uarray(( RTred_int_array, RTred_std_array ))  
u_blue_int_array=unumpy.uarray(( RTblue_int_array, RTblue_std_array ))  

#calc intensity ratios
ratio = (u_red_int_array/u_blue_int_array) #/(u_red_int_array[experime]/u_blue_int_array[0])
unumpy_error_ratio = unumpy.std_devs(ratio) 
ratio2 = (RTred_int_array/RTblue_int_array)/(np.nanmean(RTred_int_array,axis=(0,1))/np.nanmean(RTblue_int_array, axis=(0,1)))
ax1.errorbar(x_vec, ratio2, fmt='ks',markersize=7.5)
#ax1.errorbar(x_vec, ratio2, yerr=unumpy_error_ratio, fmt='ks',markersize=7.5)
del u_red_int_array, u_blue_int_array
gc.collect()

ratio2 = np.zeros([6])

for experim in np.arange(0,6):
    
    print(experim) 
    x_vec = il_data[:,experim]
    u_red_int_array=unumpy.uarray(( red_int_array[:,experim], red_std_array[:,experim] ))  
    u_blue_int_array=unumpy.uarray(( blue_int_array[:,experim], blue_std_array[:,experim] ))  

    #calc intensity ratios
    #ratio = (u_red_int_array/u_blue_int_array) #/(u_red_int_array[experime]/u_blue_int_array[0])
    #unumpy_error_ratio = unumpy.std_devs(ratio) 
    ratio2 = (red_int_array[:,experim]/blue_int_array[:,experim])/(np.average(RTred_int_array,axis=(0,1))/np.average(RTblue_int_array, axis=(0,1)))

    ax1.errorbar(x_vec, ratio2, fmt='ks',markersize=7.5)
    #ax1.errorbar(x_vec, ratio2, yerr=unumpy_error_ratio, fmt='ks',markersize=7.5)
    del u_red_int_array, u_blue_int_array
    gc.collect()
    
#MEAN OF FITS  
#RT + other Ts
x_vec = np.concatenate((np.average(RTil_data,axis = 0),np.average(il_data,axis = 0)),axis=0)
#xx = 1.0 #np.average(RTred_int_array,axis=(0,1))/np.average(RTblue_int_array, axis=(0,1))
xx = np.ones(1)
ax1.errorbar(x_vec, np.concatenate((xx,(np.nanmean(red_int_array,axis=(0))/np.nanmean(blue_int_array, axis=(0)))/(np.nanmean(RTred_int_array,axis=(0,1))/np.nanmean(RTblue_int_array, axis=(0,1)))),axis=0), color='b', marker = 'o', markersize=10)

#
ax1.set_ylabel('Red to green \n intensity ratio',fontsize=fsizepl)
##ax1.set_ylim([0, 1.05])
##ax1.set_yticks([0.5,0.75,1.0])
##ax1.errorbar(x_vec, ratio2, yerr=unumpy_error_ratio, fmt='ks',markersize=7.5)
plt.tick_params(labelsize=fsizenb)
##plt.xlim([20,100])
##plt.xticks([30,40,50,60,70])

fig43= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig43.set_size_inches(1200./fig42.dpi,900./fig42.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')   

ax2 = plt.subplot2grid((1,5), (0,0), colspan=5, rowspan=1)
x_vec = np.concatenate((np.average(RTil_data,axis = 0), np.average(il_data,axis = 0)),axis=0)
#LONG/SHORT green
ax2.errorbar(x_vec,np.concatenate((np.nanmean(RTe_array_blue,axis = 0)/np.nanmean(RTb_array_blue, axis = 0), np.nanmean(e_array_blue,axis = 0)/np.nanmean(b_array_blue, axis = 0)),axis=0),color='g', markersize=5,linewidth=2)
#LONG/SHORT red
ax2.errorbar(x_vec,np.concatenate((np.nanmean(RTe_array_red,axis = 0)/np.nanmean(RTb_array_red, axis = 0),np.nanmean(e_array_red,axis = 0)/np.nanmean(b_array_red, axis = 0)), axis=0),color='r', markersize=5,linewidth=2)
#LONG/LONG green/red
ax2.errorbar(x_vec,np.concatenate((np.nanmean(RTe_array_blue,axis = 0)/np.nanmean(RTe_array_red, axis = 0),np.nanmean(e_array_blue,axis = 0)/np.nanmean(e_array_red, axis = 0)),axis=0),color = 'k', markersize=7,linewidth=3)
#SHORT/SHORT green/red
ax2.errorbar(x_vec, np.concatenate((np.nanmean(RTb_array_blue,axis = 0)/np.nanmean(RTb_array_red, axis = 0),np.nanmean(b_array_blue,axis = 0)/np.nanmean(b_array_red, axis = 0)),axis=0),color='k', markersize=3,linewidth=1)

multipage_longer('ZZZZZZZZSummary.pdf',dpi=80)

