

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
prefix = 'varCprevtau'

Time_bin = 1000#in ns; 
nominal_time_on = 150.0 #time during which e-beam nominally on, in mus
totaltrpoints = 1400 #total number of time-resolved points

nametr = ['2017-01-13-1730_ImageSequence__150.000kX_10.000kV_30mu_15',
          '2017-01-13-1810_ImageSequence__150.000kX_10.000kV_30mu_17',
          '2017-01-13-1935_ImageSequence__150.000kX_10.000kV_30mu_3',
          '2017-01-13-2011_ImageSequence__150.000kX_10.000kV_30mu_4',
          '2017-01-13-2050_ImageSequence__150.000kX_10.000kV_30mu_7',
          '2017-01-13-2132_ImageSequence__150.000kX_10.000kV_30mu_8',
          '2017-01-13-2213_ImageSequence__150.000kX_10.000kV_30mu_9',
          '2017-01-13-2251_ImageSequence__150.000kX_10.000kV_30mu_10']


Pixel_size = 2.5*np.ones(8) #nm
Ps = Pixel_size #pixel size in nm, the numbers above with round nm precision
No_experiments = [4,4,4,4,4,4,4,4]

description = 'Andrea small NaYF4:Er'    # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
kv = [10]

let = ['N102pt5', 'N92pt9', 'N66pt4', 'N72pt8', 'N60pt6', 'N48pt5', 'N39pt8', 'N31pt2']
######################################## Plot with dose for different apertures
##files below exist 


pisize =Pixel_size

listofindex =np.arange(0,7)#,11]

index = 0

##### ONCE ALL FITS WORK, 
###### NEED TO RUN THESE LINES BELOW SO THAT ALL NPZ FILES ARE CREATED 
#mycode = 'Red_int_array = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Red_int_array', data = red_int_array)
#
#mycode = 'Blue_int_array = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Blue_int_array', data = blue_int_array)
#
#mycode = 'Red_std_array = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Red_std_array', data = red_std_array)
#
#mycode = 'Blue_std_array = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Blue_std_array', data = blue_std_array)
#
#il_data = [ 41.79260051,  29.8911322,   36.69970937,  48.82985395, 69.84299812]
#mycode = 'Il_data = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Il_data', data = il_data)
#
#b_array_red = [50.861, 38.772, 27.493, 30.914, 42.129]
#mycode = 'B_array_red = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('B_array_red', data = b_array_red)
#
#be_array_red = [8.813, 6.071, 4.560, 3.537, 9.797]
#mycode ='Be_array_red = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Be_array_red', data = be_array_red)
#
#e_array_red = [232.240, 260.251, 213.217, 192.565, 164.092]
#mycode = 'E_array_red = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('E_array_red', data = e_array_red)
#
#ee_array_red = [17.787, 22.585, 13.830, 14.539, 7.922]
#mycode = 'Ee_array_red = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Ee_array_red', data = ee_array_red)
#
#b_array_blue = [34.478, 32.995, 26.309, 21.116, 42.616]
#mycode = 'B_array_blue = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('B_array_blue', data = b_array_blue)
#
#be_array_blue = [3.954, 3.458, 2.751, 2.761, 4.138]
#mycode ='Be_array_blue = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Be_array_blue', data = be_array_blue)
#
#e_array_blue = [178.036,206.513, 194.171, 152.834, 152.166]
#mycode = 'E_array_blue = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('E_array_blue', data = e_array_blue)
#
#ee_array_blue = [8.243, 13.458, 11.513, 7.545, 7.311]
#mycode = 'Ee_array_blue = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Ee_array_blue', data = ee_array_blue)

#####FIG WITH SUMMARY   
##### WHEN FILES ABOVE ARE CREATED, CREATE FIGURE BELOW WITH REGION SIZES VS LIFETIMES AND REGION SIZES VS S/N OR BRIGHTNESS OR <SIGNAL INT>/<BACKGROUND INT>
fastfactor = 1

fsizepl = 24
fsizenb = 20

if len(prefix) == 0:
    Red_int_array = np.load('Red_int_array.npz') 
    red_int_array = Red_int_array['data']
    Blue_int_array = np.load('Blue_int_array.npz') 
    blue_int_array = Blue_int_array['data']
    
    Red_std_array = np.load('Red_std_array.npz') 
    red_std_array = Red_std_array['data']
    Blue_std_array = np.load('Blue_std_array.npz') 
    blue_std_array = Blue_std_array['data']
    
    B_array_red= np.load('B_array_red.npz')
    b_array_red = B_array_red['data']  
    Be_array_red = np.load('Be_array_red.npz')
    be_array_red = Be_array_red['data'] 
    E_array_red = np.load('E_array_red.npz')
    e_array_red = E_array_red['data']   
    Ee_array_red = np.load('Ee_array_red.npz')
    ee_array_red = Ee_array_red['data']   
    
    B_array_blue= np.load('B_array_blue.npz')
    b_array_blue = B_array_blue['data']  
    Be_array_blue = np.load('Be_array_blue.npz')
    be_array_blue = Be_array_blue['data'] 
    E_array_blue = np.load('E_array_blue.npz')
    e_array_blue = E_array_blue['data']   
    Ee_array_blue = np.load('Ee_array_blue.npz')
    ee_array_blue = Ee_array_blue['data'] 
    
    Il_data = np.load('Il_data.npz')
    il_data = Il_data['data']  
    Il_data_std = np.load('Il_data_std.npz')
    il_data_std = Il_data_std['data']  
    
else:
    Red_int_array = np.load(prefix +'Red_int_array.npz') 
    red_int_array = Red_int_array['data']
    Blue_int_array = np.load(prefix +'Blue_int_array.npz') 
    blue_int_array = Blue_int_array['data']
    
    Red_std_array = np.load(prefix +'Red_std_array.npz') 
    red_std_array = Red_std_array['data']
    Blue_std_array = np.load(prefix +'Blue_std_array.npz') 
    blue_std_array = Blue_std_array['data']
    
    B_array_red= np.load(prefix +'B_array_red.npz')
    b_array_red = B_array_red['data']  
    Be_array_red = np.load(prefix +'Be_array_red.npz')
    be_array_red = Be_array_red['data'] 
    E_array_red = np.load(prefix +'E_array_red.npz')
    e_array_red = E_array_red['data']   
    Ee_array_red = np.load(prefix +'Ee_array_red.npz')
    ee_array_red = Ee_array_red['data']   
    
    B_array_blue= np.load(prefix +'B_array_blue.npz')
    b_array_blue = B_array_blue['data']  
    Be_array_blue = np.load(prefix +'Be_array_blue.npz')
    be_array_blue = Be_array_blue['data'] 
    E_array_blue = np.load(prefix +'E_array_blue.npz')
    e_array_blue = E_array_blue['data']   
    Ee_array_blue = np.load(prefix +'Ee_array_blue.npz')
    ee_array_blue = Ee_array_blue['data'] 
    
    Il_data = np.load(prefix +'Il_data.npz')
    il_data = Il_data['data']  
    Il_data_std = np.load(prefix +'Il_data_std.npz')
    il_data_std = Il_data_std['data']  

################### deleting data points
todel = [0]
red_int_array = np.delete(red_int_array, todel)
blue_int_array = np.delete(blue_int_array, todel)
red_std_array = np.delete(red_std_array, todel)
blue_std_array = np.delete(blue_std_array, todel)
b_array_red = np.delete(b_array_red, todel)
be_array_red = np.delete(be_array_red, todel)
e_array_red = np.delete(e_array_red, todel)
ee_array_red = np.delete(ee_array_red, todel)
b_array_blue = np.delete(b_array_blue, todel)
be_array_blue = np.delete(be_array_blue, todel)
e_array_blue = np.delete(e_array_blue, todel)
ee_array_blue = np.delete(ee_array_blue, todel)
il_data = np.delete(il_data, todel)
il_data_std = np.delete(il_data_std, todel)
il_data_std = il_data_std/np.sqrt(4) #taking into account that each il_data comes from 1 different averages 
####
length_scalebar = 100.0 #in nm 
scalebar_legend = '100nm'
length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[index]))
###let = ['V0','V0pt25' ,'V0pt5', 'V0pt75','V1']
fig42= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig42.set_size_inches(1200./fig42.dpi,900./fig42.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    
#fig42.suptitle('Lifetimes fitted after 1300$\mu$s decay, small NaYF$_4$ CS from Andrea \n (10kV, 30$\mu$m aperture, 40ns time bins)',fontsize=fsizetit)     

nofigs = 7

ax0 = plt.subplot2grid((3,nofigs), (0,0), colspan=1, rowspan=1)
se = np.load(let[1] +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(ax0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
ax0.add_artist(sbar)

ax00 = plt.subplot2grid((3,nofigs), (0,1), colspan=1, rowspan=1)
se = np.load(let[2] +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(ax00.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
ax00.add_artist(sbar)

ax000 = plt.subplot2grid((3,nofigs), (0,2), colspan=1, rowspan=1)
se = np.load(let[3] +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(ax000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
ax000.add_artist(sbar)

ax0000 = plt.subplot2grid((3,nofigs), (0,3), colspan=1, rowspan=1)
se = np.load(let[4] +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(ax0000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
ax0000.add_artist(sbar)

ax00000 = plt.subplot2grid((3,nofigs), (0,4), colspan=1, rowspan=1)
se = np.load(let[5] +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(ax00000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
ax00000.add_artist(sbar)

axb = plt.subplot2grid((3,nofigs), (0,5), colspan=1, rowspan=1)
se = np.load(let[6] +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(axb.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
axb.add_artist(sbar)

axb0 = plt.subplot2grid((3,nofigs), (0,6), colspan=1, rowspan=1)
se = np.load(let[7] +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(axb0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
axb0.add_artist(sbar)

ax2 = plt.subplot2grid((3,6), (1,0), colspan=6, rowspan=1)
ax1 = plt.subplot2grid((3,6), (2,0), colspan=6, sharex=ax2)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

x_vec = il_data
#x_vec = [27.8, 30.5, 44.4, 50.5, 74.6, 94.9]

ax2.errorbar(x_vec, b_array_red, yerr=be_array_red, xerr=il_data_std, fmt='ro',markersize=5)
ax2.errorbar(x_vec, e_array_red, yerr=ee_array_red, xerr=il_data_std,fmt='ro', markersize=10)
ax2.errorbar(x_vec, b_array_blue, yerr=be_array_blue, xerr=il_data_std,fmt='go',markersize=5)
ax2.errorbar(x_vec, e_array_blue, yerr=ee_array_blue, xerr=il_data_std, fmt='go', markersize=10)

ax2.set_ylabel('Time constants ($\mu$s)',fontsize=fsizepl)

ax2.tick_params(labelsize=fsizenb)
ax1.set_xlabel('Temperature ($^{\circ}$C)',fontsize=fsizepl)

plt.sca(ax1) 

plt.setp(ax2.get_xticklabels(), visible=False)
plt.xlim([0,100])
ax2.set_ylim([0,400])
###

ind_RT = 6
#### index of room temperature, after deletions

### error
import uncertainties as unc  
import uncertainties.unumpy as unumpy  
#import nemmen 
u_red_int_array=unumpy.uarray(( red_int_array, red_std_array ))  
u_blue_int_array=unumpy.uarray(( blue_int_array, blue_std_array ))  

#calc intensity ratios
ratio = (u_red_int_array/u_blue_int_array)/(u_red_int_array[ind_RT]/u_blue_int_array[ind_RT])
unumpy_error_ratio = unumpy.std_devs(ratio) 
ratio2 = (red_int_array/blue_int_array)/(red_int_array[ind_RT]/blue_int_array[ind_RT])

ratioun = (u_red_int_array/u_blue_int_array)
unumpy_error_ratioun = unumpy.std_devs(ratioun) 

#ax3 = ax2.twinx()
#ax1.set_ylim([0, 1.05])
#ax1.set_yticks([0.5,0.75,1.0])
#ax1.errorbar(x_vec, ratio2, yerr=unumpy_error_ratio, fmt='ks',markersize=7.5)
ax1.errorbar(x_vec, ratio2, yerr=unumpy_error_ratio, xerr=il_data_std,fmt='ko',markersize=10,label='normalized by RT')
ax1.errorbar(x_vec, red_int_array/blue_int_array,xerr=il_data_std, yerr=unumpy_error_ratioun, fmt='ko',markersize=5,label='unnormalized')
ax1.set_ylabel('Red to green \n intensity ratios',fontsize=fsizepl)
plt.tick_params(labelsize=fsizenb)
plt.xlim([20,100])
plt.legend(loc ='best')
#ax1.set_ylim([0,1.05])
#plt.xticks([30,40,50,60,70])

############ SECOND PAGE PLOTS
###############################################################################

fig42= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig42.set_size_inches(1200./fig42.dpi,900./fig42.dpi)

x_vec = il_data

ax2 = plt.subplot2grid((2,4), (0,0), colspan=2, rowspan=1)

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

ax2.errorbar(x_vec, e_array_red/b_array_red,  fmt='ro',markersize=10,ls='-')
ax2.errorbar(x_vec, e_array_blue/b_array_blue, fmt='go', markersize=10,ls='-')
ax2.errorbar(x_vec, e_array_red/e_array_blue,  fmt='ko',markersize=10,ls='-',label='red/green long')
ax2.errorbar(x_vec, b_array_red/b_array_blue, fmt='ko', markersize=7,ls='-',label='red/green short')
ax2.set_ylabel('Time constant ratios',fontsize=fsizepl)
ax2.tick_params(labelsize=fsizenb)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.legend(loc='best')

ax1 = plt.subplot2grid((2,4), (1,0), colspan=2, sharex=ax2)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')


ax1.errorbar(x_vec, e_array_red/b_array_red/(e_array_red[ind_RT]/b_array_red[ind_RT]),  fmt='ro',markersize=10,ls='-')
ax1.errorbar(x_vec, e_array_blue/b_array_blue/(e_array_blue[ind_RT]/b_array_blue[ind_RT]), fmt='go', markersize=10,ls='-')
ax1.errorbar(x_vec, e_array_red/e_array_blue/(e_array_red[ind_RT]/e_array_blue[ind_RT]),  fmt='ko',markersize=10,ls='-')
ax1.errorbar(x_vec, b_array_red/b_array_blue/(b_array_red[ind_RT]/b_array_blue[ind_RT]), fmt='ko', markersize=7,ls='-')

ax1.set_ylabel('Time constant ratios,\n normalized by RT',fontsize=fsizepl)
ax1.tick_params(labelsize=fsizenb)
ax1.set_xlabel('Temperature ($^{\circ}$C)',fontsize=fsizepl)
plt.xlim([20,100])

ax2 = plt.subplot2grid((2,4), (0,2), colspan=2, rowspan=1)

ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('right')
ax2.yaxis.set_label_position("right")

no_pixels = 300.0*300.0
ax2.errorbar(x_vec, red_int_array/Ps[0]/Ps[0], fmt='ro',markersize=10, label='kHz/nm$^2$',ls='-')
ax2.errorbar(x_vec, blue_int_array/Ps[0]/Ps[0], fmt='go',markersize=10, label='kHz/nm$^2$',ls='-')
ax2.set_ylabel('Intensities (kHz/nm$^2$)\n (light per area)',fontsize=fsizepl)
plt.tick_params(labelsize=fsizenb)
plt.setp(ax2.get_xticklabels(), visible=False)

ax1 = plt.subplot2grid((2,4), (1,2), colspan=2, sharex=ax2)

ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('right')
ax1.yaxis.set_label_position("right")

ax1.errorbar(x_vec, red_int_array*no_pixels/1.0e3, fmt='ro',markersize=10, label='MHz/frame$',ls='-')
ax1.errorbar(x_vec, blue_int_array*no_pixels/1.0e3, fmt='go',markersize=10, label='MHz/frame$',ls='-')
ax1.set_ylabel('Intensities (MHz/frame)\n (total collected light)',fontsize=fsizepl)
plt.tick_params(labelsize=fsizenb)
ax1.set_xlabel('Temperature ($^{\circ}$C)',fontsize=fsizepl)
plt.xlim([20,90])

if len(prefix) == 0:
    multipage_longer('ZZZZZZZZSummaryFINALSHAPE.pdf',dpi=80)
else:
     multipage_longer(prefix + 'ZZZZZZZZSummaryFINALSHAPE.pdf',dpi=80)