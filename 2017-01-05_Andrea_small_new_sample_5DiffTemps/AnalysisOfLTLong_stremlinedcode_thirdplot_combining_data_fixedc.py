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
#prefix = 'fixedCprevtau'
#prefix = 'varCprevtau' #'varCprevtau'
#prefix = '' #var c, same tau
prefix = 'fixedC' #same tau

Time_bin = 1000#in ns; 
nominal_time_on = 150.0 #time during which e-beam nominally on, in mus
totalpoints = 1500 #total number of time-resolved points

tminx = 20
tmaxx = 100

nametr = ['2017-01-05-1452_ImageSequence__250.000kX_10.000kV_30mu_10',
          '2017-01-05-1557_ImageSequence__250.000kX_10.000kV_30mu_15',
          '2017-01-05-1634_ImageSequence__250.000kX_10.000kV_30mu_20',
          '2017-01-05-1709_ImageSequence__250.000kX_10.000kV_30mu_23',
          '2017-01-05-1745_ImageSequence__250.000kX_10.000kV_30mu_26',
          '2017-01-05-1831_ImageSequence__250.000kX_10.000kV_30mu_30',
          '2017-01-05-1906_ImageSequence__250.000kX_10.000kV_30mu_32']

No_experiments = 5*np.ones([6])

description = 'Andrea small NaYF4:Er'    # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
kv = [10]

let = ['RT','V0','V0pt25' ,'V0pt5', 'V0pt5b','V0pt75','V1']
######################################## Plot with dose for different apertures
##files below exist 

index = 0
listofindex =np.arange(0,7)#,11]

#####FIG WITH SUMMARY   
##### WHEN FILES ABOVE ARE CREATED, CREATE FIGURE BELOW WITH REGION SIZES VS LIFETIMES AND REGION SIZES VS S/N OR BRIGHTNESS OR <SIGNAL INT>/<BACKGROUND INT>
fastfactor = 1

fsizepl = 24
fsizenb = 20

######## Load this data  #####LARGEST ZOOM

Red_int_array = np.load(prefix +'Red_int_array.npz') 
red_int_array = Red_int_array['data']
Blue_int_array = np.load(prefix +'Blue_int_array.npz') 
blue_int_array = Blue_int_array['data']

Red_std_array = np.load(prefix +'Red_std_array.npz') 
red_std_array = Red_int_array['data']
Blue_std_array = np.load(prefix +'Blue_std_array.npz') 
blue_std_array = Blue_int_array['data']

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

############ load other data #####SMALLEST ZOOM

Red_int_array2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/' + prefix +'Red_int_array.npz') 
red_int_array2 = Red_int_array2['data']
Blue_int_array2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/' + prefix +'Blue_int_array.npz') 
blue_int_array2 = Blue_int_array2['data']

Red_std_array2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/' + prefix +'Red_std_array.npz') 
red_std_array2 = Red_int_array2['data']
Blue_std_array2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/' + prefix +'Blue_std_array.npz') 
blue_std_array2 = Blue_int_array2['data']

B_array_red2= np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/' + prefix +'B_array_red.npz')
b_array_red2 = B_array_red2['data']  
Be_array_red2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/' + prefix +'Be_array_red.npz')
be_array_red2 = Be_array_red2['data'] 
E_array_red2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/' + prefix +'E_array_red.npz')
e_array_red2 = E_array_red2['data']   
Ee_array_red2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/' + prefix +'Ee_array_red.npz')
ee_array_red2 = Ee_array_red2['data']   

B_array_blue2= np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/' + prefix +'B_array_blue.npz')
b_array_blue2 = B_array_blue2['data']  
Be_array_blue2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/' + prefix +'Be_array_blue.npz')
be_array_blue2 = Be_array_blue2['data'] 
E_array_blue2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/' + prefix +'E_array_blue.npz')
e_array_blue2 = E_array_blue2['data']   
Ee_array_blue2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/' + prefix +'Ee_array_blue.npz')
ee_array_blue2 = Ee_array_blue2['data'] 

Il_data2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/' + prefix +'Il_data.npz')
il_data2 = Il_data2['data']  
Il_data_std2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/' + prefix +'Il_data_std.npz')
il_data_std2 = Il_data_std2['data']  

############ load other data #####MEDIUM ZOOM

Red_int_array3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + prefix +'Red_int_array.npz') 
red_int_array3 = Red_int_array3['data']
Blue_int_array3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + prefix +'Blue_int_array.npz') 
blue_int_array3 = Blue_int_array3['data']

Red_std_array3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + prefix +'Red_std_array.npz') 
red_std_array3 = Red_int_array3['data']
Blue_std_array3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + prefix +'Blue_std_array.npz') 
blue_std_array3 = Blue_int_array3['data']

B_array_red3= np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + prefix +'B_array_red.npz')
b_array_red3 = B_array_red3['data']  
Be_array_red3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + prefix +'Be_array_red.npz')
be_array_red3 = Be_array_red3['data'] 
E_array_red3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + prefix +'E_array_red.npz')
e_array_red3 = E_array_red3['data']   
Ee_array_red3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + prefix +'Ee_array_red.npz')
ee_array_red3 = Ee_array_red3['data']   

B_array_blue3= np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + prefix +'B_array_blue.npz')
b_array_blue3 = B_array_blue3['data']  
Be_array_blue3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + prefix +'Be_array_blue.npz')
be_array_blue3 = Be_array_blue3['data'] 
E_array_blue3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + prefix +'E_array_blue.npz')
e_array_blue3 = E_array_blue3['data']   
Ee_array_blue3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + prefix +'Ee_array_blue.npz')
ee_array_blue3 = Ee_array_blue3['data'] 

Il_data3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + prefix +'Il_data.npz')
il_data3 = Il_data3['data']  
Il_data_std3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + prefix +'Il_data_std.npz')
il_data_std3 = Il_data_std3['data']  

####
fig42= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig42.set_size_inches(1200./fig42.dpi,900./fig42.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    

# titulo

titulo = 'NaYF$_4$:Yb,Er UC NPs (sample by Andrea Pickel)\n (10 kV, 30 $\mu$m aperture, 1 $\mu$s time bins, 1.4 ms transient, cathodoluminescence green/red: $</>$ 593 nm)'
plt.suptitle(titulo,fontsize=fsizetit)

#plot figs

length_scalebar = 100.0 #in nm 
scalebar_legend = '100nm'
Pixel_size = 0.89
length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size))

noplots = 13
nolines = 4


ax0 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)
se = np.load('V0' +'SEchannel.npz',mmap_mode='r') 
xlen = se['data'].shape[0]
ylen = se['data'].shape[1]
delx = 0#+28
dely = 0
xval = 107
yval = 107
plt.imshow(se['data'][np.floor(xlen/2)-xval+delx:np.floor(xlen/2)+xval+delx,np.floor(ylen/2)-yval+dely:np.floor(ylen/2)+yval+dely],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(ax0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
ax0.add_artist(sbar)

ax00 = plt.subplot2grid((nolines,noplots), (0,2), colspan=1, rowspan=1)
se = np.load('V0pt25' +'SEchannel.npz',mmap_mode='r') 
xlen = se['data'].shape[0]
ylen = se['data'].shape[1]
delx = 0#+28
dely = 0 #+26
xval = 96
yval = 106
cutx = 0 #32
cutxtop = 0 #10
plt.imshow(se['data'][np.floor(xlen/2)-xval+delx+cutxtop:np.floor(xlen/2)+xval+delx-cutx,np.floor(ylen/2)-yval+dely:np.floor(ylen/2)+yval+dely],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(ax00.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
ax00.add_artist(sbar)

ax0000 = plt.subplot2grid((nolines,noplots), (0,8), colspan=1, rowspan=1)
se = np.load('V0pt5b' +'SEchannel.npz',mmap_mode='r')
xlen = se['data'].shape[0]
ylen = se['data'].shape[1] 
delx = 0#+28
dely = 0#00
xval = 80
yval = 86
plt.imshow(se['data'][np.floor(xlen/2)-xval+delx:np.floor(xlen/2)+xval+delx,np.floor(ylen/2)-yval+dely:np.floor(ylen/2)+yval+dely],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(ax0000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
ax0000.add_artist(sbar)

ax00000 = plt.subplot2grid((nolines,noplots), (0,11), colspan=1, rowspan=1)
se = np.load('V0pt75' +'SEchannel.npz',mmap_mode='r') 
xlen = se['data'].shape[0]
ylen = se['data'].shape[1]
delx = 0#+28
dely = 0
xval = 125
yval = 97
cutx = 0 #75
plt.imshow(se['data'][np.floor(xlen/2)-xval+delx:np.floor(xlen/2)+xval+delx-cutx,np.floor(ylen/2)-yval+dely:np.floor(ylen/2)+yval+dely],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(ax00000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
ax00000.add_artist(sbar)

######## OLD DSET PICS
length_scalebar = 100.0 #in nm 
scalebar_legend = '100nm'
Pixel_size = 2.2
length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size))
   
letold = ['V0pt25' ,'V0pt5', 'V0', 'V0pt75','V1']

axb0 = plt.subplot2grid((nolines,noplots), (0,1), colspan=1, rowspan=1)
se = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/'+ letold[0] +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(axb0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
axb0.add_artist(sbar)

axb0 = plt.subplot2grid((nolines,noplots), (0,5), colspan=1, rowspan=1)
se = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/'+ letold[1] +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(axb0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
axb0.add_artist(sbar)

axb0 = plt.subplot2grid((nolines,noplots), (0,7), colspan=1, rowspan=1)
se = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/'+ letold[2] +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(axb0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
axb0.add_artist(sbar)

axb0 = plt.subplot2grid((nolines,noplots), (0,9), colspan=1, rowspan=1)
se = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/'+ letold[3] +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(axb0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
axb0.add_artist(sbar)

axb0 = plt.subplot2grid((nolines,noplots), (0,12), colspan=1, rowspan=1)
se = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/'+ letold[4] +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(axb0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
axb0.add_artist(sbar)

######## 

length_scalebar = 100.0 #in nm 
scalebar_legend = '100nm'
Pixel_size = 2.5
length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size))

leto = ['N102pt5', 'N92pt9', 'N66pt4', 'N72pt8', 'N60pt6', 'N48pt5', 'N39pt8', 'N31pt2']

axc0 = plt.subplot2grid((nolines,noplots), (0,3), colspan=1, rowspan=1)
se = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[4] +'SEchannel.npz',mmap_mode='r') 
xlen = se['data'].shape[0]
ylen = se['data'].shape[1] 
delx = 0#+28
dely = 0#00
xval = 144
yval = 142
plt.imshow(se['data'][np.floor(xlen/2)-xval+delx:np.floor(xlen/2)+xval+delx,np.floor(ylen/2)-yval+dely:np.floor(ylen/2)+yval+dely],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(axc0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
axc0.add_artist(sbar)

axc1 = plt.subplot2grid((nolines,noplots), (0,4), colspan=1, rowspan=1)
se = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[5] +'SEchannel.npz',mmap_mode='r') 
delx = 0#+28
dely = 0#00
xval = 133
yval = 122
plt.imshow(se['data'][np.floor(xlen/2)-xval+delx:np.floor(xlen/2)+xval+delx,np.floor(ylen/2)-yval+dely:np.floor(ylen/2)+yval+dely],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(axc1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
axc1.add_artist(sbar)

axc2 = plt.subplot2grid((nolines,noplots), (0,6), colspan=1, rowspan=1)
se = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[6] +'SEchannel.npz',mmap_mode='r') 
delx = 0#+28
dely = 0#00
xval = 135
yval = 105
plt.imshow(se['data'][np.floor(xlen/2)-xval+delx:np.floor(xlen/2)+xval+delx,np.floor(ylen/2)-yval+dely:np.floor(ylen/2)+yval+dely],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(axc2.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
axc2.add_artist(sbar)

axc3 = plt.subplot2grid((nolines,noplots), (0,10), colspan=1, rowspan=1)
se = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[7] +'SEchannel.npz',mmap_mode='r')
delx = 0#+28
dely = 0#00
xval = 144
yval = 120
plt.imshow(se['data'][np.floor(xlen/2)-xval+delx:np.floor(xlen/2)+xval+delx,np.floor(ylen/2)-yval+dely:np.floor(ylen/2)+yval+dely],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(axc3.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
axc3.add_artist(sbar)

#########

################### deleting data points first dset
todel = [0,3,6]#[0,1,3,6]#[0,1] #
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
il_data_std = il_data_std/np.sqrt(5) #taking into account that each il_data comes from 5 different averages

################### deleting data points third dset
todel = [0,1,2,3]#  [0,1,2,3]
red_int_array3 = np.delete(red_int_array3, todel)
blue_int_array3 = np.delete(blue_int_array3, todel)
red_std_array3 = np.delete(red_std_array3, todel)
blue_std_array3 = np.delete(blue_std_array3, todel)
b_array_red3 = np.delete(b_array_red3, todel)
be_array_red3 = np.delete(be_array_red3, todel)
e_array_red3 = np.delete(e_array_red3, todel)
ee_array_red3 = np.delete(ee_array_red3, todel)
b_array_blue3 = np.delete(b_array_blue3, todel)
be_array_blue3 = np.delete(be_array_blue3, todel)
e_array_blue3 = np.delete(e_array_blue3, todel)
ee_array_blue3 = np.delete(ee_array_blue3, todel)
il_data3 = np.delete(il_data3, todel)
il_data_std3 = np.delete(il_data_std3, todel)
il_data_std3 = il_data_std3/np.sqrt(4) #taking into account that each il_data comes from 5 different averages

#####################################################

ax2 = plt.subplot2grid((nolines,noplots), (1,0), colspan=noplots, rowspan=1)
ax1 = plt.subplot2grid((nolines,noplots), (3,0), colspan=noplots, sharex=ax2)
ax3 = plt.subplot2grid((nolines,noplots), (2,0), colspan=noplots, sharex=ax2)


ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')

x_vec = il_data
ax2.errorbar(x_vec, b_array_red, yerr=be_array_red,  fmt='ro',markersize=5,ls='-')
ax2.errorbar(x_vec, e_array_red, yerr=ee_array_red, fmt='ro', markersize=10,ls='-',label='largest zoom')
ax2.errorbar(x_vec, b_array_blue, yerr=be_array_blue, fmt='go',markersize=5,ls='-')
ax2.errorbar(x_vec, e_array_blue, yerr=ee_array_blue, fmt='go', markersize=10,ls='-')

x_vec3 = il_data3
ax2.errorbar(x_vec3, b_array_red3, yerr=be_array_red3,  fmt='rs',markersize=5,ls='--')
ax2.errorbar(x_vec3, e_array_red3, yerr=ee_array_red3, fmt='rs', markersize=10,ls='--',label='medium zoom')
ax2.errorbar(x_vec3, b_array_blue3, yerr=be_array_blue3, fmt='gs',markersize=5,ls='--')
ax2.errorbar(x_vec3, e_array_blue3, yerr=ee_array_blue3, fmt='gs', markersize=10,ls='--')

x_vec2 = il_data2
ax2.errorbar(x_vec2, b_array_red2, yerr=be_array_red2,  fmt='r^',markersize=5,ls='dotted')
ax2.errorbar(x_vec2, e_array_red2, yerr=ee_array_red2, fmt='r^', markersize=10,ls='dotted',label='smallest zoom')
ax2.errorbar(x_vec2, b_array_blue2, yerr=be_array_blue2, fmt='g^',markersize=5,ls='dotted')
ax2.errorbar(x_vec2, e_array_blue2, yerr=ee_array_blue2, fmt='g^', markersize=10,ls='dotted')


ax2.set_ylabel('Time constants ($\mu$s)',fontsize=fsizepl)
ax2.tick_params(labelsize=fsizenb)
ax1.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)

plt.sca(ax1) 

plt.setp(ax2.get_xticklabels(), visible=False)
ax2.legend(loc='best')
#ax2.set_ylim([0,300])
###
### error
import uncertainties as unc  
import uncertainties.unumpy as unumpy  

u_red_int_array=unumpy.uarray(( red_int_array, red_std_array ))  
u_blue_int_array=unumpy.uarray(( blue_int_array, blue_std_array ))  
#calc intensity ratios
ratio = (u_red_int_array/u_blue_int_array)/(u_red_int_array[1]/u_blue_int_array[1])
unumpy_error_ratio = unumpy.std_devs(ratio) 
ratio2 = (red_int_array/blue_int_array)/(red_int_array[1]/blue_int_array[1])
ratioun = (u_red_int_array/u_blue_int_array)
unumpy_error_ratioun = unumpy.std_devs(ratioun) 
#ax1.errorbar(x_vec, ratio2, yerr=unumpy_error_ratio, fmt='ko',markersize=10,label='normalized by RT')
#ax1.errorbar(x_vec, red_int_array/blue_int_array, yerr=unumpy_error_ratioun, fmt='ko',markersize=5,label='unnormalized')
ax1.errorbar(x_vec, ratio2, fmt='ko',markersize=10,label='normalized by RT',ls='-')
ax1.errorbar(x_vec, red_int_array/blue_int_array, fmt='ko',markersize=5,label='unnormalized',ls='-')

ax3.errorbar(x_vec, (red_int_array-blue_int_array)/(red_int_array+blue_int_array)/( (red_int_array[1]-blue_int_array[1])/(red_int_array[1]+blue_int_array[1])), fmt='ko',markersize=10,label='normalized by RT',ls='-')
ax3.errorbar(x_vec, (red_int_array-blue_int_array)/(red_int_array+blue_int_array), fmt='ko',markersize=5,label='unnormalized',ls='-')
ax3.errorbar(x_vec2, (red_int_array2-blue_int_array2)/(red_int_array2+blue_int_array2), fmt='k^',markersize=5,ls='dotted')
ax3.errorbar(x_vec2, (red_int_array2-blue_int_array2)/(red_int_array2+blue_int_array2)/( (red_int_array2[0]-blue_int_array2[0])/(red_int_array2[0]+blue_int_array2[0])), fmt='k^',markersize=10,ls='dotted')
ax3.errorbar(x_vec3, (red_int_array3-blue_int_array3)/(red_int_array3+blue_int_array3), fmt='ks',markersize=5,ls='--')
ax3.errorbar(x_vec3, (red_int_array3-blue_int_array3)/(red_int_array3+blue_int_array3)/( (red_int_array3[-1]-blue_int_array3[-1])/(red_int_array3[-1]+blue_int_array3[-1])), fmt='ks',markersize=10,ls='--')
ax3.set_ylabel('Red to green \n visibility',fontsize=fsizepl)
ax3.tick_params(labelsize=fsizenb)
ax3.legend(loc='best')
plt.setp(ax3.get_xticklabels(), visible=False)
#plt.legend(loc ='best')


u_red_int_array2=unumpy.uarray(( red_int_array2, red_std_array2 ))  
u_blue_int_array2=unumpy.uarray(( blue_int_array2, blue_std_array2 ))  
#calc intensity ratios
ratio22 = (u_red_int_array2/u_blue_int_array2)/(u_red_int_array2[0]/u_blue_int_array2[0])
unumpy_error_ratio22 = unumpy.std_devs(ratio22) 
ratio222 = (red_int_array2/blue_int_array2)/(red_int_array2[0]/blue_int_array2[0])
ratioun2 = (u_red_int_array2/u_blue_int_array2)
unumpy_error_ratioun2 = unumpy.std_devs(ratioun2) 
#ax1.errorbar(x_vec2, ratio222, yerr=unumpy_error_ratio22,fmt='k^',markersize=10)
#ax1.errorbar(x_vec2, red_int_array2/blue_int_array2, yerr=unumpy_error_ratioun2, fmt='k^',markersize=5)
ax1.errorbar(x_vec2, ratio222, fmt='k^',markersize=10,ls='dotted')
ax1.errorbar(x_vec2, red_int_array2/blue_int_array2,fmt='k^',markersize=5,ls='--')

u_red_int_array3=unumpy.uarray(( red_int_array3, red_std_array3 ))  
u_blue_int_array3=unumpy.uarray(( blue_int_array3, blue_std_array3 ))  
#calc intensity ratios
ratio33 = (u_red_int_array3/u_blue_int_array3)/(u_red_int_array3[-1]/u_blue_int_array3[-1])
unumpy_error_ratio33 = unumpy.std_devs(ratio33) 
ratio333 = (red_int_array3/blue_int_array3)/(red_int_array3[-1]/blue_int_array3[-1])
ratioun3 = (u_red_int_array3/u_blue_int_array3)
unumpy_error_ratioun3 = unumpy.std_devs(ratioun3) 
ax1.errorbar(x_vec3, ratio333, fmt='ks',markersize=10,ls='--')
ax1.errorbar(x_vec3, red_int_array3/blue_int_array3,fmt='ks',markersize=5,ls='--')

ax1.set_ylabel('Red to green \n intensity ratios',fontsize=fsizepl)
plt.tick_params(labelsize=fsizenb)
plt.xlim([tminx,tmaxx])
plt.legend(loc ='best')
#ax1.set_ylim([0,1.05])




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

ax2.errorbar(x_vec2, e_array_red2/b_array_red2,  fmt='r^',markersize=10,ls='dotted')
ax2.errorbar(x_vec2, e_array_blue2/b_array_blue2, fmt='g^', markersize=10,ls='dotted')
ax2.errorbar(x_vec2, e_array_red2/e_array_blue2,  fmt='k^',markersize=10,ls='dotted')
ax2.errorbar(x_vec2, b_array_red2/b_array_blue2, fmt='k^', markersize=7,ls='dotted')

ax2.errorbar(x_vec3, e_array_red3/b_array_red3,  fmt='rs',markersize=10,ls='--')
ax2.errorbar(x_vec3, e_array_blue3/b_array_blue3, fmt='gs', markersize=10,ls='--')
ax2.errorbar(x_vec3, e_array_red3/e_array_blue3,  fmt='ks',markersize=10,ls='--')
ax2.errorbar(x_vec3, b_array_red3/b_array_blue3, fmt='ks', markersize=7,ls='--')

ax2.set_ylabel('Time constant ratios',fontsize=fsizepl)
ax2.tick_params(labelsize=fsizenb)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.legend(loc='best')

ax1 = plt.subplot2grid((2,4), (1,0), colspan=2, sharex=ax2)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax1.errorbar(x_vec, e_array_red/b_array_red/(e_array_red[1]/b_array_red[1]),  fmt='ro',markersize=10,ls='-')
ax1.errorbar(x_vec, e_array_blue/b_array_blue/(e_array_blue[1]/b_array_blue[1]), fmt='go', markersize=10,ls='-')
ax1.errorbar(x_vec, e_array_red/e_array_blue/(e_array_red[1]/e_array_blue[1]),  fmt='ko',markersize=10,ls='-')
ax1.errorbar(x_vec, b_array_red/b_array_blue/(b_array_red[1]/b_array_blue[1]), fmt='ko', markersize=7,ls='-')

ax1.errorbar(x_vec2, e_array_red2/b_array_red2/(e_array_red2[0]/b_array_red2[0]),  fmt='r^',markersize=10,ls='dotted')
ax1.errorbar(x_vec2, e_array_blue2/b_array_blue2/(e_array_blue2[0]/b_array_blue2[0]), fmt='g^', markersize=10,ls='dotted')
ax1.errorbar(x_vec2, e_array_red2/e_array_blue2/(e_array_red2[0]/e_array_blue2[0]),  fmt='k^',markersize=10,ls='dotted')
ax1.errorbar(x_vec2, b_array_red2/b_array_blue2/(b_array_red2[0]/b_array_blue2[0]), fmt='k^', markersize=7,ls='dotted')

ax1.errorbar(x_vec3, e_array_red3/b_array_red3/(e_array_red3[-1]/b_array_red3[-1]),  fmt='rs',markersize=10,ls='--')
ax1.errorbar(x_vec3, e_array_blue3/b_array_blue3/(e_array_blue3[-1]/b_array_blue3[-1]), fmt='gs', markersize=10,ls='--')
ax1.errorbar(x_vec3, e_array_red3/e_array_blue3/(e_array_red3[-1]/e_array_blue3[-1]),  fmt='ks',markersize=10,ls='--')
ax1.errorbar(x_vec3, b_array_red3/b_array_blue3/(b_array_red3[-1]/b_array_blue3[-1]), fmt='ks', markersize=7,ls='--')

ax1.set_ylabel('Time constant ratios,\n normalized by RT',fontsize=fsizepl)
ax1.tick_params(labelsize=fsizenb)
ax1.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)
plt.xlim([tminx,tmaxx])

ax2 = plt.subplot2grid((2,4), (0,2), colspan=2, rowspan=1)

ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('right')
ax2.yaxis.set_label_position("right")

Ps = 0.89
ax2.errorbar(x_vec, red_int_array/Ps/Ps, fmt='ro',markersize=10, label='kHz/nm$^2$',ls='-')
ax2.errorbar(x_vec, blue_int_array/Ps/Ps, fmt='go',markersize=10, label='kHz/nm$^2$',ls='-')

Ps = 2.2
ax2.errorbar(x_vec2, red_int_array2/Ps/Ps, fmt='r^',markersize=10, label='kHz/nm$^2$',ls='dotted')
ax2.errorbar(x_vec2, blue_int_array2/Ps/Ps, fmt='g^',markersize=10, label='kHz/nm$^2$',ls='dotted')

Ps = 2.5
ax2.errorbar(x_vec3, red_int_array3/Ps/Ps, fmt='rs',markersize=10, label='kHz/nm$^2$',ls='--')
ax2.errorbar(x_vec3, blue_int_array3/Ps/Ps, fmt='gs',markersize=10, label='kHz/nm$^2$',ls='--')

ax2.set_ylabel('Intensities (kHz/nm$^2$)\n (light per area)',fontsize=fsizepl)
plt.tick_params(labelsize=fsizenb)
plt.setp(ax2.get_xticklabels(), visible=False)

ax1 = plt.subplot2grid((2,4), (1,2), colspan=2, sharex=ax2)

ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('right')
ax1.yaxis.set_label_position("right")

no_pixels = 250.0*250.0
ax1.errorbar(x_vec, red_int_array*no_pixels/1.0e3, fmt='ro',markersize=10, label='MHz/frame$',ls='-')
ax1.errorbar(x_vec, blue_int_array*no_pixels/1.0e3, fmt='go',markersize=10, label='MHz/frame$',ls='-')

no_pixels = 500.0*500.0
ax1.errorbar(x_vec2, red_int_array2*no_pixels/1.0e3, fmt='r^',markersize=10, label='MHz/frame$',ls='dotted')
ax1.errorbar(x_vec2, blue_int_array2*no_pixels/1.0e3, fmt='g^',markersize=10, label='MHz/frame$',ls='dotted')

no_pixels = 300.0*300.0
ax1.errorbar(x_vec3, red_int_array3*no_pixels/1.0e3, fmt='rs',markersize=10, label='MHz/frame$',ls='--')
ax1.errorbar(x_vec3, blue_int_array3*no_pixels/1.0e3, fmt='gs',markersize=10, label='MHz/frame$',ls='--')

ax1.set_ylabel('Intensities (MHz/frame)\n (total collected light)',fontsize=fsizepl)
plt.tick_params(labelsize=fsizenb)
ax1.set_xlabel('Temperature at heater($^{\circ}$C)',fontsize=fsizepl)
plt.xlim([tminx,tmaxx])

################################## PAGE 3plots


fig45= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig45.set_size_inches(1200./fig42.dpi,900./fig42.dpi)

nolines = 4
noplots = 1

ax4 = plt.subplot2grid((nolines,noplots), (1,0), colspan=noplots, sharex=ax2)
ax2 = plt.subplot2grid((nolines,noplots), (0,0), colspan=noplots, rowspan=1)
ax1 = plt.subplot2grid((nolines,noplots), (3,0), colspan=noplots, sharex=ax2)
ax3 = plt.subplot2grid((nolines,noplots), (2,0), colspan=noplots, sharex=ax2)


ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')

ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.xaxis.set_ticks_position('bottom')
ax4.yaxis.set_ticks_position('left')

x_vec = il_data
ax4.errorbar(x_vec, b_array_red, yerr=be_array_red,  fmt='ro',markersize=5,ls='-')
ax2.errorbar(x_vec, e_array_red, yerr=ee_array_red, fmt='ro', markersize=10,ls='-',label='largest zoom')
ax4.errorbar(x_vec, b_array_blue, yerr=be_array_blue, fmt='go',markersize=5,ls='-')
ax2.errorbar(x_vec, e_array_blue, yerr=ee_array_blue, fmt='go', markersize=10,ls='-')

x_vec3 = il_data3
ax4.errorbar(x_vec3, b_array_red3, yerr=be_array_red3,  fmt='rs',markersize=5,ls='--')
ax2.errorbar(x_vec3, e_array_red3, yerr=ee_array_red3, fmt='rs', markersize=10,ls='--',label='medium zoom')
ax4.errorbar(x_vec3, b_array_blue3, yerr=be_array_blue3, fmt='gs',markersize=5,ls='--')
ax2.errorbar(x_vec3, e_array_blue3, yerr=ee_array_blue3, fmt='gs', markersize=10,ls='--')

x_vec2 = il_data2
ax4.errorbar(x_vec2, b_array_red2, yerr=be_array_red2,  fmt='r^',markersize=5,ls='dotted')
ax2.errorbar(x_vec2, e_array_red2, yerr=ee_array_red2, fmt='r^', markersize=10,ls='dotted',label='smallest zoom')
ax4.errorbar(x_vec2, b_array_blue2, yerr=be_array_blue2, fmt='g^',markersize=5,ls='dotted')
ax2.errorbar(x_vec2, e_array_blue2, yerr=ee_array_blue2, fmt='g^', markersize=10,ls='dotted')


ax2.set_ylabel('Long time \n constants \n ($\mu$s)',fontsize=fsizepl)
ax2.tick_params(labelsize=fsizenb)
ax1.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)


ax4.set_ylabel('Short time \n constants \n ($\mu$s)',fontsize=fsizepl)
ax4.tick_params(labelsize=fsizenb)


plt.sca(ax1) 

plt.setp(ax2.get_xticklabels(), visible=False)
ax2.legend(loc='best')
#ax2.set_ylim([0,300])
###
### error
import uncertainties as unc  
import uncertainties.unumpy as unumpy  

u_red_int_array=unumpy.uarray(( red_int_array, red_std_array ))  
u_blue_int_array=unumpy.uarray(( blue_int_array, blue_std_array ))  
#calc intensity ratios
ratio = (u_red_int_array/u_blue_int_array)/(u_red_int_array[1]/u_blue_int_array[1])
unumpy_error_ratio = unumpy.std_devs(ratio) 
ratio2 = (red_int_array/blue_int_array)/(red_int_array[1]/blue_int_array[1])
ratioun = (u_red_int_array/u_blue_int_array)
unumpy_error_ratioun = unumpy.std_devs(ratioun) 
#ax1.errorbar(x_vec, ratio2, yerr=unumpy_error_ratio, fmt='ko',markersize=10,label='normalized by RT')
#ax1.errorbar(x_vec, red_int_array/blue_int_array, yerr=unumpy_error_ratioun, fmt='ko',markersize=5,label='unnormalized')
ax1.errorbar(x_vec, ratio2, fmt='ko',markersize=10,label='normalized by RT',ls='-')
#ax1.errorbar(x_vec, red_int_array/blue_int_array, fmt='ko',markersize=5,label='unnormalized',ls='-')

ax3.errorbar(x_vec, (red_int_array-blue_int_array)/(red_int_array+blue_int_array)/( (red_int_array[1]-blue_int_array[1])/(red_int_array[1]+blue_int_array[1])), fmt='ko',markersize=10,label='normalized by RT',ls='-')
#ax3.errorbar(x_vec, (red_int_array-blue_int_array)/(red_int_array+blue_int_array), fmt='ko',markersize=5,label='unnormalized',ls='-')
#ax3.errorbar(x_vec2, (red_int_array2-blue_int_array2)/(red_int_array2+blue_int_array2), fmt='k^',markersize=5,ls='dotted')
ax3.errorbar(x_vec2, (red_int_array2-blue_int_array2)/(red_int_array2+blue_int_array2)/( (red_int_array2[0]-blue_int_array2[0])/(red_int_array2[0]+blue_int_array2[0])), fmt='k^',markersize=10,ls='dotted')
#ax3.errorbar(x_vec3, (red_int_array3-blue_int_array3)/(red_int_array3+blue_int_array3), fmt='ks',markersize=5,ls='--')
ax3.errorbar(x_vec3, (red_int_array3-blue_int_array3)/(red_int_array3+blue_int_array3)/( (red_int_array3[-1]-blue_int_array3[-1])/(red_int_array3[-1]+blue_int_array3[-1])), fmt='ks',markersize=10,ls='--')
ax3.set_ylabel('Red to green \n visibility',fontsize=fsizepl)
ax3.tick_params(labelsize=fsizenb)
ax3.legend(loc='best')
plt.setp(ax3.get_xticklabels(), visible=False)
#plt.legend(loc ='best')


u_red_int_array2=unumpy.uarray(( red_int_array2, red_std_array2 ))  
u_blue_int_array2=unumpy.uarray(( blue_int_array2, blue_std_array2 ))  
#calc intensity ratios
ratio22 = (u_red_int_array2/u_blue_int_array2)/(u_red_int_array2[0]/u_blue_int_array2[0])
unumpy_error_ratio22 = unumpy.std_devs(ratio22) 
ratio222 = (red_int_array2/blue_int_array2)/(red_int_array2[0]/blue_int_array2[0])
ratioun2 = (u_red_int_array2/u_blue_int_array2)
unumpy_error_ratioun2 = unumpy.std_devs(ratioun2) 
#ax1.errorbar(x_vec2, ratio222, yerr=unumpy_error_ratio22,fmt='k^',markersize=10)
#ax1.errorbar(x_vec2, red_int_array2/blue_int_array2, yerr=unumpy_error_ratioun2, fmt='k^',markersize=5)
ax1.errorbar(x_vec2, ratio222, fmt='k^',markersize=10,ls='dotted')
#ax1.errorbar(x_vec2, red_int_array2/blue_int_array2,fmt='k^',markersize=5,ls='--')

u_red_int_array3=unumpy.uarray(( red_int_array3, red_std_array3 ))  
u_blue_int_array3=unumpy.uarray(( blue_int_array3, blue_std_array3 ))  
#calc intensity ratios
ratio33 = (u_red_int_array3/u_blue_int_array3)/(u_red_int_array3[-1]/u_blue_int_array3[-1])
unumpy_error_ratio33 = unumpy.std_devs(ratio33) 
#ratio333 = (red_int_array3/blue_int_array3)/(red_int_array3[3]/blue_int_array3[3])
ratioun3 = (u_red_int_array3/u_blue_int_array3)
unumpy_error_ratioun3 = unumpy.std_devs(ratioun3) 
ax1.errorbar(x_vec3, ratio333, fmt='ks',markersize=10,ls='--')
#ax1.errorbar(x_vec3, red_int_array3/blue_int_array3,fmt='ks',markersize=5,ls='--')

ax1.set_ylabel('Red to green \n intensity ratios',fontsize=fsizepl)
plt.tick_params(labelsize=fsizenb)
plt.xlim([tminx,tmaxx])
plt.legend(loc ='best')
#ax1.set_ylim([0,1.05])

################################## PAGE  4 plots

area = [0.89*0.89, 2.2*2.2, 2.5*2.5]


fig45= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig45.set_size_inches(1200./fig42.dpi,900./fig42.dpi)

nolines = 4
noplots = 1

ax4 = plt.subplot2grid((nolines,noplots), (1,0), colspan=noplots, sharex=ax2)
ax2 = plt.subplot2grid((nolines,noplots), (0,0), colspan=noplots, rowspan=1)
ax1 = plt.subplot2grid((nolines,noplots), (3,0), colspan=noplots, sharex=ax2)
ax3 = plt.subplot2grid((nolines,noplots), (2,0), colspan=noplots, sharex=ax2)


ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')

ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.xaxis.set_ticks_position('bottom')
ax4.yaxis.set_ticks_position('left')

x_vec = il_data
ax4.errorbar(x_vec, b_array_red/area[0], yerr=be_array_red,  fmt='ro',markersize=5,ls='-')
ax2.errorbar(x_vec, e_array_red/area[0], yerr=ee_array_red, fmt='ro', markersize=10,ls='-',label='largest zoom')
ax4.errorbar(x_vec, b_array_blue/area[0], yerr=be_array_blue, fmt='go',markersize=5,ls='-')
ax2.errorbar(x_vec, e_array_blue/area[0], yerr=ee_array_blue, fmt='go', markersize=10,ls='-')

x_vec3 = il_data3
ax4.errorbar(x_vec3, b_array_red3/area[2], yerr=be_array_red3,  fmt='rs',markersize=5,ls='--')
ax2.errorbar(x_vec3, e_array_red3/area[2], yerr=ee_array_red3, fmt='rs', markersize=10,ls='--',label='medium zoom')
ax4.errorbar(x_vec3, b_array_blue3/area[2], yerr=be_array_blue3, fmt='gs',markersize=5,ls='--')
ax2.errorbar(x_vec3, e_array_blue3/area[2], yerr=ee_array_blue3, fmt='gs', markersize=10,ls='--')

x_vec2 = il_data2
ax4.errorbar(x_vec2, b_array_red2/area[1], yerr=be_array_red2,  fmt='r^',markersize=5,ls='dotted')
ax2.errorbar(x_vec2, e_array_red2/area[1], yerr=ee_array_red2, fmt='r^', markersize=10,ls='dotted',label='smallest zoom')
ax4.errorbar(x_vec2, b_array_blue2/area[1], yerr=be_array_blue2, fmt='g^',markersize=5,ls='dotted')
ax2.errorbar(x_vec2, e_array_blue2/area[1], yerr=ee_array_blue2, fmt='g^', markersize=10,ls='dotted')


ax2.set_ylabel('Long time \n csts per area \n ($\mu$s/nm$^2$)',fontsize=fsizepl)
ax2.tick_params(labelsize=fsizenb)
ax1.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)


ax4.set_ylabel('Short time \n csts per area \n ($\mu$s/nm$^2$)',fontsize=fsizepl)
ax4.tick_params(labelsize=fsizenb)


plt.sca(ax1) 

plt.setp(ax2.get_xticklabels(), visible=False)
ax2.legend(loc='best')
#ax2.set_ylim([0,300])
###
### error
import uncertainties as unc  
import uncertainties.unumpy as unumpy  

u_red_int_array=unumpy.uarray(( red_int_array, red_std_array ))  
u_blue_int_array=unumpy.uarray(( blue_int_array, blue_std_array ))  
#calc intensity ratios
ratio = (u_red_int_array/u_blue_int_array)/(u_red_int_array[1]/u_blue_int_array[1])
unumpy_error_ratio = unumpy.std_devs(ratio) 
ratio2 = (red_int_array/blue_int_array)/(red_int_array[1]/blue_int_array[1])
ratioun = (u_red_int_array/u_blue_int_array)
unumpy_error_ratioun = unumpy.std_devs(ratioun) 
#ax1.errorbar(x_vec, ratio2/area[0], yerr=unumpy_error_ratio, fmt='ko',markersize=10,label='normalized by RT')
#ax1.errorbar(x_vec, red_int_array/blue_int_array, yerr=unumpy_error_ratioun, fmt='ko',markersize=5,label='unnormalized')
ax1.errorbar(x_vec, ratio2/area[0], fmt='ko',markersize=10,label='normalized by RT',ls='-')
#ax1.errorbar(x_vec, red_int_array/blue_int_array, fmt='ko',markersize=5,label='unnormalized',ls='-')

ax3.errorbar(x_vec, (red_int_array-blue_int_array)/(red_int_array+blue_int_array)/( (red_int_array[1]-blue_int_array[1])/(red_int_array[1]+blue_int_array[1]))/area[0], fmt='ko',markersize=10,label='normalized by RT',ls='-')
#ax3.errorbar(x_vec, (red_int_array-blue_int_array)/(red_int_array+blue_int_array), fmt='ko',markersize=5,label='unnormalized',ls='-')
#ax3.errorbar(x_vec2, (red_int_array2-blue_int_array2)/(red_int_array2+blue_int_array2), fmt='k^',markersize=5,ls='dotted')
ax3.errorbar(x_vec2, (red_int_array2-blue_int_array2)/(red_int_array2+blue_int_array2)/( (red_int_array2[0]-blue_int_array2[0])/(red_int_array2[0]+blue_int_array2[0]))/area[1], fmt='k^',markersize=10,ls='dotted')
#ax3.errorbar(x_vec3, (red_int_array3-blue_int_array3)/(red_int_array3+blue_int_array3), fmt='ks',markersize=5,ls='--')
ax3.errorbar(x_vec3, (red_int_array3-blue_int_array3)/(red_int_array3+blue_int_array3)/( (red_int_array3[-1]-blue_int_array3[-1])/(red_int_array3[-1]+blue_int_array3[-1]))/area[2], fmt='ks',markersize=10,ls='--')
ax3.set_ylabel('Red to green \n visibility \n per area (1/nm$^2$)',fontsize=fsizepl)
ax3.tick_params(labelsize=fsizenb)
ax3.legend(loc='best')
plt.setp(ax3.get_xticklabels(), visible=False)
#plt.legend(loc ='best')


u_red_int_array2=unumpy.uarray(( red_int_array2, red_std_array2 ))  
u_blue_int_array2=unumpy.uarray(( blue_int_array2, blue_std_array2 ))  
#calc intensity ratios
ratio22 = (u_red_int_array2/u_blue_int_array2)/(u_red_int_array2[0]/u_blue_int_array2[0])
unumpy_error_ratio22 = unumpy.std_devs(ratio22) 
ratio222 = (red_int_array2/blue_int_array2)/(red_int_array2[0]/blue_int_array2[0])
ratioun2 = (u_red_int_array2/u_blue_int_array2)
unumpy_error_ratioun2 = unumpy.std_devs(ratioun2) 
#ax1.errorbar(x_vec2, ratio222, yerr=unumpy_error_ratio22,fmt='k^',markersize=10)
#ax1.errorbar(x_vec2, red_int_array2/blue_int_array2, yerr=unumpy_error_ratioun2, fmt='k^',markersize=5)
ax1.errorbar(x_vec2, ratio222, fmt='k^',markersize=10,ls='dotted')
#ax1.errorbar(x_vec2, red_int_array2/blue_int_array2/area[1],fmt='k^',markersize=5,ls='--')

u_red_int_array3=unumpy.uarray(( red_int_array3, red_std_array3 ))  
u_blue_int_array3=unumpy.uarray(( blue_int_array3, blue_std_array3 ))  
#calc intensity ratios
ratio33 = (u_red_int_array3/u_blue_int_array3)/(u_red_int_array3[-1]/u_blue_int_array3[-1])
unumpy_error_ratio33 = unumpy.std_devs(ratio33) 
#ratio333 = (red_int_array3/blue_int_array3)/(red_int_array3[3]/blue_int_array3[3])
ratioun3 = (u_red_int_array3/u_blue_int_array3)
unumpy_error_ratioun3 = unumpy.std_devs(ratioun3) 
ax1.errorbar(x_vec3, ratio333/area[2], fmt='ks',markersize=10,ls='--')
#ax1.errorbar(x_vec3, red_int_array3/blue_int_array3/area[2],fmt='ks',markersize=5,ls='--')

ax1.set_ylabel('Red to green \n intensity ratios \n per area (1/nm$^2$)',fontsize=fsizepl)
plt.tick_params(labelsize=fsizenb)
plt.xlim([tminx,tmaxx])
plt.legend(loc ='best')
#ax1.set_ylim([0,1.05])

############  new PAGE 5 PLOTS
###############################################################################

fig42= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig42.set_size_inches(1200./fig42.dpi,900./fig42.dpi)

x_vec = il_data

ax2 = plt.subplot2grid((2,4), (0,0), colspan=2, rowspan=1)

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

ax2.errorbar(x_vec, e_array_red/b_array_red/area[0],  fmt='ro',markersize=10,ls='-')
ax2.errorbar(x_vec, e_array_blue/b_array_blue/area[0], fmt='go', markersize=10,ls='-')
ax2.errorbar(x_vec, e_array_red/e_array_blue/area[0],  fmt='ko',markersize=10,ls='-',label='red/green long')
ax2.errorbar(x_vec, b_array_red/b_array_blue/area[0], fmt='ko', markersize=7,ls='-',label='red/green short')

ax2.errorbar(x_vec2, e_array_red2/b_array_red2/area[1],  fmt='r^',markersize=10,ls='dotted')
ax2.errorbar(x_vec2, e_array_blue2/b_array_blue2/area[1], fmt='g^', markersize=10,ls='dotted')
ax2.errorbar(x_vec2, e_array_red2/e_array_blue2/area[1],  fmt='k^',markersize=10,ls='dotted')
ax2.errorbar(x_vec2, b_array_red2/b_array_blue2/area[1], fmt='k^', markersize=7,ls='dotted')

ax2.errorbar(x_vec3, e_array_red3/b_array_red3/area[2],  fmt='rs',markersize=10,ls='--')
ax2.errorbar(x_vec3, e_array_blue3/b_array_blue3/area[2], fmt='gs', markersize=10,ls='--')
ax2.errorbar(x_vec3, e_array_red3/e_array_blue3/area[2],  fmt='ks',markersize=10,ls='--')
ax2.errorbar(x_vec3, b_array_red3/b_array_blue3/area[2], fmt='ks', markersize=7,ls='--')

ax2.set_ylabel('Time constant ratios, \n norm by area (1/nm$^2$)',fontsize=fsizepl)
ax2.tick_params(labelsize=fsizenb)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.legend(loc='best')

ax1 = plt.subplot2grid((2,4), (1,0), colspan=2, sharex=ax2)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax1.errorbar(x_vec, e_array_red/b_array_red/(e_array_red[1]/b_array_red[1])/area[0],  fmt='ro',markersize=10,ls='-')
ax1.errorbar(x_vec, e_array_blue/b_array_blue/(e_array_blue[1]/b_array_blue[1])/area[0], fmt='go', markersize=10,ls='-')
ax1.errorbar(x_vec, e_array_red/e_array_blue/(e_array_red[1]/e_array_blue[1])/area[0],  fmt='ko',markersize=10,ls='-')
ax1.errorbar(x_vec, b_array_red/b_array_blue/(b_array_red[1]/b_array_blue[1])/area[0], fmt='ko', markersize=7,ls='-')

ax1.errorbar(x_vec2, e_array_red2/b_array_red2/(e_array_red2[0]/b_array_red2[0])/area[1],  fmt='r^',markersize=10,ls='dotted')
ax1.errorbar(x_vec2, e_array_blue2/b_array_blue2/(e_array_blue2[0]/b_array_blue2[0])/area[1], fmt='g^', markersize=10,ls='dotted')
ax1.errorbar(x_vec2, e_array_red2/e_array_blue2/(e_array_red2[0]/e_array_blue2[0])/area[1],  fmt='k^',markersize=10,ls='dotted')
ax1.errorbar(x_vec2, b_array_red2/b_array_blue2/(b_array_red2[0]/b_array_blue2[0])/area[1], fmt='k^', markersize=7,ls='dotted')

ax1.errorbar(x_vec3, e_array_red3/b_array_red3/(e_array_red3[-1]/b_array_red3[-1])/area[2],  fmt='rs',markersize=10,ls='--')
ax1.errorbar(x_vec3, e_array_blue3/b_array_blue3/(e_array_blue3[-1]/b_array_blue3[-1])/area[2], fmt='gs', markersize=10,ls='--')
ax1.errorbar(x_vec3, e_array_red3/e_array_blue3/(e_array_red3[-1]/e_array_blue3[-1])/area[2],  fmt='ks',markersize=10,ls='--')
ax1.errorbar(x_vec3, b_array_red3/b_array_blue3/(b_array_red3[-1]/b_array_blue3[-1])/area[2], fmt='ks', markersize=7,ls='--')

ax1.set_ylabel('Time constant ratios,\n normalized by RT and area (1/nm$^2$)',fontsize=fsizepl)
ax1.tick_params(labelsize=fsizenb)
ax1.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)
plt.xlim([tminx,tmaxx])



################################### PAGE  5 plots
#
#
########### inverting area!!!!
area = [1.0/(0.89*0.89), 1.0/(2.2*2.2), 1.0/(2.5*2.5)]


fig45= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig45.set_size_inches(1200./fig42.dpi,900./fig42.dpi)

nolines = 4
noplots = 1

ax4 = plt.subplot2grid((nolines,noplots), (1,0), colspan=noplots, sharex=ax2)
ax2 = plt.subplot2grid((nolines,noplots), (0,0), colspan=noplots, rowspan=1)
ax1 = plt.subplot2grid((nolines,noplots), (3,0), colspan=noplots, sharex=ax2)
ax3 = plt.subplot2grid((nolines,noplots), (2,0), colspan=noplots, sharex=ax2)


ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')

ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.xaxis.set_ticks_position('bottom')
ax4.yaxis.set_ticks_position('left')

x_vec = il_data
ax4.errorbar(x_vec, b_array_red/area[0], yerr=be_array_red,  fmt='ro',markersize=5,ls='-')
ax2.errorbar(x_vec, e_array_red/area[0], yerr=ee_array_red, fmt='ro', markersize=10,ls='-',label='largest zoom')
ax4.errorbar(x_vec, b_array_blue/area[0], yerr=be_array_blue, fmt='go',markersize=5,ls='-')
ax2.errorbar(x_vec, e_array_blue/area[0], yerr=ee_array_blue, fmt='go', markersize=10,ls='-')

x_vec3 = il_data3
ax4.errorbar(x_vec3, b_array_red3/area[2], yerr=be_array_red3,  fmt='rs',markersize=5,ls='--')
ax2.errorbar(x_vec3, e_array_red3/area[2], yerr=ee_array_red3, fmt='rs', markersize=10,ls='--',label='medium zoom')
ax4.errorbar(x_vec3, b_array_blue3/area[2], yerr=be_array_blue3, fmt='gs',markersize=5,ls='--')
ax2.errorbar(x_vec3, e_array_blue3/area[2], yerr=ee_array_blue3, fmt='gs', markersize=10,ls='--')

x_vec2 = il_data2
ax4.errorbar(x_vec2, b_array_red2/area[1], yerr=be_array_red2,  fmt='r^',markersize=5,ls='dotted')
ax2.errorbar(x_vec2, e_array_red2/area[1], yerr=ee_array_red2, fmt='r^', markersize=10,ls='dotted',label='smallest zoom')
ax4.errorbar(x_vec2, b_array_blue2/area[1], yerr=be_array_blue2, fmt='g^',markersize=5,ls='dotted')
ax2.errorbar(x_vec2, e_array_blue2/area[1], yerr=ee_array_blue2, fmt='g^', markersize=10,ls='dotted')


ax2.set_ylabel('Long time \n csts x area \n ($\mu$s x nm$^2$)',fontsize=fsizepl)
ax2.tick_params(labelsize=fsizenb)
ax1.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)


ax4.set_ylabel('Short time \n csts x area \n ($\mu$s x nm$^2$)',fontsize=fsizepl)
ax4.tick_params(labelsize=fsizenb)


plt.sca(ax1) 

plt.setp(ax2.get_xticklabels(), visible=False)
ax2.legend(loc='best')
#ax2.set_ylim([0,300])
###
### error
import uncertainties as unc  
import uncertainties.unumpy as unumpy  

u_red_int_array=unumpy.uarray(( red_int_array, red_std_array ))  
u_blue_int_array=unumpy.uarray(( blue_int_array, blue_std_array ))  
#calc intensity ratios
ratio = (u_red_int_array/u_blue_int_array)/(u_red_int_array[0]/u_blue_int_array[0])
unumpy_error_ratio = unumpy.std_devs(ratio) 
ratio2 = (red_int_array/blue_int_array)/(red_int_array[0]/blue_int_array[0])
ratioun = (u_red_int_array/u_blue_int_array)
unumpy_error_ratioun = unumpy.std_devs(ratioun) 
#ax1.errorbar(x_vec, ratio2/area[0], yerr=unumpy_error_ratio, fmt='ko',markersize=10,label='normalized by RT')
#ax1.errorbar(x_vec, red_int_array/blue_int_array, yerr=unumpy_error_ratioun, fmt='ko',markersize=5,label='unnormalized')
ax1.errorbar(x_vec, ratio2/area[0], fmt='ko',markersize=10,label='normalized by RT',ls='-')
#ax1.errorbar(x_vec, red_int_array/blue_int_array, fmt='ko',markersize=5,label='unnormalized',ls='-')

ax3.errorbar(x_vec, (red_int_array-blue_int_array)/(red_int_array+blue_int_array)/( (red_int_array[1]-blue_int_array[1])/(red_int_array[1]+blue_int_array[1]))/area[0], fmt='ko',markersize=10,label='normalized by RT',ls='-')
#ax3.errorbar(x_vec, (red_int_array-blue_int_array)/(red_int_array+blue_int_array), fmt='ko',markersize=5,label='unnormalized',ls='-')
#ax3.errorbar(x_vec2, (red_int_array2-blue_int_array2)/(red_int_array2+blue_int_array2), fmt='k^',markersize=5,ls='dotted')
ax3.errorbar(x_vec2, (red_int_array2-blue_int_array2)/(red_int_array2+blue_int_array2)/( (red_int_array2[0]-blue_int_array2[0])/(red_int_array2[0]+blue_int_array2[0]))/area[1], fmt='k^',markersize=10,ls='dotted')
#ax3.errorbar(x_vec3, (red_int_array3-blue_int_array3)/(red_int_array3+blue_int_array3), fmt='ks',markersize=5,ls='--')
ax3.errorbar(x_vec3, (red_int_array3-blue_int_array3)/(red_int_array3+blue_int_array3)/( (red_int_array3[-1]-blue_int_array3[-1])/(red_int_array3[-1]+blue_int_array3[-1]))/area[2], fmt='ks',markersize=10,ls='--')
ax3.set_ylabel('Red to green \n visibility \n x area (nm$^2$)',fontsize=fsizepl)
ax3.tick_params(labelsize=fsizenb)
ax3.legend(loc='best')
plt.setp(ax3.get_xticklabels(), visible=False)
#plt.legend(loc ='best')


u_red_int_array2=unumpy.uarray(( red_int_array2, red_std_array2 ))  
u_blue_int_array2=unumpy.uarray(( blue_int_array2, blue_std_array2 ))  
#calc intensity ratios
ratio22 = (u_red_int_array2/u_blue_int_array2)/(u_red_int_array2[0]/u_blue_int_array2[0])
unumpy_error_ratio22 = unumpy.std_devs(ratio22) 
ratio222 = (red_int_array2/blue_int_array2)/(red_int_array2[0]/blue_int_array2[0])
ratioun2 = (u_red_int_array2/u_blue_int_array2)
unumpy_error_ratioun2 = unumpy.std_devs(ratioun2) 
#ax1.errorbar(x_vec2, ratio222, yerr=unumpy_error_ratio22,fmt='k^',markersize=10)
#ax1.errorbar(x_vec2, red_int_array2/blue_int_array2, yerr=unumpy_error_ratioun2, fmt='k^',markersize=5)
ax1.errorbar(x_vec2, ratio222, fmt='k^',markersize=10,ls='dotted')
#ax1.errorbar(x_vec2, red_int_array2/blue_int_array2/area[1],fmt='k^',markersize=5,ls='--')

u_red_int_array3=unumpy.uarray(( red_int_array3, red_std_array3 ))  
u_blue_int_array3=unumpy.uarray(( blue_int_array3, blue_std_array3 ))  
#calc intensity ratios
ratio33 = (u_red_int_array3/u_blue_int_array3)/(u_red_int_array3[3]/u_blue_int_array3[3])
unumpy_error_ratio33 = unumpy.std_devs(ratio33) 
#ratio333 = (red_int_array3/blue_int_array3)/(red_int_array3[3]/blue_int_array3[3])
ratioun3 = (u_red_int_array3/u_blue_int_array3)
unumpy_error_ratioun3 = unumpy.std_devs(ratioun3) 
ax1.errorbar(x_vec3, ratio333/area[2], fmt='ks',markersize=10,ls='--')
#ax1.errorbar(x_vec3, red_int_array3/blue_int_array3/area[2],fmt='ks',markersize=5,ls='--')

ax1.set_ylabel('Red to green \n intensity ratios \n x area (nm$^2$)',fontsize=fsizepl)
plt.tick_params(labelsize=fsizenb)
plt.xlim([tminx,tmaxx])
plt.legend(loc ='best')
#ax1.set_ylim([0,1.05])

multipage_longer(prefix + 'ZZZZZZZZSummaryCOMBINED.pdf',dpi=80)