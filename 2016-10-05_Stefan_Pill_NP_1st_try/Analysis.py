import os
import sys
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../Python modules/") # necessary for the tex fonts
import scipy as sp
import scipy.misc
import matplotlib.pyplot as plt
import h5py
import numpy as np
from BackgroundCorrection import *
import matplotlib.cm as cm
import scipy.ndimage as ndimage
#from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from MakePdf import *
from PlottingFcts import *


import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = DeprecationWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.simplefilter(action = "ignore", category = PendingDeprecationWarning)
#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from Registration import reg_images

from sklearn.mixture import GMM 
import matplotlib.cm as cm

#import scalebars as sb

No_pixels = 250

name = ['2016-10-05-1626_ImageSequence_pill_155.956kX_20.000kV_30mu_26',
        '2016-10-05-1655_ImageSequence_pill_229.126kX_20.000kV_30mu_27',
       '2016-10-05-1705_ImageSequence_pill_129.338kX_20.000kV_30mu_29',
       '2016-10-05-1724_ImageSequence_pill_146.525kX_20.000kV_30mu_32',
       '2016-10-05-2013_ImageSequence__373.509kX_20.000kV_30mu_3',
       '2016-10-05-2121_ImageSequence__1000.000kX_20.000kV_30mu_10'] #Last one is single! At 1000kX (max mag)!!!
Pixel_size = [7.15905e-10,4.87286e-10, 8.63243e-10,  7.61988e-10, 2.98922e-10, 1.1165e-10] #as given by accompanying text file
Ps = [0.72, 0.49,0.86, 0.77, 0.30, 0.11] #in nm
lag = [100, 100,100,100, 50, 50]
frames = [50,100,50,50, 100,20] 
#obs = ['Region 1']

index = 5

### data
file1    = h5py.File(name[index] + '.hdf5', 'r')  
se   = file1['/data/Analog channel 2 : InLens/data'] #50 frames x250 x 250 pixels ####Achtung: Not SE, but InLens
red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
blue  = file1['/data/Counter channel 2 : PMT blue/PMT blue/data']#50 frames x 200 tr pts x250 x 250 pixels

se = np.array(se)
red = np.array(red)
blue = np.array(blue)

#convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
red = red/1.0e3
blue = blue/1.0e3
unit = '(kHz)'

### 
se_dset = se
red_dset = red
blue_dset = blue

### register
#independently
#se_dset_reg = reg_images(se_dset)
#blue_dset_reg = reg_images(blue_dset)
#red_dset_reg = reg_images(red_dset)

#when SE fails
#blue_dset_reg, red_dset_reg = reg_images(blue_dset, red_dset)
#se_dset_reg = np.average(se_dset, axis = 0)

#based on registering of se
se_dset_reg ,blue_dset_reg, red_dset_reg = reg_images(se_dset,blue_dset, red_dset)

# CUtting if necessary
#### cut only inside window: these are the base images!!!!!!!!!!!!!!!!!!!!!!!!!!!!
##trying circular mask at center a,b
#a, b = 247,255 #y was 255  x was 243
#n = blue_dset_reg.shape[0] #not square matrix anymore; does not matter, only approximatively
#r = 160 #was 170
#y,x = np.ogrid[-a:n-a, -b:n-b]
#mask = x*x + y*y <= r*r
## cutting 3 channels
#blue_dset_cut = np.empty([blue_dset_reg.shape[0],blue_dset_reg.shape[1]])
#blue_dset_cut[:] = np.nan
#blue_dset_cut[mask] = blue_dset_reg[mask]
#
#red_dset_cut = np.empty([blue_dset_reg.shape[0],blue_dset_reg.shape[1]])
#red_dset_cut[:] = np.nan
#red_dset_cut[mask] = red_dset_reg[mask]
#
#se_dset_cut = np.empty([blue_dset_reg.shape[0],blue_dset_reg.shape[1]])
#se_dset_cut[:] = np.nan
#se_dset_cut[mask] = se_dset_reg[mask]

#not cutting
blue_dset_cut = blue_dset_reg
red_dset_cut = red_dset_reg
se_dset_cut = se_dset_reg


fig4 = plt.figure(figsize=(8, 6), dpi=80)
ax1 = plt.subplot2grid((1,3), (0, 1), colspan=1)
ax1.set_title('Blue channel base')
plt.imshow(blue_dset_cut,cmap='Blues')
ax1 = plt.subplot2grid((1,3), (0, 2), colspan=1)
ax1.set_title('Red channel base')
plt.imshow(red_dset_cut,cmap='Reds')
ax1 = plt.subplot2grid((1,3), (0, 0), colspan=1)
ax1.set_title('InLens channel base')
plt.imshow(se_dset_cut,cmap='Greys')

#plt.close("all")
title =  'Stefan NaYF4:Er (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(lag[index]) + '$\mu$s lag per pixel, ' + str(frames[index]) + 'expts., InLens registered)' #, \n' #+ obs[index] 
if index == 0:
    scinti_channel = '500-700nm' 
else:
    scinti_channel = '$<$ 650nm' 
sample_channel = '$>$ 715nm'
if index == 5:
    length_scalebar = 10.0 #in nm (1000nm == 1mum)
    scalebar_legend = '10 nm'
else:
    length_scalebar = 50.0 #in nm (1000nm == 1mum)
    scalebar_legend = '50 nm'

plot_3_channels(se_dset_cut,blue_dset_cut, red_dset_cut, Pixel_size[index], title, scinti_channel, sample_channel, length_scalebar, scalebar_legend,unit,work_red_channel=True)
multipage('ZZ' + name[index] + '.pdf')

plot_3_channels_stretch(se_dset_cut,blue_dset_cut, red_dset_cut, Pixel_size[index], title, scinti_channel, sample_channel, length_scalebar, scalebar_legend,unit)
multipage('ZZStretch' + name[index] + '.pdf')

plot_2_channels_divide(se_dset_cut,blue_dset_cut, red_dset_cut, Pixel_size[index], title, scinti_channel, sample_channel, length_scalebar, scalebar_legend,unit,work_red_channel=True)
multipage('ZZDivide' + name[index] + '.pdf')

fig40 = plt.figure(figsize=(8, 6), dpi=80)
plt.imshow(blue_dset_cut/red_dset_cut)
plt.colorbar()
#plt.clim([0,200])

plt.show()

klklklk

###################################################################### OPTIONAL
want_gaussian_filter_correction_blue = False
want_gaussian_filter_correction_red = False

if want_gaussian_filter_correction_blue:
   sigma_blue = 1 
   blue_dset_cut1 = gaussian_filter_correction(blue_dset_cut, 'Blue',sigma_blue)
   blue_dset_cut = blue_dset_cut1  

if want_gaussian_filter_correction_red:
   sigma_red = 1 
   red_dset_cut1 = gaussian_filter_correction(red_dset_cut, 'Red',sigma_red)
   red_dset_cut = red_dset_cut1  

############################################################### END OF OPTIONAL

###################################################################### OPTIONAL
### Suggested:
# 1- Blue True, 3, [0] + Red False
# 2 - Blue True, 3, [2] + Red False
# 3 - Blue True, 3, [0] + Red True, 21, [1]
# 4 - Blue True, 3, [2] + Red True, 21, [1]
# 5 - Blue False, Red False

want_background_correction_blue = False
want_background_correction_red = False

filterset = ['white_tophat','black_tophat','medfilt']

if want_background_correction_blue:
    # Available algo types:
    # 'white_tophat' -> needs to change disk size
    # 'black_tophat' -> needs to change disk size
    # 'medfilt' -> needs to changer kernel size
    
    # New base dsets: blue_dset_cut, red_dset_cut
    size_blue = 3
    blue_dset_cut1 = background_correction(blue_dset_cut, filterset[0], 'Blue',size_blue)
    #blue_dset_cut2 = background_correction(blue_dset_cut, filterset[1], 'Blue',size_blue)
    blue_dset_cut3 = background_correction(blue_dset_cut, filterset[2], 'Blue',size_blue)
    #both [0] and [2] acceptable; min size_blue that makes sense = 3
    
    blue_dset_cut = blue_dset_cut1     #1 or 3
       
if want_background_correction_red:    
    size_red = 21
    #red_dset_cut1 = background_correction(red_dset_cut, filterset[0], 'Red',size_red)
    red_dset_cut2 = background_correction(red_dset_cut, filterset[1], 'Red',size_red)
    #red_dset_cut3 = background_correction(red_dset_cut, filterset[2], 'Red',size_red)
    # [1] can be good. Or no correction.
    red_dset_cut = red_dset_cut2

############################################################### END OF OPTIONAL

#plt.close("all")

from CreateDatasets import *

do_avg_dset = True
do_median_dset = True
do_arb_thr_one = True
do_gmmred_dset = True
do_gmmboth_dset = True
do_threshold_adaptive = True
do_random_walker = True
do_otsu = True

### construct different datasets
### 1) Simple average
if do_avg_dset:
    below_blue, above_blue, below_red, above_red = above_below_avg(blue_dset_cut, red_dset_cut)
    do_analysis(blue_dset_cut, red_dset_cut, below_blue, above_blue, below_red, above_red, 'YAP', 'Chlor','Above/Below avg', 'below avg', 'above avg',Pixel_size[index])

### 1) Simple median
if do_median_dset:
    belowm_blue, abovem_blue, belowm_red, abovem_red = above_below_median(blue_dset_cut, red_dset_cut)
    do_analysis(blue_dset_cut, red_dset_cut, belowm_blue, abovem_blue, belowm_red, abovem_red, 'YAP', 'Chlor','Above/Below median', 'below median', 'above median',Pixel_size[index])

### 1) Arb thresh in red
if do_arb_thr_one:
    arb_threshold = 0.6 #fraction of max
    belowarb_blue, abovearb_blue, belowarb_red, abovearb_red = arb_thr_one(red_dset_cut, blue_dset_cut, arb_threshold)
    do_analysis(blue_dset_cut, red_dset_cut, belowarb_blue, abovearb_blue, belowarb_red, abovearb_red, 'YAP', 'Chlor','Above/Below arb thr = ' + str(arb_threshold) + ' of red max', 'below red thr', 'above red thr',Pixel_size[index])

### 2) GMM with red mask, where red has been recognized as fluorescence
if do_gmmred_dset:
    gmmred_blue_dark_dset, gmmred_blue_bright_dset, gmmred_red_dark_dset, gmmred_red_bright_dset = gmmone(red_dset_cut, blue_dset_cut)
    do_analysis(blue_dset_cut, red_dset_cut, gmmred_blue_dark_dset, gmmred_blue_bright_dset, gmmred_red_dark_dset, gmmred_red_bright_dset, 'YAP', 'Chlor','GMM red', 'red dark spots', 'red bright spots',Pixel_size[index])

### 3) GMM with independent masks in both channels
if do_gmmboth_dset:
    gmmboth_blue_dark_dset, gmmboth_blue_bright_dset, gmmboth_red_dark_dset, gmmboth_red_bright_dset = gmmboth(red_dset_cut, blue_dset_cut)
    do_analysis(blue_dset_cut, red_dset_cut, gmmboth_blue_dark_dset, gmmboth_blue_bright_dset, gmmboth_red_dark_dset, gmmboth_red_bright_dset, 'YAP', 'Chlor','GMM both', 'dark spots', 'bright spots',Pixel_size[index])

### 4) Threshold adapative
if do_threshold_adaptive:
   blocksize = 11
   offset = 0
   th_below_blue, th_above_blue, th_below_red, th_above_red = threshold_adaptive_dset(red_dset_cut, blue_dset_cut,blocksize, offset)
   do_analysis(blue_dset_cut, red_dset_cut, th_below_blue, th_above_blue, th_below_red, th_above_red, 'YAP', 'Chlor','Threshold adaptive' + '(blocksize, offset =' + str(blocksize) + ', ' + str(offset) + ')', 'below thr', 'above thr',Pixel_size[index])

### 5) random_walker not yet working
## http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_random_walker_segmentation.html#example-segmentation-plot-random-walker-segmentation-py
if do_random_walker:
    cutofflow = 0.89
    cutoffhigh = 0.9
    rw_below_blue, rw_above_blue, rw_below_red, rw_above_red = random_walker_dset(red_dset_cut, blue_dset_cut,cutofflow, cutoffhigh)
    do_analysis(blue_dset_cut, red_dset_cut, rw_below_blue, rw_above_blue, rw_below_red, rw_above_red, 'YAP', 'Chlor','Random walker'+ '(cutoffs high, low =' + str(cutoffhigh) + ', ' + str(cutofflow) + ')', 'background', 'foreground',Pixel_size[index])

### 6) Otsu thresholding 
if do_otsu:
   ot_below_blue, ot_above_blue, ot_below_red, ot_above_red = thr_otsu(red_dset_cut, blue_dset_cut)
   do_analysis(blue_dset_cut, red_dset_cut, ot_below_blue, ot_above_blue, ot_below_red, ot_above_red, 'YAP', 'Chlor','Otsu threshold', 'background', 'foreground',Pixel_size[index])
 
#print('here1') 
#log_dog_doh(blue_dset_cut)
#print('here2') 
#log_dog_doh(blue_dset_cut)


multipage(name[index] + '.pdf')
plt.show()