#file 2016-07-05-1901_ImageSequence__11.236kX_3.000kV_12mu_5.hdf5

import os
import sys
import scipy as sp
import scipy.misc
import matplotlib.pyplot as plt
import h5py
import numpy as np
from BackgroundCorrection import *
import matplotlib.cm as cm
import scipy.ndimage as ndimage
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from MakePdf import *

import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = DeprecationWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.simplefilter(action = "ignore", category = PendingDeprecationWarning)

from Registration import reg_images
from tifffile import *

from sklearn.mixture import GMM 
import matplotlib.cm as cm

No_pixels = 512

### data
#file    = h5py.File('data.hdf5', 'r')  #file 2016-07-05-1901_ImageSequence__11.236kX_3.000kV_12mu_5.hdf5
Pixel_size = 9.937221e-09 #as given by txt accompanying file

se_dset_tot   = file['/data/Analog channel 1 : SE2/data'] # in pikachu, InLens is switched in name with SE2; we're actually taking SE2 image here)
red_dset_tot   = file['/data/Counter channel 2 : PMT red More 509/PMT red More 509/data']
blue_dset_tot   = file['/data/Counter channel 1 : PMT blue Less 509/PMT blue Less 509/data']

### remove last 20 images because I know they're black - no need to do this in Woodstock
se_dset = se_dset_tot#[0:40,:,:]
red_dset = red_dset_tot#[0:40,:,:]
blue_dset = blue_dset_tot#[0:40,:,:]

### register
#independently
#se_dset_reg = reg_images(se_dset)
#blue_dset_reg = reg_images(blue_dset)
#red_dset_reg = reg_images(red_dset)
#based on registering of blue channel
se_dset_reg, red_dset_reg, blue_dset_reg = reg_images(se_dset, red_dset, blue_dset)

### cut only inside window: these are the base images!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#trying circular mask at center a,b
#a, b = 247,255 #y was 255  x was 243
#n = blue_dset_reg.shape[0] #not square matrix anymore; does not matter, only approximatively
#r = 160 #was 170
#y,x = np.ogrid[-a:n-a, -b:n-b]
#mask = x*x + y*y <= r*r
# cutting 3 channels
blue_dset_cut = blue_dset_reg #np.empty([blue_dset_reg.shape[0],blue_dset_reg.shape[1]])
#blue_dset_cut[:] = np.nan
#blue_dset_cut[mask] = blue_dset_reg[mask]

red_dset_cut = red_dset_reg #np.empty([blue_dset_reg.shape[0],blue_dset_reg.shape[1]])
#red_dset_cut[:] = np.nan
#red_dset_cut[mask] = red_dset_reg[mask]

se_dset_cut = se_dset_reg #np.empty([blue_dset_reg.shape[0],blue_dset_reg.shape[1]])
#se_dset_cut[:] = np.nan
#se_dset_cut[mask] = se_dset_reg[mask]

fig4 = plt.figure(figsize=(8, 6), dpi=80)
ax1 = plt.subplot2grid((1,3), (0, 1), colspan=1)
ax1.set_title('Blue channel base')
plt.imshow(blue_dset_cut)
scalebar = ScaleBar(Pixel_size, frameon = True, box_alpha = 0.001, location = 'upper right') # 1 pixel = Pixel_size in meter
plt.gca().add_artist(scalebar)
ax1 = plt.subplot2grid((1,3), (0, 2), colspan=1)
ax1.set_title('Red channel base')
plt.imshow(red_dset_cut)
scalebar = ScaleBar(Pixel_size, frameon = True, box_alpha = 0.001, location = 'upper right') # 1 pixel = Pixel_size in meter
plt.gca().add_artist(scalebar)
ax1 = plt.subplot2grid((1,3), (0, 0), colspan=1)
ax1.set_title('SE channel base')
plt.imshow(se_dset_cut)
scalebar = ScaleBar(Pixel_size, frameon = True, box_alpha = 0.001, location = 'upper right') # 1 pixel = Pixel_size in meter
plt.gca().add_artist(scalebar)

#plt.close("all")

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
do_arb_thr_one = False
do_gmmred_dset = False
do_gmmboth_dset = True
do_threshold_adaptive = False
do_random_walker = False
do_otsu = True

### construct different datasets
### 1) Simple average
if do_avg_dset:
    print('doing avg')
    below_blue, above_blue, below_red, above_red = above_below_avg(blue_dset_cut, red_dset_cut)
    do_analysis(blue_dset_cut, red_dset_cut, below_blue, above_blue, below_red, above_red, 'YAP', 'Chlor','Above/Below avg', 'below avg', 'above avg',Pixel_size)

### 1) Simple median
if do_median_dset:
    print('doing median')
    belowm_blue, abovem_blue, belowm_red, abovem_red = above_below_median(blue_dset_cut, red_dset_cut)
    do_analysis(blue_dset_cut, red_dset_cut, belowm_blue, abovem_blue, belowm_red, abovem_red, 'YAP', 'Chlor','Above/Below median', 'below median', 'above median',Pixel_size)

### 1) Arb thresh in red
if do_arb_thr_one:
    print('doing arb thres')
    arb_threshold = 0.6 #fraction of max
    belowarb_blue, abovearb_blue, belowarb_red, abovearb_red = arb_thr_one(red_dset_cut, blue_dset_cut, arb_threshold)
    do_analysis(blue_dset_cut, red_dset_cut, belowarb_blue, abovearb_blue, belowarb_red, abovearb_red, 'YAP', 'Chlor','Above/Below arb thr = ' + str(arb_threshold) + ' of red max', 'below red thr', 'above red thr',Pixel_size)

### 2) GMM with red mask, where red has been recognized as fluorescence
if do_gmmred_dset:
    print('doing gmm red')
    gmmred_blue_dark_dset, gmmred_blue_bright_dset, gmmred_red_dark_dset, gmmred_red_bright_dset = gmmone(red_dset_cut, blue_dset_cut)
    do_analysis(blue_dset_cut, red_dset_cut, gmmred_blue_dark_dset, gmmred_blue_bright_dset, gmmred_red_dark_dset, gmmred_red_bright_dset, 'YAP', 'Chlor','GMM red', 'red dark spots', 'red bright spots',Pixel_size)

### 3) GMM with independent masks in both channels
if do_gmmboth_dset:
    print('doing gmm both')
    gmmboth_blue_dark_dset, gmmboth_blue_bright_dset, gmmboth_red_dark_dset, gmmboth_red_bright_dset = gmmboth(red_dset_cut, blue_dset_cut)
    do_analysis(blue_dset_cut, red_dset_cut, gmmboth_blue_dark_dset, gmmboth_blue_bright_dset, gmmboth_red_dark_dset, gmmboth_red_bright_dset, 'YAP', 'Chlor','GMM both', 'dark spots', 'bright spots',Pixel_size)

### 4) Threshold adapative
if do_threshold_adaptive:
   print('doing thr adap')
   blocksize = 50
   offset = 0
   th_below_blue, th_above_blue, th_below_red, th_above_red = threshold_adaptive_dset(red_dset_cut, blue_dset_cut,blocksize, offset)
   do_analysis(blue_dset_cut, red_dset_cut, th_below_blue, th_above_blue, th_below_red, th_above_red, 'YAP', 'Chlor','Threshold adaptive' + '(blocksize, offset =' + str(blocksize) + ', ' + str(offset) + ')', 'below thr', 'above thr',Pixel_size)

### 5) random_walker not yet working
## http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_random_walker_segmentation.html#example-segmentation-plot-random-walker-segmentation-py
if do_random_walker:
    print('doing random walk')
    cutofflow = 0.89
    cutoffhigh = 0.9
    rw_below_blue, rw_above_blue, rw_below_red, rw_above_red = random_walker_dset(red_dset_cut, blue_dset_cut,cutofflow, cutoffhigh)
    do_analysis(blue_dset_cut, red_dset_cut, rw_below_blue, rw_above_blue, rw_below_red, rw_above_red, 'YAP', 'Chlor','Random walker'+ '(cutoffs high, low =' + str(cutoffhigh) + ', ' + str(cutofflow) + ')', 'background', 'foreground',Pixel_size)

### 6) Otsu thresholding 
if do_otsu:
   print('doing otsu')
   ot_below_blue, ot_above_blue, ot_below_red, ot_above_red = thr_otsu(red_dset_cut, blue_dset_cut)
   do_analysis(blue_dset_cut, red_dset_cut, ot_below_blue, ot_above_blue, ot_below_red, ot_above_red, 'YAP', 'Chlor','Otsu threshold', 'background', 'foreground',Pixel_size)
 
#print('here1') 
#log_dog_doh(blue_dset_cut)
#print('here2') 
#log_dog_doh(blue_dset_cut)


multipage('multipage.pdf')
plt.show()
