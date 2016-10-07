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

#Trying to get frames from mov file
#import cv2
#vidcap = cv2.VideoCapture('Bibi.mov')
#success,image = vidcap.read()
#count = 0
#success = True
#while success:
#  success,image = vidcap.read()
#  print('Read a new frame: ', success)
#  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
#  count += 1

#klklklklkkl



#import scalebars as sb

#import Image
#from PIL import Image
#im = Image.open("Bibiana-original.gif")
#import images2gif as i2g
#im = i2g.readGif('Bibiana-original.gif', False)

#from PIL import Image, ImageSequence
#filename = 'Bibiana-original.gif'
#ima = Image.open(filename)
#original_duration = ima.info['duration']
#frames = [frame.copy() for frame in ImageSequence.Iterator(ima)]    
#im = frames.reverse()

# extracted all frames in jpg format from mov file using command:
#ffmpeg -i Bibi.mov frame%d.jpg

#from PIL import Image
#im = np.zeros([15,396,439])
#for kk in np.arange(0,14):
#    hlp =  Image.open("frame" + str(kk+1) + ".jpg")
#    im[kk] = np.average(hlp, axis = 2) #Averaging over R,G,B values

###### Boerge loaded all frames from Bibi.mov in a file called all_frames.npy; in grayscale
imaa = np.load('all_frames.npy')
print(imaa.shape)

No_pixels = 250 #439x396 

name = ['bibiana']
Pixel_size = [5.6e-08] #as given by accompanying text file
Ps = [56]
lag = [500]
frames = [20] 
obs = ['Region I']

index = 4

### data

red = np.array(imaa)

red_dset = red

#red_dset_cut = np.empty([red_dset.shape[0],red_dset.shape[1]])
#red_dset_cut[:] = np.nan
#mask = x*x + y*y <= r*r
#red_dset_cut[mask] = red_dset_reg[mask]


# Cut to hide axes - frames are #439x396 
print(red_dset.shape)
#red_dset_cut = red_dset[:,0:396, 0:439]
#red_dset_cut = red_dset[:,18:359, 37:422] #cutting margins
red_dset_cut = red_dset[:,20:359, 40:400]
#red_dset_cut = red_dset[:,50:350, 50:350]

#plt.imshow(red_dset_cut[0])
#plt.show()
### register
#independently
#se_dset_reg = reg_images(se_dset)
#blue_dset_reg = reg_images(blue_dset)
red_dset_reg, red_dset_reg_all = reg_images(red_dset_cut)

#when SE fails
#blue_dset_reg, red_dset_reg = reg_images(blue_dset, red_dset)
#se_dset_reg = np.average(se_dset, axis = 0)

#based on registering of se
#se_dset_reg ,blue_dset_reg, red_dset_reg = reg_images(se_dset,blue_dset, red_dset)

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





fig4 = plt.figure(figsize=(8, 6), dpi=80)
#ax1 = plt.subplot2grid((1,3), (0, 1), colspan=1)
#ax1.set_title('Blue channel base')
#plt.imshow(blue_dset_cut,cmap='Blues')
#ax1 = plt.subplot2grid((1,3), (0, 2), colspan=1)
#ax1.set_title('Red channel base')
#plt.imshow(red_dset_cut,cmap='Reds')
ax1 = plt.subplot2grid((1,1), (0, 0), colspan=1)
ax1.set_title('SE channel base')
plt.imshow(red_dset_reg,cmap='Greys')

#plt.close("all")
title =  'Bibiana' #'Grana + IL on ZnO:Ga (4kV, 30 $\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(lag[index]) + '$\mu$s lag per pixel, ' + str(frames[index]) + 'expts., SE registered), \n' + obs[index] 
#scinti_channel = '$<$ 593nm'
#sample_channel = '$>$ 647nm'
#length_scalebar = 5000.0 #in nm (1000nm == 1mum)
#scalebar_legend = '5 $\mu$m'
#plot_3_channels(se_dset_cut,blue_dset_cut, red_dset_cut, Pixel_size[index], title, scinti_channel, sample_channel, length_scalebar, scalebar_legend,unit,work_red_channel=True)
#multipage('ZZ' + name[index] + '.pdf')

#plot_3_channels_stretch(se_dset_cut,blue_dset_cut, red_dset_cut, Pixel_size[index], title, scinti_channel, sample_channel, length_scalebar, scalebar_legend,unit)
#multipage('ZZStretch' + name[index] + '.pdf')

#plot_2_channels_divide(se_dset_cut,blue_dset_cut, red_dset_cut, Pixel_size[index], title, scinti_channel, sample_channel, length_scalebar, scalebar_legend,unit,work_red_channel=True)
#multipage('ZZDivide' + name[index] + '.pdf')

#fig40 = plt.figure(figsize=(8, 6), dpi=80)
#plt.imshow(blue_dset_cut/red_dset_cut)
#plt.colorbar()
#plt.clim([0,200])

plt.show()

import matplotlib.animation as animation
dpi = 100


fig = plt.figure()
#ax = fig.add_subplot(111)
ax0 = plt.subplot2grid((1,3), (0, 0), colspan=1)
ax0.set_aspect('equal')
ax0.get_xaxis().set_visible(False)
ax0.get_yaxis().set_visible(False)
ax0.set_title('Original')
im0 = ax0.imshow(red_dset_cut[0,:,:],cmap='gray') #,interpolation='nearest')


ax = plt.subplot2grid((1,3), (0, 1), colspan=1)
ax.set_aspect('equal')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title('Registered')
im = ax.imshow(red_dset_reg_all[0,:,:],cmap='gray') #,interpolation='nearest')

ax3 = plt.subplot2grid((1,3), (0, 2), colspan=1)
ax3.set_aspect('equal')
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
ax3.set_title('Orig - Reg')
#im3 = ax3.imshow(red_dset_cut[0,:,:]-red_dset_reg_all[0,:,:],cmap='gray') #,interpolation='nearest')
#im.set_clim([0,1])
#fig.set_size_inches([5,5])


#tight_layout()


def update_img(n):
    tmp = red_dset_reg_all[n,:,:]
    im.set_data(tmp)
    tmp0 = red_dset_cut[n,:,:]
    im0.set_data(tmp0)
    #tmp3 = red_dset_cut[n,:,:] - red_dset_reg_all[n,:,:]
    #im3.set_data(tmp3)
    #print(np.average(tmp3))
    return im

    #legend(loc=0)
ani = animation.FuncAnimation(fig,update_img,14)#,interval=1)
writer = animation.writers['ffmpeg'](fps=1)

ani.save('demo.mp4',writer=writer,dpi=dpi)
#return ani



##### Try to use trackpy
import pims
import trackpy as tp
import pandas as pd
from pandas import DataFrame, Series  # for convenience

f = tp.locate(red_dset_cut[0,:,:], 15, invert=False) #False looks for bright features
plt.figure()  # make a new figure
tp.annotate(f, red_dset_cut[0,:,:]);

fig, ax = plt.subplots()
ax.hist(f['mass'], bins=20)

# Optionally, label the axes.
ax.set(xlabel='mass', ylabel='count');

f = tp.locate(red_dset_cut[0,:,:], 15, invert=False, minmass=500) #False looks for bright features
plt.figure()  # make a new figure
tp.annotate(f, red_dset_cut[0,:,:]);


f = tp.batch(red_dset_cut[0:13,:,:], 15, minmass=500, invert=False);
t = tp.link_df(f, 5, memory=3)

t1 = tp.filter_stubs(t, 50)
# Compare the number of particles in the unfiltered and filtered data.
print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())

plt.figure()
tp.mass_size(t1.groupby('particle').mean()); # convenience function -- just plots size vs. mass

plt.figure()
tp.plot_traj(t1);

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