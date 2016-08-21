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
#from BackgroundCorrection import *
import matplotlib.cm as cm
import scipy.ndimage as ndimage
#from matplotlib_scalebar.scalebar import ScaleBar #### Has issue with plotting using latex font. only import when needed, then unimport
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
from sklearn.mixture import GMM 
import matplotlib.cm as cm
#from FluoDecay import *
#from PlottingFcts import *
sys.path.append("/usr/bin") # necessary for the tex fonts
import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile
import scalebars as sb

fsizetit = 18
fsizepl = 16
sizex = 8
sizey = 6
dpi_no = 80
lw = 2

#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################

#Large area -> file 2016-07-29-0848_ImageSequence__0.234kX_15.000kV_12mu_8
#Small area -> file 2016-07-29-0856_ImageSequence__0.663kX_15.000kV_12mu_9

Time_bin = 2000 #2microsec in ns; 1/clock of 500kHz 
totalpoints = 20 #total number of time-resolved points
### data
name = ['2016-07-29-0848_ImageSequence__0.234kX_15.000kV_12mu_8.hdf5','2016-07-29-0856_ImageSequence__0.663kX_15.000kV_12mu_9.hdf5']

file1 = h5py.File('2016-07-29-0848_ImageSequence__0.234kX_15.000kV_12mu_8.hdf5', 'r')  
sebroad_dset   = file1['/data/Analog channel 1 : SE2/data'] #20 frames x250 x 250 pixels
ilbroad_dset  =  file1['/data/Analog channel 2 : InLens/data']#50 frames x 200 tr pts x250 x 250 pixels

file2 = h5py.File('2016-07-29-0856_ImageSequence__0.663kX_15.000kV_12mu_9.hdf5', 'r')  
sezoom_dset   = file2['/data/Analog channel 1 : SE2/data'] #20 frames x250 x 250 pixels
ilzoom_dset  =  file2['/data/Analog channel 2 : InLens/data']#50 frames x 200 tr pts x250 x 250 pixels

sebroad_dset  = np.array(sebroad_dset)
ilbroad_dset  = np.array(ilbroad_dset)
sezoom_dset  = np.array(sezoom_dset)
ilzoom_dset  = np.array(ilzoom_dset)

file1.close()    
file2.close()

#BEGIN Enhance constrast
#from PIL import Image, ImageEnhance
#contrast = ImageEnhance.Contrast(sezoom_dset)
#contrast2 = ImageEnhance.Contrast(ilzoom_dset)
#sezoom_dset= contrast.enhance(2)
#ilzoom_dset= contrast1.enhance(2)

#from PIL import Image
#from PIL import ImageEnhance
#brightness = 3.0
#peak = sezoom_dset
#enhancer = np.zeros([50,250,250])
#for kk in np.arange(0,50):
#    enhancer[kk,:,:] = ImageEnhance.Brightness(sezoom_dset[kk,:,:])
#sezoom_dset = enhancer.enhance(brightness)

#from skimage import exposure
#camera = sezoom_dset 
#sezoom_dset  = exposure.equalize_hist(camera)
#camera1 = ilzoom_dset 
#ilzoom_dset  = exposure.equalize_hist(camera1)

# histogram equalization
#import operator
#def equalize(h):
#    lut = []
#    for b in range(0, len(h), 256):
#        # step size
#        step = reduce(operator.add, h[b:b+256]) / 255
#
#        # create equalization lookup table
#        n = 0
#        for i in range(256):
#            lut.append(n / step)
#            n = n + h[i+b]
#    return lut
#
## calculate lookup table
#enhancer = np.zeros([50,250,250])
#for kk in np.arange(0,50):
#    lut = equalize(np.histogram(sebroad_dset[kk,:,:]))
#    # map image through lookup table
#    enhancer[kk,:,:] = sebroad_dset[kk,:,:].point(lut)
#sebroad_dset = np.copy(enhancer)

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf
    
data_equalized = np.zeros(ilzoom_dset.shape)
for i in range(ilzoom_dset.shape[0]):
    image = ilzoom_dset[i, :, :]
    data_equalized[i, :, :] = image_histogram_equalization(image)[0]
ilzoom_dset = np.copy(data_equalized)
#END Enhance constrast

Pixelbroad = 476.27 #in nm
Pixelzoom = 168.39 #in nm

length_scalebarbroad = 25000.0 #in nm (1000nm == 1mum)
length_scalebarzoom = 25000.0 #in nm (1000nm == 1mum)
scalebar_legend = '25 $\mu$m'
length_scalebar_in_pixelsbroad = np.ceil(length_scalebarbroad/(Pixelbroad)) #length_scalebar in pixel size (nm), rounded up for fairness
length_scalebar_in_pixelszoom = np.ceil(length_scalebarzoom/(Pixelzoom)) #length_scalebar in pixel size (nm), rounded up for fairness
 
fig40= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig40.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    
plt.suptitle("Gas (H$_2$?) trapped inside liquid, o-ring/CNC liquid cell, mixture of 100nm Ag NPs, 1$\mu$m yellow and red polymer NPs \n (15kV, 120$\mu$m, 2$\mu$s time bins)", fontsize=fsizetit)
   
gc.collect()
ax1 = plt.subplot2grid((1,4), (0, 0), colspan=1)
ax1.set_title('SE broad',fontsize=fsizepl)
im0 = plt.imshow(sebroad_dset[0,:,:],cmap=cm.Greys_r)#,vmin=np.min(sebroad_dset), vmax=np.max(sebroad_dset))
sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixelsbroad, scalebar_legend, style = 'dark', loc = 4)
ax1.add_artist(sbar)
plt.axis('off')

gc.collect()
ax1 = plt.subplot2grid((1,4), (0, 1), colspan=1)
ax1.set_title('InLens broad',fontsize=fsizepl)
im = plt.imshow(ilbroad_dset[0,:,:],cmap=cm.Greys_r)#,vmin=np.min(ilbroad_dset), vmax=np.max(ilbroad_dset)) #or 'OrRd'
sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixelsbroad, scalebar_legend, style = 'dark', loc = 4)
ax1.add_artist(sbar)
plt.axis('off')
  
gc.collect()
ax1 = plt.subplot2grid((1,4), (0,2), colspan=1)
ax1.set_title('SE zoom',fontsize=fsizepl)
imbright = ax1.imshow(sezoom_dset[0,:,:],cmap=cm.Greys_r)#,vmin=np.min(sezoom_dset), vmax=np.max(sezoom_dset))
sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixelszoom, scalebar_legend, style = 'dark', loc = 4)
ax1.add_artist(sbar)
plt.axis('off')

gc.collect()
ax1 = plt.subplot2grid((1,4), (0,3), colspan=1)
ax1.set_title('InLens zoom',fontsize=fsizepl)
imdark = ax1.imshow(ilzoom_dset[0,:,:],cmap=cm.Greys_r)#,vmin=np.min(ilzoom_dset), vmax=np.max(ilzoom_dset))
sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixelszoom, scalebar_legend, style = 'dark', loc = 4)
ax1.add_artist(sbar)
plt.axis('off')

def updatered(j):
        # set the data in the axesimage object
        im0.set_array(sebroad_dset[j,:,:])
        im.set_array(ilbroad_dset[j,:,:])
        imbright.set_array(sezoom_dset[j,:,:])
        imdark.set_array(ilzoom_dset[j,:,:])
       
        # return the artists set
        return [im0, im,imbright,imdark]
        
import matplotlib.animation as animation

updatered(0) # resets the plots to point 0
anim = animation.FuncAnimation(fig40, updatered,frames=np.arange(sebroad_dset.shape[0]-1), interval=200,repeat_delay=100) #interval was 10

plt.show()

Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800) #was 15 fps
anim.save('ZZZ-1video.avi', writer=writer)#, dpi=400)