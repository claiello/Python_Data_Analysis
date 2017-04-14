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
from CreateDatasets import *
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = DeprecationWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.simplefilter(action = "ignore", category = PendingDeprecationWarning)
from Registration import * 
#from tifffile import *
from sklearn.mixture import GMM 
import matplotlib.cm as cm
#from FluoDecay import *
from PlottingFcts import *
sys.path.append("/usr/bin") # necessary for the tex fonts
import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile
import scalebars as sb

fsizetit = 36
fsizepl = 24
sizex = 8
sizey = 6
dpi_no = 80
lw = 2

name = ['2017-03-25-1636_ImageSequence__5.408kX_2.000kV_30mu_18'] #10avgs
        
Pixel_size = [41.3] #nm

ax0 = plt.subplot2grid((1,2), (0,0), colspan=1, rowspan=1)

for index in np.arange(0,len(name)):
#if True is False:    
    print(index)

    file1    = h5py.File(name[index] + '.hdf5', 'r')  
    title =  'Mithrene CL 17/03/25'
    se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
    red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
    blue  = file1['/data/Counter channel 2 : PMT blue/PMT blue/data']#50 frames x 200 tr pts x250 x 250 pixels
    
    cutx = 335
    cutxend = 95
    cuty = 140
    cutyend = 300
    se = se[:,cutx:-cutxend,cuty:-cutyend]
    red = red[:,cutx:-cutxend,cuty:-cutyend]
    blue = blue[:,cutx:-cutxend,cuty:-cutyend]
    
    se = np.array(se)
    red = np.array(red)
    blue = np.array(blue)
    red = red + blue
    del blue
    gc.collect()
    
    #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
    red = red/1.0e3
    unit = '(kHz)'
    
    reg_se, reg_red, reg_se_all, reg_red_all = reg_images(se,red)
    
    length_scalebar = 1000.0 #in nm (1000nm == 1mum)
    scalebar_legend = '1 $\mu$m'
    
    plot_2_channels(reg_se, reg_red, Pixel_size[index], title, length_scalebar, scalebar_legend,'kHz',work_red_channel=True)
    
    #darkred, brightred, darkonese, brightse = gmmone(reg_se, reg_red)
    ax0.imshow(reg_se, cmap='Greys')
    ax0.set_title('Cut terrace',fontsize=fsizetit)
    ax0.axis('off')
    #br[index] = np.nanmean(brightred,axis=(0,1))
    
#multipage_longer('CutTerrace.pdf',dpi=80)
#plt.show()   
