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

#Grab info on mag, pixel size, aperture, kV from .txt file
### can't make textcolor to work

name = ['2017-03-25-1636_ImageSequence__5.408kX_2.000kV_30mu_18',
        '2017-03-25-1645_ImageSequence__15.125kX_2.000kV_30mu_20',
        '2017-03-25-1650_ImageSequence__15.125kX_2.000kV_30mu_21',
        '2017-03-25-1702_ImageSequence__129.960kX_2.000kV_30mu_27',
        '2017-03-25-1707_ImageSequence__20.209kX_2.000kV_30mu_30',
        '2017-03-25-1710_ImageSequence__20.209kX_2.000kV_30mu_31',
        '2017-03-25-1736_ImageSequence__20.209kX_2.000kV_30mu_32'] #10avgs
        
Pixel_size = [41.3, 4.21, 2.1, 1.72, 11.0, 11.0, 11.0] #nm

for index in np.arange(0,len(name)):
#if True is False:    
    print(index)

    file1    = h5py.File(name[index] + '.hdf5', 'r')  
    title =  'Mithrene CL 17/03/29'
    se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
    red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
    blue  = file1['/data/Counter channel 2 : PMT blue/PMT blue/data']#50 frames x 200 tr pts x250 x 250 pixels
    
    se = np.array(se)
    red = np.array(red)
    blue = np.array(blue)
    red = red + blue
    del blue
    gc.collect()
    
    #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
    red = red/1.0e3
    unit = '(kHz)'

    length_scalebar = 1000.0 #in nm (1000nm == 1mum)
    scalebar_legend = '1 $\mu$m'
    
    reg_se, reg_red, reg_se_all, reg_red_all = reg_images(se,red)
    
    plot_2_channels(reg_se, reg_red, Pixel_size[index], title, length_scalebar, scalebar_legend,'kHz',work_red_channel=True)
    
    