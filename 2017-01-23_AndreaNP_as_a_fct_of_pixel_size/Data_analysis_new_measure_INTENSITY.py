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

import skimage
from skimage import exposure

from my_fits import *

#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################


initbin = (150+50+3)-1
backgdinit = 50
dset= 1

###############################################################################
###############################################################################
###############################################################################

if dset == 1:
    nametr = ['2017-01-23-1547_ImageSequence__250.000kX_10.000kV_30mu_7',
          '2017-01-23-1608_ImageSequence__250.000kX_10.000kV_30mu_8',
          '2017-01-23-1633_ImageSequence__250.000kX_10.000kV_30mu_9',
          '2017-01-23-1736_ImageSequence__250.000kX_10.000kV_30mu_10',
          '2017-01-23-1818_ImageSequence__63.372kX_10.000kV_30mu_11']

    let = ['pix1', 'pix2', 'pix3', 'pix4', 'pix5'] #pixel size is decreasing
    
    Varying_variable = [2.23, 1.79, 1.49, 1.28, 1.12]
    Label_varying_variable = 'Pixel size (nm) [data taken LARGE to SMALL]' 
    
    listofindex = [0,1,2,3,4]
    
    loadprefix = ''
    
###############################################################################
###############################################################################
###############################################################################

if dset == 1:
    pref0 = 'Different_pixel_sizes_'
    
Time_bin = 1000.0 #in ns

#fig1, ax1 = plt.subplots()
#fig2, ax2 = plt.subplots()
#fig3, ax3 = plt.subplots()
#fig4, ax4 = plt.subplots()
#fig5, ax5 = plt.subplots()
#
##fig10, ax10 = plt.subplots()
#fig20, ax20 = plt.subplots()
#fig200, ax200 = plt.subplots()
#fig2000, ax2000 = plt.subplots()
#fig30, ax30 = plt.subplots()
#
#fig40, ax40 = plt.subplots()
#fig400, ax400 = plt.subplots()

red_vec = np.empty([len(Varying_variable),1498])
blue_vec = np.empty([len(Varying_variable),1498])

for index in listofindex: 
    
    print(index)
    
    print('bef loading')
    redd = np.load(loadprefix + let[index] + 'Redbright.npz',mmap_mode='r') 
    blued = np.load(loadprefix + let[index] + 'Bluebright.npz',mmap_mode='r') 
    red_int_array = np.average(redd['data'][:,backgdinit:initbin,:,:],axis=(0,1,2,3))
    blue_int_array = np.average(blued['data'][:,backgdinit:initbin,:,:],axis=(0,1,2,3))
    red_std_array = np.std(redd['data'][:,backgdinit:initbin,:,:],axis=(0,1,2,3))
    blue_std_array = np.std(blued['data'][:,backgdinit:initbin,:,:],axis=(0,1,2,3))
    
    ured_int_array = unumpy.uarray(red_int_array,red_std_array)
    ublue_int_array = unumpy.uarray(blue_int_array,blue_std_array)
  
    yerr1 = unumpy.std_devs(((ublue_int_array-ured_int_array)/(ured_int_array+ublue_int_array)))
 
    x_vec = Varying_variable
    
    label1,res1 = my_fits.fit_with_plot_small_ratio_visib(ax3, 
                                                          x_vec, 
                                                          ((blue_int_array-red_int_array)/(red_int_array+blue_int_array)),
                                                          yerr = yerr1, 
                                                          my_color = 'k', 
                                                          my_edgecolor=None, 
                                                          my_facecolor=None, 
                                                          my_linestyle = 'dotted',
                                                          normtemp=x_vec[RTindex[0]]+273.15,
                                                          axnew=None,
                                                          my_hatch="/")



