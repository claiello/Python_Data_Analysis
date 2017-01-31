#nominally
#50mus moving
#150mus on
#1400mus off
#clock 1MHz = 1mus time bon
#250kX, 50%x 50% scale
#250x250 pixels
#5 frames per temperature (ie, per voltage )

##### Temperatures were not read, estimated before in the program
#29.89
#36.70
#41.79
#48.83
#69.84

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

import skimage
from skimage import exposure
#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################

prefix = ''

nametr = ['2017-01-05-1452_ImageSequence__250.000kX_10.000kV_30mu_10',
          '2017-01-05-1557_ImageSequence__250.000kX_10.000kV_30mu_15',
          '2017-01-05-1634_ImageSequence__250.000kX_10.000kV_30mu_20',
          '2017-01-05-1709_ImageSequence__250.000kX_10.000kV_30mu_23',
          '2017-01-05-1745_ImageSequence__250.000kX_10.000kV_30mu_26',
          '2017-01-05-1831_ImageSequence__250.000kX_10.000kV_30mu_30',
          '2017-01-05-1906_ImageSequence__250.000kX_10.000kV_30mu_32']

let = ['RT','V0','V0pt25' ,'V0pt5', 'V0pt5b','V0pt75','V1']

######################################## Plot with dose for different apertures


backgdinit = 50
initbin = (150+50+3)-1

##############################################################################
###############################################################################
#Try poisson

import scipy.stats
import scipy.optimize
import statsmodels.api as sm 
from scipy.optimize import minimize
from scipy.misc import factorial

length_of_array = 1398

#index = 2
#if index == 2:
for index in [1,2,4,5]: #listofindex:

    gc.collect()    
    
    print(index)
      
    red = np.load(str(let[index]) +'Redbright.npz',mmap_mode='r') 
    gc.collect() 
    for jj in [initbin,length_of_array+initbin]: #np.arange(0, aa.shape[1]):
        data= np.array(red['data'][:,jj,:,:])
        data = data.flatten()
        res = sm.Poisson(data,np.ones_like(data)).fit()
        print(res.predict())
        print(res.summary())

    
    


klklklkk
##############################################################################
###############################################################################
#Try poisson





##files below exist 
back_array_red = np.zeros(len(nametr))
back_array_blue = np.zeros(len(nametr))

#index = 0
#red = np.load(str(let[index]) +'Redbright.npz',mmap_mode='r') 
#length_of_array = red['data'].shape[1] - initbin
#del red
#gc.collect()

#length_of_array = 1398





#index = 3
#if index is 3:
for index in [1,2,4,5]: #listofindex:

    gc.collect()    
    
    print(index)
      
    red = np.load(str(let[index]) +'Redbright.npz',mmap_mode='r') 
    aa = np.average(red['data'][:,0:backgdinit,:,:], axis = (0,1,2,3))
    del red
    gc.collect()
    back_array_red[index] = aa
    del aa
    gc.collect()
    
    blue = np.load(str(let[index]) +'Bluebright.npz',mmap_mode='r') 
    aa =np.average(blue['data'][:,0:backgdinit,:,:], axis = (0,1,2,3))
    del blue
    gc.collect()
    back_array_blue[index] = aa
    del aa
    gc.collect()
    
    print('here3')

mycode =prefix + 'Back_array_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Back_array_red', data = back_array_red)

mycode =prefix + 'Back_array_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Back_array_blue', data = back_array_blue)

##### ONCE ALL FITS WORK, 
###### NEED TO RUN THESE LINES BELOW SO THAT ALL NPZ FILES ARE CREATED 
klklklk

#std_array_red = np.zeros([len(nametr),length_of_array])
#std_array_blue = np.zeros([len(nametr),length_of_array])

#index = 3
#if index is 3:
for index in [1,2,4,5]: #listofindex:

    gc.collect()    
    
    print(index)
      
    red = np.load(str(let[index]) +'Redbright.npz',mmap_mode='r') 
    aa = np.std(red['data'][:,initbin:,:,:], axis = (0,2,3))
    del red
    gc.collect() 
    std_array_red[index,:] = aa
    del aa 
    gc.collect()
    
    blue = np.load(str(let[index]) +'Bluebright.npz',mmap_mode='r') 
    aa = np.std(blue['data'][:,initbin:,:,:], axis = (0,2,3))
    del blue
    gc.collect()
    std_array_blue[index,:] = aa
    del aa
    gc.collect()
    
    print('here3')

mycode =prefix + 'Std_array_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Std_array_red', data = std_array_red)

mycode =prefix + 'Std_array_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Std_array_blue', data = std_array_blue)



klklklklk


