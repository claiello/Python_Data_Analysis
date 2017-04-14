import numpy as np
import matplotlib.pyplot as plt
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
#from BackgroundCorrection import *
#from TConversionThermocoupler import *
import matplotlib.cm as cm
import scipy.ndimage as ndimage
#from matplotlib_scalebar.scalebar import ScaleBar #### Has issue with plotting using latex font. only import when needed, then unimport
#from mpl_toolkits.axes_grid1 import make_axes_locatable
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.misc
import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile

import skimage
from skimage import exposure
from my_fits import *

import pickle

from numpy import genfromtxt
from uncertainties import unumpy

from get_data import get_data

### settings
fsizepl = 24
fsizenb = 20
sizex = 8
sizey=6
dpi_no = 80

fig1= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig1.set_size_inches(800./fig1.dpi,600./fig1.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    

noplots = 2
nolines = 2

#ax100 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)

d = np.load('data_2nm.npz')

bin_r = d['arr_0']
bin_z = d['arr_1']
result = d['arr_2']
all_r = d['arr_3']
all_z = d['arr_4']

result = np.array(result)

result[~np.isfinite(result)] = 0.0
result[np.isnan(result)] = 0.0

print(result.shape)
print(bin_r.shape)
print(bin_z.shape)

from scipy import ndimage
res = ndimage.measurements.center_of_mass(result[0:27,0:12])
print(res)

klklk

bin_r[12] = 26.57
bin_z[27] = 121

deltar = 2.2140811635188982
deltaz = 5.8852084000000104

I think that the above gives the center of mass within one nanoparticle

res is (15.639559624678359 (Z), 3.8323182106314655 (R))