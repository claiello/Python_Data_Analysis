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
from tifffile import *
from sklearn.mixture import GMM 
import matplotlib.cm as cm
from FluoDecay import *
from PlottingFcts import *

import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile
#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################
titulo = 'o'
fsizetit = 18 #22 #18
fsizepl = 16 #20 #16
sizex = 8 #10 #8
sizey = 6# 10 #6
dpi_no = 80
lw = 2
print('here')

fig50= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig50.set_size_inches(1200./fig50.dpi,900./fig50.dpi)
zno = np.load('ZZZZnO3.npz')  #yap['datared'], yap['datablue']
ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
ax1.set_title("ZnO:Ga",fontsize=fsizepl)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
blue = np.average(zno['datablue'],axis=(0))/1.0e3 #MHz
last_pt_offset = -10 #sometimes use -1, last point, but sometimes this gives 0. -10 seems to work
Time_bin = 40 #ns
middlept = -30
initdecay = 83
init_guess = [np.average(blue[initdecay,:,:]), 2.0, np.average(blue[last_pt_offset,:,:]), np.average(blue[middlept,:,:]), 0.05] #e init was 0.5

init_guess = [0.005, 0.2, 0.001, 0.05, 0.005]
print('here')
b,e,be,ee,b2,e2,be2,ee2,chisquared, chisquared2 = calcdecay_subplot_nan(blue[initdecay-2+4:,:,:], 
                              time_detail= Time_bin*1e-9,
                              titulo='Cathodoluminescence rate decay, bi-exponential fit, \n ',
                              single=False,
                              other_dset1=None,
                              other_dset2=None,
                              init_guess=init_guess,
                              unit='MHz')    
print('here')

#plt.xlim([0,2])
#major_ticks0 = [1,2]
plt.ylabel("Average luminescence \n of each time bin, per pixel (MHz)",fontsize=fsizepl)
plt.xlim([0,2.5])
plt.show() 

    
multipage('ComparisonYAPZnO.pdf',dpi=80)  
