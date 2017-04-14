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
import matplotlib.cm as cm
import scipy.ndimage as ndimage
from matplotlib.backends.backend_pdf import PdfPages
from MakePdf import *
from matplotlib.pyplot import cm #to plot following colors of rainbow
from matplotlib import rc
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = DeprecationWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.simplefilter(action = "ignore", category = PendingDeprecationWarning)
from Registration import * 
from tifffile import *
from sklearn.mixture import GMM 
import matplotlib.cm as cm
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
import my_fits
from uncertainties import unumpy
from numpy import genfromtxt

### settings
fsizepl = 24
fsizenb = 20
mkstry = ['8','11','5'] #marker size for different dsets Med Zoom/Large Zoom/Small Zoom
dpi_no=80
sizex = 8
sizey =6
###
sizex = 8
sizey=3

fig1= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig1.set_size_inches(1200./fig1.dpi,900./fig1.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    

noplots = 3
nolines = 3

################ C
ax0 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)
ax1 = plt.subplot2grid((nolines,noplots), (0,1), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((nolines,noplots), (0,2), colspan=1, rowspan=1)
################ C
ax3 = plt.subplot2grid((nolines,noplots), (2,0), colspan=1, rowspan=1)
ax4 = plt.subplot2grid((nolines,noplots), (2,1), colspan=1, rowspan=1)
ax5 = plt.subplot2grid((nolines,noplots), (2,2), colspan=1, rowspan=1)

ax3a = plt.subplot2grid((nolines,noplots), (1,0), colspan=1, rowspan=1)
ax4a = plt.subplot2grid((nolines,noplots), (1,1), colspan=1, rowspan=1)
ax5a = plt.subplot2grid((nolines,noplots), (1,2), colspan=1, rowspan=1)

ax0.text(-0.4, 1.0, 'a', transform=ax0.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})
ax3.text(-0.4, 1.0, 'c', transform=ax3.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})
ax3a.text(-0.4, 1.0, 'b', transform=ax3a.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})


sys.path.append("../2017-03-28_AndreaNPS_varyotherparams_nostagezmovement/") #Same as above for pixel, new for kv and current
from Combining_data_with_prefix_onlyIntensVisibRatio import do_visib_other_qttties
do_visib_other_qttties(ax0,ax1,ax2,ax3,ax4,ax5,ax3a,ax4a,ax5a)
ax0.spines["top"].set_visible(False)
ax0.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax0.xaxis.set_ticks_position('bottom')
ax0.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)
ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')
ax4.xaxis.set_ticks_position('bottom')
ax4.yaxis.set_ticks_position('left')
ax5.xaxis.set_ticks_position('bottom')
ax5.yaxis.set_ticks_position('left')

ax3a.spines["top"].set_visible(False)
ax3a.spines["right"].set_visible(False)
ax4a.spines["top"].set_visible(False)
ax4a.spines["right"].set_visible(False)
ax5a.spines["top"].set_visible(False)
ax5a.spines["right"].set_visible(False)
ax3a.xaxis.set_ticks_position('bottom')
ax3a.yaxis.set_ticks_position('left')
ax4a.xaxis.set_ticks_position('bottom')
ax4a.yaxis.set_ticks_position('left')
ax5a.xaxis.set_ticks_position('bottom')
ax5a.yaxis.set_ticks_position('left')

plt.tight_layout()
   
multipage_longer_desired_aspect_ratio('Fig4SI.pdf',1600,1200,dpi=80,)
