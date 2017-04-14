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
fig1.set_size_inches(1200./fig1.dpi,900./fig1.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    

noplots = 2
nolines = 2

ax100 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)
#ax100.text(-0.1, 1.0, 'a', transform=ax100.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})       
ax100.spines['right'].set_visible(False)
ax100.spines['top'].set_visible(False)
ax100.xaxis.set_ticks_position('bottom')
ax100.yaxis.set_ticks_position('left')

d = get_data('test.dat')



# calculate distance from center line (x, y, z) -> (r, z)

dist = np.array([])
for n in range(len(d)):
    print(n)
    for k in range(d[n].shape[0]):
        print(k)
        x = d[n][k, 0]
        y = d[n][k, 1]
        z = d[n][k, 2]
        r = np.sqrt( x**2 + y**2 )

        hlp = np.array([z, r])
        dist = np.vstack((dist, hlp)) if dist.size else hlp

# binning

all_z = dist[:, 0]
all_r = dist[:, 1]

bins = np.linspace(1, 1200, 500)
inds = np.digitize(all_z, bins)

dist_r = []
std_z = []
std_r = []
for k in range(len(bins)):
    hlp = all_r[k == inds]
    dist_r.append(hlp)
    # we want the mean of the radius
    std_r.append(np.mean(hlp))

    hlp = bins[k]
    std_z.append(hlp)


ax100.plot(std_z, std_r, 'o',color='k')
ax100.set_xlabel('Depth (nm)',fontsize=fsizepl)
ax100.set_ylabel('Mean simulated radial extent \n of electron trajectories (nm)',fontsize=fsizepl)

ax100.axvspan(0,120, alpha=0.25, color='grey')
ax100.axvspan(120,120+200, alpha=0.25, color='yellow')
ax100.text(5, 10, 'Nanoparticle', fontsize=fsizenb) 
ax100.text(125, 35, r'SiO$_2$', fontsize=fsizenb) 
ax100.text(325, 60, 'Si substrate', fontsize=fsizenb) 

ax100.set_xticks([250,500,750,1000])
ax100.set_xlim([0,1200])
ax100.set_yticks([100,200,300,400,500])#,200])#
ax100.set_ylim([0,550])#
ax100.tick_params(labelsize=fsizenb)

plt.tight_layout() 

multipage_longer('CasinoOLD.pdf',dpi=80)

