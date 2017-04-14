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

plt.pcolor(bin_r, bin_z, np.abs(result)/np.max(np.abs(result)))#/np.nanmax(np.abs(result)))
plt.xlabel('Simulated radial extent \n of electron trajectories (nm)',fontsize=fsizepl)
plt.ylabel('Simulated depth \n of electron trajectories (nm)',fontsize=fsizepl)
cb = plt.colorbar(ticks=[0.5, 1])
#cb.ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
cb.set_label(label='Energy deposited (a.u.)',size=fsizepl)
cb.ax.tick_params(labelsize=fsizenb)

plt.xlim(xmin=0,xmax=225)
plt.ylim(ymin=500,ymax=0)

plt.clim(vmin=0, vmax = 1)#0.004/np.nanmax(np.abs(result)))#-0.004, vmax=0)

plt.hlines(y=120, xmin=0, xmax=25, color = 'white',lw=1)
plt.vlines(x=25, ymin=0, ymax=120, color = 'white',lw=1)
#plt.grid(color='yellow',which='major')

plt.axhline(y=120, linewidth=0.5, color = 'white',lw=0.5, linestyle='--')
plt.axhline(y=320, linewidth=0.5, color = 'white',lw=0.5, linestyle='--')
plt.axvline(x=25,  linewidth=0.5, color = 'white',lw=0.5, linestyle='--')
plt.axvline(x=75,  linewidth=0.5, color = 'white',lw=0.5, linestyle='--')
plt.axvline(x=125,  linewidth=0.5, color = 'white',lw=0.5, linestyle='--')
plt.axvline(x=175,  linewidth=0.5, color = 'white',lw=0.5, linestyle='--')

plt.xticks([25, 75, 125, 175, 225])
plt.yticks([120, 320, 500])
#plt.xaxis.set_ticks_position('bottom')
#plt.yaxis.set_ticks_position('left')
plt.tick_params(labelsize=fsizenb)

plt.text(125,122, 'Bottom of nanoparticles', fontsize=fsizenb, va='center',ha='center',color='white')
plt.text(125,325, r'Bottom of SiO$_2$', fontsize=fsizenb, va='center',ha='center',color='white')
plt.text(125,410, 'Si substrate', fontsize=fsizenb, va='center',ha='center',color='white')

#plt.figure()
#plt.plot(all_r, all_z)

#plt.show()

plt.tight_layout() 

multipage_longer_desired_aspect_ratio('CasinoNEW.pdf',xsize=800, ysize=600, dpi=80)


