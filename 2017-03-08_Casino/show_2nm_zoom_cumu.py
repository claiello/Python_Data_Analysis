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

#plt.plot(bin_r[0:-1],np.cumsum(np.sum(np.abs(result[0:27,:]),axis=0))/np.cumsum(np.sum(np.abs(result[0:27,:]),axis=0))[12])
#plt.plot(bin_r[0:-1],np.sum(np.abs(result[0:27,:]),axis=0)/np.sum(np.abs(result[0:27,:]),axis=0)[12])

deltar = 2.2140811635188982
deltaz = 5.8852084000000104

plt.pcolor(bin_r, bin_z, np.abs(result)*deltar*deltaz*1000.0/2.29388 > 1.0) #2.29eV is the green transition
#>1.0 : area where cancreate green excitations
plt.colorbar()

mask = np.abs(result[0:27,0:12])*deltar*deltaz*1000.0/2.29388 > 1.0 #where energy dumped can drive a green transition or higher
from scipy import ndimage
resu = ndimage.measurements.center_of_mass(mask* result[0:27,0:12])
print(resu)

kklkl
resu is (15.793515146933403 Z, 3.6889794250989261 R)

from scipy import ndimage
res = ndimage.measurements.center_of_mass( np.abs(result[0:27,0:12])*deltar*deltaz*1000.0/2.29388 > 1.0)
print(res)

print('np vol in pcentage then nm^3')
print(np.sum((np.abs(result[0:27,0:12])*deltar*deltaz*1000.0/2.29388 > 1.0))/(12*27)*100)
print(np.sum((np.abs(result[0:27,0:12])*deltar*deltaz*1000.0/2.29388 > 1.0))/(12*27)*120*1623.8)

print(np.sum(np.sum((np.abs(result[0:27,0:12])*deltar*deltaz*1000.0/2.29388 > 1.0), axis = 0)))

print(np.sum((np.abs(result[0:27,0:12])*deltar*deltaz*1000.0/2.29388 > 1.0))/(12*27)*120*1623.8)



ljjlj

#center of mass
#(18.031055900621119, 5.2608695652173916)
#(Z,R)

plt.show()

klklk

#plt.ylim([0,1.05])
plt.axvline(25)
plt.axhline(1)
plt.axhline(0.5)
#plt.xlim([0,27])
plt.show()





klklk

plt.pcolor(bin_r, bin_z, np.sum(np.abs(result)))#/np.nanmax(np.abs(result)))
plt.xlabel('Simulated radial extent \n of electron trajectories (nm)',fontsize=fsizepl)
plt.ylabel('Simulated depth \n of electron trajectories (nm)',fontsize=fsizepl)
cb = plt.colorbar(ticks=[0.5, 1])
#cb.ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
cb.set_label(label='Energy deposited (a.u.)',size=fsizepl)
cb.ax.tick_params(labelsize=fsizenb)

plt.xlim(xmin=0,xmax=75)
plt.ylim(ymin=130,ymax=0)

plt.clim(vmin=0, vmax = 1)#0.004/np.nanmax(np.abs(result)))#-0.004, vmax=0)

plt.hlines(y=120, xmin=0, xmax=25, color = 'white',lw=1)
plt.vlines(x=25, ymin=0, ymax=120, color = 'white',lw=1)
#plt.grid(color='yellow',which='major')

#plt.axhline(y=120, linewidth=0.5, color = 'white',lw=0.5, linestyle='--')
#plt.axhline(y=320, linewidth=0.5, color = 'white',lw=0.5, linestyle='--')
#plt.axvline(x=25,  linewidth=0.5, color = 'white',lw=0.5, linestyle='--')
#plt.axvline(x=75,  linewidth=0.5, color = 'white',lw=0.5, linestyle='--')
#plt.axvline(x=125,  linewidth=0.5, color = 'white',lw=0.5, linestyle='--')
#plt.axvline(x=175,  linewidth=0.5, color = 'white',lw=0.5, linestyle='--')

plt.xticks([5,10,15,20,25, 75])
plt.yticks([10,20,30,40,50,60,70,80,90,100,110,120,130])
#plt.xaxis.set_ticks_position('bottom')
#plt.yaxis.set_ticks_position('left')
plt.tick_params(labelsize=fsizenb)

#plt.text(125,122, 'Bottom of nanoparticles', fontsize=fsizenb, va='center',ha='center',color='white')
#plt.text(125,325, r'Bottom of SiO$_2$', fontsize=fsizenb, va='center',ha='center',color='white')
#plt.text(125,410, 'Si substrate', fontsize=fsizenb, va='center',ha='center',color='white')

#plt.figure()
#plt.plot(all_r, all_z)

#plt.show()

plt.tight_layout() 

multipage_longer_desired_aspect_ratio('CasinoNEWzoomCUMU.pdf',xsize=800, ysize=600, dpi=80)


