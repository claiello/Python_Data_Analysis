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

fsizepl = 24 #20 #16
fsizenb = 20
sizex = 8 #10 #8
sizey = 6# 10 #6
dpi_no = 80
lw = 2

#######

ax1 = plt.subplot2grid((2,3), (0, 2), colspan=1)
time_bin = 0.04
cutpointsatbeginning = 75
yap = np.load('ZZZYAP3.npz') #yap['datared'], yap['datablue']
#######
#######

fig50= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig50.set_size_inches(1200./fig50.dpi,900./fig50.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    
#plt.suptitle("YAP and ZnO:Ga decay upon near-impulsive excitation",fontsize=fsizetit)

#YAP decay
ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
#ax1.set_title("YAP",fontsize=fsizepl)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
blue = np.average(yap['datablue'],axis=(0))/1.0e3 #MHz
last_pt_offset = -10 #sometimes use -1, last point, but sometimes this gives 0. -10 seems to work
Time_bin = 40 #ns
middlept = -30
initdecay = 83
#init_guess = [np.average(blue[initdecay,:,:]), 2.0, np.average(blue[last_pt_offset,:,:]), np.average(blue[middlept,:,:]), 0.05] #e init was 0.5

signal = blue[initdecay:,:,:]/np.max(np.average(blue[initdecay:,:,:],axis=(1,2)))

Time_bin = 40
cut_longa = 6
cut_shorta =3
init_tau_longa = 0.5
init_tau_shorta = 0.05
my_t = np.arange(0,blue[initdecay:,:,:].shape[0])*Time_bin
init_guess = calc_double_fit(my_t,np.average(signal,axis=(1,2)), dt=Time_bin*1e-9, cut_long=cut_longa, cut_short=cut_shorta, init_tau_long=init_tau_longa, init_tau_short=init_tau_shorta)
init_guess[2] = np.average(blue[0:75,:,:],axis=(0,1,2))/np.max(np.average(blue[initdecay:,:,:],axis=(1,2)))

init_guess = [0.84, 0.019, 0.0158,0.132, 0.22]
b,e,be,ee = calcdecay_subplot(signal, 
                              time_detail= Time_bin*1e-9,
                              titulo='Cathodoluminescence rate decay, bi-exponential fit, \n ',
                              single=False,
                              other_dset1=None,
                              other_dset2=None,
                              init_guess=init_guess,
                              unit='MHz')    
          

plt.semilogy(my_t/Time_bin,  0.84961314*np.exp(-my_t/Time_bin/b), 'g', lw=3) 
plt.semilogy(my_t/Time_bin,   0.13284771*np.exp(-my_t/Time_bin/e), 'g', lw=3)              
#plt.semilogy(my_t/Time_bin, np.exp(-my_t/e), 'r', lw=3)                    
                              
plt.xlim([0,2.5])
#major_ticks0 = [1,2]
plt.ylabel("Average cathodoluminescence per pixel (a.u.)",fontsize=fsizepl)
#ax1.set_xticks(major_ticks0) 

plt.ylim(ymin=1.0e-3)
#major_ticks0 = [1,2]
plt.xlabel("Transient cathodoluminescence \n acquisition time ($\mu$s)",fontsize=fsizepl)
plt.xlim([0,2.5])

ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
indu = 83
plt.plot(np.arange(1,blue[indu:,:,:].shape[0]+1)*Time_bin/1000, np.average(blue[indu:,:,:]/np.max(np.average(blue[indu:,:,:],axis=(1,2))),axis=(1,2)), 'r',marker = 'o')
plt.xlim([0,2.5])
plt.xticks([1,2])
plt.ylim([0,1.1])
plt.yticks([0.5,1])
plt.xlabel("Transient cathodoluminescence \n acquisition time ($\mu$s)",fontsize=fsizepl)
plt.ylabel("Approx. instrument \n response function (a.u.)",fontsize=fsizepl)




plt.show() 

    
multipage('Oneplot.pdf',dpi=80)  