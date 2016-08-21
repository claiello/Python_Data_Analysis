import os
import sys
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
#from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation
import gc
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../Python modules/") # necessary for the tex fonts
from MakePdf import *

fsizetit = 18
fsizepl = 16
sizex = 8
sizey = 6
dpi_no = 80
lw = 2

import scalebars as sb
import matplotlib.gridspec as gridspec

tau1red =       [0.2,0.26,0.36,0.37,0.77,0.96,0.45,0.23,0.50,0.16]
tau1red_error = [0.02, 0.03, 0.07, 0.06, 0.09, 0.12, 0.26, 0.09, 0.24, 0.04]

tau2red =       [0.02, 0.03, 0.03, 0.01, 0.03, 0.04, 0.03, 0.01, 0.02, 0.01, 0.02]
tau2red_error = [0.005,0.005, 0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01, 0.005]

tau1blue =       [0.3,0.24,0.21,0.32,0.33,0.35,0.12,0.2,0.18,0.19,0.12]
tau1blue_error = [0.02, 0.02, 0.03, 0.05, 0.03, 0.03, 0.04, 0.05, 0.07, 0.07,0.02]

tau2blue =       [0.03, 0.02, 0.02, 0.01, 0.03, 0.03, 0.02, 0.01, 0.02, 0.02, 0.01]
tau2blue_error = [0.005,0.005, 0.005, 0.005, 0.005, 0.005, 0.02, 0.01, 0.01, 0.01, 0.005]

tau1all =       [0.28, 0.24, 0.23,0.34,0.40,0.44,0.15,0.16,0.18,0.16]
tau1all_error = [0.02, 0.02, 0.03, 0.04, 0.03,0.03,0.04, 0.005, 0.06, 0.04]

tau2all =       [0.03, 0.02, 0.02, 0.01, 0.03, 0.03, 0.02, 0.01, 0.02, 0.02, 0.02]
tau2all_error = [0.005, 0.005,0.005,0.005,0.005,0.005,0.01,0.005,0.005,0.01,0.005]  

x_vec = [2,2,20,20,36,36,44,44,80,80,-2]

plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')
plt.suptitle('Cathodoluminescence decay rates as a function of Er $\%$', fontsize=fsizetit)

ax1 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((2,2), (1,0), colspan=1, sharex=ax1)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

#ax1.set_ylim([0.55,0.85])
#ax2.set_ylim([0.0,0.1])


ax1.errorbar(x_vec[:-1], tau1all, yerr=tau1all_error, fmt='ko',markersize=10)
ax2.errorbar(x_vec, tau2all, yerr=tau2all_error, fmt='ks', markersize=5)
ax1.set_ylabel('Longer time constant ($\mu$s)',fontsize=fsizepl)
ax2.set_ylabel('Shorter time constant ($\mu$s)',fontsize=fsizepl)
plt.xlabel(r"Er content ($\%$)",fontsize=fsizepl) 
major_ticks = [-2,2,20,36,44,80]
ax1.set_xticks(major_ticks)
ax1.set_xticklabels(['OA','2','20','40 CS','40 C','80'])
plt.xlim([-6,84])

ax1.set_yticks([0.25, 0.5, 0.75, 1.0, 1.25])
ax2.set_yticks([0.025, 0.050])

ax1.set_title('For all pixels',fontsize=fsizepl)
 
ax1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((2,2), (1,1), colspan=1, sharex=ax1)

ax1.set_title(r'For $\tau$' +  ' of red/blue photons ($>$/$<$ 458nm)',fontsize=fsizepl)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

ax1.errorbar(x_vec[:-1], tau1red, yerr=tau1red_error, fmt='ro',markersize=10)
ax2.errorbar(x_vec, tau2red, yerr=tau2red_error, fmt='rs', markersize=5)
ax1.errorbar(x_vec, tau1blue, yerr=tau1blue_error, fmt='bo',markersize=10)
ax2.errorbar(x_vec, tau2blue, yerr=tau2blue_error, fmt='bs', markersize=5)

ax1.set_xticks(major_ticks)
ax1.set_xticklabels(['OA','2','20','40 CS','40 C','80'])
plt.xlim([-4,84])

ax1.set_yticks([0.25, 0.5, 0.75, 1.0, 1.25])
ax2.set_yticks([0.025, 0.050])
#from uncertainties import unumpy
#tau1U = unumpy.uarray(tau1,tau1_error)
#tau2U = unumpy.uarray(tau2,tau2_error)
#tau1bgU = unumpy.uarray(tau1bg,tau1bg_error)
#tau2bgU = unumpy.uarray(tau2bg,tau2bg_error)
#
#ratio_large = tau1U/tau1bgU
#ratio_small = tau2U/tau2bgU
#large_nb = np.zeros(len(ratio_large))
#large_err = np.zeros(len(ratio_large))
#small_nb = np.zeros(len(ratio_large))
#small_err = np.zeros(len(ratio_large))
#for jj in np.arange(len(ratio_large)):
#    print(jj)
#    large_nb[jj] = float(str(ratio_large[jj]).partition('+/-')[0])
#    large_err[jj] = float(str(ratio_large[jj]).partition('+/-')[2])
#    small_nb[jj] = float(str(ratio_small[jj]).partition('+/-')[0])
#    small_err[jj] = float(str(ratio_small[jj]).partition('+/-')[2])
#
##ax1.set_ylim([0.55,0.85])
##ax2.set_ylim([0.0,0.1])
#
#ax1.errorbar(x_vec, large_nb, yerr=large_err, fmt='ko',markersize=10)
#ax1.axhline(y=1, xmin=0, xmax=60,linewidth=2, color = 'k', ls = '--')
#
#ax2.errorbar(x_vec, small_nb, yerr=small_err, fmt='ks', markersize=5)
#ax2.axhline(y=1, xmin=0, xmax=60,  linewidth=2, color = 'k',ls = '--')
#
#ax1.set_ylabel('Ratio of longer time constants',fontsize=fsizepl)
#ax2.set_ylabel('Ratio of shorter time constants',fontsize=fsizepl)
#plt.xlabel(r"Er content ($\%$)",fontsize=fsizepl) 
#major_ticks = [2,10,20,40,60]
#ax1.set_xticks(major_ticks) 
#plt.xlim([0,62])
 
multipage('TausComparison.pdf',dpi=80)
    
