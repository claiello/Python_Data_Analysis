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
#from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation
import gc
sys.path.append("/usr/bin") # necessary for the tex fonts
from MakePdf import *

fsizetit = 26
fsizepl = 20
sizex = 8
sizey = 6
dpi_no = 80
lw = 2

import scalebars as sb
import matplotlib.gridspec as gridspec

#III - Vb - Vr - 11B - 11r
#'Sigma-AldrichA', 'Er 2$\%$ Blue', 'Er 2$\%$ Red', 'Er 10$\%$ Blue', 'Er 10$\%$ Red'

##### CL
tau1 =       [0.08, 0.16,0.06,  0.05, 0.03,   0.06,  0.19,           0.14, 0.12 ]
tau1_error = [0.01, 0.009,0.039,  0.021, 0.01,  0.004,0.034,     0.002,0.035]

tau2 =       [0.42, 0.812,0.365,  0.5, 1.5,   1.116,  9.122,    0.516, 0.446]
tau2_error = [0.48,0.225,0.055,  0.456,  1.925,   0.145, 34.641,       0.015, 0.166]

###### Fluo
tau1bg =       [0.12,0.09,0.044,  np.nan, np.nan, np.nan, np.nan,       0.09,0.04]
tau1bg_error = [0.01,0.01,0.002,  np.nan, np.nan, np.nan, np.nan,          0.01,0.01]

tau2bg =       [1.17,1.12,0.98,  np.nan, np.nan, np.nan, np.nan,            1.12,0.98]
tau2bg_error = [0.04,0.03,0.03,  np.nan, np.nan, np.nan, np.nan,            0.03,0.03]  

x_vec = [1,2,3,4,5,6,7,8,9]

plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')
plt.suptitle('Comparison TR-CL vs. Fluorometer (7$\mu$s-long decay, 100ns time bins): \n close enough, but not within error bars', fontsize=fsizetit)

ax2 = plt.subplot2grid((2,1), (0,0), colspan=1, rowspan=1)
ax1 = plt.subplot2grid((2,1), (1,0), colspan=1, sharex=ax2)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

#ax1.set_ylim([0.55,0.85])
ax2.set_ylim([0.25,1.75])
ax2.set_yticks([0.5, 1.0, 1.5])
ax1.set_yticks([0.05, 0.15, 0.25])
ax1.set_ylim([0,0.3])

ax1.errorbar(x_vec, tau1, yerr=tau1_error, fmt='yo',markersize=5, label='TR-CL')

ax2.errorbar(x_vec, tau2, yerr=tau2_error, fmt='yo', markersize=10, label='TR-CL')
ax1.errorbar(x_vec, tau1bg, yerr=tau1bg_error, fmt='ro',markersize=5, label='Fluorometer')
ax2.errorbar(x_vec, tau2bg, yerr=tau2bg_error, fmt='ro', markersize=10,label='Fluorometer')
ax1.set_ylabel('Shorter time constant ($\mu$s)',fontsize=fsizepl)
ax2.set_ylabel('Longer time constant ($\mu$s)',fontsize=fsizepl)
#plt.xlabel(r"Samples",fontsize=fsizepl) 
#major_ticks = [1,2,3,4,5]
#ax1.set_xticks(major_ticks) 
#ax1.set_xtick_labels(['SA', 'Er 2$\%$ Blue', 'Er 2$\%$ Red', 'Er 10$\%$ Blue', 'Er 10$\%$ Red']) 
plt.xlim([0,10])
ax2.legend(fontsize=fsizepl)
#ax1.set_xticks([1,2,3,4,5,6,7,8,9])

plt.sca(ax1) 
plt.xticks([1,2,3,4,5,6,7,8,9],[
'Sigma Aldrich (Ho) \n on SiO$_2$', 
'Er 2$\%$ core-shell \n on SiO$_2$, blue', 
'Er 2$\%$ core-shell \n on SiO$_2$, red',
'Er 2$\%$ core-shell \n on Si, blue', 
'Er 2$\%$ core-shell \n on Si, red' ,  
'Er 2$\%$ core only \n on Si, blue', 
'Er 2$\%$ core only \n on Si, red', 
'Er 10$\%$ core-shell \n on SiO$_2$, blue', 
'Er 10$\%$ core-shell \n on SiO$_2$, red'],fontsize=10)#, rotation='30')
plt.setp(ax2.get_xticklabels(), visible=False)

ax2.annotate('9 $\pm$ 34 (!)', xy=(7, 1.75), xytext=(6.3, 1.5),arrowprops=dict(facecolor='black', shrink=0.05))

#ax1.set_title('For signal pixels',fontsize=fsizepl)
 
#ax1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
#ax2 = plt.subplot2grid((2,2), (1,1), colspan=1, sharex=ax1)
#
#ax1.set_title(r'For $\tau$' +  ' of signal pixels $\div$' +  r' $\tau$' + ' of background pixels',fontsize=fsizepl)
#
#ax1.spines['right'].set_visible(False)
#ax1.spines['top'].set_visible(False)
#ax1.xaxis.set_ticks_position('bottom')
#ax1.yaxis.set_ticks_position('left')
#
#ax2.spines['right'].set_visible(False)
#ax2.spines['top'].set_visible(False)
#ax2.xaxis.set_ticks_position('bottom')
#ax2.yaxis.set_ticks_position('left')
#
#plt.xlabel(r"Samples",fontsize=fsizepl) 
#plt.xticks([1,2,3,4,5], ['SA', 'Er 2$\%$ Blue', 'Er 2$\%$ Red', 'Er 10$\%$ Blue', 'Er 10$\%$ Red'], rotation='vertical')

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
    
