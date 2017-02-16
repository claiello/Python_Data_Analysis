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
#from sklearn.mixture import GMM 
import matplotlib.cm as cm
#from FluoDecay import *
#from PlottingFcts import *
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.misc
#import matplotlib.animation as animation
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
###

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

ax1 = plt.subplot2grid((nolines,noplots), (0,1), colspan=1, rowspan=1)
ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('right')

ax3 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')

ax4 = plt.subplot2grid((nolines,noplots), (1,0), colspan=1, rowspan=1)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.xaxis.set_ticks_position('bottom')
ax4.yaxis.set_ticks_position('left')

def tauestimate(counts_red, error_red, counts_blue, error_blue):
    
    print(counts_red.shape[1])
    
    ucounts_red = unumpy.uarray(counts_red, error_red)
    ucounts_blue = unumpy.uarray(counts_blue, error_blue)
    
    def helper(arrayx):
         return np.cumsum(arrayx*np.arange(1,counts_red.shape[1]+1), axis = 1)/np.cumsum(arrayx, axis = 1)
    
    return helper(ucounts_red),helper(ucounts_blue)

d_ap = pickle.load( open( "d0.p", "rb" ) ) #4#
d_kv = pickle.load( open( "d1.p", "rb" ) ) #3#
d_pixel = pickle.load( open( "d2.p", "rb" ) ) #5#
d_temp = pickle.load( open( "d4.p", "rb" ) ) #4#

import matplotlib.cm as cm
ct = 3
for data in [d_temp]: #[d_ap, d_kv, d_pixel, d_temp]:
    
    if ct == 3:
        xvec = np.array([0.002996,0.0031089,0.0031954,0.003285])
        fno = 30
    
    (taured,taublue) = tauestimate(data['red1D'], np.sqrt(data['red1D']),data['blue1D'], np.sqrt(data['blue1D']))
    ts_b = np.arange(0,data['red1D'].shape[1])
    
    aa = np.empty([taured.shape[1]])    
    cc = np.empty([taured.shape[1]])   
    aaerr = np.empty([taured.shape[1]])    
    ccerr = np.empty([taured.shape[1]])   
    bb = np.empty([taured.shape[1]])    
    dd = np.empty([taured.shape[1]]) 
    for jjj in np.arange(1,taured.shape[1]):
        print(jjj)
        (a,b,result) = linear_fit_with_error((1.0/xvec - 273.15), unumpy.nominal_values(taured[:,jjj]), unumpy.std_devs(taured[:,jjj]))
        (c,d,result2) = linear_fit_with_error((1.0/xvec - 273.15) , unumpy.nominal_values(taublue[:,jjj]), unumpy.std_devs(taublue[:,jjj]))
        aa[jjj] = a
        cc[jjj] = c
        bb[jjj] = b
        dd[jjj] = d
        aaerr[jjj] = result.params['a'].stderr
        ccerr[jjj] = result2.params['a'].stderr
        
    step = 50
    colors = iter(cm.rainbow(np.linspace(0, 1, len(np.arange(0,taured.shape[1],step)))))                 
    for jjj in np.arange(0,taured.shape[1],step):
        colorful = next(colors)
        hlp1= unumpy.nominal_values(taured[:,jjj])
        hlp2= unumpy.nominal_values(taublue[:,jjj])
        #hlpcumu below is VISIB of CUMU of TAUS
        hlpcumu = (np.cumsum(unumpy.nominal_values(taured[:,jjj]))-np.cumsum(unumpy.nominal_values(taublue[:,jjj])))/(np.cumsum(unumpy.nominal_values(taured[:,jjj]))+np.cumsum(unumpy.nominal_values(taublue[:,jjj])))
        
        ax1.plot(1/(xvec) - 273.15,hlp1 ,marker='o',ls='None',color=colorful,markersize=8)
        ax1.plot(1/(xvec) - 273.15,aa[jjj]*(1/(xvec) - 273.15) + bb[jjj],ls='-',color=colorful,markersize=8)   
      
        ax3.plot(1/(xvec) - 273.15,hlp2 ,marker='o',ls='None',color=colorful,markersize=8)
        ax3.plot(1/(xvec) - 273.15,cc[jjj]*(1/(xvec) - 273.15) + dd[jjj],ls='-',color=colorful,markersize=8)
        
        if jjj == 0:
            ax1.text(55, 175, '   ' + 'time in\n 50 $\mu$s intervals', fontsize=fsizenb, va='center',ha='center')
            ax1.annotate('', xy=(55,250), xytext=(55,200),
                arrowprops=dict(facecolor='black', shrink=0.05))   
            ax3.text(-0.1, 1.0, 'a', transform=ax3.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})
        
        #'Time in 50 $\mu$s \n    intervals
        ax1.set_ylabel(r'Red band $\tau$ ($\mu$s)',fontsize=fsizepl)
        ax1.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)
        ax1.tick_params(labelsize=fsizenb)
        ax1.set_ylim([0,305])
        ax1.set_xlim([25, 65])
        ax1.set_xticks([30,40,50,60]) 
        ax1.set_yticks([100,200,300])
        ax1.yaxis.set_label_position("right")
        
        ax3.set_ylabel(r'Green band $\tau$ ($\mu$s)',fontsize=fsizepl)
       
        ax3.set_ylim([0,305])
        ax3.set_xlim([25,65])
        ax3.set_yticks([100,200,300])
        ax3.set_xticks([30,40,50,60])
        ax3.tick_params(labelsize=fsizenb)
        ax3.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)
       
    #labels
    xmin = 0
    xmax = 1390
    ax4.plot(np.arange(1,taured.shape[1]+1),aa,ls='--',color='r',lw=2)
    ax4.plot(np.arange(1,taured.shape[1]+1),cc,ls='--',color='g',lw=2)
    ax4.text(-0.1, 1.0, 'b', transform=ax4.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})
    
    #fit a linearly
    initpt=750
    (redslopea,redslopeb,reso) = linear_fit_with_error(np.arange(1+initpt,taured.shape[1]+1),aa[initpt:],aaerr[initpt:])
    (blueslopea,blueslopeb,reso2) = linear_fit_with_error(np.arange(1+initpt,taured.shape[1]+1),cc[initpt:],ccerr[initpt:])
    ax4.plot(np.arange(1+initpt,taured.shape[1]+1),redslopea*np.arange(1+initpt,taured.shape[1]+1) + redslopeb,ls='-',color='r',lw=2)
    ax4.plot(np.arange(1+initpt,taured.shape[1]+1),blueslopea*np.arange(1+initpt,taured.shape[1]+1) + blueslopeb,ls='-',color='g',lw=2)
    ax4.text(-1, 500,'test1', fontsize=fsizenb) 
    ax4.text(-2, 1000,'test2', fontsize=fsizenb)
    annotatered = 'slope \n ' + str("{0:.2f}".format(redslopea*1000))+ ' $\cdot$ 10$^{-3}$ $^{\circ}$C$^{-1}$'
    ax4.annotate(annotatered, xy=(1000, -0.9), xytext=(1000,0.1), fontsize=fsizenb,
            arrowprops=dict(facecolor='r', edgecolor='None', shrink=0.05), va='center',ha='center')
    annotategreen = 'slope \n ' + str("{0:.2f}".format(blueslopea*1000))+ ' $\cdot$ 10$^{-3}$ $^{\circ}$C$^{-1}$'
    ax4.annotate(annotategreen, xy=(900, -2.2), xytext=(900,-3.75), fontsize=fsizenb,
            arrowprops=dict(facecolor='g', edgecolor='None', shrink=0.05), va='center',ha='center')
    ax4.set_ylabel('Slope ' +  r'$\alpha$ = $\partial\tau$/$\partial$T' + ' ($\mu$s $\cdot$ $^{\circ}$C$^{-1}$)',fontsize=fsizepl)
    ax4.set_xlabel('Transient cathodoluminescence \n acquisition time ($\mu$s)',fontsize=fsizepl)
    ax4.tick_params(labelsize=fsizenb)
    ax4.set_xticks([500,1000])#,1500])
    ax4.set_ylim((-4.2,0.5))
    ax4.set_yticks([-4,-3,-2,-1,0])
        
    sys.path.append("../2017-01-28_Gradient_Sensitivity/") # necessary for the tex fonts
    from calc_sens import get_sens  
    (xxx, eta_taured)  = get_sens(taured, 1/(xvec) - 273.15,np.arange(1,taured.shape[1]+1))
    (xxx, eta_taublue)  = get_sens(taublue, 1/(xvec) - 273.15,np.arange(1,taured.shape[1]+1))
           
    ax0 = plt.subplot2grid((nolines,noplots), (1,1), colspan=1, rowspan=1)
    ax0.semilogy(np.arange(1+300,taured.shape[1]+1),eta_taured[300:] ,color='r',ls='--',lw=2)
    ax0.semilogy(np.arange(1+2,taured.shape[1]+1),eta_taublue[2:] ,color='g',ls='--',lw=2)
    ax0.text(-0.1, 1.0, 'c', transform=ax0.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.xaxis.set_ticks_position('bottom')
    ax0.yaxis.set_ticks_position('left')
    ax0.set_xticks([500,1000])#,1500])
    ax0.set_ylim((3.8,21))
    ax0.set_ylabel(r'$\delta$T $\equiv$ $\sigma_{\tau}$/$\vert\partial\tau$/$\partial$T$\vert$ ($^{\circ}$C)',fontsize=fsizepl)
    ax0.set_xlabel('Transient cathodoluminescence \n acquisition time ($\mu$s)',fontsize=fsizepl)
    ax0.set_yticks([10])
    ax0.set_yticklabels(['10'])
    ax0.set_xlim([xmin, xmax])
    ax0.tick_params(labelsize=fsizenb) 

plt.tight_layout()
multipage_longer('Fig5.pdf',dpi=80)

klklk