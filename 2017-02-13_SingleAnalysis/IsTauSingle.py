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

### settings
fsizepl = 24
fsizenb = 20
sizex = 8
sizey=6
dpi_no = 80

do_plottau = True
if do_plottau:
    
    fig1= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig1.set_size_inches(1200./fig1.dpi,900./fig1.dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')    
    
    noplots = 2
    nolines = 2
    
    ax00b = plt.subplot2grid((nolines,noplots), (1,1), colspan=1, rowspan=1)
    ax00b.spines['left'].set_visible(False)
    ax00b.spines['top'].set_visible(False)
    ax00b.xaxis.set_ticks_position('bottom')
    ax00b.yaxis.set_ticks_position('right')
    
    ax100b = plt.subplot2grid((nolines,noplots), (1,0), colspan=1, rowspan=1)
    ax100b.text(-0.1, 1.0, 'b', transform=ax100b.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})       
    ax100b.spines['right'].set_visible(False)
    ax100b.spines['top'].set_visible(False)
    ax100b.xaxis.set_ticks_position('bottom')
    ax100b.yaxis.set_ticks_position('left')
    
    ax00 = plt.subplot2grid((nolines,noplots), (0,1), colspan=1, rowspan=1)
    ax00.spines['left'].set_visible(False)
    ax00.spines['top'].set_visible(False)
    ax00.xaxis.set_ticks_position('bottom')
    ax00.yaxis.set_ticks_position('right')
    
    ax100 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)
    ax100.text(-0.1, 1.0, 'a', transform=ax100.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})       
    ax100.spines['right'].set_visible(False)
    ax100.spines['top'].set_visible(False)
    ax100.xaxis.set_ticks_position('bottom')
    ax100.yaxis.set_ticks_position('left')
    
    ax00.set_ylabel(r'Red band $\tau$ ($\mu$s)',fontsize=fsizepl)
    ax00.yaxis.set_label_position("right")
    ax100.set_ylabel(r'Green band $\tau$ ($\mu$s)',fontsize=fsizepl)
    
    ax00b.set_ylabel(r'Slope $\partial$(red band $\tau$)/$\partial$t (a.u.)',fontsize=fsizepl)
    ax00b.yaxis.set_label_position("right")
    ax100b.set_ylabel(r'Slope $\partial$(green band $\tau$)/$\partial$t (a.u.)',fontsize=fsizepl)
    
    def tauestimate(counts_red, error_red, counts_blue, error_blue):
    
        print(counts_red.shape[1])
        
        ucounts_red = unumpy.uarray(counts_red, error_red)
        ucounts_blue = unumpy.uarray(counts_blue, error_blue)
        
        def helper(arrayx):
             return np.cumsum(arrayx*np.arange(1,counts_red.shape[1]+1), axis = 1)/np.cumsum(arrayx, axis = 1)
        
        return helper(ucounts_red),helper(ucounts_blue)
        
    def moving_average(a,n=3):
        ret = np.cumsum(a)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n-1:]/n

    def plotinho(ax0, dset,my_color,ax0b,my_edgecolor, my_facecolor):   
        
        movav = 100
        
#        if my_color is 'r':
#            ax0.text(1000, 165, 'temperature \n increases', fontsize=fsizenb, va='center',ha='center')
#            ax0.annotate('', xy=(1000,75), xytext=(1000,150),
#                arrowprops=dict(facecolor='black', shrink=0.05))              
        
        ax0.plot(moving_average(np.arange(1,taured.shape[1]+1),movav),moving_average(unumpy.nominal_values(dset[0,:]),movav),color=my_color,ls='--',lw=2)
       
        ax0.fill_between(moving_average(np.arange(1,taured.shape[1]+1),movav),
                                                 moving_average(unumpy.nominal_values(dset[0,:]),movav)-moving_average(unumpy.std_devs(dset[0,:]),movav),
                                                 moving_average(unumpy.nominal_values(dset[0,:]),movav)+moving_average(unumpy.std_devs(dset[0,:]),movav),
                                                 edgecolor=my_edgecolor,
                                                 facecolor=my_facecolor,
                                                 alpha=0.5,
                                                 linewidth=0.0)
         
        ax0.plot(moving_average(np.arange(1,taured.shape[1]+1),movav),moving_average(np.arange(1,taured.shape[1]+1)/2,movav),color='k',ls='--',lw=2)        
        
        ax0b.plot(moving_average(np.arange(2,taured.shape[1]+1),movav),moving_average(np.diff(unumpy.nominal_values(dset[0,:])),movav),color=my_color,ls='--',lw=2)
      
        ax0b.axhline(y=0.5, xmin=0, xmax=130, lw=2, color = 'k', ls='--')
        ax0b.set_yticks([0.25,0.5])#,200])#
        ax0b.set_ylim([0,0.55])#
        
        my_x= moving_average(np.arange(1,taured.shape[1]+1),movav)
      
        ax0b.set_xlim(xmin= my_x[0],xmax = my_x[-1])#    
        ax0.set_xlim(xmin= my_x[0],xmax = my_x[-1])       
        
        ax0.set_xticks([500,1000])
        ax0b.set_xticks([500,1000])
        ax0b.set_xticklabels([500,1000])
        ax0.set_xticklabels([])
        ax0b.set_xlabel('Transient cathodoluminescence \n acquisition time ($\mu$s)',fontsize=fsizepl)
        
        ax0.set_yticks([100,200,300,400,500,600])#,200])#
        ax0.set_ylim([0,610])#
        ax0.tick_params(labelsize=fsizenb)
        ax0b.tick_params(labelsize=fsizenb)
        
    print('bef loading')
    C = pickle.load( open( "C.p", "rb" ) ) #pickle.load( open( "d4.p", "rb" ) ) #4#
    non_cut_k = C['non_cut_k']  
    import matplotlib.cm as cm
    print(C['taured'][non_cut_k,:].shape)
    taured = unumpy.uarray(C['taured'][non_cut_k,:], C['tauredstd'][non_cut_k,:])
    taublue =  unumpy.uarray(C['taublue'][non_cut_k,:], C['taubluestd'][non_cut_k,:])
    print('here') 
    plotinho(ax00, taured,'r',ax00b ,my_edgecolor='#ff3232', my_facecolor='#ff6666')
    plotinho(ax100, taublue,'g',ax100b,my_edgecolor='#74C365', my_facecolor='#74C365')
    
    D = pickle.load( open( "D.p", "rb" ) ) #pickle.load( open( "d4.p", "rb" ) ) #4#
    non_cut_k = D['non_cut_k']  
    taured = unumpy.uarray(D['taured'][non_cut_k,:], D['tauredstd'][non_cut_k,:])
    taublue =  unumpy.uarray(D['taublue'][non_cut_k,:], D['taubluestd'][non_cut_k,:])
    plotinho(ax00, taured,'DarkRed',ax00b ,my_edgecolor='#801515', my_facecolor='#801515')
    plotinho(ax100, taublue,'#003100',ax100b,my_edgecolor='#003D1B', my_facecolor='#003D1B')
    
    plt.tight_layout() 
    #plt.show()  
    multipage_longer('IsTauSingle.pdf',dpi=80)




