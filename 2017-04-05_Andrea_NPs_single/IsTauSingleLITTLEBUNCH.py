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
        
        movav = 400
        
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
        
    #No_experiments = 1*np.ones([10]) #50 avgs
    #nametr = ['2017-04-05-1420_ImageSequence__400.000kX_10.000kV_30mu_9']
    #description = ['Andrea small NaYF4:Er'] 
    #let = ['x400littlebunch'] #no of pixels
    
    blue = np.load('Blue_decay_arrayLITTLEBUNCH.npz',mmap_mode='r')     
    red = np.load('Red_decay_arrayLITTLEBUNCH.npz',mmap_mode='r')
    bgblue = np.load('bgBlue_decay_arrayLITTLEBUNCH.npz',mmap_mode='r')     
    bgred = np.load('bgRed_decay_arrayLITTLEBUNCH.npz',mmap_mode='r') 

    red = red['data']
    blue = blue['data']
    bgred = bgred['data']
    bgblue = bgblue['data']      
    
    print('finite')
    print(np.sum(np.isfinite(red)))
    print(np.sum(np.isfinite(blue)))
    print(np.sum(np.isfinite(bgred)))
    print(np.sum(np.isfinite(bgblue)))
    
    taured, taublue = tauestimate(red,np.nanstd(red)/(50*np.sum(np.isfinite(red))),blue,np.nanstd(blue)/(50*np.sum(np.isfinite(blue))))
    bgtaured, bgtaublue = tauestimate(bgred,np.nanstd(bgred)/(50*np.sum(np.isfinite(bgred))),bgblue,np.nanstd(bgblue)/(50*np.sum(np.isfinite(bgblue))))
    
    print('here') 
    plotinho(ax00, taured,'r',ax00b ,my_edgecolor='#ff3232', my_facecolor='#ff6666')
    plotinho(ax100, taublue,'g',ax100b,my_edgecolor='#74C365', my_facecolor='#74C365')
    
    plotinho(ax00, bgtaured,'DarkRed',ax00b ,my_edgecolor='#801515', my_facecolor='#801515')
    plotinho(ax100, bgtaublue,'#003100',ax100b,my_edgecolor='#003D1B', my_facecolor='#003D1B')
    
    plt.tight_layout() 
    #plt.show()  
    multipage_longer('IsTauSingleLITTLEBUNCH.pdf',dpi=80)
    
    plt.figure()
    
        
    
    ####### TAU ABOVE
    ####### INTENSITY BELOW
    
    del red, blue, bgred, bgblue
    
    blue = np.load('Blue_int_arrayLITTLEBUNCH.npz',mmap_mode='r')     
    red = np.load('Red_int_arrayLITTLEBUNCH.npz',mmap_mode='r')
    bgblue = np.load('bgBlue_int_arrayLITTLEBUNCH.npz',mmap_mode='r')     
    bgred = np.load('bgRed_int_arrayLITTLEBUNCH.npz',mmap_mode='r') 
    
    red = red['data']
    blue = blue['data']
    bgred = bgred['data']
    bgblue = bgblue['data']      

    plt.figure()

    plt.plot([0],np.array(red),marker='o',color='r',markersize=16)
    plt.plot([1],np.array(blue),marker='o',color='g',markersize=16)
    plt.plot([0],np.array(bgred),marker='o',color='DarkRed',markersize=16)
    plt.plot([1],np.array(bgblue),marker='o',color='#003100',markersize=16)
    plt.plot([2],np.array(red)/np.array(blue),marker='o',color='k',markersize=16)
    plt.plot([2],np.array(bgred)/np.array(bgblue),marker='x',color='k',markersize=16)
    plt.show()
    
    
    

