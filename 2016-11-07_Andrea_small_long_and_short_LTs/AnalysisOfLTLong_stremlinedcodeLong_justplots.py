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
#from TConversionThermocoupler import *
import matplotlib.cm as cm
import scipy.ndimage as ndimage
#from matplotlib_scalebar.scalebar import ScaleBar #### Has issue with plotting using latex font. only import when needed, then unimport
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
from FluoDecay import *
from PlottingFcts import *

from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.misc
import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile
#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################
Time_bin = 2000#in ns; 1/clock of 25MHz 
nominal_time_on = 198.0 #time during which e-beam nominally on, in mus
totalpoints = 500 #total number of time-resolved points

Pixel_size = [1.4, 0.8] #nm
Ps = Pixel_size #pixel size in nm, the numbers above with round nm precision
No_experiments = [1,1]

description = 'Andrea small NaYF4:Er'     # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
kv = [10]

let = ['I','II']

nametr = ['','']
######################################## Plot with dose for different apertures

pisize =Pixel_size

listofindex =np.arange(0,2)#,11]

consider_whole_light = []
#4,5,7 segmentation ok-ish
#index = 6
#if index is 6:
for index in [0]: #listofindex:
    
    Ps = str("{0:.2f}".format(Pixel_size[index])) 

    se = np.load(str(let[index]) +'SEchannel.npz') 
    segmm = np.load(str(let[index]) +'SEchannelGMM.npz') 
    red = np.load(str(let[index]) +'Redbright.npz') 
    blue = np.load(str(let[index]) +'Bluebright.npz') 
    
    fsizetit = 18 
    fsizepl = 16 
    sizex = 8 
    sizey = 6
    dpi_no = 80
    lw = 2
    
    length_scalebar = 50.0 #in nm 
    scalebar_legend = '50nm'
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[index]))
    
    
    fig40= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig40.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}') #not working
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}')  #not working
    plt.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]  #not working
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')         

    gc.collect()
    
    ax1 = plt.subplot2grid((2,4), (0, 0), colspan=1)
    ax1.set_title('SE channel',fontsize=fsizepl) #as per accompanying txt files
    plt.imshow(se['data'],cmap=cm.Greys_r)
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    gc.collect()
    ax1 = plt.subplot2grid((2,4), (0, 1), colspan=1)
    ax1.set_title('SE channel, signal pixels',fontsize=fsizepl)
    hlpse = np.copy(segmm['bright'])
    hlpse[~np.isnan(hlpse)] = 0.0 #inside
    if index in consider_whole_light:
        hlpse[np.isnan(hlpse)] = 0.0 #outside, consider all light
    else:    
        hlpse[np.isnan(hlpse)] = 1.0 #outside
    im = plt.imshow(hlpse,cmap=cm.Greys) #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    del hlpse, se
    gc.collect()
    
    ax1 = plt.subplot2grid((2,4), (1, 0), colspan=1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
       
    # INSIDE
    hlp = np.copy(segmm['bright'])
    
    if index in consider_whole_light:
        hlp = 1.0 #outside, consider all light
    else:
        hlp[~np.isnan(hlp)] = 1.0  #inside
    
    # OUTSIDE
    hlpd = np.copy(segmm['bright'])
    if index in consider_whole_light:
        hlpd  = 0.0 #consider all light
    else:
        hlpd[~np.isnan(hlpd)] = 0.0 
        hlpd[np.isnan(hlpd)] = 1.0
   
    plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1e3,np.nanmean(red['data']*hlp,axis = (0,2,3)),c='r',lw=2) #in mus, in MHz
    plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1e3,np.nanmean(red['data']*hlpd,axis = (0,2,3)),c='DarkRed',lw=2) #in mus, in MHz
    plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1e3,np.nanmean(blue['data']*hlp,axis = (0,2,3)),c='g',lw=2) #in mus, in MHz
    plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1e3,np.nanmean(blue['data']*hlpd,axis = (0,2,3)),c='DarkGreen',lw=2) #in mus, in MHz
    ax1.axvspan(2.0,200.0, alpha=0.25, color='yellow')
    unit = 'kHz'
    plt.ylabel("Average cathodoluminescence per pixel (" + unit + ")",fontsize=fsizepl)
    plt.xlabel("Experiment time ($\mu$s)",fontsize=fsizepl)
    ax1.set_xticks([200,400]) 
    ax1.set_yticks([5,10]) 
    plt.xlim([7,410])
    plt.ylim([0,11])
    
    ax1 = plt.subplot2grid((2,3), (1, 1), colspan=1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    datared = np.average(red['data'], axis = (0))
    datablue = np.average(blue['data'], axis = (0))
    
    if True is False:
        pass
    else:
        initbin = (102)-1 #init bin for decay
        backgdinit = 1
        ### 700ns /40ns = 7. ....
        datared_init = datared[0:backgdinit,:,:]
        datared = datared[initbin:,:,:]
        datablue_init = datablue[0:backgdinit,:,:]
        datablue = datablue[initbin:,:,:]

    fastfactor = 1
    
    cinit = np.nanmean(datared_init*hlp,axis=(0,1,2))
    cinitb = np.nanmean(datablue_init*hlp,axis=(0,1,2))
   
    aaa = datared*hlp
    xx_array = np.arange(0,aaa.shape[0])*Time_bin*1e-9*fastfactor
    ax1.semilogy(xx_array/1e-6,np.nanmean(datared*hlp,axis=(1,2)),'o',color='r',markersize=3)  
    ax1.semilogy(xx_array/1e-6,np.nanmean(datablue*hlp,axis=(1,2)),'o',color='g',markersize=3)  
    ax1.set_ylabel("Average cathodoluminescence per pixel (" + unit + ")",fontsize=fsizepl)
    ax1.set_xlabel('Transient cathodoluminescence \n acquisition time ($\mu$s)',  fontsize=fsizepl)
    ax1.set_xlim(xmax=1000.0) #2000
    
    
    #Plot whole of background decay
    ax1.semilogy(xx_array/1e-6,np.nanmean(datared*hlp,axis=(1,2)),'o',color='r',markersize=5)   
    ax1.semilogy(xx_array/1e-6,np.nanmean(datablue*hlp,axis=(1,2)),'o',color='g',markersize=5) 
    
    ax1.semilogy(xx_array/1e-6,np.nanmean(datared*hlpd,axis=(1,2)),'o',color='DarkRed',markersize=3)   
    ax1.semilogy(xx_array/1e-6,np.nanmean(datablue*hlpd,axis=(1,2)),'o',color='DarkGreen',markersize=3) 
    
    
    my_edgecolor='#ff3232'
    my_facecolor='#ff6666'
    ax1.fill_between(xx_array/1e-6,np.nanmean(datared*hlp,axis=(1,2))+np.sqrt(np.nanmean(datared*hlp,axis=(1,2))),np.nanmean(datared*hlp,axis=(1,2))-np.sqrt(np.nanmean(datared*hlp,axis=(1,2))),alpha=0.5, edgecolor=my_edgecolor, facecolor= my_facecolor)   
    
    #Plot mean signal before e-beam on
    ax1.semilogy(xx_array/1e-6,np.nanmean(datared_init*hlp,axis=(0,1,2))*np.ones(len(xx_array)),'r--',lw=2)
    #Plot mean background
    ax1.semilogy(xx_array/1e-6,np.nanmean(datared_init*hlpd,axis=(0,1,2))*np.ones(len(xx_array)),'--',color='DarkRed',lw=2)
    
    #Plot mean signal before e-beam on
    ax1.semilogy(xx_array/1e-6,np.nanmean(datablue_init*hlp,axis=(0,1,2))*np.ones(len(xx_array)),'g--',lw=2)
    #Plot mean background
    ax1.semilogy(xx_array/1e-6,np.nanmean(datablue_init*hlpd,axis=(0,1,2))*np.ones(len(xx_array)),'--',color='DarkGreen',lw=2)
    
 
    multipage_longer('ZZZZSingle-'+ let[index] + '.pdf',dpi=80)

klklklkl
#my_edgecolor='#74C365', my_facecolor='#74C365'