#nominally
#1mus moving
#2mus on
#7mus off
#in practise,
#moving until 0.5mus
#excitation between 0.5 and 2.5mus
#decay rest

#### THIS FILE CONSIDERS ONLY THE SE IMAGE TAKEN PRIOR TO TR

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
from BackgroundCorrection import *
from TConversionThermocoupler import *
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
##files below exist 
#b_array_red = np.zeros(len(nametr))
#be_array_red = np.zeros(len(nametr))
#e_array_red = np.zeros(len(nametr))
#ee_array_red = np.zeros(len(nametr))
#b_array_blue = np.zeros(len(nametr))
#be_array_blue = np.zeros(len(nametr))
#e_array_blue = np.zeros(len(nametr))
#ee_array_blue = np.zeros(len(nametr))
#blue_int_array = np.zeros(len(nametr))
#red_int_array = np.zeros(len(nametr))
#cumu_blue = np.zeros([len(nametr),50])
#cumu_red = np.zeros([len(nametr),50])
#il_data = np.zeros([len(nametr),50])
#size_signal = np.zeros(len(nametr))
#sig_to_back = np.zeros(len(nametr))
#sig_to_initred = np.zeros(len(nametr))
#sig_to_back_blue = np.zeros(len(nametr))
#sig_to_initblue = np.zeros(len(nametr))

pisize =Pixel_size

listofindex =np.arange(0,2)#,11]

consider_whole_light = []
#4,5,7 segmentation ok-ish
index =0
if index is 0:
#for index in listofindex:
    
    Ps = str("{0:.2f}".format(Pixel_size[index])) 

    se = np.load(str(let[index]) +'SEchannel.npz') 
    segmm = np.load(str(let[index]) +'SEchannelGMM.npz') 
    red = np.load(str(let[index]) +'Redbright.npz') 
    blue = np.load(str(let[index]) +'Bluebright.npz') 
    
    fsizetit = 18 
    fsizepl = 24
    sizex = 8 
    sizey = 6
    dpi_no = 80
    lw = 2
    
    length_scalebar = 50.0 #in nm 
    scalebar_legend = '50nm'
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[index]))
    
    titulo = description + ' (10kV, 30$\mu$m aperture, 2$\mu$s time bins, ' + str(Ps)+ 'nm pixels, blue/red: $</>$ 593nm)'
    
    fig40= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig40.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}') #not working
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}')  #not working
    plt.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]  #not working
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')         
    
    #plt.suptitle("Segmentation (model: 2-GMM) of cathodoluminescence signal using SE channel, \n" + titulo,fontsize=fsizetit)

    gc.collect()
    
    ax1 = plt.subplot2grid((2,6), (0, 0), colspan=2)
  
    ax1.set_title('Electron topography, \n segmented',fontsize=fsizepl)
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
    
    ax1 = plt.subplot2grid((2,6), (0, 2), colspan=2)
    ax1.set_title('Cathodoluminescence, \n green band',fontsize=fsizepl)
    im = plt.imshow(np.nanmean(blue['data'],axis = (0,1)),cmap=cm.Greens) #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    unit = '(kHz)'
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor = plt.axes([box.x0, box.y0*1 , box.width,0.01 ])    
    cb2 = plt.colorbar(im, cax = axColor, orientation="horizontal",label='Photon counts ' + unit,ticks=[5.0,10.0]) 
    cb2.ax.tick_params(labelsize=fsizepl)
    cb2.set_label('Photon counts (kHz)', size = fsizepl)
    
    ax1 = plt.subplot2grid((2,6), (0, 4), colspan=2)
    ax1.set_title('Cathodoluminescence, \n red band',fontsize=fsizepl)
    imb = plt.imshow(np.nanmean(red['data'],axis = (0,1)),cmap=cm.Reds) #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor = plt.axes([box.x0, box.y0*1 , box.width,0.01 ])    
    cb1 = plt.colorbar(imb, cax = axColor, orientation="horizontal",label='Photon counts ' + unit,ticks=[4.0,8.0]) 
    cb1.ax.tick_params(labelsize=fsizepl)
    cb1.set_label('Photon counts (kHz)', size = fsizepl)
    
    ax1 = plt.subplot2grid((2,6), (1, 0), colspan=3)
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
    #plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1e3,np.nanmean(red['data']*hlpd,axis = (0,2,3)),c='DarkRed',lw=2) #in mus, in MHz
    plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1e3,np.nanmean(blue['data']*hlp,axis = (0,2,3)),c='g',lw=2) #in mus, in MHz
    #plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1e3,np.nanmean(blue['data']*hlpd,axis = (0,2,3)),c='DarkBlue',lw=2) #in mus, in MHz
    ax1.axvspan(2.0,200.0, alpha=0.25, color='yellow')
    unit = 'kHz'
    plt.ylabel("Average luminescence, \n per signal pixel (" + unit + ")",fontsize=fsizepl)
    plt.xlabel("Time as electron beam on/off ($\mu$s)",fontsize=fsizepl)
    major_ticks0 = [250,500,750,1000]
    ax1.set_xticks(major_ticks0) 
    #ax1.set_yticks([15,30,45]) 
    plt.ylim([0,10])
    ax1.tick_params(labelsize=fsizepl)
    
    ax1 = plt.subplot2grid((2,6), (1, 3), colspan=3)
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
   
    #hlp_red = datared * hlp
    #hlp_blue = datablue * hlp
   
    #hlpred = np.nanmean(red['data'][:,initbin:,:,:]*hlp,axis = (2,3))
    #hlpblue = np.nanmean(blue['data'][:,initbin:,:,:]*hlp,axis = (2,3))
    
    #error_arrayr = np.nanstd(hlpred,axis = 0)
    #error_arrayr[error_arrayr < 0.05 * np.max(hlpred)]  = 0.05 * np.max(hlpred) #arbitrary 5% of max  
    
    cut_longa = 6
    cut_shorta = 3
    
    print('bef init')
    
    init_guess = calc_double_fit(np.arange(0,datared.shape[0])*Time_bin*1e-9*fastfactor,np.nanmean(datared*hlp,axis=(1,2)),dt= Time_bin*1e-9*fastfactor,cut_long=cut_longa,cut_short=cut_shorta)
    init_guessb = calc_double_fit(np.arange(0,datablue.shape[0])*Time_bin*1e-9*fastfactor,np.nanmean(datablue*hlp,axis=(1,2)),dt= Time_bin*1e-9*fastfactor,cut_long=cut_longa,cut_short=cut_shorta)
    
    #replace c with cinit
    init_guess[2] = cinit 
    init_guessb[2] = cinitb
    
    print('bef fit')
    
    b,e,be,ee,b2,e2,be2,ee2 = calcdecay_subplot_nan(datared*hlp, time_detail= Time_bin*1e-9*fastfactor,titulo='',single=False,other_dset1=None,other_dset2=datablue*hlp,init_guess=init_guess,init_guess2=init_guessb,unit='kHz',error_array=None) #,error_array=error_arrayr, error_array2=error_arrayb)    
        
#    b_array_red[index] = b
#    e_array_red[index] = e
#    be_array_red[index] = be    
#    ee_array_red[index] = ee  
#    b_array_blue[index] = b2
#    e_array_blue[index] = e2
#    be_array_blue[index] = be2    
#    ee_array_blue[index] = ee2  
 
    #plt.ylabel("Average luminescence \n of each time bin, per signal pixel (" + unit + ")",fontsize=fsizepl)
    plt.xlabel(r'Time after blanking the electron beam ($\mu$s)',  fontsize=fsizepl)
    plt.xlim(xmax=1000.0) #2000
    major_ticks0 = [250,500,750,1000]
    ax1.set_xticks(major_ticks0) 
    ax1.tick_params(labelsize=fsizepl)
    
    # Extra plots        
    aaa = datared*hlp
    xx_array = np.arange(0,aaa.shape[0])*Time_bin*1e-9*fastfactor
    #Plot whole of background decay
    #plt.semilogy(xx_array/1e-6,np.nanmean(datared*hlpd,axis=(1,2)),'o',color='DarkRed',markersize=3,label='Transient CL from red background')   
    #Plot mean signal before e-beam on
    plt.semilogy(xx_array/1e-6,np.nanmean(datared_init*hlp,axis=(0,1,2))*np.ones(len(xx_array)),'r--',lw=2,label='Mean CL from signal pixels, before e-beam')
    #Plot mean background
    #plt.semilogy(xx_array/1e-6,np.nanmean(datared_init*hlpd,axis=(0,1,2))*np.ones(len(xx_array)),'--',color='DarkRed',lw=1,label='Mean CL from background, before e-beam')
    
    #Plot whole of background decay
    #plt.semilogy(xx_array/1e-6,np.nanmean(datablue*hlpd,axis=(1,2)),'o',color='DarkGreen',markersize=3,label='Transient CL from blue background')   
    #Plot mean signal before e-beam on
    plt.semilogy(xx_array/1e-6,np.nanmean(datablue_init*hlp,axis=(0,1,2))*np.ones(len(xx_array)),'g--',lw=2,label='Mean CL from blue signal pixels, before e-beam')
    #Plot mean background
    #plt.semilogy(xx_array/1e-6,np.nanmean(datablue_init*hlpd,axis=(0,1,2))*np.ones(len(xx_array)),'--',color='DarkBlue',lw=1,label='Mean CL from background, before e-beam')
    major_ticksy = [0.1,1.0,10.0]
    ax1.set_yticks(major_ticksy) 
    
    #plt.legend(loc='best')
        
    #ax1.set_xticks(major_ticks0)
    
    
    
    multipage_longer('ZZZZZZZGreenSingle-'+ let[index] + '.pdf',dpi=80)

