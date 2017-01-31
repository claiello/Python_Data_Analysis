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
Time_bin = 1000#in ns; 
nominal_time_on = 150.0 #time during which e-beam nominally on, in mus
totalpoints = 1500 #total number of time-resolved points

Pixel_size = [2.2,2.2,2.2,2.2,2.2] #nm
Ps = Pixel_size #pixel size in nm, the numbers above with round nm precision
No_experiments = [1,1,1,1,1]

description = 'Andrea large NaYF4:Er'    # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
kv = [10]

let = ['V0','V0pt25' ,'V0pt5', 'V0pt75','V1']

nametr = ['','','','','','','','']
######################################## Plot with dose for different apertures
##files below exist 

#blue_int_array = np.zeros(len(nametr))
#red_int_array = np.zeros(len(nametr))
#blue_std_array = np.zeros(len(nametr))
#red_std_array = np.zeros(len(nametr))


pisize =Pixel_size

listofindex =np.arange(0,5)#,11]

consider_whole_light = [0,1,2,3,4]
#4,5,7 segmentation ok-ish
index = 4
if index is 5:
#for index in listofindex:
    
    print(index)
    
    Ps = str("{0:.2f}".format(Pixel_size[index])) 

    #se = np.load(str(let[index]) +'SEchannel.npz',mmap_mode='r') 
    #segmm = np.load(str(let[index]) +'SEchannelGMM.npz',mmap_mode='r') 
    red = np.load(str(let[index]) +'Redbright.npz',mmap_mode='r') 
    blue = np.load(str(let[index]) +'Bluebright.npz',mmap_mode='r') 
    
#    fsizetit = 18 
#    fsizepl = 16 
#    sizex = 8 
#    sizey = 6
#    dpi_no = 80
#    lw = 2
#    
#    length_scalebar = 50.0 #in nm 
#    scalebar_legend = '50nm'
#    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[index]))
#    
#    titulo = description + ' ' +  let[index] +  ' (10kV, 30$\mu$m aperture, 1$\mu$s time bins, ' + str(Ps)+ 'nm pixels, blue/red: $</>$ 593nm)'
#    
#    fig40= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
#    fig40.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
#    plt.rc('text.latex', preamble=r'\usepackage{xcolor}') #not working
#    plt.rc('text', usetex=True)
#    plt.rc('text.latex', preamble=r'\usepackage{xcolor}')  #not working
#    plt.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]  #not working
#    plt.rc('font', family='serif')
#    plt.rc('font', serif='Palatino')         
#    
#    plt.suptitle("Segmentation (model: 2-GMM) of cathodoluminescence signal using SE channel, \n" + titulo,fontsize=fsizetit)
#
#    gc.collect()
#    
#    ax1 = plt.subplot2grid((2,4), (0, 0), colspan=1)
#    ax1.set_title('SE channel',fontsize=fsizepl) #as per accompanying txt files
#    plt.imshow(se['data'],cmap=cm.Greys_r)
#    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#    ax1.add_artist(sbar)
#    plt.axis('off')
#    
#    gc.collect()
#    ax1 = plt.subplot2grid((2,4), (0, 1), colspan=1)
#    ax1.set_title('SE channel, signal pixels',fontsize=fsizepl)
#    hlpse = np.copy(segmm['bright'])
#    hlpse[~np.isnan(hlpse)] = 0.0 #inside
#    if index in consider_whole_light:
#        hlpse[np.isnan(hlpse)] = 0.0 #outside, consider all light
#    else:    
#        hlpse[np.isnan(hlpse)] = 1.0 #outside
#    im = plt.imshow(hlpse,cmap=cm.Greys) #or 'OrRd'
#    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#    ax1.add_artist(sbar)
#    plt.axis('off')
#    
#    del hlpse, se
    gc.collect()
    
#    ax1 = plt.subplot2grid((2,4), (0, 2), colspan=1)
#    ax1.set_title('CL data averaged over time',fontsize=fsizepl)
#    im = plt.imshow(np.nanmean(blue['data'],axis = (0,1)),cmap=cm.Blues) #or 'OrRd'
#    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
#    ax1.add_artist(sbar)
#    plt.axis('off')
#    unit = '(kHz)'
#    box = ax1.get_position()
#    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
#    axColor = plt.axes([box.x0, box.y0*1.1 , box.width,0.01 ])    
#    cb2 = plt.colorbar(im, cax = axColor, orientation="horizontal",label='Photon counts ' + unit) 
#    
#    ax1 = plt.subplot2grid((2,4), (0, 3), colspan=1)
#    ax1.set_title('CL data averaged over time',fontsize=fsizepl)
#    imb = plt.imshow(np.nanmean(red['data'],axis = (0,1)),cmap=cm.Reds) #or 'OrRd'
#    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
#    ax1.add_artist(sbar)
#    plt.axis('off')
#    
#    box = ax1.get_position()
#    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
#    axColor = plt.axes([box.x0, box.y0*1.1 , box.width,0.01 ])    
#    cb1 = plt.colorbar(imb, cax = axColor, orientation="horizontal",label='Photon counts ' + unit) 
#    
#    ax1 = plt.subplot2grid((2,4), (1, 0), colspan=1)
#    ax1.spines['right'].set_visible(False)
#    ax1.spines['top'].set_visible(False)
#    ax1.xaxis.set_ticks_position('bottom')
#    ax1.yaxis.set_ticks_position('left')
       
    # INSIDE
    #hlp = np.copy(segmm['bright'])
    
    if index in consider_whole_light:
        hlp = 1.0 #outside, consider all light
    else:
        hlp[~np.isnan(hlp)] = 1.0  #inside
    
    # OUTSIDE
    #hlpd = np.copy(segmm['bright'])
    if index in consider_whole_light:
        hlpd  = 0.0 #consider all light
    else:
        hlpd[~np.isnan(hlpd)] = 0.0 
        hlpd[np.isnan(hlpd)] = 1.0
   
#    plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1e3,np.nanmean(red['data']*hlp,axis = (0,2,3)),c='r',lw=2) #in mus, in MHz
#    plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1e3,np.nanmean(red['data']*hlpd,axis = (0,2,3)),c='DarkRed',lw=2) #in mus, in MHz
#    plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1e3,np.nanmean(blue['data']*hlp,axis = (0,2,3)),c='b',lw=2) #in mus, in MHz
#    plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1e3,np.nanmean(blue['data']*hlpd,axis = (0,2,3)),c='DarkBlue',lw=2) #in mus, in MHz
#    ax1.axvspan(50.0,50.0+150.0, alpha=0.25, color='yellow')
#    unit = 'kHz'
#    plt.ylabel("Average luminescence \n of each time bin, per signal pixel (" + unit + ")",fontsize=fsizepl)
#    plt.xlabel("Behaviour of e-beam during each experiment: \n 150-ON + 1500-OFF ($\mu$s)",fontsize=fsizepl)
#    major_ticks0 = [250,500,750,1000,1250,1500]
#    ax1.set_xticks(major_ticks0) 
#    #ax1.set_yticks([15,30,45]) 
#    plt.ylim([0,15])
#    
#    ax1 = plt.subplot2grid((2,3), (1, 1), colspan=1)
#    ax1.spines['right'].set_visible(False)
#    ax1.spines['top'].set_visible(False)
#    ax1.xaxis.set_ticks_position('bottom')
#    ax1.yaxis.set_ticks_position('left')
    
    #datared = np.average(red['data'], axis = (0))
    #datablue = np.average(blue['data'], axis = (0))
    
    if True is False:
        pass
    else:
        initbin = (150+50+3)-1 #init bin for decay
        backgdinit = 50
        ### 700ns /40ns = 7. ....
#        datared_init = datared[0:backgdinit,:,:]
#        datared = datared[initbin:,:,:]
#        datablue_init = datablue[0:backgdinit,:,:]
#        datablue = datablue[initbin:,:,:]

#    fastfactor = 1
#    
#    #hlp_red = datared * hlp
#    #hlp_blue = datablue * hlp
#   
#    #hlpred = np.nanmean(red['data'][:,initbin:,:,:]*hlp,axis = (2,3))
#    #hlpblue = np.nanmean(blue['data'][:,initbin:,:,:]*hlp,axis = (2,3))
#    
#    #error_arrayr = np.nanstd(hlpred,axis = 0)
#    #error_arrayr[error_arrayr < 0.05 * np.max(hlpred)]  = 0.05 * np.max(hlpred) #arbitrary 5% of max  
#    
#    cut_longa = 6
#    cut_shorta = 3
#    
#    print('bef init')
#    gc.collect()
#    
#    init_guess = calc_double_fit(np.arange(0,datared.shape[0])*Time_bin*1e-9*fastfactor,np.nanmean(datared*hlp,axis=(1,2)),dt= Time_bin*1e-9*fastfactor,cut_long=cut_longa,cut_short=cut_shorta)
#    
#    gc.collect()    
#    
#    init_guessb = calc_double_fit(np.arange(0,datablue.shape[0])*Time_bin*1e-9*fastfactor,np.nanmean(datablue*hlp,axis=(1,2)),dt= Time_bin*1e-9*fastfactor,cut_long=cut_longa,cut_short=cut_shorta)
#    
#    cinit = np.nanmean(datared_init*hlp,axis=(0,1,2))
#    cinitb = np.nanmean(datablue_init*hlp,axis=(0,1,2))    
#    
#    #replace c with cinit
#    init_guess[2] = cinit 
#    init_guessb[2] = cinitb
#    if index == 4:
#        print('here blue')
#        init_guessb[1] = 5.0
#    
#    print('bef fit')
#    gc.collect()
    
#    b,e,be,ee,b2,e2,be2,ee2 = calcdecay_subplot_nan(datared*hlp, time_detail= Time_bin*1e-9*fastfactor,titulo='',single=False,other_dset1=None,other_dset2=datablue*hlp,init_guess=init_guess,init_guess2=init_guessb,unit='kHz',error_array=None) #,error_array=error_arrayr, error_array2=error_arrayb)    
#        
#    b_array_red[index] = b
#    e_array_red[index] = e
#    be_array_red[index] = be    
#    ee_array_red[index] = ee  
#    b_array_blue[index] = b2
#    e_array_blue[index] = e2
##    be_array_blue[index] = be2    
##    ee_array_blue[index] = ee2  
# 
#    plt.ylabel("Average luminescence \n of each time bin, per signal pixel (" + unit + ")",fontsize=fsizepl)
#    plt.xlabel(r'Time after blanking the electron beam ($\mu$s)',  fontsize=fsizepl)
#    plt.xlim(xmax=1500.0) #2000
#    
#    # Extra plots        
#    aaa = datared*hlp
#    xx_array = np.arange(0,aaa.shape[0])*Time_bin*1e-9*fastfactor
#    #Plot whole of background decay
#    plt.semilogy(xx_array/1e-6,np.nanmean(datared*hlpd,axis=(1,2)),'o',color='DarkRed',markersize=3,label='Transient CL from red background')   
#    #Plot mean signal before e-beam on
#    plt.semilogy(xx_array/1e-6,np.nanmean(datared_init*hlp,axis=(0,1,2))*np.ones(len(xx_array)),'r--',lw=2,label='Mean CL from signal pixels, before e-beam')
#    #Plot mean background
#    plt.semilogy(xx_array/1e-6,np.nanmean(datared_init*hlpd,axis=(0,1,2))*np.ones(len(xx_array)),'--',color='DarkRed',lw=1,label='Mean CL from background, before e-beam')
#    
#    #Plot whole of background decay
#    plt.semilogy(xx_array/1e-6,np.nanmean(datablue*hlpd,axis=(1,2)),'o',color='DarkBlue',markersize=3,label='Transient CL from blue background')   
#    #Plot mean signal before e-beam on
#    plt.semilogy(xx_array/1e-6,np.nanmean(datablue_init*hlp,axis=(0,1,2))*np.ones(len(xx_array)),'b--',lw=2,label='Mean CL from blue signal pixels, before e-beam')
#    #Plot mean background
#    plt.semilogy(xx_array/1e-6,np.nanmean(datablue_init*hlpd,axis=(0,1,2))*np.ones(len(xx_array)),'--',color='DarkBlue',lw=1,label='Mean CL from background, before e-beam')
#    
#    
#    #plt.legend(loc='best')
#        
#    #ax1.set_xticks(major_ticks0)
#    
#    ax1 = plt.subplot2grid((2,3), (1, 2), colspan=1)
#    ax1.spines['left'].set_visible(False)
#    ax1.spines['top'].set_visible(False)
#    ax1.xaxis.set_ticks_position('bottom')
#    ax1.yaxis.set_ticks_position('right')
#    ax1.yaxis.set_label_position("right")
#    
#    del datared, datablue, datablue_init, datared_init
#    gc.collect()
    dataALLred = red['data'][:,:,:,:]
    dataALLblue = blue['data'][:,:,:,:]
#    nominal_time_on = 150.0
    
    del red, blue
    gc.collect()
#    
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)),c='r',linestyle='None', marker='o',markersize=4) #in mus, in MHz
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='r',  linewidth= lw) #in mus, in MHz
#    #background intensity
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3)),c='DarkRed', label='',linestyle='None', marker='o',markersize=3) #in mus, in MHz
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='DarkRed',  linewidth= lw) #in mus, in MHz
#    #init red intensity
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred[:,0:backgdinit,:,:]*hlp,axis=(1,2,3)),c='r', linestyle='None', marker='o',markersize=3) #in mus, in MHz
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred[:,0:backgdinit,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),'--',c='r',  linewidth= lw) #in mus, in MHz
#
#
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)),c='b', linestyle='None', marker='o',markersize=4) #in mus, in MHz
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='b',  linewidth= lw) #in mus, in MHz
#    #background intensity
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3)),c='DarkBlue', label='',linestyle='None', marker='o',markersize=3) #in mus, in MHz
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='DarkBlue',  linewidth= lw) #in mus, in MHz
#    #init red intensity
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,0:backgdinit,:,:]*hlp,axis=(1,2,3)),c='b', linestyle='None', marker='o',markersize=3) #in mus, in MHz
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,0:backgdinit,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),'--',c='b',  linewidth= lw) #in mus, in MHz
#
#    aa = np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3))/np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3))
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3))/np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)),c='k', label='blue/red = ' + str("{0:.2f}".format(aa[0])),linestyle='None', marker='s',markersize=6) #in mus, in MHz
#    plt.legend(loc='best')
    
    red_int_array[index] = np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)) #np.nanmean(dataALLred*hlp,axis=(0,1,2,3))
    #cumu_red[index,:] =  np.nanmean(dataALLred*hlp,axis=(1,2,3))
    blue_int_array[index] = np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)) #np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))
    #cumu_blue[index,:] =  np.nanmean(dataALLred*hlp,axis=(1,2,3))
    
    red_std_array[index] = np.nanstd(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)) #np.nanmean(dataALLred*hlp,axis=(0,1,2,3))
    #cumu_red[index,:] =  np.nanmean(dataALLred*hlp,axis=(1,2,3))
    blue_std_array[index] = np.nanstd(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)) #np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))
    #cumu_blue[index,:] =  np.nanmean(dataALLred*hlp,axis=(1,2,3))
    
#    plt.ylabel("Average luminescence \n for each experiment, per signal pixel while excited (kHz)",fontsize=fsizepl)
#    plt.xlabel("Cumulative e-beam exposure time \n per pixel (nominal, $\mu$s)",fontsize=fsizepl)
#    
#    plt.xlim([nominal_time_on - 1.0,nominal_time_on*No_experiments[index]*fastfactor +1.0])
#    
#    #sig_to_back[index] =   np.nanmean(dataALLred*hlp,axis=(0,1,2,3))/np.nanmean(dataALLred*hlpd,axis=(0,1,2,3))
#    #sig_to_initred[index] =   np.nanmean(dataALLred*hlp,axis=(0,1,2,3))/np.nanmean(dataALLred[:,0:16,:,:]*hlp,axis=(0,1,2,3))
#    
#    
#    #sig_to_back_blue[index] =   np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))/np.nanmean(dataALLblue*hlpd,axis=(0,1,2,3))
#    #sig_to_initblue[index] =   np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))/np.nanmean(dataALLblue[:,0:16,:,:]*hlp,axis=(0,1,2,3))
#    
#    multipage_longer('ZZZZSingle-'+ let[index] + '.pdf',dpi=80)

##### ONCE ALL FITS WORK, 
###### NEED TO RUN THESE LINES BELOW SO THAT ALL NPZ FILES ARE CREATED 
#mycode = 'Red_int_array = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Red_int_array', data = red_int_array)
#
#mycode = 'Blue_int_array = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Blue_int_array', data = blue_int_array)
#
#mycode = 'Red_std_array = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Red_std_array', data = red_std_array)
#
#mycode = 'Blue_std_array = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Blue_std_array', data = blue_std_array)
#
#il_data = [ 41.79260051,  29.8911322,   36.69970937,  48.82985395, 69.84299812]
#mycode = 'Il_data = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Il_data', data = il_data)
#
#b_array_red = [50.861, 38.772, 27.493, 30.914, 42.129]
#mycode = 'B_array_red = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('B_array_red', data = b_array_red)
#
#be_array_red = [8.813, 6.071, 4.560, 3.537, 9.797]
#mycode ='Be_array_red = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Be_array_red', data = be_array_red)
#
#e_array_red = [232.240, 260.251, 213.217, 192.565, 164.092]
#mycode = 'E_array_red = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('E_array_red', data = e_array_red)
#
#ee_array_red = [17.787, 22.585, 13.830, 14.539, 7.922]
#mycode = 'Ee_array_red = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Ee_array_red', data = ee_array_red)
#
#b_array_blue = [34.478, 32.995, 26.309, 21.116, 42.616]
#mycode = 'B_array_blue = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('B_array_blue', data = b_array_blue)
#
#be_array_blue = [3.954, 3.458, 2.751, 2.761, 4.138]
#mycode ='Be_array_blue = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Be_array_blue', data = be_array_blue)
#
#e_array_blue = [178.036,206.513, 194.171, 152.834, 152.166]
#mycode = 'E_array_blue = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('E_array_blue', data = e_array_blue)
#
#ee_array_blue = [8.243, 13.458, 11.513, 7.545, 7.311]
#mycode = 'Ee_array_blue = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Ee_array_blue', data = ee_array_blue)

#####FIG WITH SUMMARY   
##### WHEN FILES ABOVE ARE CREATED, CREATE FIGURE BELOW WITH REGION SIZES VS LIFETIMES AND REGION SIZES VS S/N OR BRIGHTNESS OR <SIGNAL INT>/<BACKGROUND INT>
fastfactor = 1

fsizepl = 24
fsizenb = 20

Red_int_array = np.load('Red_int_array.npz') 
red_int_array = Red_int_array['data']
Blue_int_array = np.load('Blue_int_array.npz') 
blue_int_array = Blue_int_array['data']

Red_std_array = np.load('Red_std_array.npz') 
red_std_array = Red_int_array['data']
Blue_std_array = np.load('Blue_std_array.npz') 
blue_std_array = Blue_int_array['data']

B_array_red= np.load('B_array_red.npz')
b_array_red = B_array_red['data']  
Be_array_red = np.load('Be_array_red.npz')
be_array_red = Be_array_red['data'] 
E_array_red = np.load('E_array_red.npz')
e_array_red = E_array_red['data']   
Ee_array_red = np.load('Ee_array_red.npz')
ee_array_red = Ee_array_red['data']   

B_array_blue= np.load('B_array_blue.npz')
b_array_blue = B_array_blue['data']  
Be_array_blue = np.load('Be_array_blue.npz')
be_array_blue = Be_array_blue['data'] 
E_array_blue = np.load('E_array_blue.npz')
e_array_blue = E_array_blue['data']   
Ee_array_blue = np.load('Ee_array_blue.npz')
ee_array_blue = Ee_array_blue['data'] 

Il_data = np.load('Il_data.npz')
il_data = Il_data['data']   


####
length_scalebar = 100.0 #in nm 
scalebar_legend = '100nm'
length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[index]))
###let = ['V0','V0pt25' ,'V0pt5', 'V0pt75','V1']
fig42= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig42.set_size_inches(1200./fig42.dpi,900./fig42.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    
#fig42.suptitle('Lifetimes fitted after 1300$\mu$s decay, small NaYF$_4$ CS from Andrea \n (10kV, 30$\mu$m aperture, 40ns time bins)',fontsize=fsizetit)     
ax0 = plt.subplot2grid((3,5), (0,0), colspan=1, rowspan=1)
se = np.load('V0pt25' +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(ax0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
ax0.add_artist(sbar)

ax00 = plt.subplot2grid((3,5), (0,1), colspan=1, rowspan=1)
se = np.load('V0pt5' +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(ax00.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
ax00.add_artist(sbar)

ax000 = plt.subplot2grid((3,5), (0,2), colspan=1, rowspan=1)
se = np.load('V0' +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(ax000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
ax000.add_artist(sbar)

ax0000 = plt.subplot2grid((3,5), (0,3), colspan=1, rowspan=1)
se = np.load('V0pt75' +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(ax0000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
ax0000.add_artist(sbar)

ax00000 = plt.subplot2grid((3,5), (0,4), colspan=1, rowspan=1)
se = np.load('V1' +'SEchannel.npz',mmap_mode='r') 
plt.imshow(se['data'],cmap=cm.Greys_r)
plt.axis('off')
sbar = sb.AnchoredScaleBar(ax00000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
ax00000.add_artist(sbar)

ax2 = plt.subplot2grid((3,5), (1,0), colspan=5, rowspan=1)
ax1 = plt.subplot2grid((3,5), (2,0), colspan=5, sharex=ax2)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

x_vec = il_data

ax2.errorbar(x_vec, b_array_red, yerr=be_array_red, fmt='ro',markersize=5)
ax2.errorbar(x_vec, e_array_red, yerr=ee_array_red, fmt='ro', markersize=10)
ax2.errorbar(x_vec, b_array_blue, yerr=be_array_blue, fmt='go',markersize=5)
ax2.errorbar(x_vec, e_array_blue, yerr=ee_array_blue, fmt='go', markersize=10)
#ax1.set_ylabel('Shorter time constant ($\mu$s)',fontsize=fsizepl)
ax2.set_ylabel('Time constants ($\mu$s)',fontsize=fsizepl)
#ax2.set_ylim([0,1.5])
#plt.xlim([0,90])
#ax2.legend(fontsize=fsizepl)
ax2.tick_params(labelsize=fsizenb)
ax1.set_xlabel('Temperature ($^{\circ}$C)',fontsize=fsizepl)

plt.sca(ax1) 

plt.setp(ax2.get_xticklabels(), visible=False)
plt.xlim([0,100])
ax2.set_ylim([10,290])
###



### error
import uncertainties as unc  
import uncertainties.unumpy as unumpy  
#import nemmen 
u_red_int_array=unumpy.uarray(( red_int_array, red_std_array ))  
u_blue_int_array=unumpy.uarray(( blue_int_array, blue_std_array ))  

#calc intensity ratios
ratio = (u_red_int_array[0:5]/u_blue_int_array[0:5])/(u_red_int_array[1]/u_blue_int_array[1])
unumpy_error_ratio = unumpy.std_devs(ratio) 
ratio2 = (red_int_array[0:5]/blue_int_array[0:5])/(red_int_array[1]/blue_int_array[1])

#ax3 = ax2.twinx()
ax1.set_ylim([0.45, 1.05])
#ax1.set_yticks([0.5,0.75,1.0])
#ax1.errorbar(x_vec, ratio2, yerr=unumpy_error_ratio, fmt='ks',markersize=7.5)
ax1.errorbar(x_vec, ratio2, fmt='ks',markersize=7.5)
ax1.set_ylabel('Red to green \n intensity ratio',fontsize=fsizepl)
plt.tick_params(labelsize=fsizenb)
plt.xlim([25,75])
plt.xticks([30,40,50,60,70])

multipage_longer('ZZZZZZZZSummary.pdf',dpi=80)
