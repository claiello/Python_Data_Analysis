#nominally
#50mus moving
#150mus on
#1400mus off
#clock 1MHz = 1mus time bon
#250kX, 50%x 50% scale
#250x250 pixels
#5 frames per temperature (ie, per voltage )

##### Temperatures read by the multimeter just before acquisition of chosenexp = 0 were:
#27.8C
#30.5C
#44.4C
##50.5
#74.6C
#94.9C

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

nametr = ['2017-01-05-1557_ImageSequence__250.000kX_10.000kV_30mu_15',
          '2017-01-05-1634_ImageSequence__250.000kX_10.000kV_30mu_20',
          '2017-01-05-1709_ImageSequence__250.000kX_10.000kV_30mu_23',
          '2017-01-05-1745_ImageSequence__250.000kX_10.000kV_30mu_26',
          '2017-01-05-1831_ImageSequence__250.000kX_10.000kV_30mu_30',
          '2017-01-05-1906_ImageSequence__250.000kX_10.000kV_30mu_32']

Pixel_size = 0.89*np.ones(len(nametr)) #nm
Ps = Pixel_size #pixel size in nm, the numbers above with round nm precision
No_experiments = [1,1,1,1,1,1]

description = 'Andrea small NaYF4:Er'    # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
kv = [10]

let = ['V0','V0pt25' ,'V0pt5', 'V0pt5b','V0pt75','V1']


######################################## Plot with dose for different apertures
##files below exist 
b_array_red = np.zeros(len(nametr))
be_array_red = np.zeros(len(nametr))
e_array_red = np.zeros(len(nametr))
ee_array_red = np.zeros(len(nametr))
b_array_blue = np.zeros(len(nametr))
be_array_blue = np.zeros(len(nametr))
e_array_blue = np.zeros(len(nametr))
ee_array_blue = np.zeros(len(nametr))
blue_int_array = np.zeros(len(nametr))
red_int_array = np.zeros(len(nametr))
cumu_blue = np.zeros([len(nametr),No_experiments[0]])
cumu_red = np.zeros([len(nametr),No_experiments[0]])
il_data = np.zeros([len(nametr),No_experiments[0]])
il_data_std = np.zeros([len(nametr),No_experiments[0]])

#size_signal = np.zeros(len(nametr))
#sig_to_back = np.zeros(len(nametr))
#sig_to_initred = np.zeros(len(nametr))
#sig_to_back_blue = np.zeros(len(nametr))
#sig_to_initblue = np.zeros(len(nametr))
blue_std_array = np.zeros(len(nametr))
red_std_array = np.zeros(len(nametr))

pisize =Pixel_size

listofindex =np.arange(0,len(nametr))#,11]

chosenexp = 4 #0 1 2 3 4

consider_whole_light = [0,1,2,3,4,5]
#4,5,7 segmentation ok-ish
#index = 5
#if index is 5:
for index in listofindex:
    
    Ps = str("{0:.2f}".format(Pixel_size[index])) 

    se = np.load(str(let[index]) + 'ChosenExp' + str(chosenexp)+'SEchannel.npz',mmap_mode='r') 
    segmm = np.load(str(let[index]) + 'ChosenExp' + str(chosenexp)+'SEchannelGMM.npz',mmap_mode='r') 
    red = np.load(str(let[index]) + 'ChosenExp' + str(chosenexp)+'Redbright.npz',mmap_mode='r') 
    blue = np.load(str(let[index]) + 'ChosenExp' + str(chosenexp)+'Bluebright.npz',mmap_mode='r') 
    il = np.load(str(let[index]) + 'ChosenExp' + str(chosenexp)+'ILchannel.npz') 
    
    fsizetit = 18 
    fsizepl = 16 
    sizex = 8 
    sizey = 6
    dpi_no = 80
    lw = 2
    
    length_scalebar = 50.0 #in nm 
    scalebar_legend = '50nm'
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[index]))
    
    titulo = description + ' ' +  let[index] +  ' (10kV, 30$\mu$m aperture, 1$\mu$s time bins, ' + str(Ps)+ 'nm pixels, green/red: $</>$ 593nm)'
    
    fig40= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig40.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}') #not working
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}')  #not working
    plt.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]  #not working
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')         
    
    plt.suptitle("Segmentation (model: 2-GMM) of cathodoluminescence signal using SE channel, \n" + titulo,fontsize=fsizetit)

    gc.collect()
    
    ax1 = plt.subplot2grid((2,12), (0, 0), colspan=3)
    ax1.set_title('SE channel',fontsize=fsizepl) #as per accompanying txt files
    plt.imshow(se['data'],cmap=cm.Greys_r)   
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    ax1.add_artist(sbar)
    #sbar.set_font_size(fsizepl)
    plt.axis('off')
    
    gc.collect()
    ax1 = plt.subplot2grid((2,12), (0, 3), colspan=3)
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
    
    
    ######### stretch contrasts
    import skimage
    from skimage import exposure
    # Work RED
    redhelp = np.nanmean(red['data'],axis = (0,1))
    nred = (redhelp-np.min(redhelp))/(np.max(redhelp)-np.min(redhelp))
    redg = (nred*(255.0-0.0)) + 0.0
    imgr =   np.array(redg,dtype=np.uint8)
    # Contrast stretching
    p2, p98 = np.percentile(imgr, (2, 98))
    img_rescaler =  exposure.rescale_intensity(imgr, in_range=(p2, p98))
    # Equalization
    img_eqr =  exposure.equalize_hist(imgr)
    # Adaptive Equalization
    img_adapteqr =  exposure.equalize_adapthist(imgr, clip_limit=0.03)
    
    # Work BLUE
    bluehelp = np.nanmean(blue['data'],axis = (0,1))
    nblue = (bluehelp-np.min(bluehelp))/(np.max(bluehelp)-np.min(bluehelp))
    blueg = (nblue*(255.0-0.0)) + 0.0
    imgb =   np.array(blueg,dtype=np.uint8)
    # Contrast stretching
    p2, p98 = np.percentile(imgb, (2, 98))
    img_rescaleb =  exposure.rescale_intensity(imgb, in_range=(p2, p98))
    # Equalization
    img_eqb =  exposure.equalize_hist(imgb)
    # Adaptive Equalization
    img_adapteqb =  exposure.equalize_adapthist(imgb, clip_limit=0.03)
    
    
    #################
    
    ax1 = plt.subplot2grid((2,12), (0, 6), colspan=3)
    ax1.set_title('CL data averaged over time \n and 5 experiments',fontsize=fsizepl)
    #im = plt.imshow(np.nanmean(blue['data'],axis = (0,1)),cmap=cm.Greens) #or 'OrRd'
    im = plt.imshow(img_adapteqb,cmap=cm.Greens) #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    unit = '(kHz)'
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor = plt.axes([box.x0, box.y0*1.075 , box.width,0.01 ])    
    cb2 = plt.colorbar(im, cax = axColor, orientation="horizontal",ticks=[np.nanmean(blue['data'],axis = (0,1,2,3)), 10.0*np.nanmean(blue['data'],axis = (0,1,2,3))] ) #,np.nanmean(blue['data'],axis = (0,1,2,3))+np.nanstd(blue['data'],axis = (0,1,2,3)),np.nanmean(blue['data'],axis = (0,1,2,3))+3.0*np.nanstd(blue['data'],axis = (0,1,2,3))] )#/2.0,np.nanmax(blue['data'],axis = (0,1,2,3))]) 
    cb2.ax.tick_params(labelsize=fsizepl)
    cb2.set_label('Photon counts (kHz)', size = fsizepl)
    
    ax1 = plt.subplot2grid((2,12), (0, 9), colspan=3)
    ax1.set_title('CL data averaged over time \n and 5 experiments',fontsize=fsizepl)
    #imb = plt.imshow(np.nanmean(red['data'],axis = (0,1)),cmap=cm.Reds) #or 'OrRd'
    imb = plt.imshow(img_adapteqr,cmap=cm.Reds) #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor = plt.axes([box.x0, box.y0*1.075 , box.width,0.01 ])    
    #cb1 = plt.colorbar(imb, cax = axColor, orientation="horizontal",label='Photon counts ' + unit) 
    cb1 = plt.colorbar(imb, cax = axColor, orientation="horizontal",ticks=[np.nanmean(red['data'],axis = (0,1,2,3)), 10.0*np.nanmean(red['data'],axis = (0,1,2,3))]) 
    cb1.ax.tick_params(labelsize=fsizepl)
    cb1.set_label('Photon counts (kHz)', size = fsizepl)
    
    ax1 = plt.subplot2grid((2,12), (1, 0), colspan=3)
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
    ax1.axvspan(50.0,50.0+150.0, alpha=0.25, color='yellow')
    unit = 'kHz'
    plt.ylabel("Average luminescence,\n per signal pixel (" + unit + ")",fontsize=fsizepl)
    plt.xlabel("Time as electron beam on/off ($\mu$s)",fontsize=fsizepl)
    major_ticks0 = [500,1000,1500]
    ax1.set_xticks(major_ticks0) 
    #ax1.set_yticks([15,30,45]) 
    plt.ylim([0,10])
    ax1.tick_params(labelsize=fsizepl)
    
    ax1 = plt.subplot2grid((2,12), (1, 4), colspan=4)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    datared = np.average(red['data'], axis = (0))
    datablue = np.average(blue['data'], axis = (0))
    
    if True is False:
        pass
    else:
        initbin = (150+50+3)-1  #init bin for decay
        backgdinit = 50
        ### 700ns /40ns = 7. ....
        datared_init = datared[0:backgdinit,:,:]
        datared = datared[initbin:,:,:]
        datablue_init = datablue[0:backgdinit,:,:]
        datablue = datablue[initbin:,:,:]

    fastfactor = 1
    
    #hlp_red = datared * hlp
    #hlp_blue = datablue * hlp
   
    #hlpred = np.nanmean(red['data'][:,initbin:,:,:]*hlp,axis = (2,3))
    #hlpblue = np.nanmean(blue['data'][:,initbin:,:,:]*hlp,axis = (2,3))
    
    #error_arrayr = np.nanstd(hlpred,axis = 0)
    #error_arrayr[error_arrayr < 0.05 * np.max(hlpred)]  = 0.05 * np.max(hlpred) #arbitrary 5% of max  
    
    cut_longa = 6#8#6
    cut_shorta =  3#5#3
    
    print('bef init')
    gc.collect()
    
    init_guess = calc_double_fit(np.arange(0,datared.shape[0])*Time_bin*1e-9*fastfactor,np.nanmean(datared*hlp,axis=(1,2)),dt= Time_bin*1e-9*fastfactor,cut_long=cut_longa,cut_short=cut_shorta)
    
    gc.collect()    
    
    init_guessb = calc_double_fit(np.arange(0,datablue.shape[0])*Time_bin*1e-9*fastfactor,np.nanmean(datablue*hlp,axis=(1,2)),dt= Time_bin*1e-9*fastfactor,cut_long=cut_longa,cut_short=cut_shorta)
    
    cinit = np.nanmean(datared_init*hlp,axis=(0,1,2))
    cinitb = np.nanmean(datablue_init*hlp,axis=(0,1,2))    
    
    #replace c with cinit
    init_guess[2] = cinit 
    init_guessb[2] = cinitb
    if index == 4:
        print('here blue')
        init_guessb[1] = 5.0
    
    print('bef fit')
    gc.collect()
    
    b,e,be,ee,b2,e2,be2,ee2 = calcdecay_subplot_nan(datared*hlp, time_detail= Time_bin*1e-9*fastfactor,titulo='',single=False,other_dset1=None,other_dset2=datablue*hlp,init_guess=init_guess,init_guess2=init_guessb,unit='kHz',error_array=None) #,error_array=error_arrayr, error_array2=error_arrayb)    
        
    b_array_red[index] = b
    e_array_red[index] = e
    be_array_red[index] = be    
    ee_array_red[index] = ee  
    b_array_blue[index] = b2
    e_array_blue[index] = e2
    be_array_blue[index] = be2    
    ee_array_blue[index] = ee2  
 
    plt.ylabel("Average luminescence,\n per signal pixel (" + unit + ")",fontsize=fsizepl)
    plt.xlabel(r'Time after blanking the electron beam ($\mu$s)',  fontsize=fsizepl)
    plt.xlim(xmax=1400.0) #1400
    #major_ticks0 = [250,500,750,1000,1250]
    #ax1.set_xticks(major_ticks0) 
    ax1.tick_params(labelsize=fsizepl)
    
    # Extra plots        
    aaa = datared*hlp
    xx_array = np.arange(0,aaa.shape[0])*Time_bin*1e-9*fastfactor
    #Plot whole of background decay
    plt.semilogy(xx_array/1e-6,np.nanmean(datared*hlpd,axis=(1,2)),'o',color='DarkRed',markersize=3,label='Transient CL from red background')   
    #Plot mean signal before e-beam on
    plt.semilogy(xx_array/1e-6,np.nanmean(datared_init*hlp,axis=(0,1,2))*np.ones(len(xx_array)),'r--',lw=2,label='Mean CL from signal pixels, before e-beam')
    #Plot mean background
    plt.semilogy(xx_array/1e-6,np.nanmean(datared_init*hlpd,axis=(0,1,2))*np.ones(len(xx_array)),'--',color='DarkRed',lw=1,label='Mean CL from background, before e-beam')
    
    #Plot whole of background decay
    plt.semilogy(xx_array/1e-6,np.nanmean(datablue*hlpd,axis=(1,2)),'o',color='DarkGreen',markersize=3,label='Transient CL from blue background')   
    #Plot mean signal before e-beam on
    plt.semilogy(xx_array/1e-6,np.nanmean(datablue_init*hlp,axis=(0,1,2))*np.ones(len(xx_array)),'g--',lw=2,label='Mean CL from blue signal pixels, before e-beam')
    #Plot mean background
    plt.semilogy(xx_array/1e-6,np.nanmean(datablue_init*hlpd,axis=(0,1,2))*np.ones(len(xx_array)),'--',color='DarkGreen',lw=1,label='Mean CL from background, before e-beam')
    
    
    #plt.legend(loc='southwest')
        
    #ax1.set_xticks(major_ticks0)
    
    ax1 = plt.subplot2grid((2,12), (1, 9), colspan=3)
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('right')
    ax1.yaxis.set_label_position("right")
    
    del datared, datablue, datablue_init, datared_init
    gc.collect()
    if index is not 5:
        dataALLred = red['data'][:,:,:,:]
        dataALLblue = blue['data'][:,:,:,:]
    else:
        dataALLred = red['data'][0:3,:,:,:]
        dataALLblue = blue['data'][0:3,:,:,:]
    nominal_time_on = 150.0
    
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)),c='r',linestyle='None', marker='o',markersize=6) #in mus, in MHz
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='r',  linewidth= lw) #in mus, in MHz
    #background intensity
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3)),c='DarkRed', label='',linestyle='None', marker='o',markersize=5) #in mus, in MHz
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='DarkRed',  linewidth= lw) #in mus, in MHz
    #init red intensity
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred[:,0:backgdinit,:,:]*hlp,axis=(1,2,3)),c='r', linestyle='None', marker='o',markersize=5) #in mus, in MHz
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred[:,0:backgdinit,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),'--',c='r',  linewidth= lw) #in mus, in MHz


    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)),c='g', linestyle='None', marker='o',markersize=6) #in mus, in MHz
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='g',  linewidth= lw) #in mus, in MHz
    #background intensity
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3)),c='DarkGreen', label='',linestyle='None', marker='o',markersize=5) #in mus, in MHz
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='DarkGreen',  linewidth= lw) #in mus, in MHz
    #init red intensity
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,0:backgdinit,:,:]*hlp,axis=(1,2,3)),c='g', linestyle='None', marker='o',markersize=5) #in mus, in MHz
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,0:backgdinit,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),'--',c='g',  linewidth= lw) #in mus, in MHz

    aa = np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3))/np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3))
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3))/np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)),c='k',linestyle='None', marker='s',markersize=5)# + str("{0:.2f}".format(aa[0])),linestyle='None', marker='s',markersize=6) #in mus, in MHz
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3))/np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='k',  linewidth= lw,label='green/red signal' ) #in mus, in MHz
    #plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3))/np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)),c='k',linestyle='None', marker='s',markersize=4,label='green/red' )# + str("{0:.2f}".format(aa[0])),linestyle='None', marker='s',markersize=6) #in mus, in MHz
    
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3))/np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3)),c='y',linestyle='None', marker='s',markersize=5)# + str("{0:.2f}".format(aa[0])),linestyle='None', marker='s',markersize=6) #in mus, in MHz
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))/np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='y',  linewidth= lw,label='green/red backgd' ) #in mus, in MHz

    plt.legend(loc='best')
    
    red_int_array[index] = np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) #np.nanmean(dataALLred*hlp,axis=(0,1,2,3))
    cumu_red[index,:] =  np.nanmean(dataALLred*hlp,axis=(1,2,3))
    blue_int_array[index] = np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) #np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))
    cumu_blue[index,:] =  np.nanmean(dataALLred*hlp,axis=(1,2,3))
    
    ############################## FFT to cut noise in Temperature/IL data
    result = []
    #print(dataALLred.shape[1])
    total_time = dataALLred.shape[1]*Time_bin/1000.0 # in us
    #print(total_time)

    se = np.array(il['data'])
    #se = se[0, :, :]
    t = se.flatten()
    x = np.linspace(0, total_time, len(t))
   
    #plt.figure(1)
    #plt.plot( x * 1e-6/1e-3, t)
    #plt.figure(2)
   
    fft_y = np.fft.fft(t)
   
    n = t.size
    timestep = Time_bin
    freq = np.fft.fftfreq(n, d = timestep)
   
   #plt.semilogy(freq, np.abs(fft_y))
   
    ind = np.abs(freq) > 1000
   
    fft_y_cut = np.copy(fft_y)
    fft_y_cut[ind] = 0.0
   
   #plt.semilogy(freq, np.abs(fft_y_cut), 'r')
   
    new_y = np.abs(np.fft.ifft(fft_y_cut))
   
   #plt.figure(1)
   #plt.plot( x * 1e-6/1e-3, new_y, label = str(k))
   #plt.legend()

    result.append(np.mean(new_y[:-10]))

    result = np.array(result)

    deltav =0# -0.190e-3
    #il_data[index,:] = KThermocouplerconversion(result + 1.0e-3 + deltav)
    il_data_std[index,:] = KThermocouplerconversion(np.std(new_y[:-10]) )
    #print(il_data)
    #print(il_data_std)
    
    if index == 0:
        deltav2 = +0.238e-3
    if index == 1:
        deltav2 = +0.238e-3
    if index == 2:
        deltav2 = +0.238e-3
    if index == 3:
        deltav2 = +0.359e-3
    if index == 4:
        deltav2 = +0.505e-3
    if index == 5:
        deltav2 = +0.691e-3
    il_data[index,:] = KThermocouplerconversion(np.average(il['data'], axis = (1,2)) + 1.0e-3 + deltav2) #take average per frame
    #il_data_std[index,:] = KThermocouplerconversion(np.std(il['data'], axis = (1,2)) )
    print(il_data)
    print(il_data_std)
    
    ##### ABOVE, IMPORTANT: ADD 1mV, WHICH IS THE VOLTAGE AT 25c, RT
    
    red_std_array[index] = np.nanstd(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) #np.nanmean(dataALLred*hlp,axis=(0,1,2,3))
    blue_std_array[index] = np.nanstd(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) #np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))
    
    plt.ylabel("Average luminescence for each experiment,\n per signal pixel while excited (kHz)",fontsize=fsizepl)
    plt.xlabel("Cumulative e-beam exposure time \n per pixel ($\mu$s)",fontsize=fsizepl)
    plt.ylim([0,8])
    
    #plt.xlim([nominal_time_on - 1.0,nominal_time_on*No_experiments[index]*fastfactor +1.0])
    plt.xlim([140,760])
    major_ticks0 = [150,300,450,600,750]
    ax1.set_xticks(major_ticks0) 
    ax1.tick_params(labelsize=fsizepl)
    
    
    #sig_to_back[index] =   np.nanmean(dataALLred*hlp,axis=(0,1,2,3))/np.nanmean(dataALLred*hlpd,axis=(0,1,2,3))
    #sig_to_initred[index] =   np.nanmean(dataALLred*hlp,axis=(0,1,2,3))/np.nanmean(dataALLred[:,0:16,:,:]*hlp,axis=(0,1,2,3))
    
    
    #sig_to_back_blue[index] =   np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))/np.nanmean(dataALLblue*hlpd,axis=(0,1,2,3))
    #sig_to_initblue[index] =   np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))/np.nanmean(dataALLblue[:,0:16,:,:]*hlp,axis=(0,1,2,3))
    
    multipage_longer('ChosenExp' + str(chosenexp) + 'ZZZZSingle-'+ let[index] + '.pdf',dpi=80)

##### ONCE ALL FITS WORK, 
###### NEED TO RUN THESE LINES BELOW SO THAT ALL NPZ FILES ARE CREATED 

mycode = 'Red_std_array' +  'ChosenExp' + str(chosenexp) + '= tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Red_std_array'+  'ChosenExp' + str(chosenexp), data = red_std_array)

mycode = 'Blue_std_array'  + 'ChosenExp' + str(chosenexp) + '= tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Blue_std_array' +  'ChosenExp' + str(chosenexp), data = blue_std_array)

mycode = 'Red_int_array' +  'ChosenExp' + str(chosenexp) + '= tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Red_int_array'+  'ChosenExp' + str(chosenexp), data = red_int_array)

mycode = 'Blue_int_array'+  'ChosenExp' + str(chosenexp) + ' = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Blue_int_array'+  'ChosenExp' + str(chosenexp), data = blue_int_array)

mycode = 'Cumu_red'+  'ChosenExp' + str(chosenexp) + ' = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Cumu_red'+  'ChosenExp' + str(chosenexp), data = cumu_red)

mycode = 'Cumu_blue'+  'ChosenExp' + str(chosenexp) + ' = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Cumu_blue'+  'ChosenExp' + str(chosenexp), data = cumu_blue)

#il_data will be [6,5] in shape
mycode = 'Il_data'+  'ChosenExp' + str(chosenexp) + ' = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Il_data'+  'ChosenExp' + str(chosenexp), data = il_data)

mycode = 'Il_data_std'+  'ChosenExp' + str(chosenexp) + ' = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Il_data_std'+  'ChosenExp' + str(chosenexp), data = il_data_std)

mycode = 'B_array_red'+  'ChosenExp' + str(chosenexp) + ' = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('B_array_red'+  'ChosenExp' + str(chosenexp), data = b_array_red)

mycode ='Be_array_red'+  'ChosenExp' + str(chosenexp) + ' = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Be_array_red'+  'ChosenExp' + str(chosenexp), data = be_array_red)

mycode = 'E_array_red'+  'ChosenExp' + str(chosenexp) + ' = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('E_array_red'+  'ChosenExp' + str(chosenexp), data = e_array_red)

mycode = 'Ee_array_red'+  'ChosenExp' + str(chosenexp) + ' = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Ee_array_red'+  'ChosenExp' + str(chosenexp), data = ee_array_red)

mycode = 'B_array_blue'+  'ChosenExp' + str(chosenexp) + ' = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('B_array_blue'+  'ChosenExp' + str(chosenexp), data = b_array_blue)

mycode ='Be_array_blue'+  'ChosenExp' + str(chosenexp) + ' = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Be_array_blue'+  'ChosenExp' + str(chosenexp), data = be_array_blue)

mycode = 'E_array_blue'+  'ChosenExp' + str(chosenexp) + ' = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('E_array_blue'+  'ChosenExp' + str(chosenexp), data = e_array_blue)

mycode = 'Ee_array_blue'+  'ChosenExp' + str(chosenexp) + ' = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Ee_array_blue'+  'ChosenExp' + str(chosenexp), data = ee_array_blue)

#mycode = 'Size_signal = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Size_signal', data = size_signal)
#
#mycode = 'Sig_to_back = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Sig_to_back', data = sig_to_back)
#
#mycode = 'Sig_to_initred = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Sig_to_initred', data = sig_to_initred)

#####FIG WITH SUMMARY   
##### WHEN FILES ABOVE ARE CREATED, CREATE FIGURE BELOW WITH REGION SIZES VS LIFETIMES AND REGION SIZES VS S/N OR BRIGHTNESS OR <SIGNAL INT>/<BACKGROUND INT>
klklklklk
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b-')

ax1.set_xlabel('X data')
ax1.set_ylabel('Y1 data', color='g')
ax2.set_ylabel('Y2 data', color='b')




fastfactor = 1

Red_int_array = np.load('Red_int_array.npz') 
red_int_array = Red_int_array['data']
Blue_int_array = np.load('Blue_int_array.npz') 
blue_int_array = Blue_int_array['data']

B_array_red= np.load('B_array_red.npz')
b_array_red = B_array_red['data']  
Be_array_red = np.load('Be_array_red.npz')
be_array_red = Be_array_red['data'] 
E_array_red = np.load('E_array_red.npz')
e_array_red = E_array_red['data']   
Ee_array_red = np.load('Ee_array_red.npz')
ee_array_red = Ee_array_red['data']   

Size_signal = np.load('Size_signal.npz') 
size_signal = Size_signal['data']

B_array_blue= np.load('B_array_blue.npz')
b_array_blue = B_array_blue['data']  
Be_array_blue = np.load('Be_array_blue.npz')
be_array_blue = Be_array_blue['data'] 
E_array_blue = np.load('E_array_blue.npz')
e_array_blue = E_array_blue['data']   
Ee_array_blue = np.load('Ee_array_blue.npz')
ee_array_blue = Ee_array_blue['data'] 

Size_signal = np.load('Size_signal.npz') 
size_signal = Size_signal['data']

Sig_to_back = np.load('Sig_to_back.npz') 
sig_to_back = Sig_to_back['data']

Sig_to_initred = np.load('Sig_to_initred.npz') 
sig_to_initred = Sig_to_initred['data']

####
    
fig41= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig41.set_size_inches(1200./fig41.dpi,900./fig41.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    
fig41.suptitle('Lifetimes fitted after 1300$\mu$s decay, small NaYF$_4$ CS from Andrea \n (10kV, 30$\mu$m aperture, 40ns time bins)',fontsize=fsizetit)     

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

#ax2.set_ylim([0.0,10.0])
#ax2.set_yticks([2.5, 5, 7.5, 10.0])
#ax1.set_yticks([0.025, 0.05, 0.075,0.1])
#ax1.set_ylim([0,0.1])
#ax1.set_xticks([10,20,30,40,50,60,70,80])
#ax1.set_xticklabels(let)

cut = [0,1,2,3,4,5,6]

rest_points = range(cut[-1]+1, len(size_signal))



x_vec = size_signal[cut]

ax1.errorbar(x_vec, b_array_red[cut], yerr=be_array_red[cut], fmt='ro',markersize=5)
ax2.errorbar(x_vec, e_array_red[cut], yerr=ee_array_red[cut], fmt='ro', markersize=10)

x_vec = size_signal[rest_points]

ax1.errorbar(x_vec, b_array_red[rest_points], yerr=be_array_red[rest_points], fmt='go',markersize=5)
ax2.errorbar(x_vec, e_array_red[rest_points], yerr=ee_array_red[rest_points], fmt='go', markersize=10)


ax1.set_ylabel('Shorter time constant ($\mu$s)',fontsize=fsizepl)
ax2.set_ylabel('Longer time constant ($\mu$s)',fontsize=fsizepl)
ax2.set_ylim([0,1.5])
#plt.xlim([0,90])
#ax2.legend(fontsize=fsizepl)
ax1.set_xlabel('Size nanodiamond (nm$^2$)',fontsize=fsizepl)

plt.sca(ax1) 

plt.setp(ax2.get_xticklabels(), visible=False)

####
    
fig42= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig42.set_size_inches(1200./fig41.dpi,900./fig41.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    
fig42.suptitle('Lifetimes fitted after 1300$\mu$s decay, small NaYF$_4$ CS from Andrea \n (10kV, 30$\mu$m aperture, 40ns time bins)',fontsize=fsizetit)     

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

#ax2.set_ylim([0.0,10.0])
#ax2.set_yticks([2.5, 5, 7.5, 10.0])
#ax1.set_yticks([0.025, 0.05, 0.075,0.1])
#ax1.set_ylim([0,0.1])
#ax1.set_xticks([10,20,30,40,50,60,70,80])
#ax1.set_xticklabels(let)
x_vec = pisize

ax1.errorbar(x_vec, b_array_red, yerr=be_array_red, fmt='ro',markersize=5)
ax2.errorbar(x_vec, e_array_red, yerr=ee_array_red, fmt='ro', markersize=10)
ax1.set_ylabel('Shorter time constant ($\mu$s)',fontsize=fsizepl)
ax2.set_ylabel('Longer time constant ($\mu$s)',fontsize=fsizepl)
ax2.set_ylim([0,1.5])
#plt.xlim([0,90])
#ax2.legend(fontsize=fsizepl)
ax1.set_xlabel('Pixel size (nm)',fontsize=fsizepl)

plt.sca(ax1) 

plt.setp(ax2.get_xticklabels(), visible=False)

####
    
fig43= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig43.set_size_inches(1200./fig41.dpi,900./fig41.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    
fig43.suptitle('Lifetimes fitted after 1300$\mu$s decay, small NaYF$_4$ CS from Andrea \n (10kV, 30$\mu$m aperture, 40ns time bins)',fontsize=fsizetit)     

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

#ax2.set_ylim([0.0,10.0])
#ax2.set_yticks([2.5, 5, 7.5, 10.0])
#ax1.set_yticks([0.025, 0.05, 0.075,0.1])
#ax1.set_ylim([0,0.1])
#ax1.set_xticks([10,20,30,40,50,60,70,80])
#ax1.set_xticklabels(let)
x_vec = sig_to_back

ax1.errorbar(x_vec, b_array_red, yerr=be_array_red, fmt='ro',markersize=5)
ax2.errorbar(x_vec, e_array_red, yerr=ee_array_red, fmt='ro', markersize=10)
ax1.set_ylabel('Shorter time constant ($\mu$s)',fontsize=fsizepl)
ax2.set_ylabel('Longer time constant ($\mu$s)',fontsize=fsizepl)
ax2.set_ylim([0,1.5])
#plt.xlim([0,90])
#ax2.legend(fontsize=fsizepl)
ax1.set_xlabel('Signal to background',fontsize=fsizepl)

plt.sca(ax1) 

plt.setp(ax2.get_xticklabels(), visible=False)

####
    
fig44= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig44.set_size_inches(1200./fig41.dpi,900./fig41.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    
fig44.suptitle('Lifetimes fitted after 1300$\mu$s decay, small NaYF$_4$ CS from Andrea \n (10kV, 30$\mu$m aperture, 40ns time bins)',fontsize=fsizetit)     

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

#ax2.set_ylim([0.0,10.0])
#ax2.set_yticks([2.5, 5, 7.5, 10.0])
#ax1.set_yticks([0.025, 0.05, 0.075,0.1])
#ax1.set_ylim([0,0.1])
#ax1.set_xticks([10,20,30,40,50,60,70,80])
#ax1.set_xticklabels(let)
x_vec = sig_to_initred

ax1.errorbar(x_vec, b_array_red, yerr=be_array_red, fmt='ro',markersize=5)
ax2.errorbar(x_vec, e_array_red, yerr=ee_array_red, fmt='ro', markersize=10)
ax1.set_ylabel('Shorter time constant ($\mu$s)',fontsize=fsizepl)
ax2.set_ylabel('Longer time constant ($\mu$s)',fontsize=fsizepl)
ax2.set_ylim([0,1.5])
#plt.xlim([0,90])
#ax2.legend(fontsize=fsizepl)
ax1.set_xlabel('Signal to initial signal, before e-beam',fontsize=fsizepl)

plt.sca(ax1) 

plt.setp(ax2.get_xticklabels(), visible=False)

multipage_longer('ZZZZZSummary.pdf',dpi=80)
