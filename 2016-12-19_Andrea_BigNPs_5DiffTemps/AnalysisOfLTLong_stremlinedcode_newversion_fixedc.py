#nominally
#50mus moving
#150mus on
#1400mus off
#clock 1MHz = 1mus time bon
#250kX, 50%x 50% scale
#250x250 pixels
#5 frames per temperature (ie, per voltage )

##### Temperatures were not read, estimated before in the program
#29.89
#36.70
#41.79
#48.83
#69.84

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

import skimage
from skimage import exposure
#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################

#MIND THE PREFIX!!!!!!!!111111111111111111111111111111111111111111

prefix = 'fixedCprevtau'

Time_bin = 1000#in ns; 
nominal_time_on = 150.0 #time during which e-beam nominally on, in mus
totalpoints = 1500 #total number of time-resolved points

nametr = ['2016-12-19-1924_ImageSequence__100.000kX_10.000kV_30mu_4',
          '2016-12-19-1950_ImageSequence__100.000kX_10.000kV_30mu_5',
          '2016-12-19-2130_ImageSequence__100.000kX_10.000kV_30mu_8',
          '2016-12-19-2015_ImageSequence__100.000kX_10.000kV_30mu_6',
          '2016-12-19-2056_ImageSequence__27.836kX_10.000kV_30mu_7']
          
Pixel_size = 2.2*np.ones(len(nametr)) #nm
Ps = Pixel_size #pixel size in nm, the numbers above with round nm precision
No_experiments = [1,1,1,1,1]

description = 'Andrea small NaYF4:Er'    # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
kv = [10]

let = ['V0pt25' ,'V0pt5', 'V0', 'V0pt75','V1']


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
#cumu_blue = np.zeros([len(nametr),No_experiments[0]])
#cumu_red = np.zeros([len(nametr),No_experiments[0]])
il_data = np.zeros([len(nametr)])
il_data_std = np.zeros([len(nametr)])
#size_signal = np.zeros(len(nametr))
#sig_to_back = np.zeros(len(nametr))
#sig_to_initred = np.zeros(len(nametr))
#sig_to_back_blue = np.zeros(len(nametr))
#sig_to_initblue = np.zeros(len(nametr))
blue_std_array = np.zeros(len(nametr))
red_std_array = np.zeros(len(nametr))

pisize =Pixel_size

listofindex =np.arange(0,len(nametr))#,11]

consider_whole_light = [0,1,2,3,4]
#4,5,7 segmentation ok-ish
#index = 6
#if index is 6:
for index in listofindex:
    
    print(index)
    
    Ps = str("{0:.2f}".format(Pixel_size[index])) 
    print('bef loading')
    se = np.load(str(let[index]) +'SEchannel.npz',mmap_mode='r') 
    segmm = np.load(str(let[index]) +'SEchannelGMM.npz',mmap_mode='r') 
    red = np.load(str(let[index]) +'Redbright.npz',mmap_mode='r') 
    blue = np.load(str(let[index]) +'Bluebright.npz',mmap_mode='r') 
    il = np.load(str(let[index]) +'ILchannel.npz') 
    print('after loading')
    
    ###IL
    if index == 0:
        il_data[index] = 29.89
#    elif index == 1:
#        il_data[index] = 27.8
#    elif index == 2:
#        il_data[index] = 30.5
    else:
        #calibration done using 2017 jan 5 data
        aparam = 0.233801
        bparam = 0.000144
        delta = aparam*np.average(il['data'], axis = (0,1,2)) + bparam
        il_data[index] = KThermocouplerconversion(np.average(il['data'], axis = (0,1,2)) + 1.0e-3 + delta)
        
    
#    if index == 0:
#       deltav2 = -0.046e-3  # KThermocouplerconversion(1.0e-3) #take average per frame 
#    if index == 1:
#        deltav2 = +0.238e-3
#    if index == 2:
#        deltav2 = +0.156e-3
#    if index == 3:
#        deltav2 = +0.225e-3
#    if index == 4:
#        deltav2 = +0.359e-3
#    if index == 5:
#        deltav2 = +0.505e-3
#    if index == 6:
#       deltav2 = +0.660e-3
#    
#    il_data[index] = KThermocouplerconversion(np.average(il['data'], axis = (0,1,2)) + 1.0e-3 + deltav2) #take average per frame   
#    print(np.average(il['data'], axis = (0,1,2)))
    print(il_data)
    
    ############################## FFT to cut noise in Temperature/IL data
    #result = []
    #print(dataALLred.shape[1])
    total_time = red['data'].shape[1]*Time_bin/1000.0 # in us
    #print(total_time)
    se1 = np.array(il['data'])
    #se = se[0, :, :]
    t = se1.flatten()
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
    #result.append(np.mean(new_y[:-10]))
    #result = np.array(result)
    #deltav =0# -0.190e-3
    #il_data[index] = KThermocouplerconversion(np.array(np.mean(new_y[:-10])) + 1.0e-3)
    il_data_std[index] = KThermocouplerconversion(np.std(new_y[:-10]) )
    #print(il_data)
    #print(il_data_std)
#    
   
    #il_data_std[index] = KThermocouplerconversion(np.std(il['data'], axis = (1,2)) )
    
    print(il_data_std) 
    del il, se1, t, fft_y, fft_y_cut, freq, ind, x, new_y
    gc.collect()
    
    print('after il data')
    
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
    
    if index in consider_whole_light:
        hlpse = 0.0
        #hlpse[np.isnan(hlpse)] = 0.0 #outside, consider all light
    else:    
        hlpse = np.copy(segmm['bright'])
        hlpse[~np.isnan(hlpse)] = 0.0 #inside
        hlpse[np.isnan(hlpse)] = 1.0 #outside
    #im = plt.imshow(hlpse,cmap=cm.Greys) #or 'OrRd'
    #sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    #ax1.add_artist(sbar)
    plt.axis('off')
    
    del hlpse, se, segmm
    gc.collect()
    
#==============================================================================
#    print('bef skimage')
#    ######### stretch contrasts
#    ######## COPIED FROM BELOW
    backgdinit = 50
    initbin = (150+50+3)-1
#    # Work RED
#    #try_vec_red = red['data'][:,backgdinit:initbin,:,:]
#    #print(try_vec_red.shape)
#    redhelp = np.average(red['data'][:,backgdinit:initbin,:,:],axis = (0,1))
#    print(np.average(redhelp, axis = (0,1)))
#    #print(np.average(redhelp, axis = (0,1)))
#    nred = (redhelp-np.min(redhelp))/(np.max(redhelp)-np.min(redhelp))
#    redg = (nred*(255.0-0.0)) + 0.0
#    imgr =   np.array(redg,dtype=np.uint8)
# # Contrast stretching
# #p2, p98 = np.percentile(imgr, (2, 98))
# #img_rescaler =  exposure.rescale_intensity(imgr, in_range=(p2, p98))
# # Equalization
# #img_eqr =  exposure.equalize_hist(imgr)
# # Adaptive Equalization
#    img_adapteqr =  exposure.equalize_adapthist(imgr, clip_limit=0.03)
# 
# # Work BLUE
#   # try_vec_blue = blue['data'][:,backgdinit:initbin,:,:]
#    bluehelp =np.average(blue['data'][:,backgdinit:initbin,:,:],axis = (0,1)) # np.average(try_vec_blue,axis = (0,1)) #np.average(blue['data'][:,backgdinit:initbin,:,:],axis = (0,1))
#    nblue = (bluehelp-np.min(bluehelp))/(np.max(bluehelp)-np.min(bluehelp))
#    blueg = (nblue*(255.0-0.0)) + 0.0
#    imgb =   np.array(blueg,dtype=np.uint8)
# # Contrast stretching
# #p2, p98 = np.percentile(imgb, (2, 98))
# #img_rescaleb =  exposure.rescale_intensity(imgb, in_range=(p2, p98))
# # Equalization
# #img_eqb =  exposure.equalize_hist(imgb)
# # Adaptive Equalization
#    img_adapteqb =  exposure.equalize_adapthist(imgb, clip_limit=0.03)
#==============================================================================
    print('after skimage')
    
    #################
    
    ax1 = plt.subplot2grid((2,12), (0, 6), colspan=3)
    ax1.set_title('CL data while e-beam on',fontsize=fsizepl)
    im = plt.imshow(np.average(blue['data'][:,backgdinit:initbin,:,:],axis = (0,1)),cmap=cm.Greens) #or 'OrRd'
    #im = plt.imshow(img_adapteqb,cmap=cm.Greens) #or 'OrRd'
    print('after imshow')
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    unit = '(kHz)'
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor = plt.axes([box.x0, box.y0*1.075 , box.width,0.01 ])    
    print(np.average(blue['data'][:,backgdinit:initbin,:,:],axis = (0,1,2,3)))
    tickval = np.average(blue['data'][:,backgdinit:initbin,:,:],axis = (0,1,2,3))
    cb2 = plt.colorbar(im, cax = axColor, orientation="horizontal",ticks=[0.5*tickval,tickval,2.0*tickval],format='%0.2f' ) #,np.nanmean(blue['data'],axis = (0,1,2,3))+np.nanstd(blue['data'],axis = (0,1,2,3)),np.nanmean(blue['data'],axis = (0,1,2,3))+3.0*np.nanstd(blue['data'],axis = (0,1,2,3))] )#/2.0,np.nanmax(blue['data'],axis = (0,1,2,3))]) 
    #cb2 = plt.colorbar(im, cax = axColor, orientation="horizontal",ticks=[0.1*np.average(blue['data'][:,backgdinit:initbin,:,:],axis = (0,1,2,3)),np.average(blue['data'][:,backgdinit:initbin,:,:],axis = (0,1,2,3)), 10.0*np.average(blue['data'][:,backgdinit:initbin,:,:],axis = (0,1,2,3))] ) #,np.nanmean(blue['data'],axis = (0,1,2,3))+np.nanstd(blue['data'],axis = (0,1,2,3)),np.nanmean(blue['data'],axis = (0,1,2,3))+3.0*np.nanstd(blue['data'],axis = (0,1,2,3))] )#/2.0,np.nanmax(blue['data'],axis = (0,1,2,3))]) 
    print('after colorbar')
    cb2.ax.tick_params(labelsize=fsizepl)
    cb2.set_label('Photon counts (kHz)', size = fsizepl)
    
    ax1 = plt.subplot2grid((2,12), (0, 9), colspan=3)
    ax1.set_title('CL data while e-beam on',fontsize=fsizepl)
    imb = plt.imshow(np.average(red['data'][:,backgdinit:initbin,:,:],axis = (0,1)),cmap=cm.Reds) #or 'OrRd'
    #imb = plt.imshow(img_adapteqr,cmap=cm.Reds) #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor = plt.axes([box.x0, box.y0*1.075 , box.width,0.01 ])    
    #cb1 = plt.colorbar(imb, cax = axColor, orientation="horizontal",label='Photon counts ' + unit) 
    tickval =  np.average(red['data'][:,backgdinit:initbin,:,:],axis = (0,1,2,3))
    cb1 = plt.colorbar(imb, cax = axColor, orientation="horizontal",ticks=[0.5*tickval,tickval,2.0*tickval],format='%0.2f') 
    #cb1 = plt.colorbar(imb, cax = axColor, orientation="horizontal",ticks=[0.1*np.average(red['data'][:,backgdinit:initbin,:,:],axis = (0,1,2,3)),np.average(red['data'][:,backgdinit:initbin,:,:],axis = (0,1,2,3)), 10.0*np.average(red['data'][:,backgdinit:initbin,:,:],axis = (0,1,2,3))]) 
    cb1.ax.tick_params(labelsize=fsizepl)
    cb1.set_label('Photon counts (kHz)', size = fsizepl)
    
    ax1 = plt.subplot2grid((2,12), (1, 0), colspan=3)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
     
    print('after pic plotting') 
       
    # INSIDE
    
    
    if index in consider_whole_light:
        hlp = 1.0 #outside, consider all light
    else:
        hlp = np.copy(segmm['bright'])
        hlp[~np.isnan(hlp)] = 1.0  #inside
    
    # OUTSIDE
    
    if index in consider_whole_light:
        hlpd  = 0.0 #consider all light
    else:
        hlpd = np.copy(segmm['bright'])
        hlpd[~np.isnan(hlpd)] = 0.0 
        hlpd[np.isnan(hlpd)] = 1.0
        
    gc.collect()    
    print('bef plotting of plot1')
    plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1.0e3,np.average(red['data']*hlp,axis = (0,2,3)),c='r',lw=2) #in mus, in MHz
    print('after 1st plot statement of plot1')
    #plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1e3,np.average(red['data']*hlpd,axis = (0,2,3)),c='DarkRed',lw=2) #in mus, in MHz
    plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1.0e3,np.average(blue['data']*hlp,axis = (0,2,3)),c='g',lw=2) #in mus, in MHz
    #plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1e3,np.average(blue['data']*hlpd,axis = (0,2,3)),c='DarkGreen',lw=2) #in mus, in MHz
    print('after plotting of plot1')
    ax1.axvspan(50.0,50.0+150.0, alpha=0.25, color='yellow')
    unit = 'kHz'
    plt.ylabel("Average luminescence,\n per signal pixel (" + unit + ")",fontsize=fsizepl)
    plt.xlabel("Time as electron beam on/off ($\mu$s)",fontsize=fsizepl)
    major_ticks0 = [500,1000,1500]
    ax1.set_xticks(major_ticks0) 
    #ax1.set_yticks([15,30,45]) 
    plt.ylim([0,10])
    ax1.tick_params(labelsize=fsizepl)
    
    print('after plot 1') 
    
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
        initbin = (150+50+3)-1 #init bin for decay
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
    cut_shorta = 3#5#3
    
    print('bef init')
    gc.collect()
    
    init_guess = calc_double_fit(np.arange(0,datared.shape[0])*Time_bin*1.0e-9*fastfactor,np.nanmean(datared*hlp,axis=(1,2)),dt= Time_bin*1.0e-9*fastfactor,cut_long=cut_longa,cut_short=cut_shorta)
    
    gc.collect()    
    
    init_guessb = calc_double_fit(np.arange(0,datablue.shape[0])*Time_bin*1.0e-9*fastfactor,np.nanmean(datablue*hlp,axis=(1,2)),dt= Time_bin*1.0e-9*fastfactor,cut_long=cut_longa,cut_short=cut_shorta)
    
    cinit = np.nanmean(datared_init*hlp,axis=(0,1,2))
    cinitb = np.nanmean(datablue_init*hlp,axis=(0,1,2))    
    
    #replace c with cinit
    init_guess[2] = cinit 
    init_guessb[2] = cinitb
#    if index == 4:
#        print('here blue')
#        init_guessb[1] = 5.0
        
    ##################################### THIS IS TO GIVE PREV TAU AS INIT
    if index == 0:
        pass
    else:
        init_guess[1] = b_array_red[index-1]
        init_guess[4] = e_array_red[index-1]
        init_guessb[1] = b_array_blue[index-1]
        init_guessb[4] = e_array_blue[index-1]
     ##################################### THIS IS TO GIVE PREV TAU AS INIT
    
    print('bef fit')
    gc.collect()
    
    b,e,be,ee,b2,e2,be2,ee2 = calcdecay_subplot_nan(datared*hlp, time_detail= Time_bin*1.0e-9*fastfactor,titulo='',single=False,other_dset1=None,other_dset2=datablue*hlp,init_guess=init_guess,init_guess2=init_guessb,unit='kHz',error_array=None) #,error_array=error_arrayr, error_array2=error_arrayb)    
        
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
    major_ticks0 = [250,500,750,1000,1250]
    ax1.set_xticks(major_ticks0) 
    ax1.tick_params(labelsize=fsizepl)
    
    # Extra plots        
    aaa = datared*hlp
    xx_array = np.arange(0,aaa.shape[0])*Time_bin*1.0e-9*fastfactor
    #Plot whole of background decay
    #plt.semilogy(xx_array/1e-6,np.average(datared*hlpd,axis=(1,2)),'o',color='DarkRed',markersize=3,label='Transient CL from red background')   
    #Plot mean signal before e-beam on
    plt.semilogy(xx_array/1.0e-6,np.average(datared_init*hlp,axis=(0,1,2))*np.ones(len(xx_array)),'r--',lw=2,label='Mean CL from signal pixels, before e-beam')
    #Plot mean background
    #plt.semilogy(xx_array/1e-6,np.average(datared_init*hlpd,axis=(0,1,2))*np.ones(len(xx_array)),'--',color='DarkRed',lw=1,label='Mean CL from background, before e-beam')
    
    #Plot whole of background decay
    #plt.semilogy(xx_array/1e-6,np.average(datablue*hlpd,axis=(1,2)),'o',color='DarkGreen',markersize=3,label='Transient CL from blue background')   
    #Plot mean signal before e-beam on
    plt.semilogy(xx_array/1.0e-6,np.average(datablue_init*hlp,axis=(0,1,2))*np.ones(len(xx_array)),'g--',lw=2,label='Mean CL from blue signal pixels, before e-beam')
    #Plot mean background
    #plt.semilogy(xx_array/1e-6,np.average(datablue_init*hlpd,axis=(0,1,2))*np.ones(len(xx_array)),'--',color='DarkGreen',lw=1,label='Mean CL from background, before e-beam')
    
    del aaa, xx_array
    gc.collect()
    
    #plt.legend(loc='southwest')
        
    #ax1.set_xticks(major_ticks0)
        
    print('after plot 2') 
    
    ax1 = plt.subplot2grid((2,12), (1, 9), colspan=3)
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('right')
    ax1.yaxis.set_label_position("right")
    
    del datared, datablue, datablue_init, datared_init
    gc.collect()
    #if index is not 5:
    dataALLred = red['data'][:,:,:,:]
    dataALLblue = blue['data'][:,:,:,:]
    #else:
#        dataALLred = red['data'][0:3,:,:,:]
#        dataALLblue = blue['data'][0:3,:,:,:]
    nominal_time_on = 150.0
    
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)),c='r',linestyle='None', marker='o',markersize=4) #in mus, in MHz
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='r',  linewidth= lw) #in mus, in MHz
    #background intensity
    #plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3)),c='DarkRed', label='',linestyle='None', marker='o',markersize=3) #in mus, in MHz
    #plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='DarkRed',  linewidth= lw) #in mus, in MHz
    #init red intensity
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLred[:,0:backgdinit,:,:]*hlp,axis=(1,2,3)),c='r', linestyle='None', marker='o',markersize=3) #in mus, in MHz
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLred[:,0:backgdinit,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),'--',c='r',  linewidth= lw) #in mus, in MHz


    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)),c='g', linestyle='None', marker='o',markersize=4) #in mus, in MHz
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='g',  linewidth= lw) #in mus, in MHz
    #background intensity
    #plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3)),c='DarkGreen', label='',linestyle='None', marker='o',markersize=3) #in mus, in MHz
    #plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='DarkGreen',  linewidth= lw) #in mus, in MHz
    #init red intensity
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLblue[:,0:backgdinit,:,:]*hlp,axis=(1,2,3)),c='g', linestyle='None', marker='o',markersize=3) #in mus, in MHz
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLblue[:,0:backgdinit,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),'--',c='g',  linewidth= lw) #in mus, in MHz

    aa = np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3))/np.average(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3))
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3))/np.average(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)),c='k',linestyle='None', marker='s',markersize=3)# + str("{0:.2f}".format(aa[0])),linestyle='None', marker='s',markersize=6) #in mus, in MHz
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3))/np.average(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='k',  linewidth= lw,label='green/red signal (a.u.)' ) #in mus, in MHz
    #plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3))/np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)),c='k',linestyle='None', marker='s',markersize=4,label='green/red' )# + str("{0:.2f}".format(aa[0])),linestyle='None', marker='s',markersize=6) #in mus, in MHz
    
    #plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3))/np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3)),c='y',linestyle='None', marker='s',markersize=3)# + str("{0:.2f}".format(aa[0])),linestyle='None', marker='s',markersize=6) #in mus, in MHz
    #plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))/np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='y',  linewidth= lw,label='green/red backgd' ) #in mus, in MHz

    plt.legend(loc='best')
    
    print('after plot 3, bef ft') 
    
   
    red_int_array[index] = np.average(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) #np.nanmean(dataALLred*hlp,axis=(0,1,2,3))
    #cumu_red[index,:] =  np.nanmean(dataALLred*hlp,axis=(1,2,3))
    blue_int_array[index] = np.average(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) #np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))
    #cumu_blue[index,:] =  np.nanmean(dataALLred*hlp,axis=(1,2,3))
    ##### ABOVE, IMPORTANT: ADD 1mV, WHICH IS THE VOLTAGE AT 25c, RT
    
    print('here2')
    red_std_array[index] = np.std(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) #np.nanmean(dataALLred*hlp,axis=(0,1,2,3))
    blue_std_array[index] = np.std(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) #np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))
    
    plt.ylabel("Average luminescence while e-beam on,\n for each experiment and per signal pixel (kHz)",fontsize=fsizepl)
    plt.xlabel("Cumulative e-beam exposure time \n per pixel ($\mu$s)",fontsize=fsizepl)
    plt.ylim([0,8])
    
    #plt.xlim([nominal_time_on - 1.0,nominal_time_on*No_experiments[index]*fastfactor +1.0])
    plt.xlim([140,760])
    major_ticks0 = [150,300,450,600,750]
    ax1.set_xticks(major_ticks0) 
    ax1.tick_params(labelsize=fsizepl)
    
    del dataALLred, dataALLblue
    gc.collect()
    
    
    #sig_to_back[index] =   np.nanmean(dataALLred*hlp,axis=(0,1,2,3))/np.nanmean(dataALLred*hlpd,axis=(0,1,2,3))
    #sig_to_initred[index] =   np.nanmean(dataALLred*hlp,axis=(0,1,2,3))/np.nanmean(dataALLred[:,0:16,:,:]*hlp,axis=(0,1,2,3))
    
    
    #sig_to_back_blue[index] =   np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))/np.nanmean(dataALLblue*hlpd,axis=(0,1,2,3))
    #sig_to_initblue[index] =   np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))/np.nanmean(dataALLblue[:,0:16,:,:]*hlp,axis=(0,1,2,3))
    print('here3')
    multipage_longer('ZZZZSingle-'+ let[index] + prefix + '.pdf',dpi=80)
    

##### ONCE ALL FITS WORK, 
###### NEED TO RUN THESE LINES BELOW SO THAT ALL NPZ FILES ARE CREATED 

mycode = prefix +'Red_std_array = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Red_std_array', data = red_std_array)

mycode = prefix +'Blue_std_array = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Blue_std_array', data = blue_std_array)

mycode = prefix + 'Red_int_array = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Red_int_array', data = red_int_array)

mycode = prefix +'Blue_int_array = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Blue_int_array', data = blue_int_array)

#mycode = 'Cumu_red = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Cumu_red', data = cumu_red)

#mycode = 'Cumu_blue = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Cumu_blue', data = cumu_blue)

#il_data will be [6,5] in shape
mycode = prefix +'Il_data = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Il_data', data = il_data)

mycode =prefix + 'Il_data_std = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Il_data_std', data = il_data_std)

mycode =prefix + 'B_array_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'B_array_red', data = b_array_red)

mycode =prefix +'Be_array_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Be_array_red', data = be_array_red)

mycode = prefix +'E_array_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'E_array_red', data = e_array_red)

mycode = prefix +'Ee_array_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Ee_array_red', data = ee_array_red)

mycode = prefix +'B_array_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'B_array_blue', data = b_array_blue)

mycode =prefix +'Be_array_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Be_array_blue', data = be_array_blue)

mycode = prefix +'E_array_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'E_array_blue', data = e_array_blue)

mycode = prefix +'Ee_array_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Ee_array_blue', data = ee_array_blue)

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
