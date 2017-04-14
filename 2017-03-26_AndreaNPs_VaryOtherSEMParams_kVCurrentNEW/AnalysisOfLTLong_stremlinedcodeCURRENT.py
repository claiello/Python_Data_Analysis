#==============================================================================
# 1.6ms, 50 move, 150 excite, 1400 transient
# 1MHz clock rate (1mus timebins)
# 5 frames
# 150kX mag
# standard: 300pixels
# 10kV
# 30mum == 379pA
#with filter: 592 dicrhoic + 550/***32***nm in blue pmt + 650/54nm in red pmt, semrock brightline
#==============================================================================

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
#######

Time_bin = 1000#in ns; 
nominal_time_on = 150.0 #time during which e-beam nominally on, in mus
totaltrpoints = 1400 #total number of time-resolved points

prefix =  'current'

No_experiments = 5*np.ones([4])
                
nametr = ['2017-03-24-1827_ImageSequence__1.434kX_10.000kV_60mu_12',
          '2017-03-24-1855_ImageSequence__150.000kX_10.000kV_60mu_13',
          '2017-03-24-1925_ImageSequence__0.898kX_10.000kV_12mu_14',
          '2017-03-24-1946_ImageSequence__150.000kX_10.000kV_12mu_15']

description = ['Andrea small NaYF4:Er'] # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
kv = [10]

#nominal Temps
let = ['ap30','ap60','ap10','ap120'] #aperture
Current = [379, 1800, 28,5700] #pA


######################################## Plot with dose for different apertures
##files below exist 

blue_int_array = np.zeros(len(nametr))
red_int_array = np.zeros(len(nametr))
blue_std_array = np.zeros(len(nametr))
red_std_array = np.zeros(len(nametr))

red_decay_array = np.zeros([len(nametr),1398])
blue_decay_array = np.zeros([len(nametr),1398])

listofindex =np.arange(0,len(nametr))#,11]


consider_whole_light = [0,1,2,3,4]

#plt.figure(1)

for index in listofindex:
    
    print(index)
   
    red0 = np.load(str(let[index]) +'Redbright.npz',mmap_mode='r')  
    blue0 = np.load(str(let[index]) +'Bluebright.npz',mmap_mode='r')  
    se = np.load(str(let[index]) +'SEchannel.npz',mmap_mode='r') 
    segmm = np.load(str(let[index]) +'SEchannelGMM.npz',mmap_mode='r')  
    
    red = red0['data']
    blue = blue0['data']
    del red0, blue0
    gc.collect()
    
    ##############################################################
    
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
    
    #ax1 = plt.subplot2grid((1,12), (0,index), colspan=3)
   # im = plt.imshow(se['data'],cmap=cm.Greys) #or 'OrRd'
   # print('after imshow')
#    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
#    ax1.add_artist(sbar)
#    plt.axis('off')
#    unit = '(kHz)'
#    box = ax1.get_position()
#    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
#    axColor = plt.axes([box.x0, box.y0*1.075 , box.width,0.01 ])    
#    print(np.average(blue[:,backgdinit:initbin,:,:],axis = (0,1,2,3)))
#    tickval = np.average(blue[:,backgdinit:initbin,:,:],axis = (0,1,2,3))
#    cb2 = plt.colorbar(im, cax = axColor, orientation="horizontal",ticks=[0.5*tickval,tickval,2.0*tickval],format='%0.2f' ) #,np.nanmean(blue['data'],axis = (0,1,2,3))+np.nanstd(blue['data'],axis = (0,1,2,3)),np.nanmean(blue['data'],axis = (0,1,2,3))+3.0*np.nanstd(blue['data'],axis = (0,1,2,3))] )#/2.0,np.nanmax(blue['data'],axis = (0,1,2,3))]) 
#    print('after colorbar')
#    cb2.ax.tick_params(labelsize=fsizepl)
#    cb2.set_label('Photon counts (kHz)', size = fsizepl)
    
   # ax1 = plt.subplot2grid((2,12), (1, index), colspan=3)
    #ax1.set_title('CL data while e-beam on,\n averaged over 3 experiments',fontsize=fsizepl)
   # imb = plt.imshow(np.average(red[:,backgdinit:initbin,:,:],axis = (0,1)),cmap=cm.Reds) #or 'OrRd'
#    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
#    ax1.add_artist(sbar)
#    plt.axis('off')
#    
#    box = ax1.get_position()
#    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
#    axColor = plt.axes([box.x0, box.y0*1.075 , box.width,0.01 ])    
#    tickval =  np.average(red[:,backgdinit:initbin,:,:],axis = (0,1,2,3))
#    cb1 = plt.colorbar(imb, cax = axColor, orientation="horizontal",ticks=[0.5*tickval,tickval,2.0*tickval],format='%0.2f') 
#    cb1.ax.tick_params(labelsize=fsizepl)
#    cb1.set_label('Photon counts (kHz)', size = fsizepl)
#    
#    ax1 = plt.subplot2grid((2,12), (1, 0), colspan=3)
#    ax1.spines['right'].set_visible(False)
#    ax1.spines['top'].set_visible(False)
#    ax1.xaxis.set_ticks_position('bottom')
#    ax1.yaxis.set_ticks_position('left')
     
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
        
#    gc.collect()    
#    print('bef plotting of plot1')
#    plt.plot(np.arange(0,red.shape[1])*Time_bin/1.0e3,np.average(red*hlp,axis = (0,2,3)),c='r',lw=2) #in mus, in MHz
#    print('after 1st plot statement of plot1')
#    #plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1e3,np.average(red['data']*hlpd,axis = (0,2,3)),c='DarkRed',lw=2) #in mus, in MHz
#    plt.plot(np.arange(0,red.shape[1])*Time_bin/1.0e3,np.average(blue*hlp,axis = (0,2,3)),c='g',lw=2) #in mus, in MHz
#    #plt.plot(np.arange(0,red['data'].shape[1])*Time_bin/1e3,np.average(blue['data']*hlpd,axis = (0,2,3)),c='DarkGreen',lw=2) #in mus, in MHz
#    print('after plotting of plot1')
#    ax1.axvspan(50.0,50.0+150.0, alpha=0.25, color='yellow')
#    unit = 'kHz'
#    plt.ylabel("Average luminescence,\n per signal pixel (" + unit + ")",fontsize=fsizepl)
#    plt.xlabel("Time as electron beam on/off ($\mu$s)",fontsize=fsizepl)
#    major_ticks0 = [500,1000,1500]
#    ax1.set_xticks(major_ticks0) 
#    #ax1.set_yticks([15,30,45]) 
#    plt.ylim([0,10])
#    ax1.tick_params(labelsize=fsizepl)
#    
#    print('after plot 1') 
#    
#    ax1 = plt.subplot2grid((2,12), (1, 4), colspan=4)
#    ax1.spines['right'].set_visible(False)
#    ax1.spines['top'].set_visible(False)
#    ax1.xaxis.set_ticks_position('bottom')
#    ax1.yaxis.set_ticks_position('left')
    
    datared = np.average(red, axis = (0))
    datablue = np.average(blue, axis = (0))
    
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
    
#    cut_longa = 6#8#6
#    cut_shorta = 3#5#3
#    
#    print('bef init')
#    gc.collect()
#    
#    init_guess = calc_double_fit(np.arange(0,datared.shape[0])*Time_bin*1.0e-9*fastfactor,np.nanmean(datared*hlp,axis=(1,2)),dt= Time_bin*1.0e-9*fastfactor,cut_long=cut_longa,cut_short=cut_shorta)
#    
#    gc.collect()    
#    
#    init_guessb = calc_double_fit(np.arange(0,datablue.shape[0])*Time_bin*1.0e-9*fastfactor,np.nanmean(datablue*hlp,axis=(1,2)),dt= Time_bin*1.0e-9*fastfactor,cut_long=cut_longa,cut_short=cut_shorta)
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
    
#    ##################################### THIS IS TO GIVE PREV TAU AS INIT
#    if index == 1:
#        pass
#    else:
#        init_guess[1] = b_array_red[index-1]
#        init_guess[4] = e_array_red[index-1]
#        init_guessb[1] = b_array_blue[index-1]
#        init_guessb[4] = e_array_blue[index-1]
#     ##################################### THIS IS TO GIVE PREV TAU AS INIT
    
#    print('bef fit')
#    gc.collect()
#    
#    b,e,be,ee,b2,e2,be2,ee2 = calcdecay_subplot_nan(datared*hlp, time_detail= Time_bin*1.0e-9*fastfactor,titulo='',single=False,other_dset1=None,other_dset2=datablue*hlp,init_guess=init_guess,init_guess2=init_guessb,unit='kHz',error_array=None) #,error_array=error_arrayr, error_array2=error_arrayb)    
#        
#    b_array_red[index] = b
#    e_array_red[index] = e
#    be_array_red[index] = be    
#    ee_array_red[index] = ee  
#    b_array_blue[index] = b2
#    e_array_blue[index] = e2
#    be_array_blue[index] = be2    
#    ee_array_blue[index] = ee2  
# 
#    plt.ylabel("Average luminescence,\n per signal pixel (" + unit + ")",fontsize=fsizepl)
#    plt.xlabel(r'Time after blanking the electron beam ($\mu$s)',  fontsize=fsizepl)
#    plt.xlim(xmax=1400.0) #1400
#    major_ticks0 = [250,500,750,1000,1250]
#    ax1.set_xticks(major_ticks0) 
#    ax1.tick_params(labelsize=fsizepl)
    
#    # Extra plots        
#    aaa = datared*hlp
#    xx_array = np.arange(0,aaa.shape[0])*Time_bin*1.0e-9*fastfactor
#    #Plot whole of background decay
#    #plt.semilogy(xx_array/1e-6,np.average(datared*hlpd,axis=(1,2)),'o',color='DarkRed',markersize=3,label='Transient CL from red background')   
#    #Plot mean signal before e-beam on
#    plt.semilogy(xx_array/1.0e-6,np.average(datared_init*hlp,axis=(0,1,2))*np.ones(len(xx_array)),'r--',lw=2,label='Mean CL from signal pixels, before e-beam')
#    #Plot mean background
#    #plt.semilogy(xx_array/1e-6,np.average(datared_init*hlpd,axis=(0,1,2))*np.ones(len(xx_array)),'--',color='DarkRed',lw=1,label='Mean CL from background, before e-beam')
#    
#    #Plot whole of background decay
#    #plt.semilogy(xx_array/1e-6,np.average(datablue*hlpd,axis=(1,2)),'o',color='DarkGreen',markersize=3,label='Transient CL from blue background')   
#    #Plot mean signal before e-beam on
#    plt.semilogy(xx_array/1.0e-6,np.average(datablue_init*hlp,axis=(0,1,2))*np.ones(len(xx_array)),'g--',lw=2,label='Mean CL from blue signal pixels, before e-beam')
#    #Plot mean background
#    #plt.semilogy(xx_array/1e-6,np.average(datablue_init*hlpd,axis=(0,1,2))*np.ones(len(xx_array)),'--',color='DarkGreen',lw=1,label='Mean CL from background, before e-beam')
#    
#    del aaa, xx_array
#    gc.collect()
    
    #plt.legend(loc='southwest')
        
    #ax1.set_xticks(major_ticks0)
        
#    print('after plot 2') 
#    
#    ax1 = plt.subplot2grid((2,12), (1, 9), colspan=3)
#    ax1.spines['left'].set_visible(False)
#    ax1.spines['top'].set_visible(False)
#    ax1.xaxis.set_ticks_position('bottom')
#    ax1.yaxis.set_ticks_position('right')
#    ax1.yaxis.set_label_position("right")
    
    del datared, datablue, datablue_init, datared_init
    gc.collect()
    #if index is not 5:
    dataALLred = red[:,:,:,:]
    dataALLblue = blue[:,:,:,:]
    #else:
#        dataALLred = red['data'][0:3,:,:,:]
#        dataALLblue = blue['data'][0:3,:,:,:]
    #nominal_time_on = 150.0
    
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)),c='r',linestyle='None', marker='o',markersize=4) #in mus, in MHz
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='r',  linewidth= lw) #in mus, in MHz
#    #background intensity
#    #plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3)),c='DarkRed', label='',linestyle='None', marker='o',markersize=3) #in mus, in MHz
#    #plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='DarkRed',  linewidth= lw) #in mus, in MHz
#    #init red intensity
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLred[:,0:backgdinit,:,:]*hlp,axis=(1,2,3)),c='r', linestyle='None', marker='o',markersize=3) #in mus, in MHz
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLred[:,0:backgdinit,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),'--',c='r',  linewidth= lw) #in mus, in MHz
#
#
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)),c='g', linestyle='None', marker='o',markersize=4) #in mus, in MHz
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='g',  linewidth= lw) #in mus, in MHz
#    #background intensity
#    #plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3)),c='DarkGreen', label='',linestyle='None', marker='o',markersize=3) #in mus, in MHz
#    #plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='DarkGreen',  linewidth= lw) #in mus, in MHz
#    #init red intensity
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLblue[:,0:backgdinit,:,:]*hlp,axis=(1,2,3)),c='g', linestyle='None', marker='o',markersize=3) #in mus, in MHz
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLblue[:,0:backgdinit,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),'--',c='g',  linewidth= lw) #in mus, in MHz
#
#    aa = np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3))/np.average(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3))
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3))/np.average(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)),c='k',linestyle='None', marker='s',markersize=3)# + str("{0:.2f}".format(aa[0])),linestyle='None', marker='s',markersize=6) #in mus, in MHz
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.average(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3))/np.average(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='k',  linewidth= lw,label='green/red signal (a.u.)' ) #in mus, in MHz
#    #plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3))/np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(1,2,3)),c='k',linestyle='None', marker='s',markersize=4,label='green/red' )# + str("{0:.2f}".format(aa[0])),linestyle='None', marker='s',markersize=6) #in mus, in MHz
#    
#    #plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3))/np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(1,2,3)),c='y',linestyle='None', marker='s',markersize=3)# + str("{0:.2f}".format(aa[0])),linestyle='None', marker='s',markersize=6) #in mus, in MHz
#    #plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))/np.nanmean(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='y',  linewidth= lw,label='green/red backgd' ) #in mus, in MHz
#
#    plt.legend(loc='best')
#    
#    print('after plot 3, bef ft') 
    
   
    red_int_array[index] = np.average(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) #np.nanmean(dataALLred*hlp,axis=(0,1,2,3))
    #cumu_red[index,:] =  np.nanmean(dataALLred*hlp,axis=(1,2,3))
    blue_int_array[index] = np.average(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) #np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))
    #cumu_blue[index,:] =  np.nanmean(dataALLred*hlp,axis=(1,2,3))
    ##### ABOVE, IMPORTANT: ADD 1mV, WHICH IS THE VOLTAGE AT 25c, RT
    
    red_decay_array[index,:] = np.average(dataALLred[:,initbin:,:,:]*hlp,axis=(0,2,3))
    blue_decay_array[index,:] = np.average(dataALLblue[:,initbin:,:,:]*hlp,axis=(0,2,3))
    
    print('here2')
    red_std_array[index] = np.std(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) #np.nanmean(dataALLred*hlp,axis=(0,1,2,3))
    blue_std_array[index] = np.std(dataALLblue[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) #np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))
    
#    plt.ylabel("Average luminescence while e-beam on,\n for each experiment and per signal pixel (kHz)",fontsize=fsizepl)
#    plt.xlabel("Cumulative e-beam exposure time \n per pixel ($\mu$s)",fontsize=fsizepl)
#    plt.ylim([0,8])
#    
#    #plt.xlim([nominal_time_on - 1.0,nominal_time_on*No_experiments[index]*fastfactor +1.0])
#    plt.xlim([140,760])
#    major_ticks0 = [150,300,450,600,750]
#    ax1.set_xticks(major_ticks0) 
#    ax1.tick_params(labelsize=fsizepl)
    
    del dataALLred, dataALLblue
    gc.collect()
    
    
    #sig_to_back[index] =   np.nanmean(dataALLred*hlp,axis=(0,1,2,3))/np.nanmean(dataALLred*hlpd,axis=(0,1,2,3))
    #sig_to_initred[index] =   np.nanmean(dataALLred*hlp,axis=(0,1,2,3))/np.nanmean(dataALLred[:,0:16,:,:]*hlp,axis=(0,1,2,3))
    
    
    #sig_to_back_blue[index] =   np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))/np.nanmean(dataALLblue*hlpd,axis=(0,1,2,3))
    #sig_to_initblue[index] =   np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))/np.nanmean(dataALLblue[:,0:16,:,:]*hlp,axis=(0,1,2,3))
#    print('here3')
#    if len(prefix) == 0:
#        multipage_longer('ZZZZSingle-'+ let[index] +  '.pdf',dpi=80) 
#    else:
#        multipage_longer('ZZZZSingle-'+ let[index] + prefix + '.pdf',dpi=80)

##### ONCE ALL FITS WORK, 
###### NEED TO RUN THESE LINES BELOW SO THAT ALL NPZ FILES ARE CREATED 
 
mycode = prefix +'Red_std_array = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Red_std_array', data = red_std_array)

mycode = prefix +'Blue_std_array = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Blue_std_array', data = blue_std_array)

mycode = prefix +'Red_int_array = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Red_int_array', data = red_int_array)

mycode =prefix +'Blue_int_array = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Blue_int_array', data = blue_int_array)

mycode = prefix +'Red_decay_array = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Red_decay_array', data = red_decay_array)

mycode = prefix +'Blue_decay_array = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefix +'Blue_decay_array', data = blue_decay_array)
    
#####FIG WITH SUMMARY   
##### WHEN FILES ABOVE ARE CREATED, CREATE FIGURE BELOW WITH REGION SIZES VS LIFETIMES AND REGION SIZES VS S/N OR BRIGHTNESS OR <SIGNAL INT>/<BACKGROUND INT>
klklklklk
