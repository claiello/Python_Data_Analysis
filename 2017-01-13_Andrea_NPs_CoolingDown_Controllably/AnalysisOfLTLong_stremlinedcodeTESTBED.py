#nominally
#50mus moving
#150mus on
#1400mus off
#clock 1MHz = 1mus time bon
#150kX
#300x300 pixels
#4 frames per temperature (ie, per voltage )

##### Temperatures read by the multimeter just before acquisition of chosenexp = 0 were:
###RT (no voltage)
#102.5
#92.9
#66.4
#72.8
#60.6
#48.5
#39.8
#31.2

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
#######
MIND THE PREFIX11111111111111111111111111111111111111111111111111111111111111

#######
prefix = 'varCprevtau'

Time_bin = 1000#in ns; 
nominal_time_on = 150.0 #time during which e-beam nominally on, in mus
totaltrpoints = 1400 #total number of time-resolved points

nametr = ['2017-01-13-1730_ImageSequence__150.000kX_10.000kV_30mu_15',
          '2017-01-13-1810_ImageSequence__150.000kX_10.000kV_30mu_17',
          '2017-01-13-1935_ImageSequence__150.000kX_10.000kV_30mu_3',
          '2017-01-13-2011_ImageSequence__150.000kX_10.000kV_30mu_4',
          '2017-01-13-2050_ImageSequence__150.000kX_10.000kV_30mu_7',
          '2017-01-13-2132_ImageSequence__150.000kX_10.000kV_30mu_8',
          '2017-01-13-2213_ImageSequence__150.000kX_10.000kV_30mu_9',
          '2017-01-13-2251_ImageSequence__150.000kX_10.000kV_30mu_10']


Pixel_size = 2.5 #nm
Ps = Pixel_size #pixel size in nm, the numbers above with round nm precision
No_experiments = [4,4,4,4,4,4,4,4]

description = 'Andrea small NaYF4:Er'    # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
kv = [10]

let = ['N102pt5', 'N92pt9', 'N66pt4', 'N72pt8', 'N60pt6', 'N48pt5', 'N39pt8', 'N31pt2']


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

listofindex =np.arange(1,len(nametr))#,11]

temp = [102.5, 92.9, 66.4, 72.8, 60.6, 48.5, 39.8, 31.2]

consider_whole_light = [0,1,2,3,4,5,6,7]
#4,5,7 segmentation ok-ish
#index =2
#if index is 2:
for index in listofindex:
    
    print(index)
    
    #IL
    il = np.load(str(let[index]) +'ILchannel.npz') 
#    ###IL
#    if index == 7:
#        il_data[index] = 31.2
#    else:
#        aparam = 0.233801
#        bparam = 0.000144
#        delta = aparam*np.average(il['data'], axis = (0,1,2)) + bparam
#        il_data[index] = KThermocouplerconversion(np.average(il['data'], axis = (0,1,2)) + 1.0e-3 + delta)
        
    
#    if index == 0:
#       deltav2 = +0.240e-3  # KThermocouplerconversion(1.0e-3) #take average per frame 
#    if index == 1:
#        deltav2 = +0.229e-3
#    if index == 2:
#        deltav2 = +0.100e-3
#    if index == 3:
#        deltav2 = +0.225e-3
#    if index == 4:
#        deltav2 = +0.359e-3
#    if index == 5:
#        deltav2 = +0.505e-3
#    if index == 6:
#       deltav2 = +0.660e-3
#    
#    
#    hulper = np.average(il['data'], axis = (1,2))   
#    il_data[index] = KThermocouplerconversion(hulper[0] + 1.0e-3 + deltav2) #take average per frame   
     
    il_data[index] = temp[index] 
    #print(hulper[0])
    #print(np.average(il['data'], axis = (0,1,2)))
    #print(il_data)
    
    
    
    
    ############################## FFT to cut noise in Temperature/IL data
    #result = []
    #print(dataALLred.shape[1])
    total_time = totaltrpoints*Time_bin/1000.0 # in us
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
    
    print('before loading')
    
#    red = np.load(str(let[index]) +'Redbright.npz',mmap_mode='r') 
#    blue = np.load(str(let[index]) +'Bluebright.npz',mmap_mode='r') 
    
    red1 = np.load(str(let[index]) +'Redbright1.npz',mmap_mode='r') 
    red2 = np.load(str(let[index]) +'Redbright2.npz',mmap_mode='r') 
    red = np.concatenate((np.reshape(np.array(red1['data'],dtype=np.float32), [1,red1['data'].shape[0],red1['data'].shape[1],red1['data'].shape[2] ]), np.reshape(np.array(red2['data'],dtype=np.float32), [1,red1['data'].shape[0],red1['data'].shape[1],red1['data'].shape[2] ])), axis = 0)
    del red1, red2
    gc.collect()
    red3 = np.load(str(let[index]) +'Redbright3.npz',mmap_mode='r') 
    red = np.concatenate((red, np.reshape(np.array(red3['data'],dtype=np.float32), [1,red.shape[1],red.shape[2],red.shape[3]])), axis = 0)  
    del red3 
    gc.collect()
    red4 = np.load(str(let[index]) +'Redbright4.npz',mmap_mode='r') 
    red = np.concatenate((red, np.reshape(np.array(red4['data'],dtype=np.float32), [1,red.shape[1],red.shape[2],red.shape[3]])), axis = 0)  
    del red4
    gc.collect()
    
    blue1 = np.load(str(let[index]) +'Bluebright1.npz',mmap_mode='r') 
    blue2 = np.load(str(let[index]) +'Bluebright2.npz',mmap_mode='r') 
    blue= np.concatenate((np.reshape(np.array(blue1['data'],dtype=np.float32), [1,blue1['data'].shape[0],blue1['data'].shape[1],blue1['data'].shape[2] ]), np.reshape(np.array(blue2['data'],dtype=np.float32), [1,blue1['data'].shape[0],blue1['data'].shape[1],blue1['data'].shape[2] ])), axis = 0)
    del blue1, blue2
    gc.collect()
    blue3 = np.load(str(let[index]) +'Bluebright3.npz',mmap_mode='r') 
    blue = np.concatenate((blue, np.reshape(np.array(blue3['data'],dtype=np.float32), [1,red.shape[1],red.shape[2],red.shape[3]])), axis = 0)  
    del blue3 
    gc.collect()
    blue4 = np.load(str(let[index]) +'Bluebright4.npz',mmap_mode='r') 
    blue = np.concatenate((blue, np.reshape(np.array(blue4['data'],dtype=np.float32),[1,red.shape[1],red.shape[2],red.shape[3]])), axis = 0)  
    del blue4
    gc.collect()
    
    se = np.load(str(let[index]) +'SEchannel.npz',mmap_mode='r') 
    segmm = np.load(str(let[index]) +'SEchannelGMM.npz',mmap_mode='r')     
    
    ##############################################################
    print('after loading')
    
    fsizetit = 18 
    fsizepl = 16 
    sizex = 8 
    sizey = 6
    dpi_no = 80
    lw = 2
    
    length_scalebar = 50.0 #in nm 
    scalebar_legend = '50nm'
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size))
    
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
    
    del hlpse, se  #, segmm
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
    ax1.set_title('CL data while e-beam on,\n averaged over 5 experiments',fontsize=fsizepl)
    im = plt.imshow(np.average(blue[:,backgdinit:initbin,:,:],axis = (0,1)),cmap=cm.Greens) #or 'OrRd'
    #im = plt.imshow(img_adapteqb,cmap=cm.Greens) #or 'OrRd'
    print('after imshow')
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    unit = '(kHz)'
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor = plt.axes([box.x0, box.y0*1.075 , box.width,0.01 ])    
    print(np.average(blue[:,backgdinit:initbin,:,:],axis = (0,1,2,3)))
    tickval = np.average(blue[:,backgdinit:initbin,:,:],axis = (0,1,2,3))
    cb2 = plt.colorbar(im, cax = axColor, orientation="horizontal",ticks=[0.5*tickval,tickval,2.0*tickval],format='%0.2f' ) #,np.nanmean(blue['data'],axis = (0,1,2,3))+np.nanstd(blue['data'],axis = (0,1,2,3)),np.nanmean(blue['data'],axis = (0,1,2,3))+3.0*np.nanstd(blue['data'],axis = (0,1,2,3))] )#/2.0,np.nanmax(blue['data'],axis = (0,1,2,3))]) 
    #cb2 = plt.colorbar(im, cax = axColor, orientation="horizontal",ticks=[0.1*np.average(blue['data'][:,backgdinit:initbin,:,:],axis = (0,1,2,3)),np.average(blue['data'][:,backgdinit:initbin,:,:],axis = (0,1,2,3)), 10.0*np.average(blue['data'][:,backgdinit:initbin,:,:],axis = (0,1,2,3))] ) #,np.nanmean(blue['data'],axis = (0,1,2,3))+np.nanstd(blue['data'],axis = (0,1,2,3)),np.nanmean(blue['data'],axis = (0,1,2,3))+3.0*np.nanstd(blue['data'],axis = (0,1,2,3))] )#/2.0,np.nanmax(blue['data'],axis = (0,1,2,3))]) 
    print('after colorbar')
    cb2.ax.tick_params(labelsize=fsizepl)
    cb2.set_label('Photon counts (kHz)', size = fsizepl)
    
    ax1 = plt.subplot2grid((2,12), (0, 9), colspan=3)
    ax1.set_title('CL data while e-beam on,\n averaged over 5 experiments',fontsize=fsizepl)
    imb = plt.imshow(np.average(red[:,backgdinit:initbin,:,:],axis = (0,1)),cmap=cm.Reds) #or 'OrRd'
    #imb = plt.imshow(img_adapteqr,cmap=cm.Reds) #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor = plt.axes([box.x0, box.y0*1.075 , box.width,0.01 ])    
    #cb1 = plt.colorbar(imb, cax = axColor, orientation="horizontal",label='Photon counts ' + unit) 
    tickval =  np.average(red[:,backgdinit:initbin,:,:],axis = (0,1,2,3))
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
  
    multipage_longer('ZZZZSingle-'+ let[index] + prefix + '.pdf',dpi=80)

