#NDs 10kV 30mum 
#509nm dicroic, acquisition with red PMT
#lag 4mus
#moving 400ns
#excitation 2mus
#transient 1600ns
#clock 25MHZ --> 40ns time bins
#500x500 pixels
#se pics: 100mus lag per pixel, 500kHz clock 

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

import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile
#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################
#index = 0
calc_blue = True
#Excitation is impulsive, 120ns per pulse,turn on at point no 80, off at point no 82

#No_pixels = 250
Time_bin = 40#in ns; 1/clock of 25MHz 
nominal_time_on = 2.0 #time during which e-beam nominally on, in mus
totalpoints = 100 #total number of time-resolved points
### data
if calc_blue is False:
    pmt = ['PMT red']
    channel = ['1']
else:
    pmt = ['PMT blue']
    channel = ['2']
    
nametr = ['2016-12-16-1501_ImageSequence_NV140nmA_250.000kX_10.000kV_30mu_2',
         '2016-12-16-1515_ImageSequence_NV140nmB_250.000kX_10.000kV_30mu_2',
         '2016-12-16-1528_ImageSequence_NV140nmC_250.000kX_10.000kV_30mu_2',
         '2016-12-16-1540_ImageSequence_NV140nmD_250.000kX_10.000kV_30mu_2',
         '2016-12-16-1554_ImageSequence_NV140nmE_250.000kX_10.000kV_30mu_2',
         '2016-12-16-1616_ImageSequence_NV100nmA_250.000kX_10.000kV_30mu_4',
         '2016-12-16-1631_ImageSequence_NV100nmB_250.000kX_10.000kV_30mu_3',
         '2016-12-16-1646_ImageSequence_NV100nmC_250.000kX_10.000kV_30mu_3',
         '2016-12-16-1700_ImageSequence_NV100nmCII_1000.000kX_10.000kV_30mu_3',
         '2016-12-16-1712_ImageSequence_NV100nmCII_1000.000kX_10.000kV_30mu_5',
         '2016-12-16-1737_ImageSequence_NV100nmD_500.000kX_10.000kV_30mu_3',
         '2016-12-16-1758_ImageSequence_NV100nmE_1000.000kX_10.000kV_30mu_6',
         '2016-12-16-1810_ImageSequence_NV100nmE_500.000kX_10.000kV_30mu_9',
         '2016-12-16-2206_ImageSequence_NV40nmA_250.000kX_10.000kV_30mu_2',
         '2016-12-16-2228_ImageSequence_NV40nmB_250.000kX_10.000kV_30mu_7',
         '2016-12-16-2246_ImageSequence_NV40nmC_300.000kX_10.000kV_30mu_5',
         '2016-12-16-2307_ImageSequence_NV40nmD_500.000kX_10.000kV_30mu_5',
         '2016-12-16-2348_ImageSequence_NV40nmE_1000.000kX_10.000kV_30mu_9']
                  
No_experiments = 10.0*np.ones(len(nametr))

description = ['Adamas NDs'] # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
kv = [10]

let = ['A140','B140','C140','D140','E140','A100','B100','C100BBpb','C100IIBBpb','C100IIIBBpb','D100','E100','E100II','A40','B40','C40','D40','E40']

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
cumu_blue = np.zeros([len(nametr),10])
cumu_red = np.zeros([len(nametr),10])
il_data = np.zeros([len(nametr),10])
size_signal = np.zeros(len(nametr))
sig_to_back = np.zeros(len(nametr))
sig_to_initred = np.zeros(len(nametr))

pisize = [0.89, 0.89, 0.89,0.89, 0.89, 0.89,0.89,0.89,0.22,0.22,0.45,0.22,0.45,0.89,0.89,0.74,0.45,0.22] #in nm

listofindex =np.arange(0,len(nametr)) # #[0,1,2]  [5]  ###goes until 17
index = 1
if index is 17:
#for index in listofindex:
    
    file1    = h5py.File(nametr[index] + '.hdf5', 'r')  
    Pixel_size = pisize[index] 
    Ps = str("{0:.2f}".format(Pixel_size)) 

    se = np.load(str(let[index]) +'SEchannelONE.npz') 
    segmm = np.load(str(let[index]) +'SEchannelGMMONE.npz') 
    red = np.load(str(let[index]) +'Redbright.npz') 
    
    fsizetit = 18 
    fsizepl = 16 
    sizex = 8 
    sizey = 6
    dpi_no = 80
    lw = 2
    
    length_scalebar = 50.0 #in nm 
    scalebar_legend = '50nm'
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size))
    
    hlp = np.copy(segmm['bright'])
    hlp[~np.isnan(hlp)] = 1.0 
    size_signal[index] = len(hlp[~np.isnan(hlp)])*Pixel_size*Pixel_size #in nm^2 
    del hlp
    
    titulo = 'Nanodiamond ' + let[index] + ', of area ' +  str("{0:.2f}".format(size_signal[index]/1000.0)) + ' $\cdot$ 10$^3$nm$^2$ (10kV, 30$\mu$m aperture, 40ns time bins, ' + str(Ps)+ 'nm pixels, $>$ 509nm from NV0)'
    
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
    
    ax1 = plt.subplot2grid((2,3), (0, 0), colspan=1)
    ax1.set_title('SE channel',fontsize=fsizepl) #as per accompanying txt files
    plt.imshow(se['data'],cmap=cm.Greys_r)
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    gc.collect()
    ax1 = plt.subplot2grid((2,3), (0, 1), colspan=1)
    ax1.set_title('SE channel, signal pixels',fontsize=fsizepl)
    hlpse = np.copy(segmm['bright'])
    hlpse[~np.isnan(hlpse)] = 0.0 #inside
    hlpse[np.isnan(hlpse)] = 1.0 #outside
    im = plt.imshow(hlpse,cmap=cm.Greys) #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    ax1 = plt.subplot2grid((2,3), (0, 2), colspan=1)
    ax1.set_title('CL data averaged over time \n and 10 experiments',fontsize=fsizepl)
    im = plt.imshow(np.nanmean(red['data'],axis = (0,1)),cmap=cm.Reds) #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    ax1 = plt.subplot2grid((2,3), (1, 0), colspan=1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
       
    # INSIDE
    hlp = np.copy(segmm['bright'])
    hlp[~np.isnan(hlp)] = 1.0  #inside
    #hlp[np.isnan(hlp)] = 0.0 #outside
    
    # OUTSIDE
    hlpd = np.copy(segmm['bright'])
    hlpd[~np.isnan(hlpd)] = 0.0 
    hlpd[np.isnan(hlpd)] = 1.0
   
    plt.plot(np.arange(0,100)*Time_bin/1e3,np.nanmean(red['data']*hlp,axis = (0,2,3)),c='r',lw=2) #in mus, in MHz
    plt.plot(np.arange(0,100)*Time_bin/1e3,np.nanmean(red['data']*hlpd,axis = (0,2,3)),c='k',lw=2) #in mus, in MHz
    ax1.axvspan(0.4+0.3,2.4+0.3, alpha=0.25, color='yellow')
    unit = 'kHz'
    plt.ylabel("Average luminescence \n of each time bin, per signal pixel (" + unit + ")",fontsize=fsizepl)
    plt.xlabel("Behaviour of e-beam during each experiment: \n 2-ON + 1.3-OFF ($\mu$s)",fontsize=fsizepl)
    major_ticks0 = [1,2,3,4]
    ax1.set_xticks(major_ticks0) 
    #ax1.set_yticks([15,30,45]) 
    plt.xlim([0,4])
    
    ax1 = plt.subplot2grid((2,3), (1, 1), colspan=1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    datared = np.average(red['data'], axis = (0))
    
    if (index == 4) or (index==5) or (index==7)or (index==8)or (index==9)or (index==11)or (index==12):
        initbin = 69-1
        datared_init = datared[0:17-1,:,:]
        datared = datared[initbin:,:,:]
    else:
        initbin = 68-1
        ### 700ns /40ns = 7. ....
        datared_init = datared[0:17-1,:,:]
        datared = datared[initbin:,:,:]

    fastfactor = 1
    last_pt_offset = -10 #sometimes use -1, last point, but sometimes this gives 0. -10 seems to work
    medium_pt_offset = -20

    cinit = np.nanmean(datared_init*hlp,axis=(0,1,2))
    
    #init_guess = [np.average(datared[0,:,:]), 0.05, cinit, np.average(datared[medium_pt_offset:,:,:]), 0.33] #e init was 0.5
   
    hlp_red = datared * hlp
   
    hlpred = np.nanmean(red['data'][:,initbin:,:,:]*hlp,axis = (2,3))
    
    #error_arrayr = np.nanstd(hlpred,axis = 0)
    #error_arrayr[error_arrayr < 0.05 * np.max(hlpred)]  = 0.05 * np.max(hlpred) #arbitrary 5% of max  
    
    cut_long = 6
    cut_short = 4
    init_guess = calc_double_fit(np.arange(0,datared.shape[0])*Time_bin*1e-9*fastfactor,np.nanmean(datared*hlp,axis=(1,2)),dt= Time_bin*1e-9*fastfactor,cut_long = 6,cut_short = 4)
    
    #replace c with cinit
    init_guess[2] = cinit 
    
    b,e,be,ee,b2,e2,be2,ee2 = calcdecay_subplot_nan(datared*hlp, time_detail= Time_bin*1e-9*fastfactor,titulo='',single=False,other_dset1=None,other_dset2=None,init_guess=init_guess,unit='kHz',error_array=None) #,error_array=error_arrayr, error_array2=error_arrayb)    
        
    b_array_red[index] = b
    e_array_red[index] = e
    be_array_red[index] = be    
    ee_array_red[index] = ee  
 
    plt.ylabel("Average luminescence \n of each time bin, per signal pixel (" + unit + ")",fontsize=fsizepl)
    plt.xlabel(r'Time after blanking the electron beam ($\mu$s)',  fontsize=fsizepl)
    plt.xlim(xmax=1.3)
    
    # Extra plots        
    aaa = datared*hlp
    xx_array = np.arange(0,aaa.shape[0])*Time_bin*1e-9*fastfactor
    #Plot whole of background decay
    plt.semilogy(xx_array/1e-6,np.nanmean(datared*hlpd,axis=(1,2)),'ko',markersize=2,label='Transient CL from background')   
    #Plot mean signal before e-beam on
    plt.semilogy(xx_array/1e-6,np.nanmean(datared_init*hlp,axis=(0,1,2))*np.ones(len(xx_array)),'r--',lw=2,label='Mean CL from signal pixels, before e-beam')
    #Plot mean background
    plt.semilogy(xx_array/1e-6,np.nanmean(datared_init*hlpd,axis=(0,1,2))*np.ones(len(xx_array)),'k--',lw=1,label='Mean CL from background, before e-beam')
    
    plt.legend(loc='best')
        
    #ax1.set_xticks(major_ticks0)
    
    ax1 = plt.subplot2grid((2,3), (1, 2), colspan=1)
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('right')
    ax1.yaxis.set_label_position("right")
    
    dataALLred = red['data'][:,:,:,:]
    nominal_time_on = 2.0
    
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred*hlp,axis=(1,2,3)),c='r', label='From photons $>$ 509nm',linestyle='None', marker='o',markersize=4) #in mus, in MHz
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='r',  linewidth= lw) #in mus, in MHz
    #background intensity
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred*hlpd,axis=(1,2,3)),c='k', label='',linestyle='None', marker='o',markersize=2) #in mus, in MHz
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred*hlpd,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='k',  linewidth= lw) #in mus, in MHz
    #init red intensity
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred[:,0:16,:,:]*hlp,axis=(1,2,3)),c='r', label='',linestyle='None', marker='o',markersize=2) #in mus, in MHz
    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred[:,0:16,:,:]*hlp,axis=(0,1,2,3))*np.ones(dataALLred.shape[0]),c='r',  linewidth= lw) #in mus, in MHz


    red_int_array[index] = np.nanmean(dataALLred*hlp,axis=(0,1,2,3))
    cumu_red[index,:] =  np.nanmean(dataALLred*hlp,axis=(1,2,3))
    
    plt.ylabel("Average luminescence \n for each experiment, per signal pixel  (kHz)",fontsize=fsizepl)
    plt.xlabel("Cumulative e-beam exposure time \n per pixel (nominal, $\mu$s)",fontsize=fsizepl)
    
    plt.xlim([nominal_time_on - 1.0,nominal_time_on*No_experiments[index]*fastfactor +1.0])
    
    sig_to_back[index] =   np.nanmean(dataALLred*hlp,axis=(0,1,2,3))/np.nanmean(dataALLred*hlpd,axis=(0,1,2,3))
    sig_to_initred[index] =   np.nanmean(dataALLred*hlp,axis=(0,1,2,3))/np.nanmean(dataALLred[:,0:16,:,:]*hlp,axis=(0,1,2,3))
    
    multipage_longer('ZZZZ-'+ let[index] + '.pdf',dpi=80)

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
#mycode = 'Cumu_red = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Cumu_red', data = cumu_red)
#
#mycode = 'Cumu_blue = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Cumu_blue', data = cumu_blue)
#
#mycode = 'Il_data = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Il_data', data = il_data)
#
#mycode = 'B_array_red = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('B_array_red', data = b_array_red)
#
#mycode ='Be_array_red = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Be_array_red', data = be_array_red)
#
#mycode = 'E_array_red = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('E_array_red', data = e_array_red)
#
#mycode = 'Ee_array_red = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Ee_array_red', data = ee_array_red)
#
#mycode = 'B_array_blue = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('B_array_blue', data = b_array_blue)
#
#mycode ='Be_array_blue = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Be_array_blue', data = be_array_blue)
#
#mycode = 'E_array_blue = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('E_array_blue', data = e_array_blue)
#
#mycode = 'Ee_array_blue = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('Ee_array_blue', data = ee_array_blue)
#
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
