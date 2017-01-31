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
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
#from BackgroundCorrection import *
from TConversionThermocoupler import *
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
index = 0
if index is 0:
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


plt.show()

import pickle

pickle.dump( red['data'], open( "save.p", "wb" ) )


