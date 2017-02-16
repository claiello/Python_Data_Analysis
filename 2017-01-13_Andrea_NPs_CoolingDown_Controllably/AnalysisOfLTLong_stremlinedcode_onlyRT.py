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

consider_whole_light = [] #[0,1,2,3,4,5,6,7]
#4,5,7 segmentation ok-ish
#index =2
#if index is 2:
for index in [7]: #listofindex:
       
    ##############################################################
    
    fig40= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig40.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}') #not working
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}')  #not working
    plt.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]  #not working
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')     
    
    fsizetit = 18 
    fsizepl = 16 
    sizex = 8 
    sizey = 6
    dpi_no = 80
    lw = 2
       
    leto = ['N102pt5', 'N92pt9', 'N66pt4', 'N72pt8', 'N60pt6', 'N48pt5', 'N39pt8', 'N31pt2']
    se30 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[7] +'SEchannel.npz',mmap_mode='r')
    segmm = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[7] +'SEchannelGMM.npz',mmap_mode='r')
    xlen = se30['data'].shape[0]
    ylen = se30['data'].shape[1]
    delx = 0#+28
    dely = 0#00
    xval = 144
    yval = 120
    arr_img30 = se30['data'][np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely]    
     
    ax1 = plt.subplot2grid((1,3), (0, 0), colspan=3)
    ax1.set_title('SE channel',fontsize=fsizepl) #as per accompanying txt files
    ax1.imshow(arr_img30,cmap=cm.Greys)
    plt.axis('off')

    ax2 = plt.subplot2grid((1,3), (0,1), colspan=3)
    ax2.set_title('SE channel, signal pixels',fontsize=fsizepl)
    hlpse = np.copy(segmm['bright'])
    hlpse[~np.isnan(hlpse)] = 0.0 #inside
    hlpse[np.isnan(hlpse)] = 1.0 #outside
    ax2 = plt.imshow(hlpse[np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely] ,cmap=cm.Greys) #or 'OrRd'
    plt.axis('off')

#    backgdinit = 50
#    initbin = (150+50+3)-1
#    
#    red = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[7] +'Redbright.npz',mmap_mode='r') 
#    blue = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[7] +'Bluebright.npz',mmap_mode='r') 
#    
#    print(red['data'].shape)
#    datared = np.average(red['data'], axis = (0))
#    datablue = np.average(blue['data'], axis = (0))
#    
#    if True is False:
#        pass
#    else:
#        initbin = (150+50+3)-1 #init bin for decay
#        backgdinit = 50
#        ### 700ns /40ns = 7. ....
#        datared_init = datared[0:backgdinit,:,:]
#        datared = datared[initbin:,:,:]
#        datablue_init = datablue[0:backgdinit,:,:]
#        datablue = datablue[initbin:,:,:]

    multipage_longer('ZZZZzzzzzz.pdf',dpi=80) 
   


klklklklk