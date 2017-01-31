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

#prefix = 'fixCsametautriple'
#prefix = 'fixCprevtautriple'
prefix = 'varCsametautriple'
#prefix = 'varCprevtautriple'

prefixend = prefix
#prefixend = 'fixCsametautriple_nodiv'
#prefixend = 'fixCprevtautriple_nodiv'
#prefixend = 'varCsametautriple_nodiv'
#prefixend = 'varCprevtautriple_nodiv'

Time_bin = 1000.0

nametr = ['2017-01-13-1730_ImageSequence__150.000kX_10.000kV_30mu_15',
          '2017-01-13-1810_ImageSequence__150.000kX_10.000kV_30mu_17',
          '2017-01-13-1935_ImageSequence__150.000kX_10.000kV_30mu_3',
          '2017-01-13-2011_ImageSequence__150.000kX_10.000kV_30mu_4',
          '2017-01-13-2050_ImageSequence__150.000kX_10.000kV_30mu_7',
          '2017-01-13-2132_ImageSequence__150.000kX_10.000kV_30mu_8',
          '2017-01-13-2213_ImageSequence__150.000kX_10.000kV_30mu_9',
          '2017-01-13-2251_ImageSequence__150.000kX_10.000kV_30mu_10']
          
Pixel_size = 2.5*np.ones(len(nametr)) #nm
Ps = Pixel_size #pixel size in nm, the numbers above with round nm precision

description = 'Andrea small NaYF4:Er'
               
kv = [10]

#nominal Temps
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

chiresult_red = np.zeros(len(nametr))
chiresult_blue = np.zeros(len(nametr))

g_array_red = np.zeros(len(nametr))
ge_array_red = np.zeros(len(nametr))
g_array_blue = np.zeros(len(nametr))
ge_array_blue = np.zeros(len(nametr))

pisize =Pixel_size

listofindex =np.arange(0,len(nametr))#,11]


cut_longa = 100#200#8#6
cut_shorta = 10#60#5#3
cut_shortissima = 5
init_tau_long = 100.0
init_tau_short = 10.0
init_tau_shortissimo = 1.0

#index = 4
#if index is 4:
for index in [4,5,6,7]: #listofindex:
    
    print(index)
    
    print('bef loading')
    redd = np.load(prefix + let[index] + 'RED1D.npz',mmap_mode='r') 
    blued = np.load(prefix + let[index] + 'BLUE1D.npz',mmap_mode='r') 
    red = redd['data']
    blue = blued['data']
    print(redd['data'].shape)
    print(red.shape)
    del redd, blued
    gc.collect()
    
    print('after loading')
    
    fsizetit = 18 
    fsizepl = 16 
    sizex = 8 
    sizey = 6
    dpi_no = 80
    lw = 2
    
    titulo = description + ' ' +  let[index] +  ' (10kV, 30$\mu$m aperture, 1$\mu$s time bins, ' + str(Ps)+ 'nm pixels, green/red: $</>$ 593nm)'
    
    fig40= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig40.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}') #not working
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}')  #not working
    plt.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]  #not working
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')         
    
    ax1 = plt.subplot2grid((2,12), (1, 4), colspan=4)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    
    fastfactor=1
    print('bef init')
    gc.collect()
    
    init_guess = calc_triple_fit(np.arange(0,red.shape[0])*Time_bin*1.0e-9*fastfactor,red, Time_bin*1.0e-9*fastfactor,cut_longa,cut_shorta,cut_shortissima,init_tau_long,init_tau_short,init_tau_shortissimo)
    
    gc.collect()    
    
    init_guessb = calc_triple_fit(np.arange(0,blue.shape[0])*Time_bin*1.0e-9*fastfactor,blue,Time_bin*1.0e-9*fastfactor,cut_longa,cut_shorta,cut_shortissima,init_tau_long,init_tau_short,init_tau_shortissimo)
    
    
    lastpts = 20
    cinit = np.average(red[-lastpts:-1])
    cinitb = np.average(blue[-lastpts:-1])
    
    #replace c with cinit
    init_guess[2] = cinit 
    init_guessb[2] = cinitb
#    if index == 4:
#        print('here blue')
#        init_guessb[1] = 5.0
        
    ##################################### THIS IS TO GIVE PREV TAU AS INIT
    if (prefix == 'fixCprevtautriple') or (prefix=='varCprevtautriple'):
        print('doing its thing')
        if index == 4:
            pass
        else:
            print('doing its thing')
            init_guess[1] = np.average(b_array_red[4:index])
            init_guess[4] = np.average(e_array_red[4:index])
            init_guess[6] = np.average(g_array_red[4:index])
            init_guessb[1] = np.average(b_array_blue[4:index])
            init_guessb[4] = np.average(e_array_blue[4:index]) 
            init_guessb[6] = np.average(g_array_blue[4:index])
  
            
  
     ##################################### THIS IS TO GIVE PREV TAU AS INIT
    if (prefixend == 'varCsametautriple'): #and ((index == 4) or (index ==6) or (index == 7)):
        init_guessb[1] = init_guess[1] #33.0
        init_guessb[4] = init_guess[4] #165.0
        init_guessb[6] = init_guess[6] #7.0
    
    print('bef fit')
    gc.collect()
    
    b,e,be,ee,b2,e2,be2,ee2,g,ge,g2,ge2,chisquared,chisquared2 = calcdecay_subplot_nan_triple_1D(red, time_detail= Time_bin*1.0e-9*fastfactor,titulo='',single=False,other_dset1=None,other_dset2=blue,init_guess=init_guess,init_guess2=init_guessb,unit='kHz',error_array=None) #,error_array=error_arrayr, error_array2=error_arrayb)    
        
    b_array_red[index] = b
    e_array_red[index] = e
    be_array_red[index] = be    
    ee_array_red[index] = ee  
    b_array_blue[index] = b2
    e_array_blue[index] = e2
    be_array_blue[index] = be2    
    ee_array_blue[index] = ee2  
    g_array_red[index] = g
    ge_array_red[index] = ge    
    g_array_blue[index] = g2
    ge_array_blue[index] = ge2   
    chiresult_red[index] = chisquared
    chiresult_blue[index] = chisquared2

    
    plt.ylabel("Average luminescence,\n per signal pixel (kHz)",fontsize=fsizepl)
    plt.xlabel(r'Time after blanking the electron beam ($\mu$s)',  fontsize=fsizepl)
    plt.xlim(xmax=1400.0) #1400
    major_ticks0 = [250,500,750,1000,1250]
    ax1.set_xticks(major_ticks0) 
    ax1.tick_params(labelsize=fsizepl)
    
    # Extra plots        

    xx_array = np.arange(0,red.shape[0])*Time_bin*1.0e-9*fastfactor
    plt.semilogy(xx_array/1.0e-6,cinit*np.ones(len(xx_array)),'r--',lw=2,label='Mean CL from signal pixels, before e-beam')
    plt.semilogy(xx_array/1.0e-6,cinitb*np.ones(len(xx_array)),'g--',lw=2,label='Mean CL from blue signal pixels, before e-beam')
    #Plot mean background
    #plt.semilogy(xx_array/1e-6,np.average(datablue_init*hlpd,axis=(0,1,2))*np.ones(len(xx_array)),'--',color='DarkGreen',lw=1,label='Mean CL from background, before e-beam')
    
    del xx_array,red,blue
    gc.collect()
    
    print('here3')
    multipage_longer('ZZZZSingle-'+ let[index] + prefix + '.pdf',dpi=80)
    

##### ONCE ALL FITS WORK, 
###### NEED TO RUN THESE LINES BELOW SO THAT ALL NPZ FILES ARE CREATED 

mycode =prefixend + 'B_array_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'B_array_red', data = b_array_red)

mycode =prefixend +'Be_array_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'Be_array_red', data = be_array_red)

mycode = prefixend +'E_array_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'E_array_red', data = e_array_red)

mycode = prefixend +'Ee_array_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'Ee_array_red', data = ee_array_red)

mycode = prefixend +'B_array_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'B_array_blue', data = b_array_blue)

mycode =prefixend +'Be_array_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'Be_array_blue', data = be_array_blue)

mycode = prefixend +'E_array_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'E_array_blue', data = e_array_blue)

mycode = prefixend +'Ee_array_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'Ee_array_blue', data = ee_array_blue)

mycode = prefixend +'G_array_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'G_array_red', data = g_array_red)

mycode = prefixend +'Ge_array_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'Ge_array_red', data = ge_array_red)

mycode = prefixend +'G_array_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'G_array_blue', data = g_array_blue)

mycode = prefixend +'Ge_array_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'Ge_array_blue', data = ge_array_blue)

mycode = prefixend +'Chiresult_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'Chiresult_red', data = chiresult_red)

mycode = prefixend +'Chiresult_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'Chiresult_blue', data = chiresult_blue)

klklklklk
