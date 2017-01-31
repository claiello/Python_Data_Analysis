import os
import sys
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../Python modules/") # necessary for the tex fonts
import scipy as sp
import scipy.misc
import matplotlib
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

varyC = True
#varyC = False

#dotriple = True
dotriple = False

dset = 1

cut_longa = 200#8#6
cut_shorta = 60#15#5#3
cut_shortissima = 3

if dotriple == True:
    init_tau_longa = 100.0 #10.0
    init_tau_shorta = 10.0 #1.0
    init_tau_shortissima = 1.0
else:
    init_tau_longa = 100.0 #10.0
    init_tau_shorta = 10.0 #1.0
    
backgdinit = 50
initbin = (150+50+3)-1

dofitwitherror = True
assumepoisson = True

extrapref = 'prevtau'
   
###############################################################################
###############################################################################
###############################################################################

if dset == 1:
    nametr = ['2017-01-23-1547_ImageSequence__250.000kX_10.000kV_30mu_7',
          '2017-01-23-1608_ImageSequence__250.000kX_10.000kV_30mu_8',
          '2017-01-23-1633_ImageSequence__250.000kX_10.000kV_30mu_9',
          '2017-01-23-1736_ImageSequence__250.000kX_10.000kV_30mu_10',
          '2017-01-23-1818_ImageSequence__63.372kX_10.000kV_30mu_11']

    let = ['pix1', 'pix2', 'pix3', 'pix4', 'pix5']
    
    listofindex = [0,1,2,3,4]
    
    loadprefix = ''
    
###############################################################################
###############################################################################
###############################################################################



if dset == 1:
    pref0 = 'Different_pixel_sizes_'
    
if varyC == True:
    pref1 = 'varC_'
else:
    pref1 = 'fixedC_'
    
if dotriple == True:
    pref2 = 'triple_'
else:
    pref2 = 'double_'
    
pref3= 'assumepoisson'


    
prefixend = pref0 + pref1 + pref2 + pref3 + extrapref
print(prefixend)

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
if dotriple == True:
    g_array_red = np.zeros(len(nametr))
    ge_array_red = np.zeros(len(nametr))
    g_array_blue = np.zeros(len(nametr))
    ge_array_blue = np.zeros(len(nametr))

Time_bin = 1000.0

for index in listofindex: 
    
    print(index)
    
    print('bef loading')
    redd = np.load(loadprefix + let[index] + 'RED1D.npz',mmap_mode='r') 
    blued = np.load(loadprefix + let[index] + 'BLUE1D.npz',mmap_mode='r') 
    red = redd['data'][initbin:]
    blue = blued['data'][initbin:]
    back_init_red = np.average(redd['data'][0:backgdinit])
    back_init_blue = np.average(blued['data'][0:backgdinit])
    
    if assumepoisson == True: #If poisson distributed, std is sqrt of average
        redstd = np.sqrt(red)
        bluestd = np.sqrt(blue)
    else:
        std_red = np.load(loadprefix +'Std_array_red.npz',mmap_mode='r') #any prefix will do
        std_blue = np.load(loadprefix +'Std_array_blue.npz',mmap_mode='r') #any prefix will do
        redstd = std_red['data'][index,:]
        bluestd = std_blue['data'][index,:]
    
    #round off small values
    val = 1.0e-10
    redstd[redstd < val] = val
    bluestd[bluestd < val] = val
    red[red < val] = val
    blue[blue < val] = val
    
    del redd, blued
    gc.collect()
    print('after loading')

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
    
    print('bef init')
    gc.collect()
    
    if dotriple == True:
        init_guess = calc_triple_fit( np.arange(0,red.shape[0])*Time_bin*1.0e-9,  red, Time_bin*1.0e-9,cut_longa,cut_shorta,cut_shortissima,init_tau_longa,init_tau_shorta,init_tau_shortissima)
        init_guessb = calc_triple_fit(np.arange(0,blue.shape[0])*Time_bin*1.0e-9,blue, Time_bin*1.0e-9,cut_longa,cut_shorta,cut_shortissima,init_tau_longa,init_tau_shorta,init_tau_shortissima) 
    else:
        init_guess = calc_double_fit(np.arange(0,red.shape[0])*Time_bin*1.0e-9 ,  red, Time_bin*1.0e-9,cut_longa,cut_shorta,init_tau_longa,init_tau_shorta)  
        init_guessb = calc_double_fit(np.arange(0,blue.shape[0])*Time_bin*1.0e-9,blue, Time_bin*1.0e-9,cut_longa,cut_shorta,init_tau_longa,init_tau_shorta)
    
    init_guess[2] = back_init_red
    init_guessb[2] = back_init_blue
           
    print('bef fit')
    gc.collect()
    
    ###########################################################################
    if extrapref == 'prevtau':
        print('doing prevtau')
        if index == 0:
            pass
        else:
             if dotriple == True:
                init_guess[6] = g_array_red[index-1]
                init_guessb[6] = g_array_blue[index-1]
             
             init_guess[1] = b_array_red[index-1]
             init_guess[4] = e_array_red[index-1]
             init_guessb[1] = b_array_blue[index-1]
             init_guessb[4] = e_array_blue[index-1]
    
#    #manual adjustment:
#    if (index == 0):# or (index == 5):
#       init_guess[6] = 1.0
#    if (index == 2):
#       init_guess[1] = 50.0
#       init_guess[4] = 70.0
#       init_guess[6] = 10.0
###        init_guess[4] = 260.0
##        init_guess[6] = 0.12
    
    if dotriple == True:
        if dofitwitherror == True:
            b,e,be,ee,b2,e2,be2,ee2,g,ge,g2,ge2,chisquared,chisquared2 = calcdecay_subplot_nan_triple_1D_with_error(red, time_detail= Time_bin*1.0e-9,titulo='',single=False,other_dset1=None,other_dset2=blue,init_guess=init_guess,init_guess2=init_guessb,unit='kHz',vary_offset_desired = varyC,error_array=redstd, error_array2 = bluestd,minimizepoisson=assumepoisson)    
        else:
            b,e,be,ee,b2,e2,be2,ee2,g,ge,g2,ge2,chisquared,chisquared2 = calcdecay_subplot_nan_triple_1D(red, time_detail= Time_bin*1.0e-9,titulo='',single=False,other_dset1=None,other_dset2=blue,init_guess=init_guess,init_guess2=init_guessb,unit='kHz',vary_offset_desired = varyC,error_array=None, error_array2 = None,minimizepoisson=assumepoisson)    

    else:
        if dofitwitherror == True:
            b,e,be,ee,b2,e2,be2,ee2,chisquared,chisquared2 = calcdecay_subplot_nan_1D_with_error(red, time_detail= Time_bin*1.0e-9,titulo='',single=False,other_dset1=None,other_dset2=blue,init_guess=init_guess,init_guess2=init_guessb,unit='kHz',error_array=None, vary_offset_desired = varyC, yerrinput = redstd, yerr2input = bluestd,minimizepoisson=assumepoisson)   
        else:
            b,e,be,ee,b2,e2,be2,ee2,chisquared,chisquared2 = calcdecay_subplot_nan_1D(red, time_detail= Time_bin*1.0e-9,titulo='',single=False,other_dset1=None,other_dset2=blue,init_guess=init_guess,init_guess2=init_guessb,unit='kHz',error_array=None, vary_offset_desired = varyC,minimizepoisson=assumepoisson)   
 
        
    b_array_red[index] = b
    e_array_red[index] = e
    be_array_red[index] = be    
    ee_array_red[index] = ee  
    b_array_blue[index] = b2
    e_array_blue[index] = e2
    be_array_blue[index] = be2    
    ee_array_blue[index] = ee2  
    chiresult_red[index] = chisquared
    chiresult_blue[index] = chisquared2
    if dotriple == True:
        g_array_red[index] = g
        ge_array_red[index] = ge    
        g_array_blue[index] = g2
        ge_array_blue[index] = ge2  
    
    plt.ylabel("Average luminescence,\n per signal pixel (kHz)",fontsize=fsizepl)
    plt.xlabel(r'Time after blanking the electron beam ($\mu$s)',  fontsize=fsizepl)
    plt.xlim(xmax=1400.0) #1400
    major_ticks0 = [250,500,750,1000,1250]
    ax1.set_xticks(major_ticks0) 
    ax1.tick_params(labelsize=fsizepl)  

    xx_array = np.arange(0,red.shape[0])*Time_bin*1.0e-9
    plt.semilogy(xx_array/1.0e-6,back_init_red*np.ones(len(xx_array)),'r--',lw=2,label='Mean CL from signal pixels, before e-beam')
    plt.semilogy(xx_array/1.0e-6,back_init_blue*np.ones(len(xx_array)),'g--',lw=2,label='Mean CL from blue signal pixels, before e-beam')
   
    del xx_array,red,blue
    gc.collect()
    
    print('here3')
  
multipage_longer(prefixend + '.pdf',dpi=80)
#notsaved

Ps = [2.23, 1.79, 1.49, 1.28, 1.12]
mycode =prefixend + 'Pixel_size = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'Pixel_size', data = Ps)
   
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

mycode = prefixend +'Chiresult_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'Chiresult_red', data = chiresult_red)

mycode = prefixend +'Chiresult_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez(prefixend +'Chiresult_blue', data = chiresult_blue)

if dotriple == True:
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



klklklk