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

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

import my_fits

##############################################################################
##############################################################################
#6 different prefixes

#dotriple = True
dotriple = False


#fitlinearwitherror = True
fitlinearwitherror = False
    

#PARAMS THAT NEED TO BE CHANGED
###############################################################################

#if dset == 1:
#    nametr = ['2017-01-05-1452_ImageSequence__250.000kX_10.000kV_30mu_10',
#          '2017-01-05-1557_ImageSequence__250.000kX_10.000kV_30mu_15',
#          '2017-01-05-1634_ImageSequence__250.000kX_10.000kV_30mu_20',
#          '2017-01-05-1709_ImageSequence__250.000kX_10.000kV_30mu_23',
#          '2017-01-05-1745_ImageSequence__250.000kX_10.000kV_30mu_26',
#          '2017-01-05-1831_ImageSequence__250.000kX_10.000kV_30mu_30',
#          '2017-01-05-1906_ImageSequence__250.000kX_10.000kV_30mu_32']
#
#    let = ['RT','V0','V0pt25' ,'V0pt5', 'V0pt5b','V0pt75','V1']
#    
#    listofindex = [1,2,4,5]
#    
#    loadprefix = '../2017-01-05_Andrea_small_new_sample_5DiffTemps/'
#    
#if dset == 2:
#    nametr = ['2016-12-19-1924_ImageSequence__100.000kX_10.000kV_30mu_4',
#          '2016-12-19-1950_ImageSequence__100.000kX_10.000kV_30mu_5',
#          '2016-12-19-2130_ImageSequence__100.000kX_10.000kV_30mu_8',
#          '2016-12-19-2015_ImageSequence__100.000kX_10.000kV_30mu_6',
#          '2016-12-19-2056_ImageSequence__27.836kX_10.000kV_30mu_7']
#          
#    let = ['V0pt25' ,'V0pt5', 'V0', 'V0pt75','V1']
#    
#    listofindex = [0,1,2,3,4]
#    
#    loadprefix = '../2016-12-19_Andrea_BigNPs_5DiffTemps/'
#    
#if dset == 3:
#    nametr = ['2017-01-13-1730_ImageSequence__150.000kX_10.000kV_30mu_15',
#          '2017-01-13-1810_ImageSequence__150.000kX_10.000kV_30mu_17',
#          '2017-01-13-1935_ImageSequence__150.000kX_10.000kV_30mu_3',
#          '2017-01-13-2011_ImageSequence__150.000kX_10.000kV_30mu_4',
#          '2017-01-13-2050_ImageSequence__150.000kX_10.000kV_30mu_7',
#          '2017-01-13-2132_ImageSequence__150.000kX_10.000kV_30mu_8',
#          '2017-01-13-2213_ImageSequence__150.000kX_10.000kV_30mu_9',
#          '2017-01-13-2251_ImageSequence__150.000kX_10.000kV_30mu_10']
#         
#    let = ['N102pt5', 'N92pt9', 'N66pt4', 'N72pt8', 'N60pt6', 'N48pt5', 'N39pt8', 'N31pt2']
#
#    listofindex =  [4,5,6,7]
#    
#    loadprefix = '../2017-01-13_Andrea_NPs_CoolingDown_Controllably/'

if dotriple == True:
    pref2 = 'triple_'
else:
    pref2 = 'double_'

pref3= 'assumepoisson'

Time_bin = 1000#in ns; 
nominal_time_on = 150.0 #time during which e-beam nominally on, in mus
totalpoints = 1500 #total number of time-resolved points

tminx = 20
tmaxx = 100

fsizepl = 24
fsizenb = 20

######## LARGEST ZOOM
pref0 = 'SmallArea_'
prefix = pref0 + 'varC_' + pref2 + pref3 
Il_data = np.load('../2017-01-05_Andrea_small_new_sample_5DiffTemps/Il_data.npz')
il_data = Il_data['data']  
Il_data_std = np.load('../2017-01-05_Andrea_small_new_sample_5DiffTemps/Il_data_std.npz')
il_data_std = Il_data_std['data']  

Red_int_array = np.load('../2017-01-05_Andrea_small_new_sample_5DiffTemps/Red_int_array.npz') 
red_int_array = Red_int_array['data']
Blue_int_array = np.load('../2017-01-05_Andrea_small_new_sample_5DiffTemps/Blue_int_array.npz') 
blue_int_array = Blue_int_array['data']

Red_std_array = np.load('../2017-01-05_Andrea_small_new_sample_5DiffTemps/Red_std_array.npz') 
red_std_array = Red_int_array['data']
Blue_std_array = np.load('../2017-01-05_Andrea_small_new_sample_5DiffTemps/Blue_std_array.npz') 
blue_std_array = Blue_int_array['data']

B_array_red= np.load(prefix +'B_array_red.npz')
b_array_red = B_array_red['data']  
Be_array_red = np.load(prefix +'Be_array_red.npz')
be_array_red = Be_array_red['data'] 
E_array_red = np.load(prefix +'E_array_red.npz')
e_array_red = E_array_red['data']   
Ee_array_red = np.load(prefix +'Ee_array_red.npz')
ee_array_red = Ee_array_red['data']  
try: 
    G_array_red= np.load(prefix +'G_array_red.npz')
    g_array_red = G_array_red['data']  
    Ge_array_red = np.load(prefix +'Ge_array_red.npz')
    ge_array_red = Ge_array_red['data'] 
except:
    pass

B_array_blue= np.load(prefix +'B_array_blue.npz')
b_array_blue = B_array_blue['data']  
Be_array_blue = np.load(prefix +'Be_array_blue.npz')
be_array_blue = Be_array_blue['data'] 
E_array_blue = np.load(prefix +'E_array_blue.npz')
e_array_blue = E_array_blue['data']   
Ee_array_blue = np.load(prefix +'Ee_array_blue.npz')
ee_array_blue = Ee_array_blue['data'] 
try:
    G_array_blue= np.load(prefix +'G_array_blue.npz')
    g_array_blue = G_array_blue['data']  
    Ge_array_blue = np.load(prefix +'Ge_array_blue.npz')
    ge_array_blue = Ge_array_blue['data'] 
except:
    pass

Chiresult_red = np.load(prefix +'Chiresult_red.npz')
chiresult_red = Chiresult_red['data'] 
Chiresult_blue = np.load(prefix +'Chiresult_blue.npz')
chiresult_blue = Chiresult_blue['data'] 

######## SMALLEST ZOOM
pref0 = 'LargeArea_'
prefix= pref0 + 'varC_' + pref2 + pref3 
Il_data2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/Il_data.npz')
il_data2 = Il_data2['data']  
Il_data_std2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/Il_data_std.npz')
il_data_std2 = Il_data_std2['data'] 

Red_int_array2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/Red_int_array.npz') 
red_int_array2 = Red_int_array2['data']
Blue_int_array2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/Blue_int_array.npz') 
blue_int_array2 = Blue_int_array2['data']

Red_std_array2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/Red_std_array.npz') 
red_std_array2 = Red_int_array2['data']
Blue_std_array2 = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/Blue_std_array.npz') 
blue_std_array2 = Blue_int_array2['data']

B_array_red2= np.load(prefix +'B_array_red.npz')
b_array_red2 = B_array_red2['data']  
Be_array_red2 = np.load(prefix +'Be_array_red.npz')
be_array_red2 = Be_array_red2['data'] 
E_array_red2 = np.load(prefix +'E_array_red.npz')
e_array_red2 = E_array_red2['data']   
Ee_array_red2 = np.load(prefix +'Ee_array_red.npz')
ee_array_red2 = Ee_array_red2['data']   
try:
    G_array_red2 = np.load(prefix +'G_array_red.npz')
    g_array_red2 = G_array_red2['data']   
    Ge_array_red2 = np.load(prefix +'Ge_array_red.npz')
    ge_array_red2 = Ge_array_red2['data']  
except:
    pass

B_array_blue2= np.load(prefix +'B_array_blue.npz')
b_array_blue2 = B_array_blue2['data']  
Be_array_blue2 = np.load(prefix +'Be_array_blue.npz')
be_array_blue2 = Be_array_blue2['data'] 
E_array_blue2 = np.load(prefix +'E_array_blue.npz')
e_array_blue2 = E_array_blue2['data']   
Ee_array_blue2 = np.load(prefix +'Ee_array_blue.npz')
ee_array_blue2 = Ee_array_blue2['data'] 
try:
    G_array_blue2 = np.load(prefix +'G_array_blue.npz')
    g_array_blue2 = G_array_blue2['data']   
    Ge_array_blue2 = np.load(prefix +'Ge_array_blue.npz')
    ge_array_blue2 = Ge_array_blue2['data']  
except:
    pass

Chiresult_red2 = np.load(prefix +'Chiresult_red.npz')
chiresult_red2 = Chiresult_red2['data'] 
Chiresult_blue2 = np.load(prefix +'Chiresult_blue.npz')
chiresult_blue2 = Chiresult_blue2['data'] 


######## MEDIUM ZOOM
pref0 = 'MediumArea_'
prefix = pref0 + 'varC_' + pref2 + pref3 
Il_data3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/Il_data.npz')
il_data3 = Il_data3['data']  
Il_data_std3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/Il_data_std.npz')
il_data_std3 = Il_data_std3['data']  

Red_int_array3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/Red_int_array.npz') 
red_int_array3 = Red_int_array3['data']
Blue_int_array3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/Blue_int_array.npz') 
blue_int_array3 = Blue_int_array3['data']

Red_std_array3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/Red_std_array.npz') 
red_std_array3 = Red_int_array3['data']
Blue_std_array3 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/Blue_std_array.npz') 
blue_std_array3 = Blue_int_array3['data']

B_array_red3= np.load(prefix +'B_array_red.npz')
b_array_red3 = B_array_red3['data']  
Be_array_red3 = np.load(prefix +'Be_array_red.npz')
be_array_red3 = Be_array_red3['data'] 
E_array_red3 = np.load(prefix +'E_array_red.npz')
e_array_red3 = E_array_red3['data']   
Ee_array_red3 = np.load(prefix +'Ee_array_red.npz')
ee_array_red3 = Ee_array_red3['data']   
try:
    G_array_red3 = np.load(prefix +'G_array_red.npz')
    g_array_red3 = G_array_red3['data']   
    Ge_array_red3 = np.load(prefix +'Ge_array_red.npz')
    ge_array_red3 = Ge_array_red3['data']  
except:
    pass

B_array_blue3= np.load(prefix +'B_array_blue.npz')
b_array_blue3 = B_array_blue3['data']  
Be_array_blue3 = np.load(prefix +'Be_array_blue.npz')
be_array_blue3 = Be_array_blue3['data'] 
E_array_blue3 = np.load(prefix +'E_array_blue.npz')
e_array_blue3 = E_array_blue3['data']   
Ee_array_blue3 = np.load(prefix +'Ee_array_blue.npz')
ee_array_blue3 = Ee_array_blue3['data'] 
try:
    G_array_blue3 = np.load(prefix +'G_array_blue.npz')
    g_array_blue3 = G_array_blue3['data']   
    Ge_array_blue3 = np.load(prefix +'Ge_array_blue.npz')
    ge_array_blue3 = Ge_array_blue3['data'] 
except:
    pass

Chiresult_red3 = np.load(prefix +'Chiresult_red.npz')
chiresult_red3 = Chiresult_red3['data'] 
Chiresult_blue3 = np.load(prefix +'Chiresult_blue.npz')
chiresult_blue3 = Chiresult_blue3['data'] 


######## TOP PICS


Pixel_size = np.array([0.89,2.2,2.5]) #nm  #largest/smallest/medium

Multiplication_factor =np.zeros((5,3))
Multiplication_factor[0,:] = [1,1,1]
Multiplication_factor[1,:] = Pixel_size*Pixel_size
Multiplication_factor[2,:] = 1.0/(Pixel_size*Pixel_size)
Multiplication_factor[3,:] = Pixel_size
Multiplication_factor[4,:] = 1.0/(Pixel_size)
description = ['no area norm.', r'$\times$ area', '$\div$ area', r'$\times$ length', '$\div$ length']

RTindex = [1,0,-1] #largest/smallest/medium

for index in np.arange(0,Multiplication_factor.shape[0]):

    
    fig42= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig42.set_size_inches(1200./fig42.dpi,900./fig42.dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')    


    titulo = 'NaYF$_4$:Yb,Er UC NPs (sample by Andrea Pickel)\n (10 kV, 30 $\mu$m aperture, 1 $\mu$s time bins, 1.4 ms transient, cathodoluminescence green/red: $</>$ 593 nm) $\Rightarrow$ ' + description[index]
    plt.suptitle(titulo,fontsize=fsizetit)

    
    length_scalebar = 100.0 #in nm 
    scalebar_legend = '100nm'
    noplots = 13
    nolines = 5
    
    ######## LARGEST ZOOM
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[0]))
    let = ['RT','V0','V0pt25' ,'V0pt5', 'V0pt5b','V0pt75','V1']
    
    ax0 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)
    se = np.load('../2017-01-05_Andrea_small_new_sample_5DiffTemps/'+ let[1] +'SEchannel.npz',mmap_mode='r') 
    xlen = se['data'].shape[0]
    ylen = se['data'].shape[1]
    delx = 0#+28
    dely = 0
    xval = 107
    yval = 107
    plt.imshow(se['data'][np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely],cmap=cm.Greys_r)
    plt.axis('off')
    sbar = sb.AnchoredScaleBar(ax0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    ax0.add_artist(sbar)

    ax00 = plt.subplot2grid((nolines,noplots), (0,2), colspan=1, rowspan=1)
    se = np.load('../2017-01-05_Andrea_small_new_sample_5DiffTemps/'+let[2] +'SEchannel.npz',mmap_mode='r') 
    xlen = se['data'].shape[0]
    ylen = se['data'].shape[1]
    delx = 0#+28
    dely = 0 #+26
    xval = 96
    yval = 106
    cutx = 0 #32
    cutxtop = 0 #10
    plt.imshow(se['data'][np.floor(xlen/2.0)-xval+delx+cutxtop:np.floor(xlen/2.0)+xval+delx-cutx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely],cmap=cm.Greys_r)
    plt.axis('off')
    sbar = sb.AnchoredScaleBar(ax00.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    ax00.add_artist(sbar)

    ax0000 = plt.subplot2grid((nolines,noplots), (0,8), colspan=1, rowspan=1)
    se = np.load('../2017-01-05_Andrea_small_new_sample_5DiffTemps/'+let[4] +'SEchannel.npz',mmap_mode='r')
    xlen = se['data'].shape[0]
    ylen = se['data'].shape[1] 
    delx = 0#+28
    dely = 0#00
    xval = 80
    yval = 86
    plt.imshow(se['data'][np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely],cmap=cm.Greys_r)
    plt.axis('off')
    sbar = sb.AnchoredScaleBar(ax0000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    ax0000.add_artist(sbar)

    ax00000 = plt.subplot2grid((nolines,noplots), (0,11), colspan=1, rowspan=1)
    se = np.load('../2017-01-05_Andrea_small_new_sample_5DiffTemps/'+let[5] +'SEchannel.npz',mmap_mode='r') 
    xlen = se['data'].shape[0]
    ylen = se['data'].shape[1]
    delx = 0#+28
    dely = 0
    xval = 125
    yval = 97
    cutx = 0 #75
    plt.imshow(se['data'][np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx-cutx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely],cmap=cm.Greys_r)
    plt.axis('off')
    sbar = sb.AnchoredScaleBar(ax00000.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    ax00000.add_artist(sbar)
    
    ######## SMALLEST ZOOM
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[1]))
    letold = ['V0pt25' ,'V0pt5', 'V0', 'V0pt75','V1']

    axb0 = plt.subplot2grid((nolines,noplots), (0,1), colspan=1, rowspan=1)
    se = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/'+ letold[0] +'SEchannel.npz',mmap_mode='r') 
    plt.imshow(se['data'],cmap=cm.Greys_r)
    plt.axis('off')
    sbar = sb.AnchoredScaleBar(axb0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    axb0.add_artist(sbar)

    axb0 = plt.subplot2grid((nolines,noplots), (0,5), colspan=1, rowspan=1)
    se = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/'+ letold[1] +'SEchannel.npz',mmap_mode='r') 
    plt.imshow(se['data'],cmap=cm.Greys_r)
    plt.axis('off')
    sbar = sb.AnchoredScaleBar(axb0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    axb0.add_artist(sbar)

    axb0 = plt.subplot2grid((nolines,noplots), (0,7), colspan=1, rowspan=1)
    se = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/'+ letold[2] +'SEchannel.npz',mmap_mode='r') 
    plt.imshow(se['data'],cmap=cm.Greys_r)
    plt.axis('off')
    sbar = sb.AnchoredScaleBar(axb0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    axb0.add_artist(sbar)
    
    axb0 = plt.subplot2grid((nolines,noplots), (0,9), colspan=1, rowspan=1)
    se = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/'+ letold[3] +'SEchannel.npz',mmap_mode='r') 
    plt.imshow(se['data'],cmap=cm.Greys_r)
    plt.axis('off')
    sbar = sb.AnchoredScaleBar(axb0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    axb0.add_artist(sbar)
    
    axb0 = plt.subplot2grid((nolines,noplots), (0,12), colspan=1, rowspan=1)
    se = np.load('../2016-12-19_Andrea_BigNPs_5DiffTemps/'+ letold[4] +'SEchannel.npz',mmap_mode='r') 
    plt.imshow(se['data'],cmap=cm.Greys_r)
    plt.axis('off')
    sbar = sb.AnchoredScaleBar(axb0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    axb0.add_artist(sbar)


    ######## MEDIUM ZOOM
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[2]))
    leto = ['N102pt5', 'N92pt9', 'N66pt4', 'N72pt8', 'N60pt6', 'N48pt5', 'N39pt8', 'N31pt2']

    axc0 = plt.subplot2grid((nolines,noplots), (0,3), colspan=1, rowspan=1)
    se = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[4] +'SEchannel.npz',mmap_mode='r') 
    xlen = se['data'].shape[0]
    ylen = se['data'].shape[1] 
    delx = 0#+28
    dely = 0#00
    xval = 144
    yval = 142
    plt.imshow(se['data'][np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely],cmap=cm.Greys_r)
    plt.axis('off')
    sbar = sb.AnchoredScaleBar(axc0.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    axc0.add_artist(sbar)

    axc1 = plt.subplot2grid((nolines,noplots), (0,4), colspan=1, rowspan=1)
    se = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[5] +'SEchannel.npz',mmap_mode='r') 
    delx = 0#+28
    dely = 0#00
    xval = 133
    yval = 122
    plt.imshow(se['data'][np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely],cmap=cm.Greys_r)
    plt.axis('off')
    sbar = sb.AnchoredScaleBar(axc1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    axc1.add_artist(sbar)
    
    axc2 = plt.subplot2grid((nolines,noplots), (0,6), colspan=1, rowspan=1)
    se = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[6] +'SEchannel.npz',mmap_mode='r') 
    delx = 0#+28
    dely = 0#00
    xval = 135
    yval = 105
    plt.imshow(se['data'][np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely],cmap=cm.Greys_r)
    plt.axis('off')
    sbar = sb.AnchoredScaleBar(axc2.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    axc2.add_artist(sbar)
    
    axc3 = plt.subplot2grid((nolines,noplots), (0,10), colspan=1, rowspan=1)
    se = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[7] +'SEchannel.npz',mmap_mode='r')
    delx = 0#+28
    dely = 0#00
    xval = 144
    yval = 120
    plt.imshow(se['data'][np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely],cmap=cm.Greys_r)
    plt.axis('off')
    sbar = sb.AnchoredScaleBar(axc3.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
    axc3.add_artist(sbar)

    ######## DELETE POINTS

    if index == 0: #of course, only delete in first pass
        ################### LARGEST ZOOM
        todel = [0,3,6] #using [1,2,4,5]
        red_int_array = np.delete(red_int_array, todel)
        blue_int_array = np.delete(blue_int_array, todel)
        red_std_array = np.delete(red_std_array, todel)
        blue_std_array = np.delete(blue_std_array, todel)
        b_array_red = np.delete(b_array_red, todel)
        be_array_red = np.delete(be_array_red, todel)
        e_array_red = np.delete(e_array_red, todel)
        ee_array_red = np.delete(ee_array_red, todel)
        b_array_blue = np.delete(b_array_blue, todel)
        be_array_blue = np.delete(be_array_blue, todel)
        e_array_blue = np.delete(e_array_blue, todel)
        ee_array_blue = np.delete(ee_array_blue, todel)
        try:
            g_array_red = np.delete(g_array_red, todel)
            ge_array_red = np.delete(ge_array_red, todel)
            g_array_blue = np.delete(g_array_blue, todel)
            ge_array_blue = np.delete(ge_array_blue, todel)
        except:
            pass
        il_data = np.delete(il_data, todel)
        il_data_std = np.delete(il_data_std, todel)
        chiresult_red = np.delete(chiresult_red, todel)
        chiresult_blue = np.delete(chiresult_blue, todel)
    
        ################### MEDIUM ZOOM
        todel = [0,1,2,3]#  using [4,5,6,7]
        red_int_array3 = np.delete(red_int_array3, todel)
        blue_int_array3 = np.delete(blue_int_array3, todel)
        red_std_array3 = np.delete(red_std_array3, todel)
        blue_std_array3 = np.delete(blue_std_array3, todel)
        b_array_red3 = np.delete(b_array_red3, todel)
        be_array_red3 = np.delete(be_array_red3, todel)
        e_array_red3 = np.delete(e_array_red3, todel)
        ee_array_red3 = np.delete(ee_array_red3, todel)
        b_array_blue3 = np.delete(b_array_blue3, todel)
        be_array_blue3 = np.delete(be_array_blue3, todel)
        e_array_blue3 = np.delete(e_array_blue3, todel)
        ee_array_blue3 = np.delete(ee_array_blue3, todel)
        try:
            g_array_red3 = np.delete(g_array_red3, todel)
            ge_array_red3 = np.delete(ge_array_red3, todel)
            g_array_blue3 = np.delete(g_array_blue3, todel)
            ge_array_blue3 = np.delete(ge_array_blue3, todel)
        except:
            pass
        il_data3 = np.delete(il_data3, todel)
        il_data_std3 = np.delete(il_data_std3, todel)
        chiresult_red3 = np.delete(chiresult_red3, todel)
        chiresult_blue3 = np.delete(chiresult_blue3, todel)

    ##################################################### PAGE 1

    ax3 = plt.subplot2grid((nolines,noplots), (1,0), colspan=noplots, rowspan=1)
    ax2 = plt.subplot2grid((nolines,noplots), (2,0), colspan=noplots, sharex=ax3)
    ax1 = plt.subplot2grid((nolines,noplots), (3,0), colspan=noplots, sharex=ax3)
    ax0 = plt.subplot2grid((nolines,noplots), (4,0), colspan=noplots, sharex=ax3)
    
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.xaxis.set_ticks_position('bottom')
    ax0.yaxis.set_ticks_position('left')

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.xaxis.set_ticks_position('bottom')
    ax3.yaxis.set_ticks_position('left')

    x_vec = il_data
    ax3.errorbar(x_vec, e_array_red*Multiplication_factor[index,0], yerr=ee_array_red*Multiplication_factor[index,0], fmt='ro', markersize=10,ls='-',label='smallest area')
    ax2.errorbar(x_vec, b_array_red*Multiplication_factor[index,0], yerr=be_array_red*Multiplication_factor[index,0],  fmt='ro',markersize=10,ls='-')
    ax3.errorbar(x_vec, e_array_blue*Multiplication_factor[index,0], yerr=ee_array_blue*Multiplication_factor[index,0], fmt='go', markersize=10,ls='-')
    ax2.errorbar(x_vec, b_array_blue*Multiplication_factor[index,0], yerr=be_array_blue*Multiplication_factor[index,0],  fmt='go',markersize=10,ls='-')
   
    ax0.errorbar(x_vec, chiresult_red, fmt='ro',markersize=10,ls='-')  
    ax0.errorbar(x_vec, chiresult_blue, fmt='go',markersize=10,ls='-')  

    x_vec3 = il_data3
    ax3.errorbar(x_vec3, e_array_red3*Multiplication_factor[index,2], yerr=ee_array_red3*Multiplication_factor[index,2], fmt='rs', markersize=10,ls='--',label='medium area')
    ax2.errorbar(x_vec3, b_array_red3*Multiplication_factor[index,2], yerr=be_array_red3*Multiplication_factor[index,2],  fmt='rs',markersize=10,ls='--')
    ax3.errorbar(x_vec3, e_array_blue3*Multiplication_factor[index,2], yerr=ee_array_blue3*Multiplication_factor[index,2], fmt='gs', markersize=10,ls='--')
    ax2.errorbar(x_vec3, b_array_blue3*Multiplication_factor[index,2], yerr=be_array_blue3*Multiplication_factor[index,2],  fmt='gs',markersize=10,ls='--')
    
    ax0.errorbar(x_vec3, chiresult_red3, fmt='rs',markersize=10,ls='--')  
    ax0.errorbar(x_vec3, chiresult_blue3, fmt='gs',markersize=10,ls='--')  
    
    x_vec2 = il_data2
    ax3.errorbar(x_vec2, e_array_red2*Multiplication_factor[index,1], yerr=ee_array_red2*Multiplication_factor[index,1], fmt='r^', markersize=10,ls='dotted',label='largest area')
    ax2.errorbar(x_vec2, b_array_red2*Multiplication_factor[index,1], yerr=be_array_red2*Multiplication_factor[index,1],  fmt='r^',markersize=10,ls='dotted')
    ax3.errorbar(x_vec2, e_array_blue2*Multiplication_factor[index,1], yerr=ee_array_blue2*Multiplication_factor[index,1], fmt='g^', markersize=10,ls='dotted')
    ax2.errorbar(x_vec2, b_array_blue2*Multiplication_factor[index,1], yerr=be_array_blue2*Multiplication_factor[index,1],  fmt='g^',markersize=10,ls='dotted')
    
    ax0.errorbar(x_vec2, chiresult_red2, fmt='r^',markersize=10,ls='dotted')  
    ax0.errorbar(x_vec2, chiresult_blue2, fmt='g^',markersize=10,ls='dotted')  
    
    legend0 = ax3.legend(loc='upper left')
    ax3.add_artist(legend0)
    
    try:
        ax1.errorbar(x_vec, g_array_red*Multiplication_factor[index,0], yerr=ge_array_red*Multiplication_factor[index,0],  fmt='ro',markersize=10,ls='-')
        ax1.errorbar(x_vec, g_array_blue*Multiplication_factor[index,0], yerr=ge_array_blue*Multiplication_factor[index,0],  fmt='go',markersize=10,ls='-')  
        ax1.errorbar(x_vec3, g_array_red3*Multiplication_factor[index,2], yerr=ge_array_red3*Multiplication_factor[index,0],  fmt='rs',markersize=10,ls='--')
        ax1.errorbar(x_vec3, g_array_blue3*Multiplication_factor[index,2], yerr=ge_array_blue3*Multiplication_factor[index,0],  fmt='gs',markersize=10,ls='--')  
        ax1.errorbar(x_vec2, g_array_red2*Multiplication_factor[index,1], yerr=ge_array_red2*Multiplication_factor[index,0],  fmt='r^',markersize=10,ls='dotted')
        ax1.errorbar(x_vec2, g_array_blue2*Multiplication_factor[index,1], yerr=ge_array_blue2*Multiplication_factor[index,0],  fmt='g^',markersize=10,ls='dotted')  
    except:
        pass 
    
    x = np.concatenate([x_vec,x_vec2,x_vec3])
    
    #Fitting long
    efit_red = np.concatenate([e_array_red*Multiplication_factor[index,0],e_array_red2*Multiplication_factor[index,1],e_array_red3*Multiplication_factor[index,2]])
    efit_blue = np.concatenate([e_array_blue*Multiplication_factor[index,0],e_array_blue2*Multiplication_factor[index,1],e_array_blue3*Multiplication_factor[index,2]])
    
    eefit_red = np.concatenate([ee_array_red*Multiplication_factor[index,0],ee_array_red2*Multiplication_factor[index,1],ee_array_red3*Multiplication_factor[index,2]])
    eefit_blue = np.concatenate([ee_array_blue*Multiplication_factor[index,0],ee_array_blue2*Multiplication_factor[index,1],ee_array_blue3*Multiplication_factor[index,2]])
    
    if fitlinearwitherror == True:
        label1 = my_fits.fit_with_plot(ax3, x, efit_red, yerr = eefit_red, my_color = 'r', my_edgecolor='#ff3232', my_facecolor='#ff6666', my_linestyle = 'solid')
        label2 = my_fits.fit_with_plot(ax3, x, efit_blue, yerr = eefit_blue, my_color = 'g', my_edgecolor='#74C365', my_facecolor='#74C365', my_linestyle = 'solid')
    else:
        label1 = my_fits.fit_with_plot(ax3, x, efit_red, yerr = None, my_color = 'r', my_edgecolor='#ff3232', my_facecolor='#ff6666', my_linestyle = 'solid')
        label2 = my_fits.fit_with_plot(ax3, x, efit_blue, yerr = None, my_color = 'g', my_edgecolor='#74C365', my_facecolor='#74C365', my_linestyle = 'solid')

    legendloc = 'upper right'
    legend1 = ax3.legend([label1,label2], loc=legendloc)
    ax3.add_artist(legend1)    
    
    #Fitting medium
    bfit_red = np.concatenate([b_array_red*Multiplication_factor[index,0],b_array_red2*Multiplication_factor[index,1],b_array_red3*Multiplication_factor[index,2]])
    bfit_blue = np.concatenate([b_array_blue*Multiplication_factor[index,0],b_array_blue2*Multiplication_factor[index,1],b_array_blue3*Multiplication_factor[index,2]])
    
    befit_red = np.concatenate([be_array_red*Multiplication_factor[index,0],be_array_red2*Multiplication_factor[index,1],be_array_red3*Multiplication_factor[index,2]])
    befit_blue = np.concatenate([be_array_blue*Multiplication_factor[index,0],be_array_blue2*Multiplication_factor[index,1],be_array_blue3*Multiplication_factor[index,2]])    
        
    if fitlinearwitherror == True:
        my_fits.fit_with_plot(ax2, x, bfit_red, yerr = befit_red, my_color = 'r', my_edgecolor='#ff3232', my_facecolor='#ff6666', my_linestyle = 'solid')
        my_fits.fit_with_plot(ax2, x, bfit_blue, yerr = befit_blue, my_color = 'g', my_edgecolor='#74C365', my_facecolor='#74C365', my_linestyle = 'solid')
    else:
        my_fits.fit_with_plot(ax2, x, bfit_red, yerr = None, my_color = 'r', my_edgecolor='#ff3232', my_facecolor='#ff6666', my_linestyle = 'solid')
        my_fits.fit_with_plot(ax2, x, bfit_blue, yerr = None, my_color = 'g', my_edgecolor='#74C365', my_facecolor='#74C365', my_linestyle = 'solid')
    
    try: #Fitting short
        gfit_red = np.concatenate([g_array_red*Multiplication_factor[index,0],g_array_red2*Multiplication_factor[index,1],g_array_red3*Multiplication_factor[index,2]])
        gfit_blue = np.concatenate([g_array_blue*Multiplication_factor[index,0],g_array_blue2*Multiplication_factor[index,1],g_array_blue3*Multiplication_factor[index,2]])
        
        gefit_red = np.concatenate([ge_array_red*Multiplication_factor[index,0],ge_array_red2*Multiplication_factor[index,1],ge_array_red3*Multiplication_factor[index,2]])
        gefit_blue = np.concatenate([ge_array_blue*Multiplication_factor[index,0],ge_array_blue2*Multiplication_factor[index,1],ge_array_blue3*Multiplication_factor[index,2]])
        
        if fitlinearwitherror == True:
            my_fits.fit_with_plot(ax1, x, gfit_red, yerr = gefit_red, my_color = 'r', my_edgecolor='#ff3232', my_facecolor='#ff6666', my_linestyle = 'solid')
            my_fits.fit_with_plot(ax1, x, gfit_blue, yerr = gefit_blue, my_color = 'g', my_edgecolor='#74C365', my_facecolor='#74C365', my_linestyle = 'solid')
        else:
            my_fits.fit_with_plot(ax1, x, gfit_red, yerr = None, my_color = 'r', my_edgecolor='#ff3232', my_facecolor='#ff6666', my_linestyle = 'solid')
            my_fits.fit_with_plot(ax1, x, gfit_blue, yerr = None, my_color = 'g', my_edgecolor='#74C365', my_facecolor='#74C365', my_linestyle = 'solid')
            
    except:
        pass
    

    ax3.set_ylabel(r'Long $\tau$ ($\mu$s)',fontsize=fsizepl)
    ax3.tick_params(labelsize=fsizenb)
    #ax3.set_ylim(ymin=0)
    #ax3.legend(loc='best')
    ax2.legend(loc=legendloc)
    ax1.legend(loc=legendloc)
    ax2.set_ylabel(r'Medium $\tau$ ($\mu$s)',fontsize=fsizepl)
    ax2.tick_params(labelsize=fsizenb)
    #ax2.set_ylim(ymin=0)
    ax1.set_ylabel(r'Short $\tau$ ($\mu$s)',fontsize=fsizepl)
    ax1.tick_params(labelsize=fsizenb)
    #ax1.set_ylim(ymin=0)
    ax0.set_ylabel('$\chi^2$',fontsize=fsizepl)
    ax0.tick_params(labelsize=fsizenb)
    
    ax0.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)

    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    ##################################################### PAGE 2
    
    fig42= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig42.set_size_inches(1200./fig42.dpi,900./fig42.dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')    
    
    titulo = 'NaYF$_4$:Yb,Er UC NPs (sample by Andrea Pickel)\n (10 kV, 30 $\mu$m aperture, 1 $\mu$s time bins, 1.4 ms transient, cathodoluminescence green/red: $</>$ 593 nm) $\Rightarrow$ ' + description[index]
    plt.suptitle(titulo,fontsize=fsizetit)
    
    ax0 = plt.subplot2grid((7,4), (0,0), colspan=4, rowspan=1)
    
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.xaxis.set_ticks_position('bottom')
    ax0.yaxis.set_ticks_position('left')

    ax0.errorbar(x_vec, chiresult_red, fmt='ro',markersize=10,ls='-')  
    ax0.errorbar(x_vec, chiresult_blue, fmt='go',markersize=10,ls='-') 
    ax0.errorbar(x_vec3, chiresult_red3, fmt='rs',markersize=10,ls='--')  
    ax0.errorbar(x_vec3, chiresult_blue3, fmt='gs',markersize=10,ls='--')  
    ax0.errorbar(x_vec2, chiresult_red2, fmt='r^',markersize=10,ls='dotted')  
    ax0.errorbar(x_vec2, chiresult_blue2, fmt='g^',markersize=10,ls='dotted')
    
    ax0.set_ylabel('$\chi^2$',fontsize=fsizepl)
    ax0.tick_params(labelsize=fsizenb)
    
    ax0.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)
    
    ax2 = plt.subplot2grid((7,4), (2,0), colspan=2, rowspan=2)
    ax1 = plt.subplot2grid((7,4), (5,0), colspan=2, rowspan=2,sharex=ax2)
    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    ax2.errorbar(x_vec, e_array_red/b_array_red*Multiplication_factor[index,0],  fmt='ro',markersize=10,ls=None)
    ax2.errorbar(x_vec, e_array_blue/b_array_blue*Multiplication_factor[index,0], fmt='go', markersize=10,ls=None)
    ax2.errorbar(x_vec2, e_array_red2/b_array_red2*Multiplication_factor[index,1],  fmt='r^',markersize=10,ls=None)
    ax2.errorbar(x_vec2, e_array_blue2/b_array_blue2*Multiplication_factor[index,1], fmt='g^', markersize=10,ls=None)
    ax2.errorbar(x_vec3, e_array_red3/b_array_red3*Multiplication_factor[index,2],  fmt='rs',markersize=10,ls=None)
    ax2.errorbar(x_vec3, e_array_blue3/b_array_blue3*Multiplication_factor[index,2], fmt='gs', markersize=10,ls=None)
    
    ax1.errorbar(x_vec, (e_array_red/b_array_red)/ (e_array_red[RTindex[0]]/b_array_red[RTindex[0]])*Multiplication_factor[index,0],  fmt='ro',markersize=10,ls=None)
    ax1.errorbar(x_vec, (e_array_blue/b_array_blue)/(e_array_blue[RTindex[0]]/b_array_blue[RTindex[0]])*Multiplication_factor[index,0], fmt='go', markersize=10,ls=None)
    ax1.errorbar(x_vec2, (e_array_red2/b_array_red2)/(e_array_red2[RTindex[1]]/b_array_red2[RTindex[1]])*Multiplication_factor[index,1],  fmt='r^',markersize=10,ls=None)
    ax1.errorbar(x_vec2, (e_array_blue2/b_array_blue2)/ (e_array_blue2[RTindex[1]]/b_array_blue2[RTindex[1]])*Multiplication_factor[index,1], fmt='g^', markersize=10,ls=None)
    ax1.errorbar(x_vec3, (e_array_red3/b_array_red3)/ (e_array_red3[RTindex[2]]/b_array_red3[RTindex[2]])*Multiplication_factor[index,2],  fmt='rs',markersize=10,ls=None)
    ax1.errorbar(x_vec3, (e_array_blue3/b_array_blue3)/(e_array_blue3[RTindex[2]]/b_array_blue3[RTindex[2]])*Multiplication_factor[index,2], fmt='gs', markersize=10,ls=None)
    
    label1 = my_fits.fit_with_plot_small(ax2, x_vec[1:], e_array_red[1:]/b_array_red[1:]*Multiplication_factor[index,0], yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
    label2 = my_fits.fit_with_plot_small(ax2, x_vec[1:], e_array_blue[1:]/b_array_blue[1:]*Multiplication_factor[index,0], yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
    label3 = my_fits.fit_with_plot_small(ax2, x_vec2, e_array_red2/b_array_red2*Multiplication_factor[index,1], yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
    label4 = my_fits.fit_with_plot_small(ax2, x_vec2, e_array_blue2/b_array_blue2*Multiplication_factor[index,1], yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
    label5 = my_fits.fit_with_plot_small(ax2, x_vec3, e_array_red3/b_array_red3*Multiplication_factor[index,2], yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
    label6 = my_fits.fit_with_plot_small(ax2, x_vec3, e_array_blue3/b_array_blue3*Multiplication_factor[index,2], yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
    
    legend1 = ax2.legend([label1,label2,label3,label4,label5,label6], loc=legendloc)
    ax2.add_artist(legend1)  
    
    label1 = my_fits.fit_with_plot_small(ax1, x_vec[1:],(e_array_red[1:]/b_array_red[1:])/ (e_array_red[RTindex[0]]/b_array_red[RTindex[0]])*Multiplication_factor[index,0] , yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
    label2 = my_fits.fit_with_plot_small(ax1, x_vec[1:],(e_array_blue[1:]/b_array_blue[1:])/(e_array_blue[RTindex[0]]/b_array_blue[RTindex[0]])*Multiplication_factor[index,0] , yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
    label3 = my_fits.fit_with_plot_small(ax1, x_vec2,(e_array_red2/b_array_red2)/(e_array_red2[RTindex[1]]/b_array_red2[RTindex[1]])*Multiplication_factor[index,1] , yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
    label4 = my_fits.fit_with_plot_small(ax1, x_vec2,(e_array_blue2/b_array_blue2)/ (e_array_blue2[RTindex[1]]/b_array_blue2[RTindex[1]])*Multiplication_factor[index,1] , yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
    label5 = my_fits.fit_with_plot_small(ax1, x_vec3,(e_array_red3/b_array_red3)/ (e_array_red3[RTindex[2]]/b_array_red3[RTindex[2]])*Multiplication_factor[index,2], yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
    label6 = my_fits.fit_with_plot_small(ax1, x_vec3,(e_array_blue3/b_array_blue3)/(e_array_blue3[RTindex[2]]/b_array_blue3[RTindex[2]])*Multiplication_factor[index,2] , yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
    
    legend1 = ax1.legend([label1,label2,label3,label4,label5,label6], loc=legendloc)
    ax1.add_artist(legend1)  
    
    ax2.set_ylabel(r'$\tau$ large to medium',fontsize=fsizepl)
    ax2.tick_params(labelsize=fsizenb)
    ax1.set_ylabel(r'$\tau$ large to medium, norm. RT',fontsize=fsizepl)
    ax1.tick_params(labelsize=fsizenb)

    ax1.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)
    
    ax4 = plt.subplot2grid((7,4), (2,2), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((7,4), (5,2), colspan=2, rowspan=2,sharex=ax2)
    
    ax4.spines['left'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.xaxis.set_ticks_position('bottom')
    ax4.yaxis.set_ticks_position('right')
    
    ax3.spines['left'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.xaxis.set_ticks_position('bottom')
    ax3.yaxis.set_ticks_position('right')
    
    try:    
        ax4.errorbar(x_vec, e_array_red/g_array_red*Multiplication_factor[index,0],  fmt='ro',markersize=10,ls=None)
        ax4.errorbar(x_vec, e_array_blue/g_array_blue*Multiplication_factor[index,0], fmt='go', markersize=10,ls=None)
        ax4.errorbar(x_vec2, e_array_red2/g_array_red2*Multiplication_factor[index,1],  fmt='r^',markersize=10,ls=None)
        ax4.errorbar(x_vec2, e_array_blue2/g_array_blue2*Multiplication_factor[index,1], fmt='g^', markersize=10,ls=None)
        ax4.errorbar(x_vec3, e_array_red3/g_array_red3*Multiplication_factor[index,2],  fmt='rs',markersize=10,ls=None)
        ax4.errorbar(x_vec3, e_array_blue3/g_array_blue3*Multiplication_factor[index,2], fmt='gs', markersize=10,ls=None)
        
        label1 = my_fits.fit_with_plot_small(ax4, x_vec[1:], e_array_red[1:]/g_array_red[1:]*Multiplication_factor[index,0]  , yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
        label2 = my_fits.fit_with_plot_small(ax4, x_vec[1:], e_array_blue[1:]/g_array_blue[1:]*Multiplication_factor[index,0] , yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
        label3 = my_fits.fit_with_plot_small(ax4, x_vec2, e_array_red2/g_array_red2*Multiplication_factor[index,1] , yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
        label4 = my_fits.fit_with_plot_small(ax4, x_vec2, e_array_blue2/g_array_blue2*Multiplication_factor[index,1] , yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
        label5 = my_fits.fit_with_plot_small(ax4, x_vec3, e_array_red3/g_array_red3*Multiplication_factor[index,2] , yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
        label6 = my_fits.fit_with_plot_small(ax4, x_vec3, e_array_blue3/g_array_blue3*Multiplication_factor[index,2] , yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
    
        legend1 = ax4.legend([label1,label2,label3,label4,label5,label6], loc=legendloc)
        ax4.add_artist(legend1) 
        
        ax3.errorbar(x_vec, (e_array_red/g_array_red)/ (e_array_red[RTindex[0]]/g_array_red[RTindex[0]])*Multiplication_factor[index,0],  fmt='ro',markersize=10,ls=None)
        ax3.errorbar(x_vec, (e_array_blue/g_array_blue)/(e_array_blue[RTindex[0]]/g_array_blue[RTindex[0]])*Multiplication_factor[index,0], fmt='go', markersize=10,ls=None)
        ax3.errorbar(x_vec2, (e_array_red2/g_array_red2)/(e_array_red2[RTindex[1]]/g_array_red2[RTindex[1]])*Multiplication_factor[index,1],  fmt='r^',markersize=10,ls=None)
        ax3.errorbar(x_vec2, (e_array_blue2/g_array_blue2)/ (e_array_blue2[RTindex[1]]/g_array_blue2[RTindex[1]])*Multiplication_factor[index,1], fmt='g^', markersize=10,ls=None)
        ax3.errorbar(x_vec3, (e_array_red3/g_array_red3)/ (e_array_red3[RTindex[2]]/g_array_red3[RTindex[2]])*Multiplication_factor[index,2],  fmt='rs',markersize=10,ls=None)
        ax3.errorbar(x_vec3, (e_array_blue3/g_array_blue3)/(e_array_blue3[RTindex[2]]/g_array_blue3[RTindex[2]])*Multiplication_factor[index,2], fmt='gs', markersize=10,ls=None)
        
        label1 = my_fits.fit_with_plot_small(ax3, x_vec[1:], (e_array_red[1:]/g_array_red[1:])/ (e_array_red[RTindex[0]]/g_array_red[RTindex[0]])*Multiplication_factor[index,0]  , yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
        label2 = my_fits.fit_with_plot_small(ax3, x_vec[1:], (e_array_blue[1:]/g_array_blue[1:])/(e_array_blue[RTindex[0]]/g_array_blue[RTindex[0]])*Multiplication_factor[index,0], yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
        label3 = my_fits.fit_with_plot_small(ax3, x_vec2,    (e_array_red2/g_array_red2)/(e_array_red2[RTindex[1]]/g_array_red2[RTindex[1]])*Multiplication_factor[index,1]    , yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
        label4 = my_fits.fit_with_plot_small(ax3, x_vec2,    (e_array_blue2/g_array_blue2)/ (e_array_blue2[RTindex[1]]/g_array_blue2[RTindex[1]])*Multiplication_factor[index,1], yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
        label5 = my_fits.fit_with_plot_small(ax3, x_vec3,    (e_array_red3/g_array_red3)/ (e_array_red3[RTindex[2]]/g_array_red3[RTindex[2]])*Multiplication_factor[index,2], yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
        label6 = my_fits.fit_with_plot_small(ax3, x_vec3,    (e_array_blue3/g_array_blue3)/(e_array_blue3[RTindex[2]]/g_array_blue3[RTindex[2]])*Multiplication_factor[index,2], yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
    
        legend1 = ax3.legend([label1,label2,label3,label4,label5,label6], loc=legendloc)
        ax3.add_artist(legend1) 
    except:
        pass

    
    ax4.set_ylabel(r'$\tau$ large to short',fontsize=fsizepl)
    ax4.tick_params(labelsize=fsizenb)
    ax4.yaxis.set_label_position("right")
    ax3.set_ylabel(r'$\tau$ large to short, norm. RT',fontsize=fsizepl)
    ax3.tick_params(labelsize=fsizenb)
    ax3.yaxis.set_label_position("right")

    ax3.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)
    
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    
    ##################################################### PAGE 3
    
    fig42= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig42.set_size_inches(1200./fig42.dpi,900./fig42.dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')    
    
    titulo = 'NaYF$_4$:Yb,Er UC NPs (sample by Andrea Pickel)\n (10 kV, 30 $\mu$m aperture, 1 $\mu$s time bins, 1.4 ms transient, cathodoluminescence green/red: $</>$ 593 nm) $\Rightarrow$ ' + description[index]
    plt.suptitle(titulo,fontsize=fsizetit)
    
    ax0 = plt.subplot2grid((7,4), (0,0), colspan=4, rowspan=1)
    
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.xaxis.set_ticks_position('bottom')
    ax0.yaxis.set_ticks_position('left')

    ax0.errorbar(x_vec, chiresult_red, fmt='ro',markersize=10,ls='-')  
    ax0.errorbar(x_vec, chiresult_blue, fmt='go',markersize=10,ls='-') 
    ax0.errorbar(x_vec3, chiresult_red3, fmt='rs',markersize=10,ls='--')  
    ax0.errorbar(x_vec3, chiresult_blue3, fmt='gs',markersize=10,ls='--')  
    ax0.errorbar(x_vec2, chiresult_red2, fmt='r^',markersize=10,ls='dotted')  
    ax0.errorbar(x_vec2, chiresult_blue2, fmt='g^',markersize=10,ls='dotted')
    
    ax0.set_ylabel('$\chi^2$',fontsize=fsizepl)
    ax0.tick_params(labelsize=fsizenb)
    
    ax0.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)
    
    ax2 = plt.subplot2grid((7,4), (2,0), colspan=2, rowspan=2)
    ax1 = plt.subplot2grid((7,4), (5,0), colspan=2, rowspan=2,sharex=ax2)
    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    try:
        ax2.errorbar(x_vec, b_array_red/g_array_red*Multiplication_factor[index,0],  fmt='ro',markersize=10,ls=None)
        ax2.errorbar(x_vec, b_array_blue/g_array_blue*Multiplication_factor[index,0], fmt='go', markersize=10,ls=None)
        ax2.errorbar(x_vec2, b_array_red2/g_array_red2*Multiplication_factor[index,1],  fmt='r^',markersize=10,ls=None)
        ax2.errorbar(x_vec2, b_array_blue2/g_array_blue2*Multiplication_factor[index,1], fmt='g^', markersize=10,ls=None)
        ax2.errorbar(x_vec3, b_array_red3/g_array_red3*Multiplication_factor[index,2],  fmt='rs',markersize=10,ls=None)
        ax2.errorbar(x_vec3, b_array_blue3/g_array_blue3*Multiplication_factor[index,2], fmt='gs', markersize=10,ls=None)
        
        label1 = my_fits.fit_with_plot_small(ax2, x_vec[1:], b_array_red[1:]/g_array_red[1:]*Multiplication_factor[index,0] , yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
        label2 = my_fits.fit_with_plot_small(ax2, x_vec[1:], b_array_blue[1:]/g_array_blue[1:]*Multiplication_factor[index,0] , yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
        label3 = my_fits.fit_with_plot_small(ax2, x_vec2,    b_array_red2/g_array_red2*Multiplication_factor[index,1]  , yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
        label4 = my_fits.fit_with_plot_small(ax2, x_vec2,    b_array_blue2/g_array_blue2*Multiplication_factor[index,1], yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
        label5 = my_fits.fit_with_plot_small(ax2, x_vec3,    b_array_red3/g_array_red3*Multiplication_factor[index,2], yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
        label6 = my_fits.fit_with_plot_small(ax2, x_vec3,    b_array_blue3/g_array_blue3*Multiplication_factor[index,2], yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
    
        legend1 = ax2.legend([label1,label2,label3,label4,label5,label6], loc=legendloc)
        ax2.add_artist(legend1) 
        
        ax1.errorbar(x_vec, (b_array_red/g_array_red)/ (b_array_red[RTindex[0]]/g_array_red[RTindex[0]])*Multiplication_factor[index,0],  fmt='ro',markersize=10,ls=None)
        ax1.errorbar(x_vec, (b_array_blue/g_array_blue)/(b_array_blue[RTindex[0]]/g_array_blue[RTindex[0]])*Multiplication_factor[index,0], fmt='go', markersize=10,ls=None)
        ax1.errorbar(x_vec2, (b_array_red2/g_array_red2)/(b_array_red2[RTindex[1]]/g_array_red2[RTindex[1]])*Multiplication_factor[index,1],  fmt='r^',markersize=10,ls=None)
        ax1.errorbar(x_vec2, (b_array_blue2/g_array_blue2)/ (b_array_blue2[RTindex[1]]/g_array_blue2[RTindex[1]])*Multiplication_factor[index,1], fmt='g^', markersize=10,ls=None)
        ax1.errorbar(x_vec3, (b_array_red3/g_array_red3)/ (b_array_red3[RTindex[2]]/g_array_red3[RTindex[2]])*Multiplication_factor[index,2],  fmt='rs',markersize=10,ls=None)
        ax1.errorbar(x_vec3, (b_array_blue3/g_array_blue3)/(b_array_blue3[RTindex[2]]/g_array_blue3[RTindex[2]])*Multiplication_factor[index,2], fmt='gs', markersize=10,ls=None)
        
        label1 = my_fits.fit_with_plot_small(ax1, x_vec[1:],  (b_array_red[1:]/g_array_red[1:])/ (b_array_red[RTindex[0]]/g_array_red[RTindex[0]])*Multiplication_factor[index,0] , yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
        label2 = my_fits.fit_with_plot_small(ax1, x_vec[1:],  (b_array_blue[1:]/g_array_blue[1:])/(b_array_blue[RTindex[0]]/g_array_blue[RTindex[0]])*Multiplication_factor[index,0] , yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
        label3 = my_fits.fit_with_plot_small(ax1, x_vec2,     (b_array_red2/g_array_red2)/(b_array_red2[RTindex[1]]/g_array_red2[RTindex[1]])*Multiplication_factor[index,1], yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
        label4 = my_fits.fit_with_plot_small(ax1, x_vec2,     (b_array_blue2/g_array_blue2)/ (b_array_blue2[RTindex[1]]/g_array_blue2[RTindex[1]])*Multiplication_factor[index,1] , yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
        label5 = my_fits.fit_with_plot_small(ax1, x_vec3,     (b_array_red3/g_array_red3)/ (b_array_red3[RTindex[2]]/g_array_red3[RTindex[2]])*Multiplication_factor[index,2] , yerr = None, my_color = 'r', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
        label6 = my_fits.fit_with_plot_small(ax1, x_vec3,     (b_array_blue3/g_array_blue3)/(b_array_blue3[RTindex[2]]/g_array_blue3[RTindex[2]])*Multiplication_factor[index,2]  , yerr = None, my_color = 'g', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
    
        legend1 = ax1.legend([label1,label2,label3,label4,label5,label6], loc=legendloc)
        ax1.add_artist(legend1) 
    except:
        pass
        
    ax2.set_ylabel(r'$\tau$ medium to short',fontsize=fsizepl)
    ax2.tick_params(labelsize=fsizenb)
    ax1.set_ylabel(r'$\tau$ medium to short, norm. RT',fontsize=fsizepl)
    ax1.tick_params(labelsize=fsizenb)

    ax1.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)
    
    ax4 = plt.subplot2grid((7,4), (2,2), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((7,4), (5,2), colspan=2, rowspan=2,sharex=ax2)
    
    ax4.spines['left'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.xaxis.set_ticks_position('bottom')
    ax4.yaxis.set_ticks_position('right')
    
    ax3.spines['left'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.xaxis.set_ticks_position('bottom')
    ax3.yaxis.set_ticks_position('right')
    
    ax4.errorbar(x_vec, e_array_red/e_array_blue*Multiplication_factor[index,0],  fmt='ko',markersize=10,ls=None)
    ax4.errorbar(x_vec, b_array_red/b_array_blue*Multiplication_factor[index,0], fmt='ko', markersize=7,ls=None)
    
    ax4.errorbar(x_vec2, e_array_red2/e_array_blue2*Multiplication_factor[index,1],  fmt='k^',markersize=10,ls=None)
    ax4.errorbar(x_vec2, b_array_red2/b_array_blue2*Multiplication_factor[index,1], fmt='k^', markersize=7,ls=None)
    
    ax4.errorbar(x_vec3, e_array_red3/e_array_blue3*Multiplication_factor[index,2],  fmt='ks',markersize=10,ls=None)
    ax4.errorbar(x_vec3, b_array_red3/b_array_blue3*Multiplication_factor[index,2], fmt='ks', markersize=7,ls=None)
    
    label1 = my_fits.fit_with_plot_small(ax4, x_vec[1:], e_array_red[1:]/e_array_blue[1:]*Multiplication_factor[index,0]  , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
    label2 = my_fits.fit_with_plot_small(ax4, x_vec[1:], b_array_red[1:]/b_array_blue[1:]*Multiplication_factor[index,0]  , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
    label3 = my_fits.fit_with_plot_small(ax4, x_vec2, e_array_red2/e_array_blue2*Multiplication_factor[index,1]    , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
    label4 = my_fits.fit_with_plot_small(ax4, x_vec2, b_array_red2/b_array_blue2*Multiplication_factor[index,1]     , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
    label5 = my_fits.fit_with_plot_small(ax4, x_vec3, e_array_red3/e_array_blue3*Multiplication_factor[index,2]   , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
    label6 = my_fits.fit_with_plot_small(ax4, x_vec3, b_array_red3/b_array_blue3*Multiplication_factor[index,2]   , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
    
    ax3.errorbar(x_vec, (e_array_red/e_array_blue)/(e_array_red[RTindex[0]]/e_array_blue[RTindex[0]])*Multiplication_factor[index,0],  fmt='ko',markersize=10,ls=None)
    ax3.errorbar(x_vec, (b_array_red/b_array_blue)/(b_array_red[RTindex[0]]/b_array_blue[RTindex[0]])*Multiplication_factor[index,0], fmt='ko', markersize=7,ls=None)
    
    ax3.errorbar(x_vec2, (e_array_red2/e_array_blue2)/(e_array_red2[RTindex[1]]/e_array_blue2[RTindex[1]])*Multiplication_factor[index,1],  fmt='k^',markersize=10,ls=None)
    ax3.errorbar(x_vec2, (b_array_red2/b_array_blue2)/(b_array_red2[RTindex[1]]/b_array_blue2[RTindex[1]])*Multiplication_factor[index,1], fmt='k^', markersize=7,ls=None)
    
    ax3.errorbar(x_vec3, (e_array_red3/e_array_blue3)/(e_array_red3[RTindex[2]]/e_array_blue3[RTindex[2]])*Multiplication_factor[index,2],  fmt='ks',markersize=10,ls=None)
    ax3.errorbar(x_vec3, (b_array_red3/b_array_blue3)/(b_array_red3[RTindex[2]]/b_array_blue3[RTindex[2]])*Multiplication_factor[index,2], fmt='ks', markersize=7,ls=None)
    
    label10 = my_fits.fit_with_plot_small(ax3, x_vec[1:], (e_array_red[1:]/e_array_blue[1:])/(e_array_red[RTindex[0]]/e_array_blue[RTindex[0]])*Multiplication_factor[index,0] , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
    label20 = my_fits.fit_with_plot_small(ax3, x_vec[1:], (b_array_red[1:]/b_array_blue[1:])/(b_array_red[RTindex[0]]/b_array_blue[RTindex[0]])*Multiplication_factor[index,0]  , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
    label30= my_fits.fit_with_plot_small(ax3, x_vec2,    (e_array_red2/e_array_blue2)/(e_array_red2[RTindex[1]]/e_array_blue2[RTindex[1]])*Multiplication_factor[index,1]  , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
    label40 = my_fits.fit_with_plot_small(ax3, x_vec2,    (b_array_red2/b_array_blue2)/(b_array_red2[RTindex[1]]/b_array_blue2[RTindex[1]])*Multiplication_factor[index,1] , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
    label50 = my_fits.fit_with_plot_small(ax3, x_vec3,    (e_array_red3/e_array_blue3)/(e_array_red3[RTindex[2]]/e_array_blue3[RTindex[2]])*Multiplication_factor[index,2] , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
    label60 = my_fits.fit_with_plot_small(ax3, x_vec3,    (b_array_red3/b_array_blue3)/(b_array_red3[RTindex[2]]/b_array_blue3[RTindex[2]])*Multiplication_factor[index,2], yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
    
   
    
    try:
        ax4.errorbar(x_vec, g_array_red/g_array_blue*Multiplication_factor[index,0], fmt='ko', markersize=4,ls=None)
        ax4.errorbar(x_vec2, g_array_red2/g_array_blue2*Multiplication_factor[index,1], fmt='k^', markersize=4,ls=None)
        ax4.errorbar(x_vec3, g_array_red3/g_array_blue3*Multiplication_factor[index,2], fmt='ks', markersize=4,ls=None)
        ax3.errorbar(x_vec, (g_array_red/g_array_blue)/(g_array_red[RTindex[0]]/g_array_blue[RTindex[0]])*Multiplication_factor[index,0], fmt='ko', markersize=4,ls=None)
        ax3.errorbar(x_vec2, (g_array_red2/g_array_blue2)/(g_array_red2[RTindex[1]]/g_array_blue2[RTindex[1]])*Multiplication_factor[index,1], fmt='k^', markersize=4,ls=None)
        ax3.errorbar(x_vec3, (g_array_red3/g_array_blue3)/(g_array_red3[RTindex[2]]/g_array_blue3[RTindex[2]])*Multiplication_factor[index,2], fmt='ks', markersize=4,ls=None)
        
        label7 = my_fits.fit_with_plot_small(ax4, x_vec[1:],  g_array_red[1:]/g_array_blue[1:]*Multiplication_factor[index,0] , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
        label8 = my_fits.fit_with_plot_small(ax4, x_vec2,     g_array_red2/g_array_blue2*Multiplication_factor[index,1] , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
        label9 = my_fits.fit_with_plot_small(ax4, x_vec3,     g_array_red3/g_array_blue3*Multiplication_factor[index,2]  , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
        label70 = my_fits.fit_with_plot_small(ax3, x_vec[1:], (g_array_red[1:]/g_array_blue[1:])/(g_array_red[RTindex[0]]/g_array_blue[RTindex[0]])*Multiplication_factor[index,0]   , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
        label80 = my_fits.fit_with_plot_small(ax3, x_vec2,    (g_array_red2/g_array_blue2)/(g_array_red2[RTindex[1]]/g_array_blue2[RTindex[1]])*Multiplication_factor[index,1]  , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
        label90 = my_fits.fit_with_plot_small(ax3, x_vec3,    (g_array_red3/g_array_blue3)/(g_array_red3[RTindex[2]]/g_array_blue3[RTindex[2]])*Multiplication_factor[index,2] , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '--') 
        
        legend1 = ax4.legend([label1,label2,label3,label4,label5,label6,label7,label8,label9], loc=legendloc)
        ax4.add_artist(legend1) 
        legend1 = ax3.legend([label10,label20,label30,label40,label50,label60,label70,label80,label90], loc=legendloc)
        ax3.add_artist(legend1) 
    except:
        legend1 = ax4.legend([label1,label2,label3,label4,label5,label6], loc=legendloc)
        ax4.add_artist(legend1) 
        legend1 = ax3.legend([label10,label20,label30,label40,label50,label60], loc=legendloc)
        ax3.add_artist(legend1) 
    
    ax4.set_ylabel(r'$\tau$ red to green',fontsize=fsizepl)
    ax4.tick_params(labelsize=fsizenb)
    ax4.yaxis.set_label_position("right")
    ax3.set_ylabel(r'$\tau$ red to green, norm. RT',fontsize=fsizepl)
    ax3.tick_params(labelsize=fsizenb)
    ax3.yaxis.set_label_position("right")

    ax3.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)
    
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
  
    ##################################################### PAGE 4

    fig42= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig42.set_size_inches(1200./fig42.dpi,900./fig42.dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')    
    
    titulo = 'NaYF$_4$:Yb,Er UC NPs (sample by Andrea Pickel)\n (10 kV, 30 $\mu$m aperture, 1 $\mu$s time bins, 1.4 ms transient, cathodoluminescence green/red: $</>$ 593 nm) $\Rightarrow$ ' + description[index]
    plt.suptitle(titulo,fontsize=fsizetit)
    
    ax2 = plt.subplot2grid((2,4), (0,0), colspan=2, rowspan=1)
    ax1 = plt.subplot2grid((2,4), (1,0), colspan=2, sharex=ax2)
    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')

    ax2.errorbar(x_vec, (red_int_array/blue_int_array)*Multiplication_factor[index,0], fmt='ko',markersize=10,ls=None)
    ax2.errorbar(x_vec2, (red_int_array2/blue_int_array2)*Multiplication_factor[index,1], fmt='k^',markersize=10,ls=None)
    ax2.errorbar(x_vec3, (red_int_array3/blue_int_array3)*Multiplication_factor[index,2], fmt='ks',markersize=10,ls=None)
    
    label1 = my_fits.fit_with_plot_small(ax2, x_vec[1:],  (red_int_array[1:]/blue_int_array[1:])*Multiplication_factor[index,0]  , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
    label2 = my_fits.fit_with_plot_small(ax2, x_vec2, (red_int_array2/blue_int_array2)*Multiplication_factor[index,1]   , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
    label3 = my_fits.fit_with_plot_small(ax2, x_vec3, (red_int_array3/blue_int_array3)*Multiplication_factor[index,2]   , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
    
    legend1 = ax2.legend([label1,label2,label3], loc=legendloc)
    ax2.add_artist(legend1) 
    
    ax1.errorbar(x_vec, (red_int_array/blue_int_array)/(red_int_array[RTindex[0]]/blue_int_array[RTindex[0]])*Multiplication_factor[index,0], fmt='ko',markersize=10,ls=None)
    ax1.errorbar(x_vec2, (red_int_array2/blue_int_array2)/(red_int_array2[RTindex[1]]/blue_int_array2[RTindex[1]])*Multiplication_factor[index,1], fmt='k^',markersize=10,ls=None)
    ax1.errorbar(x_vec3, (red_int_array3/blue_int_array3)/(red_int_array3[RTindex[2]]/blue_int_array3[RTindex[2]])*Multiplication_factor[index,2], fmt='ks',markersize=10,ls=None)
    
    label1 = my_fits.fit_with_plot_small(ax1, x_vec[1:], (red_int_array[1:]/blue_int_array[1:])/(red_int_array[RTindex[0]]/blue_int_array[RTindex[0]])*Multiplication_factor[index,0]  , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
    label2 = my_fits.fit_with_plot_small(ax1, x_vec2,    (red_int_array2/blue_int_array2)/(red_int_array2[RTindex[1]]/blue_int_array2[RTindex[1]])*Multiplication_factor[index,1] , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
    label3 = my_fits.fit_with_plot_small(ax1, x_vec3,    (red_int_array3/blue_int_array3)/(red_int_array3[RTindex[2]]/blue_int_array3[RTindex[2]])*Multiplication_factor[index,2] , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
    
    legend1 = ax1.legend([label1,label2,label3], loc=legendloc)
    ax1.add_artist(legend1) 

    ax2.set_ylabel('Intensity ratio red to green',fontsize=fsizepl)
    ax2.tick_params(labelsize=fsizenb)
    ax1.set_ylabel('Intensity ratio red to green, norm. RT',fontsize=fsizepl)
    ax1.tick_params(labelsize=fsizenb)

    ax1.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)
    
    ax4 = plt.subplot2grid((2,4), (0,2), colspan=2, rowspan=1)
    ax3 = plt.subplot2grid((2,4), (1,2), colspan=2, sharex=ax2)
    
    ax4.spines['left'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.xaxis.set_ticks_position('bottom')
    ax4.yaxis.set_ticks_position('right')
    
    ax3.spines['left'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.xaxis.set_ticks_position('bottom')
    ax3.yaxis.set_ticks_position('right')

    ax4.errorbar(x_vec, (red_int_array-blue_int_array)/(red_int_array+blue_int_array)*Multiplication_factor[index,0], fmt='ko',markersize=10,ls=None)
    ax4.errorbar(x_vec2, (red_int_array2-blue_int_array2)/(red_int_array2+blue_int_array2)*Multiplication_factor[index,1], fmt='k^',markersize=10,ls=None)
    ax4.errorbar(x_vec3,(red_int_array3-blue_int_array3)/(red_int_array3+blue_int_array3)*Multiplication_factor[index,2], fmt='ks',markersize=10,ls=None)
    
    label1 = my_fits.fit_with_plot_small(ax4, x_vec[1:],  (red_int_array[1:]-blue_int_array[1:])/(red_int_array[1:]+blue_int_array[1:])*Multiplication_factor[index,0]  , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
    label2 = my_fits.fit_with_plot_small(ax4, x_vec2,     (red_int_array2-blue_int_array2)/(red_int_array2+blue_int_array2)*Multiplication_factor[index,1]  , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
    label3 = my_fits.fit_with_plot_small(ax4, x_vec3,     (red_int_array3-blue_int_array3)/(red_int_array3+blue_int_array3)*Multiplication_factor[index,2]  , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
    
    legend1 = ax4.legend([label1,label2,label3], loc=legendloc)
    ax4.add_artist(legend1) 
    
    ax3.errorbar(x_vec, ((red_int_array-blue_int_array)/(red_int_array+blue_int_array))/((red_int_array[RTindex[0]]-blue_int_array[RTindex[0]])/(red_int_array[RTindex[0]]+blue_int_array[RTindex[0]]))*Multiplication_factor[index,0], fmt='ko',markersize=10,ls=None)
    ax3.errorbar(x_vec2, ((red_int_array2-blue_int_array2)/(red_int_array2+blue_int_array2))/ ((red_int_array2[RTindex[1]]-blue_int_array2[RTindex[1]])/(red_int_array2[RTindex[1]]+blue_int_array2[RTindex[1]]))*Multiplication_factor[index,1], fmt='k^',markersize=10,ls=None)
    ax3.errorbar(x_vec3,((red_int_array3-blue_int_array3)/(red_int_array3+blue_int_array3))/((red_int_array3[RTindex[2]]-blue_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]+blue_int_array3[RTindex[2]]))*Multiplication_factor[index,2], fmt='ks',markersize=10,ls=None)

    label1 = my_fits.fit_with_plot_small(ax3, x_vec[1:], ((red_int_array[1:]-blue_int_array[1:])/(red_int_array[1:]+blue_int_array[1:]))/((red_int_array[RTindex[0]]-blue_int_array[RTindex[0]])/(red_int_array[RTindex[0]]+blue_int_array[RTindex[0]]))*Multiplication_factor[index,0]   , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '-')
    label2 = my_fits.fit_with_plot_small(ax3, x_vec2,    ((red_int_array2-blue_int_array2)/(red_int_array2+blue_int_array2))/ ((red_int_array2[RTindex[1]]-blue_int_array2[RTindex[1]])/(red_int_array2[RTindex[1]]+blue_int_array2[RTindex[1]]))*Multiplication_factor[index,1]  , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = 'dotted')
    label3 = my_fits.fit_with_plot_small(ax3, x_vec3,    ((red_int_array3-blue_int_array3)/(red_int_array3+blue_int_array3))/((red_int_array3[RTindex[2]]-blue_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]+blue_int_array3[RTindex[2]]))*Multiplication_factor[index,2] , yerr = None, my_color = 'k', my_edgecolor=None, my_facecolor=None, my_linestyle = '--')
    
    legend1 = ax3.legend([label1,label2,label3], loc=legendloc)
    ax3.add_artist(legend1) 

    ax4.set_ylabel('Visibility red to green',fontsize=fsizepl)
    ax4.tick_params(labelsize=fsizenb)
    ax4.yaxis.set_label_position("right")
    ax3.set_ylabel('Visibility red to green, norm. RT',fontsize=fsizepl)
    ax3.tick_params(labelsize=fsizenb)
    ax3.yaxis.set_label_position("right")

    ax3.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)
    
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    
    if fitlinearwitherror == True:
        pref4 = '_linearfitweighted'
    else:
        pref4 = '_linearfitnotweighted'
    
    prefix = 'varC_' + pref2 + pref3 + pref4
    multipage_longer(prefix + '.pdf',dpi=80)
   
    
klkklkllkllk   
    
