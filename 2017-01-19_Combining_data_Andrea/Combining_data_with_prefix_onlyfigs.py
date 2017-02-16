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

#import warnings
#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

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
    
   
    ######## MEDIUM ZOOM
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[2]))
    leto = ['N102pt5', 'N92pt9', 'N66pt4', 'N72pt8', 'N60pt6', 'N48pt5', 'N39pt8', 'N31pt2']

    import boe_bar as sb

    fsizenb = 20
    fig,ax = plt.subplots()
    #axc0 = plt.subplot2grid((nolines,noplots), (0,3), colspan=1, rowspan=1)
    se = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[4] +'SEchannel.npz',mmap_mode='r') 
    xlen = se['data'].shape[0]
    ylen = se['data'].shape[1] 
    delx = 0#+28
    dely = 0#00
    xval = 144
    yval = 142
    plt.imshow(se['data'][np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely],cmap=cm.Greys_r)
    plt.axis('off')
    sbar = sb.AnchoredScaleBar(ax.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4, my_fontsize = fsizenb)
    ax.add_artist(sbar)
    fig.savefig('60C.png') 
    

#    axc1 = plt.subplot2grid((nolines,noplots), (0,4), colspan=1, rowspan=1)
#    se = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[5] +'SEchannel.npz',mmap_mode='r') 
#    delx = 0#+28
#    dely = 0#00
#    xval = 133
#    yval = 122
#    plt.imshow(se['data'][np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely],cmap=cm.Greys_r)
#    plt.axis('off')
#    sbar = sb.AnchoredScaleBar(axc1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#    axc1.add_artist(sbar)
#    
#    axc2 = plt.subplot2grid((nolines,noplots), (0,6), colspan=1, rowspan=1)
#    se = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[6] +'SEchannel.npz',mmap_mode='r') 
#    delx = 0#+28
#    dely = 0#00
#    xval = 135
#    yval = 105
#    plt.imshow(se['data'][np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely],cmap=cm.Greys_r)
#    plt.axis('off')
#    sbar = sb.AnchoredScaleBar(axc2.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#    axc2.add_artist(sbar)
#    
#    axc3 = plt.subplot2grid((nolines,noplots), (0,10), colspan=1, rowspan=1)
#    se = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[7] +'SEchannel.npz',mmap_mode='r')
#    delx = 0#+28
#    dely = 0#00
#    xval = 144
#    yval = 120
#    plt.imshow(se['data'][np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely],cmap=cm.Greys_r)
#    plt.axis('off')
#    sbar = sb.AnchoredScaleBar(axc3.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#    axc3.add_artist(sbar)

    
   
    
klkklkllkllk   
    
