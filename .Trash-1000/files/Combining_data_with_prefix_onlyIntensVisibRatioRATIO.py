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
from uncertainties import unumpy
##############################################################################
##############################################################################
#6 different prefixes

#dotriple = True
dotriple = False


#fitlinearwitherror = True
fitlinearwitherror = False
    
def do_pic(ax3,fig1):
    
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
    
#    B_array_red= np.load(prefix + 'B_array_red.npz')
#    b_array_red = B_array_red['data']  
#    Be_array_red = np.load(prefix +'Be_array_red.npz')
#    be_array_red = Be_array_red['data'] 
#    E_array_red = np.load(prefix +'E_array_red.npz')
#    e_array_red = E_array_red['data']   
#    Ee_array_red = np.load(prefix +'Ee_array_red.npz')
#    ee_array_red = Ee_array_red['data']  
#    try: 
#        G_array_red= np.load(prefix +'G_array_red.npz')
#        g_array_red = G_array_red['data']  
#        Ge_array_red = np.load(prefix +'Ge_array_red.npz')
#        ge_array_red = Ge_array_red['data'] 
#    except:
#        pass
#    
#    B_array_blue= np.load(prefix +'B_array_blue.npz')
#    b_array_blue = B_array_blue['data']  
#    Be_array_blue = np.load(prefix +'Be_array_blue.npz')
#    be_array_blue = Be_array_blue['data'] 
#    E_array_blue = np.load(prefix +'E_array_blue.npz')
#    e_array_blue = E_array_blue['data']   
#    Ee_array_blue = np.load(prefix +'Ee_array_blue.npz')
#    ee_array_blue = Ee_array_blue['data'] 
#    try:
#        G_array_blue= np.load(prefix +'G_array_blue.npz')
#        g_array_blue = G_array_blue['data']  
#        Ge_array_blue = np.load(prefix +'Ge_array_blue.npz')
#        ge_array_blue = Ge_array_blue['data'] 
#    except:
#        pass
#    
#    Chiresult_red = np.load(prefix +'Chiresult_red.npz')
#    chiresult_red = Chiresult_red['data'] 
#    Chiresult_blue = np.load(prefix +'Chiresult_blue.npz')
#    chiresult_blue = Chiresult_blue['data'] 
#    
#    ######## SMALLEST ZOOM
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
    
#    B_array_red2= np.load(prefix +'B_array_red.npz')
#    b_array_red2 = B_array_red2['data']  
#    Be_array_red2 = np.load(prefix +'Be_array_red.npz')
#    be_array_red2 = Be_array_red2['data'] 
#    E_array_red2 = np.load(prefix +'E_array_red.npz')
#    e_array_red2 = E_array_red2['data']   
#    Ee_array_red2 = np.load(prefix +'Ee_array_red.npz')
#    ee_array_red2 = Ee_array_red2['data']   
#    try:
#        G_array_red2 = np.load(prefix +'G_array_red.npz')
#        g_array_red2 = G_array_red2['data']   
#        Ge_array_red2 = np.load(prefix +'Ge_array_red.npz')
#        ge_array_red2 = Ge_array_red2['data']  
#    except:
#        pass
#    
#    B_array_blue2= np.load(prefix +'B_array_blue.npz')
#    b_array_blue2 = B_array_blue2['data']  
#    Be_array_blue2 = np.load(prefix +'Be_array_blue.npz')
#    be_array_blue2 = Be_array_blue2['data'] 
#    E_array_blue2 = np.load(prefix +'E_array_blue.npz')
#    e_array_blue2 = E_array_blue2['data']   
#    Ee_array_blue2 = np.load(prefix +'Ee_array_blue.npz')
#    ee_array_blue2 = Ee_array_blue2['data'] 
#    try:
#        G_array_blue2 = np.load(prefix +'G_array_blue.npz')
#        g_array_blue2 = G_array_blue2['data']   
#        Ge_array_blue2 = np.load(prefix +'Ge_array_blue.npz')
#        ge_array_blue2 = Ge_array_blue2['data']  
#    except:
#        pass
#    
#    Chiresult_red2 = np.load(prefix +'Chiresult_red.npz')
#    chiresult_red2 = Chiresult_red2['data'] 
#    Chiresult_blue2 = np.load(prefix +'Chiresult_blue.npz')
#    chiresult_blue2 = Chiresult_blue2['data'] 
#    
    
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
    
#    B_array_red3= np.load(prefix +'B_array_red.npz')
#    b_array_red3 = B_array_red3['data']  
#    Be_array_red3 = np.load(prefix +'Be_array_red.npz')
#    be_array_red3 = Be_array_red3['data'] 
#    E_array_red3 = np.load(prefix +'E_array_red.npz')
#    e_array_red3 = E_array_red3['data']   
#    Ee_array_red3 = np.load(prefix +'Ee_array_red.npz')
#    ee_array_red3 = Ee_array_red3['data']   
#    try:
#        G_array_red3 = np.load(prefix +'G_array_red.npz')
#        g_array_red3 = G_array_red3['data']   
#        Ge_array_red3 = np.load(prefix +'Ge_array_red.npz')
#        ge_array_red3 = Ge_array_red3['data']  
#    except:
#        pass
#    
#    B_array_blue3= np.load(prefix +'B_array_blue.npz')
#    b_array_blue3 = B_array_blue3['data']  
#    Be_array_blue3 = np.load(prefix +'Be_array_blue.npz')
#    be_array_blue3 = Be_array_blue3['data'] 
#    E_array_blue3 = np.load(prefix +'E_array_blue.npz')
#    e_array_blue3 = E_array_blue3['data']   
#    Ee_array_blue3 = np.load(prefix +'Ee_array_blue.npz')
#    ee_array_blue3 = Ee_array_blue3['data'] 
#    try:
#        G_array_blue3 = np.load(prefix +'G_array_blue.npz')
#        g_array_blue3 = G_array_blue3['data']   
#        Ge_array_blue3 = np.load(prefix +'Ge_array_blue.npz')
#        ge_array_blue3 = Ge_array_blue3['data'] 
#    except:
#        pass
#    
#    Chiresult_red3 = np.load(prefix +'Chiresult_red.npz')
#    chiresult_red3 = Chiresult_red3['data'] 
#    Chiresult_blue3 = np.load(prefix +'Chiresult_blue.npz')
#    chiresult_blue3 = Chiresult_blue3['data'] 
    
    
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
    
    for index in np.arange(0,1): #Multiplication_factor.shape[0]):
    
    
         ######## DELETE POINTS
    
        if index == 0: #of course, only delete in first pass
            ################### LARGEST ZOOM
            todel = [0,3,6] #using [1,2,4,5]
            red_int_array = np.delete(red_int_array, todel)
            blue_int_array = np.delete(blue_int_array, todel)
            red_std_array = np.delete(red_std_array, todel)
            blue_std_array = np.delete(blue_std_array, todel)
#            b_array_red = np.delete(b_array_red, todel)
#            be_array_red = np.delete(be_array_red, todel)
#            e_array_red = np.delete(e_array_red, todel)
#            ee_array_red = np.delete(ee_array_red, todel)
#            b_array_blue = np.delete(b_array_blue, todel)
#            be_array_blue = np.delete(be_array_blue, todel)
#            e_array_blue = np.delete(e_array_blue, todel)
#            ee_array_blue = np.delete(ee_array_blue, todel)
#            try:
#                g_array_red = np.delete(g_array_red, todel)
#                ge_array_red = np.delete(ge_array_red, todel)
#                g_array_blue = np.delete(g_array_blue, todel)
#                ge_array_blue = np.delete(ge_array_blue, todel)
#            except:
#                pass
            il_data = np.delete(il_data, todel)
            il_data_std = np.delete(il_data_std, todel)
#            chiresult_red = np.delete(chiresult_red, todel)
#            chiresult_blue = np.delete(chiresult_blue, todel)
        
            ################### MEDIUM ZOOM
            todel = [0,1,2,3]#  using [4,5,6,7]
            red_int_array3 = np.delete(red_int_array3, todel)
            blue_int_array3 = np.delete(blue_int_array3, todel)
            red_std_array3 = np.delete(red_std_array3, todel)
            blue_std_array3 = np.delete(blue_std_array3, todel)
#            b_array_red3 = np.delete(b_array_red3, todel)
#            be_array_red3 = np.delete(be_array_red3, todel)
#            e_array_red3 = np.delete(e_array_red3, todel)
#            ee_array_red3 = np.delete(ee_array_red3, todel)
#            b_array_blue3 = np.delete(b_array_blue3, todel)
#            be_array_blue3 = np.delete(be_array_blue3, todel)
#            e_array_blue3 = np.delete(e_array_blue3, todel)
#            ee_array_blue3 = np.delete(ee_array_blue3, todel)
#            try:
#                g_array_red3 = np.delete(g_array_red3, todel)
#                ge_array_red3 = np.delete(ge_array_red3, todel)
#                g_array_blue3 = np.delete(g_array_blue3, todel)
#                ge_array_blue3 = np.delete(ge_array_blue3, todel)
#            except:
#                pass
            il_data3 = np.delete(il_data3, todel)
            il_data_std3 = np.delete(il_data_std3, todel)
#            chiresult_red3 = np.delete(chiresult_red3, todel)
#            chiresult_blue3 = np.delete(chiresult_blue3, todel)
#    
        ##################################################### PAGE 5
       
        x_vec = il_data
        x_vec3 = il_data3
        x_vec2 = il_data2
        
        legendloc = 'upper right'
    
#        fig42= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
#        fig42.set_size_inches(1200./fig42.dpi,900./fig42.dpi)
#        plt.rc('text', usetex=True)
#        plt.rc('font', family='serif')
#        plt.rc('font', serif='Palatino')    
#        
#        ax3 = plt.subplot2grid((2,4), (0,0), colspan=2, rowspan=1)
        
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.xaxis.set_ticks_position('bottom')
        ax3.yaxis.set_ticks_position('left')
        
        #gaussian
        #ured_int_array = unumpy.uarray(red_int_array,red_std_array)
        #ured_int_array2 = unumpy.uarray(red_int_array2,red_std_array2)
        #ured_int_array3 = unumpy.uarray(red_int_array3,red_std_array3)
        
        #ublue_int_array = unumpy.uarray(blue_int_array,blue_std_array)
        #ublue_int_array2 = unumpy.uarray(blue_int_array2,blue_std_array2)
        #ublue_int_array3 = unumpy.uarray(blue_int_array3,blue_std_array3)
        
        #poisson
        #ured_int_array = unumpy.uarray(red_int_array,np.sqrt(red_int_array))
        #ured_int_array2 = unumpy.uarray(red_int_array2,np.sqrt(red_int_array2))
        ured_int_array3 = unumpy.uarray(red_int_array3,np.sqrt(red_int_array3))
        
        #ublue_int_array = unumpy.uarray(blue_int_array,np.sqrt(blue_int_array))
        #ublue_int_array2 = unumpy.uarray(blue_int_array2,np.sqrt(blue_int_array2))
        ublue_int_array3 = unumpy.uarray(blue_int_array3,np.sqrt(blue_int_array3))
        
        #yerr1 = unumpy.std_devs(((ublue_int_array[1:]-ured_int_array[1:])/(ured_int_array[1:]+ublue_int_array[1:])))    
        #yerr2 = unumpy.std_devs( ((ublue_int_array2-ured_int_array2)/(ured_int_array2+ublue_int_array2)))    
        yerr3 = unumpy.std_devs( +((ublue_int_array3)/(ured_int_array3)))    #########THERE WAS A MINUS!!!!!!!!
        
        #ax3.errorbar( 1.0/(x_vec[1:]+273.15)*1.0e3,  ((blue_int_array[1:]-red_int_array[1:])/(red_int_array[1:]+blue_int_array[1:]))/((blue_int_array[RTindex[0]]-red_int_array[RTindex[0]])/(red_int_array[RTindex[0]]+blue_int_array[RTindex[0]]))*Multiplication_factor[index,0], yerr=yerr1, marker='o',markersize=11,linestyle='',color='k')
        #ax3.errorbar( 1.0/(x_vec2+273.15)*1.0e3, ((blue_int_array2-red_int_array2)/(red_int_array2+blue_int_array2))/ ((blue_int_array2[RTindex[1]]-red_int_array2[RTindex[1]])/(red_int_array2[RTindex[1]]+blue_int_array2[RTindex[1]]))*Multiplication_factor[index,1],yerr = yerr2,  marker='o',markersize=5,ls='',color='k')
        ax3.errorbar( x_vec3, ((blue_int_array3)/(red_int_array3))/((blue_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]))*Multiplication_factor[index,2], yerr= yerr3, marker='o',markersize=12,ls='',color='k')
    
        #### JUST A TEST
        #yerrn = unumpy.std_devs( +((ublue_int_array3)/(ured_int_array3))) 
        #ax3.errorbar( x_vec3,1+((blue_int_array3)/(red_int_array3))/((blue_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]))*Multiplication_factor[index,2], yerr= yerrn, marker='o',markersize=12,ls='',color='r')
        #ax44 = plt.subplot2grid((2,2), (1,1), colspan=1, rowspan=1)
        #yerrg = unumpy.std_devs( +((ublue_int_array3)/(1))) 
        #ax44.errorbar( x_vec3,(blue_int_array3)/((blue_int_array3[RTindex[2]]))*Multiplication_factor[index,2], yerr= yerrg, marker='o',markersize=12,ls='',color='g')
        #yerrr = unumpy.std_devs( +((ured_int_array3)/(1))) 
        #ax44.errorbar( x_vec3,(red_int_array3)/((red_int_array3[RTindex[2]]))*Multiplication_factor[index,2], yerr= yerrr, marker='o',markersize=12,ls='',color='r')
    
    
#        x_vecA = np.linspace(1.0/(x_vec[-1]+273.15), 1.0/(x_vec[1]+273.15), 100)
#        ax3.plot(x_vecA*1.0e3,(1.6931052311882413*(1 + 2.285856254609182*np.exp(-14.99323181686587 + 4707.954620237778*x_vecA)))/(1 + 4.2858562546091825*np.exp(-14.99323181686587 + 4707.954620237778*x_vecA)),color='k',lw=2)
#        
#        x_vec2A = np.linspace(1.0/(x_vec2[-1]+273.15), 1.0/(x_vec2[0]+273.15), 100)
#        ax3.plot(x_vec2A*1.0e3, (16156.359998704738*(1 + 0.00012019992621814524*np.exp(-3.000619559474645 + 4707.954620237778*x_vec2A)))/(1 + 2.000120199926218*np.exp(-3.000619559474645 + 4707.954620237778*x_vec2A)), color='k', lw=2)
#        
        #x_vec3A = np.linspace(1.0/(x_vec3[0]+273.15), 1.0/(x_vec3[-1]+273.15), 100) #original, for 1/T axis
        x_vec3A = np.linspace(1.0/(x_vec3[0]+273.15), 1.0/(x_vec3[-1]+273.15), 100)
        x_vec3B = np.linspace(x_vec3[0], x_vec3[-1], 100)
        #ax3.plot(x_vec3B,(2.3115977519747477*(1 - 0.46140005619286806*np.exp(-16.15513563032235 + 4707.954620237778*x_vec3A)))/(1 + 1.538599943807132*np.exp(-16.15513563032235 + 4707.954620237778*x_vec3A))       , color = 'k', lw=2)
        
        #new with D= 0
        ax3.plot(x_vec3B,(2.150456683242829*(1 - np.exp(-16.665009616969353 + 4765.39700468485*x_vec3A)))/(1 + np.exp(-16.665009616969353 + 4765.39700468485*x_vec3A))      , color = 'k', lw=2)
        
        
        def red1(x):
           return (1.2826973413526264e6 + np.exp(4707.954620237778*x)*(0.9030156292015167 - 12.706204736174692*np.sqrt((1.3950212417059393e20 - 3.6486641690952644e14*np.exp(4707.954620237778*x) + 3.1792973266618586e8*np.exp(9415.909240475556*x) - 83.61736619972217*np.exp(14123.863860713333*x) + 6.734872222571795e-6*np.exp(18831.81848095111*x))/(757600.4832566812 + 1.*np.exp(4707.954620237778*x))**4)) - 9.626226848484278e6*np.sqrt((1.3950212417059393e20 - 3.6486641690952644e14*np.exp(4707.954620237778*x) + 3.1792973266618586e8*np.exp(9415.909240475556*x) - 83.61736619972217*np.exp(14123.863860713333*x) + 6.734872222571795e-6*np.exp(18831.81848095111*x))/(757600.4832566812 + 1.*np.exp(4707.954620237778*x))**4))/(757600.4832566811 + 1.*np.exp(4707.954620237778*x))
        def red2(x):
           return (1.2826973413526264e6 + np.exp(4707.954620237778*x)*(0.9030156292015167 + 12.706204736174692*np.sqrt((1.3950212417059393e20 - 3.6486641690952644e14*np.exp(4707.954620237778*x) + 3.1792973266618586e8*np.exp(9415.909240475556*x) - 83.61736619972217*np.exp(14123.863860713333*x) + 6.734872222571795e-6*np.exp(18831.81848095111*x))/(757600.4832566812 + 1.*np.exp(4707.954620237778*x))**4)) + 9.626226848484278e6*np.sqrt((1.3950212417059393e20 - 3.6486641690952644e14*np.exp(4707.954620237778*x) + 3.1792973266618586e8*np.exp(9415.909240475556*x) - 83.61736619972217*np.exp(14123.863860713333*x) + 6.734872222571795e-6*np.exp(18831.81848095111*x))/(757600.4832566812 + 1.*np.exp(4707.954620237778*x))**4))/(757600.4832566811 + 1.*np.exp(4707.954620237778*x))
        
        def blue1(x):
            return (162345.3832090541 + np.exp(4707.954620237778*x)*(0.9709382865438488 - 3.1824463052837073*np.sqrt((1.8493663583788663e21 - 6.895241433939969e15*np.exp(4707.954620237778*x) + 7.9116279776875e9*np.exp(9415.909240475556*x) - 2212.220777615905*np.exp(14123.863860713333*x) + 0.00018013898996649402*np.exp(18831.81848095111*x))/(10.048388574039537 + 1.*np.exp(4707.954620237778*x))**4)) - 31.978457091507146*np.sqrt((1.8493663583788663e21 - 6.895241433939969e15*np.exp(4707.954620237778*x) + 7.9116279776875e9*np.exp(9415.909240475556*x) - 2212.220777615905*np.exp(14123.863860713333*x) + 0.00018013898996649402*np.exp(18831.81848095111*x))/(10.048388574039537 + 1.*np.exp(4707.954620237778*x))**4))/(10.048388574039537 + 1.*np.exp(4707.954620237778*x))
        def blue2(x):
            return (162345.3832090541 + np.exp(4707.954620237778*x)*(0.9709382865438488 + 3.1824463052837073*np.sqrt((1.8493663583788663e21 - 6.895241433939969e15*np.exp(4707.954620237778*x) + 7.9116279776875e9*np.exp(9415.909240475556*x) - 2212.220777615905*np.exp(14123.863860713333*x) + 0.00018013898996649402*np.exp(18831.81848095111*x))/(10.048388574039537 + 1.*np.exp(4707.954620237778*x))**4)) + 31.978457091507146*np.sqrt((1.8493663583788663e21 - 6.895241433939969e15*np.exp(4707.954620237778*x) + 7.9116279776875e9*np.exp(9415.909240475556*x) - 2212.220777615905*np.exp(14123.863860713333*x) + 0.00018013898996649402*np.exp(18831.81848095111*x))/(10.048388574039537 + 1.*np.exp(4707.954620237778*x))**4))/(10.048388574039537 + 1.*np.exp(4707.954620237778*x))
    
#        def orange1(x):
#            return (1.5590958295791846e7 - 2.9019962151067033e7*np.sqrt((2.3085664491656083e26 - 3.2871925230136807e20*np.exp(4707.954620237778*x) + 1.7067853750483694e14*np.exp(9415.909240475556*x) - 3.5683921200536475e7*np.exp(14123.863860713333*x) + 2.5723452404684184*np.exp(18831.81848095111*x))/(6.744667528108135e6 + 1.*np.exp(4707.954620237778*x))**4) + np.exp(4707.954620237778*x)*(-0.6932090027361615 - 4.302652729749464*np.sqrt((2.3085664491656083e26 - 3.2871925230136807e20*np.exp(4707.954620237778*x) + 1.7067853750483694e14*np.exp(9415.909240475556*x) - 3.5683921200536475e7*np.exp(14123.863860713333*x) + 2.5723452404684184*np.exp(18831.81848095111*x))/(6.744667528108135e6 + 1.*np.exp(4707.954620237778*x))**4)))/(6.744667528108135e6 + 1.*np.exp(4707.954620237778*x))
#        def orange2(x):
#            return (1.5590958295791846e7 + 2.9019962151067033e7*np.sqrt((2.3085664491656083e26 - 3.2871925230136807e20*np.exp(4707.954620237778*x) + 1.7067853750483694e14*np.exp(9415.909240475556*x) - 3.5683921200536475e7*np.exp(14123.863860713333*x) + 2.5723452404684184*np.exp(18831.81848095111*x))/(6.744667528108135e6 + 1.*np.exp(4707.954620237778*x))**4) + np.exp(4707.954620237778*x)*(-0.6932090027361615 + 4.302652729749464*np.sqrt((2.3085664491656083e26 - 3.2871925230136807e20*np.exp(4707.954620237778*x) + 1.7067853750483694e14*np.exp(9415.909240475556*x) - 3.5683921200536475e7*np.exp(14123.863860713333*x) + 2.5723452404684184*np.exp(18831.81848095111*x))/(6.744667528108135e6 + 1.*np.exp(4707.954620237778*x))**4)))/(6.744667528108135e6 + 1.*np.exp(4707.954620237778*x))
#    
        # new with D = 0
        def orange1(x):
            return (-9.803270435612112e6*(-3.7903684907336745 + 2.1936115068609294e-7*np.exp(4765.39700468485*x) + 1.727912385068454e7*np.sqrt((7.280728078830508e6 - 1.*np.exp(4765.39700468485*x) - 2.4385483799448196e-8*np.exp(9530.7940093697*x))**2/(1.727912385068454e7 + 1.*np.exp(4765.39700468485*x))**4) + 1.*np.exp(4765.39700468485*x)*np.sqrt((7.280728078830508e6 - 1.*np.exp(4765.39700468485*x) - 2.4385483799448196e-8*np.exp(9530.7940093697*x))**2/(1.727912385068454e7 + 1.*np.exp(4765.39700468485*x))**4)))/(1.727912385068454e7 + np.exp(4765.39700468485*x))
        def orange2(x):
            return (9.803270435612112e6*(3.7903684907336745 - 2.1936115068609294e-7*np.exp(4765.39700468485*x) + 1.727912385068454e7*np.sqrt((7.280728078830508e6 - 1.*np.exp(4765.39700468485*x) - 2.4385483799448196e-8*np.exp(9530.7940093697*x))**2/(1.727912385068454e7 + 1.*np.exp(4765.39700468485*x))**4) + 1.*np.exp(4765.39700468485*x)*np.sqrt((7.280728078830508e6 - 1.*np.exp(4765.39700468485*x) - 2.4385483799448196e-8*np.exp(9530.7940093697*x))**2/(1.727912385068454e7 + 1.*np.exp(4765.39700468485*x))**4)))/(1.727912385068454e7 + np.exp(4765.39700468485*x))
    
    
    
#        ax3.fill_between(x_vecA*1.0e3,  
#                         red1(x_vecA),
#                         red2(x_vecA),  
#                         color = 'none',
#                         hatch="+",
#                         edgecolor='k',
#                         facecolor=[168/256,175/256,175/256],
#                         alpha=0.5,
#                         linewidth=0.0)
#                         
#        ax3.fill_between(x_vec2A*1.0e3,  
#                         blue1(x_vec2A),
#                         blue2(x_vec2A),  
#                         color = 'none',
#                         hatch=".",
#                         edgecolor='k',
#                         facecolor=[168/256,175/256,175/256],
#                         alpha=0.5,
#                         linewidth=0.0)
                         
        ax3.fill_between(x_vec3B,  
                         orange1(x_vec3A),
                         orange2(x_vec3A),  
                         color =[168/256,175/256,175/256],
                         #hatch="/",
                         edgecolor='k',
                         facecolor=[168/256,175/256,175/256],
                         alpha=0.5,
                         linewidth=0.0)
        
        
        # For large and small zoom, visibility is green to red
        #ax3.set_ylabel('Visibility green to red \n norm. to $\sim$30$^{\circ}$C (a.u.)',fontsize=fsizepl)
        # For  medium zoom, visibility is  red to green
        ax3.set_ylabel('Ratio of intensity \n norm. to $\sim$30 $^{\circ}$C (a.u.)',fontsize=fsizepl)
        ax3.tick_params(labelsize=fsizenb)
        ax3.yaxis.set_label_position("left")
        
        ax3.set_xlim([25,65])
        ax3.set_xticks([30,40,50,60])
        ax3.set_ylim([-0.7,2.2])   #######2.2
        ax3.set_yticks([0.5, 1,1.5,2.0])
    
        ax3.set_xlabel('Temperature at heater ($^{\circ}$C)',fontsize=fsizepl)
        ax3.axhline(y=1.0 , lw=2, color='k', ls='--')
        
    #    left, bottom, width, height = [0.4,0.7,0.2,0.2] #want 0.1 in width
    #    ax30 = fig42.add_axes([left, bottom, width, height])
        
    #    DeltaEvec = [res1.params['DeltaE'].value,res2.params['DeltaE'].value,res3.params['DeltaE'].value]
    #    DeltaEerrvec =[0.1,0.1,0.1] #[np.sqrt( res1.covar[0,0]),np.sqrt( res2.covar[0,0]),np.sqrt( res3.covar[0,0])]
    #    ax30.errorbar([1],DeltaEvec[0], yerr=DeltaEerrvec[0], marker='o', markersize=11, color='k' )
    #    ax30.errorbar([2],DeltaEvec[2], yerr=DeltaEerrvec[2], marker='o', markersize=8, color='k' )
    #    ax30.errorbar([3],DeltaEvec[1], yerr=DeltaEerrvec[1], marker='o', markersize=5, color='k' )
    #    ax30.set_xlim([0.8,3.2])
    #    #ax30.set_ylim([0.08,0.12])
    #    #ax30.set_yticks([0.08,0.1,0.12])
    #    ax30.set_xticks([])
    #    ax30.spines['left'].set_visible(False)
    #    ax30.spines['bottom'].set_visible(False)
    #    ax30.spines['top'].set_visible(False)
    #    ax30.xaxis.set_ticks_position('bottom')
    #    ax30.yaxis.set_ticks_position('right')
    #    ax30.tick_params(labelsize=fsizenb)
    #    ax30.set_ylabel('Fitted $\Delta$E (eV)',fontsize=fsizepl)
    #    ax30.yaxis.set_label_position("right")
        
        if fitlinearwitherror == True:
            pref4 = '_linearfitweighted'
        else:
            pref4 = '_linearfitnotweighted'
        
        prefix = 'varC_' + pref2 + pref3 + pref4
        #multipage_longer(prefix + '.pdf',dpi=80)
        
        from minepy import MINE
        from scipy.stats import pearsonr, spearmanr
  
        
        mine = MINE(alpha=0.6, c=15)
        mine.compute_score(x_vec3, ((blue_int_array3-red_int_array3)/(red_int_array3+blue_int_array3))/((blue_int_array3[RTindex[2]]-red_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]+blue_int_array3[RTindex[2]])))
        rp = pearsonr(x_vec3,      ((blue_int_array3-red_int_array3)/(red_int_array3+blue_int_array3))/((blue_int_array3[RTindex[2]]-red_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]+blue_int_array3[RTindex[2]])))
        rs = spearmanr(x_vec3,     ((blue_int_array3-red_int_array3)/(red_int_array3+blue_int_array3))/((blue_int_array3[RTindex[2]]-red_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]+blue_int_array3[RTindex[2]])))
        import scipy        
        dc0 = scipy.spatial.distance.correlation(x_vec3,     ((blue_int_array3-red_int_array3)/(red_int_array3+blue_int_array3))/((blue_int_array3[RTindex[2]]-red_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]+blue_int_array3[RTindex[2]])))
        import sklearn
        mi0 = sklearn.metrics.mutual_info_score(x_vec3,     ((blue_int_array3-red_int_array3)/(red_int_array3+blue_int_array3))/((blue_int_array3[RTindex[2]]-red_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]+blue_int_array3[RTindex[2]])))
        print('T')
        print(rp)
        print(rs)
        
        #######################################################################
        length_scalebar = 100.0 #in nm 
        scalebar_legend = '100 nm'
        length_scalebar_in_pixels = np.ceil(length_scalebar/(2.5))        
        import boe_bar as sb
            
        
        leto = ['N102pt5', 'N92pt9', 'N66pt4', 'N72pt8', 'N60pt6', 'N48pt5', 'N39pt8', 'N31pt2']
        #axc0 = plt.subplot2grid((nolines,noplots), (0,3), colspan=1, rowspan=1)
        se60 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[4] +'SEchannel.npz',mmap_mode='r') 
        xlen = se60['data'].shape[0]
        ylen = se60['data'].shape[1] 
        delx = 0#+28
        dely = 0#00
        xval = 144
        yval = 142
        arr_img60 = se60['data'][np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely]
    
       # axc1 = plt.subplot2grid((nolines,noplots), (0,4), colspan=1, rowspan=1)
        se50 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[5] +'SEchannel.npz',mmap_mode='r') 
        delx = 0#+28
        dely = 0#00
        xval = 133
        yval = 122
        arr_img50 = se50['data'][np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely]
    
#        axc2 = plt.subplot2grid((nolines,noplots), (0,6), colspan=1, rowspan=1)
        se40 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[6] +'SEchannel.npz',mmap_mode='r') 
        delx = 0#+28
        dely = 0#00
        xval = 135
        yval = 105
        arr_img40 = se40['data'][np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely]
#      axc3 = plt.subplot2grid((nolines,noplots), (0,10), colspan=1, rowspan=1)
        se30 = np.load('../2017-01-13_Andrea_NPs_CoolingDown_Controllably/' + leto[7] +'SEchannel.npz',mmap_mode='r')
        delx = 0#+28
        dely = 0#00
        xval = 144
        yval = 120
        arr_img30 = se30['data'][np.floor(xlen/2.0)-xval+delx:np.floor(xlen/2.0)+xval+delx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely]
   
        inset2 = fig1.add_axes([0.39, 0.6175, .105, .105]) #was 0.55
        inset2.imshow(arr_img60,cmap = cm.Greys_r)
        sbar = sb.AnchoredScaleBar(inset2.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4, my_fontsize = fsizenb)
        inset2.add_artist(sbar)    
        plt.setp(inset2, xticks=[], yticks=[])
        
        inset3 = fig1.add_axes([0.275, 0.6175, .105, .105]) #was 0.55
        inset3.imshow(arr_img50,cmap = cm.Greys_r)
        sbar = sb.AnchoredScaleBar(inset3.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4, my_fontsize = fsizenb)
        inset3.add_artist(sbar)    
        plt.setp(inset3, xticks=[], yticks=[])
        
        inset4 = fig1.add_axes([0.185, 0.6175, .105, .105]) #was 0.55
        inset4.imshow(arr_img40,cmap = cm.Greys_r)
        sbar = sb.AnchoredScaleBar(inset4.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4, my_fontsize = fsizenb)
        inset4.add_artist(sbar)    
        plt.setp(inset4, xticks=[], yticks=[])
        
        inset5 = fig1.add_axes([0.1, 0.6175, .105, .105]) #was 0.55
        inset5.imshow(arr_img30,cmap = cm.Greys_r)
        sbar = sb.AnchoredScaleBar(inset5.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4, my_fontsize = fsizenb)
        inset5.add_artist(sbar)    
        plt.setp(inset5, xticks=[], yticks=[])
        
        

      
def do_visib_other_qttties(host):
#if True:
    fsizepl = 24
    fsizenb = 20
   
    initbin = (150+50+3)-1
    backgdinit = 50
    ### PIXEL
    Pixel_size = [2.23, 1.79, 1.49, 1.28, 1.12]
#    let = ['pix1', 'pix2', 'pix3', 'pix4', 'pix5'] #pixel size is decreasing
#    loadprefix = '../2017-01-23_AndreaNP_as_a_fct_of_pixel_size/'
#    nombre = 5
    
    ### APERTURE
    Current = [28, 379, 1800, 5700] 
#    let = ['ap10','ap30', 'ap60', 'ap120']   
#    loadprefix = '../2017-01-27_Andrea_NPs_Different_apertures/'
#    nombre = 4
    
    kv = [10,15,20] 
#    let = ['kv10', 'kv15', 'kv20'] 
#    loadprefix = '../2017-01-25_Andrea_NPs_Different_kV/'
#    nombre = 3
#    
#    yerr1 = np.empty(nombre)
#    red_int_array = np.empty(nombre)
#    blue_int_array = np.empty(nombre)
#    for index in np.arange(0,nombre):
#        
#        print(index)
#    
#        #for pixel, aperture
#        #redd = np.load(loadprefix + let[index] + 'Redbright.npz',mmap_mode='r') 
#        #blued = np.load(loadprefix + let[index] + 'Bluebright.npz',mmap_mode='r') 
#        #red_int_array[index] = np.average(redd['data'][:,backgdinit:initbin,:,:],axis=(0,1,2,3))
#        #blue_int_array[index] = np.average(blued['data'][:,backgdinit:initbin,:,:],axis=(0,1,2,3))
#    
#        #for kv
#        redd = np.load(loadprefix + let[index] + 'RED1D.npz',mmap_mode='r') 
#        blued = np.load(loadprefix + let[index] + 'BLUE1D.npz',mmap_mode='r') 
#        red_int_array[index] = np.average(redd['data'][backgdinit:initbin])
#        blue_int_array[index] = np.average(blued['data'][backgdinit:initbin])
#        
#        ured_int_array = unumpy.uarray(red_int_array[index],np.sqrt(red_int_array[index]))
#        ublue_int_array = unumpy.uarray(blue_int_array[index],np.sqrt(blue_int_array[index]))
#          
#        #print(blue_int_array[index]) 
#        #print(red_int_array[index])
#        print(((blue_int_array[index]-red_int_array[index])/(red_int_array[index]+blue_int_array[index])))
#        yerr1[index] = unumpy.std_devs(((ublue_int_array-ured_int_array)/(ured_int_array+ublue_int_array)))
#        print(yerr1[index])
#        
#        del ured_int_array, ublue_int_array, redd, blued
#        gc.collect()
#    
#    
#    aklklk
    #fig, ax4 = plt.subplots()
    
    toplotpixel = [-0.331871788598/(-0.272621152559),-0.290592140046/(-0.272621152559),-0.272621152559/(-0.272621152559),-0.245528170747/(-0.272621152559),-0.251845555422/(-0.272621152559)]
    toploterrpixel = [0.234703670024,0.256254568909,0.279883426484,0.315887812131,0.359311715502]
    
    toplotcurrent = [-0.381793734929/(-0.404870033234),(-0.404870033234)/(-0.404870033234),-0.429264480422/(-0.404870033234),-0.435263215242/(-0.404870033234)]
    toploterrcurrent = [0.442812930413,0.230770711289,0.121885246395,0.0702029310963]
    
    toplotkv = [-0.443318843718/(-0.443318843718),-0.539968820839/(-0.443318843718),-0.551046216565/(-0.443318843718)]
    toploterrkv = [0.211791238488,0.190393627733,0.158564880888]
    
    from mpl_toolkits.axes_grid1 import host_subplot
    import mpl_toolkits.axisartist as AA
    import matplotlib.pyplot as plt

    #host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(bottom=0.0) #original 0.2
    par2 = host.twiny()
    par3 = host.twiny()
    offset = -60 #original -40
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis["bottom"] = new_fixed_axis(loc="bottom",
                                         axes=par2,
                                        offset=(0, 2*offset))
    par2.axis["top"].toggle(all=False)
    par3.axis["bottom"] = new_fixed_axis(loc="bottom",
                                        axes=par3,
                                        offset=(0, 1*offset))
    par3.axis["top"].toggle(all=False)
    host.set_xlim([0.98,4])
    host.set_ylim([0.4,2.2])
    host.set_ylabel('Ratio of intensity \n norm. to standard (a.u.)',fontsize=fsizepl)
   # host.set_xlabel('Pixel size (nm) ' +  r'$\blacktriangle$',fontsize=fsizepl)
    host.set_xlabel('Pixel size (nm)',fontsize=fsizepl)
    
    par2.set_xlim([-12000, 6500])
    #par2.set_xlabel('Electron beam current (pA) ' + r'$\blacksquare$',fontsize=fsizepl)
    par2.set_xlabel('Electron beam current (pA)',fontsize=fsizepl)
    par3.set_xlim([-20, 45])
    #par3.set_xlabel('Electron beam energy (kV) ' + r'$\bigstar$',fontsize=fsizepl)
    par3.set_xlabel('Electron beam energy (kV)',fontsize=fsizepl)
    
    p1 = host.errorbar( Pixel_size, toplotpixel, yerr=toploterrpixel, marker='s',markersize=12,linestyle='',color='b', label='Varying pixel size',markeredgecolor='None')
    p3 = par2.errorbar( Current, toplotcurrent, yerr=toploterrcurrent, marker='^',markersize=12,linestyle='',color='c',label='Varying current',markeredgecolor='None')
    p4 = par3.errorbar( kv, toplotkv, yerr=toploterrkv, marker='*',markersize=12,linestyle='',color='m',label='Varying current',markeredgecolor='None')
   
    host.set_xticks([1,2])
    #host.set_xticklabels(['2.2', '1.8', '1.5', '1.2', '1.1'])
    par2.set_xticks([30,5500])
    par3.set_xticks([10,20])
    
    host.axis["bottom"].label.set_color('b')
    par2.axis["bottom"].label.set_color('c')
    par3.axis["bottom"].label.set_color('m')
    
    host.tick_params(axis="x", colors="b")
    par2.tick_params(axis="x", colors="c")
    par3.tick_params(axis='x', colors='m')
    
    host.spines["bottom"].set_edgecolor('b')
    par2.spines["bottom"].set_edgecolor('c')
    par3.spines["bottom"].set_edgecolor('m')
    
#    host.spines["bottom"].set_color('b')
#    par2.spines["bottom"].set_color('c')
#    par3.spines["bottom"].set_color('m')
    
    host.axis["bottom"].label.set_size(fsizepl)
    par2.axis["bottom"].label.set_size(fsizepl)
    par3.axis["bottom"].label.set_size(fsizepl)
    host.axis["right"].label.set_size(fsizepl)
    
    host.axis["bottom"].major_ticklabels.set_size(fsizenb)
    par2.axis["bottom"].major_ticklabels.set_size(fsizenb)
    par3.axis["bottom"].major_ticklabels.set_size(fsizenb)
    host.axis["right"].major_ticklabels.set_size(fsizenb)
    
    host.axis["bottom"].major_ticklabels.set_color('b')
    par2.axis["bottom"].major_ticklabels.set_color('c')
    par3.axis["bottom"].major_ticklabels.set_color('m')
    
    host.yaxis.set_label_position("right")
    host.xaxis.set_ticks_position('bottom')
    host.yaxis.set_ticks_position('right')
    
    host.axis["top"].toggle(all=False)
    host.axis["left"].toggle(all=False)
    host.axis["right"].toggle(True)
    host.axhline(y=1.0 , lw=2, color='k', ls='--')
    
    ##### STATS
    from minepy import MINE
    from scipy.stats import pearsonr, spearmanr
    import scipy
    import sklearn
  
        
    mine = MINE(alpha=0.6, c=15)
    mine2 = MINE(alpha=0.6, c=15)
    mine3 = MINE(alpha=0.6, c=15)
    
    #PIXEL   
    mine.compute_score(Pixel_size, toplotpixel)
    rp = pearsonr(Pixel_size, toplotpixel)
    rs = spearmanr(Pixel_size, toplotpixel)
    dc = scipy.spatial.distance.correlation(Pixel_size, toplotpixel)
    mi = sklearn.metrics.mutual_info_score(Pixel_size, toplotpixel)
    print('pixel')
    print(rp)
    print(rs)
    
    # KV
    mine2.compute_score(kv, toplotkv)
    rp2 = pearsonr(kv, toplotkv)
    rs2 = spearmanr(kv, toplotkv)
    dc2 = scipy.spatial.distance.correlation(kv, toplotkv)
    mi2 = sklearn.metrics.mutual_info_score(kv, toplotkv)
    print('kv')
    print(rp2)
    print(rs2)
    
    ### Curremt
    mine3.compute_score(Current, toplotcurrent)
    rp3 = pearsonr(Current, toplotcurrent)
    rs3 = spearmanr(Current, toplotcurrent)
    dc3 = scipy.spatial.distance.correlation(Current, toplotcurrent)
    mi3 = sklearn.metrics.mutual_info_score(Current, toplotcurrent)
    print('current')
    print(rp3)
    print(rs3)

    

#from mpl_toolkits.axes_grid1 import host_subplot
#import mpl_toolkits.axisartist as AA
#ax99 = host_subplot(222, axes_class=AA.Axes)
#do_visib_other_qttties(ax99)
#    
#from ace.samples import wang04
#x, y = wang04.build_sample_ace_problem_wang04(N=200)
#print(x)
#print(y)
#from ace import model
#myace = model.Model()
#myace.build_model_from_xy(x,y)
#myace.eval([0.1, 0.2, 0.5, 0.3, 0.5])
    
#klkklkllkllk