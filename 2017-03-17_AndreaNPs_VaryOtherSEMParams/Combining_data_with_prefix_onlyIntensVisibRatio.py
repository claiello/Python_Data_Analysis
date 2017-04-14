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
from my_fits import *
##############################################################################
##############################################################################

def do_visib_other_qttties(host):
#if True:
    fsizepl = 24
    fsizenb = 20
   
    initbin = (150+50+3)-1
    backgdinit = 50
    ### PIXEL
    Pixel_size = [2.48,3.72,1.86,2.98,1.86,2.98,2.13]
#    let = ['p300','p200','p400','p250','p400b','p250b','p350'] #no of pixels
#    loadprefix = '../2017-03-17_AndreaNPs_VaryOtherSEMParams/'
#    nombre = 7
#    undex = 0
    
    ### APERTURE
    Current = [379, 1800, 28,5700] #pA
#    let = ['ap30','ap60','ap10','ap120'] #aperture
#    loadprefix ='../2017-03-17_AndreaNPs_VaryOtherSEMParams/'
#    nombre = 4
#    undex = 0
    
    kv = [10,15,5,20]
#    let = ['kv10','kv15','kv5','kv20'] #aperture
#    loadprefix ='../2017-03-17_AndreaNPs_VaryOtherSEMParams/'
#    nombre = 4
#    undex = 0
#    
#    yerrV = np.empty(nombre)
#    yerrR = np.empty(nombre)
#    red_int_array = np.empty(nombre)
#    blue_int_array = np.empty(nombre)
#    for index in np.arange(0,nombre):
#        
#        print(index)
#    
#        redd = np.load(loadprefix + let[index] + 'Redbright.npz',mmap_mode='r') 
#        red_int_array[index] = np.average(redd['data'][:,backgdinit:initbin,:,:],axis=(0,1,2,3))
#        No = redd['data'].shape[0]*redd['data'][:,backgdinit:initbin,:,:].shape[1]*redd['data'].shape[2]*redd['data'].shape[3]
#        del redd
#        blued = np.load(loadprefix + let[index] + 'Bluebright.npz',mmap_mode='r') 
#        blue_int_array[index] = np.average(blued['data'][:,backgdinit:initbin,:,:],axis=(0,1,2,3))
#        del blued
#        gc.collect()
#        
#        reddN = np.load(loadprefix + let[undex] + 'Redbright.npz',mmap_mode='r') 
#        red_int_arrayN = np.average(reddN['data'][:,backgdinit:initbin,:,:],axis=(0,1,2,3))
#        NoN = reddN['data'].shape[0]*reddN['data'][:,backgdinit:initbin,:,:].shape[1]*reddN['data'].shape[2]*reddN['data'].shape[3]
#        del reddN
#        bluedN = np.load(loadprefix + let[undex] + 'Bluebright.npz',mmap_mode='r') 
#        blue_int_arrayN = np.average(bluedN['data'][:,backgdinit:initbin,:,:],axis=(0,1,2,3))
#        del bluedN
#        gc.collect()
#    
#    
##        print(redd['data'].shape[1])
##        print(redd['data'][:,backgdinit:initbin,:,:].shape[1])
##        indeed prints different things
#        
#        ured_int_array = unumpy.uarray(red_int_array[index],np.sqrt(red_int_array[index])/np.sqrt(No))
#        ublue_int_array = unumpy.uarray(blue_int_array[index],np.sqrt(blue_int_array[index])/np.sqrt(No))
#        
#        ured_int_arrayN = unumpy.uarray(red_int_arrayN,np.sqrt(red_int_arrayN)/np.sqrt(NoN))
#        ublue_int_arrayN = unumpy.uarray(blue_int_arrayN,np.sqrt(blue_int_arrayN)/np.sqrt(NoN))
#          
#        #print(blue_int_array[index]) 
#        #print(red_int_array[index])
#        #Visibility
#        print('visib')
#        print(((blue_int_array[index]-red_int_array[index])/(red_int_array[index]+blue_int_array[index]))/((blue_int_arrayN-red_int_arrayN)/(red_int_arrayN+blue_int_arrayN)))
#        #Ratio
#        print('ratio')
#        print(((red_int_array[index])/(blue_int_array[index]))/((red_int_arrayN)/(blue_int_arrayN)))
#        #Rrror on visibility
#        print('ERR visib')
#        yerrV[index] = unumpy.std_devs(((ublue_int_array-ured_int_array)/(ured_int_array+ublue_int_array))/((ublue_int_arrayN-ured_int_arrayN)/(ured_int_arrayN+ublue_int_arrayN)))
#        print(yerrV[index])
#        print('ERR ratio')
#        yerrR[index] = unumpy.std_devs(((ured_int_array)/(ublue_int_array))/((ured_int_arrayN)/(ublue_int_arrayN)))
#        print(yerrR[index])
#        
#        
#        del red_int_arrayN   ,blue_int_arrayN, ured_int_array, ublue_int_array  ,ured_int_arrayN  ,ublue_int_arrayN   
#        gc.collect()
#    
#    
#    klklkk
#    
    ####VISIB
    toplotpixel = [1.0,1.01514637055,1.01172723577,1.01363267797,1.00636743468,1.00841453996,1.00187643191]
    toplotcurrent = [1.0,0.887955652258,0.891502853426,1.23237112522]
    toplotkv = [1.0,1.12043553436,0.854931184789,1.11161402453]

    ####RATIO
    toplotpixelratio = [1.0,1.04359332782,1.03347218482,1.03909173294,1.01794046721,1.0238252341,1.00523058474]
    toplotcurrentratio = [1.0,0.819496980284,0.824408154554,1.67355621548]
    toplotkvratio = [1.0,1.45334961564,0.704557498843,1.4086256255]
    
    ####ERRORS
    toploterrpixelV = [9.33958108437e-05,0.000120090071471,8.32887436109e-05,0.00010493886892,8.3551713833e-05,0.000104789676529,8.78389691661e-05]
    toploterrcurrentV = [0.000103219172824,8.10163464006e-05,0.000264981973824,9.24001279041e-05]
    toploterrkvV = [9.29672328019e-05,8.67007606294e-05,0.000109313688219,8.41497670496e-05] 
    
    toploterrpixelR = [0.00025918707771,0.000353597161336,0.000239131604876,0.000305944613321,0.000236186237418,0.000299727423835,0.000245171241429]
    toploterrcurrentR = [0.000194375355553,0.000131056186376,0.000374812591108,0.000248624378047]
    toploterrkvR = [0.000252707313971,0.000352547534033,0.00019057094491,0.000325104616011] 

    ###################

#    ##### STATS
#    from minepy import MINE
#    from scipy.stats import pearsonr, spearmanr
#    import scipy
#    import sklearn
#  
#        
#    mine = MINE(alpha=0.6, c=15)
#    mine2 = MINE(alpha=0.6, c=15)
#    mine3 = MINE(alpha=0.6, c=15)
#    
#    mine0 = MINE(alpha=0.6, c=15)
#    mine20 = MINE(alpha=0.6, c=15)
#    mine30 = MINE(alpha=0.6, c=15)
#    
#    #PIXEL   
#    mine.compute_score(Pixel_size, toplotpixel)
#    rp = pearsonr(Pixel_size, toplotpixel)
#    rs = spearmanr(Pixel_size, toplotpixel)
#    dc = scipy.spatial.distance.correlation(Pixel_size, toplotpixel)
#    mi = sklearn.metrics.mutual_info_score(Pixel_size, toplotpixel)
#    print('pixel visib')
#    print(mi)
#    print(rs)
#    print(rp)
#    print(dc)
#    print(mine.mic())
#    
#    mine0.compute_score(Pixel_size, toplotpixelratio)
#    rp0 = pearsonr(Pixel_size, toplotpixelratio)
#    rs0 = spearmanr(Pixel_size, toplotpixelratio)
#    dc0 = scipy.spatial.distance.correlation(Pixel_size, toplotpixelratio)
#    mi0 = sklearn.metrics.mutual_info_score(Pixel_size, toplotpixelratio)
#    print('pixel ratio')
#    print(mi0)
#    print(rs0)
#    print(rp0)
#    print(dc0)
#    print(mine0.mic())
#    
#    # KV
#    mine2.compute_score(kv, toplotkv)
#    rp2 = pearsonr(kv, toplotkv)
#    rs2 = spearmanr(kv, toplotkv)
#    dc2 = scipy.spatial.distance.correlation(kv, toplotkv)
#    mi2 = sklearn.metrics.mutual_info_score(kv, toplotkv)
#    print('kv visib')
#    print(mi2)
#    print(rs2)
#    print(rp2)
#    print(dc2)
#    print(mine2.mic())
#    mine20.compute_score(kv, toplotkvratio)
#    rp20 = pearsonr(kv, toplotkvratio)
#    rs20 = spearmanr(kv, toplotkvratio)
#    dc20 = scipy.spatial.distance.correlation(kv, toplotkvratio)
#    mi20 = sklearn.metrics.mutual_info_score(kv, toplotkvratio)
#    print('kv ratio')
#    print(mi20)
#    print(rs20)
#    print(rp20)
#    print(dc20)
#    print(mine20.mic())
#    
#    ### Curremt
#    mine3.compute_score(Current, toplotcurrent)
#    rp3 = pearsonr(Current, toplotcurrent)
#    rs3 = spearmanr(Current, toplotcurrent)
#    dc3 = scipy.spatial.distance.correlation(Current, toplotcurrent)
#    mi3 = sklearn.metrics.mutual_info_score(Current, toplotcurrent)
#    print('current visib')
#    print(mi3)
#    print(rs3)
#    print(rp3)
#    print(dc3)
#    print(mine3.mic())
#    mine30.compute_score(Current, toplotcurrentratio)
#    rp30 = pearsonr(Current, toplotcurrentratio)
#    rs30 = spearmanr(Current, toplotcurrentratio)
#    dc30 = scipy.spatial.distance.correlation(Current, toplotcurrentratio)
#    mi30 = sklearn.metrics.mutual_info_score(Current, toplotcurrentratio)
#    print('current ratio')
#    print(mi30)
#    print(rs30)
#    print(rp30)
#    print(dc30)
#    print(mine30.mic())
#    
#    klklklk
    
    ################################

    from mpl_toolkits.axes_grid1 import host_subplot
    import mpl_toolkits.axisartist as AA
    import matplotlib.pyplot as plt
    
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
    host.set_xlim([1.45,9])
    
    host.set_ylim([0.2,2.0])   #######2.2
    host.set_yticks([1,1.5,2.0])    
    
    
    host.set_ylabel('Intensity thermometry signal, \n norm. to standard (a.u.)',fontsize=fsizepl)
    host.set_xlabel('Pixel size (nm)',fontsize=fsizepl)
    
    par2.set_xlim([-12000, 6500])
    par2.set_xlabel('Electron beam current (pA)',fontsize=fsizepl)
    par3.set_xlim([-20, 45])
    par3.set_xlabel('Electron beam energy (kV)',fontsize=fsizepl)
    
    p1 = host.errorbar( Pixel_size, toplotpixel, yerr=toploterrpixelV, marker='o',markersize=12,linestyle='',color='b', label='Varying pixel size',markeredgecolor='None')
    p3 = par2.errorbar( Current, toplotcurrent, yerr=toploterrcurrentV, marker='o',markersize=12,linestyle='',color='c',label='Varying current',markeredgecolor='None')
    p4 = par3.errorbar( kv, toplotkv, yerr=toploterrkvV, marker='o',markersize=12,linestyle='',color='m',label='Varying current',markeredgecolor='None')
   
    p1 = host.errorbar( Pixel_size, toplotpixelratio, yerr=toploterrpixelR, marker='d',markersize=12,linestyle='',color='b', label='Varying pixel size',markeredgecolor='None')
    p3 = par2.errorbar( Current, toplotcurrentratio, yerr=toploterrcurrentR, marker='d',markersize=12,linestyle='',color='c',label='Varying current',markeredgecolor='None')
    p4 = par3.errorbar( kv, toplotkvratio, yerr=toploterrkvR, marker='d',markersize=12,linestyle='',color='m',label='Varying current',markeredgecolor='None')
      
    host.set_xticks([1.5,4])
    par2.set_xticks([30,5500])
    par3.set_xticks([5,20])
    
    host.axis["bottom"].label.set_color('b')
    par2.axis["bottom"].label.set_color('c')
    par3.axis["bottom"].label.set_color('m')
    
    host.tick_params(axis="x", colors="b")
    par2.tick_params(axis="x", colors="c")
    par3.tick_params(axis='x', colors='m')
    
    host.spines["bottom"].set_edgecolor('b')
    par2.spines["bottom"].set_edgecolor('c')
    par3.spines["bottom"].set_edgecolor('m')
    
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
    
#llllllll