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
#from BackgroundCorrection import *
#from TConversionThermocoupler import *
import matplotlib.cm as cm
import scipy.ndimage as ndimage
#from matplotlib_scalebar.scalebar import ScaleBar #### Has issue with plotting using latex font. only import when needed, then unimport
#from mpl_toolkits.axes_grid1 import make_axes_locatable
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
#from tifffile import *
#from sklearn.mixture import GMM 
import matplotlib.cm as cm
#from FluoDecay import *
#from PlottingFcts import *
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.misc
#import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile

import skimage
from skimage import exposure
from my_fits import *

import pickle
import my_fits
from uncertainties import unumpy
from numpy import genfromtxt
from CreateDatasets import *
#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################
#######
def moving_average(a,n=3):
    vec = np.cumsum(a)
    vec[n:] = vec[n:] - vec[:-n]
    return (1/n)*vec[n-1:]

let = ['RT','N30','N40','N50','N60', 'N70']
letd = ['N70D','N60D','N50D', 'N40D','N30D','RTD']

listofindex =np.arange(0,6)

inverse = [5,4,3,2,1,0]

ax0 = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=1)
ax1 = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((2,3), (0,2), colspan=1, rowspan=1)
ax3= plt.subplot2grid((2,3), (1,0), colspan=1, rowspan=1)
ax4 = plt.subplot2grid((2,3), (1,1), colspan=1, rowspan=1)
ax5 = plt.subplot2grid((2,3), (1,2), colspan=1, rowspan=1)

axvec = [ax0,ax1,ax2,ax3,ax4,ax5]

tit = ['RT','30C','40C','50C','60C','70C']

result_up = []

result_down = []

temp_up = [ [25.0, 24.8], [30.1, 30.7], [39.5, 40.0], [50.1, 52.0], [59.7, 60.4], [69.7, 71.1] ]

temp_down = [  [22.9, 22.9], [30.5, 30.4],  [40.0, 39.8], [50.3, 49.2],  [60.5, 57.1], [69.8, 71.2] ]

x = 89000 # = (450000 - 5000)/5
ind_hlp = np.array([0.5 * x, 1.5 * x, 2.5 * x, 3.5 * x, 4.5 * x], dtype = np.int) # helper indices for the moved_averaged array
in1 = np.array([0,x,2*x,3*x,4*x],dtype=np.int)
in2 = np.array([x,2*x,3*x,4*x,5*x],dtype=np.int)
    
for index in listofindex:
    
    print(index)
    
    if index != 0:
        print('index is')
        print(index)
        il = np.load(str(let[index]) +'ILchannel.npz') 
        axvec[index].plot(il['data'].flatten(), color='r')
    
    ild = np.load(str(letd[inverse[index]]) +'ILchannel.npz') 
    axvec[index].plot(ild['data'].flatten(), color='b')
    
    #[90000-45000,180000-45000,270000-45000,360000-45000,450000-45000]
    axvec[index].set_title(tit[index])
    if index != 0:
        axvec[index].plot([90000-45000,180000-45000,270000-45000,360000-45000,450000-45000],np.average(il['data'],axis=(1,2)), color='r',markersize=12, marker='o')
    
    axvec[index].plot([90000-45000,180000-45000,270000-45000,360000-45000,450000-45000],np.average(ild['data'],axis=(1,2)), color='b',markersize=12, marker='o')
    
#    axvec[index].set_ylim([-0.145,0.055])    

    t_down = moving_average(ild['data'].flatten(), n = 5000)
    
    if index != 0:
        t_up = moving_average(il['data'].flatten(), n = 5000)
        axvec[index].plot(t_up, color='k')

        axvec[index].plot(ind_hlp, t_up[ind_hlp], 'x', color='k')
        result_up.append([temp_up[index][0], temp_up[index][1], t_up[-1], t_up[0]])  
        
        TUP = np.array([np.average(t_up[0:x]),np.average(t_up[x:2*x]),np.average(t_up[2*x:3*x]),np.average(t_up[3*x:4*x]),np.average(t_up[4*x:5*x])])
        heating_volt_to_temp = temp_up[index][1] + (temp_up[index][0] - temp_up[index][1])/(t_up[0] - t_up[-1])  *  (TUP - t_up[-1])
        #heating_volt_to_temp = temp_up[index][1] + (temp_up[index][0] - temp_up[index][1])/(t_up[0] - t_up[-1])  *  (t_up[ind_hlp] - t_up[-1])
        
        print(heating_volt_to_temp)
    
    axvec[index].plot(t_down, color='k')
    result_down.append([temp_down[index][0], temp_down[index][1], t_down[0], t_down[-1]])
    
    #print('here')
    #print(t_down[ind_hlp])
    TDOWN = np.array([np.average(t_down[0:x]),np.average(t_down[x:2*x]),np.average(t_down[2*x:3*x]),np.average(t_down[3*x:4*x]),np.average(t_down[4*x:5*x])])
    cooling_volt_to_temp = temp_down[index][1] + (temp_down[index][0] - temp_down[index][1])/(t_down[0] - t_down[-1])  *  (TDOWN - t_down[-1])
    #cooling_volt_to_temp = temp_down[index][1] + (temp_down[index][0] - temp_down[index][1])/(t_down[0] - t_down[-1])  *  (t_down[ind_hlp] - t_down[-1])    
    print(cooling_volt_to_temp)
    
    

result_up = np.array(result_up)

result_down = np.array(result_down)



plt.show()   
lklklk

UP
DOWN (30 and 40C are bad)

0
[ 22.89999962  22.89999962  22.89999962  22.89999962  22.89999962]
1
index is
1
[ 30.24231529  30.43283653  30.54502869  30.61672211  30.67356873]
[ 30.67066002  34.57613373  35.5042572   34.56672668  32.5168457 ]
2
index is
2
[ 39.62433624  39.77347946  39.86793518  39.93108368  39.97883606]
[ 39.42526627  38.67329025  38.5286026   39.00093842  39.45114136]
3
index is
3
[ 50.28258896  50.85370255  51.2881012   51.61391068  51.85993195]
[ 47.03316879  41.10401535  40.21371078  42.79360199  46.74985123]
4
index is
4
[ 59.79606628  59.98893356  60.12455368  60.27425385  60.38433838]
[ 60.27148056  59.86600113  59.19314957  58.39229965  57.50851822]
5
index is
5
[ 69.90750122  70.30992889  70.62423706  70.83724213  71.00095367]
[ 70.01960754  70.42518616  70.72190857  70.93533325  71.11050415]
