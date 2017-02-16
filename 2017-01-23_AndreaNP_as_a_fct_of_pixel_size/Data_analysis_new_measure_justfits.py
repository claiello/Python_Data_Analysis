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

from my_fits import *

#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################


initbin = (150+50+3)-1

dset= 1

###############################################################################
###############################################################################
###############################################################################

if dset == 1:
    nametr = ['2017-01-23-1547_ImageSequence__250.000kX_10.000kV_30mu_7',
          '2017-01-23-1608_ImageSequence__250.000kX_10.000kV_30mu_8',
          '2017-01-23-1633_ImageSequence__250.000kX_10.000kV_30mu_9',
          '2017-01-23-1736_ImageSequence__250.000kX_10.000kV_30mu_10',
          '2017-01-23-1818_ImageSequence__63.372kX_10.000kV_30mu_11']

    let = ['pix1', 'pix2', 'pix3', 'pix4', 'pix5'] #pixel size is decreasing
    
    Varying_variable = [2.23, 1.79, 1.49, 1.28, 1.12]
    Label_varying_variable = 'Pixel size (nm) [data taken LARGE to SMALL]' 
    
    listofindex = [0,1,2,3,4]
    
    loadprefix = ''
    
###############################################################################
###############################################################################
###############################################################################

if dset == 1:
    pref0 = 'Different_pixel_sizes_'
    
Time_bin = 1000.0 #in ns

#fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
#fig3, ax3 = plt.subplots()
#fig4, ax4 = plt.subplots()
#fig5, ax5 = plt.subplots()

#fig10, ax10 = plt.subplots()
#fig20, ax20 = plt.subplots()
#fig200, ax200 = plt.subplots()
#fig2000, ax2000 = plt.subplots()
#fig30, ax30 = plt.subplots()

#fig40, ax40 = plt.subplots()
#fig400, ax400 = plt.subplots()

red_vec = np.empty([len(Varying_variable),1498])
blue_vec = np.empty([len(Varying_variable),1498])


def plot_things(prefix, my_color):

    for index in listofindex: 
        
        print(index)
        
        print('bef loading')
        redd = np.load(loadprefix + let[index] + prefix + '1D.npz',mmap_mode='r') 
        red = redd['data'][initbin:]
        if my_color =='r':
            print('red')
            red_vec[index,:] = red
        else:
            print('blue')
            blue_vec[index,:] = red
        
        if index == 0:
            median_tau_red = np.empty(len(listofindex))
            one_over_e_tau_red = np.empty(len(listofindex))
            mean_int_median_before = np.empty(len(listofindex))
            mean_int_median_after = np.empty(len(listofindex))
            mean_int_one_over_e_before = np.empty(len(listofindex))
            mean_int_one_over_e_after = np.empty(len(listofindex))
            aux_vec = np.empty([len(listofindex),len(red)])
            aux_vec_cumu = np.empty([len(listofindex),len(red)])
        
        #median tau
        comparison = 10000000
        indexrecorded = 10000000
        for jj in np.arange(0,len(red)):
            first_part = np.sum(red[0:jj])
            second_part = np.sum(red[jj:])
            difference = np.abs(first_part - second_part) 
            if difference < comparison:
                comparison = difference
                indexrecorded = jj
        median_tau_red[index] = indexrecorded
        mean_int_median_before[index] = np.average(red[0:indexrecorded]) 
        mean_int_median_after[index] = np.average(red[indexrecorded:]) 
      
        #1/e tau
        comparisone = 10000000
        indexrecordede = 10000000
        for jj in np.arange(0,len(red)):
            first_part = red[0]/np.exp(1)
            second_part = red[jj]
            difference = np.abs(first_part - second_part) 
            if difference < comparisone:
                comparisone = difference
                indexrecordede = jj
        one_over_e_tau_red[index] = indexrecordede
        mean_int_one_over_e_before[index] = np.average(red[0:indexrecordede]) 
        mean_int_one_over_e_after[index] = np.average(red[indexrecordede:]) 
        
        print(index)
        print( aux_vec[index,:].shape)
        print(len(red)) #SHOULD BE APPROX 1500
        for jj in np.arange(0,len(red)):
            aux_vec[index,jj] = np.sum(red[0:jj])/np.sum(red[jj:])
            aux_vec_cumu[index,jj]= np.sum(red[0:jj])
        
        
        ax2.plot(np.arange(0,len(red)), aux_vec_cumu[index,:],lw=len(listofindex)-index,color=my_color )
        
        print('index=' + str(index))
        print(np.sum(np.isnan(aux_vec_cumu[index,:])))
        #normal
        #tau, A, tau2, A2, c, reso = poly_fit(np.arange(0,len(red)), aux_vec_cumu[index,:])
        #tau, A, tau2, A2, c, reso = poly_fit(np.arange(100,500), aux_vec_cumu[index,100:500])        
        #poisson
        tau, A, tau2, A2, c, reso = poly_fit(np.arange(0,len(red)-1), aux_vec_cumu[index,1:])
        #print_result(reso)
        
        x = np.arange(0,len(red))
        #y = c*x + tau*A*(1- np.exp(-tx/tau)) + tau2*A2*(1- np.exp(-x/tau2))
        y = func_poly(x, reso.params)
        
        ax2.plot(x, y, lw=len(listofindex)-index,color='k')       
        
        
        #plt.figure(100)
        
        #y = (aux_vec_cumu[index,:] - c*x)
        #y = y/np.mean(y[-100:])
        
        #plt.plot(x, y,lw=len(listofindex)-index,color=my_color)        
        plt.figure(2)

        #y = c*x + 0*tau*A*(1- np.exp(-x/tau))        
        ax2.plot(x, y, lw=len(listofindex)-index,color='k',ls='--')        
        
        #y = 0*c*x + tau*A*(1- np.exp(-x/tau))        
        #ax2.plot(x, y, lw=len(listofindex)-index,color='k', ls=':')        
        
        ax2.set_xlabel('tau')
        ax2.set_ylabel('cumu counts')
        ax2.set_xlim(xmax=1500)
        #plt.xlim([0,20])
        ax2.set_title('Lw $\propto$' +Label_varying_variable)
        
        if index == len(listofindex)-1:
            return aux_vec, aux_vec_cumu, median_tau_red, one_over_e_tau_red,  mean_int_median_before, mean_int_median_after,  mean_int_one_over_e_before, mean_int_one_over_e_after

aux_vec_red, aux_vec_cumu_red, median_tau_red, one_over_e_tau_red,  mean_int_median_before_red, mean_int_median_after_red,  mean_int_one_over_e_before_red, mean_int_one_over_e_after_red = plot_things('RED', 'r')
aux_vec_blue, aux_vec_cumu_blue, median_tau_blue, one_over_e_tau_blue,  mean_int_median_before_blue, mean_int_median_after_blue,  mean_int_one_over_e_before_blue, mean_int_one_over_e_after_blue = plot_things('BLUE', 'g')


plt.show()