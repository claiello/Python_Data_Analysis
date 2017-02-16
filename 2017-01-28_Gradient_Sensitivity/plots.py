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

import pickle
from uncertainties import unumpy
from calc_sens import *

lab = ['Aperture','Current','Pixel','Temperature medium zoom','Temperature large zoom','Temperature small zoom']

#prefix = 'CUMU'
##prefix = 'RHO'
#

#
#for ct in np.arange(0,len(lab)):
#
#    data = pickle.load( open(lab[ct] + prefix + '.p', "rb" ) )
#    
#    plt.figure()
#    plt.suptitle(lab[ct] + ' ' + prefix,fontsize=24)
#    
#    plt.subplot(4,1,1)
#    plt.plot(data['ts'], data['eta_rho_sig_r'],color='r',ls='-')
#    plt.plot(data['ts'], data['eta_rho_sig_b'],color='g',ls='-')
#    plt.xlabel('Time',fontsize=24)
#    plt.ylabel('Sensitivity \n sig error',fontsize=24)
#    ind = 1000
#    plt.xlim(xmax=ind)
#    plt.ylim(ymax=max(np.max(data['eta_rho_sig_r'][:ind]),np.max(data['eta_rho_sig_b'][:ind])))
#    
#    plt.subplot(4,1,2)
#    plt.plot(data['ts'], data['eta_rho_r'],color='r',ls='-')
#    plt.plot(data['ts'], data['eta_rho_b'],color='g',ls='-')
#    plt.xlabel('Time',fontsize=24)
#    plt.ylabel('Sensitivity \n fit error',fontsize=24)
#    ind = 1000
#    plt.xlim(xmax=ind)
#    plt.ylim(ymax=max(np.max(data['eta_rho_r'][:ind]),np.max(data['eta_rho_b'][:ind])))
#        
#    plt.subplot(4,1,3)
#    plt.plot(data['rhos'][1:], data['eta_time_sig_r'][1:],color='r',ls='-')
#    plt.plot(data['rhos'][1:], data['eta_time_sig_b'][1:],color='g',ls='-')
#    plt.xlabel('Signal',fontsize=24)
#    plt.ylabel('Sensitivity \n sig error',fontsize=24)
#    
#    plt.subplot(4,1,4)
#    plt.plot(data['rhos'], data['eta_time_r'],color='r',ls='-')
#    plt.plot(data['rhos'], data['eta_time_b'],color='g',ls='-')
#    plt.xlabel('Signal',fontsize=24)
#    plt.ylabel('Sensitivity \n fit error',fontsize=24)
#   
#    multipage_longer(lab[ct] + prefix +'.pdf',dpi=80)    
#    
#klklkl 
   
#prefix = 'CUMURATIO'
##prefix = 'RHORATIO' 
#
##prefix = 'CUMUVISIBILITY'
##prefix = 'RHOVISIBILITY'
#
#for ct in np.arange(0,len(lab)):
#    
#    data = pickle.load( open(lab[ct] + prefix + '.p', "rb" ) )
#
#    plt.figure()
#    plt.suptitle(lab[ct] + ' ' + prefix,fontsize=24)
#    
#    plt.subplot(4,1,1)
#    plt.plot(data['ts'], data['eta_rho_sig_b'],color='k',ls='-')
#    plt.xlabel('Time',fontsize=24)
#    plt.ylabel('Sensitivity \n sig error',fontsize=24)
#    ind = 1000
#    #plt.xlim(xmax=ind)
#    #plt.ylim(ymax=np.max(data['eta_rho_sig_b'][:ind]))
#    
#    plt.subplot(4,1,2)
#    plt.plot(data['ts'], data['eta_rho_b'],color='k',ls='-')
#    plt.xlabel('Time',fontsize=24)
#    plt.ylabel('Sensitivity \n fit error',fontsize=24)
#    ind = 1000
#    #plt.xlim(xmax=ind)
#    #plt.ylim(ymax=np.max(data['eta_rho_b'][:ind]))
#    
#    plt.subplot(4,1,3)
#    plt.plot(data['rhos'][1:], data['eta_time_sig_b'][1:],color='k',ls='-')
#    plt.xlabel('Signal',fontsize=24)
#    plt.ylabel('Sensitivity \n sig error',fontsize=24)
#    
#    plt.subplot(4,1,4)
#    plt.plot(data['rhos'], data['eta_time_b'],color='k',ls='-')
#    plt.xlabel('Signal',fontsize=24)
#    plt.ylabel('Sensitivity \n fit error',fontsize=24)
#    
# 
#multipage_longer(prefix +'.pdf',dpi=80)    

##My choice of graph
#prefix = 'CUMUVISIBILITY'
#
#lab = ['Temperature medium zoom','Temperature large zoom','Temperature small zoom']
#ax0 = plt.figure()
#ct = 0
#for ct in np.arange(0,len(lab)):
#    
#    data = pickle.load( open(lab[ct] + prefix + '.p', "rb" ) )
#
#    ax0.plot(data['ts'], data['eta_rho_sig_b'],color='k',ls='-')
#    ax0.set_xlabel('Time',fontsize=24)
#    ax0.set_ylabel('Sensitivity \n sig error',fontsize=24)
#    ind = 1000
#    #plt.xlim(xmax=ind)
#    #plt.ylim(ymax=np.max(data['eta_rho_sig_b'][:ind]))
#    ax0.tick_params(labelsize=fsizenb)
#
#    
#    
#    
# 
#multipage_longer(prefix +'.pdf',dpi=80)    





#my choice of plot
prefix = 'CUMU'

for ct in np.arange(0,len(lab)):

    data = pickle.load( open(lab[ct] + prefix + '.p', "rb" ) )
    
    plt.figure()
    plt.suptitle(lab[ct] + ' ' + prefix,fontsize=24)
    
    plt.subplot(4,1,1)
    plt.plot(data['ts'], data['eta_rho_sig_r'],color='r',ls='-')
    plt.plot(data['ts'], data['eta_rho_sig_b'],color='g',ls='-')
    plt.xlabel('Time',fontsize=24)
    plt.ylabel('Sensitivity \n sig error',fontsize=24)
    ind = 1000
    plt.xlim(xmax=ind)
    plt.ylim(ymax=max(np.max(data['eta_rho_sig_r'][:ind]),np.max(data['eta_rho_sig_b'][:ind])))
    
   
multipage_longer( prefix +'.pdf',dpi=80)    
    