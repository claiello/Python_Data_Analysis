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

###############################################################################
# FUNCTIONS TO CALCULATE SIGNAL AND ERROR
def signal_cumu(counts, error):
    
    ucounts = unumpy.uarray(counts, error)
    
    return np.cumsum(ucounts, axis = 1)
    
def signal_cumu_ratio(counts_red, error_red, counts_blue, error_blue):
    
    ucounts_red = unumpy.uarray(counts_red, error_red)
    ucounts_blue = unumpy.uarray(counts_blue, error_blue)
    
    return np.cumsum(ucounts_red, axis = 1)/np.cumsum(ucounts_blue, axis = 1)
    

def signal_cumu_visibility(counts_red, error_red, counts_blue, error_blue):
    
    ucounts_red = unumpy.uarray(counts_red, error_red)
    ucounts_blue = unumpy.uarray(counts_blue, error_blue)
    
    return (np.cumsum(ucounts_red, axis = 1)-np.cumsum(ucounts_blue, axis = 1))/(np.cumsum(ucounts_red, axis = 1)+np.cumsum(ucounts_blue, axis = 1))
    
    
def signal_rho(counts, error):
    
    ucounts = unumpy.uarray(counts,error)
    rho = unumpy.uarray(counts,error)# ([counts.shape[0], counts.shape[1]])
    
    for j in np.arange(0, counts.shape[0]):
        for jj in np.arange(0,counts.shape[1]-1):
            print(jj)
            rho[j,jj] = np.sum(ucounts[j,0:jj])/np.sum(ucounts[j,jj:])
            
    print(np.sum(np.isnan(unumpy.nominal_values(rho[:,:-1]))))
    return rho[:,1:-1]
    
def signal_rho_ratio(counts_red, error_red, counts_blue, error_blue):
    
    ucounts_red = unumpy.uarray(counts_red,error_red)
    rho_red = unumpy.uarray(counts_red,error_red)
    
    ucounts_blue = unumpy.uarray(counts_blue,error_blue)
    rho_blue = unumpy.uarray(counts_blue,error_blue)
    
    for j in np.arange(0, counts_red.shape[0]):
        for jj in np.arange(0,counts_red.shape[1]-1):
            print(jj)
            try:
                rho_red[j,jj] = np.sum(ucounts_red[j,0:jj])/np.sum(ucounts_red[j,jj:])
                rho_blue[j,jj] = np.sum(ucounts_blue[j,0:jj])/np.sum(ucounts_blue[j,jj:])
            except:
                rho_red[j,jj] = np.nan
                rho_blue[j,jj] = np.nan
        
    return rho_red[:,1:-1]/rho_blue[:,1:-1]
    
def signal_rho_visibility(counts_red, error_red, counts_blue, error_blue):
    
    ucounts_red = unumpy.uarray(counts_red,error_red)
    rho_red = unumpy.uarray(counts_red,error_red)
    
    ucounts_blue = unumpy.uarray(counts_blue,error_blue)
    rho_blue = unumpy.uarray(counts_blue,error_blue)
    
    for j in np.arange(0, counts_red.shape[0]-1):
        for jj in np.arange(0,counts_red.shape[1]):
            print(jj)
            try:
                rho_red[j,jj] = np.sum(ucounts_red[j,0:jj])/np.sum(ucounts_red[j,jj:])
                rho_blue[j,jj] = np.sum(ucounts_blue[j,0:jj])/np.sum(ucounts_blue[j,jj:])
            except:
                rho_red[j,jj] = np.nan
                rho_blue[j,jj] = np.nan
        
    try:
        return (rho_red[:,1:-1]-rho_blue[:,1:-1])/(rho_red[:,1:-1]+rho_blue[:,1:-1])
    except:
        return np.nan
    

###############################################################################

d_ap = pickle.load( open( "d0.p", "rb" ) )
d_kv = pickle.load( open( "d1.p", "rb" ) )
d_pixel = pickle.load( open( "d2.p", "rb" ) )
d_temp = pickle.load( open( "d4.p", "rb" ) )
d_temp_large_zoom = pickle.load( open( "d5.p", "rb" ) )
d_temp_small_zoom = pickle.load( open( "d6.p", "rb" ) )

my_color = ['r','r','r','r','k','b']
lst = ['-.','--','dotted','-','-','-']


########

#do_cumu = True
#do_cumu = False
#
#do_ratio = True
#do_ratio = False
#
lab = ['Aperture','Current','Pixel','Temperature medium zoom','Temperature large zoom','Temperature small zoom']
######### RATIOS RED AND GREEN
for index in [3]: #[0,1,2,3]:
    
    if index == 0:
        do_cumu = True
        do_ratio = True
    elif index == 1:
        do_cumu = True
        do_ratio = False
    elif index == 2:
        do_cumu = False
        do_ratio = True
    elif index == 3:
        do_cumu = False
        do_ratio = False     

    ct = 0
    for data in [d_ap, d_kv, d_pixel, d_temp,d_temp_large_zoom, d_temp_small_zoom]:
        
        if do_cumu: #do cumu
            if do_ratio:
                black = signal_cumu_ratio(data['red1D'], np.sqrt(data['red1D']),data['blue1D'], np.sqrt(data['blue1D']))
                ts_b = np.arange(0,data['red1D'].shape[1])
                prefix = 'CUMURATIO'
            else: 
                black = signal_cumu_visibility(data['red1D'], np.sqrt(data['red1D']),data['blue1D'], np.sqrt(data['blue1D']))
                ts_b = np.arange(0,data['red1D'].shape[1])
                prefix = 'CUMUVISIBILITY'
            
        else: #do rho
            notocut = 10
            if do_ratio:
                black = signal_rho_ratio(data['red1D'][:,:-notocut], np.sqrt(data['red1D'][:,:-notocut]),data['blue1D'][:,:-notocut], np.sqrt(data['blue1D'][:,:-notocut]))
                ts_b = np.arange(0,data['red1D'].shape[1]-notocut-2)
                prefix = 'RHORATIO'
            else:
                black = signal_rho_visibility(data['red1D'][:,:-notocut], np.sqrt(data['red1D'][:,:-notocut]),data['blue1D'][:,:-notocut], np.sqrt(data['blue1D'][:,:-notocut]))
                ts_b = np.arange(0,data['red1D'].shape[1]-notocut-2)
                prefix = 'RHOVISIBILITY'
        
    
        rhos = np.linspace(0,600,100)
        (eta_rho_b, eta_time_b,eta_rho_sig_b, eta_time_sig_b)  = get_sens(black, np.array(data['x_values']),ts_b,rhos)
        
        plt.figure()
        plt.suptitle(lab[ct] + ' ' + prefix,fontsize=24)
        
        plt.subplot(2,1,1)
        plt.plot(ts_b, eta_rho_b,color='k',ls='-')
        plt.plot(ts_b, eta_rho_sig_b,color='k',ls='--')
        plt.xlabel('Time',fontsize=24)
        plt.ylabel('Sensitivity',fontsize=24)
        plt.xlim(xmax=1000)
        plt.ylim(ymax=500)
        
        plt.subplot(2,1,2)
        plt.plot(rhos, eta_time_b,color='k',ls='-')
        plt.plot(rhos, eta_time_sig_b,color='k',ls='--')
        plt.xlabel('Signal',fontsize=24)
        plt.ylabel('Sensitivity',fontsize=24)
        plt.xlim(xmax=1000)
        plt.ylim(ymax=500)
    
        import pickle
        save_data = {}
        save_data['eta_rho_b'] =  eta_rho_b
        save_data['eta_rho_sig_b'] = eta_rho_sig_b
        save_data['ts'] = ts_b
        save_data['eta_time_b'] =  eta_time_b
        save_data['eta_time_sig_b'] = eta_time_sig_b
        save_data['rhos'] = rhos
       
        pickle.dump(save_data, open( lab[ct] + prefix + '.p', "wb"))
             
        multipage_longer(lab[ct] + prefix +'.pdf',dpi=80)    
        
        ct = ct + 1


klklklk

lab = ['Aperture','Current','Pixel','Temperature medium zoom','Temperature large zoom','Temperature small zoom']
do_cumu = True
#do_cumu = False #do rho
##### RED AND GREEN
ct = 0
for data in [d_ap, d_kv, d_pixel, d_temp,d_temp_large_zoom, d_temp_small_zoom]:
    
    if do_cumu:
        ### cumu
        red = signal_cumu(data['red1D'], np.sqrt(data['red1D']))
        blue = signal_cumu(data['blue1D'], np.sqrt(data['blue1D']))
        ts_r = np.arange(0,data['red1D'].shape[1])
        ts_b = np.arange(0,data['blue1D'].shape[1])
        prefix = 'CUMU'
    else:
        ### rho
        notocut = 10 
        red = signal_rho(data['red1D'][:,:-notocut], np.sqrt(data['red1D'][:,:-notocut]))
        blue = signal_rho(data['blue1D'][:,:-notocut], np.sqrt(data['blue1D'][:,:-notocut]))
        ts_r = np.arange(0,data['red1D'].shape[1]-notocut-2)
        ts_b = np.arange(0,data['blue1D'].shape[1]-notocut-2)
        print(red.shape)
        print(ts_r.shape)
        prefix = 'RHO'
    
    rhos = np.linspace(0,600,100)
    (eta_rho_r, eta_time_r,eta_rho_sig_r, eta_time_sig_r) = get_sens(red, np.array(data['x_values']),ts_r,rhos)
    (eta_rho_b, eta_time_b,eta_rho_sig_b, eta_time_sig_b)  = get_sens(blue, np.array(data['x_values']),ts_b,rhos)
    
    plt.figure()
    plt.suptitle(lab[ct] + ' ' + prefix,fontsize=24)
    
    plt.subplot(2,1,1)
    plt.plot(ts_r, eta_rho_r,color='r',ls='-')
    plt.plot(ts_r, eta_rho_sig_r,color='r',ls='--')
    plt.plot(ts_b, eta_rho_b,color='g',ls='-')
    plt.plot(ts_b, eta_rho_sig_b,color='g',ls='--')
    plt.xlabel('Time',fontsize=24)
    plt.ylabel('Sensitivity',fontsize=24)
    ind = 1000
    plt.xlim(xmax=ind)
    plt.ylim(ymax=max(np.max(eta_rho_r[:ind]),np.max(eta_rho_sig_r[:ind]),np.max(eta_rho_b[:ind]),np.max(eta_rho_sig_b[:ind])))
    
    plt.subplot(2,1,2)
    plt.plot(rhos, eta_time_r,color='r',ls='-')
    plt.plot(rhos, eta_time_sig_r,color='r',ls='--')
    plt.plot(rhos, eta_time_b,color='g',ls='-')
    plt.plot(rhos, eta_time_sig_b,color='g',ls='--')
    plt.xlabel('Signal',fontsize=24)
    plt.ylabel('Sensitivity',fontsize=24)
    plt.ylim(ymax=2000)

    import pickle
    save_data = {}
    save_data['eta_rho_r'] =  eta_rho_r
    save_data['eta_rho_sig_r'] = eta_rho_sig_r
    save_data['eta_rho_b'] =  eta_rho_b
    save_data['eta_rho_sig_b'] = eta_rho_sig_b
    save_data['ts'] = ts_b
    save_data['eta_time_r'] =  eta_time_r
    save_data['eta_time_sig_r'] = eta_time_sig_r
    save_data['eta_time_b'] =  eta_time_b
    save_data['eta_time_sig_b'] = eta_time_sig_b
    save_data['rhos'] = rhos
   
    pickle.dump(save_data, open('../' + lab[ct] + prefix + '.p', "wb"))
    
    multipage_longer(lab[ct] + prefix +'.pdf',dpi=80)    
    
    ct = ct + 1
    

#plt.show()

klklklk

#rho const
plt.figure()
cnt = 0
counterinf = 0
for data in [d_ap, d_kv, d_pixel, d_temp,d_temp_large_zoom, d_temp_small_zoom]:

    taus = np.arange(data['aux_vec_red'].shape[1])

    grad_result = []
    tau_result = []

    for k in range(len(taus)):

        x = np.array(data['x_values'])
        y = np.array(data['aux_vec_red'][:, k])
        
        if np.sum(np.isinf(y)) == 0:
            (a, b, result) = linear_fit(x, y)
        else:
            counterinf = counterinf + 1

        tau_result.append(taus[k])
        grad_result.append(a)
    
    
    print(np.sum(tau_result))
    print(np.sum(grad_result))
    print(len(tau_result))
    print(len(grad_result))
    print("next")
    plt.plot(tau_result, grad_result, my_color[cnt])
    cnt += 1

print(counterinf) #== 7, not that many infs
plt.show()

klklklk


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    #return array[idx]
    return idx

plt.figure()
cnt = 0
for data in [d_ap, d_kv, d_pixel, d_temp,d_temp_large_zoom, d_temp_small_zoom]:

    rhos = np.linspace(0, 200, 300)

    taus = np.arange(data['aux_vec_red'].shape[1])

    grad_result = []
    rho_result = []

    for k in range(len(rhos)):
    #for k in range(2):

        x = np.array(data['x_values'])
        hlp = np.array(data['aux_vec_red'])
        y = np.array([])
        
        for ind_T in range(len(x)):
            idx = find_nearest(hlp[ind_T, :], rhos[k])
            y = np.append(y, taus[idx])
        
        
        if np.sum(np.isinf(y)) == 0:
            (a, b, result) = linear_fit(x, y)

            rho_result.append(rhos[k])
            grad_result.append(a)
        
    plt.plot(rho_result, np.abs(grad_result)/max(np.abs(grad_result)), my_color[cnt],linestyle=lst[cnt],label=lab[cnt])
    
    cnt += 1

plt.xlabel(r'$\rho$',fontsize = 24)
plt.ylabel(r'|d(Param)/d$\tau$|',fontsize = 24)
plt.legend(loc='best')

plt.show()


