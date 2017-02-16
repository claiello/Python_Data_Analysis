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
from sklearn.mixture import GMM 
import matplotlib.cm as cm
#from FluoDecay import *
#from PlottingFcts import *
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

from numpy import genfromtxt
from uncertainties import unumpy

### settings
fsizepl = 24
fsizenb = 20
sizex = 8
sizey=6
dpi_no = 80
#mkstry = ['8','11','5'] #marker size for different dsets Med Zoom/Large Zoom/Small Zoom
###

#fig1= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
#fig1.set_size_inches(1200./fig1.dpi,900./fig1.dpi)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.rc('font', serif='Palatino')    
#
#noplots = 2
#nolines = 2
#
#ax0 = plt.subplot2grid((nolines,noplots), (1,1), colspan=1, rowspan=1)
#ax0.spines['right'].set_visible(False)
#ax0.spines['top'].set_visible(False)
#ax0.xaxis.set_ticks_position('bottom')
#ax0.yaxis.set_ticks_position('left')
#     
##labels
##xmin = 0
##xmax = 1390
#ax0.set_title('Signal(t) = cumulative counts \n red and green ',fontsize=fsizepl)
##ax0.set_xlim([xmin, xmax])
#ax0.set_ylabel('Sensitivity (K)',fontsize=fsizepl)
#ax0.set_xlabel('Transient cathodoluminescence \n acquisition time ($\mu$s)',fontsize=fsizepl)
#ax0.tick_params(labelsize=fsizenb)
#ax0.set_xticks([500,1000])#,1500])
#ax0.set_ylim((0.1,100))
#lab = ['Temperature medium zoom','Temperature large zoom','Temperature small zoom']

#prefix = 'CUMU'
#for ct in np.arange(0,len(lab)):
#    
#    data = pickle.load( open(lab[ct] + prefix + '.p', "rb" ) )
#    
##    ax0.set_xscale('log')
##    ax0.set_yscale('log')
##    ax0.plot(data['ts'], data['eta_rho_sig_b'],color='g',ls='-',lw=2)
##    ax0.plot(data['ts'][0::10], data['eta_rho_sig_b'][0::10],color='g',ls='',marker='o',markersize=mkstry[ct])
##    ax0.plot(data['ts'], data['eta_rho_sig_r'],color='r',ls='-',lw=2)
##    ax0.plot(data['ts'][0::10], data['eta_rho_sig_r'][0::10],color='r',ls='',marker='o',markersize=mkstry[ct])
#    
#    ax0.semilogy(data['ts'], data['eta_rho_sig_b'],color='g',ls='-',lw=2)
#    ax0.semilogy(data['ts'][0::50], data['eta_rho_sig_b'][0::50],color='g',ls='',marker='o',markersize=mkstry[ct])
#    ax0.semilogy(data['ts'], data['eta_rho_sig_r'],color='r',ls='-',lw=2)
#    ax0.semilogy(data['ts'][0::50], data['eta_rho_sig_r'][0::50],color='r',ls='',marker='o',markersize=mkstry[ct])
    

######################################################################################################################
do_fig_taus = False
if do_fig_taus:
    fig1= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig1.set_size_inches(1200./fig1.dpi,900./fig1.dpi) #1200 900
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')    
    
    noplots = 4
    nolines = 4
    
    ax00 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)
    ax01 = plt.subplot2grid((nolines,noplots), (0,1), colspan=1, rowspan=1)
    ax02 = plt.subplot2grid((nolines,noplots), (0,2), colspan=1, rowspan=1)
    ax03 = plt.subplot2grid((nolines,noplots), (0,3), colspan=1, rowspan=1)
    
    ax10 = plt.subplot2grid((nolines,noplots), (1,0), colspan=1, rowspan=1)
    ax11 = plt.subplot2grid((nolines,noplots), (1,1), colspan=1, rowspan=1)
    ax12 = plt.subplot2grid((nolines,noplots), (1,2), colspan=1, rowspan=1)
    ax13 = plt.subplot2grid((nolines,noplots), (1,3), colspan=1, rowspan=1)
    
    ax20 = plt.subplot2grid((nolines,noplots), (2,0), colspan=1, rowspan=1)
    ax21 = plt.subplot2grid((nolines,noplots), (2,1), colspan=1, rowspan=1)
    ax22 = plt.subplot2grid((nolines,noplots), (2,2), colspan=1, rowspan=1)
    ax23 = plt.subplot2grid((nolines,noplots), (2,3), colspan=1, rowspan=1)
    
    ax40 = plt.subplot2grid((nolines,noplots), (3,0), colspan=1, rowspan=1)
    ax41 = plt.subplot2grid((nolines,noplots), (3,1), colspan=1, rowspan=1)
    ax42 = plt.subplot2grid((nolines,noplots), (3,2), colspan=1, rowspan=1)
    ax43 = plt.subplot2grid((nolines,noplots), (3,3), colspan=1, rowspan=1)
    
    def tauestimate(counts_red, error_red, counts_blue, error_blue):
        
        print(counts_red.shape[1])
        
        ucounts_red = unumpy.uarray(counts_red, error_red)
        ucounts_blue = unumpy.uarray(counts_blue, error_blue)
        
        def helper(arrayx):
             return np.cumsum(arrayx*np.arange(1,counts_red.shape[1]+1), axis = 1)/np.cumsum(arrayx, axis = 1)
        
        return helper(ucounts_red),helper(ucounts_blue)
    
    d_ap = pickle.load( open( "d0.p", "rb" ) ) #4#
    d_kv = pickle.load( open( "d1.p", "rb" ) ) #3#
    d_pixel = pickle.load( open( "d2.p", "rb" ) ) #5#
    d_temp = pickle.load( open( "d4.p", "rb" ) ) #4#
    
    import matplotlib.cm as cm
    ct = 0
    for data in [d_ap, d_kv, d_pixel, d_temp]:
        
        if ct == 0:
            xvec = np.array([28, 379, 1800, 5700])
            my_ax = [ax03,ax13,ax23,ax43]
            xlab = 'Electron beam \n current (pA)'
            xl = [-300,6000]
            xt = [30,5500]
        elif ct== 1:
            xvec =np.array( [10,15,20])
            my_ax = [ax02,ax12,ax22,ax42]
            xlab = 'Electron beam \n energy (kV)'
            xl = [5,25]
            xt = [10,20]
        elif ct == 2:
            xvec = np.array([2.23, 1.79, 1.49, 1.28, 1.12])
            xlab = 'Pixel size (nm)'
            my_ax = [ax01,ax11,ax21,ax41]
            xl = [0.9, 2.4]
            xt = [1,2]
        elif ct == 3:
            xvec = np.array([1./0.002996- 273.15,1./0.0031089- 273.15,1./0.0031954- 273.15,1./0.003285- 273.15] )
            my_ax = [ax00,ax10,ax20,ax40] 
            xlab = 'Temperature \n' + r'at heater ($^{\circ}$C)'
            xl = [25,65]
            xt = [30,40,50,60]
            
        for my_ind in my_ax:
            my_ind.xaxis.set_ticks_position('bottom')
            my_ind.yaxis.set_ticks_position('left')
            
        (taured,taublue) = tauestimate(data['red1D'], np.sqrt(data['red1D']),data['blue1D'], np.sqrt(data['blue1D']))
        ts_b = np.arange(0,data['red1D'].shape[1])
        
        aa = np.empty([taured.shape[1]])    
        cc = np.empty([taured.shape[1]])   
        bb = np.empty([taured.shape[1]])    
        dd = np.empty([taured.shape[1]]) 
        bberr = np.empty([taured.shape[1]])    
        dderr = np.empty([taured.shape[1]]) 
        for jjj in np.arange(1,taured.shape[1]):
            print(jjj)
            (a,b,result) = linear_fit_with_error(xvec, unumpy.nominal_values(taured[:,jjj]), unumpy.std_devs(taured[:,jjj]))
            (c,d,result2) = linear_fit_with_error(xvec, unumpy.nominal_values(taublue[:,jjj]), unumpy.std_devs(taublue[:,jjj]))
            aa[jjj] = a
            cc[jjj] = c
            bb[jjj] = b
            dd[jjj] = d
            bberr[jjj] = result.params['b'].stderr
            dderr[jjj] = result.params['b'].stderr
            
        step = 50
        initindex = 5
        colors = iter(cm.rainbow(np.linspace(0, 1, len(np.arange(initindex,taured.shape[1],step)))))   
        
        
      
        # visibility of cumu counts
        red = data['red1D']
        blue = data['blue1D']
        viscumucts = (np.cumsum(unumpy.nominal_values(red),axis=1)-np.cumsum(unumpy.nominal_values(blue),axis=1))/       \
                         (np.cumsum(unumpy.nominal_values(red),axis=1)+np.cumsum(unumpy.nominal_values(blue),axis=1))
        #viscumucts = (viscumucts-np.min(viscumucts))/(np.max(viscumucts)-np.min(viscumucts)) 
        viscumucts = (1 - (np.max(viscumucts)-viscumucts)/(np.max(viscumucts))) * 100.0
        
        hlp3hlp3 = np.empty([taured.shape[0],taured.shape[1]])
        for jjj in np.arange(0,taured.shape[1]):
            hlp3hlp3[:,jjj] = (unumpy.nominal_values(taured[:,jjj])-unumpy.nominal_values(taublue[:,jjj]))/(unumpy.nominal_values(taured[:,jjj])+unumpy.nominal_values(taublue[:,jjj]))
        #hlp3hlp3 = (hlp3hlp3-np.min(hlp3hlp3))/(np.max(hlp3hlp3)-np.min(hlp3hlp3))            
        hlp3hlp3 = (1 - (np.max(hlp3hlp3)-hlp3hlp3)/(np.max(hlp3hlp3)))   * 100.0         
        
        for jjj in np.arange(initindex,taured.shape[1],step):
            print(jjj)
            colored = next(colors)
            #tau red
            #hlp1= unumpy.nominal_values(taured[:,jjj])/np.max(unumpy.nominal_values(taured))   #np.linalg.norm(unumpy.nominal_values(taured))
            hlp1= (1-(np.max(unumpy.nominal_values(taured))-unumpy.nominal_values(taured[:,jjj]))/np.max(unumpy.nominal_values(taured)) )* 100.0  
            #tau blue
            #hlp2= unumpy.nominal_values(taublue[:,jjj])/np.max(unumpy.nominal_values(taublue))    #np.linalg.norm(unumpy.nominal_values(taublue))
            hlp2=(1- (np.max(unumpy.nominal_values(taublue)) -unumpy.nominal_values(taublue[:,jjj]))/np.max(unumpy.nominal_values(taublue))) * 100.0  
           
            my_ax[1].plot(xvec,hlp1,marker='o',ls='dotted',color=colored,markersize=8)
            my_ax[0].plot(xvec,hlp2,marker='o',ls='dotted',color=colored,markersize=8)
           
            
            my_ax[2].plot(xvec,hlp3hlp3[:,jjj],marker='o',ls='dotted',color=colored,markersize=8)
           
            my_ax[3].plot(xvec,viscumucts[:,jjj],marker='o',ls='dotted',color=colored,markersize=8)
                
            my_ax[3].set_xlabel(xlab, fontsize=fsizepl)
            
            my_ax[0].set_xticks(xt)
            my_ax[0].set_xlim(xl)
            my_ax[0].set_xticklabels([])
            my_ax[1].set_xticks(xt)
            my_ax[1].set_xlim(xl)
            my_ax[1].set_xticklabels([])
            my_ax[2].set_xticks(xt)
            my_ax[2].set_xlim(xl)
            my_ax[2].set_xticklabels([])
            my_ax[3].set_xticks(xt)
            my_ax[3].set_xlim(xl)
            
            my_ax[0].set_ylim([0,100])
            my_ax[1].set_ylim([0,100])
            my_ax[2].set_ylim([-350,100]) #####was -250
            my_ax[3].set_ylim([50,100])#([0.3,0.6])
            
            my_ax[0].set_yticks([50,100])
            my_ax[1].set_yticks([50,100])
            my_ax[2].set_yticks([100,0,-100,-200,-300])#([-0.25,-0.05,0.15])
            my_ax[3].set_yticks([75,100])#([0.3,0.45,0.6])
            
            if ct == 3:
                my_ax[1].set_ylabel(r'Red band $\tau$' + '\n' + r'variation ($\%$)',fontsize=fsizepl)#,va='center',ha='center')
                my_ax[0].set_ylabel(r'Green band $\tau$' + '\n' + r'variation ($\%$)',fontsize=fsizepl)#,va='center',ha='center')
                my_ax[2].set_ylabel(r'Visib. of $\tau$' + '\n' + r'variation ($\%$)',fontsize=fsizepl)#,va='center',ha='center')
                my_ax[3].set_ylabel('Visib. of cumul. \n' + r'counts variation ($\%$)',fontsize=fsizepl)
            else:
                my_ax[0].set_yticklabels([])
                my_ax[1].set_yticklabels([])
                my_ax[2].set_yticklabels([])
                my_ax[3].set_yticklabels([])
            
            
        my_ax[0].tick_params(labelsize=fsizenb)  
        my_ax[1].tick_params(labelsize=fsizenb)  
        my_ax[2].tick_params(labelsize=fsizenb)  
        my_ax[3].tick_params(labelsize=fsizenb)  
        ct = ct + 1
        
    plt.tight_layout() 
    plt.show()  
    multipage_longer('SI-TimeEvolution.pdf',dpi=80)

 
#####
#print(ct)
#If ct = 4, in memory will be data for temp
#### 
do_fig_sens = False
if do_fig_sens:
   # sizex = 8
    #sizey = 6
    fig1= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig1.set_size_inches(1200./fig1.dpi,900./fig1.dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')    
    
    noplots = 2
    nolines = 2
    
    ax00 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)
    ax00.spines['right'].set_visible(False)
    ax00.spines['top'].set_visible(False)
    ax00.xaxis.set_ticks_position('bottom')
    ax00.yaxis.set_ticks_position('left')
    
    def tauestimate(counts_red, error_red, counts_blue, error_blue):
        
        print(counts_red.shape[1])
        
        ucounts_red = unumpy.uarray(counts_red, error_red)
        ucounts_blue = unumpy.uarray(counts_blue, error_blue)
        
        def helper(arrayx):
             return np.cumsum(arrayx*np.arange(1,counts_red.shape[1]+1), axis = 1)/np.cumsum(arrayx, axis = 1)
        
        return helper(ucounts_red),helper(ucounts_blue)
    
    d_ap = pickle.load( open( "d0.p", "rb" ) ) #4#
    d_kv = pickle.load( open( "d1.p", "rb" ) ) #3#
    d_pixel = pickle.load( open( "d2.p", "rb" ) ) #5#
    d_temp = pickle.load( open( "d4.p", "rb" ) ) #4#
    
    import matplotlib.cm as cm
    ct = 3
    for data in [d_temp]:
        
        if ct == 3:
            xvec = np.array([1./0.002996- 273.15,1./0.0031089- 273.15,1./0.0031954- 273.15,1./0.003285- 273.15] )
            my_ax = [ax00]#,ax10,ax20,ax40] 
            xlab = 'Transient cathodoluminescence \n acquisition time ($\mu$s)'
            xl = [0,1390]
            xt = [500,1000]
            
        for my_ind in my_ax:
            my_ind.xaxis.set_ticks_position('bottom')
            my_ind.yaxis.set_ticks_position('left')
            
        (taured,taublue) = tauestimate(data['red1D'], np.sqrt(data['red1D']),data['blue1D'], np.sqrt(data['blue1D']))
        ts_b = np.arange(0,data['red1D'].shape[1])
        
        aa = np.empty([taured.shape[1]])    
        cc = np.empty([taured.shape[1]])   
        bb = np.empty([taured.shape[1]])    
        dd = np.empty([taured.shape[1]]) 
        bberr = np.empty([taured.shape[1]])    
        dderr = np.empty([taured.shape[1]]) 
        for jjj in np.arange(1,taured.shape[1]):
            print(jjj)
            (a,b,result) = linear_fit_with_error(xvec, unumpy.nominal_values(taured[:,jjj]), unumpy.std_devs(taured[:,jjj]))
            (c,d,result2) = linear_fit_with_error(xvec, unumpy.nominal_values(taublue[:,jjj]), unumpy.std_devs(taublue[:,jjj]))
            aa[jjj] = a
            cc[jjj] = c
            bb[jjj] = b
            dd[jjj] = d
            bberr[jjj] = result.params['b'].stderr
            dderr[jjj] = result.params['b'].stderr
            
        step = 50
        initindex = 10
        colors = iter(cm.rainbow(np.linspace(0, 1, len(np.arange(initindex,taured.shape[1],step)))))   
        
    
        # visibility of taus
        hlp3 = (taured-taublue)/(taured+taublue)
        
        # visibility of cumu counts
        red = unumpy.uarray(data['red1D'],np.sqrt(data['red1D']))
        blue = unumpy.uarray(data['blue1D'],np.sqrt(data['blue1D']))
        viscumucts = (np.cumsum(red,axis=1)-np.cumsum(blue,axis=1))/       \
                     (np.cumsum(red,axis=1)+np.cumsum(blue,axis=1))
                         
        sys.path.append("../2017-01-28_Gradient_Sensitivity/") # necessary for the tex fonts
        from calc_sens import get_sens, get_sens_Tminus  
        (xxx, eta_hlp1)  = get_sens(taured, xvec,np.arange(1,red.shape[1]+1))
        (xxx, eta_hlp2)  = get_sens(taublue, xvec,np.arange(1,red.shape[1]+1))
        (xxx, eta_hlp3)  = get_sens(hlp3, xvec,np.arange(1,red.shape[1]+1))
        (xxx, eta_viscumucts)  = get_sens(viscumucts, xvec,np.arange(1,red.shape[1]+1))
   
        delay=2
        my_ax[0].semilogy(np.arange(1+delay+300,red.shape[1]+1),eta_hlp1[delay+300:],ls='--',color='r',lw=2)#,label=r'Red $\tau$')
        my_ax[0].semilogy(np.arange(1+delay,red.shape[1]+1),eta_hlp2[delay:],ls='--',color='g',lw=2)#,label=r'Green $\tau$')
       
        sotep=100
        my_ax[0].semilogy(np.arange(1+delay,red.shape[1]+1),eta_hlp3[delay:],ls='--',color='k',lw=2)
        my_ax[0].semilogy(np.arange(1+delay,red.shape[1]+1,sotep),eta_hlp3[delay::sotep],ls='None',color='k',marker='h', markersize=12,label=r'Visib. of $\tau$')
        my_ax[0].semilogy(np.arange(1+delay,red.shape[1]+1),eta_viscumucts[delay:],ls='--',color='k',lw=2)
        my_ax[0].semilogy(np.arange(1+delay,red.shape[1]+1,sotep),eta_viscumucts[delay::sotep],ls='None',color='k',marker='d', markersize=12,label='Visib. of \n cumul. counts')
            
        my_ax[0].set_xlabel(xlab, fontsize=fsizepl)
        
        my_ax[0].set_xticks(xt)
        my_ax[0].set_xlim(xl)
      
        my_ax[0].set_ylim([3.8,21])
        my_ax[0].set_yticks([10])
        my_ax[0].set_yticklabels(['10'])

        plt.legend(loc='center right',fontsize=fsizenb,frameon=False)
       
        if ct == 3:
            my_ax[0].set_ylabel(r'$\delta$T ($^{\circ}$C)',fontsize=fsizepl)#,va='center',ha='center')
      
            my_ax[0].tick_params(labelsize=fsizenb) 
            
    plt.tight_layout() 
    plt.show()  
    multipage_longer('SI-Sensitivity.pdf',dpi=80)

do_hbar = False
if do_hbar:
    fig1= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig1.set_size_inches(1200./fig1.dpi,900./fig1.dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')    
    
    noplots = 2
    nolines = 2
    
    ax00 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)
    ax00.spines['right'].set_visible(False)
    ax00.spines['top'].set_visible(False)
    ax00.xaxis.set_ticks_position('bottom')
    ax00.yaxis.set_ticks_position('left')
    
    medidas = ['Mutual info.', r'Spearman \textit{r}', r'Pearson \textit{r}',   'Dist. corr.', 'Max. info.']
    Temp = [1.38629436112,1.0, 0.98563,1.0-0.0143673005445,1.0]
    Pixel = [1.60943791243 ,0.89999,0.97876,1.0-0.0212430905327,0.97095]
    kv = [1.09861228867,1.0,0.90897,1.0-0.0910320599631, 0.91830]
    current = [1.38629436112,1.0, 0.80581,1.0-0.19419341885,1.0]
    ind = np.arange(len(medidas))
    width = 0.2
    ax00.barh(ind + 3*width, Temp, width, color='k',edgecolor='None')
    ax00.barh(ind + 2*width, Pixel, width, color='b',edgecolor='None')
    ax00.barh(ind + 1*width, kv, width, color='m',edgecolor='None')
    ax00.barh(ind + 0*width, current, width, color='c',edgecolor='None')
    ax00.set(yticks=ind + 2*width, yticklabels=medidas, ylim=[3*width - 1, len(medidas)])
    ax00.set_xlabel('Coefficient (a.u.)', fontsize=fsizepl)
    ax00.set_xticks([0.5,1.0,1.5])
    ax00.set_xlim([0,1.7])
    ax00.tick_params(labelsize=fsizepl)
    ax00.legend(["Temperature" , "Pixel","Energy","Current"], fontsize=fsizenb,frameon=False)
    ax00.axvline(x=1 , lw=2, color='k', ls='--')
    
    plt.tight_layout() 
    plt.show()  
    multipage_longer('SI-Correlation.pdf',dpi=80)
    
do_plottau = True
if do_plottau:
    
    fig1= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig1.set_size_inches(1200./fig1.dpi,900./fig1.dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')    
    
    noplots = 2
    nolines = 2
    
    ax00b = plt.subplot2grid((nolines,noplots), (1,1), colspan=1, rowspan=1)
    ax00b.spines['left'].set_visible(False)
    ax00b.spines['top'].set_visible(False)
    ax00b.xaxis.set_ticks_position('bottom')
    ax00b.yaxis.set_ticks_position('right')
    
    ax100b = plt.subplot2grid((nolines,noplots), (1,0), colspan=1, rowspan=1)
    ax100b.text(-0.1, 1.0, 'b', transform=ax100b.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})       
    ax100b.spines['right'].set_visible(False)
    ax100b.spines['top'].set_visible(False)
    ax100b.xaxis.set_ticks_position('bottom')
    ax100b.yaxis.set_ticks_position('left')
    
    ax00 = plt.subplot2grid((nolines,noplots), (0,1), colspan=1, rowspan=1)
    ax00.spines['left'].set_visible(False)
    ax00.spines['top'].set_visible(False)
    ax00.xaxis.set_ticks_position('bottom')
    ax00.yaxis.set_ticks_position('right')
    
    ax100 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)
    ax100.text(-0.1, 1.0, 'a', transform=ax100.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})       
    ax100.spines['right'].set_visible(False)
    ax100.spines['top'].set_visible(False)
    ax100.xaxis.set_ticks_position('bottom')
    ax100.yaxis.set_ticks_position('left')
    
    ax00.set_ylabel(r'Red band $\tau$ ($\mu$s)',fontsize=fsizepl)
    ax00.yaxis.set_label_position("right")
    ax100.set_ylabel(r'Green band $\tau$ ($\mu$s)',fontsize=fsizepl)
    
    ax00b.set_ylabel(r'Slope $\partial$(red band $\tau$)/$\partial$t (a.u.)',fontsize=fsizepl)
    ax00b.yaxis.set_label_position("right")
    ax100b.set_ylabel(r'Slope $\partial$(green band $\tau$)/$\partial$t (a.u.)',fontsize=fsizepl)
    
    def tauestimate(counts_red, error_red, counts_blue, error_blue):
    
        print(counts_red.shape[1])
        
        ucounts_red = unumpy.uarray(counts_red, error_red)
        ucounts_blue = unumpy.uarray(counts_blue, error_blue)
        
        def helper(arrayx):
             return np.cumsum(arrayx*np.arange(1,counts_red.shape[1]+1), axis = 1)/np.cumsum(arrayx, axis = 1)
        
        return helper(ucounts_red),helper(ucounts_blue)
        
#    def tending_function(counts, background):
#        
#         hlp1 = np.empty([counts.shape[0], counts.shape[1]])
#         hlp2 = np.empty([counts.shape[0], counts.shape[1]])
#         for jj in np.arange(0,counts.shape[0] ):
#             hlp1[jj] = background[jj]/2.0*(np.arange(1,counts.shape[1]+1))**2
#             hlp2[jj] = background[jj]*(np.arange(1,counts.shape[1]+1))**2
#        
#         return (np.cumsum(counts*np.arange(1,counts.shape[1]+1), axis = 1) +  hlp1)/      \
#                (np.cumsum(counts, axis = 1) + hlp2 )
    
    def moving_average(a,n=3):
        ret = np.cumsum(a)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n-1:]/n

    d_temp = pickle.load( open( "d4.p", "rb" ) ) #4#

    import matplotlib.cm as cm
    ct = 3
    for data in [d_temp]: #[d_ap, d_kv, d_pixel, d_temp]:
    
        if ct == 3:
            xvec = np.array([0.002996,0.0031089,0.0031954,0.003285])
        
        (taured,taublue) = tauestimate(data['red1D'], np.sqrt(data['red1D']),data['blue1D'], np.sqrt(data['blue1D']))    
        
#        #Vector with mean background values 
#        backblue= np.array([0.02345457,0.01907375,  0.01535466,  0.02599181]) #60-> 30
#        backred=np.array([ 0.02074025,0.02155595,  0.01922453,  0.01917648])
#        
#        redto = tending_function(unumpy.nominal_values(taured), backred)
#        blueto = tending_function(unumpy.nominal_values(taublue), backblue)
        
    def plotinho(ax0, dset,my_color,ax0b,my_edgecolor, my_facecolor):   
        
        movav = 50
        
        if my_color is 'r':
            ax0.text(1000, 165, 'temperature \n increases', fontsize=fsizenb, va='center',ha='center')
            ax0.annotate('', xy=(1000,75), xytext=(1000,150),
                arrowprops=dict(facecolor='black', shrink=0.05))              
        
        ax0.plot(moving_average(np.arange(1,taured.shape[1]+1),movav),moving_average(unumpy.nominal_values(dset[3,:]),movav),color=my_color,ls='--',lw=2)
        ax0.plot(moving_average(np.arange(1,taured.shape[1]+1),movav),moving_average(unumpy.nominal_values(dset[2,:]),movav),color=my_color,ls='--',lw=3)
        ax0.plot(moving_average(np.arange(1,taured.shape[1]+1),movav),moving_average(unumpy.nominal_values(dset[1,:]),movav),color=my_color,ls='--',lw=4)
        ax0.plot(moving_average(np.arange(1,taured.shape[1]+1),movav),moving_average(unumpy.nominal_values(dset[0,:]),movav),color=my_color,ls='--',lw=5)
        
        ax0.fill_between(moving_average(np.arange(1,taured.shape[1]+1),movav),
                                                 moving_average(unumpy.nominal_values(dset[3,:]),movav)-moving_average(unumpy.std_devs(dset[3,:]),movav),
                                                 moving_average(unumpy.nominal_values(dset[3,:]),movav)+moving_average(unumpy.std_devs(dset[3,:]),movav),
                                                 edgecolor=my_edgecolor,
                                                 facecolor=my_facecolor,
                                                 alpha=0.5,
                                                 linewidth=0.0)
                                                 
        ax0.fill_between(moving_average(np.arange(1,taured.shape[1]+1),movav),
                                                 moving_average(unumpy.nominal_values(dset[0,:]),movav)-moving_average(unumpy.std_devs(dset[0,:]),movav),
                                                 moving_average(unumpy.nominal_values(dset[0,:]),movav)+moving_average(unumpy.std_devs(dset[0,:]),movav),
                                                 edgecolor=my_edgecolor,
                                                 facecolor=my_facecolor,
                                                 alpha=0.5,
                                                 linewidth=0.0)
                         
        ax0.plot(moving_average(np.arange(1,taured.shape[1]+1),movav),moving_average(np.arange(1,taured.shape[1]+1)/2,movav),color='k',ls='--',lw=2)        
        
        ax0b.plot(moving_average(np.arange(2,taured.shape[1]+1),movav),moving_average(np.diff(unumpy.nominal_values(dset[3,:])),movav),color=my_color,ls='--',lw=2)
        ax0b.plot(moving_average(np.arange(2,taured.shape[1]+1),movav),moving_average(np.diff(unumpy.nominal_values(dset[2,:])),movav),color=my_color,ls='--',lw=3)
        ax0b.plot(moving_average(np.arange(2,taured.shape[1]+1),movav),moving_average(np.diff(unumpy.nominal_values(dset[1,:])),movav),color=my_color,ls='--',lw=4)
        ax0b.plot(moving_average(np.arange(2,taured.shape[1]+1),movav),moving_average(np.diff(unumpy.nominal_values(dset[0,:])),movav),color=my_color,ls='--',lw=5)
        
        ax0b.axhline(y=0.5, xmin=0, xmax=130, lw=2, color = 'k', ls='--')
        ax0b.set_yticks([0.25,0.5])#,200])#
        ax0b.set_ylim([0,0.55])#
        
        my_x= moving_average(np.arange(1,taured.shape[1]+1),movav)
      
        ax0b.set_xlim(xmin= my_x[0],xmax = my_x[-1])#    
        ax0.set_xlim(xmin= my_x[0],xmax = my_x[-1])       
        
        ax0.set_xticks([500,1000])
        ax0b.set_xticks([500,1000])
        ax0b.set_xticklabels([500,1000])
        ax0.set_xticklabels([])
        ax0b.set_xlabel('Transient cathodoluminescence \n acquisition time ($\mu$s)',fontsize=fsizepl)
        
        ax0.set_yticks([100,200,300])#,200])#
        ax0.set_ylim([0,305])#
        ax0.tick_params(labelsize=fsizenb)
        ax0b.tick_params(labelsize=fsizenb)
        
    plotinho(ax00, taured,'r',ax00b ,my_edgecolor='#ff3232', my_facecolor='#ff6666')
    plotinho(ax100, taublue,'g',ax100b,my_edgecolor='#74C365', my_facecolor='#74C365')
                  
    plt.tight_layout() 
    plt.show()  
    multipage_longer('SI-Tau.pdf',dpi=80)


