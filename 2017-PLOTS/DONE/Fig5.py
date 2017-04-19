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
import matplotlib.cm as cm
import scipy.ndimage as ndimage
from matplotlib.backends.backend_pdf import PdfPages
from MakePdf import *
from matplotlib.pyplot import cm #to plot following colors of rainbow
from matplotlib import rc
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = DeprecationWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.simplefilter(action = "ignore", category = PendingDeprecationWarning)
import matplotlib.cm as cm
import scipy.misc
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
import matplotlib.cm as cm

### settings
fsizepl = 24
fsizenb = 20
mkstry = ['8','11','5'] #marker size for different dsets Med Zoom/Large Zoom/Small Zoom
###

sizex = 8
sizey=6
dpi_no = 80

fig1= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig1.set_size_inches(1200./fig1.dpi,900./fig1.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    

noplots = 2
nolines = 2

ax1 = plt.subplot2grid((nolines,noplots), (0,1), colspan=1, rowspan=1)
ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('right')

ax3 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')

ax4 = plt.subplot2grid((nolines,noplots), (1,0), colspan=1, rowspan=1)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.xaxis.set_ticks_position('bottom')
ax4.yaxis.set_ticks_position('left')

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx #array[idx]

def tauestimate(counts_red, error_red, counts_blue, error_blue):
    
    print(counts_red.shape[1])
    
    ucounts_red = unumpy.uarray(counts_red, error_red)
    ucounts_blue = unumpy.uarray(counts_blue, error_blue)
    
    def helper(arrayx):
         return np.cumsum(arrayx*np.arange(1,counts_red.shape[1]+1), axis = 1)/np.cumsum(arrayx, axis = 1)
    
    return helper(ucounts_red),helper(ucounts_blue)
    
def viscumcounts(counts_red, error_red, counts_blue, error_blue):
    
    print(counts_red.shape[1])
    
    ucounts_red = unumpy.uarray(counts_red, error_red)
    ucounts_blue = unumpy.uarray(counts_blue, error_blue)
    
   # return (np.cumsum(ucounts_red,axis=1)-np.cumsum(ucounts_blue,axis=1))/       \
   #                  (np.cumsum(ucounts_red,axis=1)+np.cumsum(ucounts_blue,axis=1))
    return (np.cumsum(ucounts_blue,axis=1)-np.cumsum(ucounts_red,axis=1))/       \
                     (np.cumsum(ucounts_red,axis=1)+np.cumsum(ucounts_blue,axis=1))
    
if True:
    
    Il_data3 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Il_dataGOINGDOWN.npz')
#    xvec = Il_data3['data']  
    #REPLACE WITH FITTING DATA
    xvec = np.array([29.767038939801189, 32.989664557794143, 41.60740810742837, 50.690450376594001, 59.078831716569162, 71.049952240480138])[::-1]

    Il_data_std3 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Il_data_stdGOINGDOWN.npz')
    xvecstd = Il_data_std3['data']  
    #REPLACE WITH FITTING DATA
    xvecstd = np.array([0.15004144802124503, 0.13120359603133785, 0.07976927921856572, 0.10440306773732771, 0.45758747280836515, 0.74933681763885074])[::-1]

    Red_decay_array3 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Red_decay_arrayGOINGDOWN.npz') 
    red = Red_decay_array3['data']
    Blue_decay_array3 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Blue_decay_arrayGOINGDOWN.npz') 
    blue = Blue_decay_array3['data']
    

    print(xvec.shape)
    print(red.shape)
        
    Notr = np.zeros([6,1398])
    #From file get_no_signal_pixels_only
    No_signal = [90000,90000,90000,90000,90000,90000] # [38921.0,29452.0,29608.0,34650.0,33207.0,37710.0]

    Notr[0,:] = 5.0*No_signal[0] 
    Notr[1,:] = 5.0*No_signal[1] 
    Notr[2,:] = 5.0*No_signal[2] 
    Notr[3,:] = 5.0*No_signal[3] 
    Notr[4,:] = 5.0*No_signal[4] 
    Notr[5,:] = 5.0*No_signal[5] 
     
    (taured,taublue) = tauestimate(red, np.sqrt(red)/np.sqrt(Notr),blue, np.sqrt(blue)/np.sqrt(Notr))
     
    #below is wrong way of using it as Poisson
    #(taured,taublue) = tauestimate(data['red1D'], np.sqrt(data['red1D']),data['blue1D'], np.sqrt(data['blue1D']))
    ts_b = np.arange(0,red.shape[1])
    
    aa = np.empty([taured.shape[1]])    
    cc = np.empty([taured.shape[1]])   
    aaerr = np.empty([taured.shape[1]])    
    ccerr = np.empty([taured.shape[1]])   
    bb = np.empty([taured.shape[1]])    
    dd = np.empty([taured.shape[1]]) 
    for jjj in np.arange(1,taured.shape[1]):
        print(jjj)
        #####NEEDS TO BE FITTED USING POISSON - LOOK FOR OUTPUT "BEWARE DOING POISSON"
        (a,b,result) = linear_fit_with_error((xvec), unumpy.nominal_values(taured[:,jjj]), unumpy.std_devs(taured[:,jjj]), use_poisson = True)
        (c,d,result2) = linear_fit_with_error((xvec) , unumpy.nominal_values(taublue[:,jjj]), unumpy.std_devs(taublue[:,jjj]), use_poisson = True)
        aa[jjj] = a
        cc[jjj] = c
        bb[jjj] = b
        dd[jjj] = d
        aaerr[jjj] = result.params['a'].stderr
        ccerr[jjj] = result2.params['a'].stderr
        
    step = 49 #was 50
    colors = iter(cm.rainbow(np.linspace(0, 1, len(np.arange(0,taured.shape[1],step)))))   
    inittime = 5 #was 1              
    for jjj in np.arange(inittime,taured.shape[1],step):
        colorful = next(colors)
        hlp1= unumpy.nominal_values(taured[:,jjj])
        hlp2= unumpy.nominal_values(taublue[:,jjj])
        #hlpcumu below is VISIB of CUMU of TAUS
        hlpcumu = (np.cumsum(unumpy.nominal_values(taured[:,jjj]))-np.cumsum(unumpy.nominal_values(taublue[:,jjj])))/(np.cumsum(unumpy.nominal_values(taured[:,jjj]))+np.cumsum(unumpy.nominal_values(taublue[:,jjj])))
        
        ax1.plot(xvec,hlp1 ,marker='o',ls='None',color=colorful,markersize=8)
        ax1.plot(xvec,aa[jjj]*(xvec) + bb[jjj],ls='-',color=colorful,markersize=8)   
      
        if jjj < 165:
            ax3.plot(xvec,hlp2 ,marker='o',ls='None',color=colorful,markersize=8)
            ax3.plot(xvec,cc[jjj]*(xvec) + dd[jjj],ls='-',color=colorful,markersize=8)
        
        if jjj == inittime:
            ax3.text(60, 85, '   ' + 'time in\n 50 $\mu$s intervals', fontsize=fsizenb, va='center',ha='center')
            ax3.annotate('', xy=(60,175), xytext=(60,110),
                arrowprops=dict(facecolor='black', shrink=0.05))   
            ax3.text(-0.18, 1.0, 'a', transform=ax3.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})
        
        #'Time in 50 $\mu$s \n    intervals
        ax1.set_ylabel(r'Red band $\tau$ ($\mu$s)',fontsize=fsizepl)
        ax1.set_xlabel('Temperature at sample ($^{\circ}$C)',fontsize=fsizepl)
        ax1.tick_params(labelsize=fsizenb)
        #ax1.set_ylim([0,305])
        ax1.set_xlim([25, 75])
        ax1.set_xticks([30,40,50,60,70]) 
        ax1.set_yticks([100,200,300])
        ax1.yaxis.set_label_position("right")
        
        ax3.set_ylabel(r'Green band $\tau$ ($\mu$s)',fontsize=fsizepl)
       
        #ax3.set_ylim([0,305])
        ax3.set_xlim([25,75])
        ax3.set_yticks([100,200,300])
        ax3.set_xticks([30,40,50,60,60,70])
        ax3.tick_params(labelsize=fsizenb)
        ax3.set_xlabel('Temperature at sample ($^{\circ}$C)',fontsize=fsizepl)
        
        ax1.set_ylim([-1,375])
        ax3.set_ylim([-1,375])
       
    #labels
    xmin = 0
    xmax = 1390
    indicetonorm = 5
    starrtred = 25
    starrtgreen = 15
    ax4.plot(np.arange(1+starrtred,taured.shape[1]+1),np.abs(100.0*aa[starrtred:]/unumpy.nominal_values(taured[indicetonorm,starrtred:])),ls='--',color='r',lw=2)
    ax4.plot(np.arange(1+starrtgreen,taured.shape[1]+1),np.abs(100.0*cc[starrtgreen:]/unumpy.nominal_values(taublue[indicetonorm,starrtgreen:])),ls='--',color='g',lw=2)
    ax4.text(-0.18, 1.0, 'b', transform=ax4.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})
    
    ax4.set_ylabel('Abs. of norm. slope \n' +  r'$\vert\bar{\alpha}\vert$ $\equiv$ $\vert\partial\tau$/$\partial$T$\vert$/$\tau_{\sim\,25\,^{\circ}C}$' + ' ($\%$ $^{\circ}$C$^{-1}$)',fontsize=fsizepl)
    ax4.set_xlabel('Transient cathodoluminescence \n acquisition time ($\mu$s)',fontsize=fsizepl)
    ax4.tick_params(labelsize=fsizenb)
    ax4.set_xticks([500,1000])#,1500])
    #ax4.set_yticks([0.25, -0.25, -0.75, -1.25])
    annotated = 'previously reported\n (fluorescence lifetime)'
    ax4.plot(1000,0.54, marker='o',color='g', markeredgewidth=0.0, markersize = 8)
    ax4.annotate(annotated, xy=(1000, 0.53), xytext=(1000,0.4), fontsize=fsizenb,
            arrowprops=dict(facecolor='k', edgecolor='None', shrink=0.05), va='center',ha='center')
            
    ####FIND MINIMUM
    minslopered = np.max(np.abs(100.0*aa/unumpy.nominal_values(taured[indicetonorm,:])))
    timeminslopered = find_nearest(np.abs(100.0*aa/unumpy.nominal_values(taured[indicetonorm,:])),minslopered)
    minslopegreen = np.max(np.abs(100.0*cc/unumpy.nominal_values(taublue[indicetonorm,:])))
    timeminslopegreen = find_nearest(np.abs(100.0*cc/unumpy.nominal_values(taublue[indicetonorm,:])),minslopegreen)
  
#    ax4.vlines(timeminslopered,ymin=-1, ymax=minslopered, linestyle='dashed',color='r',lw=2,zorder=1)
#    ax4.vlines(timeminslopegreen,ymin=-1, ymax=minslopegreen,linestyle='dashed',color='g',lw=2,zorder=1)
    ax4.set_xticks([500,1000]) #,timeminslopered,timeminslopegreen])
    ax4.plot(timeminslopered,minslopered, marker='o',color='r', markeredgewidth=0.0, markersize = 8)
    ax4.plot(timeminslopegreen,minslopegreen, marker='o',color='g', markeredgewidth=0.0, markersize = 8)
    ax4.hlines(minslopered,xmax=timeminslopered, xmin=15, linestyle='solid',color='r',lw=2,zorder=1)
    ax4.hlines(minslopegreen,xmax=timeminslopegreen, xmin=15, linestyle='solid',color='g',lw=2,zorder=1)
    ax4.hlines(0.54,xmax=1000, xmin=0, linestyle='solid',color='g',lw=2,zorder=1)
    ax4.set_ylim([0,1.02])
    ax4.set_yticks([0.5, 1])
    ax4.set_yticklabels(['0.5','1'])
    ax4.set_xlim([15,1400])
        
    sys.path.append("../2017-01-28_Gradient_Sensitivity/") # necessary for the tex fonts
    from calc_sens import get_sens  
    (xxx, eta_taured)  = get_sens(taured, xvec,np.arange(1,taured.shape[1]+1), use_poisson = True)
    (xxx, eta_taublue)  = get_sens(taublue, xvec,np.arange(1,taured.shape[1]+1), use_poisson = True)
    
    ##### SENSITIVITY OF OTHER QUANTITIES
#    (xxx, eta_visib) = get_sens((taublue-taured)/(taured+taublue), xvec,np.arange(1,taured.shape[1]+1))
#    (xxx, eta_ratio) = get_sens((taublue)/(taured), xvec,np.arange(1,taured.shape[1]+1), use_poisson = True)
#    print('do viscumcounts')
#    (xxx, eta_viscumcounts) = get_sens(viscumcounts(red, np.sqrt(red)/np.sqrt(Notr),blue, np.sqrt(blue)/np.sqrt(Notr)), xvec,np.arange(1,taured.shape[1]+1))
#           
    ax0 = plt.subplot2grid((nolines,noplots), (1,1), colspan=1, rowspan=1)
    ax0.semilogy(np.arange(1+starrtred,taured.shape[1]+1),eta_taured[starrtred:] ,color='r',ls='--',lw=2)
    ax0.semilogy(np.arange(1+starrtgreen,taured.shape[1]+1),eta_taublue[starrtgreen:] ,color='g',ls='--',lw=2)
    ax0.text(1.15, 1.0, 'c', transform=ax0.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})
    ax0.spines['left'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.xaxis.set_ticks_position('bottom')
    ax0.yaxis.set_ticks_position('right')
    ax0.yaxis.set_label_position("right")
    ax0.set_xticks([500,1000])#,1500])
    #ax0.set_ylim((0.0065,0.15))
    ax0.set_ylabel(r'Sensitivity $\delta$T $\equiv$ $\sigma_{\tau}$/$\vert\partial \tau$/$\partial$T$\vert$ ($^{\circ}$C)',fontsize=fsizepl)
    ax0.set_xlabel('Transient cathodoluminescence \n acquisition time ($\mu$s)',fontsize=fsizepl)
    #ax0.set_yticks([0.01,0.1])
    #ax0.set_yticklabels(['0.01','0.1'])
    ax0.set_xlim([xmin, xmax])
    ax0.tick_params(labelsize=fsizenb) 
    startpt = 10
#    ax0.semilogy(np.arange(1+startpt,taured.shape[1]+1),eta_ratio[startpt:] ,ls='--',lw=2,color='k',label=r'Ratio of $\tau$')
#    ax0.semilogy(np.arange(1+startpt,taured.shape[1]+1),eta_visib[startpt:] ,ls='dotted',lw=2,color='k',label=r'Visibility of $\tau$')
#    ax0.semilogy(np.arange(1+startpt,taured.shape[1]+1),eta_viscumcounts[startpt:] ,color='k',ls='-.',lw=2,label=r'Visibility of cumul. cts.')
    
#    stepchen = 100
#    ax0.semilogy(np.arange(1+startpt,taured.shape[1]+1,stepchen),eta_ratio[startpt::stepchen] ,marker='d',ls='None',label=r'Ratio of $\tau$', markersize=12, markeredgecolor='None', color='k')
#    ax0.semilogy(np.arange(1+startpt,taured.shape[1]+1,stepchen),eta_visib[startpt::stepchen] ,marker='o',ls='None',label=r'Visibility of $\tau$', markersize=12, markeredgecolor='None', color='k')
#    ax0.semilogy(np.arange(1+startpt,taured.shape[1]+1,stepchen),eta_viscumcounts[startpt::stepchen] ,marker='h',color='k',ls='None',label=r'Visibility of cumul. cts.', markersize=12, markeredgecolor='None')

#    ax0.legend(loc='best',numpoints=1,frameon=False, fontsize=fsizenb)
    ####FIND MINIMUM
    minslopered = np.min(eta_taured[starrtred:])
    timeminslopered = find_nearest(eta_taured[starrtred:],minslopered)
    minslopegreen = np.min(eta_taublue[starrtgreen:])
    timeminslopegreen = find_nearest(eta_taublue[starrtgreen:],minslopegreen)
    
#    print('time red') #908
#    print(timeminslopered)
#    print('time green') #211
#    print(timeminslopegreen)
  
    #ax0.vlines(timeminslopered,ymin=0.005, ymax=minslopered, linestyle='dashed',color='r',lw=2,zorder=1)
    #ax0.vlines(timeminslopegreen,ymin=0.005, ymax=minslopegreen,linestyle='dashed',color='g',lw=2,zorder=1)
    ax0.set_xticks([500,1000]) #,timeminslopered,timeminslopegreen])
    ax0.plot(timeminslopered,minslopered, marker='o',color='r', markeredgewidth=0.0, markersize = 8)
    ax0.plot(timeminslopegreen,minslopegreen, marker='o',color='g', markeredgewidth=0.0, markersize = 8)
    ax0.hlines(minslopered,xmin=timeminslopered, xmax=1400, linestyle='solid',color='r',lw=2,zorder=1)
    ax0.hlines(minslopegreen,xmin=timeminslopegreen, xmax=1400, linestyle='solid',color='g',lw=2,zorder=1)
    #ax0.set_ylim([0.009, 1.09])
    print(minslopered)
    ax0.set_yticks([0.01, 0.1, 1]) #, minslopered,minslopegreen])
    ax0.set_yticklabels(['0.01','0.1','1']) #,'0.021','0.048'])
    ax0.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax0.set_ylim([0.0095, 1.05])

plt.tight_layout()
multipage_longer('Fig5.pdf',dpi=80)

#@@###$$@#@#$@$@#$@$@$@
lklklklkklhere

###### new plot
#plt.close()
#fig10= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
#fig10.set_size_inches(1200./fig10.dpi,900./fig10.dpi)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.rc('font', serif='Palatino')    
#
#noplots = 2
#nolines = 2
#
#ax0 = plt.subplot2grid((nolines,noplots), (0,1), colspan=1, rowspan=1)
#ax0.spines['right'].set_visible(False)
#ax0.spines['top'].set_visible(False)
#ax0.xaxis.set_ticks_position('bottom')
#ax0.yaxis.set_ticks_position('left')
#
#ax4 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)
#ax4.spines['right'].set_visible(False)
#ax4.spines['top'].set_visible(False)
#ax4.xaxis.set_ticks_position('bottom')
#ax4.yaxis.set_ticks_position('left')
#ax4.plot(np.arange(1,taured.shape[1]+1),100.0*aa/unumpy.nominal_values(taured[3,:]),ls='--',color='r',lw=2)
#ax4.plot(np.arange(1,taured.shape[1]+1),100.0*cc/unumpy.nominal_values(taublue[3,:]),ls='--',color='g',lw=2)
#ax4.text(-0.25, 1.0, 'a', transform=ax4.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})
#ax4.set_ylabel('Norm. slope \n' +  r'$\bar{\alpha}$ $\equiv$ ($\partial\tau$/$\partial$T)/$\tau_{\sim 25\,^{\circ}C}$' + ' ($\%$ $^{\circ}$C$^{-1}$)',fontsize=fsizepl)
#ax4.set_xlabel('Transient cathodoluminescence \n acquisition time ($\mu$s)',fontsize=fsizepl)
#ax4.tick_params(labelsize=fsizenb)
#ax4.set_xticks([500,1000])#,1500])
##ax4.set_ylim((-4.2,0.5))
#ax4.set_yticks([0.25, -0.25, -0.75, -1.25])
#
#ax0.semilogy(np.arange(1+1,taured.shape[1]+1),eta_taured[1:] ,color='r',ls='--',lw=2)
#ax0.semilogy(np.arange(1+1,taured.shape[1]+1),eta_taublue[1:] ,color='g',ls='--',lw=2)
#ax0.text(-0.25, 1.0, 'b', transform=ax0.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})
#ax0.spines['left'].set_visible(False)
#ax0.spines['top'].set_visible(False)
#ax0.spines['right'].set_visible(True)
#ax0.xaxis.set_ticks_position('bottom')
#ax0.yaxis.set_ticks_position('right')
#ax0.yaxis.set_label_position("right")
#ax0.set_xticks([500,1000])#,1500])
#ax0.set_ylim((0.0065,150))
#ax0.set_ylabel(r'Sensitivity $\delta$T $\equiv$ $\sigma_{\tau}$/$\vert\partial\tau$/$\partial$T$\vert$ ($^{\circ}$C)',fontsize=fsizepl)
#ax0.set_xlabel('Transient cathodoluminescence \n acquisition time ($\mu$s)',fontsize=fsizepl)
#ax0.set_yticks([0.01,0.1,1,10,100])
#ax0.set_yticklabels(['0.01','0.1','1','10','100'])
#ax0.set_xlim([xmin, xmax])
#ax0.tick_params(labelsize=fsizenb) 
#
#finda = eta_taured[0:]
#findc = eta_taublue[0:] 
#ax4.axvline(np.argmax(finda)+1,color='k',lw=1)
#ax4.axvline(np.argmax(findc)+1,color='k',lw=1)
#ax0.axvline(np.argmax(finda)+1,color='k',lw=1)
#ax0.axvline(np.argmax(findc)+1,color='k',lw=1)
#
#plt.tight_layout()
#multipage_longer('Fig5AllEta.pdf',dpi=80)
#
#klklk
