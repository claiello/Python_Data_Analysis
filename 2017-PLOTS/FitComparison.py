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
from FluoDecay import *
import skimage
from skimage import exposure
from my_fits import *

import pickle
import my_fits
from uncertainties import unumpy
from numpy import genfromtxt
from my_fits import *

### settings
fsizepl = 24
fsizenb = 20
mkstry = ['8','11','5'] #marker size for different dsets Med Zoom/Large Zoom/Small Zoom
###

sizex = 8
sizey=6
dpi_no = 80

def tauestimate(counts_red, error_red, counts_blue, error_blue):
    
    print(counts_red.shape[0])
    
    ucounts_red = unumpy.uarray(counts_red, error_red)
    ucounts_blue = unumpy.uarray(counts_blue, error_blue)
    
    def helper(arrayx):
         return np.cumsum(arrayx*np.arange(1,counts_red.shape[0]+1), axis = 0)/np.cumsum(arrayx, axis = 0)
    
    return helper(ucounts_red),helper(ucounts_blue)

d_temp = pickle.load( open( "d4.p", "rb" ) ) #4#

fig1= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig1.set_size_inches(1200./fig1.dpi,900./fig1.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    

ax1 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax0 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
ax0.spines['left'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.xaxis.set_ticks_position('bottom')
ax0.yaxis.set_ticks_position('right')

axcompb = plt.subplot2grid((2,2), (1,0), colspan=1, rowspan=1)
axcompb.spines['right'].set_visible(False)
axcompb.spines['top'].set_visible(False)
axcompb.xaxis.set_ticks_position('bottom')
axcompb.yaxis.set_ticks_position('left')

axcomp = plt.subplot2grid((2,2), (1,1), colspan=1, rowspan=1)
axcomp.spines['left'].set_visible(False)
axcomp.spines['top'].set_visible(False)
axcomp.xaxis.set_ticks_position('bottom')
axcomp.yaxis.set_ticks_position('right')



import matplotlib.cm as cm
ct = 3
for data in [d_temp]: #[d_ap, d_kv, d_pixel, d_temp]:
    
    red = data['red1D'][3,:]
    blue = data['blue1D'][3,:]
    
    redstd = np.sqrt(red)
    bluestd = np.sqrt(blue)
    
    print(np.sum(np.where(np.isnan(red))))
    print(np.sum(np.where(np.isnan(blue))))
    minvalue = 1.0e-12
    redstd[np.where(redstd < minvalue)] = minvalue
    bluestd[np.where(bluestd < minvalue)] = minvalue
    red[np.where(red < minvalue)] = minvalue
    blue[np.where(blue < minvalue)] = minvalue
    
    ts_b = np.arange(0,red.shape[0])
    
    initi = 3
    doPoisson = True
    y_err = None #redstd
    y_errb = None #bluestd
    # Triple exp
    print('tripleexp')
    (result3) = choose_poly_fit(ts_b[initi:], np.cumsum(red, axis = 0)[initi:],'tripleexp',y_err,doPoisson)
    (result3b) = choose_poly_fit(ts_b[initi:], np.cumsum(blue, axis = 0)[initi:],'tripleexp',y_errb,doPoisson)
    #Double exp
    print(result3)
    print('doubleexp')
    (result2) = choose_poly_fit(ts_b[initi:], np.cumsum(red, axis = 0)[initi:],'doubleexp',y_err,doPoisson)
    (result2b) = choose_poly_fit(ts_b[initi:], np.cumsum(blue, axis = 0)[initi:],'doubleexp',y_errb,doPoisson)
    #Single exp
    print('singleexp')
    (result1) = choose_poly_fit(ts_b[initi:], np.cumsum(red, axis = 0)[initi:],'singleexp',y_err,doPoisson)
    (result1b) = choose_poly_fit(ts_b[initi:], np.cumsum(blue, axis = 0)[initi:],'singleexp',y_errb,doPoisson)
  
    print('weirdexp')
    (resultw) = choose_poly_fit(ts_b[initi:], np.cumsum(red, axis = 0)[initi:],'weirdexp',y_err,doPoisson)
    (resultwb) = choose_poly_fit(ts_b[initi:], np.cumsum(blue, axis = 0)[initi:],'weirdexp',y_errb,doPoisson)
#    
    ax0.loglog(ts_b, np.cumsum(red, axis = 0)/np.max(np.cumsum(red, axis = 0)), 'o', color='r', markeredgewidth=0.0,markersize=8)
    ax0.loglog(ts_b, choose_func_poly(ts_b, result1.params,'singleexp')/np.max(np.cumsum(red, axis = 0)), color='k',lw=2,label='Mono exp.')
    ax0.loglog(ts_b, choose_func_poly(ts_b, result2.params,'doubleexp')/np.max(np.cumsum(red, axis = 0)), color='b',lw=2,label='Double exp.')
    ax0.loglog(ts_b, choose_func_poly(ts_b, result3.params,'tripleexp')/np.max(np.cumsum(red, axis = 0)), color='m',lw=2,label='Triple exp.')
    ax0.loglog(ts_b, choose_func_poly(ts_b, resultw.params,'weirdexp')/np.max(np.cumsum(red, axis = 0)), color='c',lw=2,label='I--H model')
    
    ax0.loglog(ts_b,  (np.cumsum(red, axis = 0)-choose_func_poly(ts_b, result3.params,'tripleexp'))/np.max(np.cumsum(red, axis = 0)), color='m',ls='-.')
    ax0.loglog(ts_b,  (np.cumsum(red, axis = 0)-choose_func_poly(ts_b, result2.params,'doubleexp'))/np.max(np.cumsum(red, axis = 0)), color='b',ls='-.')
    ax0.loglog(ts_b,  (np.cumsum(red, axis = 0)-choose_func_poly(ts_b, result1.params,'singleexp'))/np.max(np.cumsum(red, axis = 0)), color='k',ls='-.')
    ax0.loglog(ts_b,  (np.cumsum(red, axis = 0)-choose_func_poly(ts_b, resultw.params,'weirdexp'))/np.max(np.cumsum(red, axis = 0)), color='c',ls='-.')
    
    ax1.loglog(ts_b, np.cumsum(blue, axis = 0)/np.max(np.cumsum(blue, axis = 0)), 'o', color='g', markeredgewidth=0.0,markersize=8)
    ax1.loglog(ts_b, choose_func_poly(ts_b, result3b.params,'tripleexp')/np.max(np.cumsum(blue, axis = 0)), color='m',lw=2)
    ax1.loglog(ts_b, choose_func_poly(ts_b, result2b.params,'doubleexp')/np.max(np.cumsum(blue, axis = 0)), color='b',lw=2)
    ax1.loglog(ts_b, choose_func_poly(ts_b, result1b.params,'singleexp')/np.max(np.cumsum(blue, axis = 0)), color='k',lw=2)
    ax1.loglog(ts_b, choose_func_poly(ts_b, resultwb.params,'weirdexp')/np.max(np.cumsum(blue, axis = 0)), color='c',lw=2)
    
    ax1.loglog(ts_b, (np.cumsum(blue, axis = 0)-choose_func_poly(ts_b, result3b.params,'tripleexp'))/np.max(np.cumsum(blue, axis = 0)), color='m',ls='-.')
    ax1.loglog(ts_b, (np.cumsum(blue, axis = 0)-choose_func_poly(ts_b, result2b.params,'doubleexp'))/np.max(np.cumsum(blue, axis = 0)), color='b',ls='-.')
    ax1.loglog(ts_b, (np.cumsum(blue, axis = 0)-choose_func_poly(ts_b, result1b.params,'singleexp'))/np.max(np.cumsum(blue, axis = 0)), color='k',ls='-.')
    ax1.loglog(ts_b, (np.cumsum(blue, axis = 0)-choose_func_poly(ts_b, resultwb.params,'weirdexp'))/np.max(np.cumsum(blue, axis = 0)), color='c',ls='-.')
          
    ax0.set_ylim([0.901e-3,1.5])  
    ax1.set_ylim([0.901e-3,1.5])  
    ax0.set_yticks([1.0e-3,1.0e-2,0.1,1])
    ax1.set_yticks([1.0e-3,1.0e-2,0.1,1])
    ax0.set_yticklabels(['0.001','0.01','0.1','1'])
    ax1.set_yticklabels(['0.001','0.01','0.1','1'])
    ax0.tick_params(labelsize=fsizenb)
    ax1.tick_params(labelsize=fsizenb)
    ax0.set_xlabel('Transient cathodoluminescence \n acquisition time ($\mu$s)',fontsize=fsizepl)
    ax0.set_xticks([500,1000])#,1500])
    ax1.set_xlabel('Transient cathodoluminescence \n acquisition time ($\mu$s)',fontsize=fsizepl)
    ax1.set_xticks([500,1000])#,1500])
    ax0.set_xlim([0.901,1498])
    ax1.set_xlim([0.901,1498]) 
    ax0.set_xticks([10,100,1000])
    ax0.set_xticklabels([10,100,1000])
    ax1.set_xticks([10,100,1000])
    ax1.set_xticklabels([10,100,1000])
    ax0.set_ylabel('Norm. cumul. red band intensity \n' + r'$\int_{0}^{t}R(t^{\prime})dt^{\prime}$ (a.u.)',fontsize=fsizepl)
    ax1.set_ylabel('Norm. cumul. green band intensity \n' + r'$\int_{0}^{t}G(t^{\prime})dt^{\prime}$ (a.u.)',fontsize=fsizepl)
    ax0.yaxis.set_label_position("right")
    
    ax0.legend(bbox_to_anchor=(0.625, 0.95),
           bbox_transform=plt.gcf().transFigure,fontsize=fsizenb,frameon=False)
       
    print('red')  
    print_result(result3)  
    print_result(result2)
    print('blue')  
    print_result(result3b)  
    print_result(result2b)
    
    #null hypothesis == restricted model is correct
    #Reject: if F > qf(.95, dfN, dfF )
      
    # Try f-test RED
    df1 = 1398 - 3 #single exponential, simpler model
    df2 = 1398 - 5 #double exponential
    #2*(data*np.log(data) - data*np.log(model) - (data-model))
    SS1 = np.sum(2*(choose_func_poly(ts_b, result1.params,'singleexp')[initi:]*np.log(choose_func_poly(ts_b, result1.params,'singleexp')[initi:]) - choose_func_poly(ts_b, result1.params,'singleexp')[initi:]*np.log(np.cumsum(red, axis = 0)[initi:]) - (choose_func_poly(ts_b, result1.params,'singleexp')[initi:]-np.cumsum(red, axis = 0)[initi:])))
    SS2 = np.sum(2*(choose_func_poly(ts_b, result2.params,'doubleexp')[initi:]*np.log(choose_func_poly(ts_b, result2.params,'doubleexp')[initi:]) - choose_func_poly(ts_b, result2.params,'doubleexp')[initi:]*np.log(np.cumsum(red, axis = 0)[initi:]) - (choose_func_poly(ts_b, result2.params,'doubleexp')[initi:]-np.cumsum(red, axis = 0)[initi:])))
    #SS1 = np.sum( (np.cumsum(red, axis = 0)[initi:]-choose_func_poly(ts_b, result1.params,'singleexp')[initi:])**2 )
    #SS2 = np.sum( (np.cumsum(red, axis = 0)[initi:]-choose_func_poly(ts_b, result2.params,'doubleexp')[initi:])**2 )
    print('F test 2 vs 1, RED')       
    print(((SS1 - SS2)/(df1-df2))/(SS2/df2))  
    #The null hypothesis (that model 1 is simpler) is rejected (ie model 2 better) if the F-statistic calculated from the
    #data is greater than the critical value of the F-distribution for some desired false-rejection probability.
    
     # Try f-test BLUE
    df1 = 1398 - 3 #single exponential, simpler model
    df2 = 1398 - 5 #double exponential
    #2*(data*np.log(data) - data*np.log(model) - (data-model))
    SS1 = np.sum(2*(choose_func_poly(ts_b, result1b.params,'singleexp')[initi:]*np.log(choose_func_poly(ts_b, result1b.params,'singleexp')[initi:]) - choose_func_poly(ts_b, result1b.params,'singleexp')[initi:]*np.log(np.cumsum(blue, axis = 0)[initi:]) - (choose_func_poly(ts_b, result1b.params,'singleexp')[initi:]-np.cumsum(blue, axis = 0)[initi:])))
    SS2 = np.sum(2*(choose_func_poly(ts_b, result2b.params,'doubleexp')[initi:]*np.log(choose_func_poly(ts_b, result2b.params,'doubleexp')[initi:]) - choose_func_poly(ts_b, result2b.params,'doubleexp')[initi:]*np.log(np.cumsum(blue, axis = 0)[initi:]) - (choose_func_poly(ts_b, result2b.params,'doubleexp')[initi:]-np.cumsum(blue, axis = 0)[initi:])))
    #SS1 = np.sum( (np.cumsum(red, axis = 0)-choose_func_poly(ts_b, result1.params,'singleexp'))**2 )
    #SS2 = np.sum( (np.cumsum(red, axis = 0)-choose_func_poly(ts_b, result2.params,'doubleexp'))**2 )
    print('F test 2 vs 1, BLUE')      
    print(((SS1 - SS2)/(df1-df2))/(SS2/df2))  
    #The null hypothesis (that model 1 is simpler) is rejected (ie model 2 better) if the F-statistic calculated from the
    #data is greater than the critical value of the F-distribution for some desired false-rejection probability.
      
    # Try f-test RED 
    df1 = 1398 - 5 #doubleexponential, simpler model
    df2 = 1398 - 7 #triple exponential
    #2*(data*np.log(data) - data*np.log(model) - (data-model))
    SS1 = np.sum(2*(choose_func_poly(ts_b, result2.params,'doubleexp')[initi:]*np.log(choose_func_poly(ts_b, result2.params,'doubleexp')[initi:]) - choose_func_poly(ts_b, result2.params,'doubleexp')[initi:]*np.log(np.cumsum(red, axis = 0)[initi:]) - (choose_func_poly(ts_b, result2.params,'doubleexp')[initi:]-np.cumsum(red, axis = 0)[initi:])))
    SS2 = np.sum(2*(choose_func_poly(ts_b, result3.params,'tripleexp')[initi:]*np.log(choose_func_poly(ts_b, result3.params,'tripleexp')[initi:]) - choose_func_poly(ts_b, result3.params,'tripleexp')[initi:]*np.log(np.cumsum(red, axis = 0)[initi:]) - (choose_func_poly(ts_b, result3.params,'tripleexp')[initi:]-np.cumsum(red, axis = 0)[initi:])))
    print('F test 3 vs 2, RED')    
    print(((SS1 - SS2)/(df1-df2))/(SS2/df2))  
    #The null hypothesis (that model 1 is simpler) is rejected (ie model 2 better) if the F-statistic calculated from the
    #data is greater than the critical value of the F-distribution for some desired false-rejection probability.
          
    # Try f-test BLUE
    df1 = 1398 - 5 #doubleexponential, simpler model
    df2 = 1398 - 7 #triple exponential
    #2*(data*np.log(data) - data*np.log(model) - (data-model))
    SS1 = np.sum(2*(choose_func_poly(ts_b, result2b.params,'doubleexp')[initi:]*np.log(choose_func_poly(ts_b, result2b.params,'doubleexp')[initi:]) - choose_func_poly(ts_b, result2b.params,'doubleexp')[initi:]*np.log(np.cumsum(blue, axis = 0)[initi:]) - (choose_func_poly(ts_b, result2b.params,'doubleexp')[initi:]-np.cumsum(blue, axis = 0)[initi:])))
    SS2 = np.sum(2*(choose_func_poly(ts_b, result3b.params,'tripleexp')[initi:]*np.log(choose_func_poly(ts_b, result3b.params,'tripleexp')[initi:]) - choose_func_poly(ts_b, result3b.params,'tripleexp')[initi:]*np.log(np.cumsum(blue, axis = 0)[initi:]) - (choose_func_poly(ts_b, result3b.params,'tripleexp')[initi:]-np.cumsum(blue, axis = 0)[initi:])))
    print('F test 3 vs 2, BLUE')    
    print(((SS1 - SS2)/(df1-df2))/(SS2/df2))  
    #The null hypothesis (that model 1 is simpler) is rejected (ie model 2 better) if the F-statistic calculated from the
    #data is greater than the critical value of the F-distribution for some desired false-rejection probability.
         
      
    x = [1,2,3,4]
    chisqr = [result1.chisqr, result2.chisqr, result3.chisqr, resultw.chisqr]
    chisqb = [result1b.chisqr, result2b.chisqr, result3b.chisqr, resultwb.chisqr]
    axcomp.semilogy(x[0:4], chisqr[0:4],'r', marker='o', ls='None', markeredgewidth=0.0,markersize=8)
    axcompb.semilogy(x[0:4], chisqb[0:4],'g', marker='o', ls='None', markeredgewidth=0.0,markersize=8)
    
#    import statsmodels.api as sm
#    bicr = [0,0,0,0] #[1,2,3,4]
#    res = sm.OLS(choose_func_poly(ts_b, result3.params,'tripleexp'), np.cumsum(red, axis = 0)).fit()
#    bicr[2] = res.aic
#    res = sm.OLS(choose_func_poly(ts_b, result2.params,'doubleexp'), np.cumsum(red, axis = 0)).fit()
#    bicr[1] = res.aic
#    res = sm.OLS(choose_func_poly(ts_b, result1.params,'singleexp'), np.cumsum(red, axis = 0)).fit()
#    bicr[0] = res.aic
#    res = sm.OLS(choose_func_poly(ts_b, resultw.params,'weirdexp'), np.cumsum(red, axis = 0)).fit()
#    bicr[3] = res.aic
    bicr = np.array([np.nan, 11864, 7306, np.nan])
    #axcomp2 = axcomp.twinx()
   # axcomp2.plot(x[0:4],bicr[0:4]/1000.0,'r', marker='s', ls='None', markeredgewidth=0.0,markersize=8)
    axcomp.set_xlim([0,5])
    axcomp.set_xticks([1,2,3,4])
    axcomp.set_xticklabels(['Mono \n exp.','Double \n exp.','Triple \n exp.','I--H \n model'])
    axcomp.tick_params(labelsize=fsizenb)
    axcompb.tick_params(labelsize=fsizenb)
    axcomp.set_ylabel(r'$\chi^2$',fontsize=fsizepl)
    axcompb.set_ylabel(r'$\chi^2$',fontsize=fsizepl)
    
#    bicb = [0,0,0,0] #[1,2,3,4]
#    res = sm.OLS(choose_func_poly(ts_b, result3b.params,'tripleexp'), np.cumsum(blue, axis = 0)).fit()
#    bicb[2] = res.aic
#    res = sm.OLS(choose_func_poly(ts_b, result2b.params,'doubleexp'), np.cumsum(blue, axis = 0)).fit()
#    bicb[1] = res.aic
#    res = sm.OLS(choose_func_poly(ts_b, result1b.params,'singleexp'), np.cumsum(blue, axis = 0)).fit()
#    bicb[0] = res.aic
#    res = sm.OLS(choose_func_poly(ts_b, resultwb.params,'weirdexp'), np.cumsum(blue, axis = 0)).fit()
#    bicb[3] = res.aic
    bicb = np.array([np.nan, 12051, 3836, np.nan])
   # axcompb2 = axcompb.twinx()
   # axcompb2.plot(x[0:4],bicb[0:4]/1000.0,'g', marker='s', ls='None', markeredgewidth=0.0,markersize=8)
    axcompb.set_xlim([0,5])
    axcompb.set_xticks([1,2,3,4])
    axcompb.set_xticklabels(['Mono \n exp.','Double \n exp.','Triple \n exp.','I--H \n model'])
  #  axcomp2.set_ylabel('F score (10$^{-3}$ a.u.)',fontsize=fsizepl)
  #  axcompb2.set_ylabel('F score (10$^{-3}$ a.u.)',fontsize=fsizepl)
    
  #  axcomp2.tick_params(labelsize=fsizenb)
  #  axcompb2.tick_params(labelsize=fsizenb)
  #  axcompb2.spines['top'].set_visible(False)
 #   axcomp2.spines['top'].set_visible(False)
    axcomp.yaxis.set_label_position("right")
    axcomp.set_ylim([0.901*1e-3,290])
    axcompb.set_ylim([0.901*1e-3,290])
    
   
    
#    axcomp2.set_ylim([0,12.9])
#    axcompb2.set_ylim([0,12.9])
#    axcomp2.set_yticks([5,10])
#    axcompb2.set_yticks([5,10])
    
    
    
    chisqr = [result1.chisqr, result2.chisqr, result3.chisqr, resultw.chisqr]
    chisqb = [result1b.chisqr, result2b.chisqr, result3b.chisqr, resultwb.chisqr]
    axcomp.semilogy(x[0], chisqr[0],'k', marker='o', ls='None', markeredgewidth=0.0,markersize=12)
    axcompb.semilogy(x[0], chisqb[0],'k', marker='o', ls='None', markeredgewidth=0.0,markersize=12)
    axcomp.semilogy(x[1], chisqr[1],'b', marker='o', ls='None', markeredgewidth=0.0,markersize=12)
    axcompb.semilogy(x[1], chisqb[1],'b', marker='o', ls='None', markeredgewidth=0.0,markersize=12)
    axcomp.semilogy(x[2], chisqr[2],'m', marker='o', ls='None', markeredgewidth=0.0,markersize=12)
    axcompb.semilogy(x[2], chisqb[2],'m', marker='o', ls='None', markeredgewidth=0.0,markersize=12)
    axcomp.semilogy(x[3], chisqr[3],'c', marker='o', ls='None', markeredgewidth=0.0,markersize=12)
    axcompb.semilogy(x[3], chisqb[3],'c', marker='o', ls='None', markeredgewidth=0.0,markersize=12)
  
    
    axcompb.text(2, 0.135, r'\hspace*{1cm} F score $\sim$ 12 $\cdot$ 10$^{3}$', fontsize=fsizenb,va='center',ha='center')
    axcompb.text(3, 0.00225, 'F score $\sim$ 7 $\cdot$ 10$^{3}$', fontsize=fsizenb,va='center',ha='center')
    axcomp.text(2, 1, 'F score $\sim$ 12 $\cdot$ 10$^{3}$', fontsize=fsizenb,va='center',ha='center')
    axcomp.text(3, 0.004, 'F score $\sim$ 4 $\cdot$ 10$^{3}$', fontsize=fsizenb,va='center',ha='center')
    
    axcomp.set_yticks([0.001,0.01, 0.1, 1,10,100])
    axcomp.set_yticklabels(['0.001','0.01','0.1', '1','10','100'])
    axcompb.set_yticks([0.001,0.01, 0.1, 1,10,100])
    axcompb.set_yticklabels(['0.001','0.01','0.1', '1','10','100'])
    
    ax1.text(-0.1, 1.0, 'a', transform=ax1.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})
    axcompb.text(-0.1, 1.0, 'b', transform=axcompb.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})

   
    plt.show()
    lklklk
  
plt.tight_layout()
multipage_longer('FitComparison.pdf',dpi=80)

klklk