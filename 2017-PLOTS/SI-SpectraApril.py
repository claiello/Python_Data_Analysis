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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from MakePdf import *
from matplotlib.pyplot import cm #to plot following colors of rainbow
from matplotlib import rc
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = DeprecationWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.simplefilter(action = "ignore", category = PendingDeprecationWarning)
from tifffile import *
from sklearn.mixture import GMM 
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.misc
import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile
from uncertainties import unumpy

import skimage
from skimage import exposure
from my_fits import *

from numpy import genfromtxt
import matplotlib.cm as cm
from my_fits import *

from subtract_background import subtract_background
from matplotlib import colors as mcolors

### aux functions
def moving_average(a,n=3):
    vec = np.cumsum(a)
    vec[n:] = vec[n:] - vec[:-n]
    return (1/n)*vec[n-1:]

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx #array[idx]
    
def fft_cut_noise(spec,wl):
    total_wl = wl[-1]
    t = wl
    x = np.linspace(0, total_wl, len(t))
    fft_y = np.fft.fft(t)
    n = t.size
    timestep = wl[1] - wl[0]
    freq = np.fft.fftfreq(n, d = timestep)
    ind = np.abs(freq) > 0.1
    fft_y_cut = np.copy(fft_y)
    fft_y_cut[ind] = 0.0
    new_y = np.abs(np.fft.ifft(fft_y_cut))
    
    return new_y
    
### settings
fsizepl = 24
fsizenb = 20
###
sizex=8
sizey=6
dpi_no=80

fig1= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig1.set_size_inches(1200./fig1.dpi,900./fig1.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    

noplots = 2
nolines = 2

ax0 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.xaxis.set_ticks_position('bottom')
ax0.yaxis.set_ticks_position('left')

ax2 = plt.subplot2grid((nolines,noplots), (1,0), colspan=1, rowspan=1)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

ax3 = plt.subplot2grid((nolines,noplots), (1,1), colspan=1, rowspan=1)
ax3.spines['left'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('right')
ax3.yaxis.set_label_position('right')
    
#Use either April or April1
    
#specBG = genfromtxt('BGApril1.txt', delimiter='')
#specRT = genfromtxt('RTApril1.txt', delimiter='')
#spec30 = genfromtxt('30April1.txt', delimiter='')
#spec40 = genfromtxt('40April1.txt', delimiter='')
#spec50 = genfromtxt('50April1b.txt', delimiter='')
#spec60 = genfromtxt('60April1b.txt', delimiter='')
#spec70 = genfromtxt('70April1b.txt', delimiter='')
##FOR APRIL 1
##April150b, April160b, April170b
#vectemp = [24.8, 30.0, 40.0, 50.3, 60.9, 70.3]
#vectempstd = [0, 0.1, 0.1, 0.1, 0.2, 0.1]
##April150a: 49.95, 0.05 
##April160a: 60.55, 0.15
##aPRIL170a: 70.0, 0.1
#1kX, 1min integration, scan speed 5

specBG = genfromtxt('BGApril.txt', delimiter='')
specRT = genfromtxt('RTApril.txt', delimiter='')
spec30 = genfromtxt('30April.txt', delimiter='')
spec40 = genfromtxt('40April.txt', delimiter='')
spec50 = genfromtxt('50April.txt', delimiter='')
spec60 = genfromtxt('60April.txt', delimiter='')
spec70 = genfromtxt('70April.txt', delimiter='')
#FOR APRIL
vectemp = [24.45, 30.05, 40.2, 50.3, 60.4, 71.9] 
vectempstd = [0.05, 0.15, 0.3, 0.4, 0.5, 0.4]
#1kX, 1min, scan speed 5, with black tarp

###get background, taken at RT
specbg = np.array(specBG[:,1])
vector =  [specRT,spec30,spec40,spec50,spec60,spec70]

indexmov = 5
colors = iter(cm.rainbow(np.linspace(0, 1, len(vector)))) 

indice = 0

labelu = [r'$\sim$ 25 $^{\circ}$C',r'$\sim$ 30 $^{\circ}$C',r'$\sim$ 40 $^{\circ}$C',r'$\sim$ 50 $^{\circ}$C',r'$\sim$ 60 $^{\circ}$C',r'$\sim$ 70 $^{\circ}$C']


array_spec_to_plot_sub_sub = np.zeros([6,919])

indice = 0
for spec in vector:
    # x, y vectors
    wavel = spec[:,0] 
    spec_to_plot = spec[:,1] - specbg
    # cut vector only between 300 and 720nm
    a = find_nearest(wavel,195)
    b = find_nearest(wavel,905)
    wavel = wavel[a:b]
    spec_to_plot = spec_to_plot[a:b]
    # moving avg to cut noise
    mov_avg_index = indexmov
    spec_to_plot = moving_average(spec_to_plot,n=mov_avg_index)
    wavel = moving_average(wavel,n=mov_avg_index)
    
    colorful = next(colors)
    
    ax0.plot(wavel,spec_to_plot/(60), lw=2, color=colorful,label=labelu[indice])
    
#    spec_to_plot_sub = subtract_background(wavel, spec_to_plot,[500,510],[580,600])
#    if indice == 4:
#        spec_to_plot_sub_sub = subtract_background(wavel, spec_to_plot_sub,[600,605],[680,700])
#    else:
#        spec_to_plot_sub_sub = subtract_background(wavel, spec_to_plot_sub,[600,620],[680,700]) #
    spec_to_plot_sub_sub = spec_to_plot
    ax2.plot(wavel,spec_to_plot_sub_sub/np.max(spec_to_plot_sub_sub), lw=2, color=colorful)
    ax3.plot(wavel,spec_to_plot_sub_sub/np.max(spec_to_plot_sub_sub), lw=2, color=colorful)
    
    array_spec_to_plot_sub_sub[indice,:] = spec_to_plot_sub_sub
   
    indice = indice + 1
    
ax2.axvspan(534,566, alpha=0.25, color='green')
ax3.axvspan(623,677, alpha=0.25, color='red')
#labels
ax0.legend(loc = 'best',frameon=False, fontsize=fsizenb)
handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles[::-1], labels[::-1],loc = 'best',frameon=False, fontsize=fsizenb)
xmin = 500 #490
xmax = 600 #610
ax2.set_xlim([xmin, xmax])
xmin = 600#590
xmax = 700#710
ax3.set_xlim([xmin, xmax])
ax0.set_ylabel('Cathodoluminescence emission \n spectrum (Hz), 1kX mag.',fontsize=fsizepl)
ax0.set_xlabel('Wavelength (nm)',fontsize=fsizepl)
ax0.tick_params(labelsize=fsizenb)
ax0.set_yticks([1,2,3,4,5,6,7])
ax0.set_ylim([0,7.1])
ax0.set_xticks([200,300,400,500,600,700,800,900])
ax0.set_xlim([195,905])

ax0.text(-0.15, 1.0, 'a', transform=ax0.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})

ax2.set_ylabel('Cathodoluminescence emission \n spectrum (a.u.), 1kX magn.',fontsize=fsizepl)
ax2.set_xlabel('Wavelength (nm)',fontsize=fsizepl)
ax2.tick_params(labelsize=fsizenb)
ax2.set_yticks([0.5,1])
ax2.set_ylim([0,1.02])
ax2.set_xticks([534,566,540.5 ])
ax2.set_xticklabels(['534','566','540.5'])
ax2.get_xaxis().majorTicks[0].label1.set_horizontalalignment('right')

ax2.text(-0.15, 1.0, 'b', transform=ax2.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})

ax3.set_ylabel('Cathodoluminescence emission \n spectrum (a.u.), 1kX magn.',fontsize=fsizepl)
ax3.set_xlabel('Wavelength (nm)',fontsize=fsizepl)
ax3.tick_params(labelsize=fsizenb)
ax3.set_yticks([0.5,1])
ax3.set_ylim([0,1.02])
ax3.set_xticks([623,677,654.8])
ax3.set_xticklabels(['623','677','654.8'])
ax3.axvline(x=654.8 , lw=2, color='r', ls='--')
ax2.axvline(x=540.5 , lw=2, color='g', ls='--')

##############
#VISIBILITY
##############

visib50nmw = np.zeros(len(vector)) #516-566 for green, 634-684 for red
ratio50nmw = np.zeros(len(vector)) #516-566 for green, 634-684 for red
cumuvisib50nmw = np.zeros(len(vector)) #516-566 for green, 634-684 for red
visib50nmw_no_correction = np.zeros(len(vector)) #516-566 for green, 634-684 for red
ratio50nmw_no_correction = np.zeros(len(vector)) #516-566 for green, 634-684 for red
cumuvisib50nmw_no_correction = np.zeros(len(vector)) #516-566 for green, 634-684 for red

uratio = np.zeros(len(vector)) #516-566 for green, 634-684 for red
ucumuvisib = np.zeros(len(vector)) #516-566 for green, 634-684 for red

greenintensity =  np.zeros(len(vector))
redintensity =  np.zeros(len(vector))

indice = 0
for spec in vector:
    
    # x, y vectors
    wavel = spec[:,0]
    spec_to_plot = array_spec_to_plot_sub_sub[indice,:] #spec[:,1]-specbg
    spec_to_plot_no_correction = spec[:,1]-specbg
    # cut vector only between 300 and 720nm
    #wavebands given by filter
    #red filter is 650/54
    #green filter is 550/32!!!!!! NEW, brightline semrock
    a = find_nearest(wavel,534)
    b = find_nearest(wavel,566)
    c = find_nearest(wavel,623)
    d = find_nearest(wavel,677)
    
    
    wavel = wavel[a:b]
    spec_to_plot_green = spec_to_plot[a:b]
    spec_to_plot_red = spec_to_plot[c:d]
    spec_to_plot_green_no_correction = spec_to_plot_no_correction[a:b]
    spec_to_plot_red_no_correction = spec_to_plot_no_correction[c:d]
    # moving avg to cut noise
    mov_avg_index = indexmov
    spec_to_plot_green = moving_average(spec_to_plot_green,n=mov_avg_index)
    spec_to_plot_red = moving_average(spec_to_plot_red,n=mov_avg_index)
    spec_to_plot_green_no_correction = moving_average(spec_to_plot_green_no_correction,n=mov_avg_index)
    spec_to_plot_red_no_correction = moving_average(spec_to_plot_red_no_correction,n=mov_avg_index)
    
    ratio50nmw[indice] = (np.average(spec_to_plot_red))/( np.average(spec_to_plot_green))
    cumuvisib50nmw[indice] = (np.sum(spec_to_plot_green) - np.sum(spec_to_plot_red))/( np.sum(spec_to_plot_green) + np.sum(spec_to_plot_red))

    ratio50nmw_no_correction[indice] = (np.average(spec_to_plot_red_no_correction))/( np.average(spec_to_plot_green_no_correction))
    cumuvisib50nmw_no_correction[indice] = (np.sum(spec_to_plot_green_no_correction) - np.sum(spec_to_plot_red_no_correction))/( np.sum(spec_to_plot_green_no_correction) + np.sum(spec_to_plot_red_no_correction))

    greenintensity[indice] = ( np.average(spec_to_plot_green))
    redintensity[indice] = (np.average(spec_to_plot_red))

    indice = indice + 1
    
plt.tight_layout()
    
fig2= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig2.set_size_inches(1200./fig2.dpi,900./fig1.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino') 

ax30 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)
ax30.spines['right'].set_visible(False)
ax30.spines['top'].set_visible(False)
ax30.xaxis.set_ticks_position('bottom')
ax30.yaxis.set_ticks_position('left')
ax30.yaxis.set_label_position('left')

ax31 = plt.subplot2grid((nolines,noplots), (1,0), colspan=1, rowspan=1)
ax31.spines['right'].set_visible(False)
ax31.spines['top'].set_visible(False)
ax31.xaxis.set_ticks_position('bottom')
ax31.yaxis.set_ticks_position('left')
ax31.yaxis.set_label_position('left')
    
normto = 0
#ax30.errorbar(vectemp,ratio50nmw/ratio50nmw[normto], xerr=vectempstd, lw=2, color='k',ls='None',label='Ratio of intensities',marker='d',markersize=12)
#ax30.errorbar(vectemp,cumuvisib50nmw/cumuvisib50nmw[normto], xerr=vectempstd,lw=2, color='k',label='Visibility of intensitites',ls='None', marker='o',markersize=12)

ax30.errorbar(vectemp,ratio50nmw_no_correction/ratio50nmw_no_correction[normto], xerr=vectempstd,lw=2, color='k',ls='None',marker='d',markersize=12,markeredgecolor='gray',label='Ratio of intensities')
ax30.errorbar(vectemp,cumuvisib50nmw_no_correction/cumuvisib50nmw_no_correction[normto], xerr=vectempstd,lw=2, color='k',ls='None', marker='o',markersize=12,markeredgecolor='gray',label='Visibility of intensitites')

#ax30.plot(vectemp,greenintensity/greenintensity[normto],'g')
#ax30.plot(vectemp,redintensity/redintensity[normto],'r')


ax30.legend(loc='best', frameon=False, fontsize=fsizenb)

# get handles
handles, labels = ax30.get_legend_handles_labels()
# remove the errorbars
handles = [h[0] for h in handles]
# use them in the legend
ax30.legend(handles, labels, loc='best',numpoints=1,frameon=False, fontsize=fsizenb)

ax30.set_ylabel('Intensity thermometry signal, \n norm. to $\sim$ 25 $^{\circ}$C (a.u.)',fontsize=fsizepl)
ax30.set_xlabel(r'Temperature at sample ($^{\circ}$C)',fontsize=fsizepl)
ax30.tick_params(labelsize=fsizenb)
ax30.set_yticks([1,0.5,1.5])
ax30.set_ylim([0.4,1.6])
ax30.set_xticks([25,30,40,50,60,70])
ax30.set_xlim([20,75])
ax30.text(-0.13, 1.0, 'a', transform=ax30.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})
ax31.text(-0.13, 1.0, 'b', transform=ax31.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})
         
#Fitting ration
def d_parab(x,aa,bb,offs):
    return aa*(x-x[0])**2 + b*(x-x[0]) + offs
    
def d_visibparab(x,aa,bb,offs):
    return -1.0 + 2.0/(1+offs+aa*(x)**2 + b*(x))
    
def d_parabdown(x,aa,bb,offs):
    return aa*(x-x[-1])**2 + b*(x-x[-1]) + offs
    
def d_paraball(x,aa,bb,c):
    return aa*(x)**2 + b*(x) + c

def d_line(x, aa):
    return aa*(x-x[0])+1.0

vectemp = np.array(vectemp)

(a,b, result) = parabola_fit_fixed_point(vectemp[:-1],ratio50nmw_no_correction[:-1]/ratio50nmw_no_correction[normto],vectemp[0], 1.0)
ax30.plot(vectemp[:-1],d_parab(vectemp[:-1],a,b,1.0),color='r',lw=2)
#choice between parabola and line     
 
x_vec3B = np.linspace(vectemp[0], vectemp[-2], 100)
 
sigma_dev = np.sqrt([result.covar[0,0],result.covar[1,1]]) # sqrt(diag elements) of pcov are the 1 sigma deviations
values = np.array([])
for s1 in [-1, +1]:
    for ss1 in [-1, +1]:
        my_hlp = d_parab( x_vec3B, a + s1*sigma_dev[0], b + ss1*sigma_dev[1], 1.0)
        values = np.vstack((values, my_hlp)) if values.size else my_hlp
fitError = np.std(values, axis=0) 

ax30.fill_between(x_vec3B,  
             d_parab(x_vec3B, a,b,1.0)-1.0*fitError,
             d_parab(x_vec3B, a,b,1.0)+1.0*fitError,  
             color ='r' , #was gray [168/256,175/256,175/256]
             edgecolor='r',
             facecolor='r', #was gray [168/256,175/256,175/256],
             alpha=0.25,
             linewidth=0.0)
             
print('a,b, parab up')
print('a=' + str(result.params['a'].value))
print('b=' + str(result.params['b'].value))   
         
#for all T points
#(a,b, result) = parabola_fit_fixed_point(vectemp,ratio50nmw_no_correction/ratio50nmw_no_correction[normto],vectemp[0], 1.0)
#ax30.plot(vectemp,d_parab(vectemp,a,b,1.0),color='b',lw=2)
#
#x_vec3B = np.linspace(vectemp[0], vectemp[-1], 100)
# 
#sigma_dev = np.sqrt([result.covar[0,0],result.covar[1,1]]) # sqrt(diag elements) of pcov are the 1 sigma deviations
#values = np.array([])
#for s1 in [-1, +1]:
#    for ss1 in [-1, +1]:
#        my_hlp = d_parab( x_vec3B, a + s1*sigma_dev[0], b + ss1*sigma_dev[1], 1.0)
#        values = np.vstack((values, my_hlp)) if values.size else my_hlp
#fitError = np.std(values, axis=0) 
#
#ax30.fill_between(x_vec3B,  
#             d_parab(x_vec3B, a,b,1.0)-1.0*fitError,
#             d_parab(x_vec3B, a,b,1.0)+1.0*fitError,  
#             color ='b' , #was gray [168/256,175/256,175/256]
#             edgecolor='b',
#             facecolor='b', #was gray [168/256,175/256,175/256],
#             alpha=0.25,
#             linewidth=0.0)
#         
print(d_visibparab(vectemp[:-1],a,b,1.0))
ax30.plot(vectemp[:-1],d_visibparab(vectemp[:-1],a,b,1.0),color='k',lw=2)   

simuint = genfromtxt('MathematicaSimuIntensity.txt', delimiter='')   
ax30.plot(simuint[:,0],simuint[:,1],'m',lw=2)   
ax30.axhline(1.12)
         
plt.show()
lllll 
         
         
         
         
         
         
         #########################################################
         
#### Fits in ax30
#fit passing thru 1.0
# y = a(x-xo) + yo, yo here being 1.0
#(a,result) = linear_fit_fixed_point(np.array(vectemp), ratio50nmw/ratio50nmw[normto], vectemp[0], 1.0)
#ax30.plot(np.array(vectemp),a*(np.array(vectemp)-vectemp[0])+1.0,color='k',lw=2)
#
#(a2,result) = visi_fit_fixed_point(np.array(vectemp), cumuvisib50nmw/cumuvisib50nmw[normto],vectemp[0], 1.0)
#ax30.plot(np.array(vectemp),-1.0 + 2.0/(1.0 + (-1-a2*vectemp[0] + 2/(1.0 +1.0)) + a2 * np.array(vectemp)),color='k',lw=2)

#(a3,result) = linear_fit_fixed_point(np.array(vectemp), ratio50nmw_no_correction/ratio50nmw_no_correction[normto], vectemp[0], 1.0)
#ax30.plot(np.array(vectemp),a3*(np.array(vectemp)-vectemp[0])+1.0,color='k',lw=2)
#

#### USING PARABOLA MODEL FOR RATIO
(a4,result) = visi_fit_fixed_point(np.array(vectemp), cumuvisib50nmw_no_correction/cumuvisib50nmw_no_correction[normto],vectemp[0], 1.0)
ax30.plot(np.array(vectemp),-1.0 + 2.0/(1.0 + (-1-a4*vectemp[0] + 2/(1.0 +1.0)) + a4 * np.array(vectemp)),color='k',lw=2)

xhere = np.linspace(vectemp[0],vectemp[-1],100)
#ax31.plot(xhere, 100.0*np.abs(np.ones(len(xhere))*a),color='k',lw=2,ls='--')
#ax31.plot(xhere, 100.0*np.abs(-2.0*a2/(1.0+a2*xhere+(-1-a2*xhere[0]+2/(1.0+1.0)))**2),color='k',lw=2,ls='--')
ax31.plot(xhere, 100.0*np.abs(np.ones(len(xhere))*a3),color='k',lw=2,ls='--')
ax31.plot(xhere, 100.0*np.abs(-2.0*a4/(1.0+a4*xhere+(-1-a4*xhere[0]+2/(1.0+1.0)))**2),color='k',lw=2,ls='--')
ax31.set_xticks([25,30,40,50,60,70])
ax31.set_xlim([20,75])
ax31.set_ylabel(r'$\vert\partial$(signal)/$\partial$T$\vert$ ($\%$ $^{\circ}$C$^{-1}$)',fontsize=fsizepl)
ax31.set_xlabel(r'Temperature at sample ($^{\circ}$C)',fontsize=fsizepl)
ax31.tick_params(labelsize=fsizenb)
#ax31.plot(xhere[0],100.0*np.abs(a),marker='d',markersize=12,color='k')
#ax31.plot(xhere[0],100.0*np.abs(-2.0*a2/(1.0+a2*xhere[0]+(-1-a2*xhere[0]+2/(1.0+1.0)))**2),marker='o',markersize=12,color='k')
ax31.plot(xhere[0],100.0*np.abs(a3),marker='d',markersize=12,color='k',markeredgecolor='k')
ax31.plot(xhere[0],100.0*np.abs(-2.0*a4/(1.0+a4*xhere[0]+(-1-a4*xhere[0]+2/(1.0+1.0)))**2),marker='o',markersize=12,color='k',markeredgecolor='gray')

ax31.fill_between(xhere,0.5,1.5, color =[168/256,175/256,175/256],edgecolor='k',
                         facecolor=[168/256,175/256,175/256],
                         alpha=0.5,
                         linewidth=0.0)
ax31.text(47.5,1.0, 'previously reported \n (fluorescence ratio of intensity)', fontsize=fsizenb, va='center',ha='center')

ax31.set_ylim([0,1.6])
ax31.set_yticks([0.5,1.0,1.5])

plt.tight_layout()
multipage_longer_desired_aspect_ratio('SI-spectraApril.pdf',1600,1200,dpi=80,)

print(vectemp)
print(ratio50nmw/ratio50nmw[normto])
#print(unumpy.std_devs(uratio/ratio[normto]))
print(cumuvisib50nmw/cumuvisib50nmw[normto])
#print(unumpy.std_devs(ucumuvisib/ucumuvisib[normto]))

print('fits')
print(a)
print(b)
#print(a2)
#print(b2)

lklklk
