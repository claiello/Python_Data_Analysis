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
    
specBG = genfromtxt('bgNEW.txt', delimiter='')
spec50 = genfromtxt('50NEW.txt', delimiter='')
spec60 = genfromtxt('60NEW.txt', delimiter='')
spec70 = genfromtxt('70NEW.txt', delimiter='')

specBGPLASMAED = genfromtxt('BGRT5minPLASMAED.txt', delimiter='')
specRTPLASMAED = genfromtxt('RTPLASMAED.txt', delimiter='')
spec30PLASMAED = genfromtxt('30PLASMAED.txt', delimiter='')
spec40PLASMAED = genfromtxt('40PLASMAED.txt', delimiter='')

###get background, taken at RT
specbg = np.array(specBG[:,1])
specbgplasmaed = np.array(specBGPLASMAED[:,1])

#ALL SAME REGION, 750x, SCAN SPEED 1, INTEGRATION TIME 10MIN
#AT 50c, VOLTAGE SOURCE MISBEHAVED, LARGER TEMP DELTA
#they were taken in the order as above
vector =  [specRTPLASMAED,spec30PLASMAED,spec40PLASMAED,spec50,spec60,spec70]
vectemp = [25.1, 30.45, 39.9, 50.15, 61.4, 71.3] #[25,30,40,50,60,70]
vectempstd = [0, 0.35, 0.1, 0.55, 1.4, 1.3]

indexmov = 5
colors = iter(cm.rainbow(np.linspace(0, 1, len(vector)))) 

indice = 0

labelu = [r'$\sim$ 25 $^{\circ}$C',r'$\sim$ 30 $^{\circ}$C',r'$\sim$ 40 $^{\circ}$C',r'$\sim$ 50 $^{\circ}$C',r'$\sim$ 60 $^{\circ}$C',r'$\sim$ 70 $^{\circ}$C']


array_spec_to_plot_sub_sub = np.zeros([6,919])

indice = 0
for spec in vector:
    # x, y vectors
    wavel = spec[:,0] 
    if indice in [0,1,2]:
        spec_to_plot = spec[:,1] - specbgplasmaed
    elif indice in [3,4,5]:
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
    
    ax0.plot(wavel,spec_to_plot/(600), lw=2, color=colorful,label=labelu[indice])
    
    spec_to_plot_sub = subtract_background(wavel, spec_to_plot,[500,510],[580,600])
    if indice == 4:
        spec_to_plot_sub_sub = subtract_background(wavel, spec_to_plot_sub,[600,605],[680,700])
    else:
        spec_to_plot_sub_sub = subtract_background(wavel, spec_to_plot_sub,[600,620],[680,700]) #
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
ax0.set_ylabel('Bulk cathodoluminescence \n emission spectrum (Hz)',fontsize=fsizepl)
ax0.set_xlabel('Wavelength (nm)',fontsize=fsizepl)
ax0.tick_params(labelsize=fsizenb)
ax0.set_yticks([1,2,3,4])
ax0.set_ylim([0,4.1])
ax0.set_xticks([200,300,400,500,600,700,800,900])
ax0.set_xlim([195,905])

ax0.text(-0.1, 1.0, 'a', transform=ax0.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})

ax2.set_ylabel('Bulk cathodoluminescence \n emission spectrum (a.u.)',fontsize=fsizepl)
ax2.set_xlabel('Wavelength (nm)',fontsize=fsizepl)
ax2.tick_params(labelsize=fsizenb)
ax2.set_yticks([0.5,1])
ax2.set_ylim([0,1.02])
ax2.set_xticks([534,566,540.5 ])
ax2.set_xticklabels(['534.0','566.0','540.5'])
ax2.get_xaxis().majorTicks[0].label1.set_horizontalalignment('right')

ax2.text(-0.1, 1.0, 'b', transform=ax2.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})

ax3.set_ylabel('Bulk cathodoluminescence \n emission spectrum (a.u.)',fontsize=fsizepl)
ax3.set_xlabel('Wavelength (nm)',fontsize=fsizepl)
ax3.tick_params(labelsize=fsizenb)
ax3.set_yticks([0.5,1])
ax3.set_ylim([0,1.02])
ax3.set_xticks([623,677,654.1])
ax3.axvline(x=654.1 , lw=2, color='r', ls='--')
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


    indice = indice + 1
    
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
ax30.errorbar(vectemp,ratio50nmw/ratio50nmw[normto], xerr=vectempstd, lw=2, color='k',ls='None',label='Ratio of intensities',marker='d',markersize=12)
ax30.errorbar(vectemp,cumuvisib50nmw/cumuvisib50nmw[normto], xerr=vectempstd,lw=2, color='k',label='Visibility of intensitites',ls='None', marker='o',markersize=12)

ax30.errorbar(vectemp,ratio50nmw_no_correction/ratio50nmw_no_correction[normto], xerr=vectempstd,lw=2, color='gray',ls='None',marker='d',markersize=12,markeredgecolor='gray',label='...without baseline correction')
ax30.errorbar(vectemp,cumuvisib50nmw_no_correction/cumuvisib50nmw_no_correction[normto], xerr=vectempstd,lw=2, color='gray',ls='None', marker='o',markersize=12,markeredgecolor='gray',label='...without baseline correction')

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
ax30.set_ylim([0.2,1.6])
ax30.set_xticks([25,30,40,50,60,70])
ax30.set_xlim([20,75])
ax30.text(-0.1, 1.0, 'a', transform=ax30.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})
ax31.text(-0.1, 1.0, 'b', transform=ax31.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})
         
#### Fits in ax30
#fit passing thru 1.0
# y = a(x-xo) + yo, yo here being 1.0
(a,result) = linear_fit_fixed_point(np.array(vectemp), ratio50nmw/ratio50nmw[normto], vectemp[0], 1.0)
ax30.plot(np.array(vectemp),a*(np.array(vectemp)-vectemp[0])+1.0,color='k',lw=2)
#
(a2,result) = visi_fit_fixed_point(np.array(vectemp), cumuvisib50nmw/cumuvisib50nmw[normto],vectemp[0], 1.0)
ax30.plot(np.array(vectemp),-1.0 + 2.0/(1.0 + (-1-a2*vectemp[0] + 2/(1.0 +1.0)) + a2 * np.array(vectemp)),color='k',lw=2)

(a3,result) = linear_fit_fixed_point(np.array(vectemp), ratio50nmw_no_correction/ratio50nmw_no_correction[normto], vectemp[0], 1.0)
ax30.plot(np.array(vectemp),a3*(np.array(vectemp)-vectemp[0])+1.0,color='gray',lw=2)
#
(a4,result) = visi_fit_fixed_point(np.array(vectemp), cumuvisib50nmw_no_correction/cumuvisib50nmw_no_correction[normto],vectemp[0], 1.0)
ax30.plot(np.array(vectemp),-1.0 + 2.0/(1.0 + (-1-a4*vectemp[0] + 2/(1.0 +1.0)) + a4 * np.array(vectemp)),color='gray',lw=2)

xhere = np.linspace(vectemp[0],vectemp[-1],100)
ax31.plot(xhere, 100.0*np.abs(np.ones(len(xhere))*a),color='k',lw=2,ls='--')
ax31.plot(xhere, 100.0*np.abs(-2.0*a2/(1.0+a2*xhere+(-1-a2*xhere[0]+2/(1.0+1.0)))**2),color='k',lw=2,ls='--')
ax31.plot(xhere, 100.0*np.abs(np.ones(len(xhere))*a3),color='gray',lw=2,ls='--')
ax31.plot(xhere, 100.0*np.abs(-2.0*a4/(1.0+a4*xhere+(-1-a4*xhere[0]+2/(1.0+1.0)))**2),color='gray',lw=2,ls='--')
ax31.set_xticks([25,30,40,50,60,70])
ax31.set_xlim([20,75])
ax31.set_ylabel(r'$\vert\partial$(signal)/$\partial$T$\vert$ ($\%$ $^{\circ}$C$^{-1}$)',fontsize=fsizepl)
ax31.set_xlabel(r'Temperature at sample ($^{\circ}$C)',fontsize=fsizepl)
ax31.tick_params(labelsize=fsizenb)
ax31.plot(xhere[0],100.0*np.abs(a),marker='d',markersize=12,color='k')
ax31.plot(xhere[0],100.0*np.abs(-2.0*a2/(1.0+a2*xhere[0]+(-1-a2*xhere[0]+2/(1.0+1.0)))**2),marker='o',markersize=12,color='k')
ax31.plot(xhere[0],100.0*np.abs(a3),marker='d',markersize=12,color='gray',markeredgecolor='gray')
ax31.plot(xhere[0],100.0*np.abs(-2.0*a4/(1.0+a4*xhere[0]+(-1-a4*xhere[0]+2/(1.0+1.0)))**2),marker='o',markersize=12,color='gray',markeredgecolor='gray')

ax31.fill_between(xhere,0.5,1.5, color =[168/256,175/256,175/256],edgecolor='k',
                         facecolor=[168/256,175/256,175/256],
                         alpha=0.5,
                         linewidth=0.0)
ax31.text(58,1.4, 'previously reported \n (fluorescence ratio of intensity)', fontsize=fsizenb, va='center',ha='center')

ax31.set_ylim([0.4,2.1])
ax31.set_yticks([0.5,1.0,1.5,2.0])

plt.tight_layout()
multipage_longer_desired_aspect_ratio('SI-spectraPLASMAED.pdf',1600,1200,dpi=80,)

print(vectemp)
print(ratio50nmw/ratio50nmw[normto])
#print(unumpy.std_devs(uratio/ratio[normto]))
print(cumuvisib50nmw/cumuvisib50nmw[normto])
#print(unumpy.std_devs(ucumuvisib/ucumuvisib[normto]))

print('fits')
print(a)
print(b)
print(a2)
#print(b2)

lklklk
#OLDDATA 
#####
#specRT = genfromtxt('RTB.txt', delimiter='')
#spec30 = genfromtxt('30C.txt', delimiter='')
#spec40 = genfromtxt('40.txt', delimiter='')
#spec40B = genfromtxt('40B.txt', delimiter='')
#spec50 = genfromtxt('50.txt', delimiter='')
#spec60 = genfromtxt('60.txt', delimiter='')
#spec70 = genfromtxt('70.txt', delimiter='')
#specRTA = genfromtxt('RT.txt', delimiter='')
#spec30A = genfromtxt('30A.txt', delimiter='')
#spec30B = genfromtxt('30B.txt', delimiter='')
#
#####Add 2 30 files, 30A and 30B
#spec30D= np.zeros([1044,2 ])
#spec30D[:,1] = (1*spec30A[:,1] + 1*spec30B[:,1] + 1*spec30[:,1])/3
#spec30D[:,0] = spec30A[:,0]
#
##absolutely comparable: spec40, spec50, spec60, spec70 (same region, all going up)
#vector =  [specRTA,spec30D,spec40,spec50,spec60,spec70]
#
##ORDER AT WHICH THEY WERE TAKEN:
##RT PLACE 1
##30a PLACE 1
##30B PLACE 1
##40 PLACE 2 UNTIL THE END
##50
##60
##70
##40b
##30c
##rtb
#visib25nmw = np.zeros(len(vector)) #516-566 for green, 634-684 for red
#ratio25nmw = np.zeros(len(vector)) #516-566 for green, 634-684 for red
#cumuvisib25nmw = np.zeros(len(vector)) #516-566 for green, 634-684 for red
#
#indice = 0
#for spec in vector:
#    
#    # x, y vectors
#    wavel = spec[:,0]
#    spec_to_plot = spec[:,1]-specbg
#    # cut vector only between 300 and 720nm
#    a = find_nearest(wavel,528.7)
#    b = find_nearest(wavel,553.7)
#    c = find_nearest(wavel,646.9)
#    d = find_nearest(wavel,671.9)
#    wavel = wavel[a:b]
#    spec_to_plot_green = spec_to_plot[a:b]
#    spec_to_plot_red = spec_to_plot[c:d]
#    # moving avg to cut noise
#    mov_avg_index = indexmov
#    spec_to_plot_green = moving_average(spec_to_plot_green,n=mov_avg_index)
#    spec_to_plot_red = moving_average(spec_to_plot_red,n=mov_avg_index)
#    #wavel = moving_average(wavel,n=mov_avg_index) #wavel + mov_avg_index/2.0 * (wavel[1] - wavel[0])
#    
#    # take out cst background in interval 300/720
#    #spec_to_plot_green = spec_to_plot_green - np.average(spec_to_plot_green)
#    #spec_to_plot_red = spec_to_plot_red - np.average(spec_to_plot_red)
#
#    # make all go to zero min
##    min_val = np.amin(spec_to_plot_green)
##    spec_to_plot_green = spec_to_plot_green + np.abs(min_val)
##    min_val = np.amin(spec_to_plot_red)
##    spec_to_plot_red = spec_to_plot_red + np.abs(min_val)
##    #normalize taking into account 300/720 interval
##    spec_to_plot_green = spec_to_plot_green/np.max(spec_to_plot_green)
##    spec_to_plot_red = spec_to_plot_red/np.max(spec_to_plot_red)
#    
#    #find index of red/green transition
##    e = find_nearest(wavel,600)
#    
#    # Find center of mass in interval 300/720, for two bands
#    #green
##    aux_y = spec_to_plot[:e]
##    aux_x = wavel[:e]
##    f = find_nearest(aux_y, np.max(aux_y))
##    print(aux_x[f])
#    #ax0.axvline(x=aux_x[f] , lw=2, color='g', ls='--')
#    
#    #red
##    aux_y = spec_to_plot[e:]
##    aux_xx = wavel[e:]
##    g = find_nearest(aux_y, np.max(aux_y))
##    print(aux_xx[g])
#    #ax0.axvline(x=aux_xx[g] , lw=2, color='r', ls='--', ymax = 0.8)
#    
#    visib25nmw[indice] = (np.average(spec_to_plot_green) - np.average(spec_to_plot_red))/( np.average(spec_to_plot_green) + np.average(spec_to_plot_red))
#    ratio25nmw[indice] = (np.average(spec_to_plot_green))/(  np.average(spec_to_plot_red))
#    cumuvisib25nmw[indice] = (np.sum(spec_to_plot_green) - np.sum(spec_to_plot_red))/( np.sum(spec_to_plot_green) + np.sum(spec_to_plot_red))
#
#    # plot
#    #ax0.plot(wavel[:e],spec_to_plot[:e], lw=2, color='g')
#    #ax0.plot(wavel[e:],spec_to_plot[e:], lw=2, color='r')
#    # plot vertical line at 593nm (dichroic)
#    #ax0.axvline(x=593, lw=2, color='k', ls='--', ymax = 0.8)
#    
#    indice = indice + 1

#visib100nmw = np.zeros(len(vector)) #500-600 for green, 600-700 for red
#ratio100nmw = np.zeros(len(vector)) #500-600 for green, 600-700 for red
#cumuvisib100nmw = np.zeros(len(vector)) #500-600 for green, 600-700 for red
#indice = 0
#for spec in vector:
#    
#    # x, y vectors
#    wavel = spec[:,0]
#    spec_to_plot = spec[:,1]-specbg
#    # cut vector only between 300 and 720nm
#    a = find_nearest(wavel,500)
#    b = find_nearest(wavel,700)
#    wavel = wavel[a:b]
#    spec_to_plot = spec_to_plot[a:b]
#    # moving avg to cut noise
#    mov_avg_index = indexmov
#    spec_to_plot = moving_average(spec_to_plot,n=mov_avg_index)
#    wavel = moving_average(wavel,n=mov_avg_index) #wavel + mov_avg_index/2.0 * (wavel[1] - wavel[0])
#    
##    # take out cst background in interval 300/720
##    spec_to_plot = spec_to_plot - np.average(spec_to_plot)
##    # make all go to zero min
##    min_val = np.amin(spec_to_plot)
##    spec_to_plot = spec_to_plot + np.abs(min_val)
##    #normalize taking into account 300/720 interval
##    spec_to_plot = spec_to_plot/np.max(spec_to_plot)
#    
#    #find index of red/green transition
#    e = find_nearest(wavel,600)
#    
#    # Find center of mass in interval 300/720, for two bands
#    #green
##    aux_y = spec_to_plot[:e]
##    aux_x = wavel[:e]
##    f = find_nearest(aux_y, np.max(aux_y))
##    print(aux_x[f])
#    #ax0.axvline(x=aux_x[f] , lw=2, color='g', ls='--')
#    
#    #red
##    aux_y = spec_to_plot[e:]
##    aux_xx = wavel[e:]
##    g = find_nearest(aux_y, np.max(aux_y))
##    print(aux_xx[g])
#    #ax0.axvline(x=aux_xx[g] , lw=2, color='r', ls='--', ymax = 0.8)
#    
#    visib100nmw[indice] = (np.average(spec_to_plot[:e]) - np.average(spec_to_plot[e:]))/( np.average(spec_to_plot[:e]) + np.average(spec_to_plot[e:]))
#    ratio100nmw[indice] = (np.average(spec_to_plot[:e]) )/(  np.average(spec_to_plot[e:]))
#    cumuvisib100nmw[indice] = (np.sum(spec_to_plot[:e]) - np.sum(spec_to_plot[e:]))/( np.sum(spec_to_plot[:e]) + np.sum(spec_to_plot[e:]))
#
#    # plot
#    #ax0.plot(wavel[:e],spec_to_plot[:e], lw=2, color='g')
#    #ax0.plot(wavel[e:],spec_to_plot[e:], lw=2, color='r')
#    # plot vertical line at 593nm (dichroic)
#    #ax0.axvline(x=593, lw=2, color='k', ls='--', ymax = 0.8)
#    
#    indice = indice + 1

#ax30.plot(vectemp,visib300720/visib300720[normto], lw=2, color='k')
#ax30.plot(vectemp,visib100nmw/visib100nmw[normto], lw=2, color='b')
#ax30.plot(vectemp,visib50nmw/visib50nmw[normto], lw=2, color='c')
#ax30.plot(vectemp,visib25nmw/visib25nmw[normto], lw=2, color='m')
#ax30.plot(vectemp,ratio300720/ratio300720[normto], lw=2, color='k',ls='--')
#ax30.plot(vectemp,ratio100nmw/ratio100nmw[normto], lw=2, color='b',ls='--')
#ax30.plot(vectemp,ratio25nmw/ratio25nmw[normto], lw=2, color='m',ls='--')
#ax30.plot(vectemp,cumuvisib300720/cumuvisib300720[normto], lw=2, color='k',ls='dotted')
#ax30.plot(vectemp,cumuvisib100nmw/cumuvisib100nmw[normto], lw=2, color='b',ls='dotted')
#ax30.plot(vectemp,cumuvisib25nmw/cumuvisib25nmw[normto], lw=2, color='m',ls='dotted')

#####VISIBILITY WHOLE PMT RANGE
#visib300720 = np.zeros(len(vector))
#ratio300720 = np.zeros(len(vector))
#cumuvisib300720 = np.zeros(len(vector))
#
#indice = 0
#for spec in vector:
#    
#    # x, y vectors
#    wavel = spec[:,0] 
#    spec_to_plot = array_spec_to_plot_sub_sub[indice,:] #spec[:,1]-specbg
#    spec_to_plot_no_correction = spec[:,1]
#    # cut vector only between 300 and 720nm
#    a = find_nearest(wavel,300)
#    b = find_nearest(wavel,720)
#    wavel = wavel[a:b]
#    spec_to_plot = spec_to_plot[a:b]
#    spec_to_plot_no_correction = spec_to_plot_no_correction[a:b]
#    # moving avg to cut noise
#    mov_avg_index = indexmov
#    spec_to_plot = moving_average(spec_to_plot,n=mov_avg_index)
#    spec_to_plot_no_correction = moving_average(spec_to_plot,n=mov_avg_index)
#    wavel = moving_average(wavel,n=mov_avg_index) 
#
#    
#    #find index of red/green transition
#    e = find_nearest(wavel,593)
#    
#    visib300720[indice] = (np.average(spec_to_plot[:e]) - np.average(spec_to_plot[e:]))/( np.average(spec_to_plot[:e]) + np.average(spec_to_plot[e:]))
#    ratio300720[indice] = (np.average(spec_to_plot[:e]))/(np.average(spec_to_plot[e:]))
#    cumuvisib300720[indice] = (np.sum(spec_to_plot[:e]) - np.sum(spec_to_plot[e:]))/( np.sum(spec_to_plot[:e]) + np.sum(spec_to_plot[e:]))
#
#    indice = indice + 1