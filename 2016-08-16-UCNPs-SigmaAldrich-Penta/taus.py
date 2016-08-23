import os
import sys
import scipy as sp
import scipy.misc
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
#from BackgroundCorrection import *
import matplotlib.cm as cm
import scipy.ndimage as ndimage
#from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation
import gc
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../Python modules/") # necessary for the tex fonts
from MakePdf import *

fsizetit = 18
fsizepl = 16
sizex = 8
sizey = 6
dpi_no = 80
lw = 2

import scalebars as sb
import matplotlib.gridspec as gridspec
from tempfile import NamedTemporaryFile

#2.8 3 3.2kV

tau1red =       [0.01,0.01,0.11]
tau1red_error = [0.0005, 0.0005, 0.131]

tau2red =       [0.001, 0.001, 0.005]
tau2red_error = [0.0005,0.0005, 0.122]

tau1blue =       [0.1,0.11,0.09]
tau1blue_error = [0.02, 0.0005, 0.0005]

tau2blue =       [0.008, 0.001, 0.001]
tau2blue_error = [0.001,0.0005, 0.0005]



x_vec = [2.8, 3.0, 3.2]

plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')
plt.suptitle('150nm upconverting NPs (30$\mu$m aperture, 40ns time bins, 36kX or 3.1nm pixels, red/blue photons ($>$/$<$ 593nm)', fontsize=fsizetit)

ax1 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((2,2), (1,0), colspan=1, sharex=ax1)

#ax1.set_title(r'For $\tau$' +  ' of red/blue photons ($>$/$<$ 593nm)',fontsize=fsizepl)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

ax1.errorbar(x_vec, tau1red, yerr=tau1red_error, fmt='ro',markersize=10)
ax2.errorbar(x_vec, tau2red, yerr=tau2red_error, fmt='rs', markersize=5)
ax1.errorbar(x_vec, tau1blue, yerr=tau1blue_error, fmt='bo',markersize=10)
ax2.errorbar(x_vec, tau2blue, yerr=tau2blue_error, fmt='bs', markersize=5)

ax1.set_xticks(x_vec)
#ax1.set_xticklabels(['OA','2','20','40 CS','40 C','80'])
plt.xlim([2.6,3.4])
plt.ylim([0,0.04])

#ax1.set_yticks([0.25, 0.5, 0.75, 1.0, 1.25])
#ax2.set_yticks([0.025, 0.050])
#from uncertainties import unumpy
#tau1U = unumpy.uarray(tau1,tau1_error)
#tau2U = unumpy.uarray(tau2,tau2_error)
#tau1bgU = unumpy.uarray(tau1bg,tau1bg_error)
#tau2bgU = unumpy.uarray(tau2bg,tau2bg_error)
#
#ratio_large = tau1U/tau1bgU
#ratio_small = tau2U/tau2bgU
#large_nb = np.zeros(len(ratio_large))
#large_err = np.zeros(len(ratio_large))
#small_nb = np.zeros(len(ratio_large))
#small_err = np.zeros(len(ratio_large))
#for jj in np.arange(len(ratio_large)):
#    print(jj)
#    large_nb[jj] = float(str(ratio_large[jj]).partition('+/-')[0])
#    large_err[jj] = float(str(ratio_large[jj]).partition('+/-')[2])
#    small_nb[jj] = float(str(ratio_small[jj]).partition('+/-')[0])
#    small_err[jj] = float(str(ratio_small[jj]).partition('+/-')[2])
#
##ax1.set_ylim([0.55,0.85])
##ax2.set_ylim([0.0,0.1])
#
#ax1.errorbar(x_vec, large_nb, yerr=large_err, fmt='ko',markersize=10)
#ax1.axhline(y=1, xmin=0, xmax=60,linewidth=2, color = 'k', ls = '--')
#
#ax2.errorbar(x_vec, small_nb, yerr=small_err, fmt='ks', markersize=5)
#ax2.axhline(y=1, xmin=0, xmax=60,  linewidth=2, color = 'k',ls = '--')
#
ax1.set_ylabel('Longer time constant ($\mu$s)',fontsize=fsizepl)
ax2.set_ylabel('Shorter time constant ($\mu$s)',fontsize=fsizepl)
plt.xlabel(r"Electron voltage (kV)",fontsize=fsizepl) 
#major_ticks = [2,10,20,40,60]
#ax1.set_xticks(major_ticks) 
#plt.xlim([0,62])

ax1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)


#ax1.set_title(r'For $\tau$' +  ' of red/blue photons ($>$/$<$ 593nm)',fontsize=fsizepl)


#taus below are actually A1/A2
tau1red =       [0.119,4.495,0.082]
#tau1red_error = [0.0005, 0.0005, 0.131]

tau1blue =       [0.032,0.035,0.037]
#tau1blue_error = [0.02, 0.0005, 0.0005]


ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

ax1.errorbar(x_vec, tau1red, fmt='ro',markersize=10)
ax1.errorbar(x_vec, tau1blue, fmt='bo',markersize=10)
ax1.set_xticks(x_vec)
ax1.set_ylim([0,4.6])
plt.xlim([2.6,3.4])
ax1.set_ylabel('Ratio of amplitudes \n longer/shorter time constants',fontsize=fsizepl)
#ax2.set_ylabel('Shorter time constant ($\mu$s)',fontsize=fsizepl)
plt.xlabel(r"Electron voltage (kV)",fontsize=fsizepl) 

#### Bleaching

##SECOND COLUMN
name = ['2016-08-16-2030_ImageSequence_SANP_36.196kX_3.000kV_30mu_2','2016-08-16-2035_ImageSequence_SANP_36.196kX_3.000kV_30mu_3','2016-08-16-2044_ImageSequence_SANP_36.957kX_3.000kV_30mu_6']
namefile = ['CheckBleach1','CheckBleach2','CheckBleach3']

index = 0
if index is 1:
    
#for Er60 only: np.arange(9,12)
#for index in np.arange(0,3):

    print(index)
    

    file1    = h5py.File(name[index] + '.hdf5', 'r')  
    se1_dset   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
    blue1_dset  = file1['/data/Counter channel 2 : PMT blue/PMT blue/data']#50 frames x 200 tr pts x250 x 250 pixels
    red1_dset  = file1['/data/Counter channel 1 : PMT red/PMT red/data']
    
    red1_dset = np.array(red1_dset) 
    #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
    red1_dset = red1_dset/1.0e3
    red1_dset = np.array(red1_dset, dtype=np.float32)  #converting the CL dset to float16 from float64 is creating infinities! bc max value is > 2^16 
    
    blue1_dset = np.array(blue1_dset) 
    #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
    blue1_dset = blue1_dset/1.0e3
    blue1_dset = np.array(blue1_dset, dtype=np.float32)  #converting the CL dset to float16 from float64 is creating infinities! bc max value is > 2^16 
    
#    if index == 1:
#        file2    = h5py.File('2016-08-16-2004_ImageSequence_ZnO_2.000kX_3.000kV_30mu_2.hdf5', 'r')  
#        blue2_dset  = file1['/data/Counter channel 2 : PMT blue/PMT blue/data']#50 frames x 200 tr pts x250 x 250 pixels
#        red2_dset  = file1['/data/Counter channel 1 : PMT red/PMT red/data']
#    
#        red2_dset = np.array(red2_dset) 
#        #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
#        red2_dset = red2_dset/1.0e3
#        red2_dset = np.array(red2_dset, dtype=np.float32)  #converting the CL dset to float16 from float64 is creating infinities! bc max value is > 2^16 
#    
#        blue2_dset = np.array(blue2_dset) 
#        #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
#        blue2_dset = blue2_dset/1.0e3
#        blue2_dset = np.array(blue2_dset, dtype=np.float32)  #converting the CL dset to float16 from float64 is creating infinities! bc max value is > 2^16 
#        
#        red1_dset = np.append(red1_dset , red2_dset, axis = 0)
#        blue1_dset = np.append(blue1_dset , blue2_dset, axis = 0)
    
    print('data loaded')
    
    mycode = 'ZZZ' + namefile[index] + ' = NamedTemporaryFile(delete=False)'
    exec(mycode)
    np.savez('ZZZ' + namefile[index], datared = red1_dset, datablue = blue1_dset)
##END OF SECOND COLUMN    

ax1 = plt.subplot2grid((2,2), (1, 1), colspan=1)
bone = np.load('ZZZCheckBleach1.npz') #yap['datared'], yap['datablue']
btwo = np.load('ZZZCheckBleach2.npz')
#boneblue = np.append(bone['datablue'],btwo['datablue'])
#bonered = np.append(bone['datared'],btwo['datared'])
bthree = np.load('ZZZCheckBleach3.npz')
#ax1.set_title('(Blue) YAP decay with e-beam',fontsize=fsizepl)
nominal_time_on = 20#in microsec
no_points = 20
#a = np.array([bone['datablue']])
#a = np.append(a,btwo['datablue'])
plt.plot(np.arange(1,no_points+1)*nominal_time_on,np.average(bone['datablue'],axis=(1,2)),c='b',lw=3)
plt.plot(np.arange(1,no_points+1)*nominal_time_on,np.average(bone['datared'],axis=(1,2)),c='r',lw=3)
plt.plot(np.arange(no_points,40)*nominal_time_on,np.average(btwo['datablue'],axis=(1,2)),c='b',lw=3)
plt.plot(np.arange(no_points,40)*nominal_time_on,np.average(btwo['datared'],axis=(1,2)),c='r',lw=3)

nominal_time_on = 100#in microsec
plt.plot(np.arange(1,no_points+1)*nominal_time_on,np.average(bthree['datablue'],axis=(1,2)),c='b',lw=3)
plt.plot(np.arange(1,no_points+1)*nominal_time_on,np.average(bthree['datared'],axis=(1,2)),c='r',lw=3)

#plt.plot(np.arange(1,no_points+1)*nominal_time_on,np.average(yap['datablue']/1.0e3,axis=(0,1,2))*np.ones([no_points]),'k')
major_ticks = [500,1000,1500,2000]
#ax1.set_xticks(major_ticks) 
plt.xlim([20,2000])
plt.ylabel('CL counts, per pixel (kHz)',fontsize=fsizepl)
#plt.ylim([0.42,0.48])
plt.xlabel('Cumulative e-beam exposure ($\mu$s)',fontsize=fsizepl)

 
multipage('TausComparison.pdf',dpi=80)
    
