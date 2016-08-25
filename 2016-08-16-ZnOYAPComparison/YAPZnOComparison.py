import os
import sys
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../Python modules/") # necessary for the tex fonts
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

import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile
#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################
titulo =  '(3kV, 30$\mu$m aperture, 2kX or 56nm pixels, 0.1ms lag per pixel, blue/red $= </>$ 409nm) \n 1) One should not worry about `hill\' at reds for ZnO:Ga 2) ZnO:Ga is resistant to e-beam damage'


##FIRST COLUMN
name = ['2016-08-16-1943_ImageSequence_YAP_2.000kX_3.000kV_30mu_1', '2016-08-16-2009_ImageSequence_ZnO_2.021kX_3.000kV_30mu_4']
namefile = ['YAP1','ZnO1']


index = 0
if index is 1:
    
#for Er60 only: np.arange(9,12)
#for index in np.arange(0,2):

    print(index)

    file1    = h5py.File(name[index] + '.hdf5', 'r')  
    #titulo =  '(3kV, 30$\mu$m aperture, 2kX or 56nm pixels, 0.1ms lag per pixel) '
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
    
    print('data loaded')
    
    mycode = 'ZZZ' + namefile[index] + ' = tempfile.NamedTemporaryFile(delete=False)'
    exec(mycode)
    np.savez('ZZZ' + namefile[index], datared = np.average(red1_dset, axis = (0,1,2)), datablue = np.average(blue1_dset, axis = (0,1,2)))
##END OF FIRST COLUMN
    
##SECOND COLUMN
name = ['2016-08-16-1953_ImageSequence_YAP_2.000kX_3.000kV_30mu_3', '2016-08-16-2002_ImageSequence_ZnO_2.000kX_3.000kV_30mu_1', '2016-08-16-2006_ImageSequence_ZnO_2.000kX_3.000kV_30mu_3']
namefile = ['YAP2','ZnO2','ZnO2long']


index = 1
if index is 0:
    
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
    
    mycode = 'ZZZ' + namefile[index] + ' = tempfile.NamedTemporaryFile(delete=False)'
    exec(mycode)
    np.savez('ZZZ' + namefile[index], datared = red1_dset, datablue = blue1_dset)
##END OF SECOND COLUMN    

##THIRD COLUMN
name = ['2016-08-16-1951_ImageSequence_YAP_2.000kX_3.000kV_30mu_2','2016-08-16-2017_ImageSequence_ZnO_2.021kX_3.000kV_30mu_5']
namefile = ['YAP3','ZnO3']


index = 1
if index is 0:
    
#for Er60 only: np.arange(9,12)
#for index in np.arange(0,2):

    print(index)
    

    file1    = h5py.File(name[index] + '.hdf5', 'r')  
    titulo =  '(3kV, 30$\mu$m aperture, 2kX or 56nm pixels, 0.1ms lag per pixel, blue/red $= </>$ 409nm) \n 1) One should not worry about `hill\' at reds for ZnO:Ga 2) ZnO:Ga is resistant to e-beam damage'
    se1_dset   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
    blue1_dset  = file1['/data/Counter channel 2 : PMT blue/PMT blue time-resolved/data']#50 frames x 200 tr pts x250 x 250 pixels
    red1_dset  = file1['/data/Counter channel 1 : PMT red/PMT red time-resolved/data']
    
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
    
    mycode = 'ZZZ' + namefile[index] + ' = tempfile.NamedTemporaryFile(delete=False)'
    exec(mycode)
    np.savez('ZZZ' + namefile[index], datared = red1_dset, datablue = blue1_dset)
##END OF THIRD COLUMN    


fsizetit = 18 #22 #18
fsizepl = 16 #20 #16
sizex = 8 #10 #8
sizey = 6# 10 #6
dpi_no = 80
lw = 2

fig40= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig40.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    
plt.suptitle("YAP and ZnO:Ga comparison " + titulo,fontsize=fsizetit)

#######

ax1 = plt.subplot2grid((2,3), (0, 0), colspan=1)
yap = np.load('ZZZYAP1.npz') #yap['datared'], yap['datablue']
ratio = yap['datablue']/yap['datared']
ax1.set_title('YAP brightness, ratio blue/red = ' + str('{:0.1f}'.format(ratio)),fontsize=fsizepl)
dados = [yap['datared']/1.0e3 + yap['datablue']/1.0e3, yap['datablue']/1.0e3, yap['datared']/1.0e3]
plt.plot(dados,'ok',ls='None')
major_ticks = [0,1,2]
ax1.set_xticks(major_ticks) 
plt.xlim([-1,3])
labels = ['Total', 'Blue', 'Red']
plt.xticks(major_ticks, labels, rotation='horizontal')
plt.ylabel('CL counts, per pixel (MHz)')
plt.ylim([0,0.5])

ax1 = plt.subplot2grid((2,3), (1, 0), colspan=1)
zno = np.load('ZZZZnO1.npz') #yap['datared'], yap['datablue']
ratio = zno['datablue']/zno['datared']
ax1.set_title('ZnO:Ga brightness, ratio blue/red = ' + str('{:0.1f}'.format(ratio)),fontsize=fsizepl)
dados = [zno['datared']/1.0e3 + zno['datablue']/1.0e3, zno['datablue']/1.0e3, zno['datared']/1.0e3]
plt.plot(dados,'ok',ls='None')
major_ticks = [0,1,2]
ax1.set_xticks(major_ticks) 
plt.xlim([-1,3])
labels = ['Total', 'Blue', 'Red']
plt.xticks(major_ticks, labels, rotation='horizontal')
plt.ylabel('CL counts, per pixel (MHz)')
plt.ylim([0,0.5])


#######

ax1 = plt.subplot2grid((2,3), (0, 1), colspan=1)
yap = np.load('ZZZYAP2.npz') #yap['datared'], yap['datablue']
ax1.set_title('(Blue) YAP decay with e-beam',fontsize=fsizepl)
nominal_time_on = 6 #in microsec
no_points = 20
plt.plot(np.arange(1,no_points+1)*nominal_time_on,np.average(yap['datablue']/1.0e3,axis=(1,2)),c='b')
plt.plot(np.arange(1,no_points+1)*nominal_time_on,np.average(yap['datablue']/1.0e3,axis=(0,1,2))*np.ones([no_points]),'k')
major_ticks = [6,50,100]
ax1.set_xticks(major_ticks) 
plt.xlim([6,120])
plt.ylabel('CL counts, per pixel (MHz)')
plt.ylim([0.42,0.48])
plt.xlabel('Cumulative e-beam exposure ($\mu$s)')

ax1 = plt.subplot2grid((2,3), (1, 1), colspan=1)
yap = np.load('ZZZZnO2.npz') #yap['datared'], yap['datablue']
ax1.set_title('(Blue) ZnO:Ga decay with e-beam',fontsize=fsizepl)
nominal_time_on = 6 #in microsec
no_points = 20
plt.plot(np.arange(1,no_points+1)*nominal_time_on,np.average(yap['datablue']/1.0e3,axis=(1,2)),c='b')
plt.hold(True)
plt.plot(np.arange(1,no_points+1)*nominal_time_on,np.average(yap['datablue']/1.0e3,axis=(0,1,2))*np.ones([no_points]),'k')
yap = np.load('ZZZZnO2long.npz') #yap['datared'], yap['datablue']
nominal_time_on = 6 #in microsec
no_points = 100
plt.plot(np.arange(1,no_points+1)*nominal_time_on,np.average(yap['datablue']/1.0e3,axis=(1,2)),c='b')
plt.hold(True)
plt.plot(np.arange(1,no_points+1)*nominal_time_on,np.average(yap['datablue']/1.0e3,axis=(0,1,2))*np.ones([no_points]),'k')
major_ticks = [100,200,400,600] #[6,50,100]
ax1.set_xticks(major_ticks) 
plt.xlim([6,600])#([6,120])
plt.ylabel('CL counts, per pixel (MHz)')
plt.ylim([0.42,0.48])
plt.xlabel('Cumulative e-beam exposure ($\mu$s)')


#######

ax1 = plt.subplot2grid((2,3), (0, 2), colspan=1)
time_bin = 0.04
cutpointsatbeginning = 75
yap = np.load('ZZZYAP3.npz') #yap['datared'], yap['datablue']
ax1.set_title('(Blue) YAP decay upon near-impulsive excitation',fontsize=fsizepl)
plt.plot(np.arange(0,yap['datablue'].shape[1]-cutpointsatbeginning)*time_bin, np.average(yap['datablue']/1.0e3,axis=(0,2,3))[cutpointsatbeginning:])
ax1.axvspan(0.20, 0.32, alpha=0.25, color='red')
major_ticks = [1,2,3]
ax1.set_xticks(major_ticks) 
plt.xlim([0,3])
plt.ylabel('CL counts, per pixel (MHz)')
plt.ylim([0,0.4])
plt.xlabel('Behaviour of e-beam: 0.12 ON - 3 OFF ($\mu$s)')

ax1 = plt.subplot2grid((2,3), (1, 2), colspan=1)
time_bin = 0.04
cutpointsatbeginning = 75
zno = np.load('ZZZZnO3.npz') #yap['datared'], yap['datablue']
ax1.set_title('(Blue) ZnO:Ga decay upon near-impulsive excitation (IRF)',fontsize=fsizepl)
plt.plot(np.arange(0,zno['datablue'].shape[1]-cutpointsatbeginning)*time_bin, np.average(zno['datablue']/1.0e3,axis=(0,2,3))[cutpointsatbeginning:])
ax1.axvspan(0.20, 0.32, alpha=0.25, color='red')
major_ticks = [1,2,3]
ax1.set_xticks(major_ticks) 
plt.xlim([0,3])
plt.ylabel('CL counts, per pixel (MHz)')
plt.ylim([0,0.4])
plt.xlabel('Behaviour of e-beam: 0.12 ON - 3 OFF ($\mu$s)')

multipage('ComparisonYAPZnO.pdf',dpi=80)  

#######
#######

fig50= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig50.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    
plt.suptitle("YAP and ZnO:Ga decay upon near-impulsive excitation",fontsize=fsizetit)

#YAP decay
ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
ax1.set_title("YAP",fontsize=fsizepl)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
blue = np.average(yap['datablue'],axis=(0))/1.0e3 #MHz
last_pt_offset = -10 #sometimes use -1, last point, but sometimes this gives 0. -10 seems to work
Time_bin = 40 #ns
middlept = -30
initdecay = 83
init_guess = [np.average(blue[initdecay,:,:]), 2.0, np.average(blue[last_pt_offset,:,:]), np.average(blue[middlept,:,:]), 0.05] #e init was 0.5

init_guess = [0.1, 0.2, 0.001, 0.05, 0.005]

b,e,be,ee = calcdecay_subplot(blue[initdecay:,:,:], 
                              time_detail= Time_bin*1e-9,
                              titulo='Cathodoluminescence rate decay, bi-exponential fit, \n ',
                              single=False,
                              other_dset1=None,
                              other_dset2=None,
                              init_guess=init_guess,
                              unit='MHz')    
plt.xlim([0,2.5])
#major_ticks0 = [1,2]
plt.ylabel("Average luminescence \n of each time bin, per pixel (MHz)",fontsize=fsizepl)
#ax1.set_xticks(major_ticks0) 

ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
ax1.set_title("ZnO:Ga",fontsize=fsizepl)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
blue = np.average(zno['datablue'],axis=(0))/1.0e3 #MHz
last_pt_offset = -10 #sometimes use -1, last point, but sometimes this gives 0. -10 seems to work
Time_bin = 40 #ns
middlept = -30
initdecay = 83
init_guess = [np.average(blue[initdecay,:,:]), 2.0, np.average(blue[last_pt_offset,:,:]), np.average(blue[middlept,:,:]), 0.05] #e init was 0.5

init_guess = [0.005, 0.2, 0.001, 0.05, 0.005]

b,e,be,ee = calcdecay_subplot(blue[initdecay:,:,:], 
                              time_detail= Time_bin*1e-9,
                              titulo='Cathodoluminescence rate decay, bi-exponential fit, \n ',
                              single=False,
                              other_dset1=None,
                              other_dset2=None,
                              init_guess=init_guess,
                              unit='MHz')    
#plt.xlim([0,2])
#major_ticks0 = [1,2]
plt.ylabel("Average luminescence \n of each time bin, per pixel (MHz)",fontsize=fsizepl)
plt.xlim([0,2.5])
plt.show() 

    
multipage('ComparisonYAPZnO.pdf',dpi=80)  