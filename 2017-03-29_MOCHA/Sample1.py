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
#from tifffile import *
from sklearn.mixture import GMM 
import matplotlib.cm as cm
#from FluoDecay import *
from PlottingFcts import *
sys.path.append("/usr/bin") # necessary for the tex fonts
import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile
import scalebars as sb
import scipy.ndimage    
from lmfit import *

fsizetit = 36
fsizepl = 24
sizex = 8
sizey = 6
dpi_no = 80
lw = 2

#Grab info on mag, pixel size, aperture, kV from .txt file
### can't make textcolor to work

name = ['2017-03-29-1407_ImageSequence__10.000kX_2.000kV_30mu_1',
        '2017-03-29-1409_ImageSequence__25.000kX_2.000kV_30mu_2',
        '2017-03-29-1426_ImageSequence__50.000kX_2.000kV_30mu_3'] #10avgs
        
Pixel_size = [21.8, 8.7, 4.4] #nm

#for index in [2]: #np.arange(0,len(name)):
if True is False:    
    print(index)

    file1    = h5py.File(name[index] + '.hdf5', 'r')  
    title =  'Mithrene CL 17/03/29'
    se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
    red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
    
    se = np.array(se)
    red = np.array(red)
    
    #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
    red = red/1.0e3
    unit = '(kHz)'

    length_scalebar = 1000.0 #in nm (1000nm == 1mum)
    scalebar_legend = '1 $\mu$m'
    
    reg_se, reg_red, reg_se_all, reg_red_all = reg_images(se,red)
    
    plot_2_channels(reg_se, reg_red, Pixel_size[index], title, length_scalebar, scalebar_legend,'kHz',work_red_channel=True)
    
    if index == 2: #calculate bleaching
        arru =  np.average(reg_red_all,axis=(1,2))
        print('bleaching')
        plt.figure(45)
        plt.plot(np.arange(1,reg_red_all.shape[0]+1), arru/arru[0], ls='None', marker='o',markersize=12)
        plt.xlabel('Average number',fontsize=fsizetit)
        plt.ylabel('Average CL intensity, norm. (a.u.)',fontsize=fsizetit)
        plt.title('Bleaching',fontsize=fsizetit)
        plt.xlim([0.5,10.5])
        plt.tick_params(labelsize=fsizepl)
        multipage_longer('Bleaching.pdf',dpi=80)
        plt.show()
        
#######
#cut thru single platelet
def moving_average(a,n=3):
    vec = np.cumsum(a)
    vec[n:] = vec[n:] - vec[:-n]
    return (1/n)*vec[n-1:]
    
def gaus(x,a,x0,sigma, c):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + c
    
def gauss_fit(x, x0, y):

    # define objective function: returns the array to be minimized
    def fcn2min(params, x, data):
        """ model decaying sine wave, subtract data"""
        a = params['a'].value
        sigma = params['sigma'].value
        c = params['c'].value

        model = gaus(x,a,x0,sigma, c)
    
        return model - data

    # create a set of Parameters
    params = Parameters()
    params.add('a', value = 2, vary = True)
    params.add('sigma', value = 100, vary = True)
    params.add('c', value = 5, vary = True)

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min, params, fcn_args=(x, y))
    result = minner.minimize()

    #print(result.params)

    return (result.params['a'], result.params['sigma'], result.params['c'], result)
    
    
name = ['2017-03-29-1426_ImageSequence__50.000kX_2.000kV_30mu_3'] #10avgs
        
Pixel_size = [ 4.4] #nm

for index in np.arange(0,len(name)):
#if True is False:    
    print(index)

    file1    = h5py.File(name[index] + '.hdf5', 'r')  
    title =  'Mithrene CL 17/03/29'
    se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
    red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
    
    se = np.array(se)
    red = np.array(red)
    
    #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
    red = red/1.0e3
    unit = '(kHz)'

    length_scalebar = 500.0 #in nm (1000nm == 1mum)
    scalebar_legend = '500 nm'
    
    reg_se, reg_red, reg_se_all, reg_red_all = reg_images(se,red)
    
    plot_2_channels(reg_se, reg_red, Pixel_size[index], title, length_scalebar, scalebar_legend,'kHz',work_red_channel=True)
    
    #-- Extract the line...
    # Make a line with "num" points...
    x0, y0 = 100, 150 # These are in _pixel_ coordinates!!
    x1, y1 = 350, 400
    num = 1000
    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    
    # Extract the values along the line, using cubic interpolation
    
    zi = scipy.ndimage.map_coordinates(reg_red, np.vstack((x,y)))
    mov_avg_index = 30
    zi = moving_average(zi,n=mov_avg_index)
    
    #-- Plot...
    fig, axes = plt.subplots(nrows=2)
    axes[0].imshow(reg_red,cmap='Blues')
    axes[0].plot([x0, x1], [y0, y1], 'ro-')
    axes[0].axis('image')
    
    axes[1].plot(Pixel_size*np.arange(1,len(zi)+1),zi)
    axes[1].set_xlabel('nm')
    axes[1].set_ylabel('CL counts (kHz)')
    
    startz = 825
    endz = 925
    y = zi[startz:endz]
    x = np.arange(startz,endz)*Pixel_size
    n = len(x)                          #the number of data
    mean = np.mean(x) #sum(x*y)/n                   #note this correction
    sigma = sum(y*(x-mean)**2)/n        #note this correction
    a,sigma,c,result = gauss_fit(x,mean,y)
    axes[1].plot(x,gaus(x,a,mean,sigma, c),'ro:')
    
    multipage_longer('SingleCut.pdf',dpi=80)
    plt.show()

        
#######
klklklkl        
        
        
name2 = ['2017-03-29-1433_ImageSequence__50.000kX_2.000kV_30mu_4',
         '2017-03-29-1440_ImageSequence__50.000kX_2.050kV_30mu_5',
         '2017-03-29-1447_ImageSequence__50.000kX_2.000kV_30mu_6',
         '2017-03-29-1454_ImageSequence__50.000kX_1.950kV_30mu_7',
         '2017-03-29-1500_ImageSequence__50.000kX_2.000kV_30mu_8',
         '2017-03-29-1510_ImageSequence__50.000kX_2.100kV_30mu_9',
         '2017-03-29-1516_ImageSequence__50.000kX_2.000kV_30mu_10',
         '2017-03-29-1523_ImageSequence__50.000kX_1.900kV_30mu_11',
         '2017-03-29-1530_ImageSequence__50.000kX_2.000kV_30mu_12']

kv = [2,2.05, 2,1.95,2,2.1,2,1.9,2]

#ax0 = plt.subplot2grid((2,5), (0,0), colspan=1, rowspan=1)
#ax1 = plt.subplot2grid((2,5), (0,1), colspan=1, rowspan=1)
#ax2 = plt.subplot2grid((2,5), (0,2), colspan=1, rowspan=1)
#ax2b = plt.subplot2grid((2,5), (0,3), colspan=1, rowspan=1)
#ax2c = plt.subplot2grid((2,5), (0,4), colspan=1, rowspan=1)
#ax3= plt.subplot2grid((2,5), (1,0), colspan=1, rowspan=1)
#ax4 = plt.subplot2grid((2,5), (1,1), colspan=1, rowspan=1)
#ax5 = plt.subplot2grid((2,5), (1,2), colspan=1, rowspan=1)
#ax5b = plt.subplot2grid((2,5), (1,3), colspan=1, rowspan=1)
#
#axvec = [ax0, ax1,ax2,ax2b, ax2c, ax3,ax4,ax5,ax5b]

if True is False:
    br = np.zeros([len(name2)])
    for index in np.arange(0,len(name2)):
        
        print(index)
        
        file1    = h5py.File(name2[index] + '.hdf5', 'r')  
        title =  'Mithrene CL 17/03/29'
        se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
        red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
        
        cutx = 15
        cutxend = 25
        se = se[:,cutx:-cutxend,:]
        red = red[:,cutx:-cutxend,:]
        
        se = np.array(se)
        red = np.array(red)
        
        #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
        red = red/1.0e3
        unit = '(kHz)'
        
        #register
        #reg_se, reg_red, reg_se_all, reg_red_all = reg_images(se,red)
        #mock
        reg_se = np.average(se, axis = 0)
        reg_red = np.average(red, axis = 0)
        reg_se_all = se
        reg_red_all = red
        
        darkred, brightred, darkonese, brightse = gmmone(reg_se, reg_red)
        #axvec[index].imshow(brightred)
        #axvec[index].set_title(str(kv[index]) + ' kV',fontsize=fsizetit)
        #axvec[index].axis('off')
        br[index] = np.nanmean(brightred,axis=(0,1))
        
    print('low kv scan')
    plt.figure(45)
    ax0 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
    ax1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
    ax0.plot(kv,br/br[0] , ls='None', marker='o',markersize=12)
    #ax0.set_xlabel('Electron energy (kV)',fontsize=fsizetit)
    ax0.set_ylabel('Avg CL, norm. (a.u.)',fontsize=fsizetit)
    ax0.set_title('Changes in kV',fontsize=fsizetit)
    ax0.set_xlim([1.85,2.15])
    ax0.tick_params(labelsize=fsizepl)
    ax1.plot(np.arange(1,10), br/br[0], ls='None', marker='o',markersize=12)
    ax1.set_xlabel('Electron energy (kV)',fontsize=fsizetit)
    ax1.set_ylabel('Avg CL, \n norm. (a.u.)',fontsize=fsizetit)
    ax1.set_title('Bleaching: \n in order of experiments',fontsize=fsizetit)
    ax1.set_xlim([0.5,9.5])
    ax1.set_xticks([1,2,3,4,5,6,7,8,9])
    ax1.set_xticklabels(['2','2.05','2','1.95','2','2.1','2','1.9','2'])
    ax1.tick_params(labelsize=fsizepl)
    ax0.set_ylim([0.5,1.05])
    ax1.set_ylim([0.5,1.05])
    ax0.set_xticks([1.9, 1.95, 2, 2.05, 2.1])
    
    
    plt.figure(46)    
    ax2 = plt.subplot2grid((1,2), (0,0), colspan=1, rowspan=1)
    #ax3 = plt.subplot2grid((2,2), (1,1), colspan=1, rowspan=1)
    brnew = np.copy(br)
    brnew[0] = 1.0
    brnew[1] = br[1]/br[0]
    brnew[2] = 1.0
    brnew[3] = br[3]/br[2]
    brnew[4] = 1.0
    brnew[5] = br[5]/br[4]
    brnew[6] = 1.0
    brnew[7] = br[7]/br[6]
    brnew[8] = 1.0
    ax2.plot(kv,brnew , ls='None', marker='o',markersize=12)
    ax2.set_xlabel('Electron energy (kV)',fontsize=fsizetit)
    ax2.set_ylabel('Avg CL, \n norm. to most recent 2kV scan (a.u.)',fontsize=fsizetit)
    #ax2.set_title('Changes in kV',fontsize=fsizetit)
    ax2.set_xlim([1.85,2.15])
    ax2.tick_params(labelsize=fsizepl)
#    ax3.plot(np.arange(1,10), brnew, ls='None', marker='o',markersize=12)
#    ax3.set_xlabel('Electron energy (kV)',fontsize=fsizetit)
#    ax3.set_ylabel('Average CL intensity, norm. (a.u.)',fontsize=fsizetit)
#    ax3.set_title('Bleaching: in order of experiments',fontsize=fsizetit)
#    ax3.set_xlim([0.5,9.5])
#    ax3.set_xticks([1,2,3,4,5,6,7,8,9])
#    ax3.set_xticklabels(['2','2.05','2','1.95','2','2.1','2','1.9','2'])
#    ax3.tick_params(labelsize=fsizepl)
    #ax2.set_ylim([0.5,1.05])
    #ax3.set_ylim([0.5,1.05])
    ax2.set_xticks([1.9, 1.95, 2, 2.05, 2.1])
    
    multipage_longer('LowkVScan.pdf',dpi=80)
    plt.show()
    
if True is False:
    Pixel_size = 21.8
    name = '2017-03-29-1536_ImageSequence__10.000kX_2.000kV_30mu_13'
    
    file1    = h5py.File(name + '.hdf5', 'r')  
    title =  'Mithrene CL 17/03/29'
    se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
    red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels

    se = np.array(se)
    red = np.array(red)
    
    #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
    red = red/1.0e3
    unit = '(kHz)'

    length_scalebar = 1000.0 #in nm (1000nm == 1mum)
    scalebar_legend = '1 $\mu$m'
    
    reg_se, reg_red, reg_se_all, reg_red_all = reg_images(se,red)
    
    plot_2_channels(reg_se, reg_red, Pixel_size, title, length_scalebar, scalebar_legend,'kHz',work_red_channel=True)

##### HIGH KV SCAN

#ax0 = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=1)
#ax1 = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=1)
#ax2 = plt.subplot2grid((2,3), (0,2), colspan=1, rowspan=1)
#ax2b = plt.subplot2grid((2,3), (1,0), colspan=1, rowspan=1)
#ax2c = plt.subplot2grid((2,3), (1,1), colspan=1, rowspan=1)
#ax3= plt.subplot2grid((2,3), (1,2), colspan=1, rowspan=1)

#axvec = [ax0, ax1,ax2,ax2b, ax2c, ax3]

#dESPITE WHAT IT SAYS, IT IS NOOOOOOOT THE METAL SAMPLE
name3 = ['2017-03-29-1805_ImageSequence_METALsample_10.000kX_2.000kV_30mu_17',
         '2017-03-29-1813_ImageSequence_METALsample_10.000kX_5.000kV_30mu_18',
         '2017-03-29-1820_ImageSequence_METALsample_10.000kX_10.000kV_30mu_19',
         '2017-03-29-1827_ImageSequence_METALsample_10.000kX_15.000kV_30mu_20',
         '2017-03-29-1834_ImageSequence_METALsample_10.000kX_20.000kV_30mu_21',
         '2017-03-29-1843_ImageSequence_METALsample_10.000kX_2.000kV_30mu_22']
kv3 = [2,5,10,15,20,2]

if True is False:
    
    br = np.zeros([len(name3)])
    brframe = np.zeros([len(name3),5]) #5 averages
    for index in np.arange(0,len(name3)):
        
        print(index)
        
        file1    = h5py.File(name3[index] + '.hdf5', 'r')  
        title =  'Mithrene CL 17/03/29'
        se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
        red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
        
        se = np.array(se)
        red = np.array(red)
        
        #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
        red = red/1.0e3
        unit = '(kHz)'
        
        #register
        #reg_se, reg_red, reg_se_all, reg_red_all = reg_images(se,red)
        #mock
        reg_se = np.average(se, axis = 0)
        reg_red = np.average(red, axis = 0)
        reg_se_all = se
        reg_red_all = red
        
#        axvec[index].imshow(reg_se,cmap='Greys')
#        axvec[index].set_title(str(kv3[index]) + ' kV',fontsize=fsizetit)
#        axvec[index].axis('off')
        br[index] = np.average(reg_red,axis=(0,1))
        brframe[index,:] = np.average(reg_red_all,axis=(1,2))
        
    print('high kv scan')
    plt.figure(45)
    ax0 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
    ax1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
    ax0.plot(kv3[:-1],br[:-1]/(br[0]) , ls='None', marker='o',markersize=12)
    ax0.set_xlabel('Electron energy (kV)',fontsize=fsizetit)
    ax0.set_ylabel('Avg CL, norm. (a.u.)',fontsize=fsizetit)
    ax0.set_title('Changes in kV',fontsize=fsizetit)
    ax0.set_xlim([1.5,20.5])
    ax0.tick_params(labelsize=fsizepl)
    ax0.set_xticks([2, 5, 10, 15, 20])
    
    ax1.plot(np.arange(1,6),brframe[0,:]/brframe[0,0],linewidth=1,label='2kV')
    ax1.plot(np.arange(1,6),brframe[1,:]/brframe[1,0],linewidth=3,label='5kV')
    ax1.plot(np.arange(1,6),brframe[2,:]/brframe[2,0],linewidth=5,label='10kV')
    ax1.plot(np.arange(1,6),brframe[3,:]/brframe[3,0],linewidth=7,label='15kV')
    ax1.plot(np.arange(1,6),brframe[4,:]/brframe[4,0],linewidth=9,label='20kV')
    ax1.plot(np.arange(1,6),brframe[5,:]/brframe[5,0],linewidth=1,label='2kV')
    ax1.set_xlabel('Averages',fontsize=fsizetit)
    ax1.set_ylabel('Avg CL, norm. (a.u.)',fontsize=fsizetit)
    ax1.set_title('Bleaching',fontsize=fsizetit)
    ax1.set_xlim([0.5,5.5])
    ax1.tick_params(labelsize=fsizepl)
    ax1.set_xticks([1,2,3,4,5])
    plt.legend(loc='best')
    
    
    multipage_longer('HighkVScan.pdf',dpi=80)
    plt.show()

#multipage_longer('SignalSetsHighkV.pdf',dpi=80)

Pixel_size = [9.9, 3.6, 2.2] #nm
name5 = ['2017-03-29-1549_ImageSequence__22.500kX_2.000kV_30mu_15',
         '2017-03-29-1555_ImageSequence__22.500kX_2.000kV_30mu_16',
         '2017-03-29-1605_ImageSequence__100.000kX_2.000kV_30mu_17']

#for index in np.arange(0,len(name5)):
if True is False:    
    print(index)

    file1    = h5py.File(name5[index] + '.hdf5', 'r')  
    title =  'Mithrene CL 17/03/29'
    se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
    red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
    
    se = np.array(se)
    red = np.array(red)
    
    #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
    red = red/1.0e3
    unit = '(kHz)'

    length_scalebar = 1000.0 #in nm (1000nm == 1mum)
    scalebar_legend = '1 $\mu$m'
    
    reg_se, reg_red, reg_se_all, reg_red_all = reg_images(se,red)
    
    plot_2_channels(reg_se, reg_red, Pixel_size[index], title, length_scalebar, scalebar_legend,'kHz',work_red_channel=True)
    
    
#### METAL SAMPLE
Pixel_size = [8.8, 11.2, 22.3, 2.2] #nm
name6 = ['2017-03-29-1615_ImageSequence_METALsample_8.597kX_2.000kV_30mu_3',
         '2017-03-29-1625_ImageSequence_METALsample_20.000kX_2.000kV_30mu_4',
         '2017-03-29-1742_ImageSequence_METALsample_10.000kX_2.000kV_30mu_15',
         '2017-03-29-1751_ImageSequence_METALsample_100.000kX_2.000kV_30mu_16']

if True is False:   
#for index in np.arange(0,len(name6)):
    
    print(index)

    file1    = h5py.File(name6[index] + '.hdf5', 'r')  
    title =  'Mithrene CL 17/03/29'
    if index == 3:
        se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
    else:
        se   = file1['/data/Analog channel 2 : InLens/data'] #50 frames x250 x 250 pixels
    red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
    
    se = np.array(se)
    red = np.array(red)
    
    #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
    red = red/1.0e3
    unit = '(kHz)'

    length_scalebar = 1000.0 #in nm (1000nm == 1mum)
    scalebar_legend = '1 $\mu$m'
    
    #reg_se, reg_red, reg_se_all, reg_red_all = reg_images(se,red)
    #mock
    reg_se = np.average(se, axis = 0)
    reg_red = np.average(red, axis = 0)
    reg_se_all = se
    reg_red_all = red
    
    plot_2_channels(reg_se, reg_red, Pixel_size[index], title, length_scalebar, scalebar_legend,'kHz',work_red_channel=True)
    

#### BLEACH METAL
name20 = ['2017-03-29-1641_ImageSequence_METALsample_50.000kX_2.000kV_30mu_6',
          '2017-03-29-1648_ImageSequence_METALsample_50.000kX_2.050kV_30mu_7',
          '2017-03-29-1654_ImageSequence_METALsample_50.000kX_2.000kV_30mu_8',
          '2017-03-29-1701_ImageSequence_METALsample_50.000kX_1.950kV_30mu_9',
          '2017-03-29-1708_ImageSequence_METALsample_50.000kX_2.000kV_30mu_10',
          '2017-03-29-1715_ImageSequence_METALsample_50.000kX_2.100kV_30mu_11',
          '2017-03-29-1722_ImageSequence_METALsample_50.000kX_2.000kV_30mu_12',
          '2017-03-29-1729_ImageSequence_METALsample_50.000kX_1.900kV_30mu_13',
          '2017-03-29-1736_ImageSequence_METALsample_50.000kX_2.000kV_30mu_14']

kv = [2,2.05, 2,1.95,2,2.1,2,1.9,2]

plt.figure(67)
ax0 = plt.subplot2grid((2,5), (0,0), colspan=1, rowspan=1)
ax1 = plt.subplot2grid((2,5), (0,1), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((2,5), (0,2), colspan=1, rowspan=1)
ax2b = plt.subplot2grid((2,5), (0,3), colspan=1, rowspan=1)
ax2c = plt.subplot2grid((2,5), (0,4), colspan=1, rowspan=1)
ax3= plt.subplot2grid((2,5), (1,0), colspan=1, rowspan=1)
ax4 = plt.subplot2grid((2,5), (1,1), colspan=1, rowspan=1)
ax5 = plt.subplot2grid((2,5), (1,2), colspan=1, rowspan=1)
ax5b = plt.subplot2grid((2,5), (1,3), colspan=1, rowspan=1)

axvec = [ax0, ax1,ax2,ax2b, ax2c, ax3,ax4,ax5,ax5b]

if True:# is False:
    br = np.zeros([len(name20)])
    for index in np.arange(0,len(name20)):
        
        print(index)
        
        file1    = h5py.File(name20[index] + '.hdf5', 'r')  
        title =  'Mithrene CL 17/03/29'
        se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
        red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
        
        cutx = 35
        cutxend = 80
        se = se[:,cutx:-cutxend,:]
        red = red[:,cutx:-cutxend,:]
        
        se = np.array(se)
        red = np.array(red)
        
        #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
        red = red/1.0e3
        unit = '(kHz)'
        
        #register
        reg_se, reg_red, reg_se_all, reg_red_all = reg_images(se,red)
        
        darkred, brightred, darkonese, brightse = gmmone(reg_se, reg_red)
#        axvec[index].imshow(brightred)
#        axvec[index].set_title(str(kv[index]) + ' kV',fontsize=fsizetit)
#        axvec[index].axis('off')
        br[index] = np.nanmean(brightred,axis=(0,1))
        
#print('here')
#multipage_longer('SignalSetLowkVMetal.pdf',dpi=80)
#plt.show()
        
    print('low kv scan')
    plt.figure(45)
    ax0 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
    ax1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
    ax0.plot(kv,br/br[0] , ls='None', marker='o',markersize=12)
    #ax0.set_xlabel('Electron energy (kV)',fontsize=fsizetit)
    ax0.set_ylabel('Avg CL, norm. (a.u.)',fontsize=fsizetit)
    ax0.set_title('Changes in kV',fontsize=fsizetit)
    ax0.set_xlim([1.85,2.15])
    ax0.tick_params(labelsize=fsizepl)
    ax1.plot(np.arange(1,10), br/br[0], ls='None', marker='o',markersize=12)
    ax1.set_xlabel('Electron energy (kV)',fontsize=fsizetit)
    ax1.set_ylabel('Avg CL, \n norm. (a.u.)',fontsize=fsizetit)
    ax1.set_title('Bleaching: \n in order of experiments',fontsize=fsizetit)
    ax1.set_xlim([0.5,9.5])
    ax1.set_xticks([1,2,3,4,5,6,7,8,9])
    ax1.set_xticklabels(['2','2.05','2','1.95','2','2.1','2','1.9','2'])
    ax1.tick_params(labelsize=fsizepl)
    ax0.set_ylim([0.5,1.05])
    ax1.set_ylim([0.5,1.05])
    ax0.set_xticks([1.9, 1.95, 2, 2.05, 2.1])
    
    
    plt.figure(46)    
    ax2 = plt.subplot2grid((1,2), (0,0), colspan=1, rowspan=1)
    #ax3 = plt.subplot2grid((2,2), (1,1), colspan=1, rowspan=1)
    brnew = np.copy(br)
    brnew[0] = 1.0
    brnew[1] = br[1]/br[0]
    brnew[2] = 1.0
    brnew[3] = br[3]/br[2]
    brnew[4] = 1.0
    brnew[5] = br[5]/br[4]
    brnew[6] = 1.0
    brnew[7] = br[7]/br[6]
    brnew[8] = 1.0
    ax2.plot(kv,brnew , ls='None', marker='o',markersize=12)
    ax2.set_xlabel('Electron energy (kV)',fontsize=fsizetit)
    ax2.set_ylabel('Avg CL, \n norm. to most recent 2kV scan (a.u.)',fontsize=fsizetit)
    #ax2.set_title('Changes in kV',fontsize=fsizetit)
    ax2.set_xlim([1.85,2.15])
    ax2.tick_params(labelsize=fsizepl)
#    ax3.plot(np.arange(1,10), brnew, ls='None', marker='o',markersize=12)
#    ax3.set_xlabel('Electron energy (kV)',fontsize=fsizetit)
#    ax3.set_ylabel('Average CL intensity, norm. (a.u.)',fontsize=fsizetit)
#    ax3.set_title('Bleaching: in order of experiments',fontsize=fsizetit)
#    ax3.set_xlim([0.5,9.5])
#    ax3.set_xticks([1,2,3,4,5,6,7,8,9])
#    ax3.set_xticklabels(['2','2.05','2','1.95','2','2.1','2','1.9','2'])
#    ax3.tick_params(labelsize=fsizepl)
    #ax2.set_ylim([0.5,1.05])
    #ax3.set_ylim([0.5,1.05])
    ax2.set_xticks([1.9, 1.95, 2, 2.05, 2.1])
    
    multipage_longer('LowkVScanMetal.pdf',dpi=80)
    plt.show()
    
   




