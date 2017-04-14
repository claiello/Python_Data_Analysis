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

fsizetit = 36
fsizepl = 24
sizex = 8
sizey = 6
dpi_no = 80
lw = 2

#Grab info on mag, pixel size, aperture, kV from .txt file

#### TIP & TERRACES

name = ['2017-04-02-1548_ImageSequence__23.425kX_2.000kV_30mu_33',
        '2017-04-02-1553_ImageSequence__3.021kX_2.000kV_30mu_35',
        '2017-04-02-1558_ImageSequence__3.249kX_2.000kV_30mu_37',
        '2017-04-02-1601_ImageSequence__3.249kX_2.000kV_30mu_38',
        '2017-04-02-1505_ImageSequence__100.000kX_10.000kV_30mu_22',
        '2017-04-02-1508_ImageSequence__100.000kX_10.000kV_30mu_23',
        '2017-04-02-1511_ImageSequence__25.882kX_10.000kV_30mu_24'] 
        
#conditions
# 5 avg, 250mus pp, 500x500 pixels, 23kX, 2kV, no filter - TERRACE
# 5 avg, 250mus pp, 500x500 pixels, 23kX, 2kV, no filter - TERRACE
# x avg, xmus pp, 500x500 pixels, xkX, 2kV, no filter - TIP
# x avg, xmus pp, 500x500 pixels, xkX, 2kV, no filter - TIP
# 5 avg, 250mus pp, 500x500 pixels, xkX, 10kV, no filter - TIP
# 5 avg, 250mus pp, 500x500 pixels, xkX, 10kV, no filter - TIP
# 5 avg, 250mus pp, 500x500 pixels, xkX, 10kV, no filter - DISTORTED TIP
        
Pixel_size = [9.53, 5.91, 4.81, 2.06, 2.23, 2.23, 8.63] #nm

#for index in np.arange(0,len(name)):
if True is False:    
    print(index)

    file1    = h5py.File(name[index] + '.hdf5', 'r')  
    title =  'Mithrene CL 17/04/02'
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
    
    
        
#######
        
name2 = ['2017-04-02-1633_ImageSequence__49.483kX_2.000kV_30mu_46',
         '2017-04-02-1639_ImageSequence__50.000kX_1.950kV_30mu_47',
         '2017-04-02-1646_ImageSequence__50.000kX_2.000kV_30mu_48',
         '2017-04-02-1652_ImageSequence__50.000kX_2.050kV_30mu_49',
         '2017-04-02-1659_ImageSequence__50.000kX_2.000kV_30mu_50',
         '2017-04-02-1705_ImageSequence__50.000kX_1.900kV_30mu_51',
         '2017-04-02-1712_ImageSequence__50.000kX_2.000kV_30mu_52',
         '2017-04-02-1718_ImageSequence__50.000kX_2.100kV_30mu_53']

kv = [2,1.95, 2, 2.05, 2, 1.9, 2, 2.1]

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

if True is False:
    br = np.zeros([len(name2)])
    for index in np.arange(0,len(name2)):
        
        print(index)
        
        file1    = h5py.File(name2[index] + '.hdf5', 'r')  
        title =  'Mithrene CL 17/04/02'
        se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
        red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
        
        cutx = 0
        cutxend = 1
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
    ax1.plot(np.arange(1,9), br/br[0], ls='None', marker='o',markersize=12)
    ax1.set_xlabel('Electron energy (kV)',fontsize=fsizetit)
    ax1.set_ylabel('Avg CL, \n norm. (a.u.)',fontsize=fsizetit)
    ax1.set_title('Bleaching: \n in order of experiments',fontsize=fsizetit)
    ax1.set_xlim([0.5,8.5])
    ax1.set_xticks([1,2,3,4,5,6,7,8])
    ax1.set_xticklabels(['2','1.95','2','2.05','2','1.9','2','2.1'])
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
    ax2.plot(kv,brnew , ls='None', marker='o',markersize=12)
    ax2.set_xlabel('Electron energy (kV)',fontsize=fsizetit)
    ax2.set_ylabel('Avg CL, \n norm. to most recent 2kV scan (a.u.)',fontsize=fsizetit)
    #ax2.set_title('Changes in kV',fontsize=fsizetit)
    ax2.set_xlim([1.85,2.15])
    ax2.tick_params(labelsize=fsizepl)
    ax2.set_xticks([1.9, 1.95, 2, 2.05, 2.1])
    
    multipage_longer('LowkVScan.pdf',dpi=80)
    plt.show()
        
        
#######
         
#FILTERS - single platelet
#50 kX, 1 avg, 250mus pp, with filters, 10kV
        
name2 = ['2017-04-02-1433_ImageSequence__50.000kX_10.000kV_30mu_10',
         '2017-04-02-1435_ImageSequence__50.000kX_10.000kV_30mu_11',
         '2017-04-02-1437_ImageSequence__50.000kX_10.000kV_30mu_12',
         '2017-04-02-1440_ImageSequence__50.000kX_10.000kV_30mu_14',
         '2017-04-02-1442_ImageSequence__50.000kX_10.000kV_30mu_15']
         
filt = ['no', '472/30nm', '503/40nm', '400/40nm', '472/30nm']
         
# no filter / 472/30 / 503/40 / 400/40 / 472/30

#ax0 = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=1)
#ax1 = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=1)
#ax2 = plt.subplot2grid((2,3), (0,2), colspan=1, rowspan=1)
#ax2b = plt.subplot2grid((2,3), (1,0), colspan=1, rowspan=1)
#ax2c = plt.subplot2grid((2,3), (1,1), colspan=1, rowspan=1)

axvec = [ax0, ax1,ax2,ax2b, ax2c]

if True is False:
    br = np.zeros([len(name2)])
    for index in np.arange(0,len(name2)):
        
        print(index)
        
        file1    = h5py.File(name2[index] + '.hdf5', 'r')  
        title =  'Mithrene CL 17/04/02'
        se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
        red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
        
        cutx = 0
        cutxend = 1
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
        #axvec[index].set_title(str(filt[index]) + ' filter',fontsize=fsizetit)
        #axvec[index].axis('off')
        br[index] = np.nanmean(brightred,axis=(0,1))

#multipage_longer('ShowCutsFilterScanPlatelet.pdf',dpi=80)
#plt.show()
        
    print('filter scan')
    plt.figure(45)
    ax0 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
    ax0.plot(np.arange(1,6),br/br[0] , ls='None', marker='o',markersize=12)
    ax0.set_ylabel('Avg CL, norm. (a.u.)',fontsize=fsizetit)
    ax0.set_title('Filters',fontsize=fsizetit)
    ax0.set_xlim([0.5,5.5])
    ax0.tick_params(labelsize=fsizepl)
    ax0.set_ylim([0.1,1.05])
    ax0.set_xticks([1,2,3,4,5])  
    ax0.set_xticklabels(['no', '472/30nm', '503/40nm', '400/40nm', '472/30nm'])
    for label in ax0.get_xmajorticklabels() :
        label.set_rotation(30)
        label.set_horizontalalignment("right")
        
    ax1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
    ax1.plot(np.arange(1,5),br[1:]/br[1] , ls='None', marker='o',markersize=12)
    ax1.set_ylabel('Avg CL, norm. (a.u.)',fontsize=fsizetit)
    ax1.set_title('Filters',fontsize=fsizetit)
    ax1.set_xlim([0.5,4.5])
    ax1.tick_params(labelsize=fsizepl)
    #ax1.set_ylim([0.1,1.05])
    ax1.set_xticks([1,2,3,4])  
    ax1.set_xticklabels(['472/30nm', '503/40nm', '400/40nm', '472/30nm'])
    for label in ax1.get_xmajorticklabels() :
        label.set_rotation(30)
        label.set_horizontalalignment("right")
    
    multipage_longer('FilterScanPlatelet.pdf',dpi=80)
    plt.show()

#######    
#FILTERS - stringy stuff try #1
#60 kX, 10 avg, 100mus pp, with filters, 10kV
        
name2 = ['2017-04-02-1408_ImageSequence__60.002kX_10.000kV_30mu_5',
         '2017-04-02-1413_ImageSequence__60.002kX_10.000kV_30mu_6',
         '2017-04-02-1417_ImageSequence__60.002kX_10.000kV_30mu_7',
         '2017-04-02-1422_ImageSequence__60.002kX_10.000kV_30mu_8',
         '2017-04-02-1427_ImageSequence__60.002kX_10.000kV_30mu_9']
         
##### scan at 1402 is identical but at 70kX, not 60kX
         
filt = ['no', '472/30nm', '503/40nm', '400/40nm', '472/30nm']
         
# no filter / 472/30 / 503/40 / 400/40 / 472/30

ax0 = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=1)
ax1 = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((2,3), (0,2), colspan=1, rowspan=1)
ax2b = plt.subplot2grid((2,3), (1,0), colspan=1, rowspan=1)
ax2c = plt.subplot2grid((2,3), (1,1), colspan=1, rowspan=1)

axvec = [ax0, ax1,ax2,ax2b, ax2c]

if True is False:
    br = np.zeros([len(name2)])
    for index in np.arange(0,len(name2)):
        
        print(index)
        
        file1    = h5py.File(name2[index] + '.hdf5', 'r')  
        title =  'Mithrene CL 17/04/02'
        se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
        red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
        
        cutx = 0
        cutxend = 200
        cuty = 100
        cutyend = 1
        se = se[:,cutx:-cutxend,cuty:-cutyend]
        red = red[:,cutx:-cutxend,cuty:-cutyend]
        
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
#        axvec[index].imshow(brightred)
#        axvec[index].set_title(str(filt[index]) + ' filter',fontsize=fsizetit)
#        axvec[index].axis('off')
        br[index] = np.nanmean(brightred,axis=(0,1))

#multipage_longer('ShowCutsFilterScanWiresTry1.pdf',dpi=80)
#plt.show()
        
    print('filter scan')
    plt.figure(45)
    ax0 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
    ax0.plot(np.arange(1,6),br/br[0] , ls='None', marker='o',markersize=12)
    ax0.set_ylabel('Avg CL, norm. (a.u.)',fontsize=fsizetit)
    ax0.set_title('Filters',fontsize=fsizetit)
    ax0.set_xlim([0.5,5.5])
    ax0.tick_params(labelsize=fsizepl)
    ax0.set_ylim([0,1.05])
    ax0.set_xticks([1,2,3,4,5])  
    ax0.set_xticklabels(['no', '472/30nm', '503/40nm', '400/40nm', '472/30nm'])
    for label in ax0.get_xmajorticklabels() :
        label.set_rotation(30)
        label.set_horizontalalignment("right")
        
    ax1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
    ax1.plot(np.arange(1,5),br[1:]/br[1] , ls='None', marker='o',markersize=12)
    ax1.set_ylabel('Avg CL, norm. (a.u.)',fontsize=fsizetit)
    ax1.set_title('Filters',fontsize=fsizetit)
    ax1.set_xlim([0.5,4.5])
    ax1.tick_params(labelsize=fsizepl)
    #ax1.set_ylim([0.1,1.05])
    ax1.set_xticks([1,2,3,4])  
    ax1.set_xticklabels(['472/30nm', '503/40nm', '400/40nm', '472/30nm'])
    for label in ax1.get_xmajorticklabels() :
        label.set_rotation(30)
        label.set_horizontalalignment("right")
    
    multipage_longer('FilterScanWiresTry1.pdf',dpi=80)
    plt.show()

#######    

#######    
#FILTERS - stringy stuff try #2
#x kX, 5 avg, 100mus pp, with filters, 2kV
        
name2 = ['2017-04-02-1604_ImageSequence__3.249kX_2.000kV_30mu_39',
         '2017-04-02-1607_ImageSequence__3.249kX_2.000kV_30mu_40',
         '2017-04-02-1609_ImageSequence__3.249kX_2.000kV_30mu_41',
         '2017-04-02-1612_ImageSequence__3.249kX_2.000kV_30mu_42',
         '2017-04-02-1614_ImageSequence__3.249kX_2.000kV_30mu_43',
         '2017-04-02-1617_ImageSequence__3.249kX_2.000kV_30mu_44']
         
# Extra scan at 4:23, no filter, 5 avgs, at 250mus per pixel just for kicks
         
filt = ['no', '472/30nm', '503/40nm', '400/40nm', '472/30nm', 'no']
         
# no filter / 472/30 / 503/40 / 400/40 / 472/30

ax0 = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=1)
ax1 = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((2,3), (0,2), colspan=1, rowspan=1)
ax2b = plt.subplot2grid((2,3), (1,0), colspan=1, rowspan=1)
ax2c = plt.subplot2grid((2,3), (1,1), colspan=1, rowspan=1)
ax2d = plt.subplot2grid((2,3), (1,2), colspan=1, rowspan=1)

axvec = [ax0, ax1,ax2,ax2b, ax2c, ax2d]

if True is False:
    br = np.zeros([len(name2)])
    for index in np.arange(0,len(name2)):
        
        print(index)
        
        file1    = h5py.File(name2[index] + '.hdf5', 'r')  
        title =  'Mithrene CL 17/04/02'
        se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
        red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
        
        cutx = 0
        cutxend = 1 #200
        cuty = 0 #100
        cutyend = 1
        se = se[:,cutx:-cutxend,cuty:-cutyend]
        red = red[:,cutx:-cutxend,cuty:-cutyend]
        
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
        axvec[index].imshow(brightred)
        axvec[index].set_title(str(filt[index]) + ' filter',fontsize=fsizetit)
        axvec[index].axis('off')
        br[index] = np.nanmean(brightred,axis=(0,1))

#multipage_longer('ShowCutsFilterScanWiresTry2.pdf',dpi=80)
#plt.show()
        
    print('filter scan')
    plt.figure(45)
    ax0 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
    ax0.plot(np.arange(1,7),br/br[0] , ls='None', marker='o',markersize=12)
    ax0.set_ylabel('Avg CL, norm. (a.u.)',fontsize=fsizetit)
    ax0.set_title('Filters',fontsize=fsizetit)
    ax0.set_xlim([0.5,6.5])
    ax0.tick_params(labelsize=fsizepl)
    ax0.set_ylim([0,1.05])
    ax0.set_xticks([1,2,3,4,5,6])  
    ax0.set_xticklabels(['no', '472/30nm', '503/40nm', '400/40nm', '472/30nm', 'no'])
    for label in ax0.get_xmajorticklabels() :
        label.set_rotation(30)
        label.set_horizontalalignment("right")
        
    ax1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
    ax1.plot(np.arange(1,5),br[1:-1]/br[1] , ls='None', marker='o',markersize=12)
    ax1.set_ylabel('Avg CL, norm. (a.u.)',fontsize=fsizetit)
    ax1.set_title('Filters',fontsize=fsizetit)
    ax1.set_xlim([0.5,4.5])
    ax1.tick_params(labelsize=fsizepl)
    #ax1.set_ylim([0.1,1.05])
    ax1.set_xticks([1,2,3,4])  
    ax1.set_xticklabels(['472/30nm', '503/40nm', '400/40nm', '472/30nm'])
    for label in ax1.get_xmajorticklabels() :
        label.set_rotation(30)
        label.set_horizontalalignment("right")
    
    multipage_longer('FilterScanWiresTry2.pdf',dpi=80)
    plt.show()

#######    
#######    
#FILTERS - stringy stuff try #3
#50 kX, 3 avg, 250mus pp, with filters, 10kV
        
name2 = ['2017-04-02-1447_ImageSequence__50.000kX_10.000kV_30mu_16',
         '2017-04-02-1451_ImageSequence__50.000kX_10.000kV_30mu_17',
         '2017-04-02-1455_ImageSequence__50.000kX_10.000kV_30mu_18',
         '2017-04-02-1459_ImageSequence__50.000kX_10.000kV_30mu_19']
       
filt = ['472/30nm', '503/40nm', '400/40nm', '472/30nm']
         
# 472/30 / 503/40 / 400/40 / 472/30

ax0 = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=1)
ax1 = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((2,3), (0,2), colspan=1, rowspan=1)
ax2b = plt.subplot2grid((2,3), (1,0), colspan=1, rowspan=1)
ax2c = plt.subplot2grid((2,3), (1,1), colspan=1, rowspan=1)
ax2d = plt.subplot2grid((2,3), (1,2), colspan=1, rowspan=1)

axvec = [ax0, ax1,ax2,ax2b, ax2c, ax2d]

if True is False:
    br = np.zeros([len(name2)])
    for index in np.arange(0,len(name2)):
        
        print(index)
        
        file1    = h5py.File(name2[index] + '.hdf5', 'r')  
        title =  'Mithrene CL 17/04/02'
        se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
        red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
        
        cutx = 0
        cutxend = 1 #200
        cuty = 0 #100
        cutyend = 1
        se = se[:,cutx:-cutxend,cuty:-cutyend]
        red = red[:,cutx:-cutxend,cuty:-cutyend]
        
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
#        axvec[index].imshow(brightred)
#        axvec[index].set_title(str(filt[index]) + ' filter',fontsize=fsizetit)
#        axvec[index].axis('off')
        br[index] = np.nanmean(brightred,axis=(0,1))

#multipage_longer('ShowCutsFilterScanWiresTry3.pdf',dpi=80)
#plt.show()
        
    print('filter scan')
    plt.figure(45)
        
    ax1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
    ax1.plot(np.arange(1,5),br/br[0] , ls='None', marker='o',markersize=12)
    ax1.set_ylabel('Avg CL, norm. (a.u.)',fontsize=fsizetit)
    ax1.set_title('Filters',fontsize=fsizetit)
    ax1.set_xlim([0.5,4.5])
    ax1.tick_params(labelsize=fsizepl)
    #ax1.set_ylim([0.1,1.05])
    ax1.set_xticks([1,2,3,4])  
    ax1.set_xticklabels(['472/30nm', '503/40nm', '400/40nm', '472/30nm'])
    for label in ax1.get_xmajorticklabels() :
        label.set_rotation(30)
        label.set_horizontalalignment("right")
    
    multipage_longer('FilterScanWiresTry3.pdf',dpi=80)
    plt.show()

#######    
#######    
#FILTERS - stringy stuff try #4
#50 kX, 3 avg, 250mus pp, with filters, 10kV
##### REMOVING PLATELET FROM SIGNAL PIXELS
        
name2 = ['2017-04-02-1447_ImageSequence__50.000kX_10.000kV_30mu_16',
         '2017-04-02-1451_ImageSequence__50.000kX_10.000kV_30mu_17',
         '2017-04-02-1455_ImageSequence__50.000kX_10.000kV_30mu_18',
         '2017-04-02-1459_ImageSequence__50.000kX_10.000kV_30mu_19']
       
filt = ['472/30nm', '503/40nm', '400/40nm', '472/30nm']
         
# 472/30 / 503/40 / 400/40 / 472/30

ax0 = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=1)
ax1 = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=1)
ax2 = plt.subplot2grid((2,3), (0,2), colspan=1, rowspan=1)
ax2b = plt.subplot2grid((2,3), (1,0), colspan=1, rowspan=1)
ax2c = plt.subplot2grid((2,3), (1,1), colspan=1, rowspan=1)
ax2d = plt.subplot2grid((2,3), (1,2), colspan=1, rowspan=1)

axvec = [ax0, ax1,ax2,ax2b, ax2c, ax2d]

if True: # is False:
    br = np.zeros([len(name2)])
    for index in np.arange(0,len(name2)):
        
        print(index)
        
        file1    = h5py.File(name2[index] + '.hdf5', 'r')  
        title =  'Mithrene CL 17/04/02'
        se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
        red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
        
        cutx = 50
        cutxend = 80
        cuty = 140 #100
        cutyend = 80
        se = se[:,cutx:-cutxend,cuty:-cutyend]
        red = red[:,cutx:-cutxend,cuty:-cutyend]
        
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
#        axvec[index].imshow(brightred)
#        axvec[index].set_title(str(filt[index]) + ' filter',fontsize=fsizetit)
#        axvec[index].axis('off')
        br[index] = np.nanmean(brightred,axis=(0,1))

#multipage_longer('ShowCutsFilterScanWiresTry4.pdf',dpi=80)
#plt.show()
        
    print('filter scan')
    plt.figure(45)
        
    ax1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
    ax1.plot(np.arange(1,5),br/br[0] , ls='None', marker='o',markersize=12)
    ax1.set_ylabel('Avg CL, norm. (a.u.)',fontsize=fsizetit)
    ax1.set_title('Filters',fontsize=fsizetit)
    ax1.set_xlim([0.5,4.5])
    ax1.set_ylim([0.8,1.17])
    ax1.tick_params(labelsize=fsizepl)
    #ax1.set_ylim([0.1,1.05])
    ax1.set_xticks([1,2,3,4])  
    ax1.set_xticklabels(['472/30nm', '503/40nm', '400/40nm', '472/30nm'])
    for label in ax1.get_xmajorticklabels() :
        label.set_rotation(30)
        label.set_horizontalalignment("right")
    
    multipage_longer('FilterScanWiresTry4.pdf',dpi=80)
    plt.show()

#######    


