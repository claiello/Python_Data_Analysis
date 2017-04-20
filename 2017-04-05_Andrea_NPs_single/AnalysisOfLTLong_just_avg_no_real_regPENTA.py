
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

import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile
import scalebars as sb

def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.get_memory_info()[0] / float(2 ** 20)
    return mem

#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################

Time_bin = 1000#in ns; 
nominal_time_on = 150.0 #time during which e-beam nominally on, in mus
totalpoints = 1400 #total number of time-resolved points
    
No_experiments = 5*np.ones([3])
                  
nametr = ['2017-04-05-1544_ImageSequence__500.000kX_10.000kV_30mu_13',
          '2017-04-05-1602_ImageSequence__500.000kX_10.000kV_30mu_14',
          '2017-04-05-1621_ImageSequence__500.000kX_10.000kV_30mu_15']

description = ['Andrea small NaYF4:Er'] # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
#nominal Temps
let = ['Try1','Try2','Try3']

#taken in the order above

index = 0
#for index in np.arange(0,len(nametr)):
if True:
    
    ###### put all dsets together
    redall = np.empty([])
    
    
    file2    = h5py.File(nametr[0] + '.hdf5', mmap_mode='r')  
    gc.collect()
    #red1_dset0  = file2['/data/Counter channel 1 : PMT red/PMT red time-resolved/data']#10 Scany points X10 frames x 150 tr pts x250 x 250 pixels
    red1_dset0  = file2['/data/Counter channel 2 : PMT blue/PMT blue time-resolved/data']#
    
    #    cutiny = 0
    #redall = red1_dset0[1:,:,:,cutiny:]
    
    redall = red1_dset0[1:,:,:,:]
    del red1_dset0
    gc.collect()
    file2.close()
    gc.collect()
    print('1 done')
    
    file3    = h5py.File(nametr[1] + '.hdf5', mmap_mode='r')  
    gc.collect()
    #red2_dset0  = file3['/data/Counter channel 1 : PMT red/PMT red time-resolved/data']#10 Scany points X10 frames x 150 tr pts x250 x 250 pixels
    red2_dset0  = file3['/data/Counter channel 2 : PMT blue/PMT blue time-resolved/data']#10 Scan
    
    
#    cutiny = 0
#    redall = np.vstack((redall,red2_dset0[1:,:,:,cutiny:])) 
    
    
    redall = np.vstack((redall,red2_dset0[1:,:,:,:])) 
    
    
    del red2_dset0
    gc.collect()
    file3.close()
    gc.collect()
    print('2 done')
    
    #file4    = h5py.File(nametr[2] + '.hdf5', mmap_mode='r')  
    #gc.collect()
    #red1_dset0  = file2['/data/Counter channel 1 : PMT red/PMT red time-resolved/data']#10 Scany points X10 frames x 150 tr pts x250 x 250 pixels
    #red3_dset0  = file4['/data/Counter channel 2 : PMT blue/PMT blue time-resolved/data']#10 
    
    #aux0 = red3_dset0[0,:,:,cutiny:].reshape([1,red3_dset0.shape[1],red3_dset0.shape[2],redall.shape[3]])
    #red3_dset0 = np.delete(red3_dset0, [0], axis = (0))
    #gc.collect()
    #print('3 done')
    #aux1 = red3_dset0[0,:,:,cutiny:].reshape([1,red3_dset0.shape[1],red3_dset0.shape[2],redall.shape[3]])
    #red3_dset0 = np.delete(red3_dset0, [0], axis = (0))
    #gc.collect()
    #print('4 done')
    #aux2 = red3_dset0[0,:,:,cutiny:].reshape([1,red3_dset0.shape[1],red3_dset0.shape[2],redall.shape[3]])
    #red3_dset0 = np.delete(red3_dset0, [0], axis = (0))
    #gc.collect()
    #print('5 done')
    #aux3 = red3_dset0[0,:,:,cutiny:].reshape([1,red3_dset0.shape[1],red3_dset0.shape[2],redall.shape[3]])
    #red3_dset0 = np.delete(red3_dset0, [0], axis = (0))
    #gc.collect()
    #print('6 done')
    #aux4 = red3_dset0[0,:,:,cutiny:].reshape([1,red3_dset0.shape[1],red3_dset0.shape[2],redall.shape[3]])
    #del red3_dset0
    #gc.collect()
    #file4.close()
    #gc.collect()
    #print('7 done')
   
    #redall = np.vstack((redall,aux0))
    #del aux0
    #gc.collect()
    #print('8 done')
    #redall = np.vstack((redall,aux1))
    #del aux1
    #gc.collect()
    #print('9 done')
    #redall = np.vstack((redall,aux2))
    #del aux2
    #gc.collect()
    #print('10 done')
    #redall = np.vstack((redall,aux3))
    #del aux3
    #gc.collect()
    #print('11 done')
    #redall = np.vstack((redall,aux4))
    #del aux4
    #print('12 done')
    #gc.collect()
    #print(redall.shape) 
    



    file2    = h5py.File(nametr[0] + '.hdf5', mmap_mode='r')  
    se1_dset0   = file2['/data/Analog channel 1 : SE2/data'] 
    se1_dset2 = np.array(se1_dset0,dtype=np.float32) 
    del se1_dset0
    gc.collect()
    
    file3    = h5py.File(nametr[1] + '.hdf5', mmap_mode='r')  
    se2_dset0   = file3['/data/Analog channel 1 : SE2/data'] 
    se2_dset2 = np.array(se2_dset0,dtype=np.float32) 
    del se2_dset0
    gc.collect()
    
    #file4    = h5py.File(nametr[2] + '.hdf5', mmap_mode='r')  
    #se3_dset0   = file4['/data/Analog channel 1 : SE2/data'] 
    #se3_dset2 = np.array(se3_dset0,dtype=np.float32) 
    #del se3_dset0
    #gc.collect()
    
    seall = np.empty([])
    seall = se1_dset2[1:,:,:]
    del se1_dset2
    gc.collect()
    seall = np.vstack((seall, se2_dset2[1:,:,:])) 
    del se2_dset2
    gc.collect()
    #seall = np.vstack((seall, se3_dset2)) 
    #del se3_dset2
    #gc.collect()
    print(seall.shape)
    
    file2.close()
    file3.close()
    #file4.close()
    gc.collect() 
    
    red1_dset = np.array(redall,dtype=np.float32)/1.0e3 #,dtype=np.float32)/1.0e3 #[0:No_experiments[index],:,:,:])/1.0e3 # np.array(red1_dset0,dtype=np.float32)/1.0e3
    del redall
    gc.collect()

    print('data loaded')
    
    #!!!!!REAL Registration - UNCOMMENT! !!!!!achtung: is giving some errors with nan and inf!!!!! WHY???????
    se1_dset_reg, red1_dset_reg, red1_dset_reg_all = reg_time_resolved_images_to_se(seall, red1_dset)
    gc.collect()

   
   
        
#    mycode = str(let[index]) + 'SEchannel = tempfile.NamedTemporaryFile(delete=False)'
#    exec(mycode)
#    np.savez(str(let[index]) + 'SEchannel', data = se1_dset_reg)
#    gc.collect()        
    
    del se1_dset_reg  , red1_dset_reg, red1_dset
    gc.collect()        
    
    mycode = str(let[index]) + 'Bluebright = tempfile.NamedTemporaryFile(delete=False)'
    exec(mycode)
    np.savez(str(let[index]) +'Bluebright', data = red1_dset_reg_all)


#    mycode = str(let[index]) + 'Redbright = tempfile.NamedTemporaryFile(delete=False)'
#    exec(mycode)
#    np.savez(str(let[index]) +'Redbright', data = red1_dset_reg_all)
        
    
klklkl
