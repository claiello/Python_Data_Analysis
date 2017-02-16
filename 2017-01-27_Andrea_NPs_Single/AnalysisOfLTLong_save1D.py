
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
#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################
    
No_experiments = [3,3,3]
                  
nametr = ['2017-01-27-1731_ImageSequence__500.000kX_10.000kV_30mu_13',
          '2017-01-27-1759_ImageSequence__500.000kX_10.000kV_30mu_14',
          '2017-01-27-1827_ImageSequence__500.000kX_10.000kV_30mu_15']
prefix =''
let = ['single']
index = 0
if True: #[7]: #np.arange(0,len(nametr)):

#    file2    = h5py.File(nametr[0] + '.hdf5', 'r')  
#    se1_dset0   = file2['/data/Analog channel 1 : SE2/data'] 
#    finalx = 150
#    se1_dset2 = np.array(se1_dset0[0:No_experiments[0],0:finalx,:],dtype=np.float32)
    
    #BELOW DSET IS VERY GOOD
    file20    = h5py.File(nametr[1] + '.hdf5', 'r')  
    se1_dset00   = file20['/data/Analog channel 1 : SE2/data'] 
#    startx = 40
#    finalx = 190
#    se1_dset20 = np.array(se1_dset00[0:No_experiments[1],startx:finalx,:],dtype=np.float32) 
    se1_dset20 = np.array(se1_dset00[0:No_experiments[1],:,:],dtype=np.float32) 
    
#    file200    = h5py.File(nametr[2] + '.hdf5', 'r')  
#    se1_dset000   = file200['/data/Analog channel 1 : SE2/data'] 
#    finalx = 150
#    se1_dset200 = np.array(se1_dset000[0:No_experiments[2],0:finalx,:],dtype=np.float32) 
#    
#    sefinal = np.vstack((se1_dset2,se1_dset20,se1_dset200)) #add together all pictures of same np
#    
#    SE, x = reg_images(se1_dset20) 
#    del x
#    gc.collect()
#    #print(SE.shape)
#    
#    plt.figure()
#    plt.imshow(SE[40:190,120:270],cmap=cm.Greys_r)
#    plt.show()
#    klklk
        
    
    
    red1_dset0  = file20['/data/Counter channel 1 : PMT red/PMT red time-resolved/data']#10 Scany points X10 frames x 150 tr pts x250 x 250 pixels
    #red1_dset0  = file20['/data/Counter channel 2 : PMT blue/PMT blue time-resolved/data']#10 Scany points X 10 frames x 150 tr pts x250 x 250 pixels
    
    print(red1_dset0.shape)
    
    #initbin = (150+50+3)-1 #1D IS JUST THE DECAY!!!!!!!
    red1_dset =np.array(red1_dset0[0:No_experiments[1],:,:,:])/1.0e3 # np.array(red1_dset0,dtype=np.float32)/1.0e3
    del red1_dset0
    gc.collect()
    print(red1_dset.shape)
    print(np.sum(np.isnan(red1_dset)))
    print(np.sum(np.isinf(red1_dset)))
   

    file20.close()
    gc.collect() 
 
    print('data loaded')
    
    
   
    se1_dset_reg, red1_dset_reg, red1_dset_reg_all = reg_time_resolved_images_to_se(se1_dset20, red1_dset)

    
    #del se1_dset2, red1_dset#, blue1_dset
   
    gc.collect()
    
    
    mycode = str(let[index]) + 'SEchannelC = tempfile.NamedTemporaryFile(delete=False)'
    exec(mycode)
    np.savez(str(let[index]) + 'SEchannelC', data = se1_dset_reg)
    
    mycode = str(let[index]) + 'RedbrightC = tempfile.NamedTemporaryFile(delete=False)'
    exec(mycode)
    np.savez(str(let[index]) +'RedbrightC', data = red1_dset_reg_all)
    
    mycode = prefix + str(let[index]) + 'RED1DC = tempfile.NamedTemporaryFile(delete=False)'
    exec(mycode)
    np.savez(prefix + str(let[index]) +'RED1DC', data = np.nanmean(red1_dset,axis = (0,2,3)))
##    
#    mycode = str(let[index]) + 'BluebrightC = tempfile.NamedTemporaryFile(delete=False)'
#    exec(mycode)
#    np.savez(str(let[index]) +'BluebrightC', data = red1_dset_reg_all)
#    
#    mycode =str(prefix + let[index]) + 'BLUE1DC = tempfile.NamedTemporaryFile(delete=False)'
#    exec(mycode)
#    np.savez(prefix + str(let[index]) +'BLUE1DC', data = np.nanmean(red1_dset,axis = (0,2,3)))
##    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    
#    do_gmmse_dset = True
#    
#    if do_gmmse_dset:
#        print('doing gmm se')
#        
#        gc.collect()
#        gmmse_red1_darkse_dset, gmmse_red1_brightse_dset, gmmse_se1_dark_dset, gmmse_se1_bright_dset, darkse_pct, brightse_pct, means, covars, weights = gmmone_tr_in_masked_channel(se1_dset_reg, red1_dset_reg)     
#
#        mycode = str(let[index]) +'SEchannelGMMA = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez(str(let[index]) +'SEchannelGMMA', bright = gmmse_se1_bright_dset, means = means, covars = covars, weights = weights)
#        
#        del red1_dset_reg, gmmse_red1_darkse_dset
#        gc.collect()
#                
#        del gmmse_se1_dark_dset#, gmmse_se1_bright_dset    
#        
#    else:
#        print('NOT doing gmm se') #assume all is bright in CL

    
klklklk
