#==============================================================================
# 1.6ms, 50 move, 150 excite, 1400 transient
# 1MHz clock rate (1mus timebins)
# 3 frames
# 150kX mag
# standard: 300pixels
# 10kV
# 30mum == 379pA
#with filter: 592 dicrhoic + 550/***32***nm in blue pmt + 650/54nm in red pmt, semrock brightline
#==============================================================================


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
    
No_experiments = 3*np.ones([7])
                  
nametr = ['2017-03-17-1635_ImageSequence__150.000kX_10.000kV_30mu_5',
          '2017-03-17-1642_ImageSequence__150.000kX_10.000kV_30mu_6',
          '2017-03-17-1701_ImageSequence__150.000kX_10.000kV_30mu_7',
          '2017-03-17-1722_ImageSequence__150.000kX_10.000kV_30mu_8',
          '2017-03-17-1747_ImageSequence__150.000kX_10.000kV_30mu_9',
          '2017-03-17-1759_ImageSequence__150.000kX_10.000kV_30mu_10',
          '2017-03-17-1816_ImageSequence__150.000kX_10.000kV_30mu_11']

description = ['Andrea small NaYF4:Er'] 
               
let = ['p300','p200','p400','p250','p400b','p250b','p350'] #no of pixels
Pixel_size = [2.48,3.72,1.86,2.98,1.86,2.98,2.13]
#taken in the order above

for index in np.arange(0,len(nametr)):
   

    print(index)

    file2    = h5py.File(nametr[index] + '.hdf5', mmap_mode='r')  
    titulo =  'Upconverting NPs'
    
    se1_dset0   = file2['/data/Analog channel 1 : SE2/data'] #10 Scany points X 10 frames X 250 x 250 pixels
    se1_dset2 = np.array(se1_dset0,dtype=np.float32) 
    del se1_dset0
    gc.collect()
    
    red1_dset0  = file2['/data/Counter channel 1 : PMT red/PMT red time-resolved/data']#10 Scany points X10 frames x 150 tr pts x250 x 250 pixels
    #red1_dset0  = file2['/data/Counter channel 2 : PMT blue/PMT blue time-resolved/data']#10 Scany points X 10 frames x 150 tr pts x250 x 250 pixels
    
    print(red1_dset0.shape)
    
    red1_dset =np.array(red1_dset0,dtype=np.float32)/1.0e3 
    del red1_dset0
    gc.collect()

    file2.close()
    gc.collect() 
    
    print('data loaded')
    
    se1_dset_reg, red1_dset_reg, red1_dset_reg_all = reg_time_resolved_images_to_se(se1_dset2, red1_dset)
    gc.collect()
    
    del se1_dset2, red1_dset   
    gc.collect()
   
    do_gmmse_dset = True
 
    
    if do_gmmse_dset:
        print('doing gmm se')
        
        gc.collect()
       
        gmmse_se1_bright_dset, means, covars, weights = gmmone_tr_in_masked_channel_modif_memory_issue(se1_dset_reg,red1_dset_reg) 
        
        del  red1_dset_reg
        gc.collect()
        
        mycode = str(let[index]) +'SEchannelGMM = tempfile.NamedTemporaryFile(delete=False)'
        exec(mycode)
        np.savez(str(let[index]) +'SEchannelGMM', bright = gmmse_se1_bright_dset, means = means, covars = covars, weights = weights)
        
        del  gmmse_se1_bright_dset,  means, covars, weights
        gc.collect()        
        
        mycode = str(let[index]) + 'SEchannel = tempfile.NamedTemporaryFile(delete=False)'
        exec(mycode)
        np.savez(str(let[index]) + 'SEchannel', data = se1_dset_reg)
        gc.collect()        
        
        del se1_dset_reg   
        gc.collect()        
    
#        mycode = str(let[index]) + 'Bluebright = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez(str(let[index]) +'Bluebright', data = red1_dset_reg_all)
    
    
        mycode = str(let[index]) + 'Redbright = tempfile.NamedTemporaryFile(delete=False)'
        exec(mycode)
        np.savez(str(let[index]) +'Redbright', data = red1_dset_reg_all)
        
            
        
    else:
        print('NOT doing gmm se') #assume all is bright in CL

    ###############################################################################
    ###############################################################################
    ###############################################################################

    
klklkl