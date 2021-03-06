
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
    
No_experiments = 1*np.ones([50]) #50 avgs
                  
nametr = ['2017-04-05-1505_ImageSequence__750.000kX_10.000kV_30mu_11']

description = ['Andrea small NaYF4:Er'] 
               
let = ['x750'] #no of pixels

#taken in the order above

for index in np.arange(0,len(nametr)):
   

    print(index)

    file2    = h5py.File(nametr[index] + '.hdf5', mmap_mode='r')  
    titulo =  'Upconverting NPs'
    
    se1_dset0   = file2['/data/Analog channel 1 : SE2/data'] #10 Scany points X 10 frames X 250 x 250 pixels
    se1_dset2 = np.array(se1_dset0,dtype=np.float32) 
    del se1_dset0
    gc.collect()
    
    #red1_dset0  = file2['/data/Counter channel 1 : PMT red/PMT red time-resolved/data']#10 Scany points X10 frames x 150 tr pts x250 x 250 pixels
    red1_dset0  = file2['/data/Counter channel 2 : PMT blue/PMT blue time-resolved/data']#10 Scany points X 10 frames x 150 tr pts x250 x 250 pixels
    
   
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
 
    mycode = str(let[index]) + 'SEchannel = tempfile.NamedTemporaryFile(delete=False)'
    exec(mycode)
    np.savez(str(let[index]) + 'SEchannel', data = se1_dset_reg)
    gc.collect()        
    
    del se1_dset_reg   
    gc.collect()        
    
    mycode = str(let[index]) + 'Bluebright = tempfile.NamedTemporaryFile(delete=False)'
    exec(mycode)
    np.savez(str(let[index]) +'Bluebright', data = red1_dset_reg_all)
    
    
#    mycode = str(let[index]) + 'Redbright = tempfile.NamedTemporaryFile(delete=False)'
#    exec(mycode)
#    np.savez(str(let[index]) +'Redbright', data = red1_dset_reg_all)
        
            
        
    

    ###############################################################################
    ###############################################################################
    ###############################################################################

    
klklkl