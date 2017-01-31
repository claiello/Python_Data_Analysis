
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
    
No_experiments = [2,2,2,2,2]
                  
nametr = ['2017-01-23-1547_ImageSequence__250.000kX_10.000kV_30mu_7',
          '2017-01-23-1608_ImageSequence__250.000kX_10.000kV_30mu_8',
          '2017-01-23-1633_ImageSequence__250.000kX_10.000kV_30mu_9',
          '2017-01-23-1736_ImageSequence__250.000kX_10.000kV_30mu_10',
          '2017-01-23-1818_ImageSequence__63.372kX_10.000kV_30mu_11']

description = ['Andrea small NaYF4:Er'] # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
kv = [10]

let = ['pix1', 'pix2', 'pix3', 'pix4', 'pix5']

for index in np.arange(0,len(nametr)):

    print(index)

    file2    = h5py.File(nametr[index] + '.hdf5', 'r')  
    titulo =  'Upconverting NPs'
    
    se1_dset0   = file2['/data/Analog channel 1 : SE2/data'] #10 Scany points X 10 frames X 250 x 250 pixels
    se1_dset2 = np.array(se1_dset0[0:No_experiments[index],:,:],dtype=np.float32) #[0:No_experiments[index],:,:]
    del se1_dset0
    gc.collect()
    
    #red1_dset0  = file2['/data/Counter channel 1 : PMT red/PMT red time-resolved/data']#10 Scany points X10 frames x 150 tr pts x250 x 250 pixels
    red1_dset0  = file2['/data/Counter channel 2 : PMT blue/PMT blue time-resolved/data']#10 Scany points X 10 frames x 150 tr pts x250 x 250 pixels
    
    print(red1_dset0.shape)
    
    red1_dset =np.array(red1_dset0[0:No_experiments[index],:,:,:])/1.0e3 # np.array(red1_dset0,dtype=np.float32)/1.0e3
    del red1_dset0
    gc.collect()
    print(red1_dset.shape)
    print(np.sum(np.isnan(red1_dset)))
    print(np.sum(np.isinf(red1_dset)))
   

    file2.close()
    gc.collect() 
 
    print('data loaded')
    
    
   
    se1_dset_reg, red1_dset_reg, red1_dset_reg_all = reg_time_resolved_images_to_se(se1_dset2, red1_dset)

    
    del se1_dset2, red1_dset#, blue1_dset
   
    gc.collect()
    
    
    mycode = str(let[index]) + 'SEchannel = tempfile.NamedTemporaryFile(delete=False)'
    exec(mycode)
    np.savez(str(let[index]) + 'SEchannel', data = se1_dset_reg)
#    
#    mycode = str(let[index]) + 'Redbright = tempfile.NamedTemporaryFile(delete=False)'
#    exec(mycode)
#    np.savez(str(let[index]) +'Redbright', data = red1_dset_reg_all)
#    
#    mycode =str(let[index]) + 'RED1D = tempfile.NamedTemporaryFile(delete=False)'
#    exec(mycode)
#    np.savez(str(let[index]) +'RED1D', data = np.nanmean(red1_dset_reg_all,axis = (0,2,3)))
    
    mycode = str(let[index]) + 'Bluebright = tempfile.NamedTemporaryFile(delete=False)'
    exec(mycode)
    np.savez(str(let[index]) +'Bluebright', data = red1_dset_reg_all)
    
    mycode =str(let[index]) + 'BLUE1D = tempfile.NamedTemporaryFile(delete=False)'
    exec(mycode)
    np.savez(str(let[index]) +'BLUE1D', data = np.nanmean(red1_dset_reg_all,axis = (0,2,3)))
#    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    
    do_gmmse_dset = True
    
    if do_gmmse_dset:
        print('doing gmm se')
        
        gc.collect()
        gmmse_red1_darkse_dset, gmmse_red1_brightse_dset, gmmse_se1_dark_dset, gmmse_se1_bright_dset, darkse_pct, brightse_pct, means, covars, weights = gmmone_tr_in_masked_channel(se1_dset_reg, red1_dset_reg)     

        mycode = str(let[index]) +'SEchannelGMM = tempfile.NamedTemporaryFile(delete=False)'
        exec(mycode)
        np.savez(str(let[index]) +'SEchannelGMM', bright = gmmse_se1_bright_dset, means = means, covars = covars, weights = weights)
        
        
        del red1_dset_reg, gmmse_red1_darkse_dset
        gc.collect()
                
        del gmmse_se1_dark_dset#, gmmse_se1_bright_dset    
        
    else:
        print('NOT doing gmm se') #assume all is bright in CL

    
klklklk
