
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
    
No_experiments = [4,4,4,4,4,4,4,4]#5*np.ones([6])
                  
nametr = ['2017-01-13-1730_ImageSequence__150.000kX_10.000kV_30mu_15',
          '2017-01-13-1810_ImageSequence__150.000kX_10.000kV_30mu_17',
          '2017-01-13-1935_ImageSequence__150.000kX_10.000kV_30mu_3',
          '2017-01-13-2011_ImageSequence__150.000kX_10.000kV_30mu_4',
          '2017-01-13-2050_ImageSequence__150.000kX_10.000kV_30mu_7',
          '2017-01-13-2132_ImageSequence__150.000kX_10.000kV_30mu_8',
          '2017-01-13-2213_ImageSequence__150.000kX_10.000kV_30mu_9',
          '2017-01-13-2251_ImageSequence__150.000kX_10.000kV_30mu_10']

description = ['Andrea small NaYF4:Er'] # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
kv = [10]

#nominal Temps
let = ['N102pt5', 'N92pt9', 'N66pt4', 'N72pt8', 'N60pt6', 'N48pt5', 'N39pt8', 'N31pt2']

#index = 6
#if index is 6:
    
for index in np.arange(0,len(nametr)):
   

    print(index)

    file2    = h5py.File(nametr[index] + '.hdf5', mmap_mode='r')  
    titulo =  'Upconverting NPs'
    il1_dset   = file2['/data/Analog channel 2 : InLens/data'] #10 Scany points X 10 frames X 250 x 250 pixels
    
    mycode = str(let[index]) + 'ILchannel = tempfile.NamedTemporaryFile(delete=False)'
    exec(mycode)
    np.savez(str(let[index]) + 'ILchannel', data = np.array(il1_dset,dtype=np.float32)) #[0:No_experiments[index],:,:],dtype=np.float32))
    del il1_dset
    gc.collect()    
    
    se1_dset0   = file2['/data/Analog channel 1 : SE2/data'] #10 Scany points X 10 frames X 250 x 250 pixels
    se1_dset2 = np.array(se1_dset0,dtype=np.float32) #[0:No_experiments[index],:,:],dtype=np.float32) #[0:No_experiments[index],:,:]
    del se1_dset0
    gc.collect()
    
   # red1_dset0  = file2['/data/Counter channel 1 : PMT red/PMT red time-resolved/data']#10 Scany points X10 frames x 150 tr pts x250 x 250 pixels
    red1_dset0  = file2['/data/Counter channel 2 : PMT blue/PMT blue time-resolved/data']#10 Scany points X 10 frames x 150 tr pts x250 x 250 pixels
    
    print(red1_dset0.shape)
    
    red1_dset =np.array(red1_dset0,dtype=np.float32)/1.0e3 #,dtype=np.float32)/1.0e3 #[0:No_experiments[index],:,:,:])/1.0e3 # np.array(red1_dset0,dtype=np.float32)/1.0e3
    del red1_dset0
    gc.collect()
#    print(red1_dset.shape)
#    print(np.sum(np.isnan(red1_dset)))
#    print(np.sum(np.isinf(red1_dset)))
    
    file2.close()
    gc.collect() 
    
    #convert to smaller data types
    #se1_dset2 = np.array(se1_dset) #, dtype=np.float16) #convert to 16 if needed
    #se1_dset2.reshape((10,250,250))
    #il1_dset2 = np.array(il1_dset) #[0:No_experiments[index],:,:]) #, dtype=np.float16)
    #il1_dset2.reshape((10,250,250))
    
    #red1_dset = np.array(red1_dset) 
    #blue1_dset = np.array(blue1_dset) 
    #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
    #red1_dset = red1_dset/1.0e3
    #red1_dset = np.array(red1_dset, dtype=np.float32)  #converting the CL dset to float16 from float64 is creating infinities! bc max value is > 2^16 
#    blue1_dset = blue1_dset/1.0e3
#    blue1_dset = np.array(blue1_dset, dtype=np.float32)  #converting the CL dset to float16 from float64 is creating infinities! bc max value is > 2^16 
#        
    print('data loaded')
    
    #REGISTRATION OF CL CHANNEL ONTO SE CHANNEL  
    ###############################################################################
    ###############################################################################
    ###############################################################################
    #independently
    #se1_dset_reg = reg_images(se1_dset)
    #The code below registers time resolved data to itself, across frames. Each one of the tr points is registered to the same time point in other frames
    #red_dset_reg_list = reg_images_tr(red1_dset) #list is across number of time resolved points #Does not work too well for this dset BUT TRY WITH OTHERS
    #Future: make tr, say, of red register to time resolved, say, of blue
    
    #process = psutil.Process(os.getpid())
    #print(process.memory_info().rss)
    
    #print('mem used' + str(memory_usage_psutil()) )   
    
    #!!!!!REAL Registration - UNCOMMENT! !!!!!achtung: is giving some errors with nan and inf!!!!! WHY???????
    se1_dset_reg, red1_dset_reg, red1_dset_reg_all = reg_time_resolved_images_to_se(se1_dset2, red1_dset)
    gc.collect()
#    se1_dset_reg, blue1_dset_reg, blue1_dset_reg_all = reg_time_resolved_images_to_se(se1_dset2, blue1_dset)
    
    ##red1_dset_reg, red1_dset_reg_all = reg_images(red1_dset)    
    ##se1_dset_reg = np.array(se1_dset, dtype=np.float16)
    
    #se1_dset_reg = np.array(se1_dset_reg, dtype=np.float16)
    #red1_dset_reg = np.array(red1_dset_reg, dtype=np.float32)
    #red1_dset_reg_all = np.array(red1_dset_reg_all, dtype=np.float32)
    
    #second step of registerist CL using bright CL: does NOT cause errors!!!
    #right now, remaining totalpoints - cut_at_beginning time resolved points, over all experiments
    #25 dark (in reality, dark until 28)/ 50 bright / 125 transient
    #cut arrays are 3 / 50 / 125
    #center_cl_index = 3 # (50 + 3)/2 # this index is going to be used as reference
    #end_left_index = 0#not used for the time being
    #end_right_index = 0#not used for the time being
    #new = reg_images_middle_cl(red1_dset_reg,center_cl_index,0,0)
    #red1_dset_reg = np.array(new, dtype=np.float32)
    
    #!!!!!END OF REAL Registration - UNCOMMENT!   
    #print(np.sum(np.isnan(red1_dset_reg)))
    #print(np.sum(np.isinf(red1_dset_reg)))
#    print(np.sum(np.isnan(blue1_dset_reg)))
#    print(np.sum(np.isinf(blue1_dset_reg)))
   
#    ###MOCK REGISTRATION, QUICK FOR HACK ####################################!!!!!!!!!!!!!!!!!!!!

    #se1_dset_reg = np.average(se1_dset2, axis=0)  ###### uncomment here
    #red1_dset_reg = np.average(red1_dset, axis=0)
    #red1_dset_reg_all = np.array(red1_dset)


    #il1_dset_reg = il1_dset
    
#    blue1_dset_reg = np.average(blue1_dset, axis=0)
#    blue1_dset_reg_all = np.array(blue1_dset)
    #il1_dset_reg = np.array(il1_dset, dtype=np.float32)
#    blue1_dset_reg = np.array(blue1_dset_reg, dtype=np.float32)
#    blue1_dset_reg_all = np.array(blue1_dset_reg_all, dtype=np.float32)    
    
    #second step
    #center_cl_index = 3 # # this index is going to be used as reference, should be no 82 in original vector
#    end_left_index = 0#not used for the time being
#    end_right_index = 0#not used for the time being
#    new = reg_images_middle_cl(red1_dset_reg,center_cl_index,0,0)
#    red1_dset_reg = np.array(new, dtype=np.float32)
    
    ## end of mock registration
    
    del se1_dset2, red1_dset   #,red1_dset_reg
    gc.collect()
   
    #del red1_dset_reg
    #gc.collect()
    
    #process = psutil.Process(os.getpid())
    #print(process.memory_info().rss)
    
    #print('mem used' + str(memory_usage_psutil()))    
    
   
   #helper_array = np.average( red1_dset_reg_all, axis = 0)
   # del red1_dset_reg_all
    #gc.collect()
    
    do_gmmse_dset = True
 
    
    if do_gmmse_dset:
        print('doing gmm se')
        #Original
        #gmmred_blue_dark_dset, gmmred_blue_bright_dset, gmmred_red_dark_dset, gmmred_red_bright_dset = gmmone(red_dset_cut, blue_dset_cut)
        #do_analysis(blue_dset_cut, red_dset_cut, gmmred_blue_dark_dset, gmmred_blue_bright_dset, gmmred_red_dark_dset, gmmred_red_bright_dset, 'YAP', 'Chlor','GMM red', 'red dark spots', 'red bright spots',Pixel_size)   
        #Version for time-resolved
        
        gc.collect()
        gmmse_se1_bright_dset, means, covars, weights = gmmone_tr_in_masked_channel_modif_memory_issue(se1_dset_reg,red1_dset_reg) #helper_array)
        
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
        
        del se1_dset_reg   #helper_array,
        gc.collect()        
        
#        del red1_dset_reg
#        gc.collect()
        
#      
#        import bz2
#        with bz2.BZ2File(str(let[index]) +'Bluebright' + '.txt.bz2', 'wb', compresslevel = 9) as f:
#            f.write(red1_dset_reg_all)
#
#        import bz2
#        with bz2.BZ2File(str(let[index]) +'Redbright' + '.txt.bz2', compresslevel = 9) as f:
#            f.write(red1_dset_reg_all)
#        
        
        ############# dont work for huge-huge dsets
#        mycode = str(let[index]) + 'Bluebright = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez_compressed(str(let[index]) +'Bluebright', data = red1_dset_reg_all)
#        gc.collect()
    
#        mycode = str(let[index]) + 'Redbright = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez(str(let[index]) +'Redbright', data = red1_dset_reg_all)
    
    
    
    
    
    
    
        mycode = str(let[index]) + 'Bluebright1 = tempfile.NamedTemporaryFile(delete=False)'
        exec(mycode)
        np.savez(str(let[index]) +'Bluebright1', data = red1_dset_reg_all[0,:,:,:])
        
        mycode = str(let[index]) + 'Bluebright2 = tempfile.NamedTemporaryFile(delete=False)'
        exec(mycode)
        np.savez(str(let[index]) +'Bluebright2', data = red1_dset_reg_all[1,:,:,:])
        
        mycode = str(let[index]) + 'Bluebright3 = tempfile.NamedTemporaryFile(delete=False)'
        exec(mycode)
        np.savez(str(let[index]) +'Bluebright3', data = red1_dset_reg_all[2,:,:,:])
        
        mycode = str(let[index]) + 'Bluebright4 = tempfile.NamedTemporaryFile(delete=False)'
        exec(mycode)
        np.savez(str(let[index]) +'Bluebright4', data = red1_dset_reg_all[3,:,:,:])    
    
        
#        mycode = str(let[index]) + 'Redbright1 = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez(str(let[index]) +'Redbright1', data = red1_dset_reg_all[0,:,:,:])
#        
#        mycode = str(let[index]) + 'Redbright2 = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez(str(let[index]) +'Redbright2', data = red1_dset_reg_all[1,:,:,:])
#        
#        mycode = str(let[index]) + 'Redbright3 = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez(str(let[index]) +'Redbright3', data = red1_dset_reg_all[2,:,:,:])
#        
#        mycode = str(let[index]) + 'Redbright4 = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez(str(let[index]) +'Redbright4', data = red1_dset_reg_all[3,:,:,:])
            
        
    else:
        print('NOT doing gmm se') #assume all is bright in CL

    ###############################################################################
    ###############################################################################
    ###############################################################################

#inputafile = bz2.BZ2File('N66pt4Bluebright.txt.bz2')
#aa = inputafile.read()
#print(aa.shape)
    
klklkl