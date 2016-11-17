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
from BackgroundCorrection import *
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
#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################
#index = 0
calc_blue = True
#Excitation is impulsive, 120ns per pulse,turn on at point no 80, off at point no 82

#No_pixels = 250
Time_bin = 2000.0#in ns; 1/clock of 25MHz 
nominal_time_on = 198.0#time during which e-beam nominally on, in mus
totalpoints = 500 #total number of time-resolved points
### data
if calc_blue is False:
    pmt = ['PMT red']
    channel = ['1']
else:
    pmt = ['PMT blue']
    channel = ['2']
    
No_experiments = [1,1,1,1,1,1]

###### Long TR scan:
#1200mus move:2mus/excite 198mus/decay 1000mus

                
nametr = ['2016-11-07-1214_ImageSequence_Bella10_4.953kX_10.000kV_30mu_3',
          '2016-11-07-1243_ImageSequence_Bella20_12.239kX_10.000kV_30mu_7',
          '2016-11-07-1303_ImageSequence_Bella40_11.987kX_10.000kV_30mu_3',
          '2016-11-07-1318_ImageSequence_Bella40_24.000kX_10.000kV_30mu_6',
          '2016-11-07-1334_ImageSequence_Bella40_48.000kX_10.000kV_30mu_9',
          '2016-11-07-1354_ImageSequence_Bella40_12.380kX_10.000kV_30mu_14']
          
#Corresponding SEs: 12:02, 12:32, 12:51, nothing, nothing (3 last:same place), 1:42
#Also, SE visible in the TR files. Not doing SE segmentation
# Note: all datasets given same initial conditions

description = ['Bella NaYF4:Er'] # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
kv = [10]

let = ['Er10','Er20','Er4012kX','Er4024kX','Er4040kX','Er80']

#tau_single = np.zeros(len(name))
#tau_single_error = np.zeros(len(name))
#tau_bi = np.zeros([len(name),2])
#tau_bi_error = np.zeros([len(name),2])

#original
#for index in np.arange(11,11):   #11):

index = 1
if index is 0:
    
#for Er60 only: np.arange(9,12)
#for index in np.arange(0,6):

    print(index)

    file2    = h5py.File(nametr[index] + '.hdf5', 'r')  
    titulo =  'Upconverting NPs'
    #il1_dset   = file2['/data/Analog channel 2 : InLens/data'] #10 Scany points X 10 frames X 250 x 250 pixels
    #se1_dset   = file2['/data/Analog channel 1 : SE2/data'] #10 Scany points X 10 frames X 250 x 250 pixels
    red0_dset  = file2['/data/Counter channel 1 : PMT red/PMT red time-resolved/data']#10 Scany points X10 frames x 150 tr pts x250 x 250 pixels
    blue0_dset  = file2['/data/Counter channel 2 : PMT blue/PMT blue time-resolved/data']#10 Scany points X 10 frames x 150 tr pts x250 x 250 pixels
    
    #red1_dset50  = file3['/data/Counter channel ' + channel[index] + ' : '+ pmt[index]+'/' + pmt[index] + ' time-resolved/data']#50 frames x 200 tr pts x250 x 250 pixels
    #red1_dset = np.append(red1_dset , red1_dset50, axis = 0)
        
    Pixel_size = red0_dset.attrs['element_size_um'][1]*1000.0 #saved attribute is in micron; converting to nm
    Ps = [str("{0:.2f}".format(Pixel_size))] #pixel size in nm, the numbers above with round nm precision    
    
    print(red0_dset.shape)
    
    #cut part of frames with ebeam off, here 25 first points
    #cut_at_beginning = 0
    #red1_dset = red1_dset[:,:,cut_at_beginning::,:,:]
    
    #hack to go faster
    fastfactor = 1
    #se1_dset = se1_dset[0::fastfactor,:,:]
    #red1_dset = red1_dset[0::fastfactor,:,:,:]
    
    #no experiments to consider
    #se1_dset = se1_dset[0:No_experiments[index]+1,:,:]
    red1_dset = red0_dset#[0:No_experiments[index]+1,:,:,:]#.reshape((No_experiments[index],:,:,:],150,250,250))
    blue1_dset = blue0_dset#[0:No_experiments[index]+1,:,:,:]#.reshape((No_experiments[index],:,:,:],150,250,250))
    
    #convert to smaller data types
    #se1_dset2 = np.array(se1_dset) #, dtype=np.float16) #convert to 16 if needed
    #se1_dset2.reshape((10,250,250))
   # il1_dset2 = np.array(il1_dset) #, dtype=np.float16)
    #il1_dset2.reshape((10,250,250))
    
    red1_dset = np.array(red1_dset) 
    blue1_dset = np.array(blue1_dset) 
    #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
    red1_dset = red1_dset/1.0e3
    red1_dset = np.array(red1_dset, dtype=np.float32)  #converting the CL dset to float16 from float64 is creating infinities! bc max value is > 2^16 
    blue1_dset = blue1_dset/1.0e3
    blue1_dset = np.array(blue1_dset, dtype=np.float32)  #converting the CL dset to float16 from float64 is creating infinities! bc max value is > 2^16 
        
    print('data loaded')
    
    #PLOT EXPT BY EXPT BEHAVIOUR ON ALL PIXELS
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    ###FIG 1
    #if index >= 8: #8,9,10,11: the 4 from Er 60% 
    #major_ticks = [2,4,6]
    #else:
        #major_ticks = [5,15,25]
    #plot_expt_by_expt_behaviour(titulo + r', all pixels', red1_dset, Time_bin, nominal_time_on,fastfactor,'r',major_ticks,dark_dset=None,plot_dark=False,unit='kHz') #pass titulo as well
    #fig_no = 'Fig#1'
    #multipage(name_str[index] + fig_no + '.pdf',dpi=80)
    ###END FIG1
    
    #REGISTRATION OF CL CHANNEL ONTO SE CHANNEL  
    ###############################################################################
    ###############################################################################
    ###############################################################################
    #independently
    #se1_dset_reg = reg_images(se1_dset)
    #The code below registers time resolved data to itself, across frames. Each one of the tr points is registered to the same time point in other frames
    #red_dset_reg_list = reg_images_tr(red1_dset) #list is across number of time resolved points #Does not work too well for this dset BUT TRY WITH OTHERS
    #Future: make tr, say, of red register to time resolved, say, of blue
    
    #!!!!!REAL Registration - UNCOMMENT! !!!!!achtung: is giving some errors with nan and inf!!!!! WHY???????
#    se1_dset_reg, red1_dset_reg, red1_dset_reg_all = reg_time_resolved_images_to_se(se1_dset2, red1_dset)
#    se1_dset_reg, blue1_dset_reg, blue1_dset_reg_all = reg_time_resolved_images_to_se(se1_dset2, blue1_dset)
#    
#    ##red1_dset_reg, red1_dset_reg_all = reg_images(red1_dset)    
#    ##se1_dset_reg = np.array(se1_dset, dtype=np.float16)
#    
#    #se1_dset_reg = np.array(se1_dset_reg, dtype=np.float16)
#    #red1_dset_reg = np.array(red1_dset_reg, dtype=np.float32)
#    #red1_dset_reg_all = np.array(red1_dset_reg_all, dtype=np.float32)
#    
#    #second step of registerist CL using bright CL: does NOT cause errors!!!
#    #right now, remaining totalpoints - cut_at_beginning time resolved points, over all experiments
#    #25 dark (in reality, dark until 28)/ 50 bright / 125 transient
#    #cut arrays are 3 / 50 / 125
#    #center_cl_index = 3 # (50 + 3)/2 # this index is going to be used as reference
#    #end_left_index = 0#not used for the time being
#    #end_right_index = 0#not used for the time being
#    #new = reg_images_middle_cl(red1_dset_reg,center_cl_index,0,0)
#    #red1_dset_reg = np.array(new, dtype=np.float32)
#    
#    #!!!!!END OF REAL Registration - UNCOMMENT!   
#    print(np.sum(np.isnan(red1_dset_reg)))
#    print(np.sum(np.isinf(red1_dset_reg)))
#    print(np.sum(np.isnan(blue1_dset_reg)))
#    print(np.sum(np.isinf(blue1_dset_reg)))
   
#    ###MOCK REGISTRATION, QUICK FOR HACK ####################################!!!!!!!!!!!!!!!!!!!!
#    if (index is 3) or (index is 6):
#        se1_dset_reg = np.average(se1_dset2, axis=0)
#        #se1_dset_reg, sth = reg_images(se1_dset2) 
#        #del sth
#        red1_dset_reg = np.average(red1_dset, axis=0)
#        red1_dset_reg_all = np.array(red1_dset)
#        blue1_dset_reg = np.average(blue1_dset, axis=0)
#        blue1_dset_reg_all = np.array(blue1_dset)
        #se1_dset_reg = np.array(se1_dset_reg, dtype=np.float16)
        #red1_dset_reg = np.array(red1_dset_reg, dtype=np.float32)
        #red1_dset_reg_all = np.array(red1_dset_reg_all, dtype=np.float32)

#    il1_dset_reg = il1_dset
#    
    blue1_dset_reg = np.average(blue1_dset, axis=0)
    red1_dset_reg = np.average(red1_dset, axis=0)
##    blue1_dset_reg_all = np.array(blue1_dset)
#    il1_dset_reg = np.array(il1_dset_reg, dtype=np.float32)
##    blue1_dset_reg = np.array(blue1_dset_reg, dtype=np.float32)
##    blue1_dset_reg_all = np.array(blue1_dset_reg_all, dtype=np.float32)    
#    
#    #second step
#    #center_cl_index = 3 # # this index is going to be used as reference, should be no 82 in original vector
##    end_left_index = 0#not used for the time being
##    end_right_index = 0#not used for the time being
##    new = reg_images_middle_cl(red1_dset_reg,center_cl_index,0,0)
##    red1_dset_reg = np.array(new, dtype=np.float32)
#    
#    ## end of mock registration
    
    del red1_dset, blue1_dset
#    file1.close()
    file2.close()
    gc.collect()
   
    
    #CUT DATASETS TO SUBSHAPE
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    ### cut only inside window: these are the base images!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #trying circular mask at center a,b
    #a, b = 247,255 #y was 255  x was 243
    #n = blue_dset_reg.shape[0] #not square matrix anymore; does not matter, only approximatively
    #r = 160 #was 170
    #y,x = np.ogrid[-a:n-a, -b:n-b]
    #mask = x*x + y*y <= r*r
    # cutting channels
    red1_dset_cut = red1_dset_reg #np.empty([blue_dset_reg.shape[0],blue_dset_reg.shape[1]])
    #red1_dset_cut_all = red1_dset_reg_all
    #red_dset_cut[:] = np.nan
    #red_dset_cut[mask] = red_dset_reg[mask]
    #se1_dset_cut = se1_dset_reg #np.empty([blue_dset_reg.shape[0],blue_dset_reg.shape[1]])
    #se_dset_cut[:] = np.nan
    #se_dset_cut[mask] = se_dset_reg[mask]
    
    blue1_dset_cut = blue1_dset_reg #np.empty([blue_dset_reg.shape[0],blue_dset_reg.shape[1]])
    #blue1_dset_cut_all = blue1_dset_reg_all
   # il1_dset_cut = il1_dset_reg #np.empty([blue_dset_reg.shape[0],blue_dset_reg.shape[1]])
    
   # se1_dset_cut = np.array(se1_dset_cut, dtype=np.float16)
    red1_dset_cut = np.array(red1_dset_cut, dtype=np.float32)
    #red1_dset_cut_all = np.array(red1_dset_cut_all, dtype=np.float32)
    
    #il1_dset_cut = np.array(il1_dset_cut, dtype=np.float16)
    blue1_dset_cut = np.array(blue1_dset_cut, dtype=np.float32)
    #blue1_dset_cut_all = np.array(blue1_dset_cut_all, dtype=np.float32)
    
    del red1_dset_reg, blue1_dset_reg
    gc.collect()
    
#    mycode = str(let[index]) + 'SEchannel = tempfile.NamedTemporaryFile(delete=False)'
#    exec(mycode)
#    np.savez(str(let[index]) + 'SEchannel', data = se1_dset_cut)
##    
    mycode = str(let[index]) + 'Redbright = tempfile.NamedTemporaryFile(delete=False)'
    exec(mycode)
    np.savez(str(let[index]) +'Redbright', data = red1_dset_cut)
    
    mycode = str(let[index]) + 'Bluebright = tempfile.NamedTemporaryFile(delete=False)'
    exec(mycode)
    np.savez(str(let[index]) +'Bluebright', data = blue1_dset_cut)
    
#    mycode = str(let[index]) + 'ILchannel = tempfile.NamedTemporaryFile(delete=False)'
#    exec(mycode)
#    np.savez(str(let[index]) + 'ILchannel', data = il1_dset_cut)
###    

        
    
    ####mock
    #start_of_transient = 83 #time-bin 75 + 300ns/40ns = 75 + ~8 = 83
    #calcdecay(red1_dset_cut[start_of_transient::,:,:], time_detail= Time_bin*1e-9*fastfactor,titulo=r'Cathodoluminescence rate decay, \n ' + titulo + ', SE `signal\' pixels',other_dset1=red1_dset_cut[start_of_transient::,:,:] ,other_dset2=red1_dset_cut[start_of_transient::,:,:])
    #multipage(name_str[index] + fig_no + '.pdf',dpi=80)
    #klklklk
    
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    ####################################################################### OPTIONAL
    #want_gaussian_filter_correction_blue = False
    #want_gaussian_filter_correction_red = False
    #
    #if want_gaussian_filter_correction_blue:
    #   sigma_blue = 1 
    #   blue_dset_cut1 = gaussian_filter_correction(blue_dset_cut, 'Blue',sigma_blue)
    #   blue_dset_cut = blue_dset_cut1  
    #
    #if want_gaussian_filter_correction_red:
    #   sigma_red = 1 
    #   red_dset_cut1 = gaussian_filter_correction(red_dset_cut, 'Red',sigma_red)
    #   red_dset_cut = red_dset_cut1  
    #
    ################################################################ END OF OPTIONAL
    #
    ####################################################################### OPTIONAL
    #### Suggested:
    ## 1- Blue True, 3, [0] + Red False
    ## 2 - Blue True, 3, [2] + Red False
    ## 3 - Blue True, 3, [0] + Red True, 21, [1]
    ## 4 - Blue True, 3, [2] + Red True, 21, [1]
    ## 5 - Blue False, Red False
    #
    #want_background_correction_blue = False
    #want_background_correction_red = False
    #
    #filterset = ['white_tophat','black_tophat','medfilt']
    #
    #if want_background_correction_blue:
    #    # Available algo types:
    #    # 'white_tophat' -> needs to change disk size
    #    # 'black_tophat' -> needs to change disk size
    #    # 'medfilt' -> needs to changer kernel size
    #    
    #    # New base dsets: blue_dset_cut, red_dset_cut
    #    size_blue = 3
    #    blue_dset_cut1 = background_correction(blue_dset_cut, filterset[0], 'Blue',size_blue)
    #    #blue_dset_cut2 = background_correction(blue_dset_cut, filterset[1], 'Blue',size_blue)
    #    blue_dset_cut3 = background_correction(blue_dset_cut, filterset[2], 'Blue',size_blue)
    #    #both [0] and [2] acceptable; min size_blue that makes sense = 3
    #    
    #    blue_dset_cut = blue_dset_cut1     #1 or 3
    #       
    #if want_background_correction_red:    
    #    size_red = 21
    #    #red_dset_cut1 = background_correction(red_dset_cut, filterset[0], 'Red',size_red)
    #    red_dset_cut2 = background_correction(red_dset_cut, filterset[1], 'Red',size_red)
    #    #red_dset_cut3 = background_correction(red_dset_cut, filterset[2], 'Red',size_red)
    #    # [1] can be good. Or no correction.
    #    red_dset_cut = red_dset_cut2
    #
    ################################################################ END OF OPTIONAL
    #
    ####TEST OTHER SEGMENTATION MODELS FOR SE, AND PLOT SE HISTOGRAM + SEGMENTATION IN THE FUTURE!!!!!
    
    ##plt.close("all")
    #
    #from CreateDatasets import *
    #
    #do_avg_dset = False
    #do_median_dset = False
    #do_arb_thr_one = False
    do_gmmse_dset = False
    #do_gmmboth_dset = False
    #do_threshold_adaptive = False
    #do_random_walker = False
    #do_otsu = False
    #
    #### construct different datasets
    #### 1) Simple average
    #if do_avg_dset:
    #    print('doing avg')
    #    below_blue, above_blue, below_red, above_red = above_below_avg(blue_dset_cut, red_dset_cut)
    #    do_analysis(blue_dset_cut, red_dset_cut, below_blue, above_blue, below_red, above_red, 'YAP', 'Chlor','Above/Below avg', 'below avg', 'above avg',Pixel_size)
    #
    #### 1) Simple median
    #if do_median_dset:
    #    print('doing median')
    #    belowm_blue, abovem_blue, belowm_red, abovem_red = above_below_median(blue_dset_cut, red_dset_cut)
    #    do_analysis(blue_dset_cut, red_dset_cut, belowm_blue, abovem_blue, belowm_red, abovem_red, 'YAP', 'Chlor','Above/Below median', 'below median', 'above median',Pixel_size)
    #
    #### 1) Arb thresh in red
    #if do_arb_thr_one:
    #    print('doing arb thres')
    #    arb_threshold = 0.6 #fraction of max
    #    belowarb_blue, abovearb_blue, belowarb_red, abovearb_red = arb_thr_one(red_dset_cut, blue_dset_cut, arb_threshold)
    #    do_analysis(blue_dset_cut, red_dset_cut, belowarb_blue, abovearb_blue, belowarb_red, abovearb_red, 'YAP', 'Chlor','Above/Below arb thr = ' + str(arb_threshold) + ' of red max', 'below red thr', 'above red thr',Pixel_size)
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### 2) GMM with red mask, where red has been recognized as fluorescence
   
    
    if do_gmmse_dset:
        print('doing gmm se')
        #Original
        #gmmred_blue_dark_dset, gmmred_blue_bright_dset, gmmred_red_dark_dset, gmmred_red_bright_dset = gmmone(red_dset_cut, blue_dset_cut)
        #do_analysis(blue_dset_cut, red_dset_cut, gmmred_blue_dark_dset, gmmred_blue_bright_dset, gmmred_red_dark_dset, gmmred_red_bright_dset, 'YAP', 'Chlor','GMM red', 'red dark spots', 'red bright spots',Pixel_size)   
        #Version for time-resolved
        gc.collect()
        gmmse_red1_darkse_dset, gmmse_red1_brightse_dset, gmmse_se1_dark_dset, gmmse_se1_bright_dset, darkse_pct, brightse_pct, means, covars, weights = gmmone_tr_in_masked_channel(se1_dset_cut, red1_dset_cut)     
        #gmmse_red1_darkse_dset, gmmse_red1_brightse_dset, gmmse_se1_dark_dset, gmmse_se1_bright_dset, darkse_pct, brightse_pct = thr_otsu_tr_in_masked_channel(se1_dset_cut, red1_dset_cut)     

        mycode = str(let[index]) +'SEchannelGMM = tempfile.NamedTemporaryFile(delete=False)'
        exec(mycode)
        np.savez(str(let[index]) +'SEchannelGMM', bright = gmmse_se1_bright_dset, means = means, covars = covars, weights = weights)
        
        #klklklk
                
        
        #do_ana0lysis(red1_dset_cut, se1_dset_cut, gmmse_red1_dark_dset, gmmse_red1_bright_dset, gmmse_se1_dark_dset, gmmse_se1_bright_dset, 'CL', 'SE','GMM SE', 'SE dark spots', 'SE bright spots',Pixel_size)
        #only for plots of intensity
        del gmmse_se1_dark_dset#, gmmse_se1_bright_dset    
        
        gmmse_red1_darkse_dset = np.array(gmmse_red1_darkse_dset, dtype=np.float32)
        gmmse_red1_brightse_dset = np.array(gmmse_red1_brightse_dset, dtype=np.float32)
        gmmse_se1_bright_dset  = np.array(gmmse_se1_bright_dset, dtype=np.float32)
        
        gc.collect()
        gmmse_red1_darkse_dset_for_4D, gmmse_red1_brightse_dset_for_4D, blah, blup, darkse_pct2, brightse_pct2 =  gmmone_tr_in_masked_channel(se1_dset_cut, red1_dset_cut_all, imagemasked_is_4D=True) 
        #gmmse_red1_darkse_dset_for_4D, gmmse_red1_brightse_dset_for_4D, blah, blup, darkse_pct2, brightse_pct2 =  thr_otsu_tr_in_masked_channel(se1_dset_cut, red1_dset_cut_all, imagemasked_is_4D=True) 
        
#        mycode = 'Redbright = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez('Redbright', data = gmmse_red1_brightse_dset_for_4D/brightse_pct2)
        
#        mycode = 'Bluebright = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez('Bluebright', data = gmmse_red1_brightse_dset_for_4D/brightse_pct2)        
        
        del blah, blup, darkse_pct2, brightse_pct2 #delete all SE masks 
        gc.collect()
        gmmse_red1_darkse_dset_for_4D = np.array(gmmse_red1_darkse_dset_for_4D, dtype=np.float32)
        gmmse_red1_brightse_dset_for_4D = np.array(gmmse_red1_brightse_dset_for_4D, dtype=np.float32)
        
        # BLUE
        gc.collect()
        gmmse_blue1_darkse_dset, gmmse_blue1_brightse_dset, gmmse_se1_dark_dset, gmmse_se1_bright_dset, darkse_pct, brightse_pct, means, covars, weights = gmmone_tr_in_masked_channel(se1_dset_cut, blue1_dset_cut)             
        del gmmse_se1_dark_dset#, gmmse_se1_bright_dset    
        gmmse_blue1_darkse_dset = np.array(gmmse_blue1_darkse_dset, dtype=np.float32)
        gmmse_blue1_brightse_dset = np.array(gmmse_blue1_brightse_dset, dtype=np.float32)
    
        gc.collect()
        gmmse_blue1_darkse_dset_for_4D, gmmse_blue1_brightse_dset_for_4D, blah, blup, darkse_pct2, brightse_pct2 =  gmmone_tr_in_masked_channel(se1_dset_cut, blue1_dset_cut_all, imagemasked_is_4D=True) 
        del blah, blup, darkse_pct2, brightse_pct2 #delete all SE masks 
        gc.collect()
        gmmse_blue1_darkse_dset_for_4D = np.array(gmmse_blue1_darkse_dset_for_4D, dtype=np.float32)
        gmmse_blue1_brightse_dset_for_4D = np.array(gmmse_blue1_brightse_dset_for_4D, dtype=np.float32)
        
    else:
        pass
#        print('NOT doing gmm se') #assume all is bright in CL
#        gmmse_red1_brightse_dset = red1_dset_cut
#        gmmse_red1_darkse_dset = 0*red1_dset_cut #or could give 0 vector
#        darkse_pct = 1.0
#        brightse_pct = 0.0
#        gmmse_red1_darkse_dset_for_4D = np.array(red1_dset_cut_all, dtype=np.float32)
    #    gmmse_red1_brightse_dset_for_4D = np.array(red1_dset_cut_all, dtype=np.float32)
        
#        mycode = 'Bluebright = tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez('Bluebright', data = gmmse_red1_brightse_dset_for_4D/brightse_pct)
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    #### 3) GMM with independent masks in both channels
    #if do_gmmboth_dset:
    #    print('doing gmm both')
    #    gmmboth_blue_dark_dset, gmmboth_blue_bright_dset, gmmboth_red_dark_dset, gmmboth_red_bright_dset = gmmboth(red_dset_cut, blue_dset_cut)
    #    do_analysis(blue_dset_cut, red_dset_cut, gmmboth_blue_dark_dset, gmmboth_blue_bright_dset, gmmboth_red_dark_dset, gmmboth_red_bright_dset, 'YAP', 'Chlor','GMM both', 'dark spots', 'bright spots',Pixel_size)
    #
    #### 4) Threshold adapative
    #if do_threshold_adaptive:
    #   print('doing thr adap')
    #   blocksize = 50
    #   offset = 0
    #   th_below_blue, th_above_blue, th_below_red, th_above_red = threshold_adaptive_dset(red_dset_cut, blue_dset_cut,blocksize, offset)
    #   do_analysis(blue_dset_cut, red_dset_cut, th_below_blue, th_above_blue, th_below_red, th_above_red, 'YAP', 'Chlor','Threshold adaptive' + '(blocksize, offset =' + str(blocksize) + ', ' + str(offset) + ')', 'below thr', 'above thr',Pixel_size)
    #
    #### 5) random_walker not yet working
    ### http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_random_walker_segmentation.html#example-segmentation-plot-random-walker-segmentation-py
    #if do_random_walker:
    #    print('doing random walk')
    #    cutofflow = 0.89
    #    cutoffhigh = 0.9
    #    rw_below_blue, rw_above_blue, rw_below_red, rw_above_red = random_walker_dset(red_dset_cut, blue_dset_cut,cutofflow, cutoffhigh)
    #    do_analysis(blue_dset_cut, red_dset_cut, rw_below_blue, rw_above_blue, rw_below_red, rw_above_red, 'YAP', 'Chlor','Random walker'+ '(cutoffs high, low =' + str(cutoffhigh) + ', ' + str(cutofflow) + ')', 'background', 'foreground',Pixel_size)
    #
    #### 6) Otsu thresholding 
    #if do_otsu:
    #   print('doing otsu')
    #   ot_below_blue, ot_above_blue, ot_below_red, ot_above_red = thr_otsu(red_dset_cut, blue_dset_cut)
    #   do_analysis(blue_dset_cut, red_dset_cut, ot_below_blue, ot_above_blue, ot_below_red, ot_above_red, 'YAP', 'Chlor','Otsu threshold', 'background', 'foreground',Pixel_size)
    #   
    #log_dog_doh(blue_dset_cut)
    
    #log_dog_doh(blue_dset_cut)
     
    ########### OUTPUT 
     
    ###### HERE: I have  red1_dset_cut, red1_dset_cut_all, se1_dset_cut, gmmse_red_darkse_dset, gmmse_red_brightse_dset, gmmse_se_darkse_dset, gmmse_se_brightse_dset
     
    ###FIG2
    #frame to be shown in static frame
    #init_plot_no = center_cl_index #around 27 or 28
    #plot_nonvideo_reg(titulo, gmmse_se1_bright_dset, red1_dset_cut,gmmse_red1_darkse_dset, gmmse_red1_brightse_dset, red1_dset_cut_all, se1_dset_cut,  gmmse_red1_brightse_dset_for_4D, gmmse_red1_darkse_dset_for_4D, Time_bin,fastfactor,nominal_time_on,Pixel_size[index],darkse_pct, brightse_pct,name_str[index] ,init_plot_no,major_ticks,unit = 'kHz')
#    del se1_dset_cut
#    gc.collect()
#    ###END FIG2
#    
#    ###FIG3
#    #fig_no = '-3plots'
#    #plot_expt_by_expt_behaviour(titulo + ', signal pixels', gmmse_red1_darkse_dset_for_4D/darkse_pct, Time_bin, nominal_time_on,fastfactor,'y',major_ticks,dark_dset=gmmse_red1_brightse_dset_for_4D/brightse_pct, plot_dark=True) #pass titulo as well
#    
#    #two lines were uncommented; now trying to delete later
#    #del gmmse_red1_brightse_dset_for_4D
#    #gc.collect()
#   # plot_expt_by_expt_behaviour(titulo + ', signal pixels', gmmse_red1_darkse_dset_for_4D/darkse_pct, Time_bin, nominal_time_on,fastfactor,'y',major_ticks,dark_dset=None, plot_dark=False,unit='kHz') #pass titulo as well
#    #del gmmse_red1_brightse_dset_for_4D
#    
#    #multipage('ZZZ' + name_str[index] + fig_no + '.pdf',dpi=80)
#    ###END FIG3
#    
#    #del gmmse_red1_darkse_dset_for_4D
#    gc.collect()
    
    ###FIG4
    #fig_no = '-3plots'
    #start_of_transient = 33; #82- cut_at_beginning + 1 #time-bin 75 + 300ns/40ns = 75 + ~8 = 83
    #last_pt_offset = -10 #sometimes use -1, last point, but sometimes this gives 0. -10 seems to work
#    if index == 7000:  ### 7 was a problem for red, added a thousand
#        print('core only, sample B')
#        init_guess = [np.average(gmmse_red1_darkse_dset[start_of_transient,:,:])/darkse_pct, 1.0, np.average(gmmse_red1_darkse_dset[last_pt_offset,:,:])/darkse_pct, np.average(gmmse_red1_darkse_dset[start_of_transient,:,:])/darkse_pct , 0.1] #e init was 0.5, d was zero before I made == a
#    elif index == 10: #### 10 (oleci acid) was a single not double, added a thousand
#        print("oleic")
#        init_guess = [np.average(gmmse_red1_darkse_dset[start_of_transient,:,:])/darkse_pct,0.05, np.average(gmmse_red1_darkse_dset[last_pt_offset,:,:])/darkse_pct, np.average(gmmse_red1_darkse_dset[start_of_transient,:,:])/darkse_pct, 0.25] #e init was 0.5
#    else:
#    init_guess = [np.average(gmmse_red1_darkse_dset[start_of_transient,:,:])/darkse_pct, 1.0, np.average(gmmse_red1_darkse_dset[last_pt_offset,:,:])/darkse_pct, np.average(gmmse_red1_darkse_dset[start_of_transient+50,:,:])/darkse_pct, 0.1] #e init was 0.5
    #if do_gmmse_dset is False:
     #  brightse_pct = 0.01 #just so that I don't have a division by 0 in the function argument below!!!!
    #b,e,be,ee = calcdecay(gmmse_red1_darkse_dset[start_of_transient::,:,:]/darkse_pct, time_detail= Time_bin*1e-9*fastfactor,titulo='Cathodoluminescence rate decay, bi-exponential fit, \n ' + titulo ,single=False,other_dset1=None ,other_dset2=None,init_guess=init_guess,unit='kHz')    
  #  b,be, be, ee = calcdecay(gmmse_red1_darkse_dset[start_of_transient::,:,:]/darkse_pct, time_detail= Time_bin*1e-9*fastfactor,titulo='Cathodoluminescence rate decay, single exponential fit, \n ' + titulo ,single=False,other_dset1=red1_dset_cut[start_of_transient::,:,:]/1.0 ,other_dset2=gmmse_red1_brightse_dset[start_of_transient::,:,:]/brightse_pct,init_guess=init_guess,unit='kHz')
    
#    tau_single[index] = b
#    tau_single_error[index] = be
#    tau_bi[index,:] = [b,e]
#    tau_bi_error[index,:] = [be,ee]
    
    #for plots of dose, in the Er 60%:
#    if index >= 9: #8,9,10,11: the 4 from Er 60% 
#        mycode = 'ZZZZZZEr60' + str(index) + '= tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez('ZZZZZZEr60' + str(index), data = np.average(gmmse_red1_darkse_dset_for_4D/darkse_pct, axis=(1,2,3)))
        
    #for plots of tau as a function of number of experiments in Er 2%, sample A (index0)
 #   if index is 0:
     #   pass
#        print('series')
#        calcdecay_series(gmmse_red1_darkse_dset_for_4D[:,start_of_transient::,:,:]/darkse_pct, time_detail= Time_bin*1e-9*fastfactor,titulo='Cathodoluminescence rate decay, bi-exponential fit, \n ' + titulo ,single=False,nominal_time_on=nominal_time_on,fastfactor=fastfactor,other_dset1=red1_dset_cut_all[:,start_of_transient::,:,:]/1.0 ,other_dset2=gmmse_red1_brightse_dset_for_4D[:,start_of_transient::,:,:]/brightse_pct,init_guess=init_guess)    
#        print('each')        
#        calcdecay_each(gmmse_red1_darkse_dset_for_4D[:,start_of_transient::,:,:]/darkse_pct, time_detail= Time_bin*1e-9*fastfactor,titulo='Cathodoluminescence rate decay, bi-exponential fit, \n ' + titulo ,single=False,nominal_time_on=nominal_time_on,fastfactor=fastfactor,other_dset1=red1_dset_cut_all[:,start_of_transient::,:,:]/1.0 ,other_dset2=gmmse_red1_brightse_dset_for_4D[:,start_of_transient::,:,:]/brightse_pct,init_guess=init_guess)    


#    del gmmse_red1_darkse_dset_for_4D, gmmse_red1_brightse_dset_for_4D #last one is trying to delete later than line372
#    gc.collect()

#    start_of_transient = 0    
#    
#    mycode = str(let[index]) +'ZZZBlueDecay = tempfile.NamedTemporaryFile(delete=False)'
#    exec(mycode)
#    np.savez(str(let[index]) +'ZZZBlueDecay', data = gmmse_blue1_darkse_dset[start_of_transient::,:,:]/darkse_pct)
#    
#    mycode = str(let[index]) +'ZZZRedDecay = tempfile.NamedTemporaryFile(delete=False)'
#    exec(mycode)
#    np.savez(str(let[index]) +'ZZZRedDecay', data = gmmse_red1_darkse_dset[start_of_transient::,:,:]/darkse_pct)
    
    #multipage(str(let[index]) +'RedZZZ.pdf',dpi=80)
    
    ###END FIG4
    
    #plt.show()
    plt.close('all')
    
#write a temporary file with all values
#outfile = tempfile.NamedTemporaryFile(delete=False)
#np.savez(outfile, tau_single=tau_single, tau_single_error=tau_single_error, tau_bi=tau_bi, tau_bi_error=tau_bi_error)
    
######################################## Plot with dose for different apertures

##files below exist
b_array_red = np.zeros(6)
be_array_red = np.zeros(6)
e_array_red = np.zeros(6)
ee_array_red = np.zeros(6)
b_array_blue = np.zeros(6)
be_array_blue = np.zeros(6)
e_array_blue = np.zeros(6)
ee_array_blue = np.zeros(6)
blue_int_array = np.zeros(6)
red_int_array = np.zeros(6)

time_bins = [2000,2000,2000,2000,2000,2000]

fig40= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig40.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
plt.rc('text.latex', preamble=r'\usepackage{xcolor}') #not working
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{xcolor}')  #not working
plt.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]  #not working
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')   
fig40.suptitle('Time-resolved cathodoluminescence, NaYF$_4$ from Bella \n (10kV, 30$\mu$m aperture, 2$\mu$s time bin)',fontsize = 18)     

index_tuple1 = [0,0,1,1,2,2]
index_tuple2 = [0,1,0,1,0,1]
Time_bin = time_bins
initbin = [100,100,100,100,100,100]

listofindex =np.arange(1,6) # #[0,1,2]  [5] 
index = 1
#if index is 0:
for index in listofindex:
    
    file1    = h5py.File(nametr[index] + '.hdf5', 'r')  
    red0_dset  = file1['/data/Counter channel 1 : PMT red/PMT red time-resolved/data']#10 Scany points X10 frames x 150 tr pts x250 x 250 pixels
    Pixel_size = red0_dset.attrs['element_size_um'][1]*1000.0 #saved attribute is in micron; converting to nm
    Ps = str("{0:.2f}".format(Pixel_size)) #pix

    ### 593 dichroic
    #se = np.load(str(let[index]) +'SEchannel.npz') 
    #segmm = np.load(str(let[index]) +'SEchannelGMM.npz') 
    red = np.load(str(let[index]) +'Redbright.npz') 
    blue = np.load(str(let[index]) +'Bluebright.npz') 
    #il = np.load(str(let[index]) +'ILchannel.npz') 
    
    fsizetit = 18 #22 #18
    fsizepl = 16 #20 #16
    sizex = 8 #10 #8
    sizey = 6# 10 #6
    dpi_no = 80
    lw = 2
    
    #Pixel_size = 3.1e-9
    length_scalebar = 1000.0 #in nm (1000nm == 1mum)
    scalebar_legend = '1$\mu$m'
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size))
    titulo = 'Region ' + let[index] + ', (10kV, 30$\mu$m aperture, ' + str(time_bins[index]) +'ns time bins, ' + str(Ps)+ 'nm pixels, blue/red: $</>$ 593nm)'
    
    
#    fig40= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
#    fig40.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
#    plt.rc('text.latex', preamble=r'\usepackage{xcolor}') #not working
#    plt.rc('text', usetex=True)
#    plt.rc('text.latex', preamble=r'\usepackage{xcolor}')  #not working
#    plt.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]  #not working
#    plt.rc('font', family='serif')
#    plt.rc('font', serif='Palatino')         
    
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    #plt.rc('font', serif='Palatino')  
    #if (index is 3) or (index is 6): 
      #  plt.suptitle("Registration and segmentation (model: simple avg.) of cathodoluminescence signal using SE channel, \n" + titulo,fontsize=fsizetit)
    #else:
   # plt.suptitle("Registration and segmentation (model: 2-GMM) of cathodoluminescence signal using SE channel, \n" + titulo,fontsize=fsizetit)

       
    gc.collect()
    
#    ax1 = plt.subplot2grid((2,2), (0, 0), colspan=1)
#    ax1.set_title('SE channel',fontsize=fsizepl) #as per accompanying txt files
#    plt.imshow(se['data'],cmap=cm.Greys_r)
#    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#    ax1.add_artist(sbar)
#    plt.axis('off')
#    
#    gc.collect()
#    ax1 = plt.subplot2grid((2,3), (0, 1), colspan=1)
#    ax1.set_title('SE channel, signal pixels',fontsize=fsizepl)
#    hlp = segmm['bright']
#    hlp[~np.isnan(hlp)] = 0.0
#    hlp[np.isnan(hlp)] = 1.0
#    im = plt.imshow(hlp,cmap=cm.Greys) #or 'OrRd'
#    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4)
#    ax1.add_artist(sbar)
#    plt.axis('off')
    
#    def make_gaussian(means, covars, weights, data, max_hist, no_pts=200):
#        
#        array_x = np.zeros([len(means), no_pts])
#        array_y = np.zeros([len(means), no_pts])
#        
#        for j in np.arange(0,len(means)):
#            array_x[j,:] = np.linspace(np.min(data),np.max(data),no_pts)
#            array_y[j,:] = weights[j] * max_hist * np.exp( -(array_x[j,:] - means[j])**2/(2*covars[j])  )
#            
#        return array_x, array_y
    
    
    #box = ax1.get_position()
    #ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    #axColor = plt.axes([box.x0, box.y0*1.1 , box.width,0.01*10 ])   
    #no_bins = 200 
    #n = axColor.hist(se['data'].flatten(),bins=no_bins)
    #
    #array_x, array_y = make_gaussian(segmm['means'][:,0], segmm['covars'][:,0],segmm['weights'], se['data'], np.max(n[0]), no_pts=200)
    #
    #axColor.plot(array_x[0],array_y[0])
    #axColor.plot(array_x[1],array_y[1])
    
    ############
#    import skimage.morphology
#    from skimage.morphology import watershed
#    from skimage.feature import peak_local_max
##    image =  np.abs(1-hlp)#hlp.astype(bool) #np.logical_or(mask_circle1, mask_circle2)
#    from scipy import ndimage
#    distance = ndimage.distance_transform_edt(image)
#    local_maxi = peak_local_max(distance, num_peaks = 1, indices = False, footprint=np.ones((50,50)),labels=image) #footprint = min dist between maxima to find #footprint was 25,25
#    markers = skimage.morphology.label(local_maxi)
#    labels_ws = watershed(-distance, markers, mask=image)
    #plt.figure()
    #ax1 = plt.subplot2grid((1,4), (0,0))
    #ax1.imshow(image)
    #ax2 = plt.subplot2grid((1,4), (0,1))
    #ax2.imshow(np.log(distance))
    #ax2 = plt.subplot2grid((1,4), (0,2))
    #ax2.imshow(markers)
    #ax2 = plt.subplot2grid((1,4), (0,3))
    #ax2.imshow(labels_ws)
    #plt.show()
    
#    ax1 = plt.subplot2grid((2,3), (0, 2), colspan=1)
#    ax1.set_title('Segmented NPs',fontsize=fsizepl)
#    im = plt.imshow(labels_ws,cmap=cm.Greys) #or 'OrRd'
#    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
#    ax1.add_artist(sbar)
#    plt.axis('off')
#    
#    ax1 = plt.subplot2grid((2,3), (1, 0), colspan=1)
#    ax1.spines['right'].set_visible(False)
#    ax1.spines['top'].set_visible(False)
#    ax1.xaxis.set_ticks_position('bottom')
#    ax1.yaxis.set_ticks_position('left')
#    
#    #TEST
##    if index is 0: #hlp does not do anything; segmentation is bad
##        hlp = segmm['bright']
##        hlp[~np.isnan(hlp)] = 1.0
##        hlp[np.isnan(hlp)] = 1.0 #hlp does not do anything; segmentation is bad
##    else:    #working for index = 1, 
##        hlp = segmm['bright']
##        hlp[~np.isnan(hlp)] = 1.0
#        
#        
#    hlp = segmm['bright']
#    hlp[~np.isnan(hlp)] = 1.0    
        

    #redn = np.average(red['data'],axis = 0)
    #bluen = np.average(blue['data'],axis = 0)
    #for jj in np.arange(0,redn.shape[0]):
    #    print(jj)
    #    redn[jj,:,:] = redn[jj,:,:] * hlp
    #    bluen[jj,:,:] = bluen[jj,:,:] * hlp
    #    
    #plt.plot(np.arange(0,91)*Time_bin/1e3,np.nanmean(redn,axis = (1,2)),c='r',label='Red photons ($>$ 593nm)',lw=3) #in mus, in MHz
    #plt.plot(np.arange(0,91)*Time_bin/1e3,np.nanmean(bluen,axis = (1,2)),c='b',label='Blue photons ($<$ 593nm)',lw=3) #in mus, in MHz
  #  plt.plot(np.arange(0,150)*Time_bin/1e3,np.nanmean(red['data']*hlp,axis = (0,2,3)),c='r',label='Red photons ($>$ 593nm)',lw=3) #in mus, in MHz
 #   plt.plot(np.arange(0,150)*Time_bin/1e3,np.nanmean(blue['data']*hlp,axis = (0,2,3)),c='b',label='Blue photons ($<$ 593nm)',lw=3) #in mus, in MHz
    
    #plt.plot(np.arange(0,91)*Time_bin/1e3,np.average(red['data'],axis = (0,2,3)),c='r',label='Red photons ($>$ 593nm)',lw=3) #in mus, in MHz
    #plt.plot(np.arange(0,91)*Time_bin/1e3,np.average(blue['data'],axis = (0,2,3)),c='b',label='Blue photons ($<$ 593nm)',lw=3) #in mus, in MHz
#    ax1.axvspan(0.5,2.3, alpha=0.25, color='yellow')
#    unit = 'kHz'
#    plt.ylabel("Average luminescence \n of each time bin, per pixel (" + unit + ")",fontsize=fsizepl)
#    plt.xlabel("Behaviour of e-beam during each experiment: \n 1.8-ON + 4-OFF ($\mu$s)",fontsize=fsizepl)
#    plt.legend() 
#    major_ticks0 = [0.5,2.3,4,6]
#    ax1.set_xticks(major_ticks0) 
#    #ax1.set_yticks([15,30,45]) 
#    plt.xlim([0,6])
    
    ax1 = plt.subplot2grid((3,2), (index_tuple1[index], index_tuple2[index]), colspan=1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_title(str(let[index]),fontsize=fsizepl)
    
    datared = red['data']
    datablue = blue['data']
    
    if True is False:
        pass
    else:
        #initbin = 10
        datared = datared[initbin[index]:,:,:]
        datablue = datablue[initbin[index]:,:,:]
    
    
    
    fastfactor = 1
    last_pt_offset = -20 #sometimes use -1, last point, but sometimes this gives 0. -10 seems to work
    init_guess = [np.average(datared[0,:,:]), 0.01, np.average(datared[last_pt_offset,:,:]), np.average(datared[-30,:,:]), 0.005] #e init was 0.5
    
    init_guess2 = [np.average(datablue[0,:,:]), 1.0, np.average(datablue[last_pt_offset,:,:]), np.average(datablue[-30,:,:]), 0.005] #e init was 0.5
    #init_guess2 = [np.average(datablue[0,:,:]), 1.0, 0.01, np.average(datablue[-30,:,:]), 0.005] #e init was 0.5
#    
    # We did not need to fix the offset c in all fits
    hlp_red = datared 
    hlp_blue = datablue 
    if index == 0: #needs to be reworked out
       init_guess = [np.nanmean(hlp_red[0,:,:]), 1.0, np.nanmean(hlp_red[last_pt_offset,:,:]), np.nanmean(hlp_red[10:50,:,:]), 10.0 ] #RED
       init_guess2 = [np.nanmean(hlp_blue[0,:,:]), 1.0,np.nanmean(hlp_blue[last_pt_offset,:,:]), np.nanmean(hlp_blue[10:50,:,:]), 10.0 ] #BLUE
       #init_guess = [1.0, 1.0, np.nanmean(hlp_red[last_pt_offset,:,:]), 0.1, 10.0 ] #RED
       #init_guess2 = [10.0, 1.0,np.nanmean(hlp_blue[last_pt_offset,:,:]), 1.0, 10.0 ] #BLUE
      
    if index == 1:
        
        init_guess = [np.nanmean(hlp_red[0,:,:]), 1.0, np.nanmean(hlp_red[last_pt_offset,:,:]), np.nanmean(hlp_red[10:50,:,:]), 20.0 ] #RED
        init_guess2 = [np.nanmean(hlp_blue[0,:,:]), 1.0,np.nanmean(hlp_blue[last_pt_offset,:,:]), np.nanmean(hlp_blue[10:50,:,:]), 20.0 ] #BLUE
     
    if index == 2:
        
        init_guess = [np.nanmean(hlp_red[0,:,:]), 1.0, np.nanmean(hlp_red[last_pt_offset,:,:]), np.nanmean(hlp_red[10:50,:,:]), 20.0 ] #RED
        init_guess2 = [np.nanmean(hlp_blue[0,:,:]), 1.0,np.nanmean(hlp_blue[last_pt_offset,:,:]), np.nanmean(hlp_blue[10:50,:,:]), 20.0 ] #BLUE
        
    if index == 3:
        
        init_guess = [np.nanmean(hlp_red[0,:,:]), 1.0, np.nanmean(hlp_red[last_pt_offset,:,:]), np.nanmean(hlp_red[10:50,:,:]), 20.0 ] #RED
        init_guess2 = [np.nanmean(hlp_blue[0,:,:]), 1.0,np.nanmean(hlp_blue[last_pt_offset,:,:]), np.nanmean(hlp_blue[10:50,:,:]), 20.0 ] #BLUE
        
    if index == 4:
        init_guess = [np.nanmean(hlp_red[0,:,:]), 1.0, np.nanmean(hlp_red[last_pt_offset,:,:]), np.nanmean(hlp_red[10:50,:,:]), 20.0 ] #RED
        init_guess2 = [np.nanmean(hlp_blue[0,:,:]), 1.0,np.nanmean(hlp_blue[last_pt_offset,:,:]), np.nanmean(hlp_blue[10:50,:,:]), 20.0 ] #BLUE
        
    if index == 5:
        init_guess = [np.nanmean(hlp_red[0,:,:]), 1.0, np.nanmean(hlp_red[last_pt_offset,:,:]), np.nanmean(hlp_red[10:50,:,:]), 20.0 ] #RED
        init_guess2 = [np.nanmean(hlp_blue[0,:,:]), 1.0,np.nanmean(hlp_blue[last_pt_offset,:,:]), np.nanmean(hlp_blue[10:50,:,:]), 20.0 ] #BLUE

    if index == 6:
        init_guess = [np.nanmean(hlp_red[0,:,:]), 1.0, np.nanmean(hlp_red[last_pt_offset,:,:]), np.nanmean(hlp_red[10:50,:,:]), 20.0 ] #RED
        init_guess2 = [np.nanmean(hlp_blue[0,:,:]), 1.0,np.nanmean(hlp_blue[last_pt_offset,:,:]), np.nanmean(hlp_blue[10:50,:,:]), 20.0 ] #BLUE

    hlpred = np.nanmean(red['data'][initbin[index]:,:,:],axis = (1,2))
    hlpblue = np.nanmean(blue['data'][initbin[index]:,:,:],axis = (1,2))
    
#    error_arrayr = np.nanstd(hlpred,axis = 0)
#    error_arrayb = np.nanstd(hlpblue, axis = 0)
#    error_arrayr[error_arrayr < 0.05 * np.max(hlpred)]  = 0.05 * np.max(hlpred) #arbitrary 5% of max
#    error_arrayb[error_arrayb < 0.05 * np.max(hlpblue)]  = 0.05 * np.max(hlpblue)
    #b,e,be,ee = calcdecay_subplot2(datared, time_detail= Time_bin*1e-9*fastfactor,titulo='Cathodoluminescence rate decay, bi-exponential fit, \n ' + titulo ,single=False,other_dset2=datablue ,other_dset1=None,init_guess=init_guess,unit='kHz',init_guess2=init_guess2)    
    b,e,be,ee,b2,e2,be2,ee2 = calcdecay_subplot_nan(datared, time_detail= Time_bin[index]*1e-9*fastfactor,titulo='Cathodoluminescence rate decay, bi-exponential fit, \n ' + titulo ,single=False,other_dset2=datablue ,other_dset1=None,init_guess=init_guess,unit='kHz',init_guess2=init_guess2) #,error_array=error_arrayr, error_array2=error_arrayb)    
    
    b_array_red[index] = b
    e_array_red[index] = e
    be_array_red[index] = be    
    ee_array_red[index] = ee  
    b_array_blue[index] = b2
    e_array_blue[index] = e2
    be_array_blue[index] = be2    
    ee_array_blue[index] = ee2 
    
    plt.xlim([0,1000])
    #major_ticks0 = [10,20,30,40]
    plt.ylabel("Average luminescence \n of each time bin, per pixel (kHz)",fontsize=fsizepl)
    plt.xlabel(r'Time after blanking the electron beam ($\mu$s)',  fontsize=fsizepl)
    
    #ax1.set_xticks(major_ticks0)

    ############    
    
#    ax1 = plt.subplot2grid((2,3), (1, 2), colspan=1)
#    ax1.spines['left'].set_visible(False)
#    ax1.spines['top'].set_visible(False)
#    ax1.xaxis.set_ticks_position('bottom')
#    ax1.yaxis.set_ticks_position('right')
#    ax1.yaxis.set_label_position("right")
#    
#    dataALLred = red['data'][:,:,:,:]
#    dataALLblue = blue['data'][:,:,:,:]
#    nominal_time_on = 1.8
#    
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue*hlp,axis=(1,2,3)),c='b', label='From blue photons ($<$ 593nm)',linestyle='None', marker='o',markersize=4) #in mus, in MHz
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred*hlp,axis=(1,2,3)),c='r', label='From red photons ($>$ 593nm)',linestyle='None', marker='o',markersize=4) #in mus, in MHz
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))*np.ones([No_experiments[index]]),c='b', linewidth= lw) #in mus, in MHz
#    plt.plot(np.arange(1,No_experiments[index]+1)*nominal_time_on*fastfactor,np.nanmean(dataALLred*hlp,axis=(0,1,2,3))*np.ones([No_experiments[index]]),c='r',  linewidth= lw) #in mus, in MHz
#    
#    red_int_array[index] = np.nanmean(dataALLred*hlp,axis=(0,1,2,3))
#    blue_int_array[index] = np.nanmean(dataALLblue*hlp,axis=(0,1,2,3))
#    cumu_red[index,:] =  np.nanmean(dataALLred*hlp,axis=(1,2,3))
#    cumu_blue[index,:] =  np.nanmean(dataALLblue*hlp,axis=(1,2,3))
#    il_data[index,:] = np.average(il['data'], axis = (1,2)) #take average per frame
#    
#    plt.ylabel("Average luminescence \n for each experiment, per pixel  (kHz)",fontsize=fsizepl)
#    plt.xlabel("Cumulative e-beam exposure time \n per pixel (nominal, $\mu$s)",fontsize=fsizepl)
#    
#    #major_ticks = [25,50,75,nominal_time_on*dset.shape[0]*fastfactor]
#   # major_ticks = [5,10,15,20]
#    #ax1.set_xticks(major_ticks) 
#    #plt.legend()
#    plt.xlim([nominal_time_on - 1.0,nominal_time_on*No_experiments[index]*fastfactor +1.0])
    
  
###comment/uncomment below
multipage_longer('ZZZZ-'+ let[index] + '.pdf',dpi=80)    

mycode = 'Red_int_array = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Red_int_array', data = red_int_array)

mycode = 'Blue_int_array = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Blue_int_array', data = blue_int_array)

mycode = 'B_array_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('B_array_red', data = b_array_red)

mycode ='Be_array_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Be_array_red', data = be_array_red)

mycode = 'E_array_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('E_array_red', data = e_array_red)

mycode = 'Ee_array_red = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Ee_array_red', data = ee_array_red)

mycode = 'B_array_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('B_array_blue', data = b_array_blue)

mycode ='Be_array_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Be_array_blue', data = be_array_blue)

mycode = 'E_array_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('E_array_blue', data = e_array_blue)

mycode = 'Ee_array_blue = tempfile.NamedTemporaryFile(delete=False)'
exec(mycode)
np.savez('Ee_array_blue', data = ee_array_blue)


#####FIG WITH SUMMARY   
fastfactor = 1

Red_int_array = np.load('Red_int_array.npz') 
red_int_array = Red_int_array['data']
Blue_int_array = np.load('Blue_int_array.npz') 
blue_int_array = Blue_int_array['data']

B_array_red= np.load('B_array_red.npz')
b_array_red = B_array_red['data']  
Be_array_red = np.load('Be_array_red.npz')
be_array_red = Be_array_red['data'] 
E_array_red = np.load('E_array_red.npz')
e_array_red = E_array_red['data']   
Ee_array_red = np.load('Ee_array_red.npz')
ee_array_red = Ee_array_red['data']   

B_array_blue= np.load('B_array_blue.npz')
b_array_blue = B_array_blue['data']  
Be_array_blue = np.load('Be_array_blue.npz')
be_array_blue = Be_array_blue['data'] 
E_array_blue = np.load('E_array_blue.npz')
e_array_blue = E_array_blue['data']   
Ee_array_blue = np.load('Ee_array_blue.npz')
ee_array_blue = Ee_array_blue['data'] 
    
fig41= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig41.set_size_inches(1200./fig41.dpi,900./fig41.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    
fig41.suptitle('Lifetimes fitted after 1000$\mu$s, NaYF$_4$ CS from Bella \n (10kV, 30$\mu$m aperture, 12kX, 2$\mu$s time bins)',fontsize=fsizetit)     

ax2 = plt.subplot2grid((2,1), (0,0), colspan=1, rowspan=1)
ax1 = plt.subplot2grid((2,1), (1,0), colspan=1, sharex=ax2)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

#ax2.set_ylim([0.0,10.0])
#ax2.set_yticks([2.5, 5, 7.5, 10.0])
#ax1.set_yticks([0.025, 0.05, 0.075,0.1])
#ax1.set_ylim([0,0.1])
ax1.set_xticks([20,37,40,43,80])
ax1.set_xticklabels(['20','40','40 \n (24kX)','40 \n (48kX)','80'])
x_vec = [10,20,37,40,43,80] #(6,10)

ax1.errorbar(x_vec, b_array_red, yerr=be_array_red, fmt='ro',markersize=5)
ax2.errorbar(x_vec, e_array_red, yerr=ee_array_red, fmt='ro', markersize=10)
ax1.errorbar(x_vec, b_array_blue, yerr=be_array_blue, fmt='bo',markersize=5)
ax2.errorbar(x_vec, e_array_blue, yerr=ee_array_blue, fmt='bo', markersize=10)
ax1.set_ylabel('Shorter time constant ($\mu$s)',fontsize=fsizepl)
ax2.set_ylabel('Longer time constant ($\mu$s)',fontsize=fsizepl)
plt.xlim([15,85])
#ax2.legend(fontsize=fsizepl)
ax1.set_xlabel('Er content (%)',fontsize=fsizepl)

plt.sca(ax1) 

plt.setp(ax2.get_xticklabels(), visible=False)

#ax3 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
#plt.plot(np.arange(1,61)*nominal_time_on*fastfactor,cumu_blue.flatten(),c='b', label='From blue photons ($<$ 593nm)',linestyle='None', marker='o',markersize=5)
#plt.plot(np.arange(1,61)*nominal_time_on*fastfactor,cumu_red.flatten(),c='r', label='From red photons ($>$ 593nm)',linestyle='None', marker='o',markersize=5)
#plt.xlabel("Cumulative e-beam exposure time \n per pixel (nominal, $\mu$s)",fontsize=fsizepl)
#plt.ylabel("Average luminescence \n for each frame, per pixel (KHz)",fontsize=fsizepl)
#plt.xlim([0,110])
#plt.ylim([0,80])
#ax3.spines['right'].set_visible(False)
#ax3.spines['top'].set_visible(False)
#ax3.xaxis.set_ticks_position('bottom')
#ax3.yaxis.set_ticks_position('left')
#ax3.arrow(20, 75, 60,0, head_width=2, head_length=6, fc='k', ec='k')
#ax3.text(25, 78, 'stage temperature decreases', fontsize=15)
#ax4 = ax3.twiny()
#ax3Ticks = ax3.get_xticks()   
#ax4Ticks = ax3Ticks
#ax4.set_xticks(ax4Ticks)
#ax4.set_xbound(ax3.get_xbound())
#ax4.set_xticklabels(x_vec.flatten())
#ax4.set_xlabel('Temperature (C)',fontsize=fsizepl)

#ax2.set_xlabel("x-transformed")
#ax2.set_xlim(0, 60)
#ax2.set_xticks([10, 30, 40])
#ax2.set_xticklabels(['7','8','99'])

#ax5 = plt.subplot2grid((2,2), (1,1), colspan=1, rowspan=1)
#plt.plot(x_vec,blue_int_array,c='b', label='From blue photons ($<$ 593nm)',linestyle='None', marker='o',markersize=10)
#plt.plot(x_vec,red_int_array,c='r', label='From red photons ($>$ 593nm)',linestyle='None', marker='o',markersize=10)
#plt.legend(loc='best')
#plt.ylabel("Average luminescence \n for each experiment, per pixel (KHz)",fontsize=fsizepl)
#ax5.spines['right'].set_visible(False)
#ax5.spines['top'].set_visible(False)
#ax5.xaxis.set_ticks_position('bottom')
#ax5.yaxis.set_ticks_position('left')
#ax5.set_xlabel('Time bin of time-resolved cathodoluminescence (nm)',fontsize=fsizepl)
#plt.xlim([40,1010])
#ax5.set_xticks([50,100,200,1000])
#plt.ylim([0,80])

multipage_longer('ZZZZZSummary.pdf',dpi=80)






