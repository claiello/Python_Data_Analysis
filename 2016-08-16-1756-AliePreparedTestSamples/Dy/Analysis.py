import os
import sys
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../../Python modules/") # necessary for the tex fonts
import scipy as sp
import scipy.misc
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
from BackgroundCorrection import *
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
titulo =  'Dy (3kV, 30$\mu$m, 40ns time bins, 36kX or 3nm pixels, blue/red $= </>$ 409nm)'

calc_blue = True
#Excitation is impulsive, 120ns per pulse,turn on at point no 80, off at point no 82

#No_pixels = 250
Time_bin = 40 #in ns; 1/clock of 25MHz 
nominal_time_on = 0.12 #time during which e-beam nominally on, in mus
totalpoints = 200 #total number of time-resolved points
### data
name = ['2016-08-16-1620_ImageSequence_DyTR_37.141kX_3.000kV_30mu_2.hdf5']
name_str = ['Dy']
na = ['2','2','20','20','40','40','40','40','80','80','00']
nl = ['A','B','A','B','A','B','A','B','A','B','A']
if calc_blue is False:
    pmt = ['PMT red','PMT red','PMT red','PMT red','PMT red','PMT red','PMT red','PMT red','PMT red','PMT red','PMT red']
    channel = ['1','1','1','1','1','1','1','1','1','1','1']
else:
    pmt = ['PMT blue','PMT blue','PMT blue','PMT blue','PMT blue','PMT blue','PMT blue','PMT blue','PMT blue','PMT blue','PMT blue']
    channel = ['2','2','2','2','2','2','2','2','2','2','2']
    name_str = [s + 'Blue' for s in name_str]
       
ap = ['30','30','30','30','30','30','30','30','30','30','30']
Pixel_size = [3.0e-09]
Ps = [3] #pixel size in nm, the numbers above with round nm precision

#no experiments to consider
No_experiments = [50,50,50,50,50,50,50,50,50,50,50,50] #only "up" in luminescence curve

tau_single = np.zeros(len(name))
tau_single_error = np.zeros(len(name))
tau_bi = np.zeros([len(name),2])
tau_bi_error = np.zeros([len(name),2])

#original
#for index in np.arange(11,11):   #11):

index = 0
if index is 1:
    
#for Er60 only: np.arange(9,12)
#for index in np.arange(12,12):

    print(index)

    file1    = h5py.File(name[index], 'r')  
    #titulo =  'Dy (3kV, 30$\mu$m, 40ns time bins, 36kX or 3nm pixels, blue/red $= </>$ 409nm)'
    se1_dset   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
    red1_dset  = file1['/data/Counter channel ' + channel[index] + ' : '+ pmt[index]+'/' + pmt[index] + ' time-resolved/data']#50 frames x 200 tr pts x250 x 250 pixels
    
    #cut part of frames with ebeam off, here 25 first points
    cut_at_beginning = 79
    red1_dset = red1_dset[:,cut_at_beginning::,:,:]
    
    #hack to go faster
    fastfactor = 1
    #se1_dset = se1_dset[0::fastfactor,:,:]
    #red1_dset = red1_dset[0::fastfactor,:,:,:]
    
    #no experiments to consider
    se1_dset = se1_dset[0:No_experiments[index]+1,:,:]
    red1_dset = red1_dset[0:No_experiments[index]+1,:,:,:]
    
    #convert to smaller data types
    se1_dset2 = np.array(se1_dset, dtype=np.float16)
    red1_dset = np.array(red1_dset) 
    #convert red1_dset to MHz!!!!!!!!!!!!!!!!!!!!!1
    red1_dset = red1_dset/1.0e3
    red1_dset = np.array(red1_dset, dtype=np.float32)  #converting the CL dset to float16 from float64 is creating infinities! bc max value is > 2^16 
    
    print('data loaded')
    
    #PLOT EXPT BY EXPT BEHAVIOUR ON ALL PIXELS
    ###############################################################################
    ###############################################################################
    ###############################################################################
    
    ###FIG 1
    #if index >= 8: #8,9,10,11: the 4 from Er 60% 
    major_ticks = [25,50,75,100]
    #else:
        #major_ticks = [5,15,25]
    plot_expt_by_expt_behaviour(titulo + r', all pixels', red1_dset, Time_bin, nominal_time_on,fastfactor,'r',major_ticks,dark_dset=None,plot_dark=False,unit='kHz') #pass titulo as well
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
#    se1_dset_reg, red1_dset_reg, red1_dset_reg_all = reg_time_resolved_images_to_se(se1_dset, red1_dset)
#    se1_dset_reg = np.array(se1_dset_reg, dtype=np.float16)
#    red1_dset_reg = np.array(red1_dset_reg, dtype=np.float32)
#    red1_dset_reg_all = np.array(red1_dset_reg_all, dtype=np.float32)
    
    #second step of registerist CL using bright CL: does NOT cause errors!!!
    #right now, remaining totalpoints - cut_at_beginning time resolved points, over all experiments
    #25 dark (in reality, dark until 28)/ 50 bright / 125 transient
    #cut arrays are 3 / 50 / 125
#    center_cl_index = 27 # (50 + 3)/2 # this index is going to be used as reference
#    end_left_index = 0#not used for the time being
#    end_right_index = 0#not used for the time being
#    new = reg_images_middle_cl(red1_dset_reg,center_cl_index,0,0)
#    red1_dset_reg = np.array(new, dtype=np.float32)
    
    #!!!!!END OF REAL Registration - UNCOMMENT!   
    #print(np.sum(np.isnan(red1_dset_reg)))
    #print(np.sum(np.isinf(red1_dset_reg)))
    #klklklk
    
#    ###MOCK REGISTRATION, QUICK FOR HACK ####################################!!!!!!!!!!!!!!!!!!!!
    se1_dset_reg = np.average(se1_dset,axis=0)
    red1_dset_reg = np.average(red1_dset, axis=0)
    red1_dset_reg_all = np.array(red1_dset)
    se1_dset_reg = np.array(se1_dset_reg, dtype=np.float16)
    red1_dset_reg = np.array(red1_dset_reg, dtype=np.float32)
    red1_dset_reg_all = np.array(red1_dset_reg_all, dtype=np.float32)
    #second step
    center_cl_index = 3 # # this index is going to be used as reference, should be no 82 in original vector
#    end_left_index = 0#not used for the time being
#    end_right_index = 0#not used for the time being
#    new = reg_images_middle_cl(red1_dset_reg,center_cl_index,0,0)
#    red1_dset_reg = np.array(new, dtype=np.float32)
    
    ## end of mock registration
    
    del se1_dset, red1_dset
    file1.close()
    
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
    red1_dset_cut_all = red1_dset_reg_all
    #red_dset_cut[:] = np.nan
    #red_dset_cut[mask] = red_dset_reg[mask]
    se1_dset_cut = se1_dset_reg #np.empty([blue_dset_reg.shape[0],blue_dset_reg.shape[1]])
    #se_dset_cut[:] = np.nan
    #se_dset_cut[mask] = se_dset_reg[mask]
    
    se1_dset_cut = np.array(se1_dset_cut, dtype=np.float16)
    red1_dset_cut = np.array(red1_dset_cut, dtype=np.float32)
    red1_dset_cut_all = np.array(red1_dset_cut_all, dtype=np.float32)
    
    del red1_dset_reg, red1_dset_reg_all, se1_dset_reg
    gc.collect()
    
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
        gmmse_red1_darkse_dset, gmmse_red1_brightse_dset, gmmse_se1_dark_dset, gmmse_se1_bright_dset, darkse_pct, brightse_pct = gmmone_tr_in_masked_channel(se1_dset_cut, red1_dset_cut)     
        #gmmse_red1_darkse_dset, gmmse_red1_brightse_dset, gmmse_se1_dark_dset, gmmse_se1_bright_dset, darkse_pct, brightse_pct = thr_otsu_tr_in_masked_channel(se1_dset_cut, red1_dset_cut)     

                
        
        #do_analysis(red1_dset_cut, se1_dset_cut, gmmse_red1_dark_dset, gmmse_red1_bright_dset, gmmse_se1_dark_dset, gmmse_se1_bright_dset, 'CL', 'SE','GMM SE', 'SE dark spots', 'SE bright spots',Pixel_size)
        #only for plots of intensity
        del gmmse_se1_dark_dset, gmmse_se1_bright_dset    
        
        gmmse_red1_darkse_dset = np.array(gmmse_red1_darkse_dset, dtype=np.float32)
        gmmse_red1_brightse_dset = np.array(gmmse_red1_brightse_dset, dtype=np.float32)
        
        gc.collect()
        gmmse_red1_darkse_dset_for_4D, gmmse_red1_brightse_dset_for_4D, blah, blup, darkse_pct2, brightse_pct2 =  gmmone_tr_in_masked_channel(se1_dset_cut, red1_dset_cut_all, imagemasked_is_4D=True) 
        #gmmse_red1_darkse_dset_for_4D, gmmse_red1_brightse_dset_for_4D, blah, blup, darkse_pct2, brightse_pct2 =  thr_otsu_tr_in_masked_channel(se1_dset_cut, red1_dset_cut_all, imagemasked_is_4D=True) 
        
        del blah, blup, darkse_pct2, brightse_pct2 #delete all SE masks 
        gc.collect()
        gmmse_red1_darkse_dset_for_4D = np.array(gmmse_red1_darkse_dset_for_4D, dtype=np.float32)
        gmmse_red1_brightse_dset_for_4D = np.array(gmmse_red1_brightse_dset_for_4D, dtype=np.float32)
        
    else:
        print('NOT doing gmm se') #assume all is bright in CL
        gmmse_red1_darkse_dset = red1_dset_cut
        gmmse_red1_brightse_dset = 0*red1_dset_cut #or could give 0 vector
        darkse_pct = 1.0
        brightse_pct = 0.0
        gmmse_red1_darkse_dset_for_4D = np.array(red1_dset_cut_all, dtype=np.float32)
        gmmse_red1_brightse_dset_for_4D = np.array(red1_dset_cut_all, dtype=np.float32)
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
    init_plot_no = center_cl_index #around 27 or 28
    plot_video_reg(titulo, red1_dset_cut,gmmse_red1_darkse_dset, gmmse_red1_brightse_dset, red1_dset_cut_all, se1_dset_cut,  gmmse_red1_brightse_dset_for_4D, gmmse_red1_darkse_dset_for_4D, Time_bin,fastfactor,nominal_time_on,Pixel_size[index],darkse_pct, brightse_pct,name_str[index] ,init_plot_no,major_ticks)
    del se1_dset_cut
    gc.collect()
    ###END FIG2
    
    ###FIG3
    #fig_no = '-3plots'
    #plot_expt_by_expt_behaviour(titulo + ', signal pixels', gmmse_red1_darkse_dset_for_4D/darkse_pct, Time_bin, nominal_time_on,fastfactor,'y',major_ticks,dark_dset=gmmse_red1_brightse_dset_for_4D/brightse_pct, plot_dark=True) #pass titulo as well
    
    #two lines were uncommented; now trying to delete later
    #del gmmse_red1_brightse_dset_for_4D
    #gc.collect()
    plot_expt_by_expt_behaviour(titulo + ', signal pixels', gmmse_red1_darkse_dset_for_4D/darkse_pct, Time_bin, nominal_time_on,fastfactor,'y',major_ticks,dark_dset=None, plot_dark=False,unit='kHz') #pass titulo as well
    #del gmmse_red1_brightse_dset_for_4D
    
    #multipage('ZZZ' + name_str[index] + fig_no + '.pdf',dpi=80)
    ###END FIG3
    
    #del gmmse_red1_darkse_dset_for_4D
    gc.collect()
    
    ###FIG4
    fig_no = '-3plots'
    start_of_transient = 82- cut_at_beginning + 1 #time-bin 75 + 300ns/40ns = 75 + ~8 = 83
    last_pt_offset = -10 #sometimes use -1, last point, but sometimes this gives 0. -10 seems to work
    if index == 7000:  ### 7 was a problem for red, added a thousand
        print('core only, sample B')
        init_guess = [np.average(gmmse_red1_darkse_dset[start_of_transient,:,:])/darkse_pct, 1.0, np.average(gmmse_red1_darkse_dset[last_pt_offset,:,:])/darkse_pct, np.average(gmmse_red1_darkse_dset[start_of_transient,:,:])/darkse_pct , 0.1] #e init was 0.5, d was zero before I made == a
    elif index == 10: #### 10 (oleci acid) was a single not double, added a thousand
        print("oleic")
        init_guess = [np.average(gmmse_red1_darkse_dset[start_of_transient,:,:])/darkse_pct,0.05, np.average(gmmse_red1_darkse_dset[last_pt_offset,:,:])/darkse_pct, np.average(gmmse_red1_darkse_dset[start_of_transient,:,:])/darkse_pct, 0.25] #e init was 0.5
    else:
        init_guess = [np.average(gmmse_red1_darkse_dset[start_of_transient,:,:])/darkse_pct, 0.5, np.average(gmmse_red1_darkse_dset[last_pt_offset,:,:])/darkse_pct, np.average(gmmse_red1_darkse_dset[start_of_transient+50,:,:])/darkse_pct, 0.05] #e init was 0.5
    #if do_gmmse_dset is False:
     #  brightse_pct = 0.01 #just so that I don't have a division by 0 in the function argument below!!!!
    b,e,be,ee = calcdecay(gmmse_red1_darkse_dset[start_of_transient::,:,:]/darkse_pct, time_detail= Time_bin*1e-9*fastfactor,titulo='Cathodoluminescence rate decay, bi-exponential fit, \n ' + titulo ,single=False,other_dset1=None ,other_dset2=None,init_guess=init_guess,unit='kHz')    
    #b,be = calcdecay(gmmse_red1_darkse_dset[start_of_transient::,:,:]/darkse_pct, time_detail= Time_bin*1e-9*fastfactor,titulo='Cathodoluminescence rate decay, single exponential fit, \n ' + titulo ,single=True,other_dset1=red1_dset_cut[start_of_transient::,:,:]/1.0 ,other_dset2=gmmse_red1_brightse_dset[start_of_transient::,:,:]/brightse_pct,init_guess=init_guess)
    
    tau_single[index] = b
    tau_single_error[index] = be
    tau_bi[index,:] = [b,e]
    tau_bi_error[index,:] = [be,ee]
    
    #for plots of dose, in the Er 60%:
#    if index >= 9: #8,9,10,11: the 4 from Er 60% 
#        mycode = 'ZZZZZZEr60' + str(index) + '= tempfile.NamedTemporaryFile(delete=False)'
#        exec(mycode)
#        np.savez('ZZZZZZEr60' + str(index), data = np.average(gmmse_red1_darkse_dset_for_4D/darkse_pct, axis=(1,2,3)))
        
    #for plots of tau as a function of number of experiments in Er 2%, sample A (index0)
    if index is 0:
        pass
        #calcdecay_series(gmmse_red1_darkse_dset_for_4D[:,start_of_transient::,:,:]/darkse_pct, time_detail= Time_bin*1e-9*fastfactor,titulo='Cathodoluminescence rate decay, bi-exponential fit, \n ' + titulo ,single=False,nominal_time_on=nominal_time_on,fastfactor=fastfactor,other_dset1=red1_dset_cut_all[:,start_of_transient::,:,:]/1.0 ,other_dset2=gmmse_red1_brightse_dset_for_4D[:,start_of_transient::,:,:]/brightse_pct,init_guess=init_guess)    
        #calcdecay_each(gmmse_red1_darkse_dset_for_4D[:,start_of_transient::,:,:]/darkse_pct, time_detail= Time_bin*1e-9*fastfactor,titulo='Cathodoluminescence rate decay, bi-exponential fit, \n ' + titulo ,single=False,nominal_time_on=nominal_time_on,fastfactor=fastfactor,other_dset1=red1_dset_cut_all[:,start_of_transient::,:,:]/1.0 ,other_dset2=gmmse_red1_brightse_dset_for_4D[:,start_of_transient::,:,:]/brightse_pct,init_guess=init_guess)    


    #del gmmse_red1_darkse_dset_for_4D, gmmse_red1_brightse_dset_for_4D #last one is trying to delete later than line372
    del gmmse_red1_brightse_dset_for_4D #last one is trying to delete later than line372
    gc.collect()
    
    multipage('ZZZ' + name_str[index] + fig_no + '.pdf',dpi=80)
    
    ###END FIG4
    
    #plt.show()
    plt.close('all')
    
#write a temporary file with all values
#outfile = tempfile.NamedTemporaryFile(delete=False)
#np.savez(outfile, tau_single=tau_single, tau_single_error=tau_single_error, tau_bi=tau_bi, tau_bi_error=tau_bi_error)

#mycode = 'ZZZred = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('ZZZred', datared = gmmse_red1_darkse_dset, datared4D = gmmse_red1_darkse_dset_for_4D)

#mycode = 'ZZZblue = tempfile.NamedTemporaryFile(delete=False)'
#exec(mycode)
#np.savez('ZZZblue', datablue = gmmse_red1_darkse_dset, datablue4D = gmmse_red1_darkse_dset_for_4D)
##########
    
red = np.load('ZZZred.npz') 
blue = np.load('ZZZblue.npz') 

unit = 'kHz'
plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')
plt.suptitle("Cathodoluminescence for near-impulsive excitation (120ns of exposure), \n " + titulo, fontsize=23)
ax1 = plt.subplot2grid((2,2), (0, 0), colspan=1)
#ax1.set_title(str(dset.shape[0]) + r" consecutive experiments (time follows rainbow: purple $\rightarrow$ red)",fontsize=fsizepl)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
color=iter(cm.rainbow(np.linspace(0,1,6)))
#lab =[r'Er 2$\%$ CS', 'Er 20$\%$ CS','Er 40$\%$ CS','Er 40$\%$ C','Er 80$\%$ CS','Oleic acid']
#new = np.zeros([6,121])
#for jj in range(5):
  #  new[jj,:] = 0.5*(data['dset_decay'][2*jj,:]+data['dset_decay'][2*jj+1,:])
#new[5,:] = data['dset_decay'][10,:]
#for k in range(6):
   # c=next(color)
    #if k in [0,9,19,29,39,49]:
plt.plot(np.arange(0,121)*Time_bin/1e3,np.average(red['datared'],axis=(1,2)),'r',lw=3) #in mus, in MHz
plt.plot(np.arange(0,121)*Time_bin/1e3,np.average(blue['datablue'],axis=(1,2)),'b',lw=3)
    #else:
        #plt.plot(np.arange(0,121)*Time_bin/1e3,data['dset_decay'][k,:],c=c, label='_nolegend_') #in mus, in MHz
#plt.plot(np.arange(0,dset.shape[1])*Time_bin/1e3,np.average(dset_avg,axis = 0),c=my_color_avg,label='average over ' + str(dset.shape[0]) + ' expts.',linewidth=lw) #in mus, in MHz
ax1.axvspan(0.04, 0.16, alpha=0.25, color='red')
plt.ylabel("Average luminescence \n of each time bin, per pixel (" + unit + ")",fontsize=20)
plt.xlabel("Behaviour of e-beam during each experiment: \n 0.12-ON + OFF ($\mu$s)",fontsize=20)
plt.legend() 
#major_ticks0 = [0.3, 2.3, 7]
major_ticks0 = [1,2]
ax1.set_xticks(major_ticks0) 
ax1.set_yticks([5,10]) 
plt.xlim([0,2])

#ax1 = plt.subplot2grid((2,2), (1, 0), colspan=1)
#major_ticks = [-2,2,20,36,44,80]
#ax1.set_xticks(major_ticks)
#ax1.set_xticklabels(['OA','2','20','40 CS','40 C','80'])
#plt.xlim([-6,84])
#x_vec = [2,2,20,20,36,36,44,44,80,80,-2]
##ax1.set_title(r"Average frame intensity",fontsize=fsizepl)
#ax1.spines['right'].set_visible(False)
#ax1.spines['top'].set_visible(False)
#ax1.xaxis.set_ticks_position('bottom')
#ax1.yaxis.set_ticks_position('left')
#plt.errorbar(x_vec,data['mean_brightness'],np.std(data['mean_brightness']),c='k',linestyle='None', marker='o',markersize=10)#@, label='_nolegend_',linewidth=lw) #in mus, in MHz
#ax1.set_yscale('log')
#
#plt.ylabel("Average luminescence \n for all experiments  (" + unit + ')',fontsize=20)
#plt.xlabel(r"Er content ($\%$)",fontsize=20)
#major_ticks = [25,50,75,nominal_time_on*dset.shape[0]*fastfactor]
#major_ticks = [2,4,6]
#ax1.set_xticks(major_ticks) 
#plt.xlim([0,11])

ax1 = plt.subplot2grid((2,2), (0, 1), colspan=1)
#ax1.set_title("For Er 40$\%$ CS",fontsize=20)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

#datared = np.load('ZZZZZZreddecayEr40CSA.npz') 
#datablue = np.load('ZZZZZZbluedecayEr40CSA.npz') 
#dataall = np.load('ZZZZZZALLdecayEr40CSA.npz') 

fastfactor = 1
last_pt_offset = -10 #sometimes use -1, last point, but sometimes this gives 0. -10 seems to work
middlept = 20
init_guess = [np.average(red['datared'][4,:,:]), 1.0, np.average(red['datared'][last_pt_offset,:,:]), np.average(red['datared'][middlept,:,:]), 0.1] #e init was 0.5
init_guess2 = [np.average(blue['datablue'][4,:,:]), 1.0, np.average(blue['datablue'][last_pt_offset,:,:]), np.average(blue['datablue'][middlept,:,:]),0.1] #e init was 0.5
b,e,be,ee = calcdecay_subplot2(red['datared'][4:,:,:], time_detail= Time_bin*1e-9*fastfactor,titulo='Cathodoluminescence rate decay, bi-exponential fit, \n ' + titulo ,single=False,other_dset2=blue['datablue'][4:,:,:] ,other_dset1=None,init_guess=init_guess,unit='kHz',init_guess2=init_guess2)    
plt.xlim([0,2])
major_ticks0 = [1,2]
plt.ylabel("Average luminescence \n of each time bin, per pixel (" + unit + ")",fontsize=20)

ax1.set_xticks(major_ticks0) 
plt.show() 

ax1 = plt.subplot2grid((2,2), (1, 1), colspan=1)
#ax1.set_title("For Er 40$\%$ CS",fontsize=fsizepl)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')

fsizetit = 22 #18
fsizepl = 20 #16
sizex = 10 #8
sizey = 10 #6
dpi_no = 80
lw = 2

plt.plot(np.arange(1,51)*nominal_time_on*fastfactor,np.average(blue['datablue4D'],axis=(1,2,3)),c='b', label='From blue photons ($<$ 409nm)',linestyle='None', marker='o',markersize=4) #in mus, in MHz
plt.plot(np.arange(1,51)*nominal_time_on*fastfactor,np.average(red['datared4D'],axis=(1,2,3)),c='r', label='From red photons ($>$ 409nm)',linestyle='None', marker='o',markersize=4) #in mus, in MHz
plt.ylabel("Average luminescence \n for each experiment, per pixel  (kHz)",fontsize=fsizepl)
plt.xlabel(r"Cumulative e-beam exposure time per pixel (nominal, $\mu$s)",fontsize=fsizepl)
#major_ticks = [25,50,75,nominal_time_on*dset.shape[0]*fastfactor]
major_ticks = [2,4,6]
ax1.set_xticks(major_ticks) 
#plt.legend()
plt.xlim([nominal_time_on,nominal_time_on*50*fastfactor])

    
multipage('FigForPosterFonts.pdf',dpi=80)  
 