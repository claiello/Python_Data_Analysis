import os
import sys
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../Python modules/") # necessary for the tex fonts
import scipy as sp
import scipy.misc
import matplotlib
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.misc
import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile

import skimage
from skimage import exposure
from my_fits import *

#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################

#backgdinit = 50
#initbin = (150+50+3)-1

val_norm = 20.0

###############################################################################
###############################################################################
###############################################################################
#if dset == 1:
#    nametr = ['2017-01-05-1452_ImageSequence__250.000kX_10.000kV_30mu_10',
#          '2017-01-05-1557_ImageSequence__250.000kX_10.000kV_30mu_15',
#          '2017-01-05-1634_ImageSequence__250.000kX_10.000kV_30mu_20',
#          '2017-01-05-1709_ImageSequence__250.000kX_10.000kV_30mu_23',
#          '2017-01-05-1745_ImageSequence__250.000kX_10.000kV_30mu_26',
#          '2017-01-05-1831_ImageSequence__250.000kX_10.000kV_30mu_30',
#          '2017-01-05-1906_ImageSequence__250.000kX_10.000kV_30mu_32']
#
#    let = ['V0','V0pt25' ,'V0pt5b','V0pt75'] 
#    Varying_variable = [27.8       ,  30.5       , 49.22101969,  74.68039129 ]
#    Label_varying_variable = 'Temperature (C)' 
#    listofindex = [0,1,2,3]#[1,2,4,5]
#    loadprefix = '../2017-01-05_Andrea_small_new_sample_5DiffTemps/'
#    my_marker = 'o'
#
#
#if dset == 2: #largest area
#    nametr = ['2016-12-19-1924_ImageSequence__100.000kX_10.000kV_30mu_4',
#          '2016-12-19-1950_ImageSequence__100.000kX_10.000kV_30mu_5',
#          '2016-12-19-2130_ImageSequence__100.000kX_10.000kV_30mu_8',
#          '2016-12-19-2015_ImageSequence__100.000kX_10.000kV_30mu_6',
#          '2016-12-19-2056_ImageSequence__27.836kX_10.000kV_30mu_7']
#          
#    let = ['V0pt25' ,'V0pt5', 'V0', 'V0pt75','V1']
#    Varying_variable = [ 29.89      ,  42.9261667 ,  49.20303894,  57.81008605,  83.7163313]
#    Label_varying_variable = 'Temperature (C)' 
#    listofindex = [0,1,2,3,4]
#    loadprefix = '../2016-12-19_Andrea_BigNPs_5DiffTemps/fixCprevtaudouble'
#    my_marker = '^'
#
#
#if dset == 3: #medium area
#
#    nametr = ['2017-01-13-1730_ImageSequence__150.000kX_10.000kV_30mu_15',
#          '2017-01-13-1810_ImageSequence__150.000kX_10.000kV_30mu_17',
#          '2017-01-13-1935_ImageSequence__150.000kX_10.000kV_30mu_3',
#          '2017-01-13-2011_ImageSequence__150.000kX_10.000kV_30mu_4',
#          '2017-01-13-2050_ImageSequence__150.000kX_10.000kV_30mu_7',
#          '2017-01-13-2132_ImageSequence__150.000kX_10.000kV_30mu_8',
#          '2017-01-13-2213_ImageSequence__150.000kX_10.000kV_30mu_9',
#          '2017-01-13-2251_ImageSequence__150.000kX_10.000kV_30mu_10']
#    
#    let = ['N60pt6', 'N48pt5', 'N39pt8', 'N31pt2']
#    Varying_variable = [60.6, 48.5, 39.8, 31.2]
#    Label_varying_variable = 'Temperature (C)' 
#    listofindex = [0,1,2,3] #corr_index = [4,5,6,7] #correction of index
#    loadprefix = '../2017-01-13_Andrea_NPs_CoolingDown_Controllably/fixCsametaudouble'
#    my_marker = 's'
        
###############################################################################
###############################################################################
###############################################################################
    
Time_bin = 1000.0 #in ns

fig1, ax1 = plt.subplots()
fig11, ax11 = plt.subplots()

fig2, ax2 = plt.subplots()
fig22, ax22 = plt.subplots()

fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()

#fig10, ax10 = plt.subplots()
fig20, ax20 = plt.subplots()
fig200, ax200 = plt.subplots()
fig2000, ax2000 = plt.subplots()
fig30, ax30 = plt.subplots()

fig40, ax40 = plt.subplots()
fig400, ax400 = plt.subplots()


red_vec = np.empty([5,1498])
blue_vec = np.empty([5,1498])

def plot_things(prefix, my_color, my_marker, axe1, axe2):

    for index in listofindex: 
        
        print(index)
        
        print('bef loading')
        redd = np.load(loadprefix + let[index] + prefix + '1D.npz',mmap_mode='r') 
        #1d DATA IS ALREADY ONLY DECAY!!!!!!!!!
        red = redd['data'] #[initbin:] TOOK OUT THE INITBIN BC 1D DATA IS ALREADY ONLY THE DECAY
        if my_color =='r':
            print('red')
            red_vec[index,:] = red
        else:
            print('blue')
            blue_vec[index,:] = red
        
        if index == 0:
            median_tau_red = np.empty(len(listofindex))
            one_over_e_tau_red = np.empty(len(listofindex))
            mean_int_median_before = np.empty(len(listofindex))
            mean_int_median_after = np.empty(len(listofindex))
            mean_int_one_over_e_before = np.empty(len(listofindex))
            mean_int_one_over_e_after = np.empty(len(listofindex))
            aux_vec = np.empty([len(listofindex),len(red)])
            aux_vec_cumu = np.empty([len(listofindex),len(red)])
        
        #median tau
        comparison = 10000000
        indexrecorded = 10000000
        for jj in np.arange(0,len(red)):
            first_part = np.sum(red[0:jj])
            second_part = np.sum(red[jj:])
            difference = np.abs(first_part - second_part) 
            if difference < comparison:
                comparison = difference
                indexrecorded = jj
        median_tau_red[index] = indexrecorded
        mean_int_median_before[index] = np.average(red[0:indexrecorded]) 
        mean_int_median_after[index] = np.average(red[indexrecorded:]) 
      
        #1/e tau
        comparisone = 10000000
        indexrecordede = 10000000
        for jj in np.arange(0,len(red)):
            first_part = red[0]/np.exp(1)
            second_part = red[jj]
            difference = np.abs(first_part - second_part) 
            if difference < comparisone:
                comparisone = difference
                indexrecordede = jj
        one_over_e_tau_red[index] = indexrecordede
        mean_int_one_over_e_before[index] = np.average(red[0:indexrecordede]) 
        mean_int_one_over_e_after[index] = np.average(red[indexrecordede:]) 
        
        
        for jj in np.arange(0,len(red)):
            aux_vec[index,jj] = np.sum(red[0:jj])/np.sum(red[jj:])
            aux_vec_cumu[index,jj]= np.sum(red[0:jj])
            
        print(index)
        print( aux_vec[index,:].shape)
        print(len(red)) #SHOULD BE APPROX 1400
        axe1.semilogy( aux_vec[index,:], np.arange(0,len(red)),lw=Varying_variable[index]/val_norm,color=my_color)
        axe1.set_xlabel('ratio (sum 0-tau)/(sum tau-1500)')
        axe1.set_ylabel('decay length tau ($\mu$s)')
        axe1.set_ylim(ymax=1500)
        #axe1.set_xlim([0,10])
        axe1.set_title('Lw $\propto$' +Label_varying_variable)
        
        axe2.plot(np.arange(0,len(red)), aux_vec_cumu[index,:],lw=Varying_variable[index]/val_norm,color=my_color)
        
         # new stuff
        
        print(np.sum(np.isnan(aux_vec_cumu[index,:])))
        tau, A, tau2, A2, c, reso = poly_fit(np.arange(0,len(red)), aux_vec_cumu[index,:])
        print_result(reso)
        
        x = np.arange(0,len(red))
        #y = c*x + tau*A*(1- np.exp(-tx/tau)) + tau2*A2*(1- np.exp(-x/tau2))
        y = func_poly(x, reso.params)
        
        ax2.plot(x, y, lw=len(listofindex)-index,color='k')       
        
        
        plt.figure(100)
        
        y = (aux_vec_cumu[index,:] - c*x)
        y = y/np.mean(y[-100:])
        
        plt.plot(x, y,lw=len(listofindex)-index,color=my_color)        
        plt.figure(2)        
        
        axe2.set_xlabel('tau')
        axe2.set_ylabel('cumu counts')
        axe2.set_xlim(xmax=1500)
        #plt.xlim([0,20])
        ax2.set_title('Lw $\propto$' +Label_varying_variable)
        
        if index == listofindex[-1]:
            print('last one')
            ax3.plot(Varying_variable, median_tau_red,label='median',color=my_color,ls='--',marker=my_marker)
            ax3.plot(Varying_variable, one_over_e_tau_red,label='1/e',color=my_color,ls='dotted',marker=my_marker)
            ax3.set_xlabel(Label_varying_variable)
            ax3.set_ylabel('tau ($\mu$s)')
            ax3.legend(loc='best')
            
            ax4.plot(Varying_variable, mean_int_median_before/mean_int_median_after,label='bef/aft median', color=my_color, ls='-',marker=my_marker)
            ax5.plot(Varying_variable, (mean_int_median_before-mean_int_median_after)/(mean_int_median_before+mean_int_median_after),label='bef/aft median', color=my_color, ls='--',marker=my_marker)
            ax4.plot(Varying_variable, mean_int_one_over_e_before/mean_int_one_over_e_after,label='bef/aft 1/e', color=my_color, ls='dotted',marker=my_marker)
            ax5.plot(Varying_variable, (mean_int_one_over_e_before-mean_int_one_over_e_after)/(mean_int_one_over_e_before+mean_int_one_over_e_after),label='bef/aft 1/e', color=my_color, ls='-.',marker=my_marker)
            ax4.set_xlabel(Label_varying_variable)
            ax4.set_ylabel('intensity ratio')
            ax4.legend(loc='best')
            ax5.set_xlabel(Label_varying_variable)
            ax5.set_ylabel('visibility of intensity')
            ax5.legend(loc='best')

            return aux_vec, aux_vec_cumu, median_tau_red, one_over_e_tau_red,  mean_int_median_before, mean_int_median_after,  mean_int_one_over_e_before, mean_int_one_over_e_after
            
          
for loopindex in [1]: #,1,2]: #,2]: #np.arange(0,3):

    if loopindex == 0:   #Small area
    ######## check how long this is, should be ~1400 after cutting
         nametr = ['2017-01-05-1452_ImageSequence__250.000kX_10.000kV_30mu_10',
          '2017-01-05-1557_ImageSequence__250.000kX_10.000kV_30mu_15',
          '2017-01-05-1634_ImageSequence__250.000kX_10.000kV_30mu_20',
          '2017-01-05-1709_ImageSequence__250.000kX_10.000kV_30mu_23',
          '2017-01-05-1745_ImageSequence__250.000kX_10.000kV_30mu_26',
          '2017-01-05-1831_ImageSequence__250.000kX_10.000kV_30mu_30',
          '2017-01-05-1906_ImageSequence__250.000kX_10.000kV_30mu_32']

         let = ['V0','V0pt25' ,'V0pt5b','V0pt75'] 
         Varying_variable = [27.8       ,  30.5       , 49.22101969,  74.68039129 ]
         Label_varying_variable = 'Temperature (C)' 
         listofindex = [0,1,2,3]#[1,2,4,5]
         loadprefix = '../2017-01-05_Andrea_small_new_sample_5DiffTemps/fixCprevtaudouble'
         my_marker = 'o'


    if loopindex == 1: #Large area
     ######## check how long this is, should be ~1400 after cutting
        nametr = ['2016-12-19-1924_ImageSequence__100.000kX_10.000kV_30mu_4',
          '2016-12-19-1950_ImageSequence__100.000kX_10.000kV_30mu_5',
          '2016-12-19-2130_ImageSequence__100.000kX_10.000kV_30mu_8',
          '2016-12-19-2015_ImageSequence__100.000kX_10.000kV_30mu_6',
          '2016-12-19-2056_ImageSequence__27.836kX_10.000kV_30mu_7']
          
        let = ['V0pt25' ,'V0pt5', 'V0', 'V0pt75','V1']
        Varying_variable = [ 29.89      ,  42.9261667 ,  49.20303894,  57.81008605,  83.7163313]
        Label_varying_variable = 'Temperature (C)' 
        listofindex = [0,1,2,3,4]
        loadprefix = '../2016-12-19_Andrea_BigNPs_5DiffTemps/fixCprevtaudouble'
        my_marker = '^'


    if loopindex == 2: #Medium area
     ######## check how long this is, should be ~1400 after cutting
        nametr = ['2017-01-13-1730_ImageSequence__150.000kX_10.000kV_30mu_15',
          '2017-01-13-1810_ImageSequence__150.000kX_10.000kV_30mu_17',
          '2017-01-13-1935_ImageSequence__150.000kX_10.000kV_30mu_3',
          '2017-01-13-2011_ImageSequence__150.000kX_10.000kV_30mu_4',
          '2017-01-13-2050_ImageSequence__150.000kX_10.000kV_30mu_7',
          '2017-01-13-2132_ImageSequence__150.000kX_10.000kV_30mu_8',
          '2017-01-13-2213_ImageSequence__150.000kX_10.000kV_30mu_9',
          '2017-01-13-2251_ImageSequence__150.000kX_10.000kV_30mu_10']
    
        let = ['N60pt6', 'N48pt5', 'N39pt8', 'N31pt2']
        Varying_variable = [60.6, 48.5, 39.8, 31.2]
        listofindex = [0,1,2,3] #corr_index = [4,5,6,7] #correction of index
        
        
#        let = ['N102pt5', 'N92pt9','N72pt8',  'N66pt4' ,'N60pt6', 'N48pt5', 'N39pt8', 'N31pt2']
#        Varying_variable = [102.5,92.9,72.8,66.4,60.6, 48.5, 39.8, 31.2] #Actual data taken seq is 102, 92,66,72,60...
#        listofindex = [0,1,2,3,4,5,6,7]
        
        
        Label_varying_variable = 'Temperature (C) [Taken HIGH to LOW]' 
       
        loadprefix = '../2017-01-13_Andrea_NPs_CoolingDown_Controllably/fixCprevtaudouble'
        my_marker = 's'


    aux_vec_red, aux_vec_cumu_red, median_tau_red, one_over_e_tau_red,  mean_int_median_before_red, mean_int_median_after_red,  mean_int_one_over_e_before_red, mean_int_one_over_e_after_red = plot_things('RED', 'r',my_marker, ax1, ax2)
    aux_vec_blue, aux_vec_cumu_blue, median_tau_blue, one_over_e_tau_blue,  mean_int_median_before_blue, mean_int_median_after_blue,  mean_int_one_over_e_before_blue, mean_int_one_over_e_after_blue = plot_things('BLUE', 'g',my_marker, ax11, ax22)
        
    
    # saving results to files
    import pickle
    save_data = {}
    save_data['aux_vec_red'] = aux_vec_red
    save_data['aux_vec_blue'] = aux_vec_blue
    save_data['x_values'] = Varying_variable
    save_data['red1D'] = red_vec
    save_data['blue1D'] = blue_vec
    pickle.dump(save_data, open("../d6.p", "wb"))
            
        
    #Relationships between taus
    ax30.plot(Varying_variable, median_tau_red/median_tau_blue,label='median ratio',color='k',ls='-',marker=my_marker)
    ax30.plot(Varying_variable, (median_tau_red-median_tau_blue)/(median_tau_red+median_tau_blue),label='median visib',color='k',ls='--',marker=my_marker)
    ax30.plot(Varying_variable, one_over_e_tau_red/one_over_e_tau_blue,label='1/e ratio',color='k',ls='dotted',marker=my_marker)
    ax30.plot(Varying_variable, (one_over_e_tau_red-one_over_e_tau_blue)/(one_over_e_tau_red+one_over_e_tau_blue),label='1/e visib',color='k',ls='-.',marker=my_marker)
    ax30.set_xlabel(Label_varying_variable)
    ax30.set_ylabel('intensity red/green')
    ax30.legend(loc='best')
    
    for index in np.arange(0,len(listofindex)):
    #    ax10.semilogy( aux_vec_red[index,:] - aux_vec_blue[index,:], np.arange(0,aux_vec_red.shape[1]),lw=len(listofindex)-index,color='k',label='minus',ls='-')
    #    ax10.semilogy( aux_vec_red[index,:]/aux_vec_blue[index,:], np.arange(0,aux_vec_red.shape[1]),lw=len(listofindex)-index,color='k',label='ratio',ls='dotted')
    #    ax10.semilogy( (aux_vec_red[index,:]-aux_vec_blue[index,:])/(aux_vec_red[index,:]+aux_vec_blue[index,:]), np.arange(0,aux_vec_red.shape[1]),lw=len(listofindex)-index,color='k',label='visib',ls='--')
    #    ax10.set_xlabel('functions')
    #    ax10.set_ylabel('decay length tau ($\mu$s)')
    #    ax10.set_ylim(ymax=1500)
    #    ax10.set_xlim([0,10])
    #    ax10.legend(loc='best')
    #    ax10.set_title('Lw $\propto$ pixel size; this plot not yet making sense')
        
        ax20.plot(np.arange(0,aux_vec_red.shape[1]), aux_vec_cumu_red[index,:]-aux_vec_cumu_blue[index,:],lw=Varying_variable[index]/val_norm,color='k',label='minus',ls='-')
        ax200.plot(np.arange(0,aux_vec_red.shape[1]), aux_vec_cumu_red[index,:]/aux_vec_cumu_blue[index,:],lw=Varying_variable[index]/val_norm,color='k',label='ratio',ls='-')
        ax2000.plot(np.arange(0,aux_vec_red.shape[1]), (aux_vec_cumu_red[index,:]-aux_vec_cumu_blue[index,:])/(aux_vec_cumu_red[index,:]+aux_vec_cumu_blue[index,:]),lw=Varying_variable[index]/val_norm,color='k',label='visi',ls='-')
        ax20.set_xlabel('tau')
        ax20.set_ylabel('cumu counts')
        ax20.set_xlim(xmax=1500)
        ax20.set_title('Red MINUS Green' + ' Lw $\propto$' +Label_varying_variable)
      
        ax200.set_xlabel('tau')
        ax200.set_ylabel('cumu counts')
        ax200.set_xlim(xmax=1500)
        ax200.set_title('Ratio Red/Green' + ' Lw $\propto$' +Label_varying_variable)
       
        ax2000.set_xlabel('tau')
        ax2000.set_ylabel('cumu counts')
        ax2000.set_xlim(xmax=1500)
        ax2000.set_title('Visibility Red/Green' + ' Lw $\propto$' +Label_varying_variable)
     
        
    ax40.plot(Varying_variable, (mean_int_median_before_red/mean_int_median_after_red)/(mean_int_median_before_blue/mean_int_median_after_blue),label='bef/aft median ratio red/green', color='k', ls='dotted', marker = my_marker)
    ax40.plot(Varying_variable, ((mean_int_median_before_red-mean_int_median_after_red)/(mean_int_median_before_red+mean_int_median_after_red))/((mean_int_median_before_blue-mean_int_median_after_blue)/(mean_int_median_before_blue+mean_int_median_after_blue)),label='vis bef/aft median ratio red/green', color='k', ls='--', marker = my_marker)
    ax40.plot(Varying_variable, mean_int_median_before_red/mean_int_median_before_blue, label='bef ratio red/green', color='k', ls='-', marker = my_marker)
    ax40.plot(Varying_variable, mean_int_median_after_red/mean_int_median_after_blue,label='aft median ratio red/green', color='k', ls='-.', marker = my_marker)
    ax40.set_xlabel(Label_varying_variable)
    ax40.set_ylabel('intensity ratio')
    ax40.legend(loc='best')
    ax40.set_xlabel(Label_varying_variable)
    ax40.set_ylabel('visibility of intensity')
    ax40.legend(loc='best')
    
    ax400.plot(Varying_variable, (mean_int_one_over_e_before_red/mean_int_one_over_e_after_red)/(mean_int_one_over_e_before_blue/mean_int_one_over_e_after_blue),label='bef/aft one_over_e ratio red/green', color='k', ls='dotted', marker = my_marker)
    ax400.plot(Varying_variable, ((mean_int_one_over_e_before_red-mean_int_one_over_e_after_red)/(mean_int_one_over_e_before_red+mean_int_one_over_e_after_red))/((mean_int_one_over_e_before_blue-mean_int_one_over_e_after_blue)/(mean_int_one_over_e_before_blue+mean_int_one_over_e_after_blue)),label='vis bef/aft one_over_e ratio red/green', color='k', ls='--', marker = my_marker)
    ax400.plot(Varying_variable, mean_int_one_over_e_before_red/mean_int_one_over_e_before_blue, label='bef ratio red/green', color='k', ls='-', marker = my_marker)
    ax400.plot(Varying_variable, mean_int_one_over_e_after_red/mean_int_one_over_e_after_blue,label='aft one_over_e ratio red/green',color='k', ls='-.', marker = my_marker)
    ax400.set_xlabel(Label_varying_variable)
    ax400.set_ylabel('intensity ratio')
    ax400.legend(loc='best')
    ax400.set_xlabel(Label_varying_variable)
    ax400.set_ylabel('visibility of intensity')
    ax400.legend(loc='best')
   
multipage_longer('Summary_new_measure.pdf',dpi=80)

plt.show()
klklklk
    
####### Needs to normalize by RT
    
###### Temperature-wise,I trust dset 2 the most
    