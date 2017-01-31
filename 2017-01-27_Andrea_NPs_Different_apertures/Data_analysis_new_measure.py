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


initbin = (150+50+3)-1

dset= 1

###############################################################################
###############################################################################
###############################################################################

if dset == 1:
    nametr = ['2017-01-27-1549_ImageSequence__250.000kX_10.000kV_10mu_10',
          '2017-01-27-1516_ImageSequence__50.415kX_10.000kV_20mu_9',
          '2017-01-27-1435_ImageSequence__250.000kX_10.000kV_30mu_8',
          '2017-01-27-1624_ImageSequence__250.000kX_10.000kV_60mu_11',
          '2017-01-27-1658_ImageSequence__17.458kX_10.000kV_30mu_12']

    #let = ['ap10', 'ap20', 'ap30', 'ap60', 'ap120']    
    #Varying_variable = [28, 180, 379, 1800, 5700] 
    
    #Removing 20mum aperture because weird
    let = ['ap10','ap30', 'ap60', 'ap120']    
    Varying_variable = [28, 379, 1800, 5700] 
    
    
    Label_varying_variable = 'Current (pA) [data taken LOW to HIGH]' 
    
    listofindex = [0,1,2,3] #,4]
    
    loadprefix = ''
    
###############################################################################
###############################################################################
###############################################################################

if dset == 1:
    pref0 = 'Different_apertures_'
    
Time_bin = 1000.0 #in ns

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
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


red_vec = np.empty([len(Varying_variable),1498])
blue_vec = np.empty([len(Varying_variable),1498])

def plot_things(prefix, my_color):

    for index in listofindex: 
        
        print(index)
        
        print('bef loading')
        redd = np.load(loadprefix + let[index] + prefix + '1D.npz',mmap_mode='r') 
        red = redd['data'][initbin:]
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
        
        print(index)
        print( aux_vec[index,:].shape)
        print(len(red)) #SHOULD BE APPROX 1500
        for jj in np.arange(0,len(red)):
            aux_vec[index,jj] = np.sum(red[0:jj])/np.sum(red[jj:])
            aux_vec_cumu[index,jj]= np.sum(red[0:jj])
            
        ax1.semilogy( aux_vec[index,:], np.arange(0,len(red)),lw=index+1,color=my_color)
        ax1.set_xlabel('ratio (sum 0-tau)/(sum tau-1500)')
        ax1.set_ylabel('decay length tau ($\mu$s)')
        ax1.set_ylim(ymax=1500)
        ax1.set_xlim([0,10])
        ax1.set_title('Lw $\propto$' +Label_varying_variable)
        
        
               
        
        
        
        ax2.plot(np.arange(0,len(red)), aux_vec_cumu[index,:],lw=index+1,color=my_color )

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
        
        
        
        ax2.set_xlabel('tau')
        ax2.set_ylabel('cumu counts')
        ax2.set_xlim(xmax=1500)
        #plt.xlim([0,20])
        ax2.set_title('Lw $\propto$' +Label_varying_variable)
        
        if index == len(listofindex)-1:
            ax3.plot(Varying_variable, median_tau_red,label='median',color=my_color,ls='--')
            ax3.plot(Varying_variable, one_over_e_tau_red,label='1/e',color=my_color,ls='dotted')
            ax3.set_xlabel(Label_varying_variable)
            ax3.set_ylabel('tau ($\mu$s)')
            ax3.legend(loc='best')
            
            ax4.plot(Varying_variable, mean_int_median_before/mean_int_median_after,label='bef/aft median', color=my_color, ls='-')
            ax5.plot(Varying_variable, (mean_int_median_before-mean_int_median_after)/(mean_int_median_before+mean_int_median_after),label='bef/aft median', color=my_color, ls='--')
            ax4.plot(Varying_variable, mean_int_one_over_e_before/mean_int_one_over_e_after,label='bef/aft 1/e', color=my_color, ls='dotted')
            ax5.plot(Varying_variable, (mean_int_one_over_e_before-mean_int_one_over_e_after)/(mean_int_one_over_e_before+mean_int_one_over_e_after),label='bef/aft 1/e', color=my_color, ls='-.')
            ax4.set_xlabel(Label_varying_variable)
            ax4.set_ylabel('intensity ratio')
            ax4.legend(loc='best')
            ax5.set_xlabel(Label_varying_variable)
            ax5.set_ylabel('intensity visibility')
            ax5.legend(loc='best')

            return aux_vec, aux_vec_cumu, median_tau_red, one_over_e_tau_red,  mean_int_median_before, mean_int_median_after,  mean_int_one_over_e_before, mean_int_one_over_e_after

aux_vec_red, aux_vec_cumu_red, median_tau_red, one_over_e_tau_red,  mean_int_median_before_red, mean_int_median_after_red,  mean_int_one_over_e_before_red, mean_int_one_over_e_after_red = plot_things('RED', 'r')
aux_vec_blue, aux_vec_cumu_blue, median_tau_blue, one_over_e_tau_blue,  mean_int_median_before_blue, mean_int_median_after_blue,  mean_int_one_over_e_before_blue, mean_int_one_over_e_after_blue = plot_things('BLUE', 'g')


# saving results to files
import pickle
save_data = {}
save_data['aux_vec_red'] = aux_vec_red
save_data['aux_vec_blue'] = aux_vec_blue
save_data['x_values'] = Varying_variable
save_data['red1D'] = red_vec
save_data['blue1D'] = blue_vec
pickle.dump(save_data, open("../d0.p", "wb"))






z_min = 0
z_max = 10
#    
#fig777, ax777 = plt.subplots()    
#ax777.pcolor(Varying_variable, np.arange(0,aux_vec_red.shape[1]), aux_vec_red.transpose(), vmin=z_min, vmax=z_max)
#
#fig778, ax778 = plt.subplots()    
#ax778.pcolor(Varying_variable, np.arange(0,aux_vec_red.shape[1]), aux_vec_blue.transpose(), vmin=z_min, vmax=z_max)
#    


#Relationships between taus
ax30.plot(Varying_variable, median_tau_red/median_tau_blue,label='median ratio',color='k',ls='-')
ax30.plot(Varying_variable, (median_tau_red-median_tau_blue)/(median_tau_red+median_tau_blue),label='median visib',color='k',ls='--')
ax30.plot(Varying_variable, one_over_e_tau_red/one_over_e_tau_blue,label='1/e ratio',color='k',ls='dotted')
ax30.plot(Varying_variable, (one_over_e_tau_red-one_over_e_tau_blue)/(one_over_e_tau_red+one_over_e_tau_blue),label='1/e visib',color='k',ls='-.')
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
    
    ax20.plot(np.arange(0,aux_vec_red.shape[1]), aux_vec_cumu_red[index,:]-aux_vec_cumu_blue[index,:],lw=index+1,color='k',ls='-') #label='minus')
    ax200.plot(np.arange(0,aux_vec_red.shape[1]), aux_vec_cumu_red[index,:]/aux_vec_cumu_blue[index,:],lw=index+1,color='k',ls='-') #,label='ratio')
    ax2000.plot(np.arange(0,aux_vec_red.shape[1]), (aux_vec_cumu_red[index,:]-aux_vec_cumu_blue[index,:])/(aux_vec_cumu_red[index,:]+aux_vec_cumu_blue[index,:]),lw=index+1,color='k',ls='-')#,label='visi')
    ax20.set_xlabel('tau')
    ax20.set_ylabel('cumu counts MINUS')
    ax20.set_xlim(xmax=1500)
    ax20.legend(loc='best')
    ax20.set_title('Lw $\propto$' +Label_varying_variable)
    ax200.set_xlabel('tau')
    ax200.set_ylabel('cumu counts RATIO')
    ax200.set_xlim(xmax=1500)
    ax200.legend(loc='best')
    ax200.set_title('Lw $\propto$' +Label_varying_variable)
    ax2000.set_xlabel('tau')
    ax2000.set_ylabel('cumu counts VISIB')
    ax2000.set_xlim(xmax=1500)
    ax2000.legend(loc='best')
    ax2000.set_title('Lw $\propto$' +Label_varying_variable)
    
ax40.plot(Varying_variable, (mean_int_median_before_red/mean_int_median_after_red)/(mean_int_median_before_blue/mean_int_median_after_blue),label='bef/aft median ratio red/green', color='k', ls='dotted')
ax40.plot(Varying_variable, ((mean_int_median_before_red-mean_int_median_after_red)/(mean_int_median_before_red+mean_int_median_after_red))/((mean_int_median_before_blue-mean_int_median_after_blue)/(mean_int_median_before_blue+mean_int_median_after_blue)),label='vis bef/aft median ratio red/green', color='k', ls='--')
ax40.plot(Varying_variable, mean_int_median_before_red/mean_int_median_before_blue, label='bef median ratio red/green', color='k', ls='-')
ax40.plot(Varying_variable, mean_int_median_after_red/mean_int_median_after_blue,label='aft median ratio red/green', color='k', ls='-.')
ax40.set_xlabel(Label_varying_variable)
ax40.set_ylabel('intensity ratio')
ax40.legend(loc='best')
ax40.set_xlabel(Label_varying_variable)
ax40.set_ylabel('intensity visibility')
ax40.legend(loc='best')

ax400.plot(Varying_variable, (mean_int_one_over_e_before_red/mean_int_one_over_e_after_red)/(mean_int_one_over_e_before_blue/mean_int_one_over_e_after_blue),label='bef/aft one_over_e ratio red/green', color='k', ls='dotted')
ax400.plot(Varying_variable, ((mean_int_one_over_e_before_red-mean_int_one_over_e_after_red)/(mean_int_one_over_e_before_red+mean_int_one_over_e_after_red))/((mean_int_one_over_e_before_blue-mean_int_one_over_e_after_blue)/(mean_int_one_over_e_before_blue+mean_int_one_over_e_after_blue)),label='vis bef/aft one_over_e ratio red/green', color='k', ls='--')
ax400.plot(Varying_variable, mean_int_one_over_e_before_red/mean_int_one_over_e_before_blue, label='bef one_over_e ratio red/green', color='k', ls='-')
ax400.plot(Varying_variable, mean_int_one_over_e_after_red/mean_int_one_over_e_after_blue,label='aft one_over_e ratio red/green',color='k', ls='-.')
ax400.set_xlabel(Label_varying_variable)
ax400.set_ylabel('intensity ratio')
ax400.legend(loc='best')
ax400.set_xlabel(Label_varying_variable)
ax400.set_ylabel('intensity visibility')
ax400.legend(loc='best')
   
   
multipage_longer('Summary.pdf',dpi=80)

plt.show()
klklklk
    

    