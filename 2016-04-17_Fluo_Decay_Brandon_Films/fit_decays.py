# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 11:17:15 2016

@author: clarice
"""


########################################################################
# Imports
########################################################################

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage as ndimage
from matplotlib_scalebar.scalebar import ScaleBar
import lmfit
from lmfit import minimize, Parameters, Parameter, report_fit

from load_data import load_data

from MakePdf import *





########################################################################
# fit functions
########################################################################

# define objective function: returns the array to be minimized
def fcn2min(params, x, data, single = False, return_plot = False, no_of_x_pts = 100):

    a = params['a'].value
    b = params['b'].value
    c = params['c'].value
    
    if single == False:        
        d = params['d'].value
        e = params['e'].value

    if return_plot == True:
        # changing x to give more values
        x = np.linspace(np.min(x), np.max(x), no_of_x_pts)
        
    if single:
        model = a*np.exp(-x/b) + c   
    else:        
        model = a*np.exp(-x/b) + c + d*np.exp(-x/e)
        #model = a*np.exp(-x/b) + c + (data[0]-a)*np.exp(-x/e)   
     
    if return_plot == False:
        return model - data
    else:
        return (x, model)
        
def fitexp(x, y, params = None, single = True, my_color = None, start_cut = None, end_cut = None):
    
    if start_cut is None:
        start_cut = 0
    if end_cut is None:
        end_cut = len(x)
        
    # create a set of Parameters
    if params is None:
        params = Parameters()
        params.add('a', value= 37.0, min=25.0,max=45.0)
        params.add('b', value= 12.0, min=4.0,max=15.0)
        params.add('c', value= y[-1], min=3.0,max=10.0) # offset
        if single == False:        
            params.add('d', value= 9.0, min=5.0,max=15.0)
            params.add('e', value= 218.0, min=50.0,max=230.0) #vary=False)
        
    result = minimize(fcn2min, params, args=(x[start_cut:end_cut], y[start_cut:end_cut], single))    
    (x_fit, y_fit) = fcn2min(result.params, x[start_cut:end_cut], y[start_cut:end_cut], single, return_plot = True, no_of_x_pts = 100)
    
    report_fit(result.params)
    if my_color is None:
        my_color = 'k'
    plt.plot(x_fit, y_fit, my_color)
    
    if single:
        return (result.params['a'].value,result.params['b'].value,result.params['c'].value,None,None)
    else:
        return (result.params['a'].value,result.params['b'].value,result.params['c'].value,result.params['d'].value,result.params['e'].value)



########################################################################
    
def plot_data(data, type, time_detail, params, opt):
        
    No_specimen = 3
    Pixel_size = 1.4886666e-08 # in meters
    
    No_of_time_points = data[type]['decay'][0]['no_of_time_points']
    
    fig2 = plt.figure(figsize=(8, 6), dpi=80)        
    
    for k in range(No_specimen):
        ax1 = plt.subplot2grid((3,3), (k, 0), colspan=1, rowspan=1)
        if k == 0:
            ax1.set_title('SE2')
        #plt.imshow(data['ce4000']['decay'][k]['image'], cmap = cm.Greys)
        plt.imshow(data['ce4000']['decay'][k]['image'])
                
        scalebar = ScaleBar(Pixel_size, frameon = True, box_alpha = 0.001, location = 'lower left') # 1 pixel = Pixel_size in meter

        plt.gca().add_artist(scalebar)
        plt.axis('off')
    
    
     #plot
    ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=3, rowspan=3)
    plt.hold(True)
    for kkk in range(No_specimen):
        
        #plt.plot(x_array[:-1]*time_detail,mean_grana_patch_blue[kkk,1:]/1000.0,'bo',markersize=(kkk+1),label='Decay region ' + str(kkk+1)) 
        x_array = np.array(range(No_of_time_points))
        x = x_array*time_detail
        y = data[type]['decay'][kkk]['blue_decay']/1000.0
        
        # fit data
        (a,b,c,d,e) = fitexp(x, y, params = params, single = opt['single_exp'], start_cut = opt['start_cut'])        
        
        plt.hold(True)
        
        if opt['single_exp']:            
            my_label = 'Decay region ' + str(kkk) + ', $ \\tau_1 $ = ' + str("{0:.2f}".format(b)) + '$\mu$s;'
        else:            
            my_label = 'Decay region ' + str(kkk) + ', $ \\tau_1 $ = ' + str("{0:.2f}".format(b)) + '$\mu$s; $ \\tau_2 $ = ' + str("{0:.2f}".format(e)) + '$\mu$s'
        plt.plot(x[opt['plot_start_cut']:], y[opt['plot_start_cut']:], 'o-', label = my_label) #'decay region = ' + str(kkk+1))                                    
    
    plt.xlabel('Time after blanking the eletron beam ($\mu$s)',  fontsize=14)
    plt.ylabel('Fluorescence rate (kcps)',  fontsize=14)
    plt.legend(loc='best')
    plt.title(opt['plot_title'])
    
    
    # plotting the average
    fig2 = plt.figure(figsize=(8, 6), dpi=80)

    y = data[type]['decay_blue_avg']/1000.0
    # fit data
    (a,b,c,d,e) = fitexp(x, y, params = params, single = opt['single_exp'], start_cut = opt['start_cut'])
    
    if opt['single_exp']:            
            my_label = 'Mean decay avg over 3 regions, $ \\tau_1 $ = ' + str("{0:.2f}".format(b)) + '$\mu$s;'
    else:            
            my_label = 'Mean decay avg over 3 regions, $ \\tau_1 $ = ' + str("{0:.2f}".format(b)) + '$\mu$s; $ \\tau_2 $ = ' + str("{0:.2f}".format(e)) + '$\mu$s'
            
    plt.plot(x[opt['plot_start_cut']:], y[opt['plot_start_cut']:], 'ro-', label = my_label)
    
    plt.xlabel('Time after blanking the eletron beam ($\mu$s)',  fontsize=14)
    plt.ylabel('Fluorescence rate (kcps)',  fontsize=14)
    plt.legend(loc='best')
    plt.title(opt['plot_title'])





########################################################################
# Main
########################################################################

data = load_data()



##########################################
# Ce 4000
##########################################

time_detail = 0.05 # in microsec; 1/expt clock rate

opt = {}
opt['start_cut'] = 6
opt['plot_start_cut'] = 6
opt['single_exp'] = True
opt['plot_title'] = 'Ce 4000'

# model = a*np.exp(-x/b) + c + d*np.exp(-x/e)
params = Parameters()
params.add('a', value= 2.0, min=0.0,max=500.0)
params.add('b', value= 10, min=0.01,max=15.0)
params.add('c', value= 0.1, min=0.0,max=10.0) # offset
#params.add('d', value= 0.01, min=0.0,max=0.01)
#params.add('e', value= 218.0, min=50.0,max=230.0) #vary=False)

plot_data(data, 'ce4000', time_detail, params, opt)

plt.ylim([0, 5])

##########################################
# Ce 3500
##########################################


time_detail = 0.050 # in microsec; 1/expt clock rate

opt = {}
opt['start_cut'] = 6
opt['plot_start_cut'] = 6
opt['single_exp'] = True
opt['plot_title'] = 'Ce 3500'

# model = a*np.exp(-x/b) + c + d*np.exp(-x/e)
params = Parameters()
params.add('a', value= 2.0, min=0.0,max=500.0)
params.add('b', value= 5.0, min=0.05,max=15.0)
params.add('c', value= 0.1, min=0.0,max=10.0) # offset
#params.add('d', value= 1.0, min=0.0,max=15.0)
#params.add('e', value= 218.0, min=50.0,max=230.0) #vary=False)

plot_data(data, 'ce3500', time_detail, params, opt)

plt.ylim([0, 4])

##########################################
# Eu 4000
##########################################

time_detail = 0.2 # in microsec; 1/expt clock rate

opt = {}
opt['start_cut'] = 3
opt['plot_start_cut'] = 3
opt['single_exp'] = False
opt['plot_title'] = 'Eu 4000'

# model = a*np.exp(-x/b) + c + d*np.exp(-x/e)
params = Parameters()
params.add('a', value= 30.0, min=25.0,max=45.0)
params.add('b', value= 3.0, min=0.1,max=15.0)
params.add('c', value= 50.0, min=3.0,max=100.0) # offset
params.add('d', value= 7.0, min=0.0,max=20.0)
params.add('e', value= 40.0, min=2.0,max=230.0) #vary=False)
        
plot_data(data, 'eu4000', time_detail, params, opt)

plt.ylim([0, 40])


##########################################
# Eu 3500
##########################################

time_detail = 0.2 # in microsec; 1/expt clock rate

opt = {}
opt['start_cut'] = 3
opt['plot_start_cut'] = 3
opt['single_exp'] = False
opt['plot_title'] = 'Eu 3500'

# model = a*np.exp(-x/b) + c + d*np.exp(-x/e)
params = Parameters()
params.add('a', value= 2.0, min=0.0,max=50.0)
params.add('b', value= 5.0, min=0.1,max=15.0)
params.add('c', value= 0.1, min=0.0,max=10.0) # offset
params.add('d', value= 1.0, min=0.0,max=5.0)
params.add('e', value= 218.0, min=1.0,max=400.0) #vary=False)
        
plot_data(data, 'eu3500', time_detail, params, opt)

plt.ylim([0, 2])


multipage('multipage.pdf')

plt.show()



        





# this function prints the contents of the fields in a list with newlines

# print '\n'.join([ str(myelement) for myelement in file['/data/Counter channel 2 : PMT blue/PMT blue'].items()])
