
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 12:35:16 2016

@author: clarice
"""


########################################################################
# Imports
########################################################################

import h5py
import numpy as np

########################################################################
# Load data
########################################################################

def get_decay(filename):
    
    mydict = {}
    
    file    = h5py.File(filename, 'r') 
    
    red_dset   = file['/data/Counter channel 1 : PMT red/PMT red time-resolved TRANSIENT/data']
    blue_dset   = file['/data/Counter channel 2 : PMT blue/PMT blue time-resolved TRANSIENT/data']

    se_dset   = file['/data/Analog channel 1 : SE2/data']
    
    mydict['image'] = se_dset[0, :, :]
    
    mydict['no_of_time_points'] = red_dset.shape[1]
    
    No_of_time_points = mydict['no_of_time_points']
    
    avg_data = np.zeros([No_of_time_points])
    for kk in range(No_of_time_points):
        avg_data[kk] = np.mean(red_dset[:,kk, :, :])
 
    mydict['red_decay'] = avg_data
        
    avg_data = np.zeros([No_of_time_points])
    for kk in range(No_of_time_points):
        avg_data[kk] = np.mean(blue_dset[:,kk, :, :])
     
    mydict['blue_decay'] = avg_data
    
    file = None
    
    return mydict
    
    


def single_set(folder, base1, base2, base3, img, img_no, img_add, decay1, decay1_add, decay2, decay2_add, decay3, decay3_add):
    
    # get image
    mydict = {}

    filename = folder + base1 + img + base3 + img_add + base2 + "_" + img_no + ".hdf5"

    try:
        file    = h5py.File(filename, 'r') 
    except:
        print(filename)

    se_dset   = file['/data/Analog channel 1 : SE2/data']

    mydict['image'] = se_dset[0, :, :]
    
    file = None

    decay1_no = '1'
    decay2_no = '1'
    decay3_no = '1'
    
    # get decay data

    filename = folder + base1 + decay1 + base3 + decay1_add + base2 + "_" + decay1_no + ".hdf5"    
    mydict['decay'] = [get_decay(filename)]    
        
    filename = folder + base1 + decay2 + base3 + decay2_add + base2 + "_" + decay2_no + ".hdf5"
    mydict['decay'].append(get_decay(filename))
    
    filename = folder + base1 + decay3 + base3 + decay3_add + base2 + "_" + decay3_no + ".hdf5"
    mydict['decay'].append(get_decay(filename))
    
    
    
    # average the blue and red data
    no_of_decay_traces = 3
    
    hlp = np.zeros(mydict['decay'][0]['blue_decay'].shape)
    for kk in range(no_of_decay_traces):
        hlp += mydict['decay'][kk]['blue_decay']        
    hlp /= 3.0
    
    mydict['decay_blue_avg'] = hlp
    
    hlp = np.zeros(mydict['decay'][0]['red_decay'].shape)
    for kk in range(no_of_decay_traces):
        hlp += mydict['decay'][kk]['red_decay']        
    hlp /= 3.0
    
    mydict['decay_red_avg'] = hlp
    
    return mydict
    


def load_data():

    data = {}
    data['ce3500'] = {}
    data['ce4000'] = {}
    data['eu3500'] = {}
    data['ce4000'] = {}

    folder1 = '2016-04-07-1143/'
    folder2 = '2016-04-07-1014/'

    base1 = '2016-04-07-'
    base2 = '7.500kX_2.000kV_30mu'

    # Load Ce 3500

    base3 = '_ImageSequence_Ce3500'

    img = '1111'
    img_no = '1'
    img_add = 'Image_'
    decay1 = '1118'
    decay1_add = 'FDClock20Decay2_'
    decay2 = '1122'
    decay2_add = 'FDClock20Decay2AnotherRegion_'
    decay3 = '1123'
    decay3_add = 'FDClock20Decay2ThirdRegion_'

    data['ce3500'] = single_set(folder2, base1, base2, base3, img, img_no, img_add, decay1, decay1_add, decay2, decay2_add, decay3, decay3_add)

    # Load Ce 4000

    base3 = '_ImageSequence_continuacao-Ce4000'

    img = '1147'
    img_no = '1'
    img_add = 'PDClock20_'
    decay1 = '1147'
    decay1_add = 'PDClock20_'
    decay2 = '1150'
    decay2_add = 'PDClock20Region2_'
    decay3 = '1152'
    decay3_add = 'PDClock20Region3_'

    data['ce4000'] = single_set(folder1, base1, base2, base3, img, img_no, img_add, decay1, decay1_add, decay2, decay2_add, decay3, decay3_add)

    # Load Eu 3500

    base3 = '_ImageSequence_continuacao-Eu3500'

    img = '1157'
    img_no = '2'
    img_add = 'Image_'
    decay1 = '1225'
    decay1_add = 'PDClock5AA_'
    decay2 = '1228'
    decay2_add = 'PDClock5BB_'
    decay3 = '1229'
    decay3_add = 'PDClock5CC_'

    data['eu3500'] = single_set(folder1, base1, base2, base3, img, img_no, img_add, decay1, decay1_add, decay2, decay2_add, decay3, decay3_add)

    # Load Eu 4000

    base3 = '_ImageSequence_continuacao-Eu4000'

    img = '1249'
    img_no = '5'
    img_add = 'Image_'
    decay1 = '1251'
    decay1_add = 'PDClock5AAA_'
    decay2 = '1254'
    decay2_add = 'PDClock5BBB_'
    decay3 = '1256'
    decay3_add = 'PDClock5CCC_'

    data['eu4000'] = single_set(folder1, base1, base2, base3, img, img_no, img_add, decay1, decay1_add, decay2, decay2_add, decay3, decay3_add)



    return data
    