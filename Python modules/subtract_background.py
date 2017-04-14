# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:34:11 2017

@author: clarice
"""

from my_fits import parabola_fit
import matplotlib.pyplot as plt
import numpy as np

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def subtract_background(wavelengths, counts, low_region, high_region):
    
    # low(high)_region =  [vmin, vmax]
    # spectrum 
    # wavelength = specBG[:,0]
    # spectrum values = specBG[:,1]
    
    wavelengths = np.array(wavelengths)
    counts = np.array(counts)
    
    # get indices of regions
    ind0_low = find_nearest(wavelengths, low_region[0])
    ind1_low = find_nearest(wavelengths, low_region[1])

    ind0_high = find_nearest(wavelengths, high_region[0])
    ind1_high = find_nearest(wavelengths, high_region[1])

    # fit low and high regions
    x = np.append(wavelengths[ind0_low:ind1_low], wavelengths[ind0_high:ind1_high])
    y = np.append(counts[ind0_low:ind1_low], counts[ind0_high:ind1_high])
    
    result = parabola_fit(x, y)
    
    a = result.params['a'].value
    b = result.params['b'].value    
    c = result.params['c'].value
    
#    plt.figure(40)
#    plt.plot(wavelengths, counts, 'go')
#    plt.plot(x, y, 'ro')
#    plt.plot(x, a*x**2+b*x+c, 'k-')
#    
#    plt.xlim(min(x), max(x))    
#    
#    plt.show()    
    xx = wavelengths[ind0_low:ind1_high]
    counts[ind0_low:ind1_high] = counts[ind0_low:ind1_high]  - (a*xx**2 + b*xx + c)
    return counts