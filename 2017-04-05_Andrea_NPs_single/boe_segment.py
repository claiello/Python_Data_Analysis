import numpy as np
import matplotlib.pyplot as plt

import scipy
from scipy import signal


def give_bolinha(filename, xinit, yinit, xfinal, yfinal, corr_threshold = 0.35, n = 30, r = 5, save_file = False, do_plot = False):

    #corr_threshold = 0.35
    #n = 30
    #r = 5
    
    SEA = np.load(filename)
    
    se = SEA['data'][xinit:xfinal,yinit:yfinal]
    
    se = np.abs(se)
    
    se = np.max(se) - se
    
    
    template = np.zeros([20, 20])
    
    
    y,x = np.ogrid[0:n, 0:n]
    
    mask = (x-n/2.0)*(x-n/2.0) + (y-n/2.0)*(y-n/2.0) <= r*r
    
    template = np.ones((n, n))
    template[mask] = 255
    
    
    corr = signal.correlate2d(se, template, boundary='symm', mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape) # find the match
    
    corr = corr/np.max(corr)
    
    corr = corr**2
    
    corr_cut = np.copy(corr)
    
    
    
    corr_cut[corr > corr_threshold] = 1.0
    corr_cut[corr <= corr_threshold] = 0.0
    
    if do_plot:
        plt.figure()
        plt.subplot(2,2,1)
        
        plt.pcolor(se)
        
        plt.xlim(0, 260)
        plt.ylim(0, 250)
        
        plt.colorbar()
        
        plt.subplot(2,2,2)
        
        plt.pcolor(template)
        
        plt.colorbar()
        
        plt.subplot(2,2,3)
        
        plt.pcolor(corr)
        
        plt.xlim(0, 260)
        plt.ylim(0, 250)
        
        plt.colorbar()
        
        plt.subplot(2,2,4)
        
        plt.pcolor(corr_cut)
        
        plt.xlim(0, 260)
        plt.ylim(0, 250)
        
        
        plt.colorbar()
        
        plt.show()
    
    
    if save_file:
        np.savez("test.py", corr = corr)

    return corr

