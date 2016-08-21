import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
import matplotlib.cm as cm
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.morphology import black_tophat, white_tophat, disk, square, rectangle

def gaussian_filter_correction(image, channelname,sigma):
    image_filtered = scipy.ndimage.filters.gaussian_filter(image, sigma)

    fig8 = plt.figure(figsize=(8, 6), dpi=80)
    ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
    ax1.set_title(channelname + ' channel')
    plt.imshow(image)
    ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
    ax1.set_title(channelname +' channel gaussian filtered')
    plt.imshow(image_filtered)
 
    return image_filtered

def background_correction(image, algo, channelname,size):
    
    if algo is 'white_tophat':
    ### correct for uneven illumination by morphological opening
    #This operation returns the bright spots of the image that are smaller than the structuring element.
    # Needed params
    
        selem = square(size) # disk(radius) rectangle(width,height)# square(width)# radius of disk, structuring element
        
        image_bc = white_tophat(image,selem)
        backgd = image-image_bc
    
    if algo is 'black_tophat':
        ### correct for uneven illumination by morphological closing?
        #This operation returns the dark spots of the image that are smaller than the structuring element.
        # Needed params

        selem = square(size) #disk(disksize) # rectangle(width,height)# square(width)# radius of disk, structuring element
    
        backgd = black_tophat(image,selem)
        image_bc = image - backgd
    
    
    if algo is 'medfilt':
        # "object should occupy less than 1/2 of region size" ex less than half of 24x24
        
        image_bc = image - scipy.signal.medfilt(image,size)
        backgd = scipy.signal.medfilt(image,size)
        
    # Plot results
    fig8 = plt.figure(figsize=(8, 6), dpi=80)
    plt.title(algo)
    ax1 = plt.subplot2grid((1,3), (0, 0), colspan=1)
    ax1.set_title(channelname + ' channel')
    plt.imshow(image)
    ax1 = plt.subplot2grid((1,3), (0, 1), colspan=1)
    ax1.set_title(channelname +' backgd,' + algo)
    plt.imshow(backgd)
    ax1 = plt.subplot2grid((1,3), (0, 2), colspan=1)
    ax1.set_title(channelname +' backgd subtr,' + algo)
    plt.imshow(image_bc)
    
    return image_bc

def find_opt_no_hist_bins(data, min_bins, max_bins):
     
    # from
    # http://toyoizumilab.brain.riken.jp/hideaki/res/histogram.html
     
    C = 1.0e9 #just huge number
    opt_no_bins = np.nan
    for N in range(min_bins, max_bins+1):
        hist = []
        bin_edges = []
        hist, bin_edges = np.histogram(data, bins=N)
        bin_width = 1.0*(bin_edges[1] - bin_edges[0]) #bin width should be cst for any difference between bin_edges[k]-bin_edges[k-1]         
        mean_algo = 1.0/N * sum(hist[xx] for xx in range(N))
        var_algo = 1.0/N * sum((hist[xx]-mean_algo)**2 for xx in range(N))
        if (2.0*mean_algo - var_algo)/(bin_width**2) < C:
            C = (2.0*mean_algo - var_algo)/(bin_width**2)
            opt_no_bins = N
       
    return opt_no_bins        