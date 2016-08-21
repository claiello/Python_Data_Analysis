#Register one or two stacks of data; optional inputs toberegistered1,2 is the "passive" stack that's going to
#overgo the same shift as the images in input images
#All images are referenced to the first image in the stack of input images
#In the future, maybe reference them to one another, ie, first-second, second-third etc, to avoid big jumps
#output is padded and no longer necessarily square matrix

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import register_translation

def reg_images(images,toberegistered1=None,toberegistered2=None):

    image0 = images[0,:,:]
    offset_images = images[1:images.shape[0],:,:]
    shift_vec = np.zeros([images.shape[0],2])
    
    oyp = 0
    oym = 0
    oxp = 0
    oxm = 0
    
    for k in range(images.shape[0]-1):
    
        # pixel precision first
        shift, error, diffphase = register_translation(image0, offset_images[k,:,:])
        shift_vec[k] = shift
        #determine max and min shifts; in order to know max padding needed
        if shift_vec[k][0] > oyp:
            oyp = shift_vec[k][0]
        if shift_vec[k][0] < oym:
            oym = shift_vec[k][0]
        if shift_vec[k][1] > oxp:
            oxp = shift_vec[k][1]
        if shift_vec[k][1] < oxm:
            oxm = shift_vec[k][1]
        
        print("Detected pixel offset (y, x):")
        print(shift)
        
        # not sure how to implement subpixel precision anyways
        # subpixel precision
        #shift, error, diffphase = register_translation(image0, offset_images[k,:,:], 100)
        #print("Detected subpixel offset (y, x):")
        #print(shift)
        
    offset_images_corr = np.zeros([images.shape[0],images.shape[1]+oyp-oym,images.shape[2]+oxp-oxm])
    
    offset_images_corr[0,-oym:images.shape[1]-oym,-oxm:images.shape[2]-oxm] = image0
    
    for k in range(images.shape[0]-1):
        
        offset_images_corr[k+1,-oym+shift_vec[k][0]:-oym+shift_vec[k][0]+images.shape[1],
                               -oxm+shift_vec[k][1]:-oxm+shift_vec[k][1]+images.shape[2]] = offset_images[k,:,:]
    
    fig1 = plt.figure(figsize=(8, 6), dpi=80)
    fig1.suptitle('Reference channel for registration')
    ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
    ax1.set_title('Simple avg')
    plt.imshow(np.average(images, axis=0))
    ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
    ax1.set_title('Registered')
    plt.imshow(np.average(offset_images_corr, axis=0))
    
    if toberegistered1 is not None:
        
        if toberegistered2 is None:
        
            offset_images_corr2 = np.zeros([images.shape[0],images.shape[1]+oyp-oym,images.shape[2]+oxp-oxm])
       
            offset_images_corr2[0,-oym:images.shape[1]-oym,-oxm:images.shape[2]-oxm] = toberegistered1[0,:,:]
            
            for k in range(images.shape[0]-1):
            
                offset_images_corr2[k+1,-oym+shift_vec[k][0]:-oym+shift_vec[k][0]+images.shape[1],
                                   -oxm+shift_vec[k][1]:-oxm+shift_vec[k][1]+images.shape[2]] = toberegistered1[k,:,:]
       
            fig2 = plt.figure(figsize=(8, 6), dpi=80)
            fig2.suptitle('First passive channel for registration')
            ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
            ax1.set_title('Simple avg')
            plt.imshow(np.average(toberegistered1, axis=0))
            ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
            ax1.set_title('Registered')
            plt.imshow(np.average(offset_images_corr2, axis=0))
    
            #plt.show() 
            return np.average(offset_images_corr, axis=0), np.average(offset_images_corr2, axis=0)
            
        else:
            
            offset_images_corr2 = np.zeros([images.shape[0],images.shape[1]+oyp-oym,images.shape[2]+oxp-oxm])
            offset_images_corr3 = np.zeros([images.shape[0],images.shape[1]+oyp-oym,images.shape[2]+oxp-oxm])
       
            offset_images_corr2[0,-oym:images.shape[1]-oym,-oxm:images.shape[2]-oxm] = toberegistered1[0,:,:]
            offset_images_corr3[0,-oym:images.shape[1]-oym,-oxm:images.shape[2]-oxm] = toberegistered2[0,:,:]
            
            for k in range(images.shape[0]-1):
            
                offset_images_corr2[k+1,-oym+shift_vec[k][0]:-oym+shift_vec[k][0]+images.shape[1],
                                   -oxm+shift_vec[k][1]:-oxm+shift_vec[k][1]+images.shape[2]] = toberegistered1[k,:,:]
       
                
                offset_images_corr3[k+1,-oym+shift_vec[k][0]:-oym+shift_vec[k][0]+images.shape[1],
                                   -oxm+shift_vec[k][1]:-oxm+shift_vec[k][1]+images.shape[2]] = toberegistered2[k,:,:]
            
            fig2 = plt.figure(figsize=(8, 6), dpi=80)
            fig2.suptitle('First passive channel for registration')
            ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
            ax1.set_title('Simple avg')
            plt.imshow(np.average(toberegistered1, axis=0))
            ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
            ax1.set_title('Registered')
            plt.imshow(np.average(offset_images_corr2, axis=0))
            
            fig3 = plt.figure(figsize=(8, 6), dpi=80)
            fig3.suptitle('Second passive channel for registration')
            ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
            ax1.set_title('Simple avg')
            plt.imshow(np.average(toberegistered2, axis=0))
            ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
            ax1.set_title('Registered')
            plt.imshow(np.average(offset_images_corr3, axis=0))
    
            #plt.show() 
            
            import tifffile as tiff
            tiff.imsave('blue.tif',np.average(offset_images_corr, axis=0))
            tiff.imsave('red.tif',np.average(offset_images_corr2, axis=0))
            tiff.imsave('se.tif',np.average(offset_images_corr3, axis=0))
    
      
            return np.average(offset_images_corr, axis=0), np.average(offset_images_corr2, axis=0), np.average(offset_images_corr3, axis=0)
       
       
       
    else: 
        #plt.show() 
        return np.average(offset_images_corr, axis=0)
        
