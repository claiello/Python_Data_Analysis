#Register one or two stacks of data; optional inputs toberegistered1,2 is the "passive" stack that's going to
#overgo the same shift as the images in input images
#All images are referenced to the first image in the stack of input images
#In the future, maybe reference them to one another, ie, first-second, second-third etc, to avoid big jumps
#output is padded and no longer necessarily square matrix

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from skimage.feature import register_translation
import gc

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
        shift, error, diffphase = register_translation(image0, offset_images[k,:,:])#,space='fourier') #,100) #upsampling is (optional) last value
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
        return np.average(offset_images_corr, axis=0), offset_images_corr
        
#time resolved image saved as
#no frames x no tr pts x pixels x pixels
        
def reg_images_tr(imagesall,toberegistered1=None,toberegistered2=None):
    
    offset_images_corr_list = []
    for jjj in range(imagesall.shape[1]-1):
        images = imagesall[:,jjj,:,:].reshape((imagesall.shape[0],imagesall.shape[2],imagesall.shape[3]))

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
        
    #    if toberegistered1 is not None:
    #        
    #        if toberegistered2 is None:
    #        
    #            offset_images_corr2 = np.zeros([images.shape[0],images.shape[1]+oyp-oym,images.shape[2]+oxp-oxm])
    #       
    #            offset_images_corr2[0,-oym:images.shape[1]-oym,-oxm:images.shape[2]-oxm] = toberegistered1[0,:,:]
    #            
    #            for k in range(images.shape[0]-1):
    #            
    #                offset_images_corr2[k+1,-oym+shift_vec[k][0]:-oym+shift_vec[k][0]+images.shape[1],
    #                                   -oxm+shift_vec[k][1]:-oxm+shift_vec[k][1]+images.shape[2]] = toberegistered1[k,:,:]
    #       
    #            fig2 = plt.figure(figsize=(8, 6), dpi=80)
    #            fig2.suptitle('First passive channel for registration')
    #            ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
    #            ax1.set_title('Simple avg')
    #            plt.imshow(np.average(toberegistered1, axis=0))
    #            ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
    #            ax1.set_title('Registered')
    #            plt.imshow(np.average(offset_images_corr2, axis=0))
    #    
    #            #plt.show() 
    #            return np.average(offset_images_corr, axis=0), np.average(offset_images_corr2, axis=0)
    #            
    #        else:
    #            
    #            offset_images_corr2 = np.zeros([images.shape[0],images.shape[1]+oyp-oym,images.shape[2]+oxp-oxm])
    #            offset_images_corr3 = np.zeros([images.shape[0],images.shape[1]+oyp-oym,images.shape[2]+oxp-oxm])
    #       
    #            offset_images_corr2[0,-oym:images.shape[1]-oym,-oxm:images.shape[2]-oxm] = toberegistered1[0,:,:]
    #            offset_images_corr3[0,-oym:images.shape[1]-oym,-oxm:images.shape[2]-oxm] = toberegistered2[0,:,:]
    #            
    #            for k in range(images.shape[0]-1):
    #            
    #                offset_images_corr2[k+1,-oym+shift_vec[k][0]:-oym+shift_vec[k][0]+images.shape[1],
    #                                   -oxm+shift_vec[k][1]:-oxm+shift_vec[k][1]+images.shape[2]] = toberegistered1[k,:,:]
    #       
    #                
    #                offset_images_corr3[k+1,-oym+shift_vec[k][0]:-oym+shift_vec[k][0]+images.shape[1],
    #                                   -oxm+shift_vec[k][1]:-oxm+shift_vec[k][1]+images.shape[2]] = toberegistered2[k,:,:]
    #            
    #            fig2 = plt.figure(figsize=(8, 6), dpi=80)
    #            fig2.suptitle('First passive channel for registration')
    #            ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
    #            ax1.set_title('Simple avg')
    #            plt.imshow(np.average(toberegistered1, axis=0))
    #            ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
    #            ax1.set_title('Registered')
    #            plt.imshow(np.average(offset_images_corr2, axis=0))
    #            
    #            fig3 = plt.figure(figsize=(8, 6), dpi=80)
    #            fig3.suptitle('Second passive channel for registration')
    #            ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
    #            ax1.set_title('Simple avg')
    #            plt.imshow(np.average(toberegistered2, axis=0))
    #            ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
    #            ax1.set_title('Registered')
    #            plt.imshow(np.average(offset_images_corr3, axis=0))
    #    
    #            #plt.show() 
    #            
    #            import tifffile as tiff
    #            tiff.imsave('blue.tif',np.average(offset_images_corr, axis=0))
    #            tiff.imsave('red.tif',np.average(offset_images_corr2, axis=0))
    #            tiff.imsave('se.tif',np.average(offset_images_corr3, axis=0))
    #    
    #      
    #            return np.average(offset_images_corr, axis=0), np.average(offset_images_corr2, axis=0), np.average(offset_images_corr3, axis=0)
    #       
    #       
    #       
    #    else: 
            #plt.show() 
        offset_images_corr_list.append(np.average(offset_images_corr, axis=0))
        
    fig1 = plt.figure(figsize=(8, 6), dpi=80)
    fig1.suptitle('Reference channel for registration')
    
    ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
    ax1.set_title('Simple avg')
    plt.imshow(np.average(images, axis=0))
    ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
    ax1.set_title('Registered')
    plt.imshow(np.average(offset_images_corr, axis=0))
    
    # function to update figure
    def updatefigs(j):
        # set the data in the axesimage object
        im1.set_array(np.average(imagesall[:,j,:,:],axis=0))
        im2.set_array(offset_images_corr_list[j])
        # return the artists set
        return im1, im2
        
    ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
    ax1.set_title('Simple avg')
    im1 = plt.imshow(np.average(imagesall[:,0,:,:].reshape([imagesall.shape[0],imagesall.shape[2],imagesall.shape[3]]),axis=0)) #first image
    
    ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
    ax1.set_title('Registered')
    im2 = plt.imshow(offset_images_corr_list[0]) #first image
    
    anim = animation.FuncAnimation(fig1, updatefigs,frames=range(imagesall.shape[1]-1), interval=10, blit=True,repeat_delay=100)                           

    plt.show() 
        
    return offset_images_corr_list

def reg_time_resolved_images_to_se(images,toberegistered1=None,toberegistered2=None):

    image0 = images[0,:,:] #se
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
    
    #Optional plotting
#    fig1 = plt.figure(figsize=(8, 6), dpi=80)
#    fig1.suptitle('Reference channel for registration')
#    ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
#    ax1.set_title('Simple avg')
#    plt.imshow(np.average(images, axis=0))
#    ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
#    ax1.set_title('Registered')
#    plt.imshow(np.average(offset_images_corr, axis=0))
    
    if toberegistered1 is not None:
        
        if toberegistered2 is None:
        
            offset_images_corr2 = np.zeros([images.shape[0],toberegistered1.shape[1],images.shape[1]+oyp-oym,images.shape[2]+oxp-oxm])
       
            for kk in range(toberegistered1.shape[1]-1):
                print(kk/(toberegistered1.shape[1]-1.0))
                offset_images_corr2[0,kk,-oym:images.shape[1]-oym,-oxm:images.shape[2]-oxm] = toberegistered1[0,kk,:,:]
            
                for k in range(images.shape[0]-1):
            
                    offset_images_corr2[k+1,kk,-oym+shift_vec[k][0]:-oym+shift_vec[k][0]+images.shape[1],
                                   -oxm+shift_vec[k][1]:-oxm+shift_vec[k][1]+images.shape[2]] = toberegistered1[k,kk,:,:]
       
       #OPTIONAL PLOTTING
#            fig2 = plt.figure(figsize=(8, 6), dpi=80)
#            fig2.suptitle('First passive channel for registration')
#               
#            avg_arr = np.empty((toberegistered1.shape[1],toberegistered1.shape[2],toberegistered1.shape[3]))
#            reg_arr = np.empty((offset_images_corr2.shape[1],offset_images_corr2.shape[2],offset_images_corr2.shape[3]))
#            for k in range(toberegistered1.shape[1]-1):
#                print(k/(toberegistered1.shape[1]-1.0))
#                avg_arr[k,:,:] = np.average(toberegistered1[:,k,:,:],axis=0)
#                reg_arr[k,:,:] = np.average(offset_images_corr2[:,k,:,:],axis=0)
                      
            # function to update figure
#            def updatefigs(j):
#                # set the data in the axesimage object
#                im1.set_array(avg_arr[j])
#                im2.set_array(reg_arr[j])
#                # return the artists set
#                return im1, im2
#                
#            ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
#            ax1.set_title('Simple avg')
#            im1 = plt.imshow(np.average(toberegistered1[:,0,:,:],axis=0)) #first image
#            
#            ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
#            ax1.set_title('Registered')
#            im2 = plt.imshow(np.average(offset_images_corr2[:,0,:,:],axis=0)) #first image
            
            #anim = animation.FuncAnimation(fig2, updatefigs,frames=range(toberegistered1.shape[1]), interval=20, blit=True,repeat_delay=1000)         
            # save the animation as an mp4.  This requires ffmpeg or mencoder to be
            # installed.  The extra_args ensure that the x264 codec is used, so that
            # the video can be embedded in html5.  You may need to adjust this for
            # your system: for more information, see
            # http://matplotlib.sourceforge.net/api/animation_api.html
           
            #mywriter = animation.FFMpegWriter()
            #anim.save('RegistrationTimeResolved.mp4', writer=mywriter,fps=30, extra_args=['-vcodec', 'libx264'])                  
    
            #plt.show() 
            
            return np.average(offset_images_corr, axis=0), np.average(offset_images_corr2, axis=0), offset_images_corr2 #2nd to last vector will be 1x no tr pts x pixels x pixels
                        
        else: #update in the future for aligning 2 time resolved channels to se; right now wont work!!!
            
            return
#            offset_images_corr2 = np.zeros([images.shape[0],images.shape[1]+oyp-oym,images.shape[2]+oxp-oxm])
#            offset_images_corr3 = np.zeros([images.shape[0],images.shape[1]+oyp-oym,images.shape[2]+oxp-oxm])
#       
#            offset_images_corr2[0,-oym:images.shape[1]-oym,-oxm:images.shape[2]-oxm] = toberegistered1[0,:,:]
#            offset_images_corr3[0,-oym:images.shape[1]-oym,-oxm:images.shape[2]-oxm] = toberegistered2[0,:,:]
#            
#            for k in range(images.shape[0]-1):
#            
#                offset_images_corr2[k+1,-oym+shift_vec[k][0]:-oym+shift_vec[k][0]+images.shape[1],
#                                   -oxm+shift_vec[k][1]:-oxm+shift_vec[k][1]+images.shape[2]] = toberegistered1[k,:,:]
#       
#                
#                offset_images_corr3[k+1,-oym+shift_vec[k][0]:-oym+shift_vec[k][0]+images.shape[1],
#                                   -oxm+shift_vec[k][1]:-oxm+shift_vec[k][1]+images.shape[2]] = toberegistered2[k,:,:]
#            
#            fig2 = plt.figure(figsize=(8, 6), dpi=80)
#            fig2.suptitle('First passive channel for registration')
#            ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
#            ax1.set_title('Simple avg')
#            plt.imshow(np.average(toberegistered1, axis=0))
#            ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
#            ax1.set_title('Registered')
#            plt.imshow(np.average(offset_images_corr2, axis=0))
#            
#            fig3 = plt.figure(figsize=(8, 6), dpi=80)
#            fig3.suptitle('Second passive channel for registration')
#            ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
#            ax1.set_title('Simple avg')
#            plt.imshow(np.average(toberegistered2, axis=0))
#            ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
#            ax1.set_title('Registered')
#            plt.imshow(np.average(offset_images_corr3, axis=0))
#    
#            #plt.show() 
#            
#            import tifffile as tiff
#            tiff.imsave('blue.tif',np.average(offset_images_corr, axis=0))
#            tiff.imsave('red.tif',np.average(offset_images_corr2, axis=0))
#            tiff.imsave('se.tif',np.average(offset_images_corr3, axis=0))
#    
#      
#            return np.average(offset_images_corr, axis=0), np.average(offset_images_corr2, axis=0), np.average(offset_images_corr3, axis=0)
       
       
       
    else: 
        #plt.show() 
        return np.average(offset_images_corr, axis=0)
        

def reg_images_middle_cl(images,middle_index,end_left_index,end_right_index,toberegistered1=None,toberegistered2=None):
    #gives back vectros of same size
    #right now, end_left and end_right indices are not used; those are the indices where registration should stop, and should use registration deltas of last index at that side

    imageref = images[middle_index,:,:]

    # copy images but delete the middle_index image
    offset_images = np.delete(images, middle_index, 0) #images[1:images.shape[0],:,:]
    shift_vec = np.zeros([images.shape[0],2])
    
    oyp = 0
    oym = 0
    oxp = 0
    oxm = 0
    
    for k in range(images.shape[0]-1):
    
        # pixel precision first
        shift, error, diffphase = register_translation(imageref, offset_images[k,:,:])
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
    
    offset_images_corr[middle_index,-oym:images.shape[1]-oym,-oxm:images.shape[2]-oxm] = imageref
    
    for k in range(images.shape[0]-1):
        
        if k >= middle_index:
            kk = k+1
        else:
            kk = k

        offset_images_corr[kk,-oym+shift_vec[k][0]:-oym+shift_vec[k][0]+images.shape[1],
                               -oxm+shift_vec[k][1]:-oxm+shift_vec[k][1]+images.shape[2]] = offset_images[k,:,:]
    
#    fig1 = plt.figure(figsize=(8, 6), dpi=80)
#    fig1.suptitle('Reference channel for registration')
#    ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
#    ax1.set_title('Simple avg')
#    plt.imshow(np.average(images, axis=0))
#    ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
#    ax1.set_title('Registered')
#    plt.imshow(np.average(offset_images_corr, axis=0))
    
    if toberegistered1 is not None:
        
        if toberegistered2 is None:
        
            offset_images_corr2 = np.zeros([images.shape[0],images.shape[1]+oyp-oym,images.shape[2]+oxp-oxm])
       
            offset_images_corr2[0,-oym:images.shape[1]-oym,-oxm:images.shape[2]-oxm] = toberegistered1[0,:,:]
            
            for k in range(images.shape[0]-1):
            
                offset_images_corr2[k+1,-oym+shift_vec[k][0]:-oym+shift_vec[k][0]+images.shape[1],
                                   -oxm+shift_vec[k][1]:-oxm+shift_vec[k][1]+images.shape[2]] = toberegistered1[k,:,:]
       
#            fig2 = plt.figure(figsize=(8, 6), dpi=80)
#            fig2.suptitle('First passive channel for registration')
#            ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
#            ax1.set_title('Simple avg')
#            plt.imshow(np.average(toberegistered1, axis=0))
#            ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
#            ax1.set_title('Registered')
#            plt.imshow(np.average(offset_images_corr2, axis=0))
    
            #plt.show() 
            return offset_images_corr, offset_images_corr2
            
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
    
      
            return offset_images_corr, offset_images_corr2, offset_images_corr3
       
       
       
    else: 
        #return offset_images_corr 

        #plt.show() 
        return offset_images_corr[:,np.round((+oyp-oym)/2.0):images.shape[1]+np.round((+oyp-oym)/2.0),np.round((+oxp-oxm)/2.0):images.shape[2]+np.round((+oxp-oxm)/2.0)] 
        #keep same size, not best way of doing this; just keeping center pixels
     
