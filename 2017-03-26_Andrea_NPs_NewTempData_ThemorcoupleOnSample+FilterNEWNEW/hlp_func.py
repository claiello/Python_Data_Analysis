import numpy as np
import gc

def calc_red_blue(let, index, filename, red_int_array, red_decay_array, red_std_array, bgred_int_array, bgred_decay_array, bgred_std_array, hlp, hlpd):

    # filename = Redbright.npz or Bluebright.npz

    red0 = np.load(str(let[index]) + filename, mmap_mode='r')  
    
    #blue0 = np.load(str(let[index]) +'Bluebright.npz',mmap_mode='r')  
    #se = np.load(str(let[index]) +'SEchannel.npz',mmap_mode='r') 
    
    red = red0['data']
    del red0
    
    gc.collect()
#       
#       #segmm, means, covars, weights = gmmone_tr_in_masked_channel_modif_memory_issue(se['data'][xinit[index]:xend[index],yinit[index]:yend[index]], red[:,:,xinit[index]:xend[index],yinit[index]:yend[index]]) 
#       ##############################################################
#       #red = red[:,:,xinit[index]:xend[index],yinit[index]:yend[index]]
#       #blue = blue[:,:,xinit[index]:xend[index],yinit[index]:yend[index]]
#       
#       #del means, covars, weights
#       #gc.collect()
#       
    backgdinit = 50
    initbin = (150+50+3)-1
#   
#       print('after skimage')
#       
#       #################
#       
#       #to plot the pics, uncomment 5 next lines
#   #    if True:
#   #        #axvec[index].imshow(se['data'][xinit[index]:xend[index],yinit[index]:yend[index]],cmap=cm.Greys) #or 'OrRd'
#   #        axvec[index].imshow(segmm,cmap=cm.Greys)
#   #        print('after imshow')   
#   #        del segmm, red, blue,se
#   #        gc.collect()
#   ##multipage_longer('Checkcuts.pdf',dpi=80)
#   #multipage_longer('Checksegmm.pdf',dpi=80)
#   #klklkk  
#   #if True:
#            
#       datared = np.average(red, axis = (0))
#        
#       if True is False:
#            pass
#       else:
#            initbin = (150+50+3)-1 #init bin for decay
#            backgdinit = 50
#            ### 700ns /40ns = 7. ....
#            datared_init = datared[0:backgdinit,:,:]
#            datared = datared[initbin:,:,:]
#   
#    del datared, datared_init
    gc.collect()

    dataALLred = red #[:,:,:,:]

    #del red

    nominal_time_on = 150.0
    
    print('bef nanmean')
     
    red_int_array[index] = np.average(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) 
    gc.collect()
    print('1')
     
    red_decay_array[index,:] = np.average(dataALLred[:,initbin:,:,:]*hlp,axis=(0,2,3))
    gc.collect()
    print('3')
     
    red_std_array[index] = np.std(dataALLred[:,backgdinit:initbin,:,:]*hlp,axis=(0,1,2,3)) 
    gc.collect()
    print('5')
    
    bgred_int_array[index] = np.average(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3)) 
    gc.collect()
    print('7')
     
    bgred_decay_array[index,:] = np.average(dataALLred[:,initbin:,:,:]*hlpd,axis=(0,2,3))
    gc.collect()
    print('9')
     
    bgred_std_array[index] = np.std(dataALLred[:,backgdinit:initbin,:,:]*hlpd,axis=(0,1,2,3))
    gc.collect()
    print('11')
    
    print('after nanmean')
    
    del dataALLred
    gc.collect()
 
