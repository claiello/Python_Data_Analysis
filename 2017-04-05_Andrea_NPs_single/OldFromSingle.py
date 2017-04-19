# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:56:29 2017

@author: clarice
"""

do_600= False #ran already - can just load data
if do_600:
    
    print('do600')
    
    length_scalebar = 100.0
    Pixel_size = np.array([1.48]) 
    
    SEA= np.load('x600SEchannel.npz') #init shape (342, 315)
    xlen = SEA['data'].shape[0]
    ylen = SEA['data'].shape[1]
    
    xinit = 100
    xfinal = -100
    yinit = 45
    yfinal = -45
    se = SEA['data'][xinit:xfinal,yinit:yfinal]
    
    new_pic = give_bolinha('x600SEchannel.npz', xinit, yinit, xfinal, yfinal, corr_threshold = 0.35, n = 30, r = 8, save_file = False, do_plot = False)
#    cutx = 20
#    cuty = 1
    #se = se[0:-cutx, cuty:-cuty]

    ax0022b.imshow(se,cmap=cm.Greys_r)
    ax0022b.axis('off')
    
    #new_pic = new_pic[0:-cutx, cuty:-cuty]
    setr = new_pic
    #binary threshold
    se_data2 = np.copy(setr)
    
    new_hlp = new_pic
    I8 = (new_hlp * 255.9).astype(np.uint8)
    bw = cv2.adaptiveThreshold(I8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 0)
    from scipy import ndimage
    hlpse2 = bw 
    hlpse2[hlpse2 > 1] = 1.0
    hlpse2[hlpse2 < 1] = 0.0
    distance = ndimage.distance_transform_edt(hlpse2)
     def tauestimate(counts_red, error_red, counts_blue, error_blue):
    
        print(counts_red.shape[1])
        
        ucounts_red = unumpy.uarray(counts_red, error_red)
        ucounts_blue = unumpy.uarray(counts_blue, error_blue)
        
        def helper(arrayx):
             return np.cumsum(arrayx*np.arange(1,counts_red.shape[1]+1), axis = 1)/np.cumsum(arrayx, axis = 1)
        
        return helper(ucounts_red),helper(ucounts_blue)
    local_maxi = peak_local_max(
        distance, 
        num_peaks = 50, 
        indices = False, 
        footprint = np.ones((50,50)),
        labels = hlpse2) #footprint = min dist between maxima to find #footprint was 25,25
    markers = skimage.morphology.label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=hlpse2)
    lab = np.unique(labels_ws)
    
    # Make random colors, not degrade
    rand_ind = np.random.permutation(lab)
    new_labels_ws = np.copy(labels_ws)
    for k in range(new_labels_ws.shape[0]):
        for j in range(new_labels_ws.shape[1]):
            new_labels_ws[k, j] = rand_ind[labels_ws[k, j]]
    labels_ws =  new_labels_ws
    
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[0]))
    sbar = sb.AnchoredScaleBar(ax0022b.transData, length_scalebar_in_pixels, "", style = 'bright', loc = 8, my_fontsize = fsizenb, my_linewidth= 2)
    ax0022b.add_artist(sbar)
    
    areas = np.array([])
    for k in lab:
        areas = np.append(areas, len( labels_ws[labels_ws == k] ))
    cut_k = []
    cut_labels_ws = np.copy(labels_ws)
    non_cut_k = []  ###### change cut_k
    for k in range(len(lab)):
        if (areas[k] < 10) or (areas[k] > 4000):# or (k == 0): 
            cut_labels_ws[cut_labels_ws == lab[k]] = 0
            cut_k.append(k)
        else:
            non_cut_k.append(k)  ###change cut_k
    
    axpicb.imshow(cut_labels_ws, cmap = cm.Greys_r) #or 'OrRd'
    axpicb.axis('off')
    
#    plt.show()
#    
#    import IPython
#    IPython.embed()
    
    del SEA, se
    gc.collect()
    
    ####### load file that exists
    C = pickle.load( open( "600.p", "rb" ) )
    areas = C['areas']
    taured = C['taured']
    tauredstd = C['tauredstd']
    taublue = C['taublue']
    taubluestd = C['taubluestd']
    intens = C['intens']
    stdintens = C['intensstd']
    intensr = C['intensr']
    stdintensr = C['intensstdr']
    intensb = C['intensb']
    stdintensb = C['intensstdb']
    non_cut_k = C['non_cut_k']  
    
    
#    REDA = np.load('x600Redbright.npz')
#    reda = REDA['data'][:,:,xinit:xfinal,yinit:yfinal] #same no pixels than C, single
#    del REDA
#    gc.collect()
#    BLUEA = np.load('x600Bluebright.npz')
#    bluea = BLUEA['data'][:,:,xinit:xfinal,yinit:yfinal]#same no pixels than C, single
#    del BLUEA
#    gc.collect() 
#    
#    no_avg = reda.shape[0]
#    intens = np.empty([len(lab),no_avg])
#    stdintens = np.empty([len(lab),no_avg])
#    intensr = np.empty([len(lab),no_avg])
#    stdintensr = np.empty([len(lab),no_avg])
#    intensb = np.empty([len(lab),no_avg])
#    stdintensb = np.empty([len(lab),no_avg])
#    
#    taured = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
#    taublue = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
#    tauredstd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
#    taubluestd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
    
    for k in non_cut_k:
        pass
    
#       print('indice600')
#       print(k)
#       
#       #Ratio
#       print('Ratio')
#       hlp = np.zeros(cut_labels_ws.shape)
#       hlp[cut_labels_ws == lab[k]] = 1.0
#       hlp[cut_labels_ws != lab[k]] = np.nan
#       ureda  =  reda[:,backgroundinit:initbin,:,:]
#       ubluea =  bluea[:,backgroundinit:initbin,:,:]
#                     
#       vecr = np.nanmean(ureda * hlp, axis = (1,2,3))
#       vecb = np.nanmean(ubluea * hlp, axis = (1,2,3))
#       
#       vec = vecr/vecb
#       
#       No = np.nansum(hlp.astype(np.float64))*reda[:,backgroundinit:initbin,:,:].shape[1]
#              
#       #import IPython
#       #IPython.embed()
#       
#       vecstd = vec * np.sqrt( 1.0/vecr/No + 1.0/vecb/No )
#       
#       vecstdr = 1.0/np.sqrt(No) * np.sqrt(vecr) 
#       vecstdb = 1.0/np.sqrt(No) * np.sqrt(vecb) 
#       del ureda, ubluea
#       gc.collect()
#       intens[k,:] = vec
#       stdintens[k,:] = vecstd
#       intensr[k,:] = vecr
#       stdintensr[k,:] = vecstdr
#       intensb[k,:] = vecb
#       stdintensb[k,:] = vecstdb
#       print(vec)
#       print(vecstd)
#       del vec, vecstd
#       gc.collect()
#       
#     
#       
#       print('Taus')
##       #Taus as a function of timeC['taublue'][C['non_cut_k']],
#            axhist300tau_blue,
#            axexp300tau_blue,
#            which_taus = [250, 500, 1000],
#            no_of_bins = 30,
#            my_color = 'b')
#       hlp = np.zeros(cut_labels_ws.shape)
#       hlp[cut_labels_ws == lab[k]] = 1.0
#       Notr = np.sum(hlp.astype(np.float64))
#       redd = np.sum(reda[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
#       blued = np.sum(bluea[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
#       hr = tauestimate(redd,np.sqrt(redd)/np.sqrt(Notr))
#       taured[k,:,:] = unumpy.nominal_values(hr)
#       tauredstd[k,:,:] = unumpy.std_devs(hr)
#       hb = tauestimate(blued,np.sqrt(blued)/np.sqrt(Notr))
#       taublue[k,:,:] = unumpy.nominal_values(hb)
#       taubluestd[k,:,:] = unumpy.std_devs(hb)
#       sizered = 1398 
#       del hr, hb, redd, blued
#       gc.collect()
   
     
    axratiomag.errorbar(3,np.nanmean(C['intens'][C['non_cut_k']],axis=(0,1)),yerr=np.nanstd(C['intens'][C['non_cut_k']],axis=(0,1)),marker='o', color='k', markersize=12) 
    axtaumag.plot(np.arange(0,1398),np.average(C['taured'][C['non_cut_k']],axis=(0,1)),lw=3,color='r')
    axtaumag.plot(np.arange(0,1398),np.average(C['taublue'][C['non_cut_k']],axis=(0,1)),lw=3,color='g')
    axstdmag.plot(np.arange(0,1398),np.std(C['taured'][C['non_cut_k']],axis=(0,1))/np.average(C['taured'][C['non_cut_k']],axis=(0,1))*100.0,lw=3,color='r')
    axstdmag.plot(np.arange(0,1398),np.std(C['taublue'][C['non_cut_k']],axis=(0,1))/np.average(C['taublue'][C['non_cut_k']],axis=(0,1))*100.0,lw=3,color='g')      

    plt.figure()
    plt.hist(C['taublue'][C['non_cut_k']][:, 0, -1].flatten())
#    del reda, bluea
#    gc.collect()
    
#    save_data = {}
#    save_data['areas'] = areas*Pixel_size**2 #in nm^2
#    save_data['taured'] = taured
#    save_data['tauredstd'] = tauredstd
#    save_data['taublue'] = taublue
#    save_data['taubluestd'] = taubluestd
#    save_data['intens'] = intens
#    save_data['intensstd'] = stdintens
#    save_data['intensr'] = intensr
#    save_data['intensstdr'] = stdintensr
#    save_data['intensb'] = intensb
#    save_data['intensstdb'] = stdintensb
#    save_data['non_cut_k'] = non_cut_k
#    
#    del taured, taublue
#    gc.collect()
#    
#    pickle.dump(save_data, open("600.p", "wb"))   
#    
#    print('radius found for C') #to check which of the found areas is background, which is signal
##    print(tor(areas[non_cut_k])) #if used saved areas
#    print(tor(areas*Pixel_size**2))
#    print(non_cut_k)

   