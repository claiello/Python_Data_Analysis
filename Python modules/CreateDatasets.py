import numpy as np
import matplotlib.pyplot as plt
from BackgroundCorrection import *
from CorrCoeffs import *
import scipy.ndimage as ndimage
from matplotlib_scalebar.scalebar import ScaleBar
from sklearn.mixture import GMM 
import matplotlib.cm as cm
from skimage.filters import threshold_otsu, threshold_adaptive 
import skimage
from skimage.segmentation import random_walker
import gc

###fig.savefig('whatever.png', facecolor=fig.get_facecolor(), edgecolor='none')

#def choose_segmentation(choice):
#    
#    if choice is gmmse:
#        print('doing gmm se')
        

def do_analysis(baseimage1, baseimage2, background1, foreground1, background2, foreground2, name1, name2, datasetname, subsetA, subsetB, Pixel_size):
   #do_analysis(blue_dset_cut, red_dset_cut, below_blue, above_blue, below_red, above_red, 'Blue', 'Red','Above/Below avg', 'below avg', 'above avg')
    # it is assumed the backgroung will appear first (in background1, background2)
    # it is assumed YAP channel will appear first (in baseimage1, background1 and foreground1)
    # all images in rate, kcps
    
    fig5 = plt.figure(figsize=(8, 6), dpi=80)
    fig5.suptitle(datasetname + ' subsets', fontsize =20)
    rect = fig5.patch
    rect.set_facecolor('white') 
    
    nrows = 9#4
    ncols = 6#2
    
    ## figures
    ax1 = plt.subplot2grid((nrows, ncols), (0, 0), colspan=3,rowspan=3)
    ax1.set_title(name1 + ' channel ' + subsetA)
    plt.imshow(background1,cmap='Blues')
    scalebar = ScaleBar(Pixel_size, frameon = True, box_alpha = 0.001, location = 'upper right') # 1 pixel = Pixel_size in meter
    plt.gca().add_artist(scalebar)
    plt.axis('off')    
    ax1 = plt.subplot2grid((nrows, ncols), (3, 0), colspan=3,rowspan=3)
    ax1.set_title(name1 + ' channel ' + subsetB)
    plt.imshow(foreground1,cmap='Reds')
    scalebar = ScaleBar(Pixel_size, frameon = True, box_alpha = 0.001, location = 'upper right') # 1 pixel = Pixel_size in meter
    plt.gca().add_artist(scalebar)
    plt.axis('off') 
    ax1 = plt.subplot2grid((nrows, ncols), (0,3), colspan=3,rowspan=3)
    ax1.set_title(name2 + ' channel ' +  subsetA)
    plt.imshow(background2,cmap='Blues')
    scalebar = ScaleBar(Pixel_size, frameon = True, box_alpha = 0.001, location = 'upper right') # 1 pixel = Pixel_size in meter
    plt.gca().add_artist(scalebar)
    plt.axis('off') 
    ax1 = plt.subplot2grid((nrows, ncols), (3, 3), colspan=3,rowspan=3)
    ax1.set_title(name2 + ' channel ' + subsetB)
    plt.imshow(foreground2,cmap='Reds')
    scalebar = ScaleBar(Pixel_size, frameon = True, box_alpha = 0.001, location = 'upper right') # 1 pixel = Pixel_size in meter
    plt.gca().add_artist(scalebar)
    plt.axis('off') 
    
    ## histograms
    min_bins = 10
    max_bins = 500
    
    ax1 = plt.subplot2grid((nrows, ncols), (6, 0), colspan=3,rowspan=2)
    
    #only do hist whenre image non nan, and counts > 0
    # (neg counts can arise from background subtraction)
    im1 = baseimage1[np.where((~np.isnan(baseimage1)) & (baseimage1 > 0))]
    No_bins = find_opt_no_hist_bins(im1/1000.0,min_bins, max_bins)
    hist, bin_edges = np.histogram(im1/1000.0, bins=No_bins)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])    
    plt.plot(bin_centers, hist, lw=2,color='k')
    plt.axvline(np.nanmean(im1)/1000.0, color='k', linestyle='dashed', linewidth=2)
    
    #Usually binary background
    im2 = background1[np.where((~np.isnan(background1)) & (background1 > 0))]
    No_bins = find_opt_no_hist_bins(im2/1000.0,min_bins, max_bins)
    hist, bin_edges = np.histogram(im2/1000.0, bins=No_bins)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])    
    plt.plot(bin_centers, hist, lw=2,color='b')
    plt.axvline(np.nanmean(im2)/1000.0, color='b', linestyle='dashed', linewidth=2)
    
    #Usually binary foreground
    im3 = foreground1[np.where((~np.isnan(foreground1)) & (foreground1 > 0))]
    No_bins = find_opt_no_hist_bins(im3/1000.0,min_bins, max_bins)
    hist, bin_edges = np.histogram(im3/1000.0, bins=No_bins)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])    
    plt.plot(bin_centers, hist, lw=2,color='r')
    plt.axvline(np.nanmean(im3)/1000.0, color='r', linestyle='dashed', linewidth=2)
    
    plt.tick_params(axis='y', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.ylabel('$\#$ Occurrences')
    plt.xlabel('Fluo rate (kcps)')
       
    ax1.set_title(name1 + ' hist: means ' 
                      + str("{0:.0f}".format(np.nanmean(im1)/1000.0)) + r' $\pm$ ' +  str("{0:.0f}".format(np.nanstd(im1)/1000.0)) + ', ' 
                      + str("{0:.0f}".format(np.nanmean(im2)/1000.0)) + r' $\pm$ ' +  str("{0:.0f}".format(np.nanstd(im2)/1000.0)) + ', ' 
                      + str("{0:.0f}".format(np.nanmean(im3)/1000.0)) + r' $\pm$ ' +  str("{0:.0f}".format(np.nanstd(im3)/1000.0)))       
       
    ax1 = plt.subplot2grid((nrows, ncols), (6, 3), colspan=3,rowspan=2)    
       
    im1 = baseimage2[np.where((~np.isnan(baseimage2)) & (baseimage2 > 0))]
    No_bins = find_opt_no_hist_bins(im1/1000.0,min_bins, max_bins)
    hist, bin_edges = np.histogram(im1/1000.0, bins=No_bins)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])    
    plt.plot(bin_centers, hist, lw=2,color='k')
    plt.axvline(np.nanmean(im1)/1000.0, color='k', linestyle='dashed', linewidth=2)
    
    # Usually binary background
    im2 = background2[np.where((~np.isnan(background2)) & (background2 > 0))]
    No_bins = find_opt_no_hist_bins(im2/1000.0,min_bins, max_bins)
    hist, bin_edges = np.histogram(im2/1000.0, bins=No_bins)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])    
    plt.plot(bin_centers, hist, lw=2,color='b')
    plt.axvline(np.nanmean(im2)/1000.0, color='b', linestyle='dashed', linewidth=2)
    
    #Usually binary foreground
    im3 = foreground2[np.where((~np.isnan(foreground2)) & (foreground2 > 0))]
    No_bins = find_opt_no_hist_bins(im3/1000.0,min_bins, max_bins)
    hist, bin_edges = np.histogram(im3/1000.0, bins=No_bins)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])    
    plt.plot(bin_centers, hist, lw=2,color='r')
    plt.axvline(np.nanmean(im3)/1000.0, color='r', linestyle='dashed', linewidth=2)
    
    #Green line is background + 2 std of background
    #Rodrigo is detecting his single molecules at a mean twice sigma to the right of background
    plt.axvline((np.nanmean(im2)+ 2*np.nanstd(im2))/1000.0, color='g', linestyle='dashed', linewidth=2)

    plt.tick_params(axis='y', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.ylabel('$\#$ Occurrences')
    plt.xlabel('Fluo rate (kcps)')
    
    ax1.set_title(name2 + ' hist: means ' 
                      + str("{0:.0f}".format(np.nanmean(im1)/1000.0)) + r' $\pm$ ' +  str("{0:.0f}".format(np.nanstd(im1)/1000.0)) + ', ' 
                      + str("{0:.0f}".format(np.nanmean(im2)/1000.0)) + r' $\pm$ ' +  str("{0:.0f}".format(np.nanstd(im2)/1000.0)) + ', ' 
                      + str("{0:.0f}".format(np.nanmean(im3)/1000.0)) + r' $\pm$ ' +  str("{0:.0f}".format(np.nanstd(im3)/1000.0)))       
       
    
    ## correlation coefficients
    ax1 = plt.subplot2grid((nrows, ncols), (8, 0), colspan=6,rowspan=1)
   # ax1.set_title('Correlation coefficients')
    
    a = calc_pearson_spearman_manders(background1,background2)
    b = calc_pearson_spearman_manders(foreground1,background2)
    c = calc_pearson_spearman_manders(background1,foreground2)
    d = calc_pearson_spearman_manders(foreground1,foreground2)
   
    plt.axis('off') 
    ax1.text(0.5,0.4,'Correlation coefficients', ha='center', va='center', transform=ax1.transAxes)
    ax1.text(0.1, +0.2, name1 + ' ' + subsetA + ' + ' + name2 + ' ' + subsetA + ' pearson, spearman, m1, m2 = ' + str("{0:+.2f}".format(a[0][0])) + ', ' + str("{0:+.2f}".format(a[1][0])) + ', ' + str("{0:+.2f}".format(a[2])) + ', ' + str("{0:+.2f}".format(a[3])), ha='left', va='center', transform=ax1.transAxes)
    ax1.text(0.1, +0.0, name1 + ' ' + subsetB + ' + ' + name2 + ' ' + subsetA + ' pearson, spearman, m1, m2 = ' + str("{0:+.2f}".format(b[0][0])) + ', ' + str("{0:+.2f}".format(b[1][0])) + ', ' + str("{0:+.2f}".format(b[2])) + ', ' + str("{0:+.2f}".format(b[3])), ha='left', va='center', transform=ax1.transAxes)
    ax1.text(0.1, -0.2, name1 + ' ' + subsetA + ' + ' + name2 + ' ' + subsetB + ' pearson, spearman, m1, m2 = ' + str("{0:+.2f}".format(c[0][0])) + ', ' + str("{0:+.2f}".format(c[1][0])) + ', ' + str("{0:+.2f}".format(c[2])) + ', ' + str("{0:+.2f}".format(b[3])), ha='left', va='center', transform=ax1.transAxes)
    ax1.text(0.1, -0.4, name1 + ' ' + subsetB + ' + ' + name2 + ' ' + subsetB + ' pearson, spearman, m1, m2 = ' + str("{0:+.2f}".format(d[0][0])) + ', ' + str("{0:+.2f}".format(d[1][0])) + ', ' + str("{0:+.2f}".format(d[2])) + ', ' + str("{0:+.2f}".format(b[3])), ha='left', va='center', transform=ax1.transAxes)
       
    #plt.tight_layout()

def above_below_avg(image, image2):
    
    below = np.copy(image) 
    below[below > np.nanmean(image)] = np.nan

    above = np.copy(image)
    above[above < np.nanmean(image)] = np.nan
    
    below2 = np.copy(image2) 
    below2[below2 > np.nanmean(image2)] = np.nan

    above2 = np.copy(image2)
    above2[above2 < np.nanmean(image2)] = np.nan

    return below, above, below2, above2
    
def above_below_median(image, image2):
    
    below = np.copy(image) 
    below[below > np.nanmedian(image)] = np.nan

    above = np.copy(image)
    above[above < np.nanmedian(image)] = np.nan
    
    below2 = np.copy(image2) 
    below2[below2 > np.nanmedian(image2)] = np.nan

    above2 = np.copy(image2)
    above2[above2 < np.nanmedian(image2)] = np.nan

    return below, above, below2, above2
    
def arb_thr_one(imagemask, imagemasked, arb_thresh):
    
    binary_img = imagemask > (arb_thresh)*np.nanmax(imagemask)
    
    # Mask applied in both channels to regions having been recognized as red foreground
    brightone = (binary_img)*imagemask
    brightone[np.where(brightone == 0)] = np.nan

    darkone = (1-binary_img)*imagemask
    darkone[np.where(darkone == 0)] = np.nan
    
    brightother = (binary_img)*imagemasked * np.isfinite(imagemask)
    brightother[np.where(brightother == 0)] = np.nan
    
    darkother = (1-binary_img)*imagemasked * np.isfinite(imagemask)
    darkother[np.where(darkother == 0)] = np.nan

    return darkother, brightother, darkone, brightone
    
def gmmone(imagemask, imagemasked):
    
    img = np.copy(imagemask)
    classif = GMM(n_components=2, covariance_type='full')
    hlp = img.flatten()[np.where(np.isnan(img.flatten()) == False)]
    hlp = hlp.reshape((hlp.size,1))
    classif.fit(hlp)
    hlp2 = classif.means_
    threshold1 = np.mean(hlp2[0:2])
    binary_img = img > (threshold1)
    
    # Mask applied in both channels to regions having been recognized as red foreground
    brightone = (binary_img)*imagemask
    brightone[np.where(brightone == 0)] = np.nan

    darkone = (1-binary_img)*imagemask
    darkone[np.where(darkone == 0)] = np.nan
    
    # need to multiply with the binary image where the red channel is finite since the mask of the red is smaller than the mask of the blue
    
    brightother = (binary_img)*imagemasked * np.isfinite(imagemask)
    brightother[np.where(brightother == 0)] = np.nan
    
    darkother = (1-binary_img)*imagemasked * np.isfinite(imagemask)
    darkother[np.where(darkother == 0)] = np.nan

    return darkother, brightother, darkone, brightone
    
def gmmboth(image1, image2):
    # image 1 is usually red

    img = np.copy(image1)
    classif = GMM(n_components=2, covariance_type='full')
    hlp = img.flatten()[np.where(np.isnan(img.flatten()) == False)]
    hlp = hlp.reshape((hlp.size,1))
    classif.fit(hlp)
    hlp2 = classif.means_
    threshold1 = np.mean(hlp2[0:2])
    binary_img = img > (threshold1)
    
    # Mask applied in both channels to regions having been recognized as red foreground
    brightone = (binary_img)*image1
    brightone[np.where(brightone == 0)] = np.nan

    darkone = (1-binary_img)*image1
    darkone[np.where(darkone == 0)] = np.nan
    
    img = np.copy(image2)
    classif = GMM(n_components=2, covariance_type='full')
    hlp = img.flatten()[np.where(np.isnan(img.flatten()) == False)]
    hlp = hlp.reshape((hlp.size,1))
    classif.fit(hlp)
    hlp2 = classif.means_
    threshold1 = np.mean(hlp2[0:2])
    binary_img = img > (threshold1)
    
    # Mask applied in both channels to regions having been recognized as red foreground
    brightother = (binary_img)*image2
    brightother[np.where(brightother == 0)] = np.nan

    darkother = (1-binary_img)*image2
    darkother[np.where(darkother == 0)] = np.nan
 
    return darkother, brightother, darkone, brightone

def threshold_adaptive_dset(image1, image2, blocksize, myoffset):
    # image 1 is usually red
    
    #need grayscale picture
    hlp = np.copy(image1)
    hlp[np.isnan(hlp)] = 0
    image_gray = hlp/np.nanmax(image1) * 255.0 
            
    brightone = threshold_adaptive(image_gray, blocksize, offset = myoffset)*image1
    darkone = (1.0-threshold_adaptive(image_gray, blocksize, offset = myoffset))*image1
    
    hlp = np.copy(image2)
    hlp[np.isnan(hlp)] = 0
    image_gray = hlp/np.nanmax(image2) * 255.0 
    
    brightother = threshold_adaptive(image_gray, blocksize, offset = myoffset) *image2
    darkother = (1.0-threshold_adaptive(image_gray, blocksize, offset = myoffset))*image2

    return darkother, brightother, darkone, brightone
    
def random_walker_dset(image1, image2, cutofflow, cutoffhigh):
    # image 1 is usually red

    hlp = np.copy(image1)
    image_gray = hlp/np.nanmax(image1) * 255.0   
    data = image_gray
    data[np.isnan(image1)] = -5.0 #just a random number lower than 0
    markers = np.zeros(data.shape, dtype=np.uint)
    markers[data <= -4.9] = 3 #labels[0], outer frame
    markers[(data > -4.9) & (data < cutofflow * data[~np.isnan(image1)].mean())] = 1 #labels[1], dark
    markers[data > cutoffhigh * data[~np.isnan(image1)].mean()] = 2 #labels[2], bright
   
    labels = random_walker(data, markers)#, beta=10, mode='bf')
    
    brightone = np.copy(image1) 
    for x in range(brightone.shape[0]):
        for y in range(brightone.shape[1]):
            if labels[x,y] == 1 or labels[x,y] == 3:
                brightone[x,y] = np.nan   
    
    darkone = np.copy(image1) 
    for x in range(darkone.shape[0]):
        for y in range(darkone.shape[1]):
            if labels[x,y] == 2 or labels[x,y] == 3:
                darkone[x,y] = np.nan   
    
    hlp = np.copy(image2)
    image_gray = hlp/np.nanmax(image2) * 255.0   
    data = image_gray
    data[np.isnan(image2)] = -5.0 #just a random number lower than 0
    markers = np.zeros(data.shape, dtype=np.uint)
    markers[data <= -4.9] = 3 #labels[0]
    markers[(data > -4.9) & (data < cutofflow * data[~np.isnan(image2)].mean())] = 1 #labels[1]
    markers[data > cutoffhigh * data[~np.isnan(image2)].mean()] = 2 #labels[2]
   
    labels = random_walker(data, markers)#, beta=10, mode='bf')
    
    brightother = np.copy(image2) 
    for x in range(brightother.shape[0]):
        for y in range(brightother.shape[1]):
            if labels[x,y] == 1 or labels[x,y] == 3:
                brightother[x,y] = np.nan   
    
    darkother = np.copy(image2) 
    for x in range(darkother.shape[0]):
        for y in range(darkother.shape[1]):
            if labels[x,y] == 2 or labels[x,y] == 3:
                darkother[x,y] = np.nan   
    
    return darkother, brightother, darkone, brightone
    

def thr_otsu(image1, image2):
    # image 1 is usually red
    
    thresh = threshold_otsu(image1)
    binary_img = image1 > (thresh)
    
    brightone = (binary_img)*image1
    brightone[np.where(brightone == 0)] = np.nan    
    
    darkone = (1-binary_img)*image1
    darkone[np.where(darkone == 0)] = np.nan
    
    thresh = threshold_otsu(image2)
    binary_img = image2 > (thresh)
    
    brightother = (binary_img)*image2
    brightother[np.where(brightother == 0)] = np.nan    
    
    darkother = (1-binary_img)*image2
    darkother[np.where(darkother == 0)] = np.nan
    
    return darkother, brightother, darkone, brightone
    
def log_dog_doh(image1):
    
    hlp = np.copy(image1)
    image_gray = hlp/np.nanmax(image1) * 255.0   
    image_gray[np.isnan(image1)] = 0.0 #just a random number lower than 0
    
    blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
    print('here3') 

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)
    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    #plt.tight_layout()
    axes = axes.ravel()
    for blobs, color, title in sequence:
        print('here4') 
        ax = axes[0]
        axes = axes[1:]
        ax.set_title(title)
        ax.imshow(image_gray, cmap = cm.Greys, interpolation='nearest')
        ax.set_axis_off()
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax.add_patch(c)

def gmmone_tr_in_masked_channel(imagemask, imagemasked,imagemasked_is_4D=False):
    
    img = np.copy(imagemask)
    classif = GMM(n_components=2, covariance_type='diag')
    hlp = img.flatten()[np.where(np.isnan(img.flatten()) == False)]
    hlp = hlp.reshape((hlp.size,1))
    classif.fit(hlp)
    hlp2 = classif.means_
    threshold1 = np.mean(hlp2[0:2])
    binary_img = img > (threshold1)
    
    # Mask applied in both channels to regions having been recognized as red foreground
    brightone = (binary_img)*imagemask
    brightone[np.where(brightone == 0)] = np.nan

    darkone = (1-binary_img)*imagemask
    darkone[np.where(darkone == 0)] = np.nan
    
    # need to multiply with the binary image where the red channel is finite since the mask of the red is smaller than the mask of the blue
    
    if not imagemasked_is_4D:
    
        brightother = np.zeros([imagemasked.shape[0],imagemasked.shape[1],imagemasked.shape[2]])
        darkother = np.zeros([imagemasked.shape[0],imagemasked.shape[1],imagemasked.shape[2]])
        
        fac1 = (binary_img)*np.isfinite(imagemask)*1
        fac2 = (1-binary_img)*np.isfinite(imagemask)*1 
        
        #brightother =  imagemasked * fac1 
        #darkother =  imagemasked * fac2        
           
        gc.collect()
        return imagemasked * fac2 , imagemasked * fac1 , darkone, brightone, np.sum(fac1)/imagemasked.shape[1]/imagemasked.shape[2], np.sum(fac2)/imagemasked.shape[1]/imagemasked.shape[2], classif.means_, classif.covars_, classif.weights_ 

        return darkother, brightother, darkone, brightone, np.sum(fac1)/imagemasked.shape[1]/imagemasked.shape[2], np.sum(fac2)/imagemasked.shape[1]/imagemasked.shape[2], classif.means_, classif.covars_, classif.weights_ 
        #last factors are area of dark and bright regions, in percentage of whole picture area
        
    else:
        
        brightother = (np.zeros([imagemasked.shape[0],imagemasked.shape[1],imagemasked.shape[2],imagemasked.shape[3]]))
        darkother = (np.zeros([imagemasked.shape[0],imagemasked.shape[1],imagemasked.shape[2],imagemasked.shape[3]]))

        fac1 = (binary_img)* np.isfinite(imagemask)*1
        fac2 = (1-binary_img)* np.isfinite(imagemask)*1
        
        brightother =  imagemasked * fac1 
        darkother =  imagemasked * fac2    
        
        gc.collect()
        return darkother, brightother, darkone, brightone, np.sum(fac1)/imagemasked.shape[2]/imagemasked.shape[3], np.sum(fac2)/imagemasked.shape[2]/imagemasked.shape[3]
        
def gmmone_tr_in_masked_channel_modif_memory_issue(imagemask, imagemasked,imagemasked_is_4D=False):
    
    img = np.copy(imagemask)
    classif = GMM(n_components=2, covariance_type='diag')
    hlp = img.flatten()[np.where(np.isnan(img.flatten()) == False)]
    hlp = hlp.reshape((hlp.size,1))
    classif.fit(hlp)
    hlp2 = classif.means_
    threshold1 = np.mean(hlp2[0:2])
    binary_img = img > (threshold1)
    
    # Mask applied in both channels to regions having been recognized as red foreground
    brightone = (binary_img)*imagemask
    brightone[np.where(brightone == 0)] = np.nan

    darkone = (1-binary_img)*imagemask
    darkone[np.where(darkone == 0)] = np.nan
    
    # need to multiply with the binary image where the red channel is finite since the mask of the red is smaller than the mask of the blue
    
    if not imagemasked_is_4D:
    
        brightother = np.zeros([imagemasked.shape[0],imagemasked.shape[1],imagemasked.shape[2]])
        darkother = np.zeros([imagemasked.shape[0],imagemasked.shape[1],imagemasked.shape[2]])
        
        fac1 = (binary_img)*np.isfinite(imagemask)*1
        fac2 = (1-binary_img)*np.isfinite(imagemask)*1 
        
        #brightother =  imagemasked * fac1 
        #darkother =  imagemasked * fac2        
           
        gc.collect()
        return brightone, classif.means_, classif.covars_, classif.weights_ 

        #return darkother, brightother, darkone, brightone, np.sum(fac1)/imagemasked.shape[1]/imagemasked.shape[2], np.sum(fac2)/imagemasked.shape[1]/imagemasked.shape[2], classif.means_, classif.covars_, classif.weights_ 
        #last factors are area of dark and bright regions, in percentage of whole picture area
        
    else:
        
        brightother = (np.zeros([imagemasked.shape[0],imagemasked.shape[1],imagemasked.shape[2],imagemasked.shape[3]]))
        darkother = (np.zeros([imagemasked.shape[0],imagemasked.shape[1],imagemasked.shape[2],imagemasked.shape[3]]))

        fac1 = (binary_img)* np.isfinite(imagemask)*1
        fac2 = (1-binary_img)* np.isfinite(imagemask)*1
        
        brightother =  imagemasked * fac1 
        darkother =  imagemasked * fac2    
        
        gc.collect()
        return darkother, brightother, darkone, brightone, np.sum(fac1)/imagemasked.shape[2]/imagemasked.shape[3], np.sum(fac2)/imagemasked.shape[2]/imagemasked.shape[3]
        
def thr_otsu_tr_in_masked_channel(imagemask, imagemasked,imagemasked_is_4D=False):
    # image 1 is usually red
    
    thresh = threshold_otsu(imagemask)
    binary_img = imagemask > (thresh)
    
    brightone = (binary_img)*imagemask
    brightone[np.where(brightone == 0)] = np.nan    
    
    darkone = (1-binary_img)*imagemask
    darkone[np.where(darkone == 0)] = np.nan
    
    if not imagemasked_is_4D:
    
        brightother = np.zeros([imagemasked.shape[0],imagemasked.shape[1],imagemasked.shape[2]])
        darkother = np.zeros([imagemasked.shape[0],imagemasked.shape[1],imagemasked.shape[2]])
        
        fac1 = (binary_img)*np.isfinite(imagemask)*1
        fac2 = (1-binary_img)*np.isfinite(imagemask)*1 
        
        brightother =  imagemasked * fac1 
        darkother =  imagemasked * fac2        
           
        gc.collect()
        #return darkother, brightother, darkone, brightone, np.sum(fac1)/imagemasked.shape[1]/imagemasked.shape[2], np.sum(fac2)/imagemasked.shape[1]/imagemasked.shape[2]
        return brightother, darkother, darkone, brightone, np.sum(fac2)/imagemasked.shape[1]/imagemasked.shape[2], np.sum(fac1)/imagemasked.shape[1]/imagemasked.shape[2]
        #last factors are area of dark and bright regions, in percentage of whole picture area
        
    else:
        
        brightother = (np.zeros([imagemasked.shape[0],imagemasked.shape[1],imagemasked.shape[2],imagemasked.shape[3]]))
        darkother = (np.zeros([imagemasked.shape[0],imagemasked.shape[1],imagemasked.shape[2],imagemasked.shape[3]]))

        fac1 = (binary_img)* np.isfinite(imagemask)*1
        fac2 = (1-binary_img)* np.isfinite(imagemask)*1
        
        brightother =  imagemasked * fac1 
        darkother =  imagemasked * fac2    
        
        gc.collect()
        #return darkother, brightother, darkone, brightone, np.sum(fac1)/imagemasked.shape[2]/imagemasked.shape[3], np.sum(fac2)/imagemasked.shape[2]/imagemasked.shape[3]
        return brightother, darkother, darkone, brightone, np.sum(fac2)/imagemasked.shape[2]/imagemasked.shape[3], np.sum(fac1)/imagemasked.shape[2]/imagemasked.shape[3]
