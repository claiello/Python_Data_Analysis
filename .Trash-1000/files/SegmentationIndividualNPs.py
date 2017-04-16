# loading cv2, works only in python 2.7
# import sys
# sys.path.append('/usr/lib/python2.7/dist-packages')
# import cv2
# have fun. give boe hot dog.

import matplotlib
import matplotlib.pyplot as plt
import skimage
from skimage import exposure
import numpy as np

import pickle
import matplotlib.cm as cm
#from uncertainties import unumpy
from numpy import genfromtxt

import skimage.morphology
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
import sklearn
from skimage import filters
from skimage import segmentation

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.io import imread, imsave, imshow
from skimage.morphology import watershed

from skimage.filters import threshold_otsu, threshold_adaptive#, threshold_local

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from skimage.filters import roberts, sobel, scharr, prewitt

import scipy as sp
import scipy.misc
from skimage.feature import peak_local_max, canny
import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
import cv2
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral

def togs(image):
    
    return  (((image - image.min()) / (image.max() - image.min())) * 255.9).astype(np.uint8)

##################
do_150 = True
if do_150:
    #SEA= np.load('x150SEchannel.npz') #init shape (342, 315)
    SEA= np.load('test.npz') #init shape (342, 315)
    xlen = SEA['corr'].shape[0]
    ylen = SEA['corr'].shape[1]
    xinit = 42
    xfinal = -42
    yinit = 15
    yfinal = -15
    se = SEA['corr'][xinit:xfinal,yinit:yfinal]
    
    # fourier transform
    myfft = np.fft.fft2(se)
    mycut = 25#15 # less means more cut here, i.e. lower frequencies only
    myfft[0+mycut:192-mycut, 0:] = 0
    myfft[0:, 0+mycut:212-mycut] = 0
    
    new_pic = np.abs(np.fft.ifft2(myfft))
    setr = new_pic
    #binary threshold
    se_data2 = np.copy(setr)
    
    tv_filt = denoise_tv_chambolle(se_data2, weight=0.0025)
    grad_x, grad_y = np.gradient(tv_filt)
    new_pic_grad = np.sqrt(grad_x**2 + grad_y**2)
    new_pic_grad = scipy.ndimage.filters.gaussian_filter(new_pic_grad,1)
    
    ########
    I8 = (((new_pic_grad - new_pic_grad.min()) / (new_pic_grad.max() - new_pic_grad.min())) * 255.9).astype(np.uint8)
    bw = cv2.adaptiveThreshold(I8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)  #101,1
    
    from scipy import ndimage
    hlpse2 = bw    #ndimage.imread('bw_minimal.png', mode='L')
    #print(hlpse2.shape)  
    
    plt.figure()
    plt.subplot(3,3,1)
    plt.imshow(se,cmap=cm.Greys_r)
    plt.subplot(3,3,2)
    plt.imshow(setr,cmap=cm.Greys_r)
    plt.subplot(3,3,3)
    plt.imshow(new_pic_grad,cmap=cm.Greys_r)
    plt.subplot(3,3,4)
    plt.imshow(I8,cmap=cm.Greys_r)
    plt.subplot(3,3,5)
    plt.imshow(bw,cmap=cm.Greys_r)
    
    hlpse2 = togs(1.-hlpse2)
    distance = ndimage.distance_transform_edt(hlpse2)
    local_maxi = peak_local_max(
            distance, 
            num_peaks = 50, 
            indices = False, 
            footprint = np.ones((100,100)),
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
        
    areas = np.array([])
    for k in lab:
        areas = np.append(areas, len( labels_ws[labels_ws == k] ))
    cut_k = []
    cut_labels_ws = np.copy(labels_ws)
    for k in range(len(lab)):
        if (areas[k] < 1200) or (areas[k] > 4000):   #1200 works #4000 works
            cut_labels_ws[cut_labels_ws == lab[k]] = 0
            cut_k.append(k)
    plt.subplot(3,3,6)
    plt.imshow(cut_labels_ws,cmap=cm.Greys)
    plt.show()
    
do_300 = False
if do_300:
    SEA= np.load('x300SEchannel.npz') #init shape (342, 315)
    xlen = SEA['data'].shape[0]
    ylen = SEA['data'].shape[1]
    xinit = 27
    xfinal = -27
    yinit = 29
    yfinal = -29
    se = SEA['data'][xinit:xfinal,yinit:yfinal]
    print(se.shape)
    
    # fourier transform
    myfft = np.fft.fft2(se)
    mycut = 25#15 # less means more cut here, i.e. lower frequencies only
    myfft[0+mycut:192-mycut, 0:] = 0
    myfft[0:, 0+mycut:212-mycut] = 0
    
    new_pic = np.abs(np.fft.ifft2(myfft))
    setr = new_pic
    #binary threshold
    se_data2 = np.copy(setr)
    
    tv_filt = denoise_tv_chambolle(se_data2, weight=0.0025)
    grad_x, grad_y = np.gradient(tv_filt)
    new_pic_grad = np.sqrt(grad_x**2 + grad_y**2)
    new_pic_grad = scipy.ndimage.filters.gaussian_filter(new_pic_grad,1)
    
    ########
    I8 = (((new_pic_grad - new_pic_grad.min()) / (new_pic_grad.max() - new_pic_grad.min())) * 255.9).astype(np.uint8)
    bw = cv2.adaptiveThreshold(I8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)  #101,1
    
    import scipy.misc
    scipy.misc.imsave('bw300.png',new_pic_grad) 
    dfssd
    
    from scipy import ndimage
    hlpse2 = bw    #ndimage.imread('bw_minimal.png', mode='L')
    #print(hlpse2.shape)  
    
    plt.figure()
    plt.subplot(3,3,1)
    plt.imshow(se,cmap=cm.Greys_r)
    plt.subplot(3,3,2)
    plt.imshow(setr,cmap=cm.Greys_r)
    plt.subplot(3,3,3)
    plt.imshow(new_pic_grad,cmap=cm.Greys_r)
    plt.subplot(3,3,4)
    plt.imshow(I8,cmap=cm.Greys_r)
    plt.subplot(3,3,5)
    plt.imshow(bw,cmap=cm.Greys_r)
    
    hlpse2 = togs(1.-hlpse2)
    distance = ndimage.distance_transform_edt(hlpse2)
    local_maxi = peak_local_max(
            distance, 
            num_peaks = 15, 
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
        
    areas = np.array([])
    for k in lab:
        areas = np.append(areas, len( labels_ws[labels_ws == k] ))
    cut_k = []
    cut_labels_ws = np.copy(labels_ws)
    for k in range(len(lab)):
        if (areas[k] < 1200) or (areas[k] > 4000):   #1200 works #4000 works
            cut_labels_ws[cut_labels_ws == lab[k]] = 0
            cut_k.append(k)
    plt.subplot(3,3,6)
    plt.imshow(cut_labels_ws,cmap=cm.Greys)
    plt.show()
    
do_450 = False
if do_450:
    SEA= np.load('x450SEchannel.npz') #init shape (342, 315)
    xlen = SEA['data'].shape[0]
    ylen = SEA['data'].shape[1]
    xinit = 200
    xfinal = -200
    yinit = 41
    yfinal = -41
    se = SEA['data'][xinit:xfinal,yinit:yfinal]
    print(se.shape)
    
    # fourier transform
    myfft = np.fft.fft2(se)
    mycut = 25#15 # less means more cut here, i.e. lower frequencies only
    myfft[0+mycut:192-mycut, 0:] = 0
    myfft[0:, 0+mycut:212-mycut] = 0
    
    new_pic = np.abs(np.fft.ifft2(myfft))
    setr = new_pic
    #binary threshold
    se_data2 = np.copy(setr)
    
    tv_filt = denoise_tv_chambolle(se_data2, weight=0.0025)
    grad_x, grad_y = np.gradient(tv_filt)
    new_pic_grad = np.sqrt(grad_x**2 + grad_y**2)
    new_pic_grad = scipy.ndimage.filters.gaussian_filter(new_pic_grad,1)
    
    ########
    I8 = (((new_pic_grad - new_pic_grad.min()) / (new_pic_grad.max() - new_pic_grad.min())) * 255.9).astype(np.uint8)
    bw = cv2.adaptiveThreshold(I8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)  #101,1
    
    from scipy import ndimage
    hlpse2 = bw    #ndimage.imread('bw_minimal.png', mode='L')
    #print(hlpse2.shape)  
    
    plt.figure()
    plt.subplot(3,3,1)
    plt.imshow(se,cmap=cm.Greys_r)
    plt.subplot(3,3,2)
    plt.imshow(setr,cmap=cm.Greys_r)
    plt.subplot(3,3,3)
    plt.imshow(new_pic_grad,cmap=cm.Greys_r)
    plt.subplot(3,3,4)
    plt.imshow(I8,cmap=cm.Greys_r)
    plt.subplot(3,3,5)
    plt.imshow(bw,cmap=cm.Greys_r)
    
    hlpse2 = togs(1.-hlpse2)
    distance = ndimage.distance_transform_edt(hlpse2)
    local_maxi = peak_local_max(
            distance, 
            num_peaks = 15, 
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
        
    areas = np.array([])
    for k in lab:
        areas = np.append(areas, len( labels_ws[labels_ws == k] ))
    cut_k = []
    cut_labels_ws = np.copy(labels_ws)
    for k in range(len(lab)):
        if (areas[k] < 1200) or (areas[k] > 4000):   #1200 works #4000 works
            cut_labels_ws[cut_labels_ws == lab[k]] = 0
            cut_k.append(k)
    plt.subplot(3,3,6)
    plt.imshow(cut_labels_ws,cmap=cm.Greys)
    plt.show()
    
do_750 = False
if do_750:
    SEA= np.load('x750SEchannel.npz') #init shape (342, 315)
    xlen = SEA['data'].shape[0]
    ylen = SEA['data'].shape[1]
    xinit = 17
    xfinal = -17
    yinit = 18
    yfinal = -18
    se = SEA['data'][xinit:xfinal,yinit:yfinal]
    print(se.shape)
    
    # fourier transform
    myfft = np.fft.fft2(se)
    mycut = 25#15 # less means more cut here, i.e. lower frequencies only
    myfft[0+mycut:192-mycut, 0:] = 0
    myfft[0:, 0+mycut:212-mycut] = 0
    
    new_pic = np.abs(np.fft.ifft2(myfft))
    setr = new_pic
    #binary threshold
    se_data2 = np.copy(setr)
    
    tv_filt = denoise_tv_chambolle(se_data2, weight=0.0025)
    grad_x, grad_y = np.gradient(tv_filt)
    new_pic_grad = np.sqrt(grad_x**2 + grad_y**2)
    new_pic_grad = scipy.ndimage.filters.gaussian_filter(new_pic_grad,1)
    
    ########
    I8 = (((new_pic_grad - new_pic_grad.min()) / (new_pic_grad.max() - new_pic_grad.min())) * 255.9).astype(np.uint8)
    bw = cv2.adaptiveThreshold(I8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)  #101,1
    
    from scipy import ndimage
    hlpse2 = bw    #ndimage.imread('bw_minimal.png', mode='L')
    #print(hlpse2.shape)  
    
    plt.figure()
    plt.subplot(3,3,1)
    plt.imshow(se,cmap=cm.Greys_r)
    plt.subplot(3,3,2)
    plt.imshow(setr,cmap=cm.Greys_r)
    plt.subplot(3,3,3)
    plt.imshow(new_pic_grad,cmap=cm.Greys_r)
    plt.subplot(3,3,4)
    plt.imshow(I8,cmap=cm.Greys_r)
    plt.subplot(3,3,5)
    plt.imshow(bw,cmap=cm.Greys_r)
    
    hlpse2 = togs(1.-hlpse2)
    distance = ndimage.distance_transform_edt(hlpse2)
    local_maxi = peak_local_max(
            distance, 
            num_peaks = 15, 
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
        
    areas = np.array([])
    for k in lab:
        areas = np.append(areas, len( labels_ws[labels_ws == k] ))
    cut_k = []
    cut_labels_ws = np.copy(labels_ws)
    for k in range(len(lab)):
        if (areas[k] < 1200) or (areas[k] > 4000):   #1200 works #4000 works
            cut_labels_ws[cut_labels_ws == lab[k]] = 0
            cut_k.append(k)
    plt.subplot(3,3,6)
    plt.imshow(cut_labels_ws,cmap=cm.Greys)
    plt.show()