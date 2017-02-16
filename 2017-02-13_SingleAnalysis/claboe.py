# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 21:37:32 2017

@author: clarice
"""
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

SEA= np.load('V0SEchannelB.npz')
xlen = SEA['data'].shape[0]
ylen = SEA['data'].shape[1]
delx = 0#+28
dely = 0 #+26
xval = 96
yval = 106
cutx = 0 #32
cutxtop = 0 #10
se = SEA['data'][np.floor(xlen/2.0)-xval+delx+cutxtop:np.floor(xlen/2.0)+xval+delx-cutx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely]
   
# fourier transform

myfft = np.fft.fft2(se)
mycut = 15 # less means more cut here, i.e. lower frequencies only
myfft[0+mycut:192-mycut, 0:] = 0
myfft[0:, 0+mycut:212-mycut] = 0

new_pic = np.abs(np.fft.ifft2(myfft))
se = new_pic
#binary threshold
se_data2 = np.copy(se)

tv_filt = denoise_tv_chambolle(se_data2, weight=0.0025)
grad_x, grad_y = np.gradient(tv_filt)
new_pic_grad = np.sqrt(grad_x**2 + grad_y**2)
new_pic_grad = scipy.ndimage.filters.gaussian_filter(new_pic_grad,1)

########
I8 = (((new_pic_grad - new_pic_grad.min()) / (new_pic_grad.max() - new_pic_grad.min())) * 255.9).astype(np.uint8)
bw = cv2.adaptiveThreshold(I8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)  #101,1

from scipy import ndimage
hlpse2 = ndimage.imread('bw_minimal.png', mode='L')
print(hlpse2.shape)  

plt.figure()
plt.subplot(3,3,1)
plt.imshow(se,cmap=cm.Greys_r)
plt.subplot(3,3,2)
plt.imshow(new_pic_grad,cmap=cm.Greys_r)
plt.subplot(3,3,3)
plt.imshow(bw,cmap=cm.Greys_r)
plt.subplot(3,3,4)
#plt.imshow(im_floodfill_inv,cmap=cm.Greys_r)
plt.subplot(3,3,5)
#plt.imshow(im_out,cmap=cm.Greys_r)
plt.subplot(3,3,6)
#plt.imshow(im_out2,cmap=cm.Greys_r)
plt.subplot(3,3,7)
#plt.imshow(im_floodfill_inv2,cmap=cm.Greys_r)
plt.subplot(3,3,7)
plt.imshow(hlpse2,cmap=cm.Greys_r)

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



lklklk
SEA= np.load('V0SEchannelB.npz')
xlen = SEA['data'].shape[0]
ylen = SEA['data'].shape[1]
delx = 0#+28
dely = 0 #+26
xval = 96
yval = 106
cutx = 0 #32
cutxtop = 0 #10
se = SEA['data'][np.floor(xlen/2.0)-xval+delx+cutxtop:np.floor(xlen/2.0)+xval+delx-cutx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely]
   
# fourier transform

myfft = np.fft.fft2(se)
mycut = 15 # less means more cut here, i.e. lower frequencies only
myfft[0+mycut:192-mycut, 0:] = 0
myfft[0:, 0+mycut:212-mycut] = 0

new_pic = np.abs(np.fft.ifft2(myfft))

se = new_pic

#binary threshold
se_data2 = np.copy(se)

tv_filt = denoise_tv_chambolle(se_data2, weight=0.0025)
grad_x, grad_y = np.gradient(tv_filt)
new_pic_grad = np.sqrt(grad_x**2 + grad_y**2)
new_pic_grad = scipy.ndimage.filters.gaussian_filter(new_pic_grad,1)

########
I8 = (((new_pic_grad - new_pic_grad.min()) / (new_pic_grad.max() - new_pic_grad.min())) * 255.9).astype(np.uint8)
bw = cv2.adaptiveThreshold(I8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)  #101,1
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)) #17,17
#bw = cv2.dilate(cv2.erode(bw, kernel), kernel)
#bw = ndi.morphology.binary_fill_holes(bw,structure=np.ones((5,5)))
######atry to floodfill bw from line86
# Copy the thresholded image.
im_floodfill = bw.copy()
# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = bw.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 255);
# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
# Combine the two images to get the foreground.
im_out = bw | im_floodfill_inv

#fig = plt.figure(frameon=False)
#fig.set_size_inches(212/30.,192/30.)
#ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.set_axis_off()
#fig.add_axes(ax)
#ax.imshow(bw, cmap=cm.Greys_r,aspect=1)
#plt.savefig('bw.png', dpi = 30, pad_inches=0)#, bbox_inches='tight', pad_inches=0)    
#lklk


########
#ret,thresh = cv2.threshold(new_pic_grad2,45,255,0) #127,255,0
#thresh =  im_floodfill_inv.copy()  #bw.copy()
#contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#size = bw.shape
#m = np.zeros(size, dtype=np.uint8)
#plt.figure()
#for i, cnt in enumerate(contours):
#    if cv2.contourArea(cnt) >= 1:
#        color = (255,255,255)
#        cv2.drawContours(m, cnt, -1, color, -1)
#cv2.imshow("Contour",m)
#plt.show()
########
#Try to fill in contours
# Copy the thresholded image.
#image = m
#im_floodfill2 = image.copy()
## Mask used to flood filling.
## Notice the size needs to be 2 pixels than the image.
#h, w = image.shape[:2]
#mask = np.zeros((h+2, w+2), np.uint8)
## Floodfill from point (0, 0)
#cv2.floodFill(im_floodfill2, mask, (0,0), 255);
## Invert floodfilled image
#im_floodfill_inv2 = cv2.bitwise_not(im_floodfill2)
## Combine the two images to get the foreground.
#im_out2 = image | im_floodfill_inv2
#######
#new_pic_grad2 = np.copy(new_pic_grad)
## Set up the detector with default parameters.
#detector = cv2.SimpleBlobDetector()
## Detect blobs.
#keypoints = detector.detect(new_pic_grad2)
## Draw detected blobs as red circles.
## cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
#im_with_keypoints = cv2.drawKeypoints(new_pic_grad2, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
## Show keypoints
#cv2.imshow("Keypoints", im_with_keypoints)
#cv2.waitKey(0)


#######
#from scipy import ndimage
#hlpse2 = ndimage.imread('im_floodfill_inv_done.png', mode='L')
#print(hlpse2.shape)  

#from scipy import ndimage
#hlpse2 = ndimage.imread('bw_done.png', mode='L')
#print(hlpse2.shape)  

from scipy import ndimage
hlpse2 = ndimage.imread('bw_minimal.png', mode='L')
print(hlpse2.shape)  

plt.figure()
plt.subplot(3,3,1)
plt.imshow(se,cmap=cm.Greys_r)
plt.subplot(3,3,2)
plt.imshow(new_pic_grad,cmap=cm.Greys_r)
plt.subplot(3,3,3)
plt.imshow(bw,cmap=cm.Greys_r)
plt.subplot(3,3,4)
plt.imshow(im_floodfill_inv,cmap=cm.Greys_r)
plt.subplot(3,3,5)
plt.imshow(im_out,cmap=cm.Greys_r)
plt.subplot(3,3,6)
#plt.imshow(im_out2,cmap=cm.Greys_r)
plt.subplot(3,3,7)
#plt.imshow(im_floodfill_inv2,cmap=cm.Greys_r)
plt.subplot(3,3,7)
plt.imshow(hlpse2,cmap=cm.Greys_r)



#hlpse2 = np.copy(im_floodfill_inv)
####### image to go into algo is hlpse2
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
    if (areas[k] < 1200) or (areas[k] > 4000):
        cut_labels_ws[cut_labels_ws == lab[k]] = 0
        cut_k.append(k)
plt.subplot(3,3,6)
plt.imshow(cut_labels_ws,cmap=cm.bone)




plt.show()




#
#from scipy import ndimage
#se = ndimage.imread('se_line.png', mode='L')
#print(se.shape)
#    
#se = np.array(se, dtype = np.float32)
#se_data2 = np.copy(se)
#se_data2[se < 1e-12] = 0.0
#
#se_data21 = scipy.ndimage.filters.gaussian_filter(se_data2,2)    
##se_data22 = dilation(se_data21,disk(3))
##se_data3 = scipy.ndimage.filters.median_filter(se_data2,3) bad! sad!      
#hlpse2 = se_data21
#
#se_data22= canny(se_data21, sigma=3) #, low_threshold =2,high_threshold =200) #, low_threshold=10, high_threshold=50)
##hlpse2 = closing(hlpse2, disk(3))
#v = np.median(se_data21)
## apply automatic Canny edge detection using the computed median
#sigma=3
#lower = np.floor(max(0, (1.0 - sigma) * v))
#upper = np.floor(min(255, (1.0 + sigma) * v))
#
#
#    # Find centers of NPs
#distance = ndimage.distance_transform_edt(hlpse2)
#local_maxi = peak_local_max(
#            distance, 
#            indices = False,
#            num_peaks = 10,  
#            footprint = np.ones((50,50)),
#            labels = hlpse2) #footprint = min dist between maxima to find #footprint was 25,25
#            
#markers = skimage.morphology.label(local_maxi)
#labels_ws = watershed(-distance, markers, mask=hlpse2)
#lab = np.unique(labels_ws)
#
#plt.figure()
#plt.subplot(2,2,1)
#plt.imshow(se)
#plt.colorbar()
#
#plt.subplot(2,2,2)
#plt.imshow(labels_ws)
#
#plt.subplot(2,2,3)
#plt.imshow(se_data21)
#
#plt.subplot(2,2,4)
#plt.imshow(se_data22)
#
klklklklk
#if True:
#  SEA= np.load('V0SEchannelB.npz')
#    
#    xlen = SEA['data'].shape[0]
#    ylen = SEA['data'].shape[1]
#    delx = 0#+28
#    dely = 0 #+26
#    xval = 96
#    yval = 106
#    cutx = 0 #32
#    cutxtop = 0 #10
#    se = SEA['data'][np.floor(xlen/2.0)-xval+delx+cutxtop:np.floor(xlen/2.0)+xval+delx-cutx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely]
#   
#
#
#    # fourier transform
#
#    myfft = np.fft.fft2(se)
#    mycut = 15 # less means more cut here, i.e. lower frequencies only
#    myfft[0+mycut:192-mycut, 0:] = 0
#    myfft[0:, 0+mycut:212-mycut] = 0
#    
#    new_pic = np.abs(np.fft.ifft2(myfft))
#    
#    se = new_pic
#    
##    plt.subplot(2,2,1)
##    plt.imshow(np.abs(se))
##    plt.subplot(2,2,2)
##    plt.imshow(new_pic)
##    plt.show()
##    
##    asdsad
##    
#   
#   
#   
##    fig = plt.figure(frameon=False)
##    fig.set_size_inches(212/30.,192/30.)
##    ax = plt.Axes(fig, [0., 0., 1., 1.])
##    ax.set_axis_off()
##    fig.add_axes(ax)
##    ax.imshow(se, aspect=1)
##    plt.savefig('se.png', dpi = 30, pad_inches=0)#, bbox_inches='tight', pad_inches=0)    
##    lklk
#   
##    from scipy import ndimage
##    se = ndimage.imread('se_line.png', mode='L')
##    print(se.shape)
#    
##    import Image
##    img = Image.open('ping.png').convert('L')
##    se = np.asarray(img)
##    plt.figure()
##    plt.imshow(se)
##    plt.show()
##    lklk
#    
#    ax001.imshow(se,cmap=cm.Greys_r)
#    ax001.axis('off')
#    Pixel_size = np.array([0.89]) 
#    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[0]))
#    sbar = sb.AnchoredScaleBar(ax001.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4, my_fontsize = fsizenb)
#    ax001.add_artist(sbar)
#    
#    #binary threshold
#    se_data2 = np.copy(se)
#    #se_data2 = scipy.ndimage.filters.gaussian_filter(se_data2,0.5)
#    #hh, hlpse2, hhh,hhhh = gmmone(se_data2,se_data2)
#    #hlpse2[np.where(~np.isnan(hlpse2))] = 1.0
#    #hlpse2[np.where(np.isnan(hlpse2))] = 0.0
#    #hlpse2[hlpse2 < -0.35] = np.nan
##    hlpse2 = se_data2 > filters.threshold_otsu(se_data2)
##    hlpse2[hlpse2 == True] = 1.0
##    hlpse2[hlpse2 == False] = 0.0
#    
#    #grad_x, grad_y = np.gradient(se_data2)
#    #grad_se = np.sqrt(grad_x**2 + grad_y**2)
#    
#    #grad_all = np.copy(se_data2)
#    #grad_all = scipy.ndimage.filters.gaussian_filter(grad_all,1)
#    #grad_all = scipy.ndimage.filters.median_filter(grad_all,9)
#    
##    grad_x, grad_y = np.gradient(grad_all)
##    grad_se = np.sqrt(grad_x**2 + grad_y**2)
#    
#    #grad_x[se_data2 > -0.343] = 1.0
#    #grad_x[se_data2 <= -0.343] = 0.0
#    
#    
#    #se_data2 = scipy.ndimage.filters.median_filter(se_data2,3)
##    #se_data2 = scipy.ndimage.filters.gaussian_filter(se_data2,0.5)
##    block_size = 29
##    hlpse2 = threshold_adaptive(se_data2, block_size)
#    
#    from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
#    tv_filt = denoise_tv_chambolle(se_data2, weight=0.0025)
#    tv_bi = denoise_bilateral(np.abs(se_data2), sigma_range = 0.05, sigma_spatial = 15, multichannel = False)
#    
#    ########
#    from skimage.feature import peak_local_max, canny
#    #hlpse2 = canny(se_data2, sigma=3, low_threshold = -0.35,high_threshold = -0.345) #, low_threshold=10, high_threshold=50)
#    sigmaa=3
#    v = np.median(se_data2)
#    lower = np.floor(max(np.min(se_data2), (1.0 - sigmaa) * v))
#    upper = np.floor(min(np.max(se_data2), (1.0 + sigmaa) * v))
#    hlpse2 = canny(se_data2, sigma=sigmaa, low_threshold = lower,high_threshold =upper)
#    hlpse2 = closing(hlpse2, disk(4)) #3 works
#    #######
#
#    closed = dilation(se_data2, disk(3))    
#    tv_closed = denoise_tv_chambolle(closed, weight=0.0025)
#    #new_pic = np.copy(closed)
#    #new_pic[closed > -0.346] = 1.0
#    #new_pic[closed <= -0.346] = 0.0        
#    grad_x, grad_y = np.gradient(tv_filt)
#    new_pic_grad = np.sqrt(grad_x**2 + grad_y**2)
#    new_pic_grad = scipy.ndimage.filters.gaussian_filter(new_pic_grad,1)
#    
#    sigmaa=3
#    v = np.median(new_pic_grad)
#    lower = np.floor(max(np.min(new_pic_grad), (1.0 - sigmaa) * v))
#    upper = np.floor(min(np.max(new_pic_grad), (1.0 + sigmaa) * v))
#    nc = canny(new_pic_grad,sigma=sigmaa,low_threshold =lower,high_threshold =upper)
# 
#    new_pic = np.copy(new_pic_grad)   
#    grad_value = 0.00045
#    new_pic[new_pic_grad > grad_value] = 1.0
#    new_pic[new_pic_grad <= grad_value] = 0.0 
#    
##    block_size = 29
#    #hlpse2 = closed > filters.threshold_otsu(closed)
#    #hlpse2[hlpse2 == True] = 1.0
#    #hlpse2[hlpse2 == False] = 0.0
#    
##    edge_roberts = roberts(closed)
##    edge_sobel = sobel(closed)
#    
#   # plt.figure()
##    plt.subplot(3,3,1)
##    plt.imshow(se)
##    plt.colorbar()
##    plt.subplot(3,3,2)
##    plt.imshow(closed)
##    plt.colorbar()  
##    plt.subplot(3,3,3)
#  #  plt.imshow(new_pic_grad,cmap=cm.Greys_r)
#   # plt.show()
#   # klklk
#    
##    fig = plt.figure(frameon=False)
##    fig.set_size_inches(212/30.,192/30.)
##    ax = plt.Axes(fig, [0., 0., 1., 1.])
##    ax.set_axis_off()
##    fig.add_axes(ax)
##    ax.imshow(new_pic_grad, aspect=1)
##    plt.savefig('edgesB.png', dpi = 30, pad_inches=0)#, bbox_inches='tight', pad_inches=0)    
##    lklk
#    
##    plt.colorbar()  
##    plt.subplot(3,3,4)
##    plt.imshow(hlpse2)
##    plt.colorbar()  
##    plt.subplot(3,3,5)
##    plt.imshow(tv_filt)
##    plt.colorbar()  
##    plt.subplot(3,3,6)
##    plt.imshow(tv_closed)
##    plt.colorbar()  
##    plt.subplot(3,3,7)
##    plt.imshow(new_pic)
##    plt.colorbar()  
##    plt.subplot(3,3,8)
##    plt.imshow(nc)
##    plt.colorbar()  
##    plt.show()
##    
#    #klklk
#    #hlpse2 = se
#    from scipy import ndimage
#    hlpse2 = ndimage.imread('im_floodfill_inv_done.png', mode='L')
#    print(hlpse2.shape)    
#    
#    # Find centers of NPs
#    distance = ndimage.distance_transform_edt(hlpse2)
#    local_maxi = peak_local_max(
#            distance, 
#            num_peaks = 15, 
#            indices = False, 
#            footprint = np.ones((50,50)),
#            labels = hlpse2) #footprint = min dist between maxima to find #footprint was 25,25
#    markers = skimage.morphology.label(local_maxi)
#    labels_ws = watershed(-distance, markers, mask=hlpse2)
#    lab = np.unique(labels_ws)
#    
#    # Make random colors, not degrade
#    rand_ind = np.random.permutation(lab)
#    new_labels_ws = np.copy(labels_ws)
#    for k in range(new_labels_ws.shape[0]):
#        for j in range(new_labels_ws.shape[1]):
#            new_labels_ws[k, j] = rand_ind[labels_ws[k, j]]
#    labels_ws =  new_labels_ws
#    
#   
#    areas = np.array([])
#    for k in lab:
#        areas = np.append(areas, len( labels_ws[labels_ws == k] ))
#    cut_k = []
#    cut_labels_ws = np.copy(labels_ws)
#    for k in range(len(lab)):
#        if (areas[k] < 100) or (areas[k] > 4000):
#            cut_labels_ws[cut_labels_ws == lab[k]] = 0
#            cut_k.append(k)
#    ax011.imshow(cut_labels_ws,cmap=cm.rainbow) #or 'OrRd'
#    ax011.axis('off')
