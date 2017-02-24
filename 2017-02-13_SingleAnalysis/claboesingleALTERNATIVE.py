

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
    
def tor(area):
    
    return 2*np.sqrt(2.*area/(3.*np.sqrt(3)))

SEA= np.load('singleSEchannelC.npz')
se = SEA['data'][40:190,120:270]   


se_data2 = np.copy(se)
hlpse2 = se_data2 > filters.threshold_otsu(se_data2)
hlpse2[hlpse2 == True] = 1.0
hlpse2[hlpse2 == False] = 0.0

# Find centers of NPs
distance = ndimage.distance_transform_edt(hlpse2)
local_maxi = peak_local_max(
        distance, 
        num_peaks = 1, 
        indices = False, 
        footprint = np.ones((50,50)),
        labels = hlpse2) #footprint = min dist between maxima to find #footprint was 25,25
markers = skimage.morphology.label(local_maxi)
labels_ws = watershed(-distance, markers, mask=hlpse2)
lab = np.unique(labels_ws)

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

print(tor(areas))
klklk

import matplotlib.patches as patches
#
#
#
#########
#I8 = (((new_pic_grad - new_pic_grad.min()) / (new_pic_grad.max() - new_pic_grad.min())) * 255.9).astype(np.uint8)
#bw = cv2.adaptiveThreshold(I8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)  #101,1
#
#from scipy import ndimage
#hlpse2 = bw #ndimage.imread('bw_minimal.png', mode='L')
#print(hlpse2.shape)  

hex_angle = 20

plt.figure()
plt.subplot(3,3,1)
plt.imshow(se,cmap=cm.Greys_r)
ax = plt.gca()
ax.add_patch(patches.RegularPolygon((80, 70), 6, 50, fill = False, color = 'r', orientation = hex_angle * np.pi/180.0))

plt.subplot(3,3,2)
plt.imshow(new_pic_grad,cmap=cm.Greys_r)

ax = plt.gca()
ax.add_patch(patches.RegularPolygon((80, 70), 6, 50, fill = False, color = 'r', orientation = hex_angle * np.pi/180.0))

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
        num_peaks = 1, 
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

print(tor(areas))


plt.show()
