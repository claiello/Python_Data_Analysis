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
from skimage.filters import threshold_otsu, threshold_adaptive, rank#, threshold_local
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

SEA= np.load('N31pt2SEchannelA.npz')
xlen = SEA['data'].shape[0]
ylen = SEA['data'].shape[1]
delx = 0#+28
dely = 0#00
xval = 144
yval = 120
cutx = 0 #32
cutxtop = 0 #10
se = SEA['data'][np.floor(xlen/2.0)-xval+delx+cutxtop:np.floor(xlen/2.0)+xval+delx-cutx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely]
   
se_data2 = np.copy(se)
tv_filt = denoise_tv_chambolle(se_data2, weight=0.0025)
grad_x, grad_y = np.gradient(tv_filt)
new_pic_grad = np.sqrt(grad_x**2 + grad_y**2)


#fig = plt.figure(frameon=False)
#fig.set_size_inches(bw.shape[1]/30.,bw.shape[0]/30.)
#ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.set_axis_off()
#fig.add_axes(ax)
#ax.imshow(new_pic_grad, cmap=cm.Greys_r,aspect=1)
#plt.savefig('new_pic_grad.png', dpi = 30, pad_inches=0)#, bbox_inches='tight', pad_inches=0)    
#lklk

#from scipy import ndimage
#hlpse2 = ndimage.imread('bw_largearea.png', mode='L')
#print(hlpse2.shape)  


#plt.figure()
#plt.subplot(1,3,1)
#plt.imshow(se, cmap = cm.Greys_r)
#plt.subplot(1,3,2)
#plt.imshow(new_pic_grad, cmap = cm.Greys_r)
hlpse2 = np.copy(new_pic_grad)
local_otsu = rank.otsu(togs(hlpse2), disk(6))
thres = threshold_otsu(togs(hlpse2))
masklarge = togs(hlpse2) >= local_otsu
#plt.subplot(1,3,3)
#plt.imshow(masklarge, cmap = cm.Greys_r)
#plt.show()

#fig = plt.figure(frameon=False)
#fig.set_size_inches(masklarge.shape[1]/30.,masklarge.shape[0]/30.)
#ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.set_axis_off()
#fig.add_axes(ax)
#ax.imshow(masklarge, cmap=cm.Greys_r,aspect=1)
#plt.savefig('masklarge.png', dpi = 30, pad_inches=0)#, bbox_inches='tight', pad_inches=0) 

from scipy import ndimage
hlpse2 = ndimage.imread('masklarge_done.png', mode='L')

hlpse2 =togs(1.-hlpse2)
distance = ndimage.distance_transform_edt(hlpse2)
local_maxi = peak_local_max(
        distance, 
        num_peaks = 10, 
        indices = False, 
        footprint = np.ones((10,10)),
        labels = hlpse2) #footprint = min dist between maxima to find #footprint was 25,25
markers = skimage.morphology.label(local_maxi)
labels_ws = watershed(-distance, markers, mask=hlpse2)
lab = np.unique(labels_ws)

#Make random colors, not degrade
#rand_ind = np.random.permutation(lab)
#new_labels_ws = np.copy(labels_ws)
#for k in range(new_labels_ws.shape[0]):
#    for j in range(new_labels_ws.shape[1]):
#        new_labels_ws[k, j] = rand_ind[labels_ws[k, j]]
#labels_ws =  new_labels_ws
    
areas = np.array([])
for k in lab:
    areas = np.append(areas, len( labels_ws[labels_ws == k] ))
cut_k = []
cut_labels_ws = np.copy(labels_ws)
non_cut_k = []  ###### change cut_k
for k in range(len(lab)):
    if (areas[k] < 10) or (areas[k] > 20000):   #1200 works #4000 works
        cut_labels_ws[cut_labels_ws == lab[k]] = 0
        cut_k.append(k)
    else:
        non_cut_k.append(k)  ###change cut_k   
        
print(len(lab))
print(len(non_cut_k))        
        
plt.subplot(2,1,1)
plt.imshow(hlpse2, cmap = cm.Greys_r)
plt.subplot(2,1,2)
plt.imshow(cut_labels_ws,cmap=cm.rainbow)

plt.show()


klklkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk

SEA= np.load('N31pt2SEchannelA.npz')
xlen = SEA['data'].shape[0]
ylen = SEA['data'].shape[1]
delx = 0#+28
dely = 0#00
xval = 144
yval = 120
cutx = 0 #32
cutxtop = 0 #10
se = SEA['data'][np.floor(xlen/2.0)-xval+delx+cutxtop:np.floor(xlen/2.0)+xval+delx-cutx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely]
   
se_data2 = np.copy(se)

#plt.imshow(se,cmap=cm.Greys_r)
#plt.show()
#lklklk

tv_filt = denoise_tv_chambolle(se_data2, weight=0.0025)
grad_x, grad_y = np.gradient(tv_filt)
new_pic_grad = np.sqrt(grad_x**2 + grad_y**2)
new_pic_grad = scipy.ndimage.filters.gaussian_filter(new_pic_grad,1)

########
#npg = np.copy(new_pic_grad)
#I8 = togs(  npg ) #((npg - npg.min()) / (npg.max() - npg.min())) * 255.9).astype(np.uint8)
#bw = cv2.adaptiveThreshold(I8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)  #for mediuum zoom, param is 7


#fig = plt.figure(frameon=False)
#fig.set_size_inches(bw.shape[1]/30.,bw.shape[0]/30.)
#ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.set_axis_off()
#fig.add_axes(ax)
#ax.imshow(new_pic_grad, cmap=cm.Greys_r,aspect=1)
#plt.savefig('new_pic_grad.png', dpi = 30, pad_inches=0)#, bbox_inches='tight', pad_inches=0)    
#lklk

#from scipy import ndimage
#hlpse2 = ndimage.imread('bw_largearea.png', mode='L')
#print(hlpse2.shape)  

plt.figure()
plt.subplot(3,3,1)
plt.imshow(se,cmap=cm.Greys_r)
plt.colorbar()

plt.subplot(3,3,2)
plt.imshow(new_pic_grad,cmap=cm.Greys_r)
plt.subplot(3,3,3)
#plt.imshow(bw,cmap=cm.Greys_r)
plt.subplot(3,3,4)
#plt.imshow(im_floodfill_inv,cmap=cm.Greys_r)
plt.subplot(3,3,5)
#plt.imshow(im_out,cmap=cm.Greys_r)
#plt.subplot(3,3,6)
#plt.imshow(im_out2,cmap=cm.Greys_r)
#plt.subplot(3,3,7)
#plt.imshow(im_floodfill_inv2,cmap=cm.Greys_r)
#plt.subplot(3,3,7)
#plt.imshow(hlpse2,cmap=cm.Greys_r)




new_se = np.copy(se)
#new_se[se < 0.0825] = 0.0

#new_se = 0.5 * new_se/np.max(new_se) * np.max(new_pic_grad)

new_se = 0.0 * new_se/np.max(new_se) * np.max(new_pic_grad)

#plt.figure()
#
#plt.subplot(2,2,1)
#plt.imshow(se, cmap = cm.Greys_r)
#plt.colorbar()
#
#plt.subplot(2,2,2)
#plt.imshow(new_pic_grad, cmap = cm.Greys_r)
#plt.colorbar()
#
#plt.subplot(2,2,3)
#plt.imshow(new_se, cmap = cm.Greys_r)
#plt.colorbar()
#
#plt.subplot(2,2,4)
#plt.imshow(togs(new_se + new_pic_grad), cmap = cm.Greys_r)
#plt.colorbar()
#
#plt.show()


#hlpse2 = ((new_se + new_pic_grad))
hlpse2 = new_pic_grad
#hlpse2 = ((new_se))

#hlpse2 = new_se

#imgOrig = hlpse2
#img = cv2.imread(imgOrig+".jpg");  
plt.figure()
plt.subplot(2,1,1)
plt.imshow(togs(hlpse2), cmap = cm.Greys_r)
plt.colorbar()


local_otsu = rank.otsu(togs(hlpse2), disk(6))
thres = threshold_otsu(togs(hlpse2))


plt.subplot(2,1,2)
plt.imshow(togs(hlpse2) >= local_otsu, cmap = cm.Greys_r)
plt.show()
asdasd



imgray = togs(hlpse2)

lap = cv2.Laplacian(imgray, cv2.IPL_DEPTH_32F)#, ksize = 3)  

#lap = cv2.Laplacian(img, cv2.IPL_DEPTH_32F, ksize = 3)  
#
#imgray = cv2.cvtColor(lap,cv2.COLOR_BGR2GRAY)  
#print lap
plt.imshow(lap, cmap = cm.Greys_r)
plt.colorbar()
#plt.show()
#asd

ret,thresh = cv2.threshold(imgray,50,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
size = imgray.shape
m = np.zeros(size, dtype=np.uint8)
for i, cnt in enumerate(contours):
    if cv2.contourArea(cnt) >= 1:
        color = (255,255,255)
        cv2.drawContours(m, cnt, -1, color, -1)
#cv2.imwrite(str(imgOrig)+"contours.jpg", m);


#cv2.imshow('image',m)

plt.figure()
plt.imshow(m, cmap = cm.Greys_r)

plt.show()

asd


#hlpse2 = togs(1.0 - new_pic_grad)
#hlpse2 = togs(1.0 - hlpse2) #1.0-hlpse2) #was 1-
distance = ndimage.distance_transform_edt(hlpse2)
local_maxi = peak_local_max(
        distance, 
        num_peaks = 30, 
        indices = False, 
        footprint = np.ones((10,10)),
        labels = hlpse2) #footprint = min dist between maxima to find #footprint was 25,25
markers = skimage.morphology.label(local_maxi)
labels_ws = watershed(-distance, markers, mask=hlpse2)
lab = np.unique(labels_ws)

asdasd

# Make random colors, not degrade
#rand_ind = np.random.permutation(lab)
#new_labels_ws = np.copy(labels_ws)
#for k in range(new_labels_ws.shape[0]):
#    for j in range(new_labels_ws.shape[1]):
#        new_labels_ws[k, j] = rand_ind[labels_ws[k, j]]
#labels_ws =  new_labels_ws
    
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
plt.imshow(cut_labels_ws,cmap=cm.rainbow)




plt.show()

#binary threshold
se_data2 = np.copy(se)
hlpse2 = se_data2 > filters.threshold_otsu(se_data2)
hlpse2[hlpse2 == True] = 1.0
hlpse2[hlpse2 == False] = 0.0