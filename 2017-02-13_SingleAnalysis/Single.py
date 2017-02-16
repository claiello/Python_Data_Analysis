#ax01 = plt.subplot2grid((nolines,noplots), (0,1), colspan=1, rowspan=1)
#x,xx,xxx,hlpse = gmmone(se, se)
#ax01.imshow(hlpse,cmap=cm.Greys_r)
#ax111 = plt.subplot2grid((nolines,noplots), (0,3), colspan=1, rowspan=1)
#se_data3 = np.copy(se)
#se_data3[se_data3 < np.median(se_data3)] = 0.0
#se_data3[se_data3 > np.median(se_data3)] = 1.0
#hlpse3 = se_data3
#ax111.imshow(hlpse3,cmap=cm.Greys_r)
#
#ax1111 = plt.subplot2grid((nolines,noplots), (0,4), colspan=1, rowspan=1)
#x,xx,xxx,hlpse4 = thr_otsu(se, se)
#ax1111.imshow(hlpse4,cmap=cm.Greys_r)
import os
import sys
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../Python modules/") # necessary for the tex fonts
import scipy as sp
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
import h5py
import numpy as np
import matplotlib.cm as cm
import scipy.ndimage as ndimage
from matplotlib.backends.backend_pdf import PdfPages
from MakePdf import *
from matplotlib.pyplot import cm #to plot following colors of rainbow
from matplotlib import rc
#from CreateDatasets import *
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = DeprecationWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.simplefilter(action = "ignore", category = PendingDeprecationWarning)
import matplotlib.cm as cm
import scipy.misc
import gc
import tempfile
from tempfile import TemporaryFile
import skimage
from skimage import exposure
from my_fits import *
import pickle
import my_fits
from uncertainties import unumpy
from numpy import genfromtxt
import boe_bar as sb
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

from uncertainties import unumpy
from skimage.filters import threshold_otsu, threshold_adaptive#, threshold_local

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from skimage.filters import roberts, sobel, scharr, prewitt

from skimage.restoration import denoise_tv_chambolle, denoise_bilateral

sys.path.append('/usr/lib/python2.7/dist-packages')
import cv2

### settings
fsizepl = 24
fsizenb = 20
mkstry = ['8','11','5'] #marker size for different dsets Med Zoom/Large Zoom/Small Zoom
###
def tauestimate(counts_red, error_red):
    
    print(counts_red.shape[0])
    
    ucounts_red = unumpy.uarray(counts_red, error_red)
    
    def helper(arrayx):
         arrayx[arrayx < 1e-8] = 1e-8   #so that no division by zero     
         return np.cumsum(arrayx*np.arange(1,counts_red.shape[0]+1), axis = 0)/np.cumsum(arrayx, axis = 0)
    
    return helper(ucounts_red)
    
def markers_from_distance_local_max(image, distance=None, min_distance=10):
    if distance is None:
        distance = ndi.distance_transform_edt(image)
    return

def togs(image):
    
    return  (((image - image.min()) / (image.max() - image.min())) * 255.9).astype(np.uint8)

# eventually, should be range(len(lab)); right now, getting some areas only to see if works
def pickvec(lab):
    
    return range(len(lab))  #np.arange(0,4)
    
sizex = 8
sizey=6
dpi_no = 80

fig1= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig1.set_size_inches(1200./fig1.dpi,900./fig1.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    

nolines = 12
noplots = 16

ax51 = plt.subplot2grid((nolines,noplots), (0,noplots/2), colspan=noplots/2, rowspan=nolines/2)
ax510 = plt.subplot2grid((nolines,noplots), (nolines/2,0), colspan=noplots/2, rowspan=nolines/2)
ax5100 = plt.subplot2grid((nolines,noplots), (nolines/2,noplots/2), colspan=noplots/2, rowspan=nolines/2)


ax00 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=2)
ax01 = plt.subplot2grid((nolines,noplots), (0,3), colspan=1, rowspan=2)

ax001 = plt.subplot2grid((nolines,noplots), (2,0), colspan=2, rowspan=2) #SE
ax0112 = plt.subplot2grid((nolines,noplots), (2,2), colspan=2, rowspan=2) 
ax0111 = plt.subplot2grid((nolines,noplots), (2,4), colspan=2, rowspan=2) 
ax011 = plt.subplot2grid((nolines,noplots), (2,6), colspan=2, rowspan=2)

ax0022 = plt.subplot2grid((nolines,noplots), (4,0), colspan=2, rowspan=2)
ax0122 = plt.subplot2grid((nolines,noplots), (4,6), colspan=2, rowspan=2)

fig2= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig2.set_size_inches(1200./fig2.dpi,900./fig2.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino') 

ax510b = plt.subplot2grid((2,2), (1,0), colspan=1, rowspan=1)
ax5100b = plt.subplot2grid((2,2), (1,1), colspan=1, rowspan=1)
ax = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
axp = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)

#scalebar
length_scalebar = 100.0 #in nm 
scalebar_legend = '100 nm'
#when beam is on and off
backgroundinit = 50
initbin = 202
#to plot time evolution with rainbow, initial index and steps
step = 50
initind = 10
   
#### A
do_A = False
if do_A:
    
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
   
    ax00.imshow(se,cmap=cm.Greys_r)
    ax00.axis('off')
    
    Pixel_size = np.array([2.5]) 
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[0]))
    sbar = sb.AnchoredScaleBar(ax00.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4, my_fontsize = fsizenb)
    ax00.add_artist(sbar)
    
    #binary threshold
    se_data2 = np.copy(se)
    hlpse2 = se_data2 > filters.threshold_otsu(se_data2)
    hlpse2[hlpse2 == True] = 1.0
    hlpse2[hlpse2 == False] = 0.0
    
    # Find centers of NPs
    distance = ndimage.distance_transform_edt(hlpse2)
    local_maxi = peak_local_max(
            distance, 
            num_peaks = 50, 
            indices = False, 
            footprint = np.ones((25,25)),
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
    non_cut_k = []  ###### change cut_k
    for k in range(len(lab)):
        if (areas[k] < 100) or (areas[k] > 4000):
            cut_labels_ws[cut_labels_ws == lab[k]] = 0
            cut_k.append(k)
        else:
            non_cut_k.append(k)  ###change cut_k
    ax01.imshow(cut_labels_ws,cmap=cm.rainbow) #or 'OrRd'
    ax01.axis('off')

    del SEA, se
    gc.collect()
    
    REDA = np.load('N31pt2RedbrightA.npz')
    reda = REDA['data'][:,:,np.floor(xlen/2.0)-xval+delx+cutxtop:np.floor(xlen/2.0)+xval+delx-cutx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely]
    del REDA
    gc.collect()
    BLUEA = np.load('N31pt2BluebrightA.npz')
    bluea = BLUEA['data'][:,:,np.floor(xlen/2.0)-xval+delx+cutxtop:np.floor(xlen/2.0)+xval+delx-cutx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely]   
    del BLUEA
    gc.collect()    
    
    intens = np.empty(len(lab))
    stdintens = np.empty(len(lab))
    intensr = np.empty(len(lab))
    stdintensr = np.empty(len(lab))
    intensb = np.empty(len(lab))
    stdintensb = np.empty(len(lab))
    
    
    taured = np.empty([len(lab),reda.shape[1]-initbin])
    taublue = np.empty([len(lab),reda.shape[1]-initbin])
    tauredstd = np.empty([len(lab),reda.shape[1]-initbin])
    taubluestd = np.empty([len(lab),reda.shape[1]-initbin])
    
    for k in non_cut_k: #pickvec(lab):
     
       print(k)
       hlp = np.zeros(cut_labels_ws.shape)
       hlp[cut_labels_ws == lab[k]] = 1.0
       
       #Visibility
       print('Visib')
       ureda  =  reda[:,backgroundinit:initbin,:,:]
       uredastd = np.sqrt(reda[:,backgroundinit:initbin,:,:])
       ubluea =  bluea[:,backgroundinit:initbin,:,:]
       ublueastd = np.sqrt(bluea[:,backgroundinit:initbin,:,:])
       vec = np.nanmean(-(ureda-ubluea)/(ureda+ubluea) * hlp, axis = (0,1,2,3))
       vecr = np.nanmean(ureda * hlp, axis = (0,1,2,3))
       vecb = np.nanmean(ubluea * hlp, axis = (0,1,2,3))
       print(vec)
       vecstd = np.nanmean(np.sqrt(2)*np.sqrt(ureda**2 + ubluea**2)/np.sqrt((ureda+ubluea)**(3)) * hlp, axis = (0,1,2,3))
       vecstdr = np.nanmean(np.sqrt(ureda) * hlp, axis = (0,1,2,3))
       vecstdb = np.nanmean(np.sqrt(ubluea) * hlp, axis = (0,1,2,3))
       print(vecstd)
       del ureda, ubluea
       gc.collect()
       intens[k] = vec
       stdintens[k] = vecstd
       intensr[k] = vecr
       stdintensr[k] = vecstdr
       intensb[k] = vecb
       stdintensb[k] = vecstdb
       del vec, vecstd
       gc.collect()
       
       print('Taus')
       #Taus as a function of time
       redd = np.average(reda[:,initbin:,:,:] * hlp, axis = (0,2,3))
       blued = np.average(bluea[:,initbin:,:,:] * hlp, axis = (0,2,3))
       hr = tauestimate(redd,np.sqrt(redd))
       taured[k,:] = unumpy.nominal_values(hr)
       tauredstd[k,:] = unumpy.std_devs(hr)
       hb = tauestimate(blued,np.sqrt(blued))
       taublue[k,:] = unumpy.nominal_values(hb)
       taubluestd[k,:] = unumpy.std_devs(hb)
       sizered = redd.shape[0]
       del hr, hb, redd, blued
       gc.collect()
       
       ax5100.errobar(areas[k]*Pixel_size**2,unumpy.nominal_values(taured[k,-1]),yerr= unumpy.stde_devs(taured[k,-1]),marker = 'o', markersize = 5, color = 'k', ls='None')
       ax510.errorbar(areas[k]*Pixel_size**2,unumpy.nominal_values(taublue[k,-1]),yerr= unumpy.std_devs(taublue[k,-1]),marker = 'o', markersize = 5, color = 'k', ls='None')
               
       colors = iter(cm.rainbow(np.linspace(0, 1, len(np.arange(0,sizered,step)))))      
       for jj in np.arange(initind,sizered,step):
           print(jj)
           colorful = next(colors)
           ax5100b.plot(areas[k]*Pixel_size**2,unumpy.nominal_values(taured[k,jj]), marker = 'o', markersize = 5, color = colorful, ls='None')
           ax510b.plot(areas[k]*Pixel_size**2,unumpy.nominal_values(taublue[k,jj]), marker = 'o', markersize = 5, color = colorful, ls='None')
               
    ax51b.errorbar(areas[non_cut_k]*Pixel_size**2,intens[non_cut_k], yerr = stdintens[non_cut_k], marker = 'o', markersize = 5, color = 'k', ls='None')

    save_data = {}
    save_data['areas'] = areas*Pixel_size**2 #in nm^2
    save_data['taured'] = taured
    save_data['tauredstd'] = tauredstd
    save_data['taublue'] = taublue
    save_data['taubluestd'] = taubluestd
    save_data['intens'] = intens
    save_data['intensstd'] = stdintens
    save_data['intensr'] = intensr
    save_data['intensstdr'] = stdintensr
    save_data['intensb'] = intensb
    save_data['intensstdb'] = stdintensb
    save_data['non_cut_k'] = non_cut_k
    pickle.dump(save_data, open("A.p", "wb"))

do_B = True #already ran, can just open files and read
if do_B:
    
    Pixel_size = np.array([0.89]) 
    
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
       
    ax001.imshow(se,cmap=cm.Greys_r)
    ax001.axis('off') 
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[0]))
    sbar = sb.AnchoredScaleBar(ax001.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4, my_fontsize = fsizenb)
    ax001.add_artist(sbar)       
       
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
    non_cut_k = []  ###### change cut_k
    for k in range(len(lab)):
        if (areas[k] < 1200) or (areas[k] > 4000):
            cut_labels_ws[cut_labels_ws == lab[k]] = 0
            cut_k.append(k)
        else:
            non_cut_k.append(k)  ###change cut_k
    ax011.imshow(cut_labels_ws,cmap=cm.rainbow) #or 'OrRd'
    ax011.axis('off')
    ax0111.imshow(bw,cmap=cm.Greys_r) #or 'OrRd'
    ax0111.axis('off')
    ax0112.imshow(new_pic_grad,cmap=cm.Greys_r) #or 'OrRd'
    ax0112.axis('off')

    del SEA, se
    gc.collect()
    
    ####### load file that exists
    B = pickle.load( open( "B.p", "rb" ) )
    areas = B['areas']
    taured = B['taured']
    tauredstd = B['tauredstd']
    taublue = B['taublue']
    taubluestd = B['taubluestd']
    intens = B['intens']
    stdintens = B['intensstd']
    intensr = B['intensr']
    stdintensr = B['intensstdr']
    intensb = B['intensb']
    stdintensb = B['intensstdb']
    non_cut_k = B['non_cut_k']  
    
#    REDA = np.load('V0RedbrightB.npz')
#    reda = REDA['data'][:,:,np.floor(xlen/2.0)-xval+delx+cutxtop:np.floor(xlen/2.0)+xval+delx-cutx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely]
#    del REDA
#    gc.collect()
#    BLUEA = np.load('V0BluebrightB.npz')
#    bluea = BLUEA['data'][:,:,np.floor(xlen/2.0)-xval+delx+cutxtop:np.floor(xlen/2.0)+xval+delx-cutx,np.floor(ylen/2.0)-yval+dely:np.floor(ylen/2.0)+yval+dely]   
#    del BLUEA
#    gc.collect()    
#    
#    intens = np.empty(len(lab))
#    stdintens = np.empty(len(lab))
#    intensr = np.empty(len(lab))
#    stdintensr = np.empty(len(lab))
#    intensb = np.empty(len(lab))
#    stdintensb = np.empty(len(lab))
#    
#    taured = np.empty([len(lab),reda.shape[1]-initbin])
#    taublue = np.empty([len(lab),reda.shape[1]-initbin])
#    tauredstd = np.empty([len(lab),reda.shape[1]-initbin])
#    taubluestd = np.empty([len(lab),reda.shape[1]-initbin])
#    
#    #print(len(lab))
#    #print(non_cut_k)
#    #print(len(non_cut_k))
    for k in non_cut_k: #pickvec(lab): ######### change cut_k
     
#       print(k)
#       hlp = np.zeros(cut_labels_ws.shape)
#       hlp[cut_labels_ws == lab[k]] = 1.0
#       
#       #Visibility
#       print('Visib')
#       ureda  =  reda[:,backgroundinit:initbin,:,:]
#       uredastd = np.sqrt(reda[:,backgroundinit:initbin,:,:])
#       ubluea =  bluea[:,backgroundinit:initbin,:,:]
#       ublueastd = np.sqrt(bluea[:,backgroundinit:initbin,:,:])
#       
#       #visibilitty
#       vec = np.nanmean(-(ureda-ubluea)/(ureda+ubluea) * hlp, axis = (0,1,2,3))
#       vecr = np.nanmean(ureda * hlp, axis = (0,1,2,3))
#       vecb = np.nanmean(ubluea * hlp, axis = (0,1,2,3))
#       print(vec)
#       vecstd = np.nanmean(np.sqrt(2)*np.sqrt(ureda**2 + ubluea**2)/np.sqrt((ureda+ubluea)**(3)) * hlp, axis = (0,1,2,3))
#       vecstdr = np.nanmean(np.sqrt(ureda) * hlp, axis = (0,1,2,3))
#       vecstdb = np.nanmean(np.sqrt(ubluea) * hlp, axis = (0,1,2,3))
#       print(vecstd)
##       #ratio
##       vec = np.nanmean((ubluea)/(ureda) * hlp, axis = (0,1,2,3))
##       vecr = np.nanmean(ureda * hlp, axis = (0,1,2,3))
##       vecb = np.nanmean(ubluea * hlp, axis = (0,1,2,3))
##       print(vec)
##       vecstd = np.nanmean(np.sqrt(ureda*(ureda + ubluea))/np.sqrt((ubluea)**(3)) * hlp, axis = (0,1,2,3))
##       vecstdr = np.nanmean(np.sqrt(ureda) * hlp, axis = (0,1,2,3))
##       vecstdb = np.nanmean(np.sqrt(ubluea) * hlp, axis = (0,1,2,3))
##       print(vecstd)
#       
#       del ureda, ubluea
#       gc.collect()
#       intens[k] = vec
#       stdintens[k] = vecstd
#       intensr[k] = vecr
#       stdintensr[k] = vecstdr
#       intensb[k] = vecb
#       stdintensb[k] = vecstdb
#       del vec, vecstd
#       gc.collect()
#       
#       print('Taus')
#       #Taus as a function of time
#       redd = np.average(reda[:,initbin:,:,:] * hlp, axis = (0,2,3))
#       blued = np.average(bluea[:,initbin:,:,:] * hlp, axis = (0,2,3))
#       hr = tauestimate(redd,np.sqrt(redd))
#       taured[k,:] = unumpy.nominal_values(hr)
#       tauredstd[k,:] = unumpy.std_devs(hr)
#       hb = tauestimate(blued,np.sqrt(blued))
#       taublue[k,:] = unumpy.nominal_values(hb)
#       taubluestd[k,:] = unumpy.std_devs(hb)
       sizered = 1398 #redd.shape[0]
#       del hr, hb, redd, blued
#       gc.collect()
               
       
       print(unumpy.std_devs(taured[k,-1]))
       print(unumpy.std_devs(taublue[k,-1]))
       #ax5100.errorbar(areas[k]*Pixel_size**2,taured[k,-1], yerr=tauredstd[k,-1], marker = 's', markersize = 8, color = 'k', ls='None')
       #ax510.errorbar(areas[k]*Pixel_size**2,taublue[k,-1], yerr=taubluestd[k,-1],marker = 's', markersize = 8, color = 'k', ls='None')
       #saved area is already multiplied by Pix^2
       ax5100.errorbar(areas[k],taured[k,-1], yerr=tauredstd[k,-1], marker = 's', markersize = 8, color = 'k', ls='None')
       ax510.errorbar(areas[k],taublue[k,-1], yerr=taubluestd[k,-1],marker = 's', markersize = 8, color = 'k', ls='None')
      
       colors = iter(cm.rainbow(np.linspace(0, 1, len(np.arange(0,sizered,step)))))      
       for jj in np.arange(initind,sizered,step):
           print(jj)
           colorful = next(colors)
           #ax5100b.plot(areas[k]*Pixel_size**2,taured[k,jj], marker = 's', markersize = 8, color = colorful, ls='None')
           #ax510b.plot(areas[k]*Pixel_size**2,taublue[k,jj], marker = 's', markersize = 8, color = colorful, ls='None')
           #saved area is already multiplied by Pix^2
           ax5100b.plot(areas[k],taured[k,jj], marker = 's', markersize = 8, color = colorful, ls='None')
           ax510b.plot(areas[k],taublue[k,jj], marker = 's', markersize = 8, color = colorful, ls='None')
               
    ######### change cut_k
    #ax51.errorbar(areas[non_cut_k]*Pixel_size**2,intens[non_cut_k], yerr = stdintens[non_cut_k], marker = 's', markersize = 8, color = 'k', ls='None')
    #saved area is already multiplied by Pix^2
    ax51.errorbar(areas[non_cut_k],intens[non_cut_k], yerr = stdintens[non_cut_k], marker = 's', markersize = 8, color = 'k', ls='None')

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
#    pickle.dump(save_data, open("B.p", "wb"))
 
#### C
do_C = True #ran already - can just load data
if do_C:
    
    Pixel_size = np.array([0.74]) 
    SEA= np.load('singleSEchannelC.npz')
    se = SEA['data'][40:190,120:270]
   
    ax0022.imshow(se,cmap=cm.Greys_r)
    ax0022.axis('off') 
    length_scalebar = 50.0 #in nm 
    scalebar_legend = '50 nm'
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[0]))
    sbar = sb.AnchoredScaleBar(ax0022.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 8, my_fontsize = fsizenb)
    ax0022.add_artist(sbar)
    
    #binary threshold
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
    non_cut_k = []  ###### change cut_k
    for k in range(len(lab)):
        if (areas[k] < 100) or (areas[k] > 17000):
            cut_labels_ws[cut_labels_ws == lab[k]] = 0
            cut_k.append(k)
        else:
            non_cut_k.append(k)  ###change cut_k
    
    print(len(lab))
    print(len(cut_k))
    print(cut_labels_ws.shape)
    ax0122.imshow(cut_labels_ws,cmap=cm.rainbow) #or 'OrRd'
    ax0122.axis('off')

    del SEA, se
    gc.collect()
    
    ####### load file that exists
    C = pickle.load( open( "C.p", "rb" ) )
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
    
#    REDA = np.load('singleRedbrightC.npz')
#    reda = REDA['data'][:,:,40:190,120:270]
#    del REDA
#    gc.collect()
#    BLUEA = np.load('singleBluebrightC.npz')
#    bluea = BLUEA['data'][:,:,40:190,120:270]
#    del BLUEA
#    gc.collect()    
#    
#    intens = np.empty(len(lab))
#    stdintens = np.empty(len(lab))
#    intensr = np.empty(len(lab))
#    stdintensr = np.empty(len(lab))
#    intensb = np.empty(len(lab))
#    stdintensb = np.empty(len(lab))
#    
#    taured = np.empty([len(lab),reda.shape[1]-initbin])
#    taublue = np.empty([len(lab),reda.shape[1]-initbin])
#    tauredstd = np.empty([len(lab),reda.shape[1]-initbin])
#    taubluestd = np.empty([len(lab),reda.shape[1]-initbin])
#    
    for k in non_cut_k: #pickvec(lab): #range(len(lab)):
     
#       print(k)
#       hlp = np.zeros(cut_labels_ws.shape)
#       hlp[cut_labels_ws == lab[k]] = 1.0
#       
#       #Visibility
#       print('Visib')
#       ureda  =  reda[:,backgroundinit:initbin,:,:]
#       uredastd = np.sqrt(reda[:,backgroundinit:initbin,:,:])
#       ubluea =  bluea[:,backgroundinit:initbin,:,:]
#       ublueastd = np.sqrt(bluea[:,backgroundinit:initbin,:,:])
#       vec = np.nanmean(-(ureda-ubluea)/(ureda+ubluea) * hlp, axis = (0,1,2,3))
#       vecr = np.nanmean(ureda * hlp, axis = (0,1,2,3))
#       vecb = np.nanmean(ubluea * hlp, axis = (0,1,2,3))
#       print(vec)
#       vecstd = np.nanmean(np.sqrt(2)*np.sqrt(ureda**2 + ubluea**2)/np.sqrt((ureda+ubluea)**(3)) * hlp, axis = (0,1,2,3))
#       vecstdr = np.nanmean(np.sqrt(ureda) * hlp, axis = (0,1,2,3))
#       vecstdb = np.nanmean(np.sqrt(ubluea) * hlp, axis = (0,1,2,3))
#       print(vecstd)
#       del ureda, ubluea
#       gc.collect()
#       intens[k] = vec
#       stdintens[k] = vecstd
#       intensr[k] = vecr
#       stdintensr[k] = vecstdr
#       intensb[k] = vecb
#       stdintensb[k] = vecstdb
#       del vec, vecstd
#       gc.collect()
#       
#       print('Taus')
#       #Taus as a function of time
#       redd = np.average(reda[:,initbin:,:,:] * hlp, axis = (0,2,3))
#       blued = np.average(bluea[:,initbin:,:,:] * hlp, axis = (0,2,3))
#       hr = tauestimate(redd,np.sqrt(redd))
#       taured[k,:] = unumpy.nominal_values(hr)
#       tauredstd[k,:] = unumpy.std_devs(hr)
#       hb = tauestimate(blued,np.sqrt(blued))
#       taublue[k,:] = unumpy.nominal_values(hb)
#       taubluestd[k,:] = unumpy.std_devs(hb)
       sizered = 1398 #redd.shape[0]
#       del hr, hb, redd, blued
#       gc.collect()
       
       #ax5100.errorbar(areas[k]*Pixel_size**2,taured[k,-1], yerr=tauredstd[k,-1],marker = '^', markersize = 11, color = 'k', ls='None')
       #ax510.errorbar(areas[k]*Pixel_size**2,taublue[k,-1],yerr=taubluestd[k,-1],marker = '^', markersize = 11, color = 'k', ls='None')
       #Saved area is already times Pix^2       
       ax5100.errorbar(areas[k],taured[k,-1], yerr=tauredstd[k,-1],marker = '^', markersize = 11, color = 'k', ls='None')
       ax510.errorbar(areas[k],taublue[k,-1],yerr=taubluestd[k,-1],marker = '^', markersize = 11, color = 'k', ls='None')
       
       
       colors = iter(cm.rainbow(np.linspace(0, 1, len(np.arange(0,sizered,step)))))      
       for jj in np.arange(initind,sizered,step):
           print(jj)
           colorful = next(colors)
           #ax5100b.plot(areas[k]*Pixel_size**2,taured[k,jj], marker = '^', markersize = 11, color = colorful, ls='None')
           #ax510b.plot(areas[k]*Pixel_size**2,taublue[k,jj], marker = '^', markersize = 11, color = colorful, ls='None')
           #Saved area is already times Pix^2       
           ax5100b.plot(areas[k],taured[k,jj], marker = '^', markersize = 11, color = colorful, ls='None')
           ax510b.plot(areas[k],taublue[k,jj], marker = '^', markersize = 11, color = colorful, ls='None')
               
    #ax51.errorbar(areas[non_cut_k]*Pixel_size**2,intens[non_cut_k], yerr = stdintens[non_cut_k], marker = '^', markersize = 11, color = 'k', ls='None')
     #saved area is already multiplied by Pix^2
    ax51.errorbar(areas[non_cut_k],intens[non_cut_k], yerr = stdintens[non_cut_k], marker = '^', markersize = 11, color = 'k', ls='None')

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
#    pickle.dump(save_data, open("C.p", "wb"))   
    
#### D
#Takes pixel size and area from C
do_D = True #####USE D WITH SAME AREA AS C!!!!!!!!!!!! #ran already - can just load data
if do_D:
    
    if do_C is False:
        errorCneedstocumpilefurDtowork
        
    ####### load file that exists
    D = pickle.load( open( "D.p", "rb" ) )
    areas = D['areas']
    taured = D['taured']
    tauredstd = D['tauredstd']
    taublue = D['taublue']
    taubluestd = D['taubluestd']
    intens = D['intens']
    stdintens = D['intensstd']
    intensr = D['intensr']
    stdintensr = D['intensstdr']
    intensb = D['intensb']
    stdintensb = D['intensstdb']
    non_cut_k = D['non_cut_k']  
    
#    REDA = np.load('backRedbrightD.npz')
#    reda = REDA['data'][:,:,40:190,120:270] #same no pixels than C, single
#    del REDA
#    gc.collect()
#    BLUEA = np.load('backBluebrightD.npz')
#    bluea = BLUEA['data'][:,:,40:190,120:270]#same no pixels than C, single
#    del BLUEA
#    gc.collect()    
#    
#    intens = np.empty(len(lab))
#    stdintens = np.empty(len(lab))
#    intensr = np.empty(len(lab))
#    stdintensr = np.empty(len(lab))
#    intensb = np.empty(len(lab))
#    stdintensb = np.empty(len(lab))
#    
#    taured = np.empty([len(lab),reda.shape[1]-initbin])
#    taublue = np.empty([len(lab),reda.shape[1]-initbin])
#    tauredstd = np.empty([len(lab),reda.shape[1]-initbin])
#    taubluestd = np.empty([len(lab),reda.shape[1]-initbin])
    
    for k in non_cut_k: #pickvec(lab): #range(len(lab)):
     
#       print(k)
#       hlp = np.zeros(cut_labels_ws.shape)
#       hlp[cut_labels_ws == lab[k]] = 1.0
#       
#       #Visibility
#       print('Visib')
#       ureda  =  reda[:,backgroundinit:initbin,:,:]
#       uredastd = np.sqrt(reda[:,backgroundinit:initbin,:,:])
#       ubluea =  bluea[:,backgroundinit:initbin,:,:]
#       ublueastd = np.sqrt(bluea[:,backgroundinit:initbin,:,:])
#       vec = np.nanmean(-(ureda-ubluea)/(ureda+ubluea) * hlp, axis = (0,1,2,3))
#       vecr = np.nanmean(ureda * hlp, axis = (0,1,2,3))
#       vecb = np.nanmean(ubluea * hlp, axis = (0,1,2,3))
#       print(vec)
#       vecstd = np.nanmean(np.sqrt(2)*np.sqrt(ureda**2 + ubluea**2)/np.sqrt((ureda+ubluea)**(3)) * hlp, axis = (0,1,2,3))
#       vecstdr = np.nanmean(np.sqrt(ureda) * hlp, axis = (0,1,2,3))
#       vecstdb = np.nanmean(np.sqrt(ubluea) * hlp, axis = (0,1,2,3))
#       print(vecstd)
#       del ureda, ubluea
#       gc.collect()
#       intens[k] = vec
#       stdintens[k] = vecstd
#       intensr[k] = vecr
#       stdintensr[k] = vecstdr
#       intensb[k] = vecb
#       stdintensb[k] = vecstdb
#       del vec, vecstd
#       gc.collect()
#       
#       print('Taus')
#       #Taus as a function of time
#       redd = np.average(reda[:,initbin:,:,:] * hlp, axis = (0,2,3))
#       blued = np.average(bluea[:,initbin:,:,:] * hlp, axis = (0,2,3))
#       hr = tauestimate(redd,np.sqrt(redd))
#       taured[k,:] = unumpy.nominal_values(hr)
#       tauredstd[k,:] = unumpy.std_devs(hr)
#       hb = tauestimate(blued,np.sqrt(blued))
#       taublue[k,:] = unumpy.nominal_values(hb)
#       taubluestd[k,:] = unumpy.std_devs(hb)
#       sizered = 1398 #redd.shape[0]
#       del hr, hb, redd, blued
#       gc.collect()
       
       ax5100.errorbar(areas[k],taured[k,-1],yerr=tauredstd[k,-1], marker = 'x', markersize = 11, color = 'k', ls='None')
       ax510.errorbar(areas[k],taublue[k,-1], yerr=taubluestd[k,-1],marker = 'x', markersize = 11, color = 'k', ls='None')
       
       colors = iter(cm.rainbow(np.linspace(0, 1, len(np.arange(0,sizered,step)))))      
       for jj in np.arange(initind,sizered,step):
           print(jj)
           colorful = next(colors)
           ax5100b.plot(areas[k],taured[k,jj], marker = 'x', markersize = 11, color = colorful, ls='None')
           ax510b.plot(areas[k],taublue[k,jj], marker = 'x', markersize = 11, color = colorful, ls='None')
               
    ax51.errorbar(areas[non_cut_k],intens[non_cut_k], yerr = stdintens[non_cut_k], marker = 'x', markersize = 11, color = 'k', ls='None')

#    save_data = {}
#    save_data['areas'] = areas #in nm^2  AREA IS THE SAME AS SINGLE PARTICLE FROM C
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
#    pickle.dump(save_data, open("D.p", "wb"))  

# if jjj == 0:
#            ax1.text(55, 175, '   ' + 'time in\n 50 $\mu$s intervals', fontsize=fsizenb, va='center',ha='center')
#            ax1.annotate('', xy=(55,250), xytext=(55,200),
#                arrowprops=dict(facecolor='black', shrink=0.05))   
#            ax3.text(-0.1, 1.0, 'a', transform=ax3.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})
 
ax51.set_ylabel('Visibility of intensity (a.u.)',fontsize=fsizepl)
ax51.set_xlabel('Nanoparticle area (nm$^{2}$)',fontsize=fsizepl)
ax51.tick_params(labelsize=fsizenb)   
ax510.set_ylabel(r'Green band $\tau$' + '\n after 1.5 ms acquisition ($\mu$s)',fontsize=fsizepl)
ax5100.set_ylabel(r'Red band $\tau$' + '\n after 1.5 ms acquisition ($\mu$s)',fontsize=fsizepl)
ax510b.set_ylabel(r'Green band $\tau$ ($\mu$s)',fontsize=fsizepl)
ax5100b.set_ylabel(r'Red band $\tau$ ($\mu$s)',fontsize=fsizepl)
ax510.set_xlabel('Nanoparticle area (nm$^{2}$)',fontsize=fsizepl)
ax5100.set_xlabel('Nanoparticle area (nm$^{2}$)',fontsize=fsizepl)
ax510.tick_params(labelsize=fsizenb)
ax5100.tick_params(labelsize=fsizenb)
ax51.spines['left'].set_visible(False)
ax51.spines['top'].set_visible(False)
ax51.xaxis.set_ticks_position('bottom')
ax51.yaxis.set_ticks_position('right')
ax51.yaxis.set_label_position("right")
ax510.spines['right'].set_visible(False)
ax510.spines['top'].set_visible(False)
ax510.xaxis.set_ticks_position('bottom')
ax510.yaxis.set_ticks_position('left')
ax5100.spines['left'].set_visible(False)
ax5100.spines['top'].set_visible(False)
ax5100.xaxis.set_ticks_position('bottom')
ax5100.yaxis.set_ticks_position('right')
ax5100.yaxis.set_label_position("right")
ax510b.tick_params(labelsize=fsizenb)
ax5100b.tick_params(labelsize=fsizenb)
ax5100b.spines['left'].set_visible(False)
ax5100b.spines['top'].set_visible(False)
ax5100b.xaxis.set_ticks_position('bottom')
ax5100b.yaxis.set_ticks_position('right')
ax5100b.yaxis.set_label_position("right")
ax510b.set_xlabel('Nanoparticle area (nm$^{2}$)',fontsize=fsizepl)
ax5100b.set_xlabel('Nanoparticle area (nm$^{2}$)',fontsize=fsizepl)

#plt.tight_layout()

if do_A:
    A = pickle.load( open( "A.p", "rb" ) )
if do_B:
    B = pickle.load( open( "B.p", "rb" ) )
if do_C:
    C = pickle.load( open( "C.p", "rb" ) )
if do_D:
    D = pickle.load( open( "D.p", "rb" ) )
areafactor = 10.0


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel(r'Green band $\tau$' + '\n after 1.5 ms of acquisition ($\mu$s)',fontsize=fsizepl)
ax.set_xlabel('Average intensity per pixel (kHz)',fontsize=fsizepl)
ax.tick_params(labelsize=fsizenb) 
if do_A:
    #green t, different areas, circleA
    ax.scatter(A['intensb'][A['non_cut_k']],A['taublue'][A['non_cut_k'],-1],s=A['areas'][A['non_cut_k']]/areafactor, color='g', alpha=0.5, marker = 'o')
if do_B:
    #green t, different areas, squareB
    ax.scatter(B['intensb'][B['non_cut_k']],B['taublue'][B['non_cut_k'],-1],s=B['areas'][B['non_cut_k']]/areafactor, color='g', alpha=0.5, marker = 's')
if do_C:
    #green t, different areas, triangleC
    ax.scatter(C['intensb'][C['non_cut_k']],C['taublue'][C['non_cut_k'],-1],s=C['areas'][C['non_cut_k']]/areafactor, color='g', alpha=0.5, marker = '^')
if do_D:
    #green t, different areas, triangleC
    ax.scatter(D['intensb'][D['non_cut_k']],D['taublue'][D['non_cut_k'],-1],s=D['areas'][D['non_cut_k']]/areafactor, color='g', alpha=0.5, marker = 'x')


axp.spines['left'].set_visible(False)
axp.spines['top'].set_visible(False)
axp.xaxis.set_ticks_position('bottom')
axp.yaxis.set_ticks_position('right')
axp.set_ylabel(r'Red band $\tau$' + '\n after 1.5 ms of acquisition ($\mu$s)',fontsize=fsizepl)
axp.set_xlabel('Average intensity per pixel (kHz)',fontsize=fsizepl)
axp.tick_params(labelsize=fsizenb) 
axp.yaxis.set_label_position("right")
if do_A:
    #red t, different areas, circleA
    axp.scatter(A['intensr'][A['non_cut_k']],A['taured'][A['non_cut_k'],-1],s=A['areas'][A['non_cut_k']]/areafactor, color='r', alpha=0.5, marker = 'o')
if do_B:
    #red t, different areas, squareB
    axp.scatter(B['intensr'][B['non_cut_k']],B['taured'][B['non_cut_k'],-1],s=B['areas'][B['non_cut_k']]/areafactor, color='r', alpha=0.5, marker = 's')
if do_C:
    #red t, different areas, triangleC
    axp.scatter(C['intensr'][C['non_cut_k']],C['taured'][C['non_cut_k'],-1],s=C['areas'][C['non_cut_k']]/areafactor, color='r', alpha=0.5, marker = '^')
if do_D:
    #red t, different areas, triangleC
    axp.scatter(D['intensr'][D['non_cut_k']],D['taured'][D['non_cut_k'],-1],s=D['areas'][C['non_cut_k']]/areafactor, color='r', alpha=0.5, marker = 'x')

plt.tight_layout()
plt.show()
    
multipage_longer('Single.pdf',dpi=80)    
klklklk