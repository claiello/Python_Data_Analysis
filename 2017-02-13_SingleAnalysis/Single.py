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
####office
from skimage.feature import peak_local_max,canny #UNCOMMENT AND RUN IN BOES COMPI
from skimage import filters 
from skimage.filters import threshold_otsu, threshold_adaptive, rank #, threshold_local
####laptop
#from skimage.feature import peak_local_max#,canny
#from skimage import filter
#from skimage.filter import threshold_otsu, threshold_adaptive, rank, canny #, threshold_local
######

from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.morphology import disk
from scipy import ndimage
import sklearn

import matplotlib.patches as patches
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.io import imread, imsave, imshow
from skimage.morphology import watershed

from uncertainties import unumpy

#from skimage.morphology import black_tophat, skeletonize, convex_hull_image
#from skimage import segmentation
#from skimage.morphology import erosion, dilation, opening, closing, white_tophat
#from skimage.filters import roberts, sobel, scharr, prewitt

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
         #arrayx[arrayx < 1e-12] = 1e-12   #so that no division by zero     
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
    
def tor(area):
    
    return 2*np.sqrt(2.*area/(3.*np.sqrt(3)))

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

az = 8 #noplots/2
aw = 6 #nolines/2
ax = plt.subplot2grid((nolines,noplots), (0,az), colspan=az, rowspan=aw)

ax51=ax.twinx()

ax510 = plt.subplot2grid((nolines,noplots), (aw,0), colspan=az, rowspan=aw)
ax5100 = plt.subplot2grid((nolines,noplots), (aw,az), colspan=az, rowspan=aw)
ax51.text(1.1, 1.0, 'f', transform=ax51.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})
ax510.text(-0.1, 1.0, 'e', transform=ax510.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})


gs = matplotlib.gridspec.GridSpec(3, 4)
#gs.update(wspace=0.1, hspace=-0.5, left=0.115, right=0.5, bottom=0.1, top=0.99) 
gs.update(wspace=0.005, hspace=0.01,left=0.05, right=0.48, bottom=0.5, top=0.985) 

ax00 = plt.subplot(gs[0,0])
ax01 = plt.subplot(gs[0,1])
ax02 = plt.subplot(gs[0,2])
ax03 = plt.subplot(gs[0,3])
ax00.text(-0.125, 1.0, 'a', transform=ax00.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})

ax001 = plt.subplot(gs[1,0])
ax0112 = plt.subplot(gs[1,1])
ax0111 = plt.subplot(gs[1,2])
ax011 = plt.subplot(gs[1,3])
ax001.text(-0.1, 1.0, 'b', transform=ax001.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})

ax0022 = plt.subplot(gs[2,0])
ax0122 = plt.subplot(gs[2,1])
axpic = plt.subplot(gs[2,2])
axpic2 = plt.subplot(gs[2,3])
ax0022.text(-0.1, 1.0, 'c', transform=ax0022.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})
axpic.text(+0.20, 1.0, 'd', transform=axpic.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})

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
do_A = True
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
    
    se = SEA['data'][int(np.floor(xlen/2.0)-xval+delx+cutxtop):int(np.floor(xlen/2.0)+xval+delx-cutx),int(np.floor(ylen/2.0)-yval+dely):int(np.floor(ylen/2.0)+yval+dely)]
   
    ax00.imshow(se,cmap=cm.Greys_r)
    ax00.axis('off')
    
    Pixel_size = np.array([2.5]) 
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[0]))
    sbar = sb.AnchoredScaleBar(ax00.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4, my_fontsize = fsizenb)
    ax00.add_artist(sbar)
    
    se_data2 = np.copy(se)
    tv_filt = denoise_tv_chambolle(se_data2, weight=0.0025)
    grad_x, grad_y = np.gradient(tv_filt)
    new_pic_grad = np.sqrt(grad_x**2 + grad_y**2)
   
    hlpse2 = np.copy(new_pic_grad)
    local_otsu = rank.otsu(togs(hlpse2), disk(6))
    thres = threshold_otsu(togs(hlpse2))
    masklarge = togs(hlpse2) >= local_otsu
    
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
        if (areas[k] < 10) or (areas[k] > 20000) or (k == 0): #k==o is modif!!! k==0 point is WEIRD
            cut_labels_ws[cut_labels_ws == lab[k]] = 0
            cut_k.append(k)
        else:
            non_cut_k.append(k)  ###change cut_k
    
    print(len(lab))
    print(len(non_cut_k))
    ax01.imshow(new_pic_grad,cmap=cm.Greys_r) #or 'OrRd'
    ax01.axis('off')
    ax02.imshow(masklarge,cmap=cm.Greys_r) #or 'OrRd'
    ax02.axis('off')
   
    #code to make all black - this will crash if running to save data
    #cut_labels_ws[cut_labels_ws > 1] = 1.0
    
    ax03.imshow(cut_labels_ws, cmap = cm.Greys_r) #or 'OrRd'
    ax03.axis('off')

    del SEA, se
    gc.collect()
    
    A = pickle.load( open( "A.p", "rb" ) )
    A = np.load("Anpz.npz")
    areas = A['areas']
    taured = A['taured']
    tauredstd = A['tauredstd']
    taublue = A['taublue']
    taubluestd = A['taubluestd']
    intens = A['intens']
    stdintens = A['stdintens']
    intensr = A['intensr']
    stdintensr = A['stdintensr']
    intensb = A['intensb']
    stdintensb = A['stdintensb']
    non_cut_k = A['non_cut_k']  

#    Anpz = tempfile.NamedTemporaryFile(delete=False)
#    np.savez('Anpz', areas = areas, 
#                   taured = taured, 
#                   tauredstd = tauredstd, 
#                   taublue = taublue, 
#                   taubluestd = taubluestd,
#                   intens = intens,
#                   stdintens = stdintens,
#                   intensr = intensr,
#                   stdintensr = stdintensr,
#                   intensb = intensb,
#                   stdintensb = stdintensb,
#                   non_cut_k = non_cut_k)
#                   
    
#    
#    REDA = np.load('N31pt2RedbrightA.npz')
#    reda = REDA['data'][:,:,int(np.floor(xlen/2.0)-xval+delx+cutxtop):int(np.floor(xlen/2.0)+xval+delx-cutx),int(np.floor(ylen/2.0)-yval+dely):int(np.floor(ylen/2.0)+yval+dely)]
#    del REDA
#    gc.collect()
#    BLUEA = np.load('N31pt2BluebrightA.npz')
#    bluea = BLUEA['data'][:,:,int(np.floor(xlen/2.0)-xval+delx+cutxtop):int(np.floor(xlen/2.0)+xval+delx-cutx),int(np.floor(ylen/2.0)-yval+dely):int(np.floor(ylen/2.0)+yval+dely)]   
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

    for k in non_cut_k: #np.arange(1,len(non_cut_k)):#non_cut_k: #pickvec(lab):
     
#       print(k)
#       hlp = np.zeros(cut_labels_ws.shape)
#       hlp[cut_labels_ws == lab[k]] = 1.0
#       
#       #Visibility
#       print('Visib')
#       ureda  =  reda[:,backgroundinit:initbin,:,:]
#       ubluea =  bluea[:,backgroundinit:initbin,:,:]
#       vec = np.nanmean(-(ureda-ubluea)/(ureda+ubluea) * hlp, axis = (0,1,2,3))
#       vecr = np.nanmean(ureda * hlp, axis = (0,1,2,3))
#       vecb = np.nanmean(ubluea * hlp, axis = (0,1,2,3))
#       No = np.sum(hlp.astype(np.float64))*reda.shape[0]*reda[:,backgroundinit:initbin,:,:].shape[1]
#       vecstd = np.nanmean(np.sqrt(2)*np.sqrt(ureda**2 + ubluea**2)/np.sqrt((ureda+ubluea)**(3)) * hlp, axis = (0,1,2,3))
#       vecstdr = 1.0/np.sqrt(No) * np.sqrt(vecr) #np.nanmean(np.sqrt(ureda) * hlp, axis = (0,1,2,3))
#       vecstdb = 1.0/np.sqrt(No) * np.sqrt(vecb) #
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
#       Notr = np.sum(hlp.astype(np.float64))*reda.shape[0]
#       hr = tauestimate(redd,np.sqrt(redd)/np.sqrt(Notr))
#       taured[k,:] = unumpy.nominal_values(hr)
#       tauredstd[k,:] = unumpy.std_devs(hr)
#       hb = tauestimate(blued,np.sqrt(blued)/np.sqrt(Notr))
#       taublue[k,:] = unumpy.nominal_values(hb)
#       taubluestd[k,:] = unumpy.std_devs(hb)
#       sizered = 1398 #redd.shape[0]
#       print(Notr)
#       del hr, hb, redd, blued
#       gc.collect()
#       
       #ax5100.errorbar(areas[k]*Pixel_size**2,taured[k,-1],yerr= tauredstd[k,-1],marker = 'o', markersize = 5, color = 'k', ls='None')
       #ax510.errorbar(areas[k]*Pixel_size**2,taublue[k,-1],yerr= taubluestd[k,-1],marker = 'o', markersize = 5, color = 'k', ls='None')
       
       #ax5100.errorbar(tor(areas[k]),taured[k,-1],yerr= 0*tauredstd[k,-1],marker = 'H', markersize = 5, color = 'r', ls='None')
       #ax510.errorbar(tor(areas[k]),taublue[k,-1],yerr= 0*taubluestd[k,-1],marker = 'H', markersize = 5, color = 'g', ls='None')
       ax5100.plot(tor(areas[k]),taured[k,-1],marker = 'H', markersize = 8, color = 'r', ls='None', markeredgewidth=0.0)
       ax510.plot(tor(areas[k]),taublue[k,-1],marker = 'H', markersize = 8, color = 'g', ls='None', markeredgewidth=0.0)
                      
           
#       colors = iter(cm.rainbow(np.linspace(0, 1, len(np.arange(0,sizered,step)))))      
#       for jj in np.arange(initind,sizered,step):
#           print(jj)
#           colorful = next(colors)
#           #ax5100b.plot(areas[k]*Pixel_size**2,taured[k,jj], marker = 'o', markersize = 5, color = colorful, ls='None')
#           #ax510b.plot(areas[k]*Pixel_size**2,taublue[k,jj], marker = 'o', markersize = 5, color = colorful, ls='None')
#           ax5100b.plot(tor(areas[k]),taured[k,jj], marker = 'o', markersize = 5, color = colorful, ls='None')
#           ax510b.plot(tor(areas[k]),taublue[k,jj], marker = 'o', markersize = 5, color = colorful, ls='None')
               
    #ax51.errorbar(areas[non_cut_k]*Pixel_size**2,intens[non_cut_k], yerr = stdintens[non_cut_k], marker = 'o', markersize = 5, color = 'k', ls='None')
    #ax51.errorbar(tor(areas[non_cut_k]),intens[non_cut_k], yerr = stdintens[non_cut_k], marker = 'o', markersize = 5, color = 'k', ls='None')


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
#    pickle.dump(save_data, open("A.p", "wb"))
#    print('endede')
#    klklk
    

do_B = True #already ran, can just open files and read
if do_B:
    
    print('dob')
    
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
    se = SEA['data'][int(np.floor(xlen/2.0)-xval+delx+cutxtop):int(np.floor(xlen/2.0)+xval+delx-cutx),int(np.floor(ylen/2.0)-yval+dely):int(np.floor(ylen/2.0)+yval+dely)]
       
    ax001.imshow(se,cmap=cm.Greys_r)
    ax001.axis('off') 
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[0]))
    sbar = sb.AnchoredScaleBar(ax001.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4, my_fontsize = fsizenb)
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
    
    #relatively new code
#    labels_ws[labels_ws == 2] = 0 # delete this label
#    my_max = np.max(labels_ws)
#    labels_ws[labels_ws == my_max] = 2 # replace highest labels with labels 2
#    lab = np.unique(labels_ws)

   #old code, was working
    #Make random colors, not degrade
#    rand_ind = np.random.permutation(lab)
#    new_labels_ws = np.copy(labels_ws)
#    for k in range(new_labels_ws.shape[0]):
#        for j in range(new_labels_ws.shape[1]):
#            new_labels_ws[k, j] = rand_ind[labels_ws[k, j]]
#    labels_ws = new_labels_ws
        
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
    
    #new code to make black white
    #cut_labels_ws[cut_labels_ws > 1] = 1.0
    ax011.imshow(cut_labels_ws, cmap = cm.Greys_r) #or 'OrRd'
    
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
#    reda = REDA['data'][:,:,int(np.floor(xlen/2.0)-xval+delx+cutxtop):int(np.floor(xlen/2.0)+xval+delx-cutx),int(np.floor(ylen/2.0)-yval+dely):int(np.floor(ylen/2.0)+yval+dely)]
#    del REDA
#    gc.collect()
#    BLUEA = np.load('V0BluebrightB.npz')
#    bluea = BLUEA['data'][:,:,int(np.floor(xlen/2.0)-xval+delx+cutxtop):int(np.floor(xlen/2.0)+xval+delx-cutx),int(np.floor(ylen/2.0)-yval+dely):int(np.floor(ylen/2.0)+yval+dely)]   
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
    
    print(len(lab))
    print(non_cut_k)
    print(len(non_cut_k))
    for k in non_cut_k: #pickvec(lab): ######### change cut_k
     
#       print(k)
#       hlp = np.zeros(cut_labels_ws.shape)
#       hlp[cut_labels_ws == lab[k]] = 1.0
#       
#       #Visibility
#       print('Visib')
#       ureda  =  reda[:,backgroundinit:initbin,:,:]
#       ubluea =  bluea[:,backgroundinit:initbin,:,:]
#       
#       #visibilitty
#       No = np.sum(hlp.astype(np.float64))*reda.shape[0]*reda[:,backgroundinit:initbin,:,:].shape[1]
#       print(np.sum(hlp))
#       print(reda.shape[0])
#       print(reda[:,backgroundinit:initbin,:,:].shape[1])
#       print(No)
#       vec = np.nanmean(-(ureda-ubluea)/(ureda+ubluea) * hlp, axis = (0,1,2,3))
#       vecr = np.nanmean(ureda * hlp, axis = (0,1,2,3))
#       vecb = np.nanmean(ubluea * hlp, axis = (0,1,2,3))
#       print(vec)
#       vecstd = np.nanmean(np.sqrt(2)*np.sqrt(ureda**2 + ubluea**2)/np.sqrt((ureda+ubluea)**(3)) * hlp, axis = (0,1,2,3))
#       vecstdr =  np.sqrt(vecr)  * 1/np.sqrt(No) #np.nanmean(np.sqrt(ureda) * hlp, axis = (0,1,2,3))
#       vecstdb =  np.sqrt(vecb)*1/np.sqrt(No)  #np.nanmean(np.sqrt(ubluea) * hlp, axis = (0,1,2,3))
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
##       
#       del ureda, ubluea
#       gc.collect()
#       intens[k] = vec
#       stdintens[k] = vecstd
#       intensr[k] = vecr
#       stdintensr[k] = vecstdr
#       intensb[k] = vecb
#       stdintensb[k] = vecstdb
#       del vec, vecstd, vecr, vecb
#       gc.collect()
#       
#       print('Taus')
#       #Taus as a function of time
#       redd = np.average(reda[:,initbin:,:,:] * hlp, axis = (0,2,3))
#       blued = np.average(bluea[:,initbin:,:,:] * hlp, axis = (0,2,3))
#       print( np.sum(hlp.astype(np.float64)))
#       print(reda.shape[0])
#       Notr = np.sum(hlp.astype(np.float64))*reda.shape[0]
#       print(Notr)
#       hr = tauestimate(redd,np.sqrt(redd)/np.sqrt(Notr))
#       taured[k,:] = unumpy.nominal_values(hr)
#       tauredstd[k,:] = unumpy.std_devs(hr)
#       hb = tauestimate(blued,np.sqrt(blued)/np.sqrt(Notr))
#       taublue[k,:] = unumpy.nominal_values(hb)
#       taubluestd[k,:] = unumpy.std_devs(hb)
#       sizered = 1398 #redd.shape[0]
#       del hr, hb, redd, blued
#       gc.collect()
#               
       
#       print(unumpy.std_devs(taured[k,-1]))
#       print(unumpy.std_devs(taublue[k,-1]))
       #ax5100.errorbar(areas[k]*Pixel_size**2,taured[k,-1], yerr=tauredstd[k,-1], marker = 's', markersize = 8, color = 'k', ls='None')
       #ax510.errorbar(areas[k]*Pixel_size**2,taublue[k,-1], yerr=taubluestd[k,-1],marker = 's', markersize = 8, color = 'k', ls='None')
       #saved area is already multiplied by Pix^2
       #ax5100.errorbar(tor(areas[k]),taured[k,-1], yerr=0*tauredstd[k,-1], marker = 'H', markersize = 8, color = 'r', ls='None', markeredgewidth=0.0)
       #ax510.errorbar(tor(areas[k]),taublue[k,-1], yerr=0*taubluestd[k,-1],marker = 'H', markersize = 8, color = 'g', ls='None', markeredgewidth=0.0)
       ax5100.plot(tor(areas[k]),taured[k,-1],  marker = 'H', markersize = 10, color = 'r', ls='None', markeredgewidth=0.0)
       ax510.plot(tor(areas[k]),taublue[k,-1], marker = 'H', markersize = 10, color = 'g', ls='None', markeredgewidth=0.0)
      
#       colors = iter(cm.rainbow(np.linspace(0, 1, len(np.arange(0,sizered,step)))))      
#       for jj in np.arange(initind,sizered,step):
#           print(jj)
#           colorful = next(colors)
#           #ax5100b.plot(areas[k]*Pixel_size**2,taured[k,jj], marker = 's', markersize = 8, color = colorful, ls='None')
#           #ax510b.plot(areas[k]*Pixel_size**2,taublue[k,jj], marker = 's', markersize = 8, color = colorful, ls='None')
#           #saved area is already multiplied by Pix^2
#           ax5100b.plot(tor(areas[k]),taured[k,jj], marker = 'o', markersize = 8, color = colorful, ls='None')
#           ax510b.plot(tor(areas[k]),taublue[k,jj], marker = 'o', markersize = 8, color = colorful, ls='None')
#               
    ######### change cut_k
    #ax51.errorbar(areas[non_cut_k]*Pixel_size**2,intens[non_cut_k], yerr = stdintens[non_cut_k], marker = 's', markersize = 8, color = 'k', ls='None')
    #saved area is already multiplied by Pix^2
    #ax51.errorbar(tor(areas[non_cut_k]),intens[non_cut_k], yerr = stdintens[non_cut_k], marker = 'o', markersize = 8, color = 'k', ls='None')

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
#    print('donee')
#    klklk
    
   
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
   

    ## Make random colors, not degrade
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
        if (areas[k] < 100) or (areas[k] > 17000):
            cut_labels_ws[cut_labels_ws == lab[k]] = 0
            cut_k.append(k)
        else:
            non_cut_k.append(k)  ###change cut_k
    
    print(len(lab))
    print(len(cut_k))
    print(cut_labels_ws.shape)
    print(tor(areas*0.74*0.74))
    
    #cut_labels_ws[cut_labels_ws > 1] = 1.0
    ax0122.imshow(cut_labels_ws, cmap = cm.Greys_r) #or 'OrRd'
 
    ax0122.axis('off')

    del SEA#, se
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
    
    print('C visib')
    #print(intens[non_cut_k])
    #print(stdintens[non_cut_k])
    
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
    
    for k in non_cut_k: #pickvec(lab): #range(len(lab)):
     
#       print(k)
#       hlp = np.zeros(cut_labels_ws.shape)
#       hlp[cut_labels_ws == lab[k]] = 1.0
#       
#       #Visibility
#       print('Visib')
#       ureda  =  reda[:,backgroundinit:initbin,:,:]
#       ubluea =  bluea[:,backgroundinit:initbin,:,:]
#       No = np.sum(hlp.astype(np.float64))*reda[:,backgroundinit:initbin,:,:].shape[0]*reda[:,backgroundinit:initbin,:,:].shape[1]
#       #working
#       vec = np.nanmean(-(ureda-ubluea)/(ureda+ubluea) * hlp, axis = (0,1,2,3))
#       vecstd = np.nanmean(np.sqrt(2)*np.sqrt(ureda**2 + ubluea**2)/np.sqrt((ureda+ubluea)**(3)) * hlp, axis = (0,1,2,3))/np.sqrt(No)
#       #TEST
##       unr = unumpy.uarray(ureda,np.sqrt(ureda)/np.sqrt(No)) 
##       unb = unumpy.uarray(ubluea,np.sqrt(ubluea)/np.sqrt(No)) 
##       vec = unumpy.nominal_values(np.nanmean(-(unr-unb)/(unr+unb) * hlp, axis = (0,1,2,3)))
##       vecstd = unumpy.std_devs(np.nanmean(-(unr-unb)/(unr+unb) * hlp, axis = (0,1,2,3)))
#       print(vec)
#       print(vecstd)
#       vecr = np.nanmean(ureda * hlp, axis = (0,1,2,3))
#       vecb = np.nanmean(ubluea * hlp, axis = (0,1,2,3))
#       vecstdr = 1/np.sqrt(No) * np.sqrt(vecr) #np.nanmean(np.sqrt(ureda) * hlp, axis = (0,1,2,3))
#       vecstdb = 1/np.sqrt(No) * np.sqrt(vecb) #
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
#       Notr = np.sum(hlp.astype(np.float64))*reda[:,initbin:,:,:].shape[0]
#       hr = tauestimate(redd,np.sqrt(redd)/np.sqrt(Notr))
#       taured[k,:] = unumpy.nominal_values(hr)
#       tauredstd[k,:] = unumpy.std_devs(hr)
#       hb = tauestimate(blued,np.sqrt(blued)/np.sqrt(Notr))
#       taublue[k,:] = unumpy.nominal_values(hb)
#       taubluestd[k,:] = unumpy.std_devs(hb)
#       sizered = 1398 #redd.shape[0]
#       del hr, hb, redd, blued
#       gc.collect()
       
       #ax5100.errorbar(areas[k]*Pixel_size**2,taured[k,-1], yerr=tauredstd[k,-1],marker = '^', markersize = 11, color = 'k', ls='None')
       #ax510.errorbar(areas[k]*Pixel_size**2,taublue[k,-1],yerr=taubluestd[k,-1],marker = '^', markersize = 11, color = 'k', ls='None')
       #Saved area is already times Pix^2       
       #ax5100.errorbar(tor(areas[k]),taured[k,-1], yerr=0*tauredstd[k,-1],marker = 'H', markersize = 11, color = 'r', ls='None',zorder=2)
       #ax510.errorbar(tor(areas[k]),taublue[k,-1],yerr=0*taubluestd[k,-1],marker = 'H', markersize = 11, color = 'g', ls='None',zorder=2)
       
       ax5100.plot(tor(areas[k]),taured[k,-1], marker = 'H', markersize = 12, color = 'r', ls='None', markeredgewidth=0.0, zorder=2)
       ax510.plot(tor(areas[k]),taublue[k,-1], marker = 'H', markersize = 12, color = 'g', ls='None',markeredgewidth=0.0, zorder=2)
       
       
#       colors = iter(cm.rainbow(np.linspace(0, 1, len(np.arange(0,sizered,step)))))      
#       for jj in np.arange(initind,sizered,step):
#           print(jj)
#           colorful = next(colors)
#           #ax5100b.plot(areas[k]*Pixel_size**2,taured[k,jj], marker = '^', markersize = 11, color = colorful, ls='None')
#           #ax510b.plot(areas[k]*Pixel_size**2,taublue[k,jj], marker = '^', markersize = 11, color = colorful, ls='None')
#           #Saved area is already times Pix^2       
#           ax5100b.plot(tor(areas[k]),taured[k,jj], marker = 'o', markersize = 11, color = colorful, ls='None',zorder=2)
#           ax510b.plot(tor(areas[k]),taublue[k,jj], marker = 'o', markersize = 11, color = colorful, ls='None',zorder=2)
               
    #ax51.errorbar(areas[non_cut_k]*Pixel_size**2,intens[non_cut_k], yerr = stdintens[non_cut_k], marker = '^', markersize = 11, color = 'k', ls='None')
     #saved area is already multiplied by Pix^2
    
    #ax51.errorbar(tor(areas[non_cut_k]),intens[non_cut_k], yerr = stdintens[non_cut_k], marker = 'o', markersize = 12, color = 'k', ls='None',zorder=2)
     #0.2 is fake, is sybolic
    ax51.errorbar(2.25,intens[non_cut_k], yerr = stdintens[non_cut_k], marker = 'H', markersize = 12, color = 'k', ls='None',zorder=2)
    

#   
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
#    

##### Plot single and hexagon
    print('radius found for C')
    print(tor(areas[non_cut_k]))
    se_data2 = np.copy(se)    
    tv_filt = denoise_tv_chambolle(se_data2, weight=0.1) #WAS0.5
    grad_x, grad_y = np.gradient(tv_filt)
    new_pic_grad = np.sqrt(grad_x**2 + grad_y**2)
    new_pic_grad = canny(togs(new_pic_grad), sigma = 3) #WAS 2 #uncomment and run in Boes's compi
    hex_angle = 18
    hex_angle2 = 28 #18#35
    axpic.imshow(new_pic_grad,cmap=cm.Greys)
    axpic2.imshow(1-cut_labels_ws,cmap=cm.Greys) #WAS IMSHOWING new_pic_grad
    #axpic2.add_patch(patches.RegularPolygon((80, 70), 6, 60, fill = True, facecolor = '#ff6666', edgecolor='None', orientation = hex_angle * np.pi/180.0, alpha=0.5))
    print(tor(areas[non_cut_k]))
    #TO PLOT HEXAGON NEEDS TO muLTIPLY BY PIXEL AREA!!! IS IT?
    axpic2.add_patch(patches.RegularPolygon((80, 70), 6, tor(areas[non_cut_k])*0.74, fill = True, facecolor = '#ff6666', edgecolor='None', orientation = hex_angle2 * np.pi/180.0, alpha=0.5))
    axpic.axis('off')
    axpic2.axis('off')
   
    
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
    
#    print('D visib')
#    print(intens[non_cut_k])
#    print(stdintens[non_cut_k])
#    
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
     
#       print('indiseD')
#       print(k)
#       hlp = np.zeros(cut_labels_ws.shape)
#       hlp[cut_labels_ws == lab[k]] = 1.0
#       
#       #Visibility
#       print('Visib')
#       ureda  =  reda[:,backgroundinit:initbin,:,:]
#       ubluea =  bluea[:,backgroundinit:initbin,:,:]
#       vec = np.nanmean(-(ureda-ubluea)/(ureda+ubluea) * hlp, axis = (0,1,2,3))
#       vecr = np.nanmean(ureda * hlp, axis = (0,1,2,3))
#       vecb = np.nanmean(ubluea * hlp, axis = (0,1,2,3))
#       print(vec)
#       No = np.sum(hlp.astype(np.float64))*reda[:,backgroundinit:initbin,:,:].shape[0]*reda[:,backgroundinit:initbin,:,:].shape[1]
#       vecstd = np.nanmean(np.sqrt(2)*np.sqrt(ureda**2 + ubluea**2)/np.sqrt((ureda+ubluea)**(3)) * hlp, axis = (0,1,2,3))/np.sqrt(No)
#       vecstdr = 1/np.sqrt(No) * np.sqrt(vecr) #np.nanmean(np.sqrt(ureda) * hlp, axis = (0,1,2,3))
#       vecstdb = 1/np.sqrt(No) * np.sqrt(vecb) #
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
#       Notr = np.sum(hlp.astype(np.float64))*reda[:,initbin:,:,:].shape[0]
#       hr = tauestimate(redd,np.sqrt(redd)/np.sqrt(Notr))
#       taured[k,:] = unumpy.nominal_values(hr)
#       tauredstd[k,:] = unumpy.std_devs(hr)
#       hb = tauestimate(blued,np.sqrt(blued)/np.sqrt(Notr))
#       taublue[k,:] = unumpy.nominal_values(hb)
#       taubluestd[k,:] = unumpy.std_devs(hb)
#       sizered = 1398 #redd.shape[0]
#       del hr, hb, redd, blued
#       gc.collect()
#       
       print('area,tau of D')
       print(tor(areas[k]*0.79*0.79))
       print(taured[k,-1])
       ax5100.plot(tor(areas[k]),taured[k,-1], marker = 'x', markersize = 12, color = 'r', ls='None',zorder=2000)
       ax510.plot(tor(areas[k]),taublue[k,-1], marker = 'x', markersize = 12, color = 'g', ls='None',zorder=2000)
       
#       colors = iter(cm.rainbow(np.linspace(0, 1, len(np.arange(0,sizered,step)))))      
#       for jj in np.arange(initind,sizered,step):
#           print(jj)
#           colorful = next(colors)
#           ax5100b.plot(tor(areas[k])+0.4,taured[k,jj], marker = '*', markersize = 11, color = colorful, ls='None',zorder=2000)
#           ax510b.plot(tor(areas[k])+0.4,taublue[k,jj], marker = '*', markersize = 11, color = colorful, ls='None',zorder=2000)
#               
    #ax51.errorbar(tor(areas[non_cut_k]),intens[non_cut_k], yerr = stdintens[non_cut_k], marker = 'x', markersize = 12, color = 'k', ls='None',zorder=2000)
    #0.2 is fake, is sybolic
    #ax51.errorbar(2.25,intens[non_cut_k], yerr = stdintens[non_cut_k], marker = 'x', markersize = 12, color = 'k', ls='None',zorder=2000)
    ax51.errorbar(2.25,intens[non_cut_k], marker = 'x', markersize = 12, color = 'k', ls='None',zorder=2000)

   

#    save_data = {}
#    save_data['areas'] = areas*Pixel_size**2 #in nm^2 #in nm^2  AREA IS THE SAME AS SINGLE PARTICLE FROM C #IF saved at sAME Time than c, needs *PIxel**2!!!!!!!!!!!!!!!!!!!!!!
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
#    kkkjjj

plt.tight_layout()
# if jjj == 0:
#            ax1.text(55, 175, '   ' + 'time in\n 50 $\mu$s intervals', fontsize=fsizenb, va='center',ha='center')
#            ax1.annotate('', xy=(55,250), xytext=(55,200),
#                arrowprops=dict(facecolor='black', shrink=0.05))   
#            ax3.text(-0.1, 1.0, 'a', transform=ax3.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', bbox={'facecolor':'None', 'pad':5})
ax51.set_ylabel('Visibility of intensity (a.u.)',fontsize=fsizepl)
#ax51.set_xlabel('Nanoparticle diameter (nm)',fontsize=fsizepl)
ax51.set_ylim([-0.12,-0.08])
ax51.set_yticks([-0.11,-0.1,-0.09])
ax51.tick_params(labelsize=fsizenb)   
ax51.yaxis.set_ticks_position('right')

#fig1.delaxes(ax51)

ax510.set_ylabel(r'Green band $\tau$' + '\n after 1.4 ms acquisition ($\mu$s)',fontsize=fsizepl)
ax5100.set_ylabel(r'Red band $\tau$' + '\n after 1.4 ms acquisition ($\mu$s)',fontsize=fsizepl)
#ax510b.set_ylabel(r'Green band $\tau$ ($\mu$s)',fontsize=fsizepl)
#ax5100b.set_ylabel(r'Red band $\tau$ ($\mu$s)',fontsize=fsizepl)
ax510.set_xlabel('Nanoparticle diameter (nm)',fontsize=fsizepl)
ax510.set_yticks([100,200,300,400,500,600])
ax510.set_ylim([0,610])
ax510.set_xlim([36,64])
ax510.set_xticks([40,50,60])
ax5100.set_xlabel('Nanoparticle diameter (nm)',fontsize=fsizepl)
ax5100.set_yticks([100,200,300,400,500,600])
ax5100.set_ylim([0,610])
ax5100.set_xlim([36,64])
ax5100.set_xticks([40,50,60])
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
#ax510b.tick_params(labelsize=fsizenb)
#ax5100b.tick_params(labelsize=fsizenb)
#ax5100b.spines['left'].set_visible(False)
#ax5100b.spines['top'].set_visible(False)
#ax5100b.xaxis.set_ticks_position('bottom')
#ax5100b.yaxis.set_ticks_position('right')
#ax5100b.yaxis.set_label_position("right")
#ax510b.set_xlabel('Nanoparticle diameter (nm)',fontsize=fsizepl)
#ax5100b.set_xlabel('Nanoparticle diameter (nm)',fontsize=fsizepl)
#ax510b.set_xlim([36,64])
#ax510b.set_xticks([40,50,60])
#ax5100b.set_xlim([36,64])
#ax5100b.set_xticks([40,50,60])
#ax510b.set_yticks([100,200,300,400,500])
#ax510b.set_ylim([0,550])
#ax5100b.set_yticks([100,200,300,400,500])
#ax5100b.set_ylim([0,550])
#ax510b.spines['right'].set_visible(False)
#ax510b.spines['top'].set_visible(False)
#ax510b.xaxis.set_ticks_position('bottom')
#ax510b.yaxis.set_ticks_position('left')

#should be uncommented


if do_A:
    A = pickle.load( open( "A.p", "rb" ) )
#    A = np.load("Anpz.npz")
#    areas = A['areas']
#    taured = A['taured']
#    tauredstd = A['tauredstd']
#    taublue = A['taublue']
#    taubluestd = A['taubluestd']
#    intens = A['intens']
#    stdintens = A['stdintens']
#    intensr = A['intensr']
#    stdintensr = A['stdintensr']
#    intensb = A['intensb']
#    stdintensb = A['stdintensb']
#    non_cut_k = A['non_cut_k'] 
if do_B:
    B = pickle.load( open( "B.p", "rb" ) )
if do_C:
    C = pickle.load( open( "C.p", "rb" ) )
if do_D:
    D = pickle.load( open( "D.p", "rb" ) )
areafactor = 10.0

ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
#ax.yaxis.set_ticks_position('left')

#label = ax.xaxis.get_label()
#x_lab_pos, y_lab_pos = label.get_position()
#label.set_position([1.0, y_lab_pos])
#label.set_horizontalalignment('left')
ax.set_xlabel(r'$\hspace*{-0.5cm}$' + 'Average intensity per pixel (kHz)',fontsize=fsizepl ,horizontalalignment='left', position=(0.07,25))#,multialignment='left')
#set_horizontalalignment('right')
#ticklab = ax.xaxis.get_ticklabels()[0]
#trans = ticklab.get_transform()
#ax.xaxis.set_label_coords(0.5, 0.1, transform=trans)
ax51.tick_params(labelsize=fsizenb) 
ax.tick_params(labelsize=fsizenb) 

#ax.yaxis.set_ticks_position('left')
#ax.yaxis.set_label_position("left")

#if do_A:
#    #green t, different areas, circleA
#    ax.scatter(A['intensb'][A['non_cut_k']],A['taublue'][A['non_cut_k'],-1],s=A['areas'][A['non_cut_k']]/areafactor, color='g', alpha=0.5, marker = 'o')
#if do_B:
#    #green t, different areas, squareB
#    ax.scatter(B['intensb'][B['non_cut_k']],B['taublue'][B['non_cut_k'],-1],s=B['areas'][B['non_cut_k']]/areafactor, color='g', alpha=0.5, marker = 's')
if do_C:
    #green t, different areas, triangleC
    ax.plot(C['intensb'][C['non_cut_k']],C['taublue'][C['non_cut_k'],-1], color='g', markeredgewidth=0.0,marker = 'H', markersize = 12)
if do_D:
    #green t, different areas, triangleC
    print(D['intensb'][D['non_cut_k']])
    print(D['taublue'][D['non_cut_k'],-1])
    ax.plot(D['intensb'][D['non_cut_k']],D['taublue'][D['non_cut_k'],-1], color='g', marker = 'x', markersize = 12)

##axp.spines['left'].set_visible(False)
##axp.spines['top'].set_visible(False)
##axp.xaxis.set_ticks_position('bottom')
##axp.yaxis.set_ticks_position('right')
##axp.set_ylabel(r'Red band $\tau$' + '\n after 1.4 ms of acquisition ($\mu$s)',fontsize=fsizepl)
##axp.set_xlabel('Average intensity per pixel (kHz)',fontsize=fsizepl)
##axp.tick_params(labelsize=fsizenb) 
###axp.yaxis.set_label_position("right")
#if do_A:
#    #red t, different areas, circleA
#    axp.scatter(A['intensr'][A['non_cut_k']],A['taured'][A['non_cut_k'],-1],s=A['areas'][A['non_cut_k']]/areafactor, color='r', alpha=0.5, marker = 'o')
#if do_B:
#    #red t, different areas, squareB
#    axp.scatter(B['intensr'][B['non_cut_k']],B['taured'][B['non_cut_k'],-1],s=B['areas'][B['non_cut_k']]/areafactor, color='r', alpha=0.5, marker = 's')
if do_C:
    #red t, different areas, triangleC
    ax.plot(C['intensr'][C['non_cut_k']],C['taured'][C['non_cut_k'],-1], color='r', marker = 'H', markersize = 12, markeredgewidth=0.0)
if do_D:
    #red t, different areas, triangleC
    print(D['intensr'][D['non_cut_k']])
    print(D['taured'][D['non_cut_k'],-1])
    ax.plot(D['intensr'][D['non_cut_k']],D['taured'][D['non_cut_k'],-1], color='r', marker = 'x', markersize = 12)
#uncommented until here
ax.set_ylabel(r'$\tau$ after 1.4 ms of acquisition ($\mu$s)',fontsize=fsizepl)
ax.set_yticks([530,550,570])
ax.set_xlim([0.4,2.5])
ax.set_xticks([0.5,1,1.5,2])
 
# my_rect2 =  patches.Rectangle(
#         (2.1, 515),
#         0.4,
#         10,
#         facecolor = 'k', alpha = 1.0, zorder = 100
#     )
# 
# my_rect2.set_clip_on(False)
# 
# ax.add_patch(my_rect2)
# 
# my_rect =  patches.Rectangle(
#         (2.1, 515),
#         0.4,
#         10,
#         fill=True,
#         facecolor = 'b', alpha = 1.0, zorder = 200
#     )
# 
# my_rect.set_clip_on(False)
# 
# ax.add_patch(my_rect)
 
my_rect1 =  patches.Rectangle(
        (2.1, -0.1225),
        0.399,
        0.005,
        fill=True,
        edgecolor="none",
        facecolor = 'w', alpha = 1.0, zorder = 1000
    )

my_rect1.set_clip_on(False)


ax51.add_patch(my_rect1)




#ax.annotate('', xy=(2.1, 520), xytext=(2.5, 520),
#                arrowprops=dict(facecolor='white', shrink=0.05), zorder = 1000000) 

#ax.set_xlim(0.5,2.0)
#ax51.set_xlim(2.1,2.5)
# hide the spines between ax and ax2
#ax.spines['right'].set_visible(False)
#ax51.spines['left'].set_visible(False)
#ax.yaxis.tick_left()
#ax.tick_params(labelright='off')
#ax51.yaxis.tick_right()
 

plt.tight_layout()

   
multipage_longer('Single.pdf',dpi=80)    
#plt.show()
klklklk
