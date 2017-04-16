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

from boe_segment import give_bolinha

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
tauindice = 250 ##-1: takes last tau, after 1.4ms of acquisition
###
def tauestimate(counts_red, error_red):
    
    print(counts_red.shape[1])
    
    ucounts_red = unumpy.uarray(counts_red, error_red)
    
    def helper(arrayx):
         arrayx[arrayx < 1e-12] = 1e-12   #so that no division by zero     
         return np.cumsum(arrayx*np.arange(1,counts_red.shape[1]+1), axis = 1)/np.cumsum(arrayx, axis = 1)
    
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

nolines = 10
noplots = 16

ax00 = plt.subplot2grid((nolines,noplots), (0,0), colspan=1, rowspan=1)
ax03 = plt.subplot2grid((nolines,noplots), (0,1), colspan=1, rowspan=1)
ax00.text(-0.125, 1.0, 'a', transform=ax00.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})

ax001 = plt.subplot2grid((nolines,noplots), (1,0), colspan=1, rowspan=1)
ax011 = plt.subplot2grid((nolines,noplots), (1,1), colspan=1, rowspan=1)
ax001.text(-0.1, 1.0, 'b', transform=ax001.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})

ax0022 = plt.subplot2grid((nolines,noplots), (2,0), colspan=1, rowspan=1)
axpic = plt.subplot2grid((nolines,noplots), (2,1), colspan=1, rowspan=1)
ax0022.text(-0.1, 1.0, 'c', transform=ax0022.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})
         
ax0022b = plt.subplot2grid((nolines,noplots), (3,0), colspan=1, rowspan=1)
axpicb = plt.subplot2grid((nolines,noplots), (3,1), colspan=1, rowspan=1)
ax0022b.text(-0.1, 1.0, 'd', transform=ax0022b.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})
         
axratiomag = plt.subplot2grid((nolines,noplots), (4,0), colspan=2, rowspan=2)
axtaumag = plt.subplot2grid((nolines,noplots), (6,0), colspan=2, rowspan=2)
axstdmag = plt.subplot2grid((nolines,noplots), (8,0), colspan=2, rowspan=2)

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
    
    SEA= np.load('x150SEchannel.npz') #init shape (342, 315)
    xlen = SEA['data'].shape[0]
    ylen = SEA['data'].shape[1]
    xinit = 42
    xfinal = -42
    yinit = 15
    yfinal = -15
    se = SEA['data'][xinit:xfinal,yinit:yfinal]
    
    new_pic = give_bolinha('x150SEchannel.npz', xinit, yinit, xfinal, yfinal, corr_threshold = 0.35, n = 30, r = 5, save_file = False, do_plot = False)
    cutx = 50
    cuty = 1
    se = se[cutx:-cutx, cuty:-cuty]

    ax00.imshow(se,cmap=cm.Greys_r)
    ax00.axis('off')
    
    new_pic = new_pic[cutx:-cutx, cuty:-cuty]
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
    
    local_maxi = peak_local_max(
        distance, 
        num_peaks = 50, 
        indices = False, 
        footprint = np.ones((27,27)),
        labels = hlpse2) #footprint = min dist between maxima to find #footprint was 25,25
    markers = skimage.morphology.label(local_maxi)
    labels_ws = watershed(-distance, markers, mask=hlpse2)
    lab = np.unique(labels_ws)
    
    # Make random colors, not degrade
#    rand_ind = np.random.permutation(lab)
#    new_labels_ws = np.copy(labels_ws)
#    for k in range(new_labels_ws.shape[0]):
#        for j in range(new_labels_ws.shape[1]):
#            new_labels_ws[k, j] = rand_ind[labels_ws[k, j]]
#    labels_ws =  new_labels_ws
    
    Pixel_size = np.array([2.5]) 
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[0]))
    sbar = sb.AnchoredScaleBar(ax00.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4, my_fontsize = fsizenb)
    ax00.add_artist(sbar)
    
    areas = np.array([])
    for k in lab:
        areas = np.append(areas, len( labels_ws[labels_ws == k] ))
    cut_k = []
    cut_labels_ws = np.copy(labels_ws)
    non_cut_k = []  ###### change cut_k
    for k in range(len(lab)):
        if (areas[k] < 10) or (areas[k] > 4000):
            cut_labels_ws[cut_labels_ws == lab[k]] = 0
            cut_k.append(k)
        else:
            non_cut_k.append(k)  ###change cut_k
    
    #print(len(lab))
    #print(len(non_cut_k))
    #ax01.imshow(new_pic_grad,cmap=cm.Greys_r) #or 'OrRd'
    #ax01.axis('off')
    #ax02.imshow(masklarge,cmap=cm.Greys_r) #or 'OrRd'
    #ax02.axis('off')
   
    #code to make all black - this will crash if running to save data
    #cut_labels_ws[cut_labels_ws > 1] = 1.0
    
    ax03.imshow(cut_labels_ws, cmap = cm.Greys_r) #or 'OrRd'
    ax03.axis('off')

    del SEA, se
    gc.collect()
    
        ####### load file that exists
#    C = pickle.load( open( "150.p", "rb" ) )
#    areas = C['areas']
#    taured = C['taured']
#    tauredstd = C['tauredstd']
#    taublue = C['taublue']
#    taubluestd = C['taubluestd']
#    intens = C['intens']
#    stdintens = C['intensstd']
#    intensr = C['intensr']
#    stdintensr = C['intensstdr']
#    intensb = C['intensb']
#    stdintensb = C['intensstdb']
#    non_cut_k = C['non_cut_k']  
    
    
    REDA = np.load('x150Redbright.npz')
    reda = REDA['data'][:,:,xinit+cutx:xfinal-cutx,yinit+cuty:yfinal-cuty] #same no pixels than C, single
    del REDA
    gc.collect()
    BLUEA = np.load('x150Bluebright.npz')
    bluea = BLUEA['data'][:,:,xinit+cutx:xfinal-cutx,yinit+cuty:yfinal-cuty]#same no pixels than C, single
    del BLUEA
    gc.collect() 
    
    no_avg = reda.shape[0]
    intens = np.empty([len(lab),no_avg])
    stdintens = np.empty([len(lab),no_avg])
    intensr = np.empty([len(lab),no_avg])
    stdintensr = np.empty([len(lab),no_avg])
    intensb = np.empty([len(lab),no_avg])
    stdintensb = np.empty([len(lab),no_avg])
    
    taured = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
    taublue = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
    tauredstd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
    taubluestd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
    
    for k in non_cut_k:
     
       print('indice150')
       print(k)
       
       #Ratio
       print('Ratio')
       hlp = np.zeros(cut_labels_ws.shape)
       hlp[cut_labels_ws == lab[k]] = 1.0
       hlp[cut_labels_ws != lab[k]] = np.nan
       ureda  =  reda[:,backgroundinit:initbin,:,:]
       ubluea =  bluea[:,backgroundinit:initbin,:,:]
                     
       vecr = np.nanmean(ureda * hlp, axis = (1,2,3))
       vecb = np.nanmean(ubluea * hlp, axis = (1,2,3))
       
       vec = vecr/vecb
       
       No = np.nansum(hlp.astype(np.float64))*reda[:,backgroundinit:initbin,:,:].shape[1]
              
       #import IPython
       #IPython.embed()
       
       vecstd = vec * np.sqrt( 1.0/vecr/No + 1.0/vecb/No )
       
       vecstdr = 1.0/np.sqrt(No) * np.sqrt(vecr) 
       vecstdb = 1.0/np.sqrt(No) * np.sqrt(vecb) 
       del ureda, ubluea
       gc.collect()
       intens[k,:] = vec
       stdintens[k,:] = vecstd
       intensr[k,:] = vecr
       stdintensr[k,:] = vecstdr
       intensb[k,:] = vecb
       stdintensb[k,:] = vecstdb
       print(vec)
       print(vecstd)
       del vec, vecstd
       gc.collect()
       
     
       
       print('Taus')
       #Taus as a function of time
       hlp = np.zeros(cut_labels_ws.shape)
       hlp[cut_labels_ws == lab[k]] = 1.0
       Notr = np.sum(hlp.astype(np.float64))
       redd = np.sum(reda[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
       blued = np.sum(bluea[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
       hr = tauestimate(redd,np.sqrt(redd)/np.sqrt(Notr))
       taured[k,:,:] = unumpy.nominal_values(hr)
       tauredstd[k,:,:] = unumpy.std_devs(hr)
       hb = tauestimate(blued,np.sqrt(blued)/np.sqrt(Notr))
       taublue[k,:,:] = unumpy.nominal_values(hb)
       taubluestd[k,:,:] = unumpy.std_devs(hb)
       sizered = 1398 
       del hr, hb, redd, blued
       gc.collect()
       
           #CONSIDER TAKING AVG OF NON ZERO ONES
#           aux = np.copy(intens[k,:])
#           aux[aux < 1.0e-12] = np.nan
#           axratiomag.errorbar(2,np.nanmean(aux),yerr=np.nanstd(aux),marker='o', color='k', markersize=12) 
#           axratiomag.set_xlim([0,10])
#           axtaumag.plot(np.arange(0,1398),np.average(taured[k,:,:],axis=0),lw=1,color='r')
#           axtaumag.plot(np.arange(0,1398),np.average(taublue[k,:,:],axis=0),lw=1,color='g')
#           axstdmag.plot(np.arange(0,1398),np.std(taured[k,:,:],axis=0)/np.average(taured[k,:,:],axis=0),lw=1,color='r')
#           axstdmag.plot(np.arange(0,1398),np.std(taublue[k,:,:],axis=0)/np.average(taublue[k,:,:],axis=0),lw=1,color='g')
    
          
    del reda, bluea
    gc.collect()
    
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
    
    del taured, taublue
    gc.collect()
    
    pickle.dump(save_data, open("150.p", "wb"))   
    
    print('radius found for C') #to check which of the found areas is background, which is signal
#    print(tor(areas[non_cut_k])) #if used saved areas
    print(tor(areas*Pixel_size**2))
    print(non_cut_k)

do_B = False #already ran, can just open files and read
if do_B:
    
    print('dob')
    
    length_scalebar = 100.0
    Pixel_size = np.array([1.25]) 
    
    SEA= np.load('x300SEchannel.npz') #init shape (342, 315)
    xlen = SEA['data'].shape[0]
    ylen = SEA['data'].shape[1]
    xinit = 27
    xfinal = -27
    yinit = 29
    yfinal = -29
    se = SEA['data'][xinit:xfinal,yinit:yfinal]
    
    new_pic = give_bolinha('x300SEchannel.npz', xinit, yinit, xfinal, yfinal, corr_threshold = 0.35, n = 30, r = 8, save_file = False, do_plot = False)
    cutx = 50
    cuty = 1
    se = se[0:-cutx, cuty:-cuty]

    ax001.imshow(se,cmap=cm.Greys_r)
    ax001.axis('off')
    
    #new_pic = new_pic[cutx:-cutx, cuty:-cuty]
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
    print(length_scalebar_in_pixels)
    sbar = sb.AnchoredScaleBar(ax001.transData, length_scalebar_in_pixels, '', style = 'dark', loc = 4, my_fontsize = fsizenb)
    ax001.add_artist(sbar)
    
    areas = np.array([])
    for k in lab:
        areas = np.append(areas, len( labels_ws[labels_ws == k] ))
    cut_k = []
    cut_labels_ws = np.copy(labels_ws)
    non_cut_k = []  ###### change cut_k
    for k in range(len(lab)):
        if (areas[k] < 10) or (areas[k] > 4000) or (k == 0): #k==o is modif!!! k==0 point is WEIRD
            cut_labels_ws[cut_labels_ws == lab[k]] = 0
            cut_k.append(k)
        else:
            non_cut_k.append(k)  ###change cut_k
    
    #print(len(lab))
    #print(len(non_cut_k))
    #ax01.imshow(new_pic_grad,cmap=cm.Greys_r) #or 'OrRd'
    #ax01.axis('off')
    #ax02.imshow(masklarge,cmap=cm.Greys_r) #or 'OrRd'
    #ax02.axis('off')
   
    #code to make all black - this will crash if running to save data
    #cut_labels_ws[cut_labels_ws > 1] = 1.0
    
    ax011.imshow(cut_labels_ws, cmap = cm.Greys_r) #or 'OrRd'
    ax011.axis('off')

    del SEA, se
    gc.collect()
    
   
    
   
#### C
do_C = False #ran already - can just load data
if do_C:
    
    print('doc')
    
    length_scalebar = 100.0
    Pixel_size = np.array([0.83]) 
    
    SEA= np.load('x450SEchannel.npz') #init shape (342, 315)
    xlen = SEA['data'].shape[0]
    ylen = SEA['data'].shape[1]
    xinit = 200
    xfinal = -200
    yinit = 41
    yfinal = -41
    se = SEA['data'][xinit:xfinal,yinit:yfinal]
    
    new_pic = give_bolinha('x450SEchannel.npz', xinit, yinit, xfinal, yfinal, corr_threshold = 0.35, n = 30, r = 8, save_file = False, do_plot = False)
#    cutx = 50
#    cuty = 1
#    se = se[0:-cutx, cuty:-cuty]

    ax0022.imshow(se,cmap=cm.Greys_r)
    ax0022.axis('off')
    
    #new_pic = new_pic[cutx:-cutx, cuty:-cuty]
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
    sbar = sb.AnchoredScaleBar(ax0022.transData, length_scalebar_in_pixels, '', style = 'bright', loc = 4, my_fontsize = fsizenb)
    ax0022.add_artist(sbar)
    
    areas = np.array([])
    for k in lab:
        areas = np.append(areas, len( labels_ws[labels_ws == k] ))
    cut_k = []
    cut_labels_ws = np.copy(labels_ws)
    non_cut_k = []  ###### change cut_k
    for k in range(len(lab)):
        if (areas[k] < 10) or (areas[k] > 20000): # or (k == 0): #k==o is modif!!! k==0 point is WEIRD
            cut_labels_ws[cut_labels_ws == lab[k]] = 0
            cut_k.append(k)
        else:
            non_cut_k.append(k)  ###change cut_k
    
    #print(len(lab))
    #print(len(non_cut_k))
    #ax01.imshow(new_pic_grad,cmap=cm.Greys_r) #or 'OrRd'
    #ax01.axis('off')
    #ax02.imshow(masklarge,cmap=cm.Greys_r) #or 'OrRd'
    #ax02.axis('off')
   
    #code to make all black - this will crash if running to save data
    #cut_labels_ws[cut_labels_ws > 1] = 1.0
    
    axpic.imshow(cut_labels_ws, cmap = cm.Greys_r) #or 'OrRd'
    axpic.axis('off')

    del SEA, se
    gc.collect()
    
    
#### C
do_750= True #ran already - can just load data
if do_750:
    
    print('do750')
    
    length_scalebar = 100.0
    Pixel_size = np.array([1.48]) 
    
    SEA= np.load('x750SEchannel.npz') #init shape (342, 315)
    xlen = SEA['data'].shape[0]
    ylen = SEA['data'].shape[1]
    xinit = 17
    xfinal = -17
    yinit = 18
    yfinal = -18
    se = SEA['data'][xinit:xfinal,yinit:yfinal]
    
    new_pic = give_bolinha('x750SEchannel.npz', xinit, yinit, xfinal, yfinal, corr_threshold = 0.35, n = 30, r = 8, save_file = False, do_plot = False)
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
    sbar = sb.AnchoredScaleBar(ax0022b.transData, length_scalebar_in_pixels, "", style = 'bright', loc = 8, my_fontsize = fsizenb)
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
#    C = pickle.load( open( "750.p", "rb" ) )
#    areas = C['areas']
#    taured = C['taured']
#    tauredstd = C['tauredstd']
#    taublue = C['taublue']
#    taubluestd = C['taubluestd']
#    intens = C['intens']
#    stdintens = C['intensstd']
#    intensr = C['intensr']
#    stdintensr = C['intensstdr']
#    intensb = C['intensb']
#    stdintensb = C['intensstdb']
#    non_cut_k = C['non_cut_k']  
    
    
    REDA = np.load('x750Redbright.npz')
    reda = REDA['data'][:,:,xinit:xfinal,yinit:yfinal] #same no pixels than C, single
    del REDA
    gc.collect()
    BLUEA = np.load('x750Bluebright.npz')
    bluea = BLUEA['data'][:,:,xinit:xfinal,yinit:yfinal]#same no pixels than C, single
    del BLUEA
    gc.collect() 
    
    no_avg = reda.shape[0]
    intens = np.empty([len(lab),no_avg])
    stdintens = np.empty([len(lab),no_avg])
    intensr = np.empty([len(lab),no_avg])
    stdintensr = np.empty([len(lab),no_avg])
    intensb = np.empty([len(lab),no_avg])
    stdintensb = np.empty([len(lab),no_avg])
    
    taured = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
    taublue = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
    tauredstd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
    taubluestd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
    
    for k in non_cut_k:
     
       print('indice750')
       print(k)
       
       #Ratio
       print('Ratio')
       hlp = np.zeros(cut_labels_ws.shape)
       hlp[cut_labels_ws == lab[k]] = 1.0
       hlp[cut_labels_ws != lab[k]] = np.nan
       ureda  =  reda[:,backgroundinit:initbin,:,:]
       ubluea =  bluea[:,backgroundinit:initbin,:,:]
                     
       vecr = np.nanmean(ureda * hlp, axis = (1,2,3))
       vecb = np.nanmean(ubluea * hlp, axis = (1,2,3))
       
       vec = vecr/vecb
       
       No = np.nansum(hlp.astype(np.float64))*reda[:,backgroundinit:initbin,:,:].shape[1]
              
       #import IPython
       #IPython.embed()
       
       vecstd = vec * np.sqrt( 1.0/vecr/No + 1.0/vecb/No )
       
       vecstdr = 1.0/np.sqrt(No) * np.sqrt(vecr) 
       vecstdb = 1.0/np.sqrt(No) * np.sqrt(vecb) 
       del ureda, ubluea
       gc.collect()
       intens[k,:] = vec
       stdintens[k,:] = vecstd
       intensr[k,:] = vecr
       stdintensr[k,:] = vecstdr
       intensb[k,:] = vecb
       stdintensb[k,:] = vecstdb
       print(vec)
       print(vecstd)
       del vec, vecstd
       gc.collect()
       
     
       
       print('Taus')
       #Taus as a function of time
       hlp = np.zeros(cut_labels_ws.shape)
       hlp[cut_labels_ws == lab[k]] = 1.0
       Notr = np.sum(hlp.astype(np.float64))
       redd = np.sum(reda[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
       blued = np.sum(bluea[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
       hr = tauestimate(redd,np.sqrt(redd)/np.sqrt(Notr))
       taured[k,:,:] = unumpy.nominal_values(hr)
       tauredstd[k,:,:] = unumpy.std_devs(hr)
       hb = tauestimate(blued,np.sqrt(blued)/np.sqrt(Notr))
       taublue[k,:,:] = unumpy.nominal_values(hb)
       taubluestd[k,:,:] = unumpy.std_devs(hb)
       sizered = 1398 
       del hr, hb, redd, blued
       gc.collect()
       
#           #CONSIDER TAKING AVG OF NON ZERO ONES
#           aux = np.copy(intens[k,:])
#           aux[aux < 1.0e-12] = np.nan
#           axratiomag.errorbar(1,np.nanmean(aux),yerr=np.nanstd(aux),marker='o', color='k', markersize=12) 
#           axratiomag.set_xlim([0,10])
#           axtaumag.plot(np.arange(0,1398),np.average(taured[k,:,:],axis=0),lw=1,color='r')
#           axtaumag.plot(np.arange(0,1398),np.average(taublue[k,:,:],axis=0),lw=1,color='g')
#           axstdmag.plot(np.arange(0,1398),np.std(taured[k,:,:],axis=0)/np.average(taured[k,:,:],axis=0),lw=1,color='r')
#           axstdmag.plot(np.arange(0,1398),np.std(taublue[k,:,:],axis=0)/np.average(taublue[k,:,:],axis=0),lw=1,color='g')
    
          
    del reda, bluea
    gc.collect()
    
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
    
    del taured, taublue
    gc.collect()
    
    pickle.dump(save_data, open("750.p", "wb"))   
    
    print('radius found for C') #to check which of the found areas is background, which is signal
#    print(tor(areas[non_cut_k])) #if used saved areas
    print(tor(areas*Pixel_size**2))
    print(non_cut_k)

plt.show()
lklk
          
do_750BG= True #ran already - can just load data
if do_750BG:
    
    print('do750BG')
    
    length_scalebar = 100.0
    Pixel_size = np.array([1.48]) 
    
    SEA= np.load('x750SEchannel.npz') #init shape (342, 315)
    xlen = SEA['data'].shape[0]
    ylen = SEA['data'].shape[1]
    xinit = 17
    xfinal = -17
    yinit = 18
    yfinal = -18
    se = SEA['data'][xinit:xfinal,yinit:yfinal]
    
    new_pic = give_bolinha('x750SEchannel.npz', xinit, yinit, xfinal, yfinal, corr_threshold = 0.35, n = 30, r = 8, save_file = False, do_plot = False)
#    cutx = 20
#    cuty = 1
    #se = se[0:-cutx, cuty:-cuty]

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
    sbar = sb.AnchoredScaleBar(ax0022b.transData, length_scalebar_in_pixels, "", style = 'bright', loc = 8, my_fontsize = fsizenb)
    ax0022b.add_artist(sbar)
    
    areas = np.array([])
    for k in lab:
        areas = np.append(areas, len( labels_ws[labels_ws == k] ))
    cut_k = []
    cut_labels_ws = np.copy(labels_ws)
    non_cut_k = []  ###### change cut_k
    for k in range(len(lab)):
        if (areas[k] < 4000):# or (k == 0): 
            cut_labels_ws[cut_labels_ws == lab[k]] = 0
            cut_k.append(k)
        else:
            non_cut_k.append(k)  ###change cut_k

    del SEA, se
    gc.collect()
    
    ####### load file that exists
#    C = pickle.load( open( "750BG.p", "rb" ) )
#    areas = C['areas']
#    taured = C['taured']
#    tauredstd = C['tauredstd']
#    taublue = C['taublue']
#    taubluestd = C['taubluestd']
#    intens = C['intens']
#    stdintens = C['intensstd']
#    intensr = C['intensr']
#    stdintensr = C['intensstdr']
#    intensb = C['intensb']
#    stdintensb = C['intensstdb']
#    non_cut_k = C['non_cut_k']  
#    
#    
    REDA = np.load('x750Redbright.npz')
    reda = REDA['data'][:,:,xinit:xfinal,yinit:yfinal] #same no pixels than C, single
    del REDA
    gc.collect()
    BLUEA = np.load('x750Bluebright.npz')
    bluea = BLUEA['data'][:,:,xinit:xfinal,yinit:yfinal]#same no pixels than C, single
    del BLUEA
    gc.collect() 
    
    no_avg = reda.shape[0]
    intens = np.empty([len(lab),no_avg])
    stdintens = np.empty([len(lab),no_avg])
    intensr = np.empty([len(lab),no_avg])
    stdintensr = np.empty([len(lab),no_avg])
    intensb = np.empty([len(lab),no_avg])
    stdintensb = np.empty([len(lab),no_avg])
    
    taured = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
    taublue = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
    tauredstd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
    taubluestd = np.empty([len(lab),no_avg,reda.shape[1]-initbin])
    
    for k in non_cut_k:
     
       print('indice750')
       print(k)
       
#Ratio
       print('Ratio')
       hlp = np.zeros(cut_labels_ws.shape)
       hlp[cut_labels_ws == lab[k]] = 1.0
       hlp[cut_labels_ws != lab[k]] = np.nan
       ureda  =  reda[:,backgroundinit:initbin,:,:]
       ubluea =  bluea[:,backgroundinit:initbin,:,:]
                     
       vecr = np.nanmean(ureda * hlp, axis = (1,2,3))
       vecb = np.nanmean(ubluea * hlp, axis = (1,2,3))
       
       vec = vecr/vecb
       
       No = np.nansum(hlp.astype(np.float64))*reda[:,backgroundinit:initbin,:,:].shape[1]
              
       #import IPython
       #IPython.embed()
       
       vecstd = vec * np.sqrt( 1.0/vecr/No + 1.0/vecb/No )
       
       vecstdr = 1.0/np.sqrt(No) * np.sqrt(vecr) 
       vecstdb = 1.0/np.sqrt(No) * np.sqrt(vecb) 
       del ureda, ubluea
       gc.collect()
       intens[k,:] = vec
       stdintens[k,:] = vecstd
       intensr[k,:] = vecr
       stdintensr[k,:] = vecstdr
       intensb[k,:] = vecb
       stdintensb[k,:] = vecstdb
       print(vec)
       print(vecstd)
       del vec, vecstd
       gc.collect()
       
       print('Taus')
       #Taus as a function of time
       hlp = np.zeros(cut_labels_ws.shape)
       hlp[cut_labels_ws == lab[k]] = 1.0
       Notr = np.sum(hlp.astype(np.float64))
       redd = np.sum(reda[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
       blued = np.sum(bluea[:,initbin:,:,:] * hlp, axis = (2,3))/Notr
       hr = tauestimate(redd,np.sqrt(redd)/np.sqrt(Notr))
       taured[k,:,:] = unumpy.nominal_values(hr)
       tauredstd[k,:,:] = unumpy.std_devs(hr)
       hb = tauestimate(blued,np.sqrt(blued)/np.sqrt(Notr))
       taublue[k,:,:] = unumpy.nominal_values(hb)
       taubluestd[k,:,:] = unumpy.std_devs(hb)
       sizered = 1398 
       del hr, hb, redd, blued
       gc.collect()
       
#           #CONSIDER TAKING AVG OF NON ZERO ONES
#           aux = np.copy(intens[k,:])
#           aux[aux < 1.0e-12] = np.nan
#           axratiomag.errorbar(1,np.nanmean(aux),yerr=np.nanstd(aux),marker='x', color='k', markersize=12) 
#           axratiomag.set_xlim([0,10])
#           axtaumag.plot(np.arange(0,1398),np.average(taured[k,:,:],axis=0),lw=1,color='DarkRed')
#           axtaumag.plot(np.arange(0,1398),np.average(taublue[k,:,:],axis=0),lw=1,color='DarkGreen')
#           axstdmag.plot(np.arange(0,1398),np.std(taured[k,:,:],axis=0)/np.average(taured[k,:,:],axis=0),lw=1,color='DarkRed')
#           axstdmag.plot(np.arange(0,1398),np.std(taublue[k,:,:],axis=0)/np.average(taublue[k,:,:],axis=0),lw=1,color='DarkGreen')
    
          
    del reda, bluea
    gc.collect()
    
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
    
    del taured, taublue
    gc.collect()
    
    pickle.dump(save_data, open("750BG.p", "wb"))   
    
    print('radius found for C') #to check which of the found areas is background, which is signal
#    print(tor(areas[non_cut_k])) #if used saved areas
    print(tor(areas*Pixel_size**2))
    print(non_cut_k)
    
##### Plot single and hexagon
#   print('radius found for C')
#   print(tor(areas[non_cut_k]))
#    se_data2 = np.copy(se)    
#    tv_filt = denoise_tv_chambolle(se_data2, weight=0.1) #WAS0.5
#    grad_x, grad_y = np.gradient(tv_filt)
#    new_pic_grad = np.sqrt(grad_x**2 + grad_y**2)
#    new_pic_grad = canny(togs(new_pic_grad), sigma = 3) #WAS 2 #uncomment and run in Boes's compi
#    hex_angle = 18
#    hex_angle2 = 28 #18#35
#    axpic.imshow(new_pic_grad,cmap=cm.Greys)
#    axpic2.imshow(1-cut_labels_ws,cmap=cm.Greys) #WAS IMSHOWING new_pic_grad
#    #axpic2.add_patch(patches.RegularPolygon((80, 70), 6, 60, fill = True, facecolor = '#ff6666', edgecolor='None', orientation = hex_angle * np.pi/180.0, alpha=0.5))
#    print(tor(areas[non_cut_k]))
#    #TO PLOT HEXAGON NEEDS TO muLTIPLY BY PIXEL AREA!!! IS IT?
#    axpic2.add_patch(patches.RegularPolygon((80, 70), 6, tor(areas[non_cut_k])*0.74, fill = True, facecolor = '#ff6666', edgecolor='None', orientation = hex_angle2 * np.pi/180.0, alpha=0.5))
#    axpic.axis('off')
#    axpic2.axis('off')
   
   
plt.show()
lkl

plt.tight_layout()
ax51.set_ylabel('Visibility of intensity (a.u.)',fontsize=fsizepl)

ax51.set_ylim([-0.12,-0.08])
ax51.set_yticks([-0.11,-0.1,-0.09])
ax51.tick_params(labelsize=fsizenb)   
ax51.yaxis.set_ticks_position('right')


ax510.set_ylabel(r'Green band $\tau$' + '\n after 250 $\mu$s acquisition ($\mu$s)',fontsize=fsizepl)
ax5100.set_ylabel(r'Red band $\tau$' + '\n after 250 $\mu$s acquisition ($\mu$s)',fontsize=fsizepl)
ax510.set_xlabel('Nanoparticle diameter (nm)',fontsize=fsizepl)
ax510.set_yticks([30,40,50,60,70,80]) #[100,200,300,400,500,600])
ax510.set_ylim([28,82])  ###1.4ms -> 0,610
ax510.set_xlim([36,64])
ax510.set_xticks([40,50,60])
ax5100.set_xlabel('Nanoparticle diameter (nm)',fontsize=fsizepl)
ax5100.set_yticks([30,40,50,60,70,80]) #[100,200,300,400,500,600])
ax5100.set_ylim([28,82]) #[0,610])
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


if do_A:
    A = pickle.load( open( "A.p", "rb" ) )
if do_B:
    B = pickle.load( open( "B.p", "rb" ) )
if do_C:
    C = pickle.load( open( "C.p", "rb" ) )
if do_D:
    D = pickle.load( open( "D.p", "rb" ) )
areafactor = 10.0

ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel(r'$\hspace*{-0.5cm}$' + 'Average intensity per pixel (kHz)',fontsize=fsizepl ,horizontalalignment='left', position=(0.07,25))#,multialignment='left')

ax51.tick_params(labelsize=fsizenb) 
ax.tick_params(labelsize=fsizenb) 
if do_C:
    #green t, different areas, triangleC
    ax.plot(C['intensb'][C['non_cut_k']],C['taublue'][C['non_cut_k'],tauindice], color='g', markeredgewidth=0.0,marker = 'H', markersize = 12)
if do_D:
    #green t, different areas, triangleC
    print(D['intensb'][D['non_cut_k']])
    print(D['taublue'][D['non_cut_k'],tauindice])
    ax.plot(D['intensb'][D['non_cut_k']],D['taublue'][D['non_cut_k'],tauindice], color='g', marker = 'x', markersize = 12)

if do_C:
    #red t, different areas, triangleC
    ax.plot(C['intensr'][C['non_cut_k']],C['taured'][C['non_cut_k'],tauindice], color='r', marker = 'H', markersize = 12, markeredgewidth=0.0)
if do_D:
    #red t, different areas, triangleC
    print(D['intensr'][D['non_cut_k']])
    print(D['taured'][D['non_cut_k'],tauindice])
    ax.plot(D['intensr'][D['non_cut_k']],D['taured'][D['non_cut_k'],tauindice], color='r', marker = 'x', markersize = 12)
#uncommented until here
ax.set_ylabel(r'$\tau$ after 250 $\mu$s of acquisition ($\mu$s)',fontsize=fsizepl)
ax.set_yticks([72,75,78])#[530,550,570])
ax.set_ylim([71,79])
ax.set_xlim([0.4,2.5])
ax.set_xticks([0.5,1,1.5,2])


plt.tight_layout()

   
multipage_longer('Single.pdf',dpi=80)    
#plt.show()
klklklk
