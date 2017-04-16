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
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = DeprecationWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.simplefilter(action = "ignore", category = PendingDeprecationWarning)
from Registration import * 
from tifffile import *
from sklearn.mixture import GMM 
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.misc
import matplotlib.animation as animation
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

### settings
fsizepl = 24
fsizenb = 20
mkstry = ['8','11','5'] #marker size for different dsets Med Zoom/Large Zoom/Small Zoom
dpi_no=80
sizex = 8
sizey =6
###
sizex = 8
sizey=3

fig1= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
fig1.set_size_inches(1200./fig1.dpi,900./fig1.dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')    

noplots = 12
nolines = 5
######## PICS
axpics = plt.subplot2grid((nolines,noplots), (0,0), colspan=6, rowspan=1)
axpics.text(-0.15, 1.0, 'a', transform=axpics.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})
axpics.spines['right'].set_visible(False)
axpics.spines['top'].set_visible(False)
axpics.spines['left'].set_visible(False)
axpics.spines['bottom'].set_visible(False)
axpics.set_xticks([])
axpics.set_yticks([])

length_scalebar = 100.0 #in nm 
scalebar_legend = '100 nm'
length_scalebar_in_pixels = np.ceil(length_scalebar/(1.49))        
import boe_bar as sb

#all cuts adjusted so that if one less pixel, loses contrast
xinit = [ 16,   5, 12,  22,  4,  13] 
xend =  [-16,  -5,-12, -22, -4, -13]
yinit = [ 12,   6,  1,  31,  2,  27] 
yend =  [-12,  -6, -1, -31, -2, -27]
    
#pics
leto =['RT','N30','N40','N50','N60', 'N70'] 

index= 5
se70 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/' + leto[index] +'SEchannel.npz',mmap_mode='r') 
arr_img70 = se70['data'][xinit[index]:xend[index],yinit[index]:yend[index]] 
   
index= 4
se60 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/' + leto[index] +'SEchannel.npz',mmap_mode='r') 
arr_img60 = se60['data'][xinit[index]:xend[index],yinit[index]:yend[index]] 

index= 3
se50 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/' + leto[index] +'SEchannel.npz',mmap_mode='r') 
arr_img50 = se50['data'][xinit[index]:xend[index],yinit[index]:yend[index]] 

index= 2
se40 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/' + leto[index] +'SEchannel.npz',mmap_mode='r') 
arr_img40 = se40['data'][xinit[index]:xend[index],yinit[index]:yend[index]] 

index= 1
se30 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/' + leto[index] +'SEchannel.npz',mmap_mode='r')
arr_img30 = se30['data'][xinit[index]:xend[index],yinit[index]:yend[index]] 

index= 0
se25 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/' + leto[index] +'SEchannel.npz',mmap_mode='r')
arr_img25 = se25['data'][xinit[index]:xend[index],yinit[index]:yend[index]] 
   
ypos = 0.85#0.6475#0.6175
sizefig = 0.09 #.105

inset1 = fig1.add_axes([0.45, ypos, sizefig, sizefig]) #was 0.55
inset1.imshow(arr_img70,cmap = cm.Greys_r)
sbar = sb.AnchoredScaleBar(inset1.transData, length_scalebar_in_pixels, '', style = 'dark', loc = 4, my_fontsize = fsizenb, my_linewidth = 2.0)
inset1.add_artist(sbar)    
plt.setp(inset1, xticks=[], yticks=[])        
inset1.set_title(r'70 $^{\circ}$C', fontsize = fsizepl)

inset2 = fig1.add_axes([0.37, ypos, sizefig, sizefig]) #was 0.55
inset2.imshow(arr_img60,cmap = cm.Greys_r)
sbar = sb.AnchoredScaleBar(inset2.transData, length_scalebar_in_pixels, '', style = 'dark', loc = 4, my_fontsize = fsizenb, my_linewidth = 2.0)
inset2.add_artist(sbar)    
plt.setp(inset2, xticks=[], yticks=[])
inset2.set_title(r'60 $^{\circ}$C', fontsize = fsizepl)

inset3 = fig1.add_axes([0.29, ypos, sizefig, sizefig]) #was 0.55
inset3.imshow(arr_img50,cmap = cm.Greys_r)
sbar = sb.AnchoredScaleBar(inset3.transData, length_scalebar_in_pixels, '', style = 'dark', loc = 4, my_fontsize = fsizenb, my_linewidth = 2.0)
inset3.add_artist(sbar)    
plt.setp(inset3, xticks=[], yticks=[])
inset3.set_title(r'50 $^{\circ}$C', fontsize = fsizepl)

inset4 = fig1.add_axes([0.21,ypos, sizefig, sizefig]) #was 0.55
inset4.imshow(arr_img40,cmap = cm.Greys_r)
sbar = sb.AnchoredScaleBar(inset4.transData, length_scalebar_in_pixels, '', style = 'dark', loc = 4, my_fontsize = fsizenb, my_linewidth = 2.0)
inset4.add_artist(sbar)    
plt.setp(inset4, xticks=[], yticks=[])
inset4.set_title(r'40 $^{\circ}$C', fontsize = fsizepl)

inset5 = fig1.add_axes([0.13, ypos, sizefig, sizefig]) #was 0.55
inset5.imshow(arr_img30,cmap = cm.Greys_r)
sbar = sb.AnchoredScaleBar(inset5.transData, length_scalebar_in_pixels,'', style = 'dark', loc = 4, my_fontsize = fsizenb, my_linewidth = 2.0)
inset5.add_artist(sbar)    
plt.setp(inset5, xticks=[], yticks=[])
inset5.set_title(r'30 $^{\circ}$C', fontsize = fsizepl)

inset6 = fig1.add_axes([0.05, ypos, sizefig, sizefig]) #was 0.55
inset6.imshow(arr_img25,cmap = cm.Greys_r)
sbar = sb.AnchoredScaleBar(inset6.transData, length_scalebar_in_pixels,scalebar_legend, style = 'dark', loc = 8, my_fontsize = fsizenb, my_linewidth = 2.0)
inset6.add_artist(sbar)    
plt.setp(inset6, xticks=[], yticks=[])
inset6.set_title(r'25 $^{\circ}$C', fontsize = fsizepl)

################ A
ax3 = plt.subplot2grid((nolines,noplots), (1,0), colspan=6, rowspan=2)
ax3.text(-0.15, 1.0, 'b', transform=ax3.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})
#NEWNEW DATA 
#sys.path.append("../2017-03-19_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEW/") # necessary 
sys.path.append("../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/") # necessary 
from Combining_data_with_prefix_onlyIntensVisibRatioNEWDATA import do_pic
result2, result = do_pic(ax3,fig1)

avisi = result2.params['a'].value
alinear = result.params['a'].value




################ B
ax30 = plt.subplot2grid((nolines,noplots), (3,0), colspan=6, rowspan=2)
ax30.text(-0.15, 1.0, 'c', transform=ax30.transAxes,fontsize=fsizepl, fontweight='bold', va='top', ha='right', 
         bbox={'facecolor':'None', 'pad':5})

xhere = np.linspace(25,70,100)
#ax30.plot(xhere, 100.0*np.abs(np.ones(len(xhere))*alinear),color='k',lw=2,ls='--')
#ax30.plot(xhere[0], 100.0*alinear,color='k',marker='d',markersize=12)

#ax30.plot(xhere, 100.0*np.abs(-2.0*avisi/(1.0+avisi*xhere+(-1-avisi*xhere[0]+2/(1.0+1.0)))**2),color='k',lw=2,ls='--')
#ax30.plot(xhere[0], 100.0*np.abs(-2.0*avisi/(1.0+avisi*xhere[0]+(-1-avisi*xhere[0]+2/(1.0+1.0)))**2),color='k',marker='o',markersize=12)

ax30.spines['right'].set_visible(False)
ax30.spines['top'].set_visible(False)
ax30.xaxis.set_ticks_position('bottom')
ax30.yaxis.set_ticks_position('left')
ax30.set_ylabel(r'$\partial$(signal)/$\partial$T ($\%$ $^{\circ}$C$^{-1}$)',fontsize=fsizepl)
ax30.set_xlabel('Temperature at sample ($^{\circ}$C)',fontsize=fsizepl)
ax30.tick_params(labelsize=fsizenb)
ax30.set_ylim([0.4,2.1])
ax30.set_xlim([20,75])
ax30.set_xticks([25,30,40,50,60,70])
#ax30.set_yticks([1,2.0,3,4])
ax30.fill_between(xhere,0.5,1.5, color =[168/256,175/256,175/256],edgecolor='k',
                         facecolor=[168/256,175/256,175/256],
                         alpha=0.5,
                         linewidth=0.0)
ax30.text(47.5,0.75, 'previously reported \n (fluorescence ratio of intensity)', fontsize=fsizenb, va='center',ha='center')
ax30.set_yticks([0.5,1,1.5,2])
#ax30.set_ylim([0.4,1.6])

###PARABOLA UP RATIO
def deriv_parab(x,aa,bb):
            return 2*aa*(x-x[0]) + b
            
#PARAB UP
a=5.40441990714e-05
b=0.00935104247854
#PARAB DOWN
ad=0.000152705595258
bd=-0.00428060304499
xhere = np.linspace(24.9,60.05,100)
xhered = np.linspace(25,58.8,100)
ax30.plot(xhere, 100.0*deriv_parab(xhere,a,b),color='r',lw=2,ls='--')
ax30.plot(xhered, 100.0*deriv_parab(xhered,ad,bd),color='b',lw=2,ls='--')

#ax30.plot(xhere[0],100.0*deriv_parab(xhere,a,b)[0] ,color='r',marker='d',markersize=12,markeredgecolor='None')
#ax30.plot(xhered[0], 100.0*deriv_parab(xhered,ad,bd)[0],color='b',marker='d',markersize=12,markeredgecolor='None')
plt.tight_layout()
   
multipage_longer_desired_aspect_ratio('Fig4.pdf',1600,1200,dpi=80,)

#OLD TANH MODEL
#Fitted C = 17.23
#Fitted DE = 0.425065
#def deriv(DE, k, T, C):
#    
#    return DE/(k * (T+273.15)**2)/(1 + np.cosh(DE/(k * (T+273.15)) - C))
#    
#T = np.linspace(30,60,50)
#T2 = np.linspace(25,65,50)
#ax30.plot(T,100.0*deriv(0.425065,8.617*1.0e-5,T,17.23),lw=2,color='k',ls='--')

#old data
#ax4 = plt.subplot2grid((nolines,noplots), (0,1), colspan=1, rowspan=1)
#sys.path.append("../2017-01-19_Combining_data_Andrea/") # necessary 

#old data
#sys.path.append("../2017-01-19_Combining_data_Andrea/") # necessary 
#from Combining_data_with_prefix_onlyIntensVisibRatio import do_pic
#new data, with filter + thermocoupler on sample
#sys.path.append("../2017-03-12_Andrea_NPs_NewTempData_ThemorcoupleOnSample+Filter/") # necessary 
