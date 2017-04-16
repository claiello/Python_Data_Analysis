import os
import sys
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../Python modules/") # necessary for the tex fonts
import scipy as sp
import scipy.misc
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
from BackgroundCorrection import *
from TConversionThermocoupler import *
import matplotlib.cm as cm
import scipy.ndimage as ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from MakePdf import *
from matplotlib.pyplot import cm #to plot following colors of rainbow
from matplotlib import rc
from CreateDatasets import *
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = DeprecationWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.simplefilter(action = "ignore", category = PendingDeprecationWarning)
from Registration import * 
from tifffile import *
from sklearn.mixture import GMM 
import matplotlib.cm as cm
from FluoDecay import *
from PlottingFcts import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.misc
import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile
import my_fits
from uncertainties import unumpy
from my_fits import *
from matplotlib.lines import Line2D
##############################################################################
##############################################################################

def do_visib_other_qttties(host,par2,par3,ax3,ax4,ax5,ax3a,ax4a,ax5a):
#if True:
    fsizepl = 24
    fsizenb = 20
   
    initbin = (150+50+3)-1
    backgdinit = 50
    ### PIXEL
    Pixel_size = [2.48,3.72,1.86,2.98,1.86,2.98,2.13]
    let = ['p300','p200','p400','p250','p400b','p250b','p350'] #no of pixels
    loadprefix = '../2017-03-28_AndreaNPS_varyotherparams_nostagezmovement/' #USED THIS OLD DSET
    nombre = 7
    undex = 0
    red0 = np.load(loadprefix + 'Red_int_arrayPixel.npz',mmap_mode='r')  
    red = red0['data']
    redbg0 = np.load(loadprefix + 'bgRed_int_arrayPixel.npz',mmap_mode='r')  
    redbg = redbg0['data']
    blue0 = np.load(loadprefix + 'Blue_int_arrayPixel.npz',mmap_mode='r')
    blue = blue0['data']
    bluebg0 = np.load(loadprefix + 'bgBlue_int_arrayPixel.npz',mmap_mode='r')
    bluebg = bluebg0['data']
    toplotpixel = (blue-red)/(red+blue)/((blue[undex]-red[undex])/(red[undex]+blue[undex]))
    toplotpixelratio = red/blue/(red[undex]/blue[undex])
    toplotpixelbg = (bluebg-redbg)/(redbg+bluebg)/((bluebg[undex]-redbg[undex])/(redbg[undex]+bluebg[undex]))
    toplotpixelratiobg = redbg/bluebg/(redbg[undex]/bluebg[undex])
    toplotpixelgreen = blue/blue[undex]
    toplotpixelred = red/red[undex]
    toplotpixelgreenbg = bluebg/blue[undex]
    toplotpixelredbg = redbg/red[undex]
    Pixel_size_full = np.copy(Pixel_size)    
    Pixel_size = np.delete(Pixel_size, [4,5])
    toplotpixel= np.delete(toplotpixel, [4,5])
    toplotpixelratio= np.delete(toplotpixelratio, [4,5])
    
    ### APERTURE
    Current = [379,48,9800,662,379,9800,6000,267] #pA
    let = ['pA379','pA48','pA9800','pA662','pA379','pA9800','pA6000','pA267'] #aperture
    loadprefix ='../2017-03-28_AndreaNPS_varyotherparams_nostagezmovement/'
    nombre = 8
    undex = 0
    red0 = np.load(loadprefix + 'Red_int_arrayCurrent.npz',mmap_mode='r')  
    red = red0['data']
    redbg0 = np.load(loadprefix + 'bgRed_int_arrayCurrent.npz',mmap_mode='r')  
    redbg = redbg0['data']
    blue0 = np.load(loadprefix + 'Blue_int_arrayCurrent.npz',mmap_mode='r')
    blue = blue0['data']
    bluebg0 = np.load(loadprefix + 'bgBlue_int_arrayCurrent.npz',mmap_mode='r')
    bluebg = bluebg0['data']
    toplotcurrent = (blue-red)/(red+blue)/((blue[undex]-red[undex])/(red[undex]+blue[undex]))
    toplotcurrentratio = red/blue/(red[undex]/blue[undex])
    toplotcurrentbg = (bluebg-redbg)/(redbg+bluebg)/((bluebg[undex]-redbg[undex])/(redbg[undex]+bluebg[undex]))
    toplotcurrentratiobg = redbg/bluebg/(redbg[undex]/bluebg[undex])
    toplotcurrentgreen = blue/blue[undex]
    toplotcurrentred = red/red[undex]
    toplotcurrentgreenbg = bluebg/blue[undex]
    toplotcurrentredbg = redbg/red[undex]
    
     #delete repetitions
    Current_full = np.copy(Current)
    print(Current_full.shape)
    Current = np.delete(Current, [4,5])
    toplotcurrent= np.delete(toplotcurrent, [4,5])
    toplotcurrentratio= np.delete(toplotcurrentratio, [4,5])
    print(Current_full.shape)
    
    kv = [10,15,5,16.8,7.5,12.5]
    let =  ['kv10','kv15','kv5','kv16pt8','kv7pt5','kv12pt5'] #kV, all at 379pA, 30mum aperture!!!!
    loadprefix ='../2017-03-28_AndreaNPS_varyotherparams_nostagezmovement/'
    nombre = 6
    undex = 0
    red0 = np.load(loadprefix + 'Red_int_arraykV.npz',mmap_mode='r')  
    red = red0['data']
    redbg0 = np.load(loadprefix + 'bgRed_int_arraykV.npz',mmap_mode='r')  
    redbg = redbg0['data']
    blue0 = np.load(loadprefix + 'Blue_int_arraykV.npz',mmap_mode='r')
    blue = blue0['data']
    bluebg0 = np.load(loadprefix + 'bgBlue_int_arraykV.npz',mmap_mode='r')
    bluebg = bluebg0['data']
    toplotkv = (blue-red)/(red+blue)/((blue[undex]-red[undex])/(red[undex]+blue[undex]))
    toplotkvratio = red/blue/(red[undex]/blue[undex])
    toplotkvbg = (bluebg-redbg)/(redbg+bluebg)/((bluebg[undex]-redbg[undex])/(redbg[undex]+bluebg[undex]))
    toplotkvratiobg = redbg/bluebg/(redbg[undex]/bluebg[undex])
    toplotkvgreen = blue/blue[undex]
    toplotkvred = red/red[undex]
    toplotkvgreenbg = bluebg/blue[undex]
    toplotkvredbg = redbg/red[undex]
    
    ###################

#    ##### STATS
#    from minepy import MINE
#    from scipy.stats import pearsonr, spearmanr
#    import scipy
#    import sklearn
#  
#        
#    mine = MINE(alpha=0.6, c=15)
#    mine2 = MINE(alpha=0.6, c=15)
#    mine3 = MINE(alpha=0.6, c=15)
#    
#    mine0 = MINE(alpha=0.6, c=15)
#    mine20 = MINE(alpha=0.6, c=15)
#    mine30 = MINE(alpha=0.6, c=15)
#    
#    #PIXEL   
#    mine.compute_score(Pixel_size, toplotpixel)a
#    rp = pearsonr(Pixel_size, toplotpixel)
#    rs = spearmanr(Pixel_size, toplotpixel)
#    dc = scipy.spatial.distance.correlation(Pixel_size, toplotpixel)
#    mi = sklearn.metrics.mutual_info_score(Pixel_size, toplotpixel)
#    print('pixel visib')
#    print(mi)
#    print(rs)
#    print(rp)
#    print(dc)
#    print(mine.mic())
#    
#    mine0.compute_score(Pixel_size, toplotpixelratio)
#    rp0 = pearsonr(Pixel_size, toplotpixelratio)
#    rs0 = spearmanr(Pixel_size, toplotpixelratio)
#    dc0 = scipy.spatial.distance.correlation(Pixel_size, toplotpixelratio)
#    mi0 = sklearn.metrics.mutual_info_score(Pixel_size, toplotpixelratio)
#    print('pixel ratio')
#    print(mi0)
#    print(rs0)
#    print(rp0)
#    print(dc0)
#    print(mine0.mic())
#    
#    # KV
#    mine2.compute_score(kv, toplotkv)
#    rp2 = pearsonr(kv, toplotkv)
#    rs2 = spearmanr(kv, toplotkv)
#    dc2 = scipy.spatial.distance.correlation(kv, toplotkv)
#    mi2 = sklearn.metrics.mutual_info_score(kv, toplotkv)
#    print('kv visib')
#    print(mi2)
#    print(rs2)
#    print(rp2)
#    print(dc2)
#    print(mine2.mic())
#    mine20.compute_score(kv, toplotkvratio)
#    rp20 = pearsonr(kv, toplotkvratio)
#    rs20 = spearmanr(kv, toplotkvratio)
#    dc20 = scipy.spatial.distance.correlation(kv, toplotkvratio)
#    mi20 = sklearn.metrics.mutual_info_score(kv, toplotkvratio)
#    print('kv ratio')
#    print(mi20)
#    print(rs20)
#    print(rp20)
#    print(dc20)
#    print(mine20.mic())
#    
#    ### Curremt
#    mine3.compute_score(Current, toplotcurrent)
#    rp3 = pearsonr(Current, toplotcurrent)
#    rs3 = spearmanr(Current, toplotcurrent)
#    dc3 = scipy.spatial.distance.correlation(Current, toplotcurrent)
#    mi3 = sklearn.metrics.mutual_info_score(Current, toplotcurrent)
#    print('current visib')
#    print(mi3)
#    print(rs3)
#    print(rp3)
#    print(dc3)
#    print(mine3.mic())
#    mine30.compute_score(Current, toplotcurrentratio)
#    rp30 = pearsonr(Current, toplotcurrentratio)
#    rs30 = spearmanr(Current, toplotcurrentratio)
#    dc30 = scipy.spatial.distance.correlation(Current, toplotcurrentratio)
#    mi30 = sklearn.metrics.mutual_info_score(Current, toplotcurrentratio)
#    print('current ratio')
#    print(mi30)
#    print(rs30)
#    print(rp30)
#    print(dc30)
#    print(mine30.mic())
#    
#    klklklk
    
    ################################

#    from mpl_toolkits.axes_grid1 import host_subplot
#    import mpl_toolkits.axisartist as AA
#    import matplotlib.pyplot as plt
    
    #par2 = host.twiny()
    #par3 = host.twiny()
   
    host.set_xlim([1.66,3.92])
    par3.set_xlim([4, 17.8])
    par2.set_xlim([10, 13000])
    
    host.set_ylabel('Intensity thermometry \n signal, \n norm. to standard (a.u.)',fontsize=fsizepl)
    host.set_xlabel('Pixel size (nm)',fontsize=fsizepl)
   
    par2.set_xlabel('Electron beam current (pA)',fontsize=fsizepl)
    par3.set_xlabel('Electron beam energy (keV)',fontsize=fsizepl)
    
    host.plot( Pixel_size, toplotpixelratio, marker='d',markersize=12,linestyle='',color='k', label='Ratio of intensities',markeredgecolor='None')

    host.plot( Pixel_size, toplotpixel,  marker='o',markersize=12,linestyle='',color='k', label='Visibility of intensities',markeredgecolor='None')
    
    #p1 = host.plot( Pixel_size, toplotpixelbg,  marker='o',markersize=12,linestyle='',color='white')
    #p1 = host.plot( Pixel_size, toplotpixelratiobg,  marker='d',markersize=12,linestyle='',color='white')
    par2.semilogx( Current, toplotcurrentratio,  marker='d',markersize=12,linestyle='',color='k',label='Ratio of intensities',markeredgecolor='None')
    par2.semilogx( Current, toplotcurrent,  marker='o',markersize=12,linestyle='',color='k',label='Visibility of intensities',markeredgecolor='None')
    #p3 = par2.plot( Current, toplotcurrentbg,  marker='o',markersize=12,linestyle='',color='white',label='Varying current',markeredgecolor='c')
    #p3 = par2.plot( Current, toplotcurrentratiobg, marker='d',markersize=12,linestyle='',color='white',label='Varying current',markeredgecolor='c')
    #par2.get_xaxis().majorTicks[0].label1.set_horizontalalignment('left')
#    par2.get_xaxis().majorTicks[1].label1.set_horizontalalignment('right')


    par3.plot( kv, toplotkv,  marker='o',markersize=12,linestyle='',color='k',label='Varying current',markeredgecolor='None')
    par3.plot( kv, toplotkvratio,  marker='d',markersize=12,linestyle='',color='k',label='Varying current',markeredgecolor='None')
    #p4 = par3.plot( kv, toplotkvbg,  marker='o',markersize=12,linestyle='',color='white',label='Varying current',markeredgecolor='m')
    #p4 = par3.plot( kv, toplotkvratiobg,  marker='d',markersize=12,linestyle='',color='white',label='Varying current',markeredgecolor='m')
      
      
         
   
    
#    host.axis["bottom"].label.set_color('b')
#    par2.axis["bottom"].label.set_color('c')
#    par3.axis["bottom"].label.set_color('m')
#    
#    host.tick_params(axis="x", colors="b")
#    par2.tick_params(axis="x", colors="c")
#    par3.tick_params(axis='x', colors='m')
    
#    host.spines["bottom"].set_edgecolor('b')
#    par2.spines["bottom"].set_edgecolor('c')
#    par3.spines["bottom"].set_edgecolor('m')
    
    host.legend(loc='best',frameon=False,fontsize=fsizenb,numpoints=1)
    
    host.tick_params(labelsize=fsizenb)
    par2.tick_params(labelsize=fsizenb)
    par3.tick_params(labelsize=fsizenb)
    
    ax3.tick_params(labelsize=fsizenb)
    ax3a.set_xlim([1.66,3.92])
    ax4.tick_params(labelsize=fsizenb)
    ax5.tick_params(labelsize=fsizenb)
    
    ax3a.tick_params(labelsize=fsizenb)
    ax4a.tick_params(labelsize=fsizenb)
    ax5a.tick_params(labelsize=fsizenb)
#    
#    host.axis["bottom"].major_ticklabels.set_color('b')
#    par2.axis["bottom"].major_ticklabels.set_color('c')
#    par3.axis["bottom"].major_ticklabels.set_color('m')
    
#    host.yaxis.set_label_position("right")
#    host.xaxis.set_ticks_position('bottom')
#    host.yaxis.set_ticks_position('right')
    
#    host.spines["top"].toggle(all=False)
#    host.spines["left"].toggle(all=False)
#    host.spines["right"].toggle(True)
    host.axhline(y=1.0 , lw=2, color='k', ls='--')
    par2.axhline(y=1.0 , lw=2, color='k', ls='--')
    par3.axhline(y=1.0 , lw=2, color='k', ls='--')
    
    ymin = 0.4
    ymax = 1.8
    host.set_ylim([ymin, ymax])
    par3.set_ylim([ymin, ymax])
    par2.set_ylim([ymin, ymax])
    host.set_yticks([0.5,1,1.5])
    par2.set_yticks([0.5,1,1.5])
    par3.set_yticks([0.5,1,1.5])
    
    host.set_xticks([2,2.5,3,3.5])
    par2.set_xscale('log')
    par2.set_xticks([50,500,5000])
    par2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    par3.set_xticks([5,10,15])
    
    ax3.set_ylabel('Cathodoluminescence \n intensity, \n norm. to standard (a.u.)',fontsize=fsizepl)
    ax3.set_xlabel('Pixel size (nm), \n in the order of experiment',fontsize=fsizepl)
   
    ax4.set_xlabel('Electron beam current (pA), \n in the order of experiment',fontsize=fsizepl)
    ax5.set_xlabel('Electron beam energy (keV), \n in the order of experiment',fontsize=fsizepl)
    
    ax3.plot(np.arange(0,len(toplotpixelred)),toplotpixelred, marker='^',markersize=12,linestyle='',color='r',markeredgecolor='None')
    ax3.plot(np.arange(0,len(toplotpixelred)),toplotpixelgreen, marker='^',markersize=12,linestyle='',color='g',markeredgecolor='None')
    marker_style = dict(color='g', linestyle='', marker='^',
                    markersize=14, markerfacecoloralt='r',markeredgecolor='white')
    ax3.plot(np.arange(0,1),toplotpixelgreen[0], fillstyle='left', **marker_style)
    ax3.set_xticks([0,1,2,3,4,5,6])
    ax3.set_xticklabels(['2.5','3.7','1.9','3.0','1.9','3.0','2.1'])
    ax3.set_xlim([-0.5, 6.5])
    ax3.set_ylim([0.875, 1.025])
    ax3.set_yticks([0.9,1.0])
    
    
    ax4.plot(np.arange(0,len(toplotcurrentred)),toplotcurrentred, marker='^',markersize=12,linestyle='',color='r',markeredgecolor='None')
    ax4.plot(np.arange(0,len(toplotcurrentred)),toplotcurrentgreen, marker='^',markersize=12,linestyle='',color='g',markeredgecolor='None')
    marker_style = dict(color='g', linestyle='', marker='^',
                    markersize=14, markerfacecoloralt='r',markeredgecolor='white')
    ax4.plot(np.arange(0,1),toplotcurrentgreen[0], fillstyle='left', **marker_style)
    ax4.set_xticks([0,1,2,3,4,5,6,7])
    ax4.set_xticklabels(['379','48','9.8k','662','379','9.8k','6k','267'])
    ax4.set_xlim([-0.5, 7.5])
    ax4.set_ylim([-0.5, 18])
    ax4.set_yticks([1,5,10,15])
    
    
    ax5.plot(np.arange(0,len(toplotkvred)),toplotkvred, marker='^',markersize=12,linestyle='',color='r',markeredgecolor='None')
    ax5.plot(np.arange(0,len(toplotkvred)),toplotkvgreen, marker='^',markersize=12,linestyle='',color='g',markeredgecolor='None')
    marker_style = dict(color='g', linestyle='', marker='^',
                    markersize=14, markerfacecoloralt='r',markeredgecolor='white')
    ax5.plot(np.arange(0,1),toplotkvgreen[0], fillstyle='left', **marker_style)
    ax5.set_xticks([0,1,2,3,4,5])
    ax5.set_xticklabels(['10','15','5','16.8','7.5','12.5'])
    ax5.set_xlim([-0.5, 5.5])
    ax5.set_ylim([0.6, 1.3])
    ax5.set_yticks([0.75, 1, 1.25])
    
    ax3a.set_ylabel('Cathodoluminescence \n intensity, \n norm. to standard (a.u.)',fontsize=fsizepl)
    ax3a.set_xlabel('Pixel size (nm)',fontsize=fsizepl)
   
    ax4a.set_xlabel('Electron beam current (pA)',fontsize=fsizepl)
    ax5a.set_xlabel('Electron beam energy (keV)',fontsize=fsizepl)
    
    ax3a.plot(Pixel_size_full,toplotpixelred, marker='^',markersize=12,linestyle='',color='r',markeredgecolor='None')
    ax3a.plot(Pixel_size_full,toplotpixelgreen, marker='^',markersize=12,linestyle='',color='g',markeredgecolor='None')
    marker_style = dict(color='g', linestyle='', marker='^',
                    markersize=14, markerfacecoloralt='r',markeredgecolor='white')
    ax3a.plot(Pixel_size[0],toplotpixelgreen[0], fillstyle='left', **marker_style)
    ax3a.set_xticks([2,2.5,3,3.5])
   # ax3a.set_xticklabels(['1.9','2.1','2.5','3.0','3.7'])
    #ax3.set_xlim([-0.5, 6.5])
    ax3a.set_ylim([0.875, 1.025])
    ax3a.set_yticks([0.9,1.0])
    
    ax4a.plot(Current_full,toplotcurrentred, marker='^',markersize=12,linestyle='',color='r',markeredgecolor='None')
    ax4a.plot(Current_full,toplotcurrentgreen, marker='^',markersize=12,linestyle='',color='g',markeredgecolor='None')
    marker_style = dict(color='g', linestyle='', marker='^',
                    markersize=14, markerfacecoloralt='r',markeredgecolor='white')
    ax4a.semilogx(Current_full[0],toplotcurrentgreen[0], fillstyle='left', **marker_style)
    ax4a.set_xscale('log')
    ax4a.set_xticks([50,500,5000])
    ax4a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #ax4a.set_xticklabels(['379','48','9.8k'])
    #ax4a.set_xlim([-0.5, 7.5])
    ax4a.set_ylim([-0.5, 18])
    ax4a.set_yticks([1,5,10,15])
    ax4a.set_xlim([10, 13000])
    
    
    ax5a.plot(kv,toplotkvred, marker='^',markersize=12,linestyle='',color='r',markeredgecolor='None')
    ax5a.plot(kv,toplotkvgreen, marker='^',markersize=12,linestyle='',color='g',markeredgecolor='None')
    marker_style = dict(color='g', linestyle='', marker='^',
                    markersize=14, markerfacecoloralt='r',markeredgecolor='white')
    ax5a.plot(kv[0],toplotkvgreen[0], fillstyle='left', **marker_style)
    ax5a.set_xticks([5,10,15])
    ax5a.set_ylim([0.6, 1.3])
    ax5a.set_yticks([0.75, 1, 1.25])
    ax5a.set_xlim([4, 17.8])
    
#    host.set_xlabel('Pixel size (nm)',fontsize=fsizepl)
    
#    ####### INTENSITIES
#    
#    ax4down.set_ylabel('Intensity, norm. to standard (a.u.)',fontsize=fsizepl)
#    ax4down.set_xlim([1.45,9])
#    
#    ax4down.spines['left'].set_visible(False)
#    ax4down.spines['top'].set_visible(False)
#    ax4down.xaxis.set_ticks_position('bottom')
#    ax4down.yaxis.set_ticks_position('right')
#    ax4down.yaxis.set_label_position("right")
#    
#    ax4down.set_ylim([0.5,1.5])
#    #par2down = ax4down.twiny()
#    #par2down.set_ylim([10,18])
#    par3down = ax4down.twiny()
#    par3down.set_ylim([0.5,1.5])
#    
#    #par2down.spines['left'].set_visible(False)
#    #par2down.spines['top'].set_visible(False)
#    #par2down.xaxis.set_ticks_position('bottom')
#    #par2down.yaxis.set_ticks_position('right')
#    par3down.spines['left'].set_visible(False)
#    par3down.spines['top'].set_visible(False)
#    par3down.xaxis.set_ticks_position('bottom')
#    par3down.yaxis.set_ticks_position('right')
#    
#    
#    #par2down.set_xlim([-12000, 10000])
#    par3down.set_xlim([-20, 45])
#   
#    #intensitites
#    ax4down.plot( Pixel_size, toplotpixelgreen, marker='^',markersize=12,linestyle='',color='g', markeredgecolor='None')
#    ax4down.plot( Pixel_size, toplotpixelred,   marker='^',markersize=12,linestyle='',color='r', markeredgecolor='None')
#    #ax4down.plot( Pixel_size, toplotpixelgreenbg, marker='^',markersize=12,linestyle='',color='white', markeredgecolor='g')
#    #ax4down.plot( Pixel_size, toplotpixelredbg,   marker='^',markersize=12,linestyle='',color='white', markeredgecolor='r')
#   
#    #par2down.plot( Current, toplotcurrentgreen, marker='^',markersize=12,linestyle='',color='g', markeredgecolor='None')
#    #par2down.plot( Current, toplotcurrentred,   marker='^',markersize=12,linestyle='',color='r', markeredgecolor='None')
#    #par2down.plot( Current, toplotcurrentgreenbg, marker='^',markersize=12,linestyle='',color='white', markeredgecolor='g')
#    #par2down.plot( Current, toplotcurrentredbg,   marker='^',markersize=12,linestyle='',color='white', markeredgecolor='r')
#   
#   
#    par3down.plot( kv, toplotkvgreen, marker='^',markersize=12,linestyle='',color='g', markeredgecolor='None')
#    par3down.plot( kv, toplotkvred,   marker='^',markersize=12,linestyle='',color='r', markeredgecolor='None')
#    #par3down.plot( kv, toplotkvgreen, marker='^',markersize=12,linestyle='',color='white', markeredgecolor='g')
#    #par3down.plot( kv, toplotkvred,   marker='^',markersize=12,linestyle='',color='white', markeredgecolor='r')
#    
#    ax4down.axhline(y=1.0 , lw=2, color='k', ls='--')
#    
#    
#lllllll
###############################################################################
###############################################################################
###############################################################################

    #OLD CODE
#    yerrV = np.empty(nombre)
#    yerrR = np.empty(nombre)
#    red_int_array = np.empty(nombre)
#    blue_int_array = np.empty(nombre)
#    for index in np.arange(0,nombre):
#        
#        print(index)
#    
#        #FOREGROUND
#        redd = np.load(loadprefix + let[index] + 'Redbright.npz',mmap_mode='r')
#        #BACKGROUND
#        #redd = np.load(loadprefix + let[index] + 'bgRedbright.npz',mmap_mode='r')
#        red_int_array[index] = np.average(redd['data'][:,backgdinit:initbin,:,:],axis=(0,1,2,3))
#        No = redd['data'].shape[0]*redd['data'][:,backgdinit:initbin,:,:].shape[1]*redd['data'].shape[2]*redd['data'].shape[3]
#        del redd
#        #FOREGROUND
#        blued = np.load(loadprefix + let[index] + 'Bluebright.npz',mmap_mode='r') 
#        #BACKGROUND
#        #blued = np.load(loadprefix + let[index] + 'bgBluebright.npz',mmap_mode='r')
#        blue_int_array[index] = np.average(blued['data'][:,backgdinit:initbin,:,:],axis=(0,1,2,3))
#        del blued
#        gc.collect()
#        
#        #VECTOR TO DO NORMALIZATION - ALWAYS FOREGROUND, NOT BACKGROUND!!!!
#        reddN = np.load(loadprefix + let[undex] + 'Redbright.npz',mmap_mode='r') 
#        red_int_arrayN = np.average(reddN['data'][:,backgdinit:initbin,:,:],axis=(0,1,2,3))
#        NoN = reddN['data'].shape[0]*reddN['data'][:,backgdinit:initbin,:,:].shape[1]*reddN['data'].shape[2]*reddN['data'].shape[3]
#        del reddN
#        bluedN = np.load(loadprefix + let[undex] + 'Bluebright.npz',mmap_mode='r') 
#        blue_int_arrayN = np.average(bluedN['data'][:,backgdinit:initbin,:,:],axis=(0,1,2,3))
#        del bluedN
#        gc.collect()
#
###        print(redd['data'].shape[1])
###        print(redd['data'][:,backgdinit:initbin,:,:].shape[1])
###        indeed prints different things
#        
#        ured_int_array = unumpy.uarray(red_int_array[index],np.sqrt(red_int_array[index])/np.sqrt(No))
#        ublue_int_array = unumpy.uarray(blue_int_array[index],np.sqrt(blue_int_array[index])/np.sqrt(No))
#        
#        ured_int_arrayN = unumpy.uarray(red_int_arrayN,np.sqrt(red_int_arrayN)/np.sqrt(NoN))
#        ublue_int_arrayN = unumpy.uarray(blue_int_arrayN,np.sqrt(blue_int_arrayN)/np.sqrt(NoN))
#          
#        #Visibility
#        print('visib')
#        print(((blue_int_array[index]-red_int_array[index])/(red_int_array[index]+blue_int_array[index]))/((blue_int_arrayN-red_int_arrayN)/(red_int_arrayN+blue_int_arrayN)))
#        #Ratio
#        print('ratio')
#        print(((red_int_array[index])/(blue_int_array[index]))/((red_int_arrayN)/(blue_int_arrayN)))
#        #Rrror on visibility
#        print('ERR visib')
#        yerrV[index] = unumpy.std_devs(((ublue_int_array-ured_int_array)/(ured_int_array+ublue_int_array))/((ublue_int_arrayN-ured_int_arrayN)/(ured_int_arrayN+ublue_int_arrayN)))
#        print(yerrV[index])
#        print('ERR ratio')
#        yerrR[index] = unumpy.std_devs(((ured_int_array)/(ublue_int_array))/((ured_int_arrayN)/(ublue_int_arrayN)))
#        print(yerrR[index])
#        
#        print('green')
#        print(blue_int_array[index]/blue_int_arrayN) 
#        print('red')
#        print(red_int_array[index]/red_int_arrayN)
#        
#        
#        del red_int_arrayN   ,blue_int_arrayN, ured_int_array, ublue_int_array  ,ured_int_arrayN  ,ublue_int_arrayN   
#        gc.collect()
#    
#    
#    klklkk
    
   ####OLD POXEL VALUES
#     ####VISIB
#    toplotpixel = [1.0,1.01514637055,1.01172723577,1.01363267797,1.00636743468,1.00841453996,1.00187643191]
#    ####RATIO
#    toplotpixelratio = [1.0,1.04359332782,1.03347218482,1.03909173294,1.01794046721,1.0238252341,1.00523058474]
#    ####ERRORS
#    toploterrpixelV = [9.33958108437e-05,0.000120090071471,8.32887436109e-05,0.00010493886892,8.3551713833e-05,0.000104789676529,8.78389691661e-05]
#    toploterrpixelR = [0.00025918707771,0.000353597161336,0.000239131604876,0.000305944613321,0.000236186237418,0.000299727423835,0.000245171241429]
