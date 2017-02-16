#nominally
#1mus moving
#2mus on
#7mus off
#in practise,
#moving until 0.5mus
#excitation between 0.5 and 2.5mus
#decay rest

#### THIS FILE CONSIDERS ONLY THE SE IMAGE TAKEN PRIOR TO TR

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
#from BackgroundCorrection import *
#from TConversionThermocoupler import *
import matplotlib.cm as cm
import scipy.ndimage as ndimage
#from matplotlib_scalebar.scalebar import ScaleBar #### Has issue with plotting using latex font. only import when needed, then unimport
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
#from Registration import * 
#from tifffile import *
from sklearn.mixture import GMM 
import matplotlib.cm as cm
#from FluoDecay import *
#from PlottingFcts import *

#from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.misc
import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile

import matplotlib.colors as colors

#import scalebars as sb

import boe_bar as sb
#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################
Time_bin = 1000#in ns; 
nominal_time_on = 150.0 #time during which e-beam nominally on, in mus
totalpoints = 1500 #total number of time-resolved points

Pixel_size = [2.2] #nm
Ps = Pixel_size #pixel size in nm, the numbers above with round nm precision
No_experiments = [1]

description = 'Andrea large NaYF4:Er'    # (20kV, 30$\mu$m, ' + str(Ps[index]) + 'nm pixels, ' + str(No_experiments[index]) + 'expts., InLens registered)'] #, \n' #+ obs[index] ']    
               
kv = [10]

let = ['Beautiful']

nametr = ['']
######################################## Plot with dose for different apertures
pisize =Pixel_size

listofindex =np.arange(0,1)#,11]

consider_whole_light = [0]
#4,5,7 segmentation ok-ish
#index = 4
#if index is 4:
def do_pic0(): #(ax11,ax1,ax2):
    
    print('here')

    for index in [0]: #listofindex:
        
        import matplotlib.gridspec as gridspec
    #    gs1 = gridspec.GridSpec(1,3)
    #    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
        
        #print(index)
        
        Ps = str("{0:.2f}".format(Pixel_size[index])) 
    
        se = np.load("../2016-12-19_Andrea_BigNPs_5DiffTemps/" + str(let[index]) +'SEchannel.npz',mmap_mode='r') 
        segmm = np.load("../2016-12-19_Andrea_BigNPs_5DiffTemps/" +str(let[index]) +'SEchannelGMM.npz',mmap_mode='r') 
        red = np.load("../2016-12-19_Andrea_BigNPs_5DiffTemps/" +str(let[index]) +'Redbright.npz',mmap_mode='r') 
        blue = np.load("../2016-12-19_Andrea_BigNPs_5DiffTemps/" +str(let[index]) +'Bluebright.npz',mmap_mode='r') 
        
        fsizepl = 24
        fsizenb = 20 
        sizex = 8 
        sizey = 6
        dpi_no = 80
        lw = 2
        
        length_scalebar = 100.0 #in nm 
        scalebar_legend = '100nm'
        length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size[index]))
        
        sizex = 8
        sizey = 6
        dpi_no = 80
    #    fig40= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    #    fig40.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
    #    plt.rc('text.latex', preamble=r'\usepackage{xcolor}') #not working
    #    plt.rc('text', usetex=True)
    #    plt.rc('text.latex', preamble=r'\usepackage{xcolor}')  #not working
    #    plt.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]  #not working
    #    plt.rc('font', family='serif')
    #    plt.rc('font', serif='Palatino')      
        
        
        
        gc.collect()
        
        ax11 = plt.subplot2grid((6,6), (3, 0), colspan=1, rowspan=1)
        ax11.set_title('Electron signal',fontsize=fsizepl) #as per accompanying txt files
        plt.imshow(se['data'],cmap=cm.Greys_r)
        
        import matplotlib.font_manager as fm
        fontprops = fm.FontProperties(size=fsizenb)
        sbar = sb.AnchoredScaleBar(ax11.transData, length_scalebar_in_pixels, scalebar_legend, style = 'bright', loc = 4, my_fontsize = fsizenb)
        ax11.add_artist(sbar)
        plt.axis('off')
        
        ax1 = plt.subplot2grid((6,6), (3, 1), colspan=1, rowspan=1)
        ax1.set_title('Green band',fontsize=fsizepl)
        plotb = np.average(blue['data'],axis = (0,1))
        plotb = plotb
        im = plt.imshow(plotb,cmap=cm.Greens,norm=colors.PowerNorm(0.5))#, vmin=plotb.min(), vmax=plotb.max())) #or 'OrRd'
        plt.axis('off')
        
        ax2 = plt.subplot2grid((6,6), (3, 2), colspan=1, rowspan=1)
        ax2.set_title('Red band',fontsize=fsizepl)
        plotr = np.average(red['data'],axis = (0,1))
        plotr = plotr
        imb = plt.imshow(plotr,cmap=cm.Reds,norm=colors.PowerNorm(0.5))#, vmin=plotr.min(), vmax=plotr.max())) #or 'OrRd'
        plt.axis('off') 
    
        plt.tight_layout()
        fig1.subplots_adjust(wspace=0, hspace=-0.5)
        
        sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4, my_fontsize = fsizenb)
        ax1.add_artist(sbar)
        unit = '(kHz)'
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
        axColor = plt.axes([box.x0, box.y0*1.05, box.width,0.01 ])    #original 1.1
        cb2 = plt.colorbar(im, cax = axColor, orientation="horizontal")
        
        cb2.set_label("Photon counts (kHz)", fontsize = fsizenb)
        cb2.set_ticks([7, 14])
        cb2.ax.tick_params(labelsize=fsizenb) 
        
        sbar = sb.AnchoredScaleBar(ax2.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4, my_fontsize = fsizenb)
        ax2.add_artist(sbar)  
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0*1.00, box.width, box.height])
        axColor = plt.axes([box.x0, box.y0*1.05 , box.width,0.01 ])    
        cb1 = plt.colorbar(imb, cax = axColor, orientation="horizontal")
        cb1.set_label('Photon counts (kHz)', fontsize = fsizenb) 
        cb1.set_ticks([3.5,7])
        
        cb1.ax.tick_params(labelsize=fsizenb) 
       
        #print(index)
  
    return
   # multipage_longer('ZZZZSingle-'+ let[index] + '.pdf',dpi=80)

#klklklk