import os
import sys
import scipy as sp
import scipy.misc
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
#from BackgroundCorrection import *
import matplotlib.cm as cm
import scipy.ndimage as ndimage
#from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation
import gc
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../Python modules/") # necessary for the tex fonts

fsizetit = 18
fsizepl = 16
sizex = 8
sizey = 6
dpi_no = 80
lw = 2

import scalebars as sb

def plot_expt_by_expt_behaviour(titulo, dset, Time_bin, nominal_time_on,fastfactor,my_color_avg,major_ticks,dark_dset=None,plot_dark=False,unit='MHz'): #dset is no_expt x no_time_bins x pixel x pixel; dset.shape[0] is no_expts, Time_bin in ns, nominal_time_on in mus
    
    plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')
    plt.suptitle("Cathodoluminescence as a function of e-beam exposure time, \n " + titulo, fontsize=fsizetit)
    ax1 = plt.subplot2grid((1,2), (0, 0), colspan=1)
    #ax1.set_title(str(dset.shape[0]) + r" consecutive experiments (time follows rainbow: purple $\rightarrow$ red)",fontsize=fsizepl)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    color=iter(cm.rainbow(np.linspace(0,1,dset.shape[0])))
    dset_avg = np.average(dset,axis=(2,3)) #used to be np.average
    for k in range(dset.shape[0]):
        c=next(color)
        if k in [0,9,19,29,39,49]:
            plt.plot(np.arange(0,dset.shape[1])*Time_bin/1e3,dset_avg[k,:],c=c,label='expt. $\#$ ' + str(k+1)) #in mus, in MHz
        else:
            plt.plot(np.arange(0,dset.shape[1])*Time_bin/1e3,dset_avg[k,:],c=c, label='_nolegend_') #in mus, in MHz
    plt.plot(np.arange(0,dset.shape[1])*Time_bin/1e3,np.average(dset_avg,axis = 0),c=my_color_avg,label='average over ' + str(dset.shape[0]) + ' expts.',linewidth=lw) #in mus, in MHz
    plt.ylabel("Average luminescence of each time bin (" + unit + "), \n for " + str(dset.shape[0]) + r" consecutive experiments (time follows rainbow: purple $\rightarrow$ red)",fontsize=fsizepl)
    plt.xlabel(r"Behaviour of e-beam during each experiment: 0.12-ON/5-OFF ($\mu$s)",fontsize=fsizepl)
    plt.legend() 
    #major_ticks0 = [0.3, 2.3, 7]
    major_ticks0 = [0.12, 5]
    ax1.set_xticks(major_ticks0) 
 
    ax1 = plt.subplot2grid((1,2), (0, 1), colspan=1)
    #ax1.set_title(r"Average frame intensity",fontsize=fsizepl)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    plt.plot(np.arange(1,dset.shape[0]+1)*nominal_time_on*fastfactor,np.average(dset_avg,axis=1),c=my_color_avg, label='_nolegend_',linewidth=lw) #in mus, in MHz
    plt.ylabel(r"Average frame luminescence of each experiment  (" + unit + ')',fontsize=fsizepl)
    plt.xlabel(r"Cumulative e-beam exposure time per pixel (nominal, $\mu$s)",fontsize=fsizepl)
    #major_ticks = [25,50,75,nominal_time_on*dset.shape[0]*fastfactor]
    major_ticks = [2,4,6]
    ax1.set_xticks(major_ticks) 
    plt.xlim([nominal_time_on,nominal_time_on*dset.shape[0]*fastfactor])
    
    if plot_dark:
        dset_avg_dark = np.average(dark_dset,axis=(2,3)) #used to be np.average
        plt.plot(np.arange(1,dset.shape[0]+1)*nominal_time_on*fastfactor,np.average(dset_avg_dark,axis=1),c='k', label='_nolegend_',linewidth=lw) #in mus, in MHz
    
    plt.show()
    
def plot_expt_by_expt_behaviour_Er60(titulo, data1, data2, data3, Time_bin, nominal_time_on,fastfactor,my_color_avg,major_ticks,no_points,aper,current,plot_inset=False,x_vec=None, tau1=None,  tau1_error=None, tau2=None,  tau2_error=None): #dset is no_expt x no_time_bins x pixel x pixel; dset.shape[0] is no_expts, Time_bin in ns, nominal_time_on in mus
    
    plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')
    plt.suptitle("Cathodoluminescence (CL) as a function of e-beam exposure time, \n " + titulo, fontsize=fsizetit)
    ax1 = plt.subplot2grid((1,1), (0, 0), colspan=1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    #color=iter(cm.rainbow(np.linspace(0,1,noplots)))
    lab = aper
    #for kk in np.arange(3):
    #c = next(color)
    plt.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,data1,c='k', label=lab[0],linewidth=lw) #in mus, in MHz
    #c = next(color)
    plt.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,data2,c='r', label=lab[1],linewidth=lw+3) #in mus, in MHz
    #c = next(color)
    plt.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,data3,c='b', label=lab[2],linewidth=lw+6) #in mus, in MHz
    
    plt.ylabel(r"Average frame luminescence of each experiment (MHz) [solid lines]",fontsize=fsizepl)
    plt.xlabel(r"Cumulative e-beam exposure time per pixel (nominal, $\mu$s)",fontsize=fsizepl)
    #major_ticks = [25,50,75,nominal_time_on*dset.shape[0]*fastfactor]
    ax1.set_xticks(major_ticks[:-1]) 
    plt.xlim([nominal_time_on,nominal_time_on*no_points*fastfactor])
    plt.legend(loc='upper center')#(loc='upper left') 
    
    ax2 = ax1.twinx()
    aper2 = [30,60,120]
    #color=iter(cm.rainbow(np.linspace(0,1,noplots)))
   # for kkk in np.arange(3):
        #c = next(color)
    ax2.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,np.arange(1,no_points+1)*nominal_time_on*fastfactor* current * np.pi* np.power(aper2[0]/2.0,2),'--', c='k',linewidth=lw)
    ax2.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,np.arange(1,no_points+1)*nominal_time_on*fastfactor* current * np.pi* np.power(aper2[1]/2.0,2),'--', c='r',linewidth=lw+3)
    ax2.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,np.arange(1,no_points+1)*nominal_time_on*fastfactor* current * np.pi* np.power(aper2[2]/2.0,2),'--', c='b',linewidth=lw+6)
    ax2.set_ylabel("Electron dose $\propto$ cumulative e-beam exposure time $\cdot$ current $\cdot$ aperture area (a.u.) [dashed lines]",fontsize=fsizepl)
    ax2.set_yticks([])
    
    #ax2.yticks([])
    
    
#    if plot_dark:
#        dset_avg_dark = np.average(dark_dset,axis=(2,3)) #used to be np.average
#        plt.plot(np.arange(1,dset.shape[0]+1)*nominal_time_on*fastfactor,np.average(dset_avg_dark,axis=1),c='k', label='_nolegend_',linewidth=lw) #in mus, in MHz
    
    if plot_inset:
        a = plt.axes([.3, .5, .1, .2]) #[.3, .5, .1, .2]
        #x_vec = [0,1,2]
        labels = ['30','60', '120']
        a.errorbar(x_vec[0], tau1[0], yerr=tau1_error[0], fmt='ko',markersize=10)
        a.errorbar(x_vec[0], tau2[0], yerr=tau2_error[0], fmt='ks', markersize=5)
        a.errorbar(x_vec[1], tau1[1], yerr=tau1_error[1], fmt='ro',markersize=10)
        a.errorbar(x_vec[1], tau2[1], yerr=tau2_error[1], fmt='rs', markersize=5)
        a.errorbar(x_vec[2], tau1[2], yerr=tau1_error[2], fmt='bo',markersize=10)
        a.errorbar(x_vec[2], tau2[2], yerr=tau2_error[2], fmt='bs', markersize=5)
        plt.xticks(x_vec, labels)
        plt.ylabel('Fitted CL transient \n time constants ($\mu$s)',fontsize=fsizepl)
#        ax2.set_ylabel('Shorter time constant ($\mu$s)',fontsize=fsizepl)
        plt.xlabel(r"Aperture ($\mu$m)",fontsize=fsizepl) 
        #major_ticks = [30,60,120]
        #ax1.set_xticks(major_ticks) 
        plt.xlim([25,125])
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.xaxis.set_ticks_position('bottom')
        a.yaxis.set_ticks_position('left')
        plt.ylim([0,2.0])
        a.set_yticks([0.05, 1.0, 2.0]) #[0.05, 0.4, 0.8]

    plt.show()
    
def plot_expt_by_expt_behaviour_Er60_1over(titulo, data1, data2, data3, Time_bin, nominal_time_on,fastfactor,my_color_avg,major_ticks,no_points,aper,current,plot_inset=False,x_vec=None, tau1=None,  tau1_error=None, tau2=None,  tau2_error=None): #dset is no_expt x no_time_bins x pixel x pixel; dset.shape[0] is no_expts, Time_bin in ns, nominal_time_on in mus
    
    plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')
    plt.suptitle("Cathodoluminescence (CL) as a function of e-beam exposure time, \n " + titulo, fontsize=fsizetit)
    ax1 = plt.subplot2grid((1,1), (0, 0), colspan=1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    #color=iter(cm.rainbow(np.linspace(0,1,noplots)))
    lab = aper
    #for kk in np.arange(3):
    #c = next(color)
    plt.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,data1,c='k', label=lab[0],linewidth=lw) #in mus, in MHz
    #c = next(color)
    plt.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,data2,c='r', label=lab[1],linewidth=lw+3) #in mus, in MHz
    #c = next(color)
    plt.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,data3,c='b', label=lab[2],linewidth=lw+6) #in mus, in MHz
    
    plt.ylabel(r"Average frame luminescence of each experiment (MHz) [solid lines]",fontsize=fsizepl)
    plt.xlabel(r"Cumulative e-beam exposure time per pixel (nominal, $\mu$s)",fontsize=fsizepl)
    #major_ticks = [25,50,75,nominal_time_on*dset.shape[0]*fastfactor]
    ax1.set_xticks(major_ticks[:-1]) 
    plt.xlim([nominal_time_on,nominal_time_on*no_points*fastfactor])
    plt.legend(loc='upper center')#(loc='upper left') 
    
    ax2 = ax1.twinx()
    aper2 = [30,60,120]
    #color=iter(cm.rainbow(np.linspace(0,1,noplots)))
   # for kkk in np.arange(3):
        #c = next(color)
    ax2.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,1/(np.arange(1,no_points+1)*nominal_time_on*fastfactor* current * np.pi* np.power(aper2[0]/2.0,2)),'--', c='k',linewidth=lw)
    ax2.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,1/(np.arange(1,no_points+1)*nominal_time_on*fastfactor* current * np.pi* np.power(aper2[1]/2.0,2)),'--', c='r',linewidth=lw+3)
    ax2.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,1/(np.arange(1,no_points+1)*nominal_time_on*fastfactor* current * np.pi* np.power(aper2[2]/2.0,2)),'--', c='b',linewidth=lw+6)
    ax2.set_ylabel("1/Electron dose $\propto$ cumulative e-beam exposure time $\cdot$ current $\cdot$ aperture area (a.u.) [dashed lines]",fontsize=fsizepl)
    ax2.set_yticks([])
    
    #ax2.yticks([])
    
    
#    if plot_dark:
#        dset_avg_dark = np.average(dark_dset,axis=(2,3)) #used to be np.average
#        plt.plot(np.arange(1,dset.shape[0]+1)*nominal_time_on*fastfactor,np.average(dset_avg_dark,axis=1),c='k', label='_nolegend_',linewidth=lw) #in mus, in MHz
    
    if plot_inset:
        a = plt.axes([.3, .5, .1, .2])
        #x_vec = [0,1,2]
        labels = ['30','60', '120']
        a.errorbar(x_vec[0], tau1[0], yerr=tau1_error[0], fmt='ko',markersize=10)
        a.errorbar(x_vec[0], tau2[0], yerr=tau2_error[0], fmt='ks', markersize=5)
        a.errorbar(x_vec[1], tau1[1], yerr=tau1_error[1], fmt='ro',markersize=10)
        a.errorbar(x_vec[1], tau2[1], yerr=tau2_error[1], fmt='rs', markersize=5)
        a.errorbar(x_vec[2], tau1[2], yerr=tau1_error[2], fmt='bo',markersize=10)
        a.errorbar(x_vec[2], tau2[2], yerr=tau2_error[2], fmt='bs', markersize=5)
        plt.xticks(x_vec, labels)
        plt.ylabel('Fitted CL transient \n time constants ($\mu$s)',fontsize=fsizepl)
#        ax2.set_ylabel('Shorter time constant ($\mu$s)',fontsize=fsizepl)
        plt.xlabel(r"Aperture ($\mu$m)",fontsize=fsizepl) 
        #major_ticks = [30,60,120]
        #ax1.set_xticks(major_ticks) 
        plt.xlim([25,125])
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.xaxis.set_ticks_position('bottom')
        a.yaxis.set_ticks_position('left')
        plt.ylim([0,0.8])
        a.set_yticks([0.05, 0.4, 0.8]) 

    plt.show()
    
def calc_avg_and_plot(x_array, dset1, my_color, my_label, my_edgecolor, my_facecolor, avg_axis,area_pct,show_1_sigma = False):


    avg1 = np.average(dset1, axis = avg_axis)/area_pct
    std_a = np.std(dset1/area_pct, axis = avg_axis)

    plt.plot(x_array, avg1, c = my_color, label = my_label,linewidth=lw)
    
    if show_1_sigma:
        plt.fill_between(x_array, avg1-std_a, avg1+std_a, alpha=0.5, edgecolor=my_edgecolor, facecolor=my_facecolor)    
    
    
def plot_video_reg(titulo, dset, dset_bright, dset_dark, dset_all, se_dset, dset_dark_4d, dset_bright_4d, Time_bin,fastfactor,nominal_time_on,Pixel_size,bright_pct,dark_pct,name_str ,init_plot_no,major_ticks,unit='MHz'):
    
    gc.collect()
    
    #size of scalebar
    length_scalebar = 100.0 #in nm (1000nm == 1mum)
    scalebar_legend = '100 nm'
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size/1.0e-9)) #length_scalebar in pixel size (nm), rounded up for fairness

    no_expts = dset_all.shape[0]

    fig40= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig40.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')    
    plt.suptitle("Registration and segmentation (model: 2-GMM) of cathodoluminescence signal using SE channel, \n" + titulo,fontsize=fsizetit)
   
    gc.collect()
    ax1 = plt.subplot2grid((2,4), (0, 0), colspan=1)
    ax1.set_title('SE channel, registered \n and averaged over ' + str(no_expts) + ' expts.',fontsize=fsizepl)
    plt.imshow(se_dset,cmap=cm.Greys_r)
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')

    gc.collect()
    ax1 = plt.subplot2grid((2,4), (0, 1), colspan=1)
    ax1.set_title('CL channel, registered using SE \n and averaged over ' + str(no_expts) + ' expts.',fontsize=fsizepl)
    im = plt.imshow(dset[0,:,:],cmap='Reds') #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
  
    gc.collect()
    ax1 = plt.subplot2grid((2,4), (0,2), colspan=1)
    ax1.set_title('CL channel \n from signal pixels',fontsize=fsizepl)
    imbright = ax1.imshow(dset_bright[0,:,:],cmap='YlOrBr',vmin=0.0, vmax=np.max(dset_bright))
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor = plt.axes([box.x0, box.y0*1.1 , box.width,0.01 ])    
    cb1 = plt.colorbar(imbright, cax = axColor, orientation="horizontal",label='Photon counts (' + unit + ')',ticks=[0,np.max(dset_bright/2.0),np.max(dset_bright)])
    cb1.ax.set_xticklabels(['0',  str("{0:.1f}".format(np.max(dset_bright/2.0))),str("{0:.1f}".format(np.max(dset_bright)))])
    
    ax1 = plt.subplot2grid((2,4), (0,3), colspan=1)
    ax1.set_title('CL channel \n from background pixels',fontsize=fsizepl)
    imdark = plt.imshow(dset_dark[0,:,:],cmap='Greys',vmin=0.0, vmax=np.max(dset_dark)) 
    plt.axis('off')
    
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor2 = plt.axes([box.x0, box.y0*1.1, box.width,0.01 ])    
    cb2 = plt.colorbar(imdark, cax = axColor2, orientation="horizontal",label='Photon counts (' + unit + ')',ticks=[0,np.max(dset_dark/2.0),np.max(dset_dark)])
    cb2.ax.set_xticklabels(['0', str("{0:.1f}".format(np.max(dset_dark/2.0))),str("{0:.1f}".format(np.max(dset_dark)))])

    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)

    gc.collect()
    ax1 = plt.subplot2grid((2,4), (1, 0), colspan=2)
    #ax1.set_title(str(no_expts) + 'consecutive expts., averaged intensity for each identified area')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    imline = ax1.axvline(0*Time_bin/1e3, color='r', linestyle='--')

   
    gc.collect()
    calc_avg_and_plot(np.arange(0,dset.shape[0])*Time_bin/1e3, dset_bright, 'y', 'CL from signal pixels', '#ffff32', '#ffff66', (1,2),bright_pct,show_1_sigma=False)
    
    gc.collect()    
    calc_avg_and_plot(np.arange(0,dset.shape[0])*Time_bin/1e3, dset, 'r', 'CL from all pixels', '#ff3232', '#ff6666', (1,2),1.0 ,show_1_sigma=False)

    gc.collect()
    calc_avg_and_plot(np.arange(0,dset.shape[0])*Time_bin/1e3, dset_dark, 'k', 'CL from background pixels', '#323232', '#666666', (1,2),dark_pct,show_1_sigma=False)

    plt.ylabel(r'Average luminescence of each time bin (' + unit + ')',fontsize=fsizepl)
    plt.xlabel(r'Behaviour of e-beam during each experiment: 2-ON/5-OFF ($\mu$s)',fontsize=fsizepl)
    plt.legend() 
    major_ticks0 = [0.3, 2.3, 7]
    ax1.set_xticks(major_ticks0) 
    plt.xlim([0,7])
    plt.ylim(ymin=0)
    
    gc.collect()
    ax1 = plt.subplot2grid((2,4), (1, 2), colspan=2)
    #ax1.set_title('Intensity averaged over' + str(no_expts) + ' expts.')

    #calc_avg_and_plot(np.arange(1,dset_bright_4d.shape[0]+1)*nominal_time_on*fastfactor,  dset_bright_4d, 'y', 'CL (+ 1$\sigma$) from SE `signal\' pixels', '#ffff32', '#ffff66', (1,2,3),bright_pct,show_1_sigma=False)
    calc_avg_and_plot(np.arange(1,dset_bright_4d.shape[0]+1)*nominal_time_on*fastfactor,  dset_bright_4d, 'y', 'CL from signal pixels', '#ffff32', '#ffff66', (1,2,3),bright_pct,show_1_sigma=False)  
    gc.collect()

    calc_avg_and_plot(np.arange(1,dset_all.shape[0]+1)*nominal_time_on*fastfactor,  dset_all, 'r', 'CL from all pixels', '#ff3232', '#ff6666', (1,2,3),1.0,show_1_sigma=False)
    gc.collect()  
    
    calc_avg_and_plot(np.arange(1,dset_dark_4d.shape[0]+1)*nominal_time_on*fastfactor,dset_dark_4d, 'k', 'CL background pixels', '#323232', '#666666', (1,2,3),dark_pct,show_1_sigma=False)
    gc.collect()
    
    plt.ylabel(r'Average luminescence of each experiment (' + unit + ')',fontsize=fsizepl)
    plt.xlabel(r'Cumulative e-beam exposure time per pixel (nominal, $\mu$s)',fontsize=fsizepl)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('right')
    major_ticks = [2,25,50,75,100]
    ax1.set_xticks(major_ticks) 
    plt.xlim([nominal_time_on,nominal_time_on*dset_all.shape[0]*fastfactor])
    plt.ylim(ymin=0)
    #plt.legend() 
    
    plt.show()

    def updatered(j):
        # set the data in the axesimage object
        im.set_array(dset[j,:,:])
        imbright.set_array(dset_bright[j,:,:])
        imdark.set_array(dset_dark[j,:,:])
        imline.set_xdata(j*Time_bin/1e3)
       
        # return the artists set
        return [im,imbright,imdark,imline]
    
    gc.collect()
    import matplotlib.animation as animation


    #updatered(28)
    #save the figure

    updatered(0) # resets the plots to point 0
    anim = animation.FuncAnimation(fig40, updatered,frames=np.arange(dset.shape[0]-1), interval=50,repeat_delay=100) #interval was 10
    
    plt.show()

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800) #was 15 fps
    anim.save('ZZZ' + name_str + '-1video.avi', writer=writer)#, dpi=400)
    
    updatered(init_plot_no) # resets the plots to point 0

def plot_nonvideo_reg(titulo, se_bright, dset, dset_bright, dset_dark, dset_all, se_dset, dset_dark_4d, dset_bright_4d, Time_bin,fastfactor,nominal_time_on,Pixel_size,bright_pct,dark_pct,name_str ,init_plot_no,major_ticks,unit='MHz'):
    
    gc.collect()
    
    #size of scalebar
    length_scalebar = 100.0 #in nm (1000nm == 1mum)
    scalebar_legend = '100 nm'
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size/1.0e-9)) #length_scalebar in pixel size (nm), rounded up for fairness

    no_expts = dset_all.shape[0]

    fig40= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig40.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')    
    plt.suptitle("Registration and segmentation (model: 2-GMM) of cathodoluminescence signal using SE channel, \n" + titulo,fontsize=fsizetit)
   
    gc.collect()
    ax1 = plt.subplot2grid((2,4), (0, 0), colspan=1)
    ax1.set_title('SE channel (0.1ms per pixel)',fontsize=fsizepl)
    plt.imshow(se_dset,cmap=cm.Greys_r)
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')

    gc.collect()
    ax1 = plt.subplot2grid((2,4), (0, 1), colspan=1)
    ax1.set_title('SE channel, signal pixels',fontsize=fsizepl)
    im = plt.imshow(se_bright,cmap=cm.Greys_r) #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
  
    gc.collect()
    ax1 = plt.subplot2grid((2,4), (0,2), colspan=1)
    ax1.set_title('CL channel \n from signal pixels',fontsize=fsizepl)
    imbright = ax1.imshow(dset_bright[0,:,:],cmap='YlOrBr',vmin=0.0, vmax=np.max(dset_bright))
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor = plt.axes([box.x0, box.y0*1.1 , box.width,0.01 ])    
    cb1 = plt.colorbar(imbright, cax = axColor, orientation="horizontal",label='Photon counts (' + unit + ')',ticks=[0,np.max(dset_bright/2.0),np.max(dset_bright)])
    cb1.ax.set_xticklabels(['0',  str("{0:.1f}".format(np.max(dset_bright/2.0))),str("{0:.1f}".format(np.max(dset_bright)))])
    
    ax1 = plt.subplot2grid((2,4), (0,3), colspan=1)
    ax1.set_title('CL channel \n from background pixels',fontsize=fsizepl)
    imdark = plt.imshow(dset_dark[0,:,:],cmap='Greys',vmin=0.0, vmax=np.max(dset_dark)) 
    plt.axis('off')
    
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor2 = plt.axes([box.x0, box.y0*1.1, box.width,0.01 ])    
    cb2 = plt.colorbar(imdark, cax = axColor2, orientation="horizontal",label='Photon counts (' + unit + ')',ticks=[0,np.max(dset_dark/2.0),np.max(dset_dark)])
    cb2.ax.set_xticklabels(['0', str("{0:.1f}".format(np.max(dset_dark/2.0))),str("{0:.1f}".format(np.max(dset_dark)))])

    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)

    gc.collect()
    ax1 = plt.subplot2grid((2,4), (1, 0), colspan=2)
    #ax1.set_title(str(no_expts) + 'consecutive expts., averaged intensity for each identified area')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    imline = ax1.axvline(0*Time_bin/1e3, color='r', linestyle='--')

   
    gc.collect()
    calc_avg_and_plot(np.arange(0,dset.shape[0])*Time_bin/1e3, dset_bright, 'y', 'CL from signal pixels', '#ffff32', '#ffff66', (1,2),bright_pct,show_1_sigma=False)
    
    gc.collect()    
    calc_avg_and_plot(np.arange(0,dset.shape[0])*Time_bin/1e3, dset, 'r', 'CL from all pixels', '#ff3232', '#ff6666', (1,2),1.0 ,show_1_sigma=False)

    gc.collect()
    calc_avg_and_plot(np.arange(0,dset.shape[0])*Time_bin/1e3, dset_dark, 'k', 'CL from background pixels', '#323232', '#666666', (1,2),dark_pct,show_1_sigma=False)

    plt.ylabel(r'Average luminescence of each time bin (' + unit + ')',fontsize=fsizepl)
    plt.xlabel(r'Behaviour of e-beam during each experiment: 2-ON/5-OFF ($\mu$s)',fontsize=fsizepl)
    plt.legend() 
    major_ticks0 = [0.3, 2.3, 7]
    ax1.set_xticks(major_ticks0) 
    plt.xlim([0,7])
    plt.ylim(ymin=0)
    
    gc.collect()
    ax1 = plt.subplot2grid((2,4), (1, 2), colspan=2)
    #ax1.set_title('Intensity averaged over' + str(no_expts) + ' expts.')

    #calc_avg_and_plot(np.arange(1,dset_bright_4d.shape[0]+1)*nominal_time_on*fastfactor,  dset_bright_4d, 'y', 'CL (+ 1$\sigma$) from SE `signal\' pixels', '#ffff32', '#ffff66', (1,2,3),bright_pct,show_1_sigma=False)
    calc_avg_and_plot(np.arange(1,dset_bright_4d.shape[0]+1)*nominal_time_on*fastfactor,  dset_bright_4d, 'y', 'CL from signal pixels', '#ffff32', '#ffff66', (1,2,3),bright_pct,show_1_sigma=False)  
    gc.collect()

    calc_avg_and_plot(np.arange(1,dset_all.shape[0]+1)*nominal_time_on*fastfactor,  dset_all, 'r', 'CL from all pixels', '#ff3232', '#ff6666', (1,2,3),1.0,show_1_sigma=False)
    gc.collect()  
    
    calc_avg_and_plot(np.arange(1,dset_dark_4d.shape[0]+1)*nominal_time_on*fastfactor,dset_dark_4d, 'k', 'CL background pixels', '#323232', '#666666', (1,2,3),dark_pct,show_1_sigma=False)
    gc.collect()
    
    plt.ylabel(r'Average luminescence of each experiment (' + unit + ')',fontsize=fsizepl)
    plt.xlabel(r'Cumulative e-beam exposure time per pixel (nominal, $\mu$s)',fontsize=fsizepl)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('right')
    major_ticks = [2,25,50,75,100]
    ax1.set_xticks(major_ticks) 
    plt.xlim([nominal_time_on,nominal_time_on*dset_all.shape[0]*fastfactor])
    plt.ylim(ymin=0)
    #plt.legend() 
    
    plt.show()

    def updatered(j):
        # set the data in the axesimage object
        im.set_array(dset[j,:,:])
        imbright.set_array(dset_bright[j,:,:])
        imdark.set_array(dset_dark[j,:,:])
        imline.set_xdata(j*Time_bin/1e3)
       
        # return the artists set
        return [im,imbright,imdark,imline]
    
    gc.collect()
    import matplotlib.animation as animation


    #updatered(28)
    #save the figure

    updatered(0) # resets the plots to point 0
    anim = animation.FuncAnimation(fig40, updatered,frames=np.arange(dset.shape[0]-1), interval=50,repeat_delay=100) #interval was 10
    
    plt.show()

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800) #was 15 fps
    anim.save('ZZZ' + name_str + '-1video.avi', writer=writer)#, dpi=400)
    
    updatered(init_plot_no) # resets the plots to point 0


def plot_3_channels(se, blue, red, Pixel_size, title, scinti_channel, sample_channel, length_scalebar, scalebar_legend,unit, work_red_channel=False):
    
    #size of scalebar
    #length_scalebar = 2000.0 #in nm (1000nm == 1mum) IN NANOMETER
    #scalebar_legend = '2 $\mu$m' #IN MICRON
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size/1.0e-9)) #length_scalebar in pixel size (nm), rounded up for fairness
    
    fig40= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig40.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}') #not working
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}')  #not working
    plt.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]  #not working
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')     
    plt.suptitle(title,fontsize=fsizetit)
   
    gc.collect()
    ax1 = plt.subplot2grid((2,3), (0, 0), colspan=1)
    ax1.set_title('SE channel',fontsize=fsizepl)
    plt.imshow(se,cmap=cm.Greys_r)
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')

    gc.collect()
    ax1 = plt.subplot2grid((2,3), (0, 1), colspan=1)
    ax1.set_title('Scintillator channel, ' + scinti_channel,fontsize=fsizepl)
    im = plt.imshow(blue,cmap='Blues') #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor = plt.axes([box.x0, box.y0*1.025 , box.width,0.01 ])    
    cb1 = plt.colorbar(im, cax = axColor, orientation="horizontal",label='Photon counts ' + unit,ticks=[0,np.max(blue/2.0),np.max(blue)])
    #cb1.ax.set_xticklabels(['0',  str("{0:.1f}".format(np.max(blue/2.0))),str("{0:.1f}".format(np.max(blue)))])
  
    gc.collect()
    ax1 = plt.subplot2grid((2,3), (0,2), colspan=1)
    ax1.set_title('Sample channel, ' + sample_channel,fontsize=fsizepl)
    imbright = ax1.imshow(red,cmap='Reds')#,vmin=0.0, vmax=np.max(red))
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor = plt.axes([box.x0, box.y0*1.025 , box.width,0.01 ])    
    cb1 = plt.colorbar(imbright, cax = axColor, orientation="horizontal",label='Photon counts ' + unit,ticks=[0,np.max(red/2.0),np.max(red)])
    #cb1.ax.set_xticklabels(['0',  str("{0:.1f}".format(np.max(red/2.0))),str("{0:.1f}".format(np.max(red)))])
    
    #plt.show()
    
    if work_red_channel:
        
        import skimage
        from skimage import data, exposure
        
        #for complete code with histogram, see
        #http://scikit-image.org/docs/dev/auto_examples/plot_equalize.html
        # Load an example image
    
        # Make a greyscale out of it
        #normalize to [0,1]
        nred = (red-np.min(red))/(np.max(red)-np.min(red))
        #scale a to [x,y] [0,255]
        #range2 = y - x;
        #a = (a*range2) + x;
        redg = (nred*(255.0-0.0)) + 0.0
        
        img =   np.array(redg,dtype=np.uint8) #data.moon() will work
        #print(img.dtype)
        
        # Contrast stretching
        p2, p98 = np.percentile(img, (2, 98))
        img_rescale =  exposure.rescale_intensity(img, in_range=(p2, p98))
        
        # Equalization
        img_eq =  exposure.equalize_hist(img)
        
        # Adaptive Equalization
        img_adapteq =  exposure.equalize_adapthist(img, clip_limit=0.03)
        
        ax1 = plt.subplot2grid((2,3), (1, 0), colspan=1)
        ax1.set_title('Greyscale sample channel, \n contrast stretching' ,fontsize=fsizepl)
        im = plt.imshow(img_rescale,cmap='Reds') #or 'OrRd'
        sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
        ax1.add_artist(sbar)
        plt.axis('off')
        
        ax1 = plt.subplot2grid((2,3), (1, 1), colspan=1)
        ax1.set_title('Greyscale sample channel, \n hist. equalization' ,fontsize=fsizepl)
        im = plt.imshow(img_eq,cmap='Reds') #or 'OrRd'
        sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
        ax1.add_artist(sbar)
        plt.axis('off')
        
        ax1 = plt.subplot2grid((2,3), (1, 2), colspan=1)
        ax1.set_title('Greyscale sample channel, \n adapt. hist. equalization' ,fontsize=fsizepl)
        im = plt.imshow(img_adapteq,cmap='Reds') #or 'OrRd'
        sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
        ax1.add_artist(sbar)
        plt.axis('off')
        
def plot_3_channels_stretch(se, blue, red, Pixel_size, title, scinti_channel, sample_channel, length_scalebar, scalebar_legend,unit):
    
    fsizepl = 10
    #size of scalebar
    #length_scalebar = 2000.0 #in nm (1000nm == 1mum) IN NANOMETER
    #scalebar_legend = '2 $\mu$m' #IN MICRON
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size/1.0e-9)) #length_scalebar in pixel size (nm), rounded up for fairness
    
    fig40= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig40.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}') #not working
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}')  #not working
    plt.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]  #not working
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')     
    plt.suptitle(title,fontsize=fsizetit)
   
    gc.collect()
    ax1 = plt.subplot2grid((4,3), (0, 0), colspan=1)
    ax1.set_title('SE channel',fontsize=fsizepl)
    plt.imshow(se,cmap=cm.Greys_r)
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')

    gc.collect()
    ax1 = plt.subplot2grid((4,3), (0, 1), colspan=1)
    ax1.set_title('Scintillator channel, ' + scinti_channel,fontsize=fsizepl)
    im = plt.imshow(blue,cmap='Blues') #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
#    box = ax1.get_position()
#    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
#    axColor = plt.axes([box.x0, box.y0*1.025 , box.width,0.01 ])    
#    cb1 = plt.colorbar(im, cax = axColor, orientation="horizontal",label='Photon counts ' + unit,ticks=[0,np.max(blue/2.0),np.max(blue)])
    #cb1.ax.set_xticklabels(['0',  str("{0:.1f}".format(np.max(blue/2.0))),str("{0:.1f}".format(np.max(blue)))])
  
    gc.collect()
    ax1 = plt.subplot2grid((4,3), (0,2), colspan=1)
    ax1.set_title('Sample channel, ' + sample_channel,fontsize=fsizepl)
    imbright = ax1.imshow(red,cmap='Reds')#,vmin=0.0, vmax=np.max(red))
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
#    box = ax1.get_position()
#    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
#    axColor = plt.axes([box.x0, box.y0*1.025 , box.width,0.01 ])    
#    cb1 = plt.colorbar(imbright, cax = axColor, orientation="horizontal",label='Photon counts ' + unit,ticks=[0,np.max(red/2.0),np.max(red)])
    #cb1.ax.set_xticklabels(['0',  str("{0:.1f}".format(np.max(red/2.0))),str("{0:.1f}".format(np.max(red)))])
    
    #plt.show()
    
        
    import skimage
    from skimage import data, exposure
    
    
    # Work SE
    nred = (se-np.min(se))/(np.max(se)-np.min(se))
    redg = (nred*(255.0-0.0)) + 0.0
    img =   np.array(redg,dtype=np.uint8)
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale =  exposure.rescale_intensity(img, in_range=(p2, p98))
    # Equalization
    img_eq =  exposure.equalize_hist(img)
    # Adaptive Equalization
    img_adapteq =  exposure.equalize_adapthist(img, clip_limit=0.03)
    ax1 = plt.subplot2grid((4,3), (1, 0), colspan=1)
    ax1.set_title('Greyscale SE channel, \n contrast stretching' ,fontsize=fsizepl)
    im = plt.imshow(img_rescale,cmap='Greys') #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    ax1 = plt.subplot2grid((4,3), (1, 1), colspan=1)
    ax1.set_title('Greyscale SE channel, \n hist. equalization' ,fontsize=fsizepl)
    im = plt.imshow(img_eq,cmap='Greys') #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    ax1 = plt.subplot2grid((4,3), (1, 2), colspan=1)
    ax1.set_title('Greyscale SE channel, \n adapt. hist. equalization' ,fontsize=fsizepl)
    im = plt.imshow(img_adapteq,cmap='Greys') #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    # Work BLUE
    nred = (blue-np.min(blue))/(np.max(blue)-np.min(blue))
    redg = (nred*(255.0-0.0)) + 0.0
    img =   np.array(redg,dtype=np.uint8)
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale =  exposure.rescale_intensity(img, in_range=(p2, p98))
    # Equalization
    img_eq =  exposure.equalize_hist(img)
    # Adaptive Equalization
    img_adapteq =  exposure.equalize_adapthist(img, clip_limit=0.03)
    ax1 = plt.subplot2grid((4,3), (2, 0), colspan=1)
    ax1.set_title('Greyscale scintillator channel, \n contrast stretching' ,fontsize=fsizepl)
    im = plt.imshow(img_rescale,cmap='Blues') #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    ax1 = plt.subplot2grid((4,3), (2, 1), colspan=1)
    ax1.set_title('Greyscale scintillator channel, \n hist. equalization' ,fontsize=fsizepl)
    im = plt.imshow(img_eq,cmap='Blues') #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    ax1 = plt.subplot2grid((4,3), (2, 2), colspan=1)
    ax1.set_title('Greyscale scintillator channel, \n adapt. hist. equalization' ,fontsize=fsizepl)
    im = plt.imshow(img_adapteq,cmap='Blues') #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    # Work RED
    nred = (red-np.min(red))/(np.max(red)-np.min(red))
    redg = (nred*(255.0-0.0)) + 0.0
    img =   np.array(redg,dtype=np.uint8)
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale =  exposure.rescale_intensity(img, in_range=(p2, p98))
    # Equalization
    img_eq =  exposure.equalize_hist(img)
    # Adaptive Equalization
    img_adapteq =  exposure.equalize_adapthist(img, clip_limit=0.03)
    ax1 = plt.subplot2grid((4,3), (3, 0), colspan=1)
    ax1.set_title('Greyscale sample channel, \n contrast stretching' ,fontsize=fsizepl)
    im = plt.imshow(img_rescale,cmap='Reds') #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    ax1 = plt.subplot2grid((4,3), (3, 1), colspan=1)
    ax1.set_title('Greyscale sample channel, \n hist. equalization' ,fontsize=fsizepl)
    im = plt.imshow(img_eq,cmap='Reds') #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    ax1 = plt.subplot2grid((4,3), (3, 2), colspan=1)
    ax1.set_title('Greyscale sample channel, \n adapt. hist. equalization' ,fontsize=fsizepl)
    im = plt.imshow(img_adapteq,cmap='Reds') #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
def plot_2_channels_divide(se, blue, red, Pixel_size, title, scinti_channel, sample_channel, length_scalebar, scalebar_legend,unit, work_red_channel=False):
    
    #size of scalebar
    #length_scalebar = 2000.0 #in nm (1000nm == 1mum) IN NANOMETER
    #scalebar_legend = '2 $\mu$m' #IN MICRON
    length_scalebar_in_pixels = np.ceil(length_scalebar/(Pixel_size/1.0e-9)) #length_scalebar in pixel size (nm), rounded up for fairness
    
    fig40= plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig40.set_size_inches(1200./fig40.dpi,900./fig40.dpi)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}') #not working
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{xcolor}')  #not working
    plt.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]  #not working
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')     
    plt.suptitle(title,fontsize=fsizetit)
   
    gc.collect()
    ax1 = plt.subplot2grid((2,3), (0, 0), colspan=1)
    ax1.set_title('Scintillator/Sample counts',fontsize=fsizepl)
    im = plt.imshow(blue/red) #,cmap=cm.Greys_r)
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor = plt.axes([box.x0, box.y0*1.025 , box.width,0.01 ])    
    cb1 = plt.colorbar(im, cax = axColor, orientation="horizontal") #,ticks=[0,50,100,150,200])#[0,np.max(blue/red/2.0),np.max(blue/red)])
    #plt.clim([0,200])

    gc.collect()
    ax1 = plt.subplot2grid((2,3), (0, 1), colspan=1)
    ax1.set_title('Scintillator channel, ' + scinti_channel,fontsize=fsizepl)
    im = plt.imshow(blue,cmap='Blues') #or 'OrRd'
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor = plt.axes([box.x0, box.y0*1.025 , box.width,0.01 ])    
    cb1 = plt.colorbar(im, cax = axColor, orientation="horizontal",label='Photon counts ' + unit,ticks=[0,np.max(blue/2.0),np.max(blue)])
    #cb1.ax.set_xticklabels(['0',  str("{0:.1f}".format(np.max(blue/2.0))),str("{0:.1f}".format(np.max(blue)))])
  
    gc.collect()
    ax1 = plt.subplot2grid((2,3), (0,2), colspan=1)
    ax1.set_title('Sample channel, ' + sample_channel,fontsize=fsizepl)
    imbright = ax1.imshow(red,cmap='Reds')#,vmin=0.0, vmax=np.max(red))
    sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
    ax1.add_artist(sbar)
    plt.axis('off')
    
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0*1.00, box.width, box.height])
    axColor = plt.axes([box.x0, box.y0*1.025 , box.width,0.01 ])    
    cb1 = plt.colorbar(imbright, cax = axColor, orientation="horizontal",label='Photon counts ' + unit,ticks=[0,np.max(red/2.0),np.max(red)])
    #cb1.ax.set_xticklabels(['0',  str("{0:.1f}".format(np.max(red/2.0))),str("{0:.1f}".format(np.max(red)))])
    
    #plt.show()
    
    if work_red_channel:
        
        import skimage
        from skimage import data, exposure
        
        #for complete code with histogram, see
        #http://scikit-image.org/docs/dev/auto_examples/plot_equalize.html
        # Load an example image
    
        # Make a greyscale out of it
        #normalize to [0,1]
        nred = (red-np.min(red))/(np.max(red)-np.min(red))
        #scale a to [x,y] [0,255]
        #range2 = y - x;
        #a = (a*range2) + x;
        redg = (nred*(255.0-0.0)) + 0.0
        
        img =   np.array(redg,dtype=np.uint8) #data.moon() will work
        #print(img.dtype)
        
        # Contrast stretching
        p2, p98 = np.percentile(img, (2, 98))
        img_rescale =  exposure.rescale_intensity(img, in_range=(p2, p98))
        
        # Equalization
        img_eq =  exposure.equalize_hist(img)
        
        # Adaptive Equalization
        img_adapteq =  exposure.equalize_adapthist(img, clip_limit=0.03)
        
        ax1 = plt.subplot2grid((2,3), (1, 0), colspan=1)
        ax1.set_title('Greyscale sample channel, \n contrast stretching' ,fontsize=fsizepl)
        im = plt.imshow(img_rescale,cmap='Reds') #or 'OrRd'
        sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
        ax1.add_artist(sbar)
        plt.axis('off')
        
        ax1 = plt.subplot2grid((2,3), (1, 1), colspan=1)
        ax1.set_title('Greyscale sample channel, \n hist. equalization' ,fontsize=fsizepl)
        im = plt.imshow(img_eq,cmap='Reds') #or 'OrRd'
        sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
        ax1.add_artist(sbar)
        plt.axis('off')
        
        ax1 = plt.subplot2grid((2,3), (1, 2), colspan=1)
        ax1.set_title('Greyscale sample channel, \n adapt. hist. equalization' ,fontsize=fsizepl)
        im = plt.imshow(img_adapteq,cmap='Reds') #or 'OrRd'
        sbar = sb.AnchoredScaleBar(ax1.transData, length_scalebar_in_pixels, scalebar_legend, style = 'dark', loc = 4)
        ax1.add_artist(sbar)
        plt.axis('off')