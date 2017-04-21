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
##############################################################################
##############################################################################

def do_pic(ax3,fig1):
    
    fsizepl = 24
    fsizenb = 20
    

    ######## MEDIUM ZOOM
    # WL stands for WHOLE LIGHT
    Il_data3 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Il_data.npz')
    #il_data3 = Il_data3['data'] 
    il_data3 = np.array([24.9, 30.786938036466221, 39.654901901625777, 50.851799349638029, 60.220330334198266, 70.652507581440247])
    Il_data_std3 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Il_data_std.npz')
#    il_data_std3 = Il_data_std3['data']  
    il_data_std3 = np.array([0.1, 1.0338612506806335, 0.70564570081038591, 1.0125697282883677, 0.5736063546274861, 0.64491532738933377])
    
    Red_int_array3 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Red_int_array.npz') 
    red_int_array3 = Red_int_array3['data']
    Blue_int_array3 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Blue_int_array.npz') 
    blue_int_array3 = Blue_int_array3['data']
    
    Red_std_array3 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Red_std_array.npz') 
    red_std_array3 = Red_std_array3['data']
    Blue_std_array3 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Blue_std_array.npz') 
    blue_std_array3 = Blue_std_array3['data']
    
    bgRed_int_array3 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/bgRed_int_array.npz') 
    bgred_int_array3 = bgRed_int_array3['data']
    bgBlue_int_array3 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/bgBlue_int_array.npz') 
    bgblue_int_array3 = bgBlue_int_array3['data']
    
    bgRed_std_array3 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/bgRed_std_array.npz') 
    bgred_std_array3 = bgRed_std_array3['data']
    bgBlue_std_array3 = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/bgBlue_std_array.npz') 
    bgblue_std_array3 = bgBlue_std_array3['data']
    
    print(red_int_array3)
    print(blue_int_array3)
    
    #GOING DOWN
    FACTOR = 1.0
    
    Il_data3d = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Il_dataGOINGDOWN.npz')
    #il_data3d = Il_data3d['data']  
    # replacing with calibrated temperature values from the heating up
    il_data3d = np.array([29.767038939801189, 32.989664557794143, 41.60740810742837, 50.690450376594001, 59.078831716569162, 71.049952240480138])[::-1]
    Il_data_std3d = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Il_data_stdGOINGDOWN.npz')
    #il_data_std3d = Il_data_std3d['data']  
    il_data_std3d = np.array([0.15004144802124503, 0.13120359603133785, 0.07976927921856572, 0.10440306773732771, 0.45758747280836515, 0.74933681763885074])[::-1]
    
    Red_int_array3d = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Red_int_arrayGOINGDOWN.npz') 
    red_int_array3d = FACTOR*Red_int_array3d['data']
    Blue_int_array3d = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Blue_int_arrayGOINGDOWN.npz') 
    blue_int_array3d = FACTOR*Blue_int_array3d['data']
    
    Red_std_array3d = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Red_std_arrayGOINGDOWN.npz') 
    red_std_array3d = FACTOR*Red_std_array3d['data']
    Blue_std_array3d = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Blue_std_arrayGOINGDOWN.npz') 
    blue_std_array3d = FACTOR*Blue_std_array3d['data']
    
    ##### Try: BAKCGROUND SUBTRACT!!!!!!!!!!!!! USING GMM INTENSITY
#    red_int_array3 = red_int_array3 - bgred_int_array3
#    blue_int_array3 = blue_int_array3 - bgblue_int_array3

    ######## TOP PICS

    ###### each avg
    Il_data3EACHAVG = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Il_dataEACHAVG .npz')
    il_data3EACHAVG = Il_data3EACHAVG['data']  
    Il_data_std3EACHAVG = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Il_data_stdEACHAVG .npz')
    il_data_std3EACHAVG = Il_data_std3EACHAVG['data']  
    
    Red_int_array3EACHAVG = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Red_int_arrayEACHAVG .npz') 
    red_int_array3EACHAVG = Red_int_array3EACHAVG['data']
    Blue_int_array3EACHAVG = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Blue_int_arrayEACHAVG .npz') 
    blue_int_array3EACHAVG = Blue_int_array3EACHAVG['data']
    
    Red_std_array3EACHAVG = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Red_std_arrayEACHAVG .npz') 
    red_std_array3EACHAVG = Red_std_array3EACHAVG['data']
    Blue_std_array3EACHAVG = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/Blue_std_arrayEACHAVG .npz') 
    blue_std_array3EACHAVG = Blue_std_array3EACHAVG['data']
    
    bgRed_int_array3EACHAVG = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/bgRed_int_arrayEACHAVG .npz') 
    bgred_int_array3EACHAVG = bgRed_int_array3EACHAVG['data']
    bgBlue_int_array3EACHAVG = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/bgBlue_int_arrayEACHAVG .npz') 
    bgblue_int_array3EACHAVG = bgBlue_int_array3EACHAVG['data']
    
    bgRed_std_array3EACHAVG = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/bgRed_std_arrayEACHAVG .npz') 
    bgred_std_array3EACHAVG = bgRed_std_array3EACHAVG['data']
    bgBlue_std_array3EACHAVG = np.load('../2017-03-26_Andrea_NPs_NewTempData_ThemorcoupleOnSample+FilterNEWNEW/bgBlue_std_arrayEACHAVG .npz') 
    bgblue_std_array3EACHAVG = bgBlue_std_array3EACHAVG['data']
    
    Pixel_size = np.array([2.48]) #nm  #largest/smallest/medium
    
    RTindex = [0,0,0] #largest/smallest/medium -> RT IS INDEX 0 IN MEDIUM
    
    for index in np.arange(0,1): #Multiplication_factor.shape[0]):
         ######## DELETE POINTS
    
        if index == 0: #of course, only delete in first pass
            pass

            ################### MEDIUM ZOOM
#            todel = [0,1,2,3]#  using [4,5,6,7]
#            red_int_array3 = np.delete(red_int_array3, todel)
#            blue_int_array3 = np.delete(blue_int_array3, todel)
#            red_std_array3 = np.delete(red_std_array3, todel)
#            blue_std_array3 = np.delete(blue_std_array3, todel)
#            il_data3 = np.delete(il_data3, todel)
#            il_data_std3 = np.delete(il_data_std3, todel)

        ##################################################### PAGE 5
       
        x_vec3 = il_data3
        x_vec3d = il_data3d
        
        legendloc = 'best'

        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.xaxis.set_ticks_position('bottom')
        ax3.yaxis.set_ticks_position('left')
        
        Notr = np.zeros([6])
        Nopointsbeamon = 152
        #From file get_no_signal_pixels_only ->>>>>> THIS NEEDS TO BE REDONE
        No_signal = [38921.0,29452.0,29608.0,34650.0,33207.0,37710.0]
        No_bg = [41159.0,34895.0,30914.0,33492.0,32320.0,36712.0]

        Notr[0] = 5.0*No_signal[0] * Nopointsbeamon
        Notr[1] = 5.0*No_signal[1] * Nopointsbeamon
        Notr[2] = 5.0*No_signal[2] * Nopointsbeamon
        Notr[3] = 5.0*No_signal[3] * Nopointsbeamon
        Notr[4] = 5.0*No_signal[4] * Nopointsbeamon
        Notr[5] = 5.0*No_signal[5] * Nopointsbeamon
        
        ured_int_array3 = unumpy.uarray(red_int_array3,np.sqrt(red_int_array3)/np.sqrt(Notr))
        ublue_int_array3 = unumpy.uarray(blue_int_array3,np.sqrt(blue_int_array3)/np.sqrt(Notr))
        yerr3 = unumpy.std_devs( +((ublue_int_array3-ured_int_array3)/(ured_int_array3+ublue_int_array3))/((blue_int_array3[RTindex[2]]-red_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]+blue_int_array3[RTindex[2]])))    #########THERE WAS A MINUS!!!!!!!!
        
        ###RATIO OF INTENSITIES
        #normalized
        print(x_vec3)
        ratiu = ((red_int_array3)/(blue_int_array3))/((red_int_array3[RTindex[2]])/(blue_int_array3[RTindex[2]]))
        ratiuEACHAVG = ((red_int_array3EACHAVG.flatten())/(blue_int_array3EACHAVG.flatten()))/((red_int_array3EACHAVG.flatten()[0])/(blue_int_array3EACHAVG.flatten()[0]))
        #bgratiu = ((bgred_int_array3)/(bgblue_int_array3))/((bgred_int_array3[RTindex[2]])/(bgblue_int_array3[RTindex[2]]))
        
        ratiud = ((red_int_array3d)/(blue_int_array3d))/((red_int_array3[0])/(blue_int_array3[0]))
        print('ratiud')
        print(ratiud)
        ax3.errorbar( x_vec3, ratiu, xerr=il_data_std3, marker='d',markersize=12,ls='',color='r',label='Ratio of intensities',markeredgecolor='None')
        ax3.errorbar( x_vec3d, ratiud, xerr=il_data_std3d, marker='d',markersize=12,ls='',color='b',markeredgecolor='None')

        #ax3.errorbar( il_data3EACHAVG.flatten(), ratiuEACHAVG, xerr=il_data_std3EACHAVG.flatten(), marker='d',markersize=12,ls='',color='k',label='Ratio of intensities',markeredgecolor='None')
        #PLOT INSETS
        left, bottom, width, height = [0.42, 0.55, 0.05,0.05]
        axinset1 = fig1.add_axes([left, bottom, width, height])
        axinset1.plot( il_data3EACHAVG.flatten()[3::6], ratiuEACHAVG[3::6], marker='d',markersize=12,ls='',color='r',label='Ratio of intensities',markeredgecolor='None')
        #axinset1.spines['left'].set_visible(False)
        #axinset1.spines['top'].set_visible(False)
        axinset1.xaxis.set_ticks_position('bottom')
        axinset1.yaxis.set_ticks_position('right')
        axinset1.set_ylabel(r'$\Delta$S = 0.1',fontsize=fsizepl)
        axinset1.set_xlabel(r'$\Delta$T $=$ 2 $^{\circ}$C',fontsize=fsizepl)
        
        label = axinset1.xaxis.get_label()
        x_lab_pos, y_lab_pos = label.get_position()
        label.set_position([0.0, y_lab_pos])
        label.set_horizontalalignment('left')
        axinset1.xaxis.set_label(label)
        
        labely = axinset1.yaxis.get_label()
        x_lab_posy, y_lab_posy = labely.get_position()
        labely.set_position([0.0,0.73])
        labely.set_verticalalignment('baseline')
        axinset1.yaxis.set_label(labely)
        
        axinset1.tick_params(labelsize=fsizenb)
        axinset1.set_ylim([1.24,1.34])
        axinset1.set_xlim([50.1, 52.1])
        axinset1.set_xticks([])
        axinset1.set_yticks([])
        ax3.annotate('', xy=(52,1.25), xytext=(58,1.13),
                arrowprops=dict(facecolor='black', shrink=0.05))  
        
        left, bottom, width, height = [0.48, 0.625, 0.05,0.05]
        axinset11 = fig1.add_axes([left, bottom, width, height])
        axinset11.plot( il_data3EACHAVG.flatten()[5::6], ratiuEACHAVG[5::6], marker='d',markersize=12,ls='',color='r',label='Ratio of intensities',markeredgecolor='None')
        #axinset11.spines['right'].set_visible(False)
        #axinset11.spines['top'].set_visible(False)
        axinset11.xaxis.set_ticks_position('bottom')
        axinset11.yaxis.set_ticks_position('left')
        axinset11.set_ylabel(r'$\Delta$S',fontsize=fsizepl)
        
        labely = axinset11.yaxis.get_label()
        x_lab_posy, y_lab_posy = labely.get_position()
        labely.set_position([0.0,0.5])
        labely.set_verticalalignment('baseline')
        axinset11.yaxis.set_label(labely)
        
        
        
        axinset11.set_xlabel('$\Delta$T',fontsize=fsizepl)
        axinset11.tick_params(labelsize=fsizenb)
        axinset11.set_yticks([])
        axinset11.set_xlim([69.5, 71.5])
        axinset11.set_xticks([])
        axinset11.set_ylim([1.435, 1.535])
        ax3.annotate('', xy=(70.5,1.44), xytext=(70.5,1.37),
                arrowprops=dict(facecolor='black', shrink=0.05)) 

        #ax3.plot( x_vec3[6:], ratiu[6:], marker='v',markersize=12,ls='',color='k',label='Ratio of intensities',markeredgecolor='None')

        #unnormalized
        #ax3.plot( x_vec3, ((red_int_array3)/(blue_int_array3)), marker='d',markersize=12,ls='',color='k',label='Ratio of intensities',markeredgecolor='None')

        ##### USUAL VISIBILITY
        #error bars smaller than marker
        #ax3.errorbar( x_vec3, ((blue_int_array3-red_int_array3)/(red_int_array3+blue_int_array3))/((blue_int_array3[RTindex[2]]-red_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]+blue_int_array3[RTindex[2]]))*Multiplication_factor[index,2], yerr= yerr3, marker='o',markersize=12,ls='',color='k',label='Visibility of intensities')
        #normalized

        #!!!!!!!uncomment
        ###############################################################################
#        visibu = ((blue_int_array3-red_int_array3)/(red_int_array3+blue_int_array3))/((blue_int_array3[RTindex[2]]-red_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]+blue_int_array3[RTindex[2]]))
#        bgvisibu = ((bgblue_int_array3-bgred_int_array3)/(bgred_int_array3+bgblue_int_array3))/((bgblue_int_array3[RTindex[2]]-bgred_int_array3[RTindex[2]])/(bgred_int_array3[RTindex[2]]+bgblue_int_array3[RTindex[2]]))
#        
#        visibud = ((blue_int_array3d-red_int_array3d)/(red_int_array3d+blue_int_array3d))/((blue_int_array3[0]-red_int_array3[0])/(red_int_array3[0]+blue_int_array3[0]))
#
#        print('visibu')
#        print(visibu)
#        ax3.errorbar( x_vec3, visibu, xerr=il_data_std3, marker='o',markersize=12,ls='',color='r',label='Visibility of intensities',markeredgecolor='None')
#        ax3.errorbar( x_vec3d, visibud, xerr=il_data_std3d, marker='o',markersize=12,ls='',color='b',markeredgecolor='k')
#        ###################################################################################
        
        #ax3.plot( x_vec3[6:], visibu[6:], marker='<',markersize=12,ls='',color='k',label='Visibility of intensities',markeredgecolor='None')
  
        #unnormalizaed
        #ax3.plot( x_vec3, ((blue_int_array3-red_int_array3)/(red_int_array3+blue_int_array3)), marker='o',markersize=12,ls='',color='k',label='Visibility of intensities',markeredgecolor='None')

        #ax3.plot( x_vec3[0:6], bgratiu[0:6], marker='d',markersize=12,ls='',color='white',label='...background',markeredgecolor='k')
        #ax3.plot( x_vec3[0:6], bgvisibu[0:6], marker='o',markersize=12,ls='',color='white',label='...background',markeredgecolor='k')

        #######INDIV CHANNELS ##### NEXT 4 LINES WORK TO PLOT INDIVIDUAL
        redu = ((red_int_array3))/((red_int_array3[RTindex[2]]))
        bluu = ((blue_int_array3))/((blue_int_array3[RTindex[2]]))
        #ax3.plot( x_vec3[0:6], redu[0:6], marker='^',markersize=12,ls='',color='r',markeredgecolor='None')
        #ax3.plot( x_vec3[0:6], bluu[0:6], marker='^',markersize=12,ls='',color='g',markeredgecolor='None')
       
        x_vec3B = np.linspace(x_vec3[0], x_vec3[-2], 100)
        x_vec3Bd = np.linspace(x_vec3d[-1], x_vec3d[1], 100)
        ######!!!!!!UNCOMMMENT!!!!!!!!!!!!!!!
        #######FITTING WITH INVERSE OF LINEAR RATIO
#        def d_visi(x, aa):
#            return -1.0 + 2.0/(1.0 + (-1-aa*x[0] + 2.0/(1.0 +1.0)) + aa * np.array(x))
#            
#        (a2,result2) = visi_fit_fixed_point(x_vec3,((blue_int_array3-red_int_array3)/(red_int_array3+blue_int_array3))/((blue_int_array3[RTindex[2]]-red_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]+blue_int_array3[RTindex[2]])),x_vec3[0], 1.0)
#        ax3.plot(np.array(x_vec3),-1.0 + 2.0/(1.0 + (-1-a2*x_vec3[0] + 2.0/(1.0 +1.0)) + a2 * np.array(x_vec3)),color='k',lw=2)
#        
#        sigma_dev2 = np.sqrt([result2.covar[0,0]]) # sqrt(diag elements) of pcov are the 1 sigma deviations
#        values2 = np.array([])
#        for s1 in [-1, +1]:
#                    my_hlp2 = d_visi( x_vec3B, result2.params['a'].value + s1*sigma_dev2[0])
#        values2 = np.vstack((values2, my_hlp2)) if values2.size else my_hlp2
#        fitError2 = np.std(values2, axis=0) 
#
#        ax3.fill_between(x_vec3B,  
#                         d_visi(x_vec3B, a2)-1.0*fitError2,
#                         d_visi(x_vec3B, a2)+1.0*fitError2,  
#                         color =[168/256,175/256,175/256],
#                         edgecolor='k',
#                         facecolor=[168/256,175/256,175/256],
#                         alpha=0.5,
#                         linewidth=0.0)
#                         
#        print('new model for visibility')
#        print('a=' + str(result2.params['a'].value))
        #######END OF FITTING WITH INVERSE OF LINEAR RATIO

        ax3.set_ylabel('Intensity thermometry signal, \n norm. to $\sim$25 $^{\circ}$C (a.u.)',fontsize=fsizepl)
        ax3.tick_params(labelsize=fsizenb)
        ax3.yaxis.set_label_position("left")
        
        ax3.set_xlim([20,75])
        ax3.set_xticks([25,30,40,50,60,70])
        ax3.set_ylim(ymin = 0.9, ymax = 1.6)
       
        ax3.set_yticks([1,1.5])
        #ax3.legend(loc='upper left',frameon=False, fontsize=fsizenb,numpoints=1)
#        # get handles
        #handles3, labels3 = ax3.get_legend_handles_labels()
#        # remove the errorbars
        #handles3 = [h[0] for h in handles3]
#        # use them in the legend
        #ax3.legend(handles3, labels3, loc='best',numpoints=1,frameon=False, fontsize=fsizenb)
    
        ax3.set_xlabel('Temperature at sample ($^{\circ}$C)',fontsize=fsizepl)
        ax3.axhline(y=1.0 , lw=2, color='k', ls='--')
        
#        #Fitting ration
        def d_parab(x,aa,bb,offs):
            return aa*(x-x[0])**2 + b*(x-x[0]) + offs
            
        def d_visibparab(x,aa,bb,offs):
            return -1.0 + 2.0/(1+offs+aa*(x)**2 + b*(x))
            
        def d_parabdown(x,aa,bb,offs):
            return aa*(x-x[-1])**2 + b*(x-x[-1]) + offs
            
        def d_paraball(x,aa,bb,c):
            return aa*(x)**2 + b*(x) + c

        def d_line(x, aa):
            return aa*(x-x[0])+1.0
        
        #LINEAR FIT
#        (a,result) = linear_fit_fixed_point(x_vec3, ((red_int_array3)/(blue_int_array3))/((red_int_array3[RTindex[2]])/(blue_int_array3[RTindex[2]])), x_vec3[0], 1.0)
        #ax3.plot(np.array(x_vec3),a*(np.array(x_vec3)-x_vec3[0])+1.0,color='k',lw=2)
        
        #PARABOLA FIT
        (a,b, result) = parabola_fit_fixed_point(x_vec3[:-1], ((red_int_array3[:-1])/(blue_int_array3[:-1]))/((red_int_array3[RTindex[2]])/(blue_int_array3[RTindex[2]])), x_vec3[0], 1.0)
        ax3.plot(np.array(x_vec3[:-1]),d_parab(x_vec3[:-1],a,b,1.0),color='r',lw=2)
        ax3.plot(np.array(x_vec3[-2:]),((red_int_array3[-2:])/(blue_int_array3[-2:]))/((red_int_array3[RTindex[2]])/(blue_int_array3[RTindex[2]])),ls='dashed',color='r',lw=2)
        #choice between parabola and line
#        df1 = 7 - 1 #linear, simpler model #7points, 1 dof
#        df2 = 6 - 2 #parabola #6points, 1 dof
#        print('F test parabola vs line')    
#        SS1 =  result.chisqr
#        SS2  = resultp.chisqr
#        print(((SS1 - SS2)/(df1-df2))/(SS2/df2))  #=17.52 ---> Reject null hypothesis (ie, that simpler model is correct) -> is parabola
#        kklll
        
        #ERROR LINE
#        sigma_dev = np.sqrt([result.covar[0,0]]) # sqrt(diag elements) of pcov are the 1 sigma deviations
#        values = np.array([])
#        for s1 in [-1, +1]:
#                    my_hlp = d_line( x_vec3B, a + s1*sigma_dev[0])
#        values = np.vstack((values, my_hlp)) if values.size else my_hlp
#        fitError = np.std(values, axis=0) 
#        
#        ax3.fill_between(x_vec3B,  
#                         d_line(x_vec3B, a)-1.0*fitError,
#                         d_line(x_vec3B, a)+1.0*fitError,  
#                         color =[168/256,175/256,175/256],
#                         edgecolor='k',
#                         facecolor=[168/256,175/256,175/256],
#                         alpha=0.5,
#                         linewidth=0.0)
#        print('ratio linear fit')
#        print('a=' + str(result.params['a'].value))
        
#        #error parabola
        sigma_dev = np.sqrt([result.covar[0,0],result.covar[1,1]]) # sqrt(diag elements) of pcov are the 1 sigma deviations
        values = np.array([])
        for s1 in [-1, +1]:
            for ss1 in [-1, +1]:
                    my_hlp = d_parab( x_vec3B, a + s1*sigma_dev[0], b + ss1*sigma_dev[1], 1.0)
                    values = np.vstack((values, my_hlp)) if values.size else my_hlp
        fitError = np.std(values, axis=0) 

        ax3.fill_between(x_vec3B,  
                         d_parab(x_vec3B, a,b,1.0)-1.0*fitError,
                         d_parab(x_vec3B, a,b,1.0)+1.0*fitError,  
                         color ='r' , #was gray [168/256,175/256,175/256]
                         edgecolor='r',
                         facecolor='r', #was gray [168/256,175/256,175/256],
                         alpha=0.25,
                         linewidth=0.0)
                         
        print('a,b, parab up')
        print('a=' + str(result.params['a'].value))
        print('b=' + str(result.params['b'].value))
        
        #### plot for visibilit d_visibparab
        #ax3.plot(np.array(x_vec3B[:-1]),d_visibparab(x_vec3B[:-1],a,b,1.0),color='c',lw=2)

        #PARABOLA DOWN
        #offsd =  ((red_int_array3d[-1])/(blue_int_array3d[-1]))/((red_int_array3[RTindex[2]])/(blue_int_array3[RTindex[2]]))
        parabvalues = ((red_int_array3d[1:])/(blue_int_array3d[1:]))/((red_int_array3[RTindex[2]])/(blue_int_array3[RTindex[2]]))
        (ad,bd, cd, resultd) = parabola_fit(x_vec3d[1:], parabvalues)
        ax3.plot(np.array(x_vec3d[1:]),ad.value*(x_vec3d[1:])**2 + bd.value*(x_vec3d[1:])+ cd.value,color='b',lw=2)
        #plot 2 remaining points
        ax3.plot(np.array(x_vec3d[0:2]),((red_int_array3d[0:2])/(blue_int_array3d[0:2]))/((red_int_array3[RTindex[2]])/(blue_int_array3[RTindex[2]])),ls='dashed',color='b',lw=2)
        
        #error parabola
        sigma_devd = np.sqrt([resultd.covar[0,0],resultd.covar[1,1],resultd.covar[2,2]]) # sqrt(diag elements) of pcov are the 1 sigma deviations
        values = np.array([])
        for s1 in [-1, +1]:
            for ss1 in [-1, +1]:
                for sss1 in [-1, +1]:
                    my_hlp = d_paraball( x_vec3Bd, ad + s1*sigma_devd[0], bd + ss1*sigma_devd[1], cd + sss1*sigma_devd[2]) 
                    values = np.vstack((values, my_hlp)) if values.size else my_hlp
        fitError = np.std(values, axis=0) 
        
        

#        def d_paraball(x,aa,bb,c):
#            return aa*(x)**2 + b*(x) + c

#ad.value*(x_vec3Bd)**2 + bd.value*(x_vec3Bd) + cd.value-1.0*fitError,
#ad.value*(x_vec3Bd)**2 + bd.value*(x_vec3Bd) + cd.value+1.0*fitError,  

        ax3.fill_between(x_vec3Bd,
                        #d_paraball( x_vec3Bd, ad.value, bd.value, cd.value ) + 1.0*fitError,
                        #d_paraball( x_vec3Bd, ad.value, bd.value, cd.value ) + 1.0*fitError,
                         ad.value*(x_vec3Bd)**2 + bd.value*(x_vec3Bd) + cd.value+1.0*fitError,
                         ad.value*(x_vec3Bd)**2 + bd.value*(x_vec3Bd) + cd.value-1.0*fitError, 
                         color ='b', #[168/256,175/256,175/256],
                         edgecolor='b',
                         facecolor='b', #[168/256,175/256,175/256],
                         alpha=0.25,
                         linewidth=0.0)
                         
        print('a,b, parab down')
        print('a=' + str(resultd.params['a'].value))
        print('b=' + str(resultd.params['b'].value))
        
        axinset1.plot(np.array(x_vec3[:-1]),d_parab(x_vec3[:-1],a,b,1.0),color='r',lw=2)
        axinset11.plot(np.array(x_vec3[:-1]),d_parab(x_vec3[:-1],a,b,1.0),color='r',lw=2)
#        ##### End of Fitting ratio 
        
#        from minepy import MINE
#        from scipy.stats import pearsonr, spearmanr
#  
#        
#        mine = MINE(alpha=0.6, c=15)
#        mine.compute_score(x_vec3, ((blue_int_array3-red_int_array3)/(red_int_array3+blue_int_array3))/((blue_int_array3[RTindex[2]]-red_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]+blue_int_array3[RTindex[2]])))
#        rp = pearsonr(x_vec3,      ((blue_int_array3-red_int_array3)/(red_int_array3+blue_int_array3))/((blue_int_array3[RTindex[2]]-red_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]+blue_int_array3[RTindex[2]])))
#        rs = spearmanr(x_vec3,     ((blue_int_array3-red_int_array3)/(red_int_array3+blue_int_array3))/((blue_int_array3[RTindex[2]]-red_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]+blue_int_array3[RTindex[2]])))
#        import scipy        
#        dc = scipy.spatial.distance.correlation(x_vec3,     ((blue_int_array3-red_int_array3)/(red_int_array3+blue_int_array3))/((blue_int_array3[RTindex[2]]-red_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]+blue_int_array3[RTindex[2]])))
#        import sklearn
#        mi = sklearn.metrics.mutual_info_score(x_vec3,     ((blue_int_array3-red_int_array3)/(red_int_array3+blue_int_array3))/((blue_int_array3[RTindex[2]]-red_int_array3[RTindex[2]])/(red_int_array3[RTindex[2]]+blue_int_array3[RTindex[2]])))
#        print('Visib')
#        print(mi)
#        print(rs)
#        print(rp)
#        print(dc)
#        print(mine.mic())
#        
#        mine2 = MINE(alpha=0.6, c=15)
#        mine2.compute_score(x_vec3, ((red_int_array3)/(blue_int_array3))/((red_int_array3[RTindex[2]])/(blue_int_array3[RTindex[2]])))
#        rp2 = pearsonr(x_vec3,      ((red_int_array3)/(blue_int_array3))/((red_int_array3[RTindex[2]])/(blue_int_array3[RTindex[2]])))
#        rs2 = spearmanr(x_vec3,     ((red_int_array3)/(blue_int_array3))/((red_int_array3[RTindex[2]])/(blue_int_array3[RTindex[2]])))
#        import scipy        
#        dc2 = scipy.spatial.distance.correlation(x_vec3,     ((red_int_array3)/(blue_int_array3))/((red_int_array3[RTindex[2]])/(blue_int_array3[RTindex[2]])))
#        import sklearn
#        mi2 = sklearn.metrics.mutual_info_score(x_vec3,     ((red_int_array3)/(blue_int_array3))/((red_int_array3[RTindex[2]])/(blue_int_array3[RTindex[2]])))
#        print('Ratio')
#        print(mi2)
#        print(rs2)
#        print(rp2)
#        print(dc2)
#        print(mine2.mic())
        
 
        
        #######################################################################

        #### Arrows explaining temp um and down
        ax3.annotate('', xy=(47,1.175), xytext=(41,1.1),
                arrowprops=dict(facecolor='red', shrink=0.05,edgecolor ='None'))  
        ax3.text(40, 1.05, 'ramping up', fontsize=fsizenb, va='center',ha='center',color='r') 
        
        ax3.annotate('', xy=(38,1.375), xytext=(44,1.45),
                arrowprops=dict(facecolor='blue', shrink=0.05,edgecolor ='None'))  
        ax3.text(45, 1.5, 'ramping down', fontsize=fsizenb, va='center',ha='center',color='b') 

    
        ###REAL!!!!!
        #return result2, result
        ###MOCK! TEMPORARY
        return result, resultd
