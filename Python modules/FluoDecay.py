import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GMM 
import matplotlib.cm as cm
import scipy.ndimage as ndimage
#from matplotlib_scalebar.scalebar import ScaleBar
import lmfit
from lmfit import minimize, Parameters, Parameter, report_fit
from MakePdf import *

sys.path.append("/usr/bin") # necessary for the tex fonts

#ORIGINAL
#fsizetit = 18
#fsizepl = 16
#sizex = 8
#sizey = 6
#dpi_no = 80
#lw = 2

fsizetit = 18
fsizepl = 16
sizex = 8
sizey = 6
dpi_no = 80
lw = 2

def d_exp(x, a, b, c, d, e):
    return a*np.exp(-x/b) + c + d*np.exp(-x/e)
    
def d_exp1(x, a, b, c):
    return a*np.exp(-x/b) + c 
 
def mean_of_nonzero(arr, binary_mask):
    # takes the 2D array and return the mean of all non-zero values
 
    hlp = (binary_mask)*arr[:,:]
      
    hlp_non_zero = np.nonzero(hlp)[0]
     
    # check if hlp_non_zero is empty or not
    if hlp_non_zero.size:
        return np.mean(arr[hlp_non_zero])
    else:
        return 0.0
         
# define objective function: returns the array to be minimized
def fcn2min1(params, x, data, return_plot = False, no_of_x_pts = 100):
 
    a = params['a'].value
    b = params['b'].value
    c = params['c'].value
 
    if return_plot == True:
        # changing x to give more values
        x = np.linspace(np.min(x), np.max(x), no_of_x_pts)
         
    model = a*np.exp(-x/b) + c   
      
    if return_plot == False:
        return model - data
    else:
        return (x, model)
              
def fcn2min2(params, x, data, return_plot = False, no_of_x_pts = 100):
 
    a = params['a'].value
    b = params['b'].value
    c = params['c'].value
     
    d = params['d'].value
    e = params['e'].value
 
    if return_plot == True:
        # changing x to give more values
        x = np.linspace(np.min(x), np.max(x), no_of_x_pts)
         
    model = a*np.exp(-x/b) + c + d*np.exp(-x/e)
        #model = a*np.exp(-x/b) + c + (data[0]-a)*np.exp(-x/e)   
      
    if return_plot == False:
        return model - data
    else:
        return (x, model)
         
def fitexp(x,y,single,my_color,my_edgecolor,my_facecolor,init_guess=None,plot_error=True,do_plot=True):
    
   
    # create a set of Parameters
    params = Parameters()
    if init_guess is not None:
        params.add('a', value= init_guess[0], min=init_guess[0]/10.0,max=init_guess[0]*10.0)
        params.add('b', value= init_guess[1], min=init_guess[1]/10.0,max=init_guess[1]*10.0)
        params.add('c', value= init_guess[2], min=init_guess[2]/10.0,max=init_guess[2]*10.0)
        if single == False:
            print('HERE')
            params.add('d', init_guess[3], min=init_guess[3]/10.0,max=init_guess[3]*10.0)
            params.add('e', init_guess[4], min=init_guess[4]/10.0,max=init_guess[4]*10.0) 
    else:
        params.add('a', value= 3700.0, min=25.0,max=4500.0)
        params.add('b', value= 1.0, min=0.1,max=15.0)
        params.add('c', value= 100.0, min=3.0,max=400.0)
        if single == False:
            print('HERE')
            params.add('e', value= 9.0, min=0.1,max=15.0)
            params.add('d', value= 218.0, min=1.0,max=230.0) #vary=False)
            
    if single:
         result1 = minimize(fcn2min1, params, args=(x,y)) #, xtol=1e-12) #just added xtol, not needed
         (x_fit, y_fit) = fcn2min1(result1.params, x, y, return_plot = True, no_of_x_pts = 100)
    else:
         result1 = minimize(fcn2min2, params, args=(x,y)) #, xtol=1e-12) #just added xtol, not needed
         (x_fit, y_fit) = fcn2min2(result1.params, x, y, return_plot = True, no_of_x_pts = 100)
   
    
    if single == False and plot_error == True: 
        sigma_dev = np.sqrt( [result1.covar[0,0],result1.covar[1,1], result1.covar[2,2],result1.covar[3,3],result1.covar[4,4]] ) # sqrt(diag elements) of pcov are the 1 sigma deviations
         
        values = np.array([
        d_exp( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value + sigma_dev[2], result1.params['d'].value + sigma_dev[3], result1.params['e'].value +  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value + sigma_dev[2], result1.params['d'].value + sigma_dev[3], result1.params['e'].value -  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value + sigma_dev[2], result1.params['d'].value - sigma_dev[3], result1.params['e'].value +  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value + sigma_dev[2], result1.params['d'].value - sigma_dev[3], result1.params['e'].value -  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value - sigma_dev[2], result1.params['d'].value + sigma_dev[3], result1.params['e'].value +  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value - sigma_dev[2], result1.params['d'].value + sigma_dev[3], result1.params['e'].value -  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value - sigma_dev[2], result1.params['d'].value - sigma_dev[3], result1.params['e'].value +  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value - sigma_dev[2], result1.params['d'].value - sigma_dev[3], result1.params['e'].value -  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value + sigma_dev[2], result1.params['d'].value + sigma_dev[3], result1.params['e'].value +  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value + sigma_dev[2], result1.params['d'].value + sigma_dev[3], result1.params['e'].value -  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value + sigma_dev[2], result1.params['d'].value - sigma_dev[3], result1.params['e'].value +  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value + sigma_dev[2], result1.params['d'].value - sigma_dev[3], result1.params['e'].value -  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value - sigma_dev[2], result1.params['d'].value + sigma_dev[3], result1.params['e'].value +  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value - sigma_dev[2], result1.params['d'].value + sigma_dev[3], result1.params['e'].value -  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value - sigma_dev[2], result1.params['d'].value - sigma_dev[3], result1.params['e'].value +  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value - sigma_dev[2], result1.params['d'].value - sigma_dev[3], result1.params['e'].value -  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value + sigma_dev[2], result1.params['d'].value + sigma_dev[3], result1.params['e'].value +  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value + sigma_dev[2], result1.params['d'].value + sigma_dev[3], result1.params['e'].value -  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value + sigma_dev[2], result1.params['d'].value - sigma_dev[3], result1.params['e'].value +  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value + sigma_dev[2], result1.params['d'].value - sigma_dev[3], result1.params['e'].value -  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value - sigma_dev[2], result1.params['d'].value + sigma_dev[3], result1.params['e'].value +  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value - sigma_dev[2], result1.params['d'].value + sigma_dev[3], result1.params['e'].value -  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value - sigma_dev[2], result1.params['d'].value - sigma_dev[3], result1.params['e'].value +  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value - sigma_dev[2], result1.params['d'].value - sigma_dev[3], result1.params['e'].value -  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value + sigma_dev[2], result1.params['d'].value + sigma_dev[3], result1.params['e'].value +  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value + sigma_dev[2], result1.params['d'].value + sigma_dev[3], result1.params['e'].value -  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value + sigma_dev[2], result1.params['d'].value - sigma_dev[3], result1.params['e'].value +  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value + sigma_dev[2], result1.params['d'].value - sigma_dev[3], result1.params['e'].value -  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value - sigma_dev[2], result1.params['d'].value + sigma_dev[3], result1.params['e'].value +  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value - sigma_dev[2], result1.params['d'].value + sigma_dev[3], result1.params['e'].value -  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value - sigma_dev[2], result1.params['d'].value - sigma_dev[3], result1.params['e'].value +  sigma_dev[4]), 
        d_exp( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value - sigma_dev[2], result1.params['d'].value - sigma_dev[3], result1.params['e'].value -  sigma_dev[4])
        ])
    
    if single == True and plot_error == True:
        
        sigma_dev = np.sqrt( [result1.covar[0,0],result1.covar[1,1], result1.covar[2,2]] )
        values = np.array([
        d_exp1( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value + sigma_dev[2]), 
        d_exp1( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value - sigma_dev[2]), 
        d_exp1( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value + sigma_dev[2]), 
        d_exp1( x_fit, result1.params['a'].value + sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value - sigma_dev[2]), 
        d_exp1( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value + sigma_dev[2]), 
        d_exp1( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value + sigma_dev[1],result1.params['c'].value - sigma_dev[2]), 
        d_exp1( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value + sigma_dev[2]), 
        d_exp1( x_fit, result1.params['a'].value - sigma_dev[0],result1.params['b'].value - sigma_dev[1],result1.params['c'].value - sigma_dev[2])
        ])
    
    # the fit error represents the standard deviation of all the possible fit +- uncertainty
    # values at each x position. One could imagine getting the min and max possible deviations,
    # but this is a one-line command that is pretty sweetly simple and fast.
    #print fitError
    if plot_error == True:
        fitError = np.std(values, axis=0)     
     
    report_fit(result1.params)
    if do_plot:
        plt.hold(True)
        plt.semilogy(x_fit, y_fit, my_color, linewidth=lw)
        if plot_error:
            plt.fill_between(x_fit, y_fit-3*fitError, y_fit+3*fitError, alpha=0.5, edgecolor=my_edgecolor, facecolor= my_facecolor)   #plotting 3sigma error
            plt.yscale('log')

    
    if single == True:
        return (result1.params['a'].value,result1.params['b'].value,result1.params['c'].value,result1.params['a'].stderr,result1.params['b'].stderr,result1.params['c'].stderr)
    else:
        return (result1.params['a'].value,result1.params['b'].value,result1.params['c'].value,result1.params['d'].value,result1.params['e'].value,result1.params['a'].stderr,result1.params['b'].stderr,result1.params['c'].stderr,result1.params['d'].stderr,result1.params['e'].stderr)
     
#START OF LOOP
def calcdecay(blue_dset,time_detail,titulo,single,other_dset1=None, other_dset2=None, init_guess=None,unit='kHz'): #blue_dset has the transient data only
    No_specimen = 1
    #No_decaypts = blue_dset.shape[0]
   
    x_array = np.arange(0,blue_dset.shape[0])*time_detail
    
    #plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    fig, ax = plt.subplots()#figsize=(sizex, sizey), dpi=dpi_no)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')
    plt.suptitle(titulo, fontsize=fsizetit)
    
    plt.hold(True)
    # for kkk in range(No_specimen):   
      
    # dofitting
    # do fit, here with leastsq model
    if single == False:
        (a,b,c,d,e,ae,be,ce,de,ee) = fitexp(x_array/1e-6,np.average(blue_dset,axis=(1,2))/No_specimen,single=False, my_color='y',my_edgecolor='#ffff32', my_facecolor='#ffff66',init_guess=init_guess,plot_error=False)       
    #e stands for "error"  
        plt.semilogy(x_array/1e-6,np.average(blue_dset,axis=(1,2))/No_specimen,'yo',label='CL from signal pixels: \n' +r'$\tau_1 $ = ' + str("{0:.2f}".format(b)) + ' $\pm$ ' + str("{0:.2f}".format(be)) + '$\mu$s;' + r'$\tau_2 $ = ' + str("{0:.2f}".format(e)) + ' $\pm$ ' + str("{0:.2f}".format(ee)) + '$\mu$s \n (3$\sigma$ error on complete fit shown)' )   
    else:
        (a,b,c,ae,be,ce) = fitexp(x_array/1e-6,np.average(blue_dset,axis=(1,2))/No_specimen,single=True, my_color='y',my_edgecolor='#ffff32',my_facecolor= '#ffff66',init_guess=init_guess[0::2],plot_error=True)     
        plt.semilogy(x_array/1e-6,np.average(blue_dset,axis=(1,2))/No_specimen,'yo',label='CL from signal pixels: \n' +  r'$\tau $ = ' + str("{0:.2f}".format(b)) + ' $\pm$ ' + str("{0:.2f}".format(be)) + '$\mu$s \n (3$\sigma$ error on complete fit shown)' )   

    #plt.hold(True)
    if other_dset1 is not None: #plotting all pixels
        if single == False:        
            (a1,b1,c1,d1,e1,ae1,be1,ce1,de1,ee1) = fitexp(x_array/1e-6,np.average(other_dset1,axis=(1,2))/No_specimen,single=False, my_color='r',my_edgecolor='#ff3232', my_facecolor='#ff6666',init_guess=init_guess,plot_error=False) 
            plt.semilogy(x_array/1e-6,np.average(other_dset1,axis=(1,2)),'ro',markersize=4, label ='CL from all pixels: \n' + r'$\tau_1 $ = ' + str("{0:.2f}".format(b1)) + ' $\pm$ ' + str("{0:.2f}".format(be1)) + '$\mu$s;' + r'$\tau_2 $ = ' + str("{0:.2f}".format(e1)) + ' $\pm$ ' + str("{0:.2f}".format(ee1)) + '$\mu$s') 
        else:
            (a1,b1,c1,ae1,be1,ce1) = fitexp(x_array/1e-6,np.average(other_dset1,axis=(1,2))/No_specimen,single=True, my_color='r',my_edgecolor='#ff3232', my_facecolor='#ff6666',init_guess=init_guess[0::2],plot_error=False)  
            plt.semilogy(x_array/1e-6,np.average(other_dset1,axis=(1,2)),'ro',markersize=4, label ='CL from all pixels: \n' + r'$\tau $ = ' + str("{0:.2f}".format(b1)) + ' $\pm$ ' + str("{0:.2f}".format(be1)) + '$\mu$s') 
       
    #plt.hold(True)
    if other_dset2 is not None:  #plotting background pixels  
        if single == False:        
            (a2,b2,c2,d2,e2,ae2,be2,ce2,de2,ee2) = fitexp(x_array/1e-6,np.average(other_dset2,axis=(1,2))/No_specimen,single=False, my_color='k',my_edgecolor='#323232', my_facecolor='#666666',init_guess=init_guess,plot_error=False) 
            plt.semilogy(x_array/1e-6,np.average(other_dset2,axis=(1,2)),'ko',markersize=4, label ='CL from background pixels: \n' + r'$\tau_1 $ = ' + str("{0:.2f}".format(b2)) + ' $\pm$ ' + str("{0:.2f}".format(be2)) + '$\mu$s;' + r'$\tau_2 $ = ' + str("{0:.2f}".format(e2)) + ' $\pm$ ' + str("{0:.2f}".format(ee2)) + '$\mu$s') 
        else:
            (a2,b2,c2,ae2,be2,ce2) = fitexp(x_array/1e-6,np.average(other_dset2,axis=(1,2))/No_specimen,single=True, my_color='k',my_edgecolor='#323232', my_facecolor='#666666',init_guess=init_guess[0::2],plot_error=True)  
            plt.semilogy(x_array/1e-6,np.average(other_dset2,axis=(1,2)),'ko',markersize=4, label ='CL from background pixels: \n' + r'$\tau $ = ' + str("{0:.2f}".format(b2)) + ' $\pm$ ' + str("{0:.2f}".format(be2)) + '$\mu$s') 

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlim(xmax=4.7)

    #plt.semilogx(x_array[:-1]*time_detail,sum_grana_blue[1:]/1000.0/No_specimen,'bo',label='Average decay, $\\tau$ = ' + str("{0:.2f}".format(1.0/b)) + '$\mu$s',markersize=(kkk+1)+2)     
    plt.xlabel(r'Time after blanking the electron beam ($\mu$s)',  fontsize=fsizepl)
    plt.ylabel(r'Average luminescence of each time bin (' + unit + ')',  fontsize=fsizepl)
    plt.legend(loc='best')
    
    if single:
        return b,0.0,be,0.0 #e and ee dont exist
    else:
        return b,e,be,ee
        
def calcdecay_subplot(blue_dset,time_detail,titulo,single,other_dset1=None, other_dset2=None, init_guess=None,unit='kHz'): #blue_dset has the transient data only
    No_specimen = 1
    #No_decaypts = blue_dset.shape[0]
   
    x_array = np.arange(0,blue_dset.shape[0])*time_detail
    
    #plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
#    fig, ax = plt.subplots()#figsize=(sizex, sizey), dpi=dpi_no)
#    plt.rc('text', usetex=True)
#    plt.rc('font', family='serif')
#    plt.rc('font', serif='Palatino')
#    plt.suptitle(titulo, fontsize=fsizetit)
    
    plt.hold(True)
    # for kkk in range(No_specimen):   
      
    # dofitting
    # do fit, here with leastsq model
    if single == False:
        (a,b,c,d,e,ae,be,ce,de,ee) = fitexp(x_array/1e-6,np.average(blue_dset,axis=(1,2))/No_specimen,single=False, my_color='k',my_edgecolor='#323232', my_facecolor='#666666',init_guess=init_guess,plot_error=True)       
    #e stands for "error"  
        plt.semilogy(x_array/1e-6,np.average(blue_dset,axis=(1,2))/No_specimen,'ko',label='CL from all pixels: \n' +r' $\tau_1 $ = ' + str("{0:.2f}".format(b)) + ' $\pm$ ' + str("{0:.3f}".format(be)) + '$\mu$s; ' + r'$\tau_2 $ = ' + str("{0:.3f}".format(e)) + ' $\pm$ ' + str("{0:.4f}".format(ee)) + '$\mu$s \n (3$\sigma$ error on complete fit shown)' )   
    else:
        (a,b,c,ae,be,ce) = fitexp(x_array/1e-6,np.average(blue_dset,axis=(1,2))/No_specimen,single=True, my_color='k',my_edgecolor='#323232',my_facecolor= '#666666',init_guess=init_guess[0::2],plot_error=True)     
        plt.semilogy(x_array/1e-6,np.average(blue_dset,axis=(1,2))/No_specimen,'ko',label='CL from all pixels: \n' +  r'$\tau $ = ' + str("{0:.2f}".format(b)) + ' $\pm$ ' + str("{0:.2f}".format(be)) + '$\mu$s \n (3$\sigma$ error on complete fit shown)' )   

    
    #plt.hold(True)
    if other_dset2 is not None:  #plotting background pixels  
        if single == False:        
            (a2,b2,c2,d2,e2,ae2,be2,ce2,de2,ee2) = fitexp(x_array/1e-6,np.average(other_dset2,axis=(1,2))/No_specimen,single=False, my_color='b',my_edgecolor='#397bff', my_facecolor='#79a6ff',init_guess=init_guess,plot_error=True) 
            plt.semilogy(x_array/1e-6,np.average(other_dset2,axis=(1,2)),'bo',markersize=4, label ='CL from blue photons ($<$ 458nm): \n' + r' $\tau_1 $ = ' + str("{0:.2f}".format(b2)) + ' $\pm$ ' + str("{0:.2f}".format(be2)) + '$\mu$s;' + r' $\tau_2 $ = ' + str("{0:.3f}".format(e2)) + ' $\pm$ ' + str("{0:.3f}".format(ee2)) + '$\mu$s') 
        else:
            (a2,b2,c2,ae2,be2,ce2) = fitexp(x_array/1e-6,np.average(other_dset2,axis=(1,2))/No_specimen,single=True, my_color='b',my_edgecolor='#323232', my_facecolor='#666666',init_guess=init_guess[0::2],plot_error=True)  
            plt.semilogy(x_array/1e-6,np.average(other_dset2,axis=(1,2)),'bo',markersize=4, label ='CL from blue photons ($<$ 458nm): \n' + r'$\tau $ = ' + str("{0:.2f}".format(b2)) + ' $\pm$ ' + str("{0:.2f}".format(be2)) + '$\mu$s') 

    #plt.hold(True)
    if other_dset1 is not None: #plotting all pixels
        if single == False:        
            (a1,b1,c1,d1,e1,ae1,be1,ce1,de1,ee1) = fitexp(x_array/1e-6,np.average(other_dset1,axis=(1,2))/No_specimen,single=False, my_color='r',my_edgecolor='#ff3232', my_facecolor='#ff6666',init_guess=init_guess,plot_error=True) 
            plt.semilogy(x_array/1e-6,np.average(other_dset1,axis=(1,2)),'ro',markersize=4, label ='CL from red photons ($>$ 458nm): \n' + r' $\tau_1 $ = ' + str("{0:.2f}".format(b1)) + ' $\pm$ ' + str("{0:.2f}".format(be1)) + '$\mu$s; ' + r'$\tau_2 $ = ' + str("{0:.3f}".format(e1)) + ' $\pm$ ' + str("{0:.3f}".format(ee1)) + '$\mu$s') 
        else:
            (a1,b1,c1,ae1,be1,ce1) = fitexp(x_array/1e-6,np.average(other_dset1,axis=(1,2))/No_specimen,single=True, my_color='r',my_edgecolor='#ff3232', my_facecolor='#ff6666',init_guess=init_guess[0::2],plot_error=False)  
            plt.semilogy(x_array/1e-6,np.average(other_dset1,axis=(1,2)),'ro',markersize=4, label ='CL from red photons ($>$ 458nm): \n' + r'$\tau $ = ' + str("{0:.2f}".format(b1)) + ' $\pm$ ' + str("{0:.2f}".format(be1)) + '$\mu$s') 
       
#    ax1.spines['right'].set_visible(False)
#    ax1.spines['top'].set_visible(False)
#    ax1.xaxis.set_ticks_position('bottom')
#    ax1.yaxis.set_ticks_position('left')
    plt.xlim(xmax=2.7)

    #plt.semilogx(x_array[:-1]*time_detail,sum_grana_blue[1:]/1000.0/No_specimen,'bo',label='Average decay, $\\tau$ = ' + str("{0:.2f}".format(1.0/b)) + '$\mu$s',markersize=(kkk+1)+2)     
    plt.xlabel(r'Time after blanking the electron beam ($\mu$s)',  fontsize=fsizepl)
    plt.ylabel(r'Average luminescence of each time bin (' + unit + ')',  fontsize=fsizepl)
    plt.legend(loc='best')
    
    if single:
        return b,0.0,be,0.0 #e and ee dont exist
    else:
        return b,e,be,ee
        
def calcdecay_subplot2(blue_dset,time_detail,titulo,single,other_dset2=None, other_dset1=None, init_guess=None,unit='kHz',init_guess2=None): #blue_dset has the transient data only
    No_specimen = 1
    #No_decaypts = blue_dset.shape[0]
   
    x_array = np.arange(0,blue_dset.shape[0])*time_detail
    
    #plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
#    fig, ax = plt.subplots()#figsize=(sizex, sizey), dpi=dpi_no)
#    plt.rc('text', usetex=True)
#    plt.rc('font', family='serif')
#    plt.rc('font', serif='Palatino')
#    plt.suptitle(titulo, fontsize=fsizetit)
    
    plt.hold(True)
    # for kkk in range(No_specimen):   
      
    # dofitting
    # do fit, here with leastsq model
    if single == False:
        (a,b,c,d,e,ae,be,ce,de,ee) = fitexp(x_array/1e-6,np.average(blue_dset,axis=(1,2))/No_specimen,single=False, my_color='r',my_edgecolor='#ff3232', my_facecolor='#ff6666',init_guess=init_guess,plot_error=False)       
    #e stands for "error"  
        #plt.semilogy(x_array/1e-6,np.average(blue_dset,axis=(1,2))/No_specimen,'ro',markersize=4,label='CL from red photons ($>$ 593nm): \n' +r' $\tau_1 $ = ' + str("{0:.2f}".format(b)) + ' $\pm$ ' + str("{0:.3f}".format(be)) + '$\mu$s; ' + r'$\tau_2 $ = ' + str("{0:.3f}".format(e)) + ' $\pm$ ' + str("{0:.3f}".format(ee)) + '$\mu$s \n (3$\sigma$ error on complete fit shown)' )   
        plt.semilogy(x_array/1e-6,np.average(blue_dset,axis=(1,2))/No_specimen,'ro',markersize=4,label='CL from red photons ($>$ 409nm): \n' +r' $\tau_1 $ = ' + str("{0:.2f}".format(b)) + ' $\pm$ ' + str("{0:.3f}".format(be)) + '$\mu$s; ' + r'$\tau_2 $ = ' + str("{0:.3f}".format(e)) + ' $\pm$ ' + str("{0:.3f}".format(ee)) + '$\mu$s' )   
    else:
        (a,b,c,ae,be,ce) = fitexp(x_array/1e-6,np.average(blue_dset,axis=(1,2))/No_specimen,single=True, my_color='k',my_edgecolor='#ffff32',my_facecolor= '#ffff66',init_guess=init_guess[0::2],plot_error=True)     
        plt.semilogy(x_array/1e-6,np.average(blue_dset,axis=(1,2))/No_specimen,'ro',label='CL from all pixels: \n' +  r'$\tau $ = ' + str("{0:.2f}".format(b)) + ' $\pm$ ' + str("{0:.2f}".format(be)) + '$\mu$s \n (3$\sigma$ error on complete fit shown)' )   

    
    #plt.hold(True)
    if other_dset2 is not None:  #plotting background pixels  
        if single == False:
            
            if init_guess2 is None:
                init_guess2 = init_guess
            
            (a2,b2,c2,d2,e2,ae2,be2,ce2,de2,ee2) = fitexp(x_array/1e-6,np.average(other_dset2,axis=(1,2))/No_specimen,single=False, my_color='b',my_edgecolor='#397bff', my_facecolor='#79a6ff',init_guess=init_guess2,plot_error=False) 
            plt.semilogy(x_array/1e-6,np.average(other_dset2,axis=(1,2)),'bo',markersize=4, label ='CL from blue photons ($<$ 409nm): \n' + r' $\tau_1 $ = ' + str("{0:.2f}".format(b2)) + ' $\pm$ ' + str("{0:.2f}".format(be2)) + '$\mu$s;' + r' $\tau_2 $ = ' + str("{0:.3f}".format(e2)) + ' $\pm$ ' + str("{0:.3f}".format(ee2)) + '$\mu$s') 
        else:
            (a2,b2,c2,ae2,be2,ce2) = fitexp(x_array/1e-6,np.average(other_dset2,axis=(1,2))/No_specimen,single=True, my_color='b',my_edgecolor='#323232', my_facecolor='#666666',init_guess=init_guess[0::2],plot_error=True)  
            plt.semilogy(x_array/1e-6,np.average(other_dset2,axis=(1,2)),'bo',markersize=4, label ='CL from blue photons ($<$ 458nm): \n' + r'$\tau $ = ' + str("{0:.2f}".format(b2)) + ' $\pm$ ' + str("{0:.2f}".format(be2)) + '$\mu$s') 

    #plt.hold(True)
    if other_dset1 is not None: #plotting all pixels
        if single == False:        
            (a1,b1,c1,d1,e1,ae1,be1,ce1,de1,ee1) = fitexp(x_array/1e-6,np.average(other_dset1,axis=(1,2))/No_specimen,single=False, my_color='r',my_edgecolor='#ff3232', my_facecolor='#ff6666',init_guess=init_guess,plot_error=False) 
            plt.semilogy(x_array/1e-6,np.average(other_dset1,axis=(1,2)),'ro',markersize=4, label ='CL from red photons ($>$ 409nm): \n' + r' $\tau_1 $ = ' + str("{0:.2f}".format(b1)) + ' $\pm$ ' + str("{0:.2f}".format(be1)) + '$\mu$s; ' + r'$\tau_2 $ = ' + str("{0:.3f}".format(e1)) + ' $\pm$ ' + str("{0:.3f}".format(ee1)) + '$\mu$s') 
        else:
            (a1,b1,c1,ae1,be1,ce1) = fitexp(x_array/1e-6,np.average(other_dset1,axis=(1,2))/No_specimen,single=True, my_color='r',my_edgecolor='#ff3232', my_facecolor='#ff6666',init_guess=init_guess[0::2],plot_error=False)  
            plt.semilogy(x_array/1e-6,np.average(other_dset1,axis=(1,2)),'ro',markersize=4, label ='CL from red photons ($>$ 458nm): \n' + r'$\tau $ = ' + str("{0:.2f}".format(b1)) + ' $\pm$ ' + str("{0:.2f}".format(be1)) + '$\mu$s') 
       
#    ax1.spines['right'].set_visible(False)
#    ax1.spines['top'].set_visible(False)
#    ax1.xaxis.set_ticks_position('bottom')
#    ax1.yaxis.set_ticks_position('left')
    plt.xlim(xmax=4.7)

    #plt.semilogx(x_array[:-1]*time_detail,sum_grana_blue[1:]/1000.0/No_specimen,'bo',label='Average decay, $\\tau$ = ' + str("{0:.2f}".format(1.0/b)) + '$\mu$s',markersize=(kkk+1)+2)     
    plt.xlabel(r'Time after blanking the electron beam ($\mu$s)',  fontsize=fsizepl)
    plt.ylabel(r'Average luminescence of each time bin (' + unit + ')',  fontsize=fsizepl)
    plt.legend(loc='best')
    
    if single:
        return b,0.0,be,0.0 #e and ee dont exist
    else:
        return b,e,be,ee
         
def calcdecay_series(blue_dset,time_detail,titulo,single,nominal_time_on,fastfactor,other_dset1=None, other_dset2=None, init_guess=None): #blue_dset has the transient data only
    # this is to see how taus change as a function of number of experiments. No plotting.

    No_specimen = 1
    
    x_array = np.arange(0,blue_dset.shape[1])*time_detail
    
    #build vectors
    tau1 = np.zeros(blue_dset.shape[0])
    tau1_error = np.zeros(blue_dset.shape[0])
    tau2 = np.zeros(blue_dset.shape[0])
    tau2_error = np.zeros(blue_dset.shape[0])
    
    tau1bg = np.zeros(blue_dset.shape[0])
    tau1_errorbg = np.zeros(blue_dset.shape[0])
    tau2bg = np.zeros(blue_dset.shape[0])
    tau2_errorbg = np.zeros(blue_dset.shape[0])
    
    tau1all = np.zeros(blue_dset.shape[0])
    tau1_errorall = np.zeros(blue_dset.shape[0])
    tau2all = np.zeros(blue_dset.shape[0])
    tau2_errorall = np.zeros(blue_dset.shape[0])
    
    for jj in np.arange(blue_dset.shape[0]):
        
        print("in loop")
        print(jj)
        print(jj/blue_dset.shape[0])
   
        # dofitting
        # do fit, here with leastsq model
        if single == False:
            (a,b,c,d,e,ae,be,ce,de,ee) = fitexp(x_array/1e-6,np.average(np.average(blue_dset[0:jj+1,:,:,:],axis = 0),axis=(1,2))/No_specimen,single=False, my_color='y',my_edgecolor='#ffff32', my_facecolor='#ffff66',init_guess=init_guess,plot_error=False,do_plot=False)       
            tau1[jj] = b
            tau1_error[jj] = be
            tau2[jj] = e
            tau2_error[jj] = ee
        #e stands for "error"  
        else:
            (a,b,c,ae,be,ce) = fitexp(x_array/1e-6,np.average(np.average(blue_dset[0:jj+1,:,:,:],axis = 0),axis=(1,2))/No_specimen,single=True, my_color='y',my_edgecolor='#ffff32',my_facecolor= '#ffff66',init_guess=init_guess[0::2],plot_error=True,do_plot=False)     
            tau1[jj] = b
            tau1_error[jj] = be        
        #plt.hold(True)
        if other_dset1 is not None: #plotting all pixels
            if single == False:        
                (a1,b1,c1,d1,e1,ae1,be1,ce1,de1,ee1) = fitexp(x_array/1e-6,np.average(np.average(other_dset1[0:jj+1,:,:,:],axis = 0),axis=(1,2))/No_specimen,single=False, my_color='r',my_edgecolor='#ff3232', my_facecolor='#ff6666',init_guess=init_guess,plot_error=False,do_plot=False) 
                tau1all[jj] = b1
                tau1_errorall[jj] = be1
                tau2all[jj] = e1
                tau2_errorall[jj] = ee1         
            else:
                (a1,b1,c1,ae1,be1,ce1) = fitexp(x_array/1e-6,np.average(np.average(other_dset1[0:jj+1,:,:,:],axis = 0),axis=(1,2))/No_specimen,single=True, my_color='r',my_edgecolor='#ff3232', my_facecolor='#ff6666',init_guess=init_guess[0::2],plot_error=False,do_plot=False)  
                tau1all[jj] = b1
                tau1_errorall[jj] = be1       
            
        #plt.hold(True)
        if other_dset2 is not None:  #plotting background pixels  
            if single == False:        
                (a2,b2,c2,d2,e2,ae2,be2,ce2,de2,ee2) = fitexp(x_array/1e-6,np.average(np.average(other_dset2[0:jj+1,:,:,:],axis = 0),axis=(1,2))/No_specimen,single=False, my_color='k',my_edgecolor='#323232', my_facecolor='#666666',init_guess=init_guess,plot_error=False,do_plot=False) 
                tau1bg[jj] = b2
                tau1_errorbg[jj] = be2
                tau2bg[jj] = e2
                tau2_errorbg[jj] = ee2
            else:
                (a2,b2,c2,ae2,be2,ce2) = fitexp(x_array/1e-6,np.average(np.average(other_dset2[0:jj+1,:,:,:],axis = 0),axis=(1,2))/No_specimen,single=True, my_color='k',my_edgecolor='#323232', my_facecolor='#666666',init_guess=init_guess[0::2],plot_error=False,do_plot=False)  
                tau1bg[jj] = b2
                tau1_errorbg[jj] = be2   
                
    
    plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')
    titulo =  'Upconverting NPs (3kV, 30$\mu$m aperture, 40ns time bins, 36kX or 3.1nm pixels)'
    #titulo =  'Er ' +  str(2)+ '$\%$ core-shell aggregates (2kX, 2kV, ' +  str(120) + '$\mu$m, 40ns time bins, ' + str(56) + 'nm pixels), sample ' + 'A'
    plt.suptitle('Cathodoluminescence decay rates as a function of e-beam exposure, \n' + titulo, fontsize=fsizetit)
    
    ax1 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid((2,2), (1,0), colspan=1, sharex=ax1)
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    
    #ax1.set_ylim([0.55,0.85])
    #ax2.set_ylim([0.0,0.1])
    x_array = np.arange(1,blue_dset.shape[0]+1)*nominal_time_on*fastfactor
    ax1.errorbar(x_array, tau1, yerr=tau1_error, fmt='yo',markersize=10,label='CL from signal pixels')
    ax2.errorbar(x_array, tau2, yerr=tau2_error, fmt='ys', markersize=5)
    
    ax1.errorbar(x_array, tau1all, yerr=tau1_errorall, fmt='ro',markersize=10,label='CL from all pixels')
    ax2.errorbar(x_array, tau2all, yerr=tau2_errorall, fmt='rs', markersize=5)  
    
    ax1.errorbar(x_array, tau1bg, yerr=tau1_errorbg, fmt='ko',markersize=10,label='CL from background pixels')
    ax2.errorbar(x_array, tau2bg, yerr=tau2_errorbg, fmt='ks', markersize=5)
      
    ax1.set_ylabel('Longer time constant \n of cumulative time bins ($\mu$s)',fontsize=fsizepl)
    ax2.set_ylabel('Shorter time constant \n of cumulative time bins ($\mu$s)',fontsize=fsizepl)
    plt.xlabel(r"Cumulative e-beam exposure time per pixel (nominal, $\mu$s)",fontsize=fsizepl)
    major_ticks = [2,4,6] #[25,50,75,100]
    ax1.set_xticks(major_ticks) 
    plt.xlim([0,6]) #[0,102]
    ax1.legend(loc='best')
    
     
    ax1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid((2,2), (1,1), colspan=1, sharex=ax1)
    
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    
    from uncertainties import unumpy
    tau1U = unumpy.uarray(tau1,tau1_error)
    tau2U = unumpy.uarray(tau2,tau2_error)
    tau1bgU = unumpy.uarray(tau1bg,tau1_errorbg)
    tau2bgU = unumpy.uarray(tau2bg,tau2_errorbg)
    
    ratio_large = tau1U/tau1bgU
    ratio_small = tau2U/tau2bgU
    large_nb = np.zeros(len(ratio_large))
    large_err = np.zeros(len(ratio_large))
    small_nb = np.zeros(len(ratio_large))
    small_err = np.zeros(len(ratio_large))
    for jj in np.arange(len(ratio_large)):
        print(jj)
        large_nb[jj] = float(str(ratio_large[jj]).partition('+/-')[0])
        large_err[jj] = float(str(ratio_large[jj]).partition('+/-')[2])
        small_nb[jj] = float(str(ratio_small[jj]).partition('+/-')[0])
        small_err[jj] = float(str(ratio_small[jj]).partition('+/-')[2])
    
    #ax1.set_ylim([0.55,0.85])
    #ax2.set_ylim([0.0,0.1])
    
    ax1.errorbar(x_array, large_nb, yerr=large_err, fmt='ko',markersize=10)
    ax2.errorbar(x_array, small_nb, yerr=small_err, fmt='ks', markersize=5)
    ax1.axhline(y=1, xmin=0, xmax=60,linewidth=2, color = 'k', ls = '--')

    ax1.set_ylabel('Ratio of larger time constants \n of cumulative time bins',fontsize=fsizepl)
    ax2.set_ylabel('Ratio of shorter time constants \n of cumulative time bins',fontsize=fsizepl)
    ax2.axhline(y=1, xmin=0, xmax=60,  linewidth=2, color = 'k',ls = '--')  
    plt.xlabel(r"Cumulative e-beam exposure time per pixel (nominal, $\mu$s)",fontsize=fsizepl) 
    major_ticks = [2,4,6] #[25,50,75,100]
    ax1.set_xticks(major_ticks) 
    plt.xlim([0,6])#[0,102]
     
   #multipage('TausFunctionBeamExposure.pdf',dpi=80)
        
def calcdecay_each(blue_dset,time_detail,titulo,single,nominal_time_on,fastfactor,other_dset1=None, other_dset2=None, init_guess=None): #blue_dset has the transient data only
    # this is to see how taus change as a function of number of experiments. No plotting.

    No_specimen = 1
    
    x_array = np.arange(0,blue_dset.shape[1])*time_detail
    
    #build vectors
    tau1 = np.zeros(blue_dset.shape[0])
    tau1_error = np.zeros(blue_dset.shape[0])
    tau2 = np.zeros(blue_dset.shape[0])
    tau2_error = np.zeros(blue_dset.shape[0])
    
    tau1bg = np.zeros(blue_dset.shape[0])
    tau1_errorbg = np.zeros(blue_dset.shape[0])
    tau2bg = np.zeros(blue_dset.shape[0])
    tau2_errorbg = np.zeros(blue_dset.shape[0])
    
    tau1all = np.zeros(blue_dset.shape[0])
    tau1_errorall = np.zeros(blue_dset.shape[0])
    tau2all = np.zeros(blue_dset.shape[0])
    tau2_errorall = np.zeros(blue_dset.shape[0])
    
    for jj in np.arange(blue_dset.shape[0]):
        
        print("in loop")
        print(jj)
        print(jj/blue_dset.shape[0])
   
        # dofitting
        # do fit, here with leastsq model
        if single == False:
            (a,b,c,d,e,ae,be,ce,de,ee) = fitexp(x_array/1e-6,np.average(blue_dset[jj,:,:,:],axis=(1,2))/No_specimen,single=False, my_color='y',my_edgecolor='#ffff32', my_facecolor='#ffff66',init_guess=init_guess,plot_error=False,do_plot=False)       
            tau1[jj] = b
            tau1_error[jj] = be
            tau2[jj] = e
            tau2_error[jj] = ee
        #e stands for "error"  
        else:
            (a,b,c,ae,be,ce) = fitexp(x_array/1e-6,np.average(blue_dset[jj,:,:,:],axis=(1,2))/No_specimen,single=True, my_color='y',my_edgecolor='#ffff32',my_facecolor= '#ffff66',init_guess=init_guess[0::2],plot_error=True,do_plot=False)     
            tau1[jj] = b
            tau1_error[jj] = be        
        #plt.hold(True)
        if other_dset1 is not None: #plotting all pixels
            if single == False:        
                (a1,b1,c1,d1,e1,ae1,be1,ce1,de1,ee1) = fitexp(x_array/1e-6,np.average(other_dset1[jj,:,:,:],axis=(1,2))/No_specimen,single=False, my_color='r',my_edgecolor='#ff3232', my_facecolor='#ff6666',init_guess=init_guess,plot_error=False,do_plot=False) 
                tau1all[jj] = b1
                tau1_errorall[jj] = be1
                tau2all[jj] = e1
                tau2_errorall[jj] = ee1         
            else:
                (a1,b1,c1,ae1,be1,ce1) = fitexp(x_array/1e-6,np.average(other_dset1[jj,:,:,:],axis=(1,2))/No_specimen,single=True, my_color='r',my_edgecolor='#ff3232', my_facecolor='#ff6666',init_guess=init_guess[0::2],plot_error=False,do_plot=False)  
                tau1all[jj] = b1
                tau1_errorall[jj] = be1       
            
        #plt.hold(True)
        if other_dset2 is not None:  #plotting background pixels  
            if single == False:        
                (a2,b2,c2,d2,e2,ae2,be2,ce2,de2,ee2) = fitexp(x_array/1e-6,np.average(other_dset2[jj,:,:,:],axis=(1,2))/No_specimen,single=False, my_color='k',my_edgecolor='#323232', my_facecolor='#666666',init_guess=init_guess,plot_error=False,do_plot=False) 
                tau1bg[jj] = b2
                tau1_errorbg[jj] = be2
                tau2bg[jj] = e2
                tau2_errorbg[jj] = ee2
            else:
                (a2,b2,c2,ae2,be2,ce2) = fitexp(x_array/1e-6,np.average(other_dset2[jj,:,:,:],axis=(1,2))/No_specimen,single=True, my_color='k',my_edgecolor='#323232', my_facecolor='#666666',init_guess=init_guess[0::2],plot_error=False,do_plot=False)  
                tau1bg[jj] = b2
                tau1_errorbg[jj] = be2   
                
    
    plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Palatino')
    titulo =  'Upconverting NPs (3kV, 30$\mu$m aperture, 40ns time bins, 36kX or 3.1nm pixels)'
    #titulo =  'Er ' +  str(2)+ '$\%$ core-shell aggregates (2kX, 2kV, ' +  str(120) + '$\mu$m, 40ns time bins, ' + str(56) + 'nm pixels), sample ' + 'A'
    plt.suptitle('Cathodoluminescence decay rates as a function of e-beam exposure, \n' + titulo, fontsize=fsizetit)
    
    ax1 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid((2,2), (1,0), colspan=1, sharex=ax1)
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    
    #ax1.set_ylim([0.55,0.85])
    #ax2.set_ylim([0.0,0.1])
    x_array = np.arange(1,blue_dset.shape[0]+1)*nominal_time_on*fastfactor
    ax1.errorbar(x_array, tau1, yerr=tau1_error, fmt='yo',markersize=10,label='CL from signal pixels')
    ax2.errorbar(x_array, tau2, yerr=tau2_error, fmt='ys', markersize=5)
    
    ax1.errorbar(x_array, tau1all, yerr=tau1_errorall, fmt='ro',markersize=10,label='CL from all pixels')
    ax2.errorbar(x_array, tau2all, yerr=tau2_errorall, fmt='rs', markersize=5)  
    
    ax1.errorbar(x_array, tau1bg, yerr=tau1_errorbg, fmt='ko',markersize=10,label='CL from background pixels')
    ax2.errorbar(x_array, tau2bg, yerr=tau2_errorbg, fmt='ks', markersize=5)
      
    ax1.set_ylabel('Longer time constant \n of individual time bins ($\mu$s)',fontsize=fsizepl)
    ax2.set_ylabel('Shorter time constant \n of individual time bins ($\mu$s)',fontsize=fsizepl)
    plt.xlabel(r"Cumulative e-beam exposure time per pixel (nominal, $\mu$s)",fontsize=fsizepl)
    major_ticks = [2,4,6] 
    ax1.set_xticks(major_ticks) 
    plt.xlim([0,6])
    ax1.legend(loc='best')
    
     
    ax1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid((2,2), (1,1), colspan=1, sharex=ax1)
    
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    
    from uncertainties import unumpy
    tau1U = unumpy.uarray(tau1,tau1_error)
    tau2U = unumpy.uarray(tau2,tau2_error)
    tau1bgU = unumpy.uarray(tau1bg,tau1_errorbg)
    tau2bgU = unumpy.uarray(tau2bg,tau2_errorbg)
    
    ratio_large = tau1U/tau1bgU
    ratio_small = tau2U/tau2bgU
    large_nb = np.zeros(len(ratio_large))
    large_err = np.zeros(len(ratio_large))
    small_nb = np.zeros(len(ratio_large))
    small_err = np.zeros(len(ratio_large))
    for jj in np.arange(len(ratio_large)):
        print(jj)
        large_nb[jj] = float(str(ratio_large[jj]).partition('+/-')[0])
        large_err[jj] = float(str(ratio_large[jj]).partition('+/-')[2])
        if jj == 9 or jj == 13:
            small_nb[jj] = 0.0
            small_err[jj] = 0.0
        else:
            small_nb[jj] = float(str(ratio_small[jj]).partition('+/-')[0])
            small_err[jj] = float(str(ratio_small[jj]).partition('+/-')[2])
    
    #ax1.set_ylim([0.55,0.85])
    #ax2.set_ylim([0.0,0.1])
    
    ax1.errorbar(x_array, large_nb, yerr=large_err, fmt='ko',markersize=10)
    ax2.errorbar(x_array, small_nb, yerr=small_err, fmt='ks', markersize=5)
    ax1.axhline(y=1, xmin=0, xmax=60,linewidth=2, color = 'k', ls = '--')

    ax1.set_ylabel('Ratio of larger time constants \n of individual time bins',fontsize=fsizepl)
    ax2.set_ylabel('Ratio of shorter time constants \n of individual time bins',fontsize=fsizepl)
    ax2.axhline(y=1, xmin=0, xmax=60,  linewidth=2, color = 'k',ls = '--')  
    plt.xlabel(r"Cumulative e-beam exposure time per pixel (nominal, $\mu$s)",fontsize=fsizepl) 
    major_ticks = [2,4,6] #[25,50,75,100]
    ax1.set_xticks(major_ticks) 
    plt.xlim([0,6]) #[0,100]
     
   # multipage('TausFunctionBeamExposure.pdf',dpi=80)
        