import numpy as np
import matplotlib.mlab as mlab
from astropy.stats import histogram
import matplotlib.pyplot as plt

fsizepl = 24
fsizenb = 20

def plot_ratio_hist(my_hist, ax1, ax2, no_of_bins = 20,my_color='grey',index=0):
    
    exp_number = np.arange(1,my_hist.shape[1]+1)

    #my_hist[NP,avgx]
    X = np.zeros([my_hist.shape[0],my_hist.shape[1]])
    for jj in np.arange(0,my_hist.shape[0]):
        X[jj,:] = my_hist[jj,:]  -  np.average(my_hist, axis = 0) # DISTANCE each NP - mean, for all expirement numbers
        
    stdbar = np.nanstd(X, axis = 1) 
    
    no_of_bins = 250
    n, bins, patches = ax1.hist(
            stdbar/np.average(my_hist, axis = (0,1)) * 100.0,
            no_of_bins, 
            #normed=1,
            facecolor='black', 
            alpha=0.75,
            edgecolor = 'None'
            )
            
    if index == 0:
       ax1.text(0.05*31,0.9*5,r'$\sigma$ $\sim$ 15.5 $\%$',fontsize=fsizepl)
       ax1.set_ylim([0,5])
       ax1.set_yticks([5])
       ax1.set_xlim([0,31])
       ax1.set_xticks([10,20,30])
    elif index == 1:
       ax1.text(0.05*17,0.9*2,r'$\sigma$ $\sim$ 7.9 $\%$',fontsize=fsizepl)
       ax1.set_ylim([0,2])
       ax1.set_yticks([2])
       ax1.set_xlim([0,17])
       ax1.set_xticks([5,10,15])

    ax2.errorbar(
            exp_number,
            np.average(my_hist, axis = 0),
            yerr = np.std(my_hist, axis = 0),
            marker='o',
            color='k',
            markersize=12,
            linestyle='None')
    
    ax2.set_xlim([0, exp_number[-1] + 1])
    ax2.set_ylim(2, 5) 
    ax2.set_yticks([2.5,4.5]) 
    ax2.set_xlabel('Exp. number',fontsize=fsizepl)
    ax2.set_ylabel('Ratio of int. (a.u.)',fontsize=fsizepl)
    ax2.set_xticks([1,7])
    
    ax1.set_xlabel(r'Std. per NP / mean$_{\overline{\tiny\textrm{NP}}\tiny, \overline{\tiny\textrm{Exp.}}}$ ($\%$)',fontsize=fsizepl) #mean is over everything
    ax1.set_ylabel('$\#$ of NP',fontsize=fsizepl)
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    
def plot_tau_hist(my_hist, ax1, ax2, which_taus = [-1], no_of_bins = 20, my_color = 'k', my_markersize = 12,my_title=""):

    ax1.plot(np.std(my_hist, axis=(0,1))/np.nanmean(my_hist, axis=(0,1))*100.0,color=my_color,linewidth=2,label=r'$\bar{\textrm{NP}},\bar{\textrm{Exp.}}$')
    ax1.set_xticks([500, 1000])
    ax1.set_xlabel('Acquisition time ($\mu$s)',fontsize=fsizepl)
    
    X = np.zeros([my_hist.shape[0],my_hist.shape[1],my_hist.shape[2]]) #[NP, avg, time]
    for jj in np.arange(0,my_hist.shape[0]):
        X[jj,:,:] = my_hist[jj,:,:]  -  np.average(my_hist, axis = 0) # DISTANCE each NP - mean, for all expirement numbers
    
    stdbar = np.nanstd(X, axis = 1) #vector [NP, time]
    
    toplot = np.average(stdbar/np.average(my_hist, axis = (0,1)) * 100.0, axis = 0)
    ax1.plot(toplot,color=my_color,linewidth=2, linestyle='dashed',label=r'$\bar{\textrm{Exp.}}$')
    
    
    XX = np.zeros([my_hist.shape[0],my_hist.shape[1],my_hist.shape[2]]) #[NP, avg, time]
    for jj in np.arange(0,my_hist.shape[1]):
        XX[:,jj,:] = my_hist[:,jj,:]  -  np.average(my_hist, axis = 1) # DISTANCE each experiment - mean, for all NPs
    
    stdbarXX = np.nanstd(XX, axis = 0) #vector [NP, time]
    
    toplotXX = np.average(stdbarXX/np.average(my_hist, axis = (0,1)) * 100.0, axis = 0)
    ax1.plot(toplotXX,color=my_color,linewidth=2,linestyle='dotted',label=r'$\bar{\textrm{NP}}$')
    
    ax1.legend(loc='best',frameon=False)
    ax1.set_ylabel(r'$\sigma_{\tau}$/$\tau$ ($\%$)',fontsize=fsizepl)

       
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    