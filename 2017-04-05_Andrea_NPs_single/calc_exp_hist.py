import numpy as np
import matplotlib.mlab as mlab
from astropy.stats import histogram
import matplotlib.pyplot as plt

fsizepl = 18 #24
fsizenb = 16 #20

def plot_ratio_hist(my_hist, ax1, ax2, no_of_bins = 20,my_color='grey'):
    
    from scipy.stats import norm

    # my_hist should be 2D array
    
    # calculating means and std of histograms of each average
    #hlp_hist = my_hist
    #hlp_avg = np.nanmean(hlp_hist, axis = 0)
    #hlp_std = np.nanstd(hlp_hist, axis = 0)
    
    #############
    
    
    ##########
    
#    (mu, sigma) = norm.fit(np.nanmean(hlp_hist, axis=(1)))
    
#    xastro, binastro = histogram(np.nanmean(hlp_hist, axis=(1)),bins='blocks')
#    print('hist params')
#    print(xastro)
#    print(binastro)    
#    
#    n, bins, patches = ax1.plot(
#            xastro, 
#            binastro, 
#            #normed=1,
#            facecolor='black', 
#            alpha=0.75,
#            edgecolor = 'None'
#            )

#    n, bins, patches = ax1.hist(
#            np.nanmean(hlp_hist, axis=(1)), 
#            no_of_bins, 
#            #normed=1,
#            facecolor='black', 
#            alpha=0.75,
#            edgecolor = 'None'
#            )
            
#    y = mlab.normpdf(bins, mu,sigma)
#    ax1.plot(bins, y , linestyle='--',lw=2,color=my_color)
    
#    ax1.text(1,1,r'$\sigma/\mu = $' + str((np.nanstd(hlp_hist, axis=(0,1))/np.nanmean(hlp_hist, axis=(0,1)))),fontsize=fsizepl)
   
    exp_number = np.arange(1,my_hist.shape[1]+1)

#    ax1.set_xlabel('Ratio of intensities',fontsize=fsizepl)
#    ax1.set_xlim([1,5])
#    ax1.set_xticks([2,4])

    #my_hist[NP,avgx]
    X = np.zeros([my_hist.shape[0],my_hist.shape[1]])
    for jj in np.arange(0,my_hist.shape[0]):
        X[jj,:] = my_hist[jj,:]  -  np.average(my_hist, axis = 0) # DISTANCE each NP - mean, for all exp values
        
    mubar = np.nanmean(X, axis = 1) #### CLOSE TO ZERO MEANS ACCURACY
    stdbar = np.nanstd(X, axis = 1) 
    
    no_of_bins = 250
    n, bins, patches = ax1.hist(
             stdbar/mubar *100.0, 
            no_of_bins, 
            #normed=1,
            facecolor='black', 
            alpha=0.75,
            edgecolor = 'None'
            )
            
    ax1.set_xlim([-5000,5000])

    #mubarbar = np.average(mubar, axis = (0))
    #stdbarbar= np.std(mubar, axis = (0))
    
    print('std stdbar/mubar')
    print(np.std(stdbar/mubar *100.0))

#    ax1.text(0.8,12,'m' + str(mubarbar) + '\n' + 'd' + str(stdbarbar),fontsize=fsizepl)



    ax2.errorbar(
            exp_number,
            np.average(my_hist, axis = 0),
            yerr = np.std(my_hist, axis = 0),
            marker='o',
            color='k',
            markersize=12,
            linestyle='None')
    

    ax2.errorbar(
            exp_number,
            my_hist[29,:],
            yerr = 0.0,
            marker='o',
            color='r',
            markersize=12,
            linestyle='None')

    ax2.set_xlim([0, exp_number[-1] + 1])
    ax2.set_ylim(2, 5) 
    ax2.set_yticks([2.5,4.5]) 
    ax2.set_xlabel('Experiment number',fontsize=fsizepl)
    ax2.set_ylabel('Ratio of intensities (a.u.)',fontsize=fsizepl)
    ax2.set_xticks([1,7])
    
#    ax1.set_xlim([0, exp_number[-1] + 1])
#    ax1.set_ylim(2, 5) 
#    ax1.set_yticks([2.5,4.5]) 
    ax1.set_xlabel('std/mean ($\%$)',fontsize=fsizepl)
    ax1.set_ylabel('nb of NP',fontsize=fsizepl)
#    ax1.set_xticks([1,7])
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    
    

def plot_tau_hist(my_hist, ax1, ax2, which_taus = [-1], no_of_bins = 20, my_color = 'k', my_markersize = 12,my_title=""):

    # my_hist should be 3D array: NP, avg number, time
    my_markersize = [6,8,10]        
    
    from matplotlib.pyplot import cm #to plot following colors of rainbow

    my_colors = cm.rainbow(np.linspace(0, 1, 3 ))
    
    ax1.plot(np.std(my_hist, axis=(0,1))/np.nanmean(my_hist, axis=(0,1)),'k',linewidth=2)
    
    
#    my_colors = ['k', 'c', 'm']    
    
    for k in np.arange(0,len(which_taus)):

        hlp = my_hist[:, :, which_taus[k]]

        # calculating means and std of histograms of each average
        hlp_hist = hlp
        hlp_avg = np.nanmean(hlp_hist, axis = 0)
        hlp_std = np.nanstd(hlp_hist, axis = 0)

#        n, bins, patches = ax1.hist(
#                np.nanmean(hlp_hist, axis=(1)), 
#                no_of_bins, 
#                #normed=1,
#                #facecolor='grey', 
#                facecolor=my_colors[k], 
#                alpha=0.75,
#                color = my_colors[k],
#                edgecolor = 'None'
#                )
                
#        ax1.set_xlim([0, 300])
        ax1.set_xticks([500, 1000])
        ax1.set_xlabel('Acquisition time ($\mu$s)',fontsize=fsizepl)

        exp_number = np.arange(1,hlp_hist.shape[1]+1)

        ax2.errorbar(
                exp_number,
                hlp_avg,
                yerr = hlp_std,
                marker='o',
                color=my_colors[k],
                markersize=my_markersize[k],
                markeredgecolor='None')

        ax2.set_xlim([0, exp_number[-1] + 1])
        ax2.set_ylim([0, 300])
        ax2.set_xticks([1,7])
        
        ax2.set_xlabel('Experiment number',fontsize=fsizepl)
        ax2.set_ylabel(my_title + r'$\tau$ ($\mu$s)',fontsize=fsizepl)
        ax2.set_ylim([0, 310])
        ax2.set_yticks([100,200,300])

        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.xaxis.set_ticks_position('bottom')
        ax1.yaxis.set_ticks_position('left')
        
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('left')
