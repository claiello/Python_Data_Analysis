import numpy as np

def plot_ratio_hist(my_hist, ax1, ax2, no_of_bins = 20):

    # my_hist should be 2D array
    
    # calculating means and std of histograms of each average
    hlp_hist = my_hist
    hlp_avg = np.nanmean(hlp_hist, axis = 0)
    hlp_std = np.nanstd(hlp_hist, axis = 0)

    n, bins, patches = ax1.hist(
            np.nanmean(hlp_hist, axis=(1)), 
            no_of_bins, 
            #normed=1,
            facecolor='grey', 
            alpha=0.75
            )
   
    exp_number = np.arange(1,hlp_hist.shape[1]+1)

    ax2.errorbar(
            exp_number,
            hlp_avg,
            yerr = hlp_std,
            marker='o',
            color='k',
            markersize=12)

    ax2.set_xlim([0, exp_number[-1] + 1])
    ax2.set_ylim(2, 5) 
    ax2.set_xlabel('Experiment number')
    ax2.set_ylabel('Intensity ratio')

def plot_tau_hist(my_hist, ax1, ax2, which_taus = [-1], no_of_bins = 20, my_color = 'k'):

    # my_hist should be 3D array: NP, avg number, time
    
    for k in which_taus:

        hlp = my_hist[:, :, k]

        # calculating means and std of histograms of each average
        hlp_hist = hlp
        hlp_avg = np.nanmean(hlp_hist, axis = 0)
        hlp_std = np.nanstd(hlp_hist, axis = 0)

        n, bins, patches = ax1.hist(
                np.nanmean(hlp_hist, axis=(1)), 
                no_of_bins, 
                normed=1,
                facecolor='grey', 
                alpha=0.75,
                color = my_color
                )

        exp_number = np.arange(1,hlp_hist.shape[1]+1)

        ax2.errorbar(
                exp_number,
                hlp_avg,
                yerr = hlp_std,
                marker='o',
                color=my_color,
                markersize=12)

        ax2.set_xlim([0, exp_number[-1] + 1])
        
        ax2.set_xlabel('Experiment number')
        ax2.set_ylabel('Tau')



