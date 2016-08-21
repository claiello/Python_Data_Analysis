import numpy as np
import scipy.stats
    
def calc_pearson_spearman_manders(image1,image2):
    
    # copy each image to hlp matrices
    hlp1 = np.copy(image1)
    hlp2 = np.copy(image2)

    hlp1 = hlp1.flatten()
    hlp2 = hlp2.flatten()
    # set all points to nan where one of the images has a nan
    ind = np.isfinite(hlp1) & np.isfinite(hlp2)

    # now, take only the values that are finite; defined by the indices in ind
    hlp1 = hlp1[ind]
    hlp2 = hlp2[ind]    
    
    pearsonc = scipy.stats.pearsonr(hlp1, hlp2)
    
    spearmanc = scipy.stats.spearmanr(hlp1, hlp2)
    
    m1 = 0
    for x in range(image1.shape[0]):
        for xx in range(image1.shape[1]):
            if not np.isnan(image2[x,xx]):
                if not np.isnan(image1[x,xx]):
                    m1 += image1[x,xx]
    m1 = m1/np.nansum(image1)
                
    m2 = 0
    for x in range(image2.shape[0]):
        for xx in range(image2.shape[1]):
            if not np.isnan(image1[x,xx]):
                if not np.isnan(image2[x,xx]):
                    m2 += image2[x,xx]
    m2 = m2/np.nansum(image2)
    
    return pearsonc, spearmanc, m1, m2
    

    
    