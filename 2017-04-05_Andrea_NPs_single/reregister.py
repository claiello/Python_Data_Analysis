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
from TConversionThermocoupler import *
import matplotlib.cm as cm
import scipy.ndimage as ndimage
#from matplotlib_scalebar.scalebar import ScaleBar #### Has issue with plotting using latex font. only import when needed, then unimport
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

import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile
import scalebars as sb

def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.get_memory_info()[0] / float(2 ** 20)
    return mem

#PARAMS THAT NEED TO BE CHANGED
###############################################################################
###############################################################################
###############################################################################

Time_bin = 1000#in ns; 
nominal_time_on = 150.0 #time during which e-beam nominally on, in mus
totalpoints = 1400 #total number of time-resolved points
    
No_experiments = 5*np.ones([7])


red1_dset_reg_all = np.load('x600Bluebright')
red1_dset_reg_all_sec = np.load('x600bBluebright')

print('data loaded')
    
se1_dset_reg, red1_dset_reg, red1_dset_reg_all = reg_time_resolved_images_to_se(se1_dset2, red1_dset)
   gc.collect()
    
    
