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

def KThermocouplerconversion(V):
    
    muV = V/1.0e-6
    
    #http://www.omega.com/techref/pdf/z198-201.pdf
    #range: 0 to 500C
    c0 = 0.0
    c1 = 2.508355*1.0e-2
    c2 = 7.860106*1.0e-8
    c3 = -2.503131*1.0e-10
    c4 = 8.315270*1.0e-14
    c5 = -1.228034*1.0e-17
    c6 = 9.804036*1.0e-22
    c7 = -4.413030*1.0e-26
    c8 = 1.057734*1.0e-30
    c9 = -1.052755*1.0e-35
    
    t = c0 + c1*muV + c2*np.power(muV,2) + c3*np.power(muV,3) + c4*np.power(muV,4) + c5*np.power(muV,5) + c6*np.power(muV,6) + c7*np.power(muV,7) + c8*np.power(muV,8) + c9*np.power(muV,9)
    
    return t
    
