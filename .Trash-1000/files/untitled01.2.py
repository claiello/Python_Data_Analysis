# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:09:48 2016

@author: clarice
"""


import os
import sys
import scipy as sp
import scipy.misc
import matplotlib.pyplot as plt
import h5py
import numpy as np
#from BackgroundCorrection import *
import matplotlib.cm as cm
import scipy.ndimage as ndimage
#from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
#from MakePdf import *
#from matplotlib.pyplot import cm #to plot following colors of rainbow
#from matplotlib import rc

#import warnings
#warnings.simplefilter(action = "ignore", category = RuntimeWarning)
#warnings.simplefilter(action = "ignore", category = DeprecationWarning)
#warnings.simplefilter(action = "ignore", category = FutureWarning)
#warnings.simplefilter(action = "ignore", category = PendingDeprecationWarning)

#from Registration import * # reg_images, reg_time_resolved_images_to_se
#from tifffile import *

#from sklearn.mixture import GMM 
#import matplotlib.cm as cm


sys.path.append("/usr/bin")

# Example data
t = np.arange(0.0, 1.0 + 0.01, 0.01)
s = np.cos(4 * np.pi * t) + 2

plt.figure(figsize=(8, 6), dpi=80)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(t, s)

plt.xlabel(r'\textbf{time} (s) $\mu$s')
plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
plt.title(r"\TeX\ is Number "
          r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
          fontsize=16, color='gray')
# Make room for the ridiculously large title.
plt.subplots_adjust(top=0.8)

plt.savefig('tex_demo')
plt.show()