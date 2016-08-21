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
import matplotlib.cm as cm
import scipy.ndimage as ndimage
#from matplotlib_scalebar.scalebar import ScaleBar #### Has issue with plotting using latex font. only import when needed, then unimport
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from MakePdf import *
from matplotlib.pyplot import cm #to plot following colors of rainbow
from matplotlib import rc
#from CreateDatasets import *
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = DeprecationWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.simplefilter(action = "ignore", category = PendingDeprecationWarning)
#from Registration import * 
#from tifffile import *
from sklearn.mixture import GMM 
import matplotlib.cm as cm
#from FluoDecay import *
from PlottingFcts import *
sys.path.append("/usr/bin") # necessary for the tex fonts
import matplotlib.animation as animation
import gc
import tempfile
from tempfile import TemporaryFile
import scalebars as sb

fsizetit = 18
fsizepl = 16
sizex = 8
sizey = 6
dpi_no = 80
lw = 2

#Grab info on mag, pixel size, aperture, kV from .txt file
### can't make textcolor to work

name = ['8.hdf5','9.hdf5','10.hdf5','12.hdf5','13.hdf5','14.hdf5','15.hdf5','16.hdf5']
kV = ['4','4','3','3','2','2','2','2']
lag = ['1ms', '5ms', '5ms', '5ms', '5ms', '5ms', '5ms', '5ms', ]
#all 30mum aperture, 2kX, 
Pixel_size = 5.5824998e-08 #in meters
Ps = 56
kk = 0
obs = ['showing CLAIRE/FRET', 'showing CLAIRE/FRET','showing CLAIRE/FRET', 'showing CLAIRE/FRET; big blob too thick to FRET','showing CLAIRE/FRET', 'showing CLAIRE/FRET; big blob too thick to FRET','showing CLAIRE/FRET; big blob too thick to FRET', 'showing CLAIRE/FRET; big blob too thick to FRET']

for index in [8,9,10,12,13,14,15,16]:
    
    print(index)

    file1    = h5py.File(str(index) + '.hdf5', 'r')  
    title =  'F888 1$\mu$m beads on ZnO:Ga, with PMMA cover (2kX, ' + kV[kk] + 'kV' + ', 30 $\mu$m, ' + str(Ps) + 'nm pixels, ' + lag[kk] + ' lag per pixel, 1 expt.), \n' + obs[kk] 
    se   = file1['/data/Analog channel 1 : SE2/data'] #50 frames x250 x 250 pixels
    red  = file1['/data/Counter channel 1 : PMT red/PMT red/data']#50 frames x 200 tr pts x250 x 250 pixels
    blue  = file1['/data/Counter channel 2 : PMT blue/PMT blue/data']#50 frames x 200 tr pts x250 x 250 pixels
    
    se = np.array(se)
    red = np.array(red)
    blue = np.array(blue)
    
    #convert red1_dset to kHz!!!!!!!!!!!!!!!!!!!!!1
    red = red/1.0e3
    blue = blue/1.0e3
    unit = '(kHz)'

    scinti_channel = '$<$ 465nm'
    sample_channel = '$>$ 465nm'
    length_scalebar = 5000.0 #in nm (1000nm == 1mum)
    scalebar_legend = '5 $\mu$m'
    plot_3_channels(np.average(se,axis=0), np.average(blue,axis=0), np.average(red,axis=0), Pixel_size, title, scinti_channel, sample_channel, length_scalebar, scalebar_legend,unit,work_red_channel=True)
    kk = kk + 1
    
#######

plt.show()
    
    
    
    
    
#######
 
#multipage('ZZZ.pdf',dpi=80)

pp = PdfPages('ZZZ.pdf')
figs = [plt.figure(n) for n in plt.get_fignums()]
for fig in figs:
    #fig.set_size_inches(1200./fig.dpi,900./fig.dpi)
    fig.savefig(pp, format='pdf')
pp.close()