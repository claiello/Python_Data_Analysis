import numpy as np
import matplotlib.pylab as plt
from lmfit import minimize, Parameters, Parameter, report_fit
import sys
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../Python modules/") # necessary for the tex fonts
#import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GMM 
import matplotlib.cm as cm
import scipy.ndimage as ndimage
#from matplotlib_scalebar.scalebar import ScaleBar
import lmfit
from lmfit import minimize, Parameters, Parameter, report_fit
#from MakePdf import *
import os

from cStringIO import StringIO
import sys

fsizetit = 18
fsizepl = 16
sizex = 8
sizey = 6
dpi_no = 80
lw = 2


def load_txt_data(filename = None):

    # loads filename or entire directory of txt files
    data = []
    files = []
    if filename is None:
        for file in os.listdir("."):
            if file.endswith(".txt"):
                data.append(np.loadtxt(file, skiprows = 10))
                files.append(file)
                print file
    else:
        data = np.loadtxt(filename, skiprows=10)

    return (data, files)

def lin_model(params, x, y, plot_fit = False):
    a = params['a'].value
    b = params['b'].value
    
    if plot_fit:
        return a*x + b
    else:
        return a*x + b - y

def double_exp(params, x, y, plot_fit = False):
    a1 = params['a1'].value
    a2 = params['a2'].value
    tau1 = params['tau1'].value
    tau2 = params['tau2'].value
    offset = params['offset'].value

    func = a1*np.exp(-x/tau1) + a2*np.exp(-x/tau2) + offset

    if plot_fit:
        return func
    else:
        return func - y
    

def fit_data(data, title = None, cut = 0, cutend = None, fit = True):

    x = data[:, 0]
    y = data[:, 1]

    if cutend is None:
        cutend = len(x)

    x = x[cut:cutend]
    y = y[cut:cutend]

    x = x * 100e-9/1e-6

    params = Parameters()
    params.add('a1', value = np.max(y))
    params.add('a2', value = np.max(y/10))
    #params.add('tau1', value = 14)
    #params.add('tau2', value = 2)

    params.add('tau1', value = 5, min = 0.0)
    params.add('tau2', value = 0.08)

    params.add('offset', value = np.min(np.mean(y[-5:])))

    result = minimize(double_exp, params, args = (x,y))

    #plt.figure()
    #plt.plot( x, y, 'o' )
    #plt.plot( x, double_exp(result.params, x, y, plot_fit = True))

    plt.figure()
    plt.semilogy( x, y, 'o' )
    if fit == True:
        plt.semilogy( x, double_exp(result.params, x, y, plot_fit = True))

    plt.title(title)
    
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    # blah blah lots of code ...

    report_fit(result.params)

    sys.stdout = old_stdout

    txt = mystdout.getvalue()
    if fit == True:
        plt.text(20, 2e4, txt)
        plt.text(2, 2e4, txt)

    return result.params



(data, files) = load_txt_data()

for k in range(len(data)-1):
    fit_data(data[k], files[k], cut = 0, cutend = 70) # cut tells how many points are left out in the beginning
    fit_data(data[k], files[k], cut = 0) # cut tells how many points are left out in the beginning

fit_data(data[-1], files[-1], fit = False)

plt.xlabel('Time (us)')

plt.show()


