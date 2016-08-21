import numpy as np
import matplotlib.pylab as plt
from lmfit import minimize, Parameters, Parameter, report_fit
import sys
sys.path.append("/usr/bin") # necessary for the tex fonts
sys.path.append("../Python modules/") # necessary for the tex fonts
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


fsizetit = 18
fsizepl = 16
sizex = 8
sizey = 6
dpi_no = 80
lw = 2



def load_data():
    
    data30= np.load('ZZZZZZEr20.npz')
    data60 = np.load('ZZZZZZEr21.npz')
    data120 = np.load('ZZZZZZEr22.npz')

    d2 = np.load('ZZZZZZEr6010.npz')
    d3 = np.load('ZZZZZZEr6011.npz')
    d4 = np.load('ZZZZZZEr609.npz')

    return (data30['data'], data60['data'], data120['data'], d2['data'], d3['data'], d4['data'])

def lin_model(params, x, data, plot_fit = False):
    a = params['a'].value
    b = params['b'].value
    
    if plot_fit:
        return a*x + b
    else:
        return a*x + b - data

def fit_model1(x, y, param_guess, my_color, do_plot=False):
    
    params = Parameters()
    params.add('a', value= param_guess[0], min=param_guess[0]/20.0, max=param_guess[0]*20.0, vary=True)
    params.add('b', value= param_guess[1], min=param_guess[1]/20.0, max=param_guess[1]*20.0, vary=True)
    
    result = minimize(lin_model, params, args=(x, y))
    
    if do_plot:
        plt.figure(1)
        plt.plot( x, y, 'o' + my_color)
        plt.plot( x, lin_model(result.params, x, y, plot_fit = True), my_color)

    report_fit(result.params)
    return result.params


def plot_with_fit(times, data, params, my_color, my_symb,label=None,ms=4):

    ax1 = plt.plot(times, data, my_symb + my_color,label=label,markersize=ms)

    a = params['a'].value
    b = params['b'].value

    
    plt.plot(times, lin_model(params, times, data, plot_fit = True), my_color) # parameter data is not used in this case
    

    plt.ylim([0,1.7])

######################################
#START HERE

(d1, d2, d3, w1, w2, w3) = load_data()

cut1 = [3,10] #confirmed  red dot
cut2 = [2, 5] #confirmed  blue dot
cut3 = [0, 2] #confirmed  black dot

d1_cut = d1[cut1[0]:cut1[1]]
d2_cut = d2[cut2[0]:cut2[1]]
d3_cut = d3[cut3[0]:cut3[1]]

wcut1 = [0,len(w1)] #confirmed  red sq
wcut2 = [0, 10] #confirmed  blue sq
wcut3 = [0, 4] #confirmed  black sq

w1_cut = w1[wcut1[0]:wcut1[1]] #red sq
w2_cut = w2[wcut2[0]:wcut2[1]]#confirmed  blue sq
w3_cut = w3[wcut3[0]:wcut3[1]]#confirmed black sq

my_fontsize = 3



params1 = []

# pixel 28

tfull = np.arange(2,102,2)

# Apt 30
t = tfull[cut1[0]:cut1[1]]
d = d1_cut
params1.append(fit_model1(t, d, [0.01, d[0]], 'r'))

# Apt 60
t = tfull[cut2[0]:cut2[1]]
d = d2_cut
params1.append(fit_model1(t, d, [0.1, d[0]], 'b')) #fits ok 

# Apt 120
t = tfull[cut3[0]:cut3[1]]
d = d3_cut
params1.append(fit_model1(t, d, [0.5, d[0]], 'k')) #fits ok


# pixel 56

# Apt 30
t = tfull[wcut1[0]:wcut1[1]] #np.arange(w1_cut.shape[0])
d = w1_cut
params1.append(fit_model1(t, d, [0.01, d[0]], 'r--'))

# Apt 60
t = tfull[wcut2[0]:wcut2[1]] #np.arange(w2_cut.shape[0])
d = w2_cut
params1.append(fit_model1(t, d, [0.05,d[0]], 'b--')) 

# Apt 120
t = tfull[wcut3[0]:wcut3[1]]  #np.arange(w3_cut.shape[0])
d = w3_cut
params1.append(fit_model1(t, d, [0.5,d[0]], 'k--'))#fits ok


#plt.figure(2)
#
#t = np.arange(d1.shape[0])
#plot_with_fit(t, d1, params1[0], 'r','o')
#plot_with_fit(t, d2, params1[1], 'b','o')
#plot_with_fit(t, d3, params1[2], 'k','o')
#
#plot_with_fit(t, w1, params1[3], 'r','s')
#plot_with_fit(t, w2, params1[4], 'b','s')
#plot_with_fit(t, w3, params1[5], 'k','s')

fit_slopes = []
for k in range(len(params1)):
    fit_slopes.append(params1[k]['a'].value)

current1 = 4.9*1.0e-9 
current2 = 4e-10
pixel1 = 56e-9
pixel2= 28e-9

maxaper = 300

Edose1 = current1  * np.array([30**2, 60**2, 120**2])/maxaper**2/pixel1/pixel1  #* np.arange(2,102,2)*1e-6
Edose2 = current1  *  np.array([30**2, 60**2, 120**2])/maxaper**2/pixel2/pixel2 #* np.arange(2,102,2)*1e-6

plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')
titulo =  'Er core-shell aggregates (2kV, 40ns time bins), `steady-state\' excitation (2$\mu$s beam on for each experiment)'
plt.suptitle('Cathodoluminescence as a function of electron dose, \n' + titulo, fontsize=fsizetit)

ax1 = plt.subplot2grid((1,1), (0,0))
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')




plt.plot(Edose1, fit_slopes[0:3], 's-',c='b',markersize=10) #Er 2%, 4kX

plt.plot(Edose2, fit_slopes[3:6], 'o-',c='r',markersize=10) #Er 60%, 2kX
plt.ylabel(r"Change of luminescence at fitted initial rise (MHz/$\mu$s) ",fontsize=fsizepl)
plt.xlabel(r"Electron flux = beam current / pixel area (A/m$^2$)",fontsize=fsizepl)

#inset
a = plt.axes([0.62, .2, .25, .25]) #[.3, .5, .1, .2]
t = tfull
plot_with_fit(t, d1, params1[0], 'r','o',label='Er 2$\%$,  30$\mu$m aperture, 28nm pixel',ms=4)
plot_with_fit(t, d2, params1[1], 'r','o',label='Er 2$\%$,  60$\mu$m aperture, 28nm pixel',ms=6)
plot_with_fit(t, d3, params1[2], 'r','o',label='Er 2$\%$, 120$\mu$m aperture, 28nm pixel',ms=8)

plot_with_fit(t, w1, params1[3], 'b','s',label='Er 60$\%$,  30$\mu$m aperture, 56nm pixel',ms=4)
plot_with_fit(t, w2, params1[4], 'b','s',label='Er 60$\%$,  60$\mu$m aperture, 56nm pixel',ms=6)
plot_with_fit(t, w3, params1[5], 'b','s',label='Er 60$\%$, 120$\mu$m aperture, 56nm pixel',ms=8)
#plt.xticks(x_vec, labels)
plt.ylabel('Average luminescence \n of each experiment (MHz)',fontsize=fsizepl)
plt.xlabel("Cumulative beam exposure time \n per pixel (nominal, $\mu$s)",fontsize=fsizepl) 
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #plots to the right
plt.legend(bbox_to_anchor=(0.5, 1.7), loc='upper center', borderaxespad=0.)
#plt.legend(loc = 'top')
major_ticks = [25,50,75,100]
a.set_xticks(major_ticks) 

a.spines['right'].set_visible(False)
a.spines['top'].set_visible(False)
a.xaxis.set_ticks_position('bottom')
a.yaxis.set_ticks_position('left')
plt.ylim([0,1.7])
#a.set_yticks([0.5,1.0,1.5]) #[0.05, 0.4, 0.8]
#major_ticks = [25/2,50/2,75/2,100/2]
a.set_xticks(major_ticks) 
#a.set_xticklabels(['25','50','75','100'])
plt.xlim([2,100])


plt.show()
multipage('ZZZ-ApertureComparison.pdf',dpi=80)
    
#[6.6279936686568065e-05, 0.01939657330513702, 0.12613826990127563, 0.00070176342138193932, 0.005144239579431986, 0.15109400144557023]