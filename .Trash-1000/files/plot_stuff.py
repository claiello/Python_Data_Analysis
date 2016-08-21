import numpy as np
import lmfit
import matplotlib.pylab as plt

def plot_expt_by_expt_behaviour_Er60(titulo, data1, data2, data3, Time_bin, nominal_time_on,fastfactor,my_color_avg,major_ticks,no_points,aper,current,plot_inset=False,x_vec=None, tau1=None, tau1_error=None, tau2=None, tau2_error=None): #dset is no_expt x no_time_bins x pixel x pixel; dset.shape[0] is no_expts, Time_bin in ns, nominal_time_on in mus

   lw = 1
   fsizepl = 10

   plt.figure(figsize=(sizex, sizey), dpi=dpi_no)
   
   plt.rc('text', usetex=True)
   
   plt.rc('font', family='serif')
   
   plt.rc('font', serif='Palatino')
   
   plt.suptitle("Cathodoluminescence (CL) as a function of e-beam exposure time, \n " + titulo, fontsize=fsizetit)
   
   ax1 = plt.subplot2grid((1,1), (0, 0), colspan=1)
   
   ax1.spines['right'].set_visible(False)
   
   ax1.spines['top'].set_visible(False)
   
   ax1.xaxis.set_ticks_position('bottom')
   
   ax1.yaxis.set_ticks_position('left')
   
   #color=iter(cm.rainbow(np.linspace(0,1,noplots)))
   
   lab = aper
   
   #for kk in np.arange(3):
   
   #c = next(color)
   
   plt.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,data1,c='k', label=lab[0],linewidth=lw) #in mus, in MHz
   
   #c = next(color)
   
   plt.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,data2,c='r', label=lab[1],linewidth=lw+3) #in mus, in MHz
   
   #c = next(color)
   
   plt.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,data3,c='b', label=lab[2],linewidth=lw+6) #in mus, in MHz
   
   plt.ylabel(r"Average frame luminescence of each experiment (MHz) [solid lines]",fontsize=fsizepl)
   
   plt.xlabel(r"Cumulative e-beam exposure time per pixel (nominal, $\mu$s)",fontsize=fsizepl)
   
   #major_ticks = [25,50,75,nominal_time_on*dset.shape[0]*fastfactor]
   
   ax1.set_xticks(major_ticks[:-1])
   
   plt.xlim([nominal_time_on,nominal_time_on*no_points*fastfactor])
   
   plt.legend(loc='upper center')#(loc='upper left')
   
   ax2 = ax1.twinx()
   
   aper2 = [30,60,120]
   
   #color=iter(cm.rainbow(np.linspace(0,1,noplots)))
   
   # for kkk in np.arange(3):
   
   #c = next(color)
   
   ax2.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,np.arange(1,no_points+1)*nominal_time_on*fastfactor* current * np.pi* np.power(aper2[0]/2.0,2),'--', c='k',linewidth=lw)
   
   ax2.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,np.arange(1,no_points+1)*nominal_time_on*fastfactor* current * np.pi* np.power(aper2[1]/2.0,2),'--', c='r',linewidth=lw+3)
   
   ax2.plot(np.arange(1,no_points+1)*nominal_time_on*fastfactor,np.arange(1,no_points+1)*nominal_time_on*fastfactor* current * np.pi* np.power(aper2[2]/2.0,2),'--', c='b',linewidth=lw+6)
   
   ax2.set_ylabel("Electron dose $\propto$ cumulative e-beam exposure time $\cdot$ current $\cdot$ aperture area (a.u.) [dashed lines]",fontsize=fsizepl)
   
   ax2.set_yticks([])

   plt.show()





data30= np.load('ZZZZZZEr20.npz')

data60 = np.load('ZZZZZZEr21.npz')

data120 = np.load('ZZZZZZEr22.npz')

no_points = data30['data'].shape[0] #no of experiments

aper = ['30$\mu$m aperture', '60$\mu$m aperture','120$\mu$m aperture']

current = 4e-10 #Amps, as given by txt files


Ps = [0,1,2]
Time_bin = np.linspace(0,200,200/40) * 1e-9
nominal_time_on = 2
sizex = 100
sizey = 100
dpi_no = 10
fsizetit = 100

titulo = 'Er ' + str(2) + '$\%$ core-shell aggregates (4kX, 2kV, 40ns time bins, ' + str(Ps[0]) + 'nm pixels), signal pixels' #0 here is arbitrary, all pixels same

fastfactor = 1

x_vec = [30,60,120]

tau1 = [1.11, 1.19, 1.37]

tau1_error = [0.07, 0.06,0.06]

tau2 = [0.06, 0.05,0.04]

tau2_error = [0.005, 0.005,0.005]

plot_expt_by_expt_behaviour_Er60(titulo, data30['data'], data60['data'], data120['data'], Time_bin, nominal_time_on,fastfactor,'r',[25,50,75,100],no_points,aper, current,plot_inset=True, x_vec=x_vec, tau1=tau1,tau1_error=tau1_error,tau2=tau2,tau2_error=tau2_error)

print data30['data']


