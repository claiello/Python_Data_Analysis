import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GMM 
import matplotlib.cm as cm
import scipy.ndimage as ndimage
from matplotlib_scalebar.scalebar import ScaleBar
import lmfit
from lmfit import minimize, Parameters, Parameter, report_fit


def mean_of_nonzero(arr, binary_mask):
    # takes the 2D array and return the mean of all non-zero values

    hlp = (binary_mask)*arr[:,:]
     
    hlp_non_zero = np.nonzero(hlp)[0]
    
    # check if hlp_non_zero is empty or not
    if hlp_non_zero.size:
        return np.mean(arr[hlp_non_zero])
    else:
        return 0.0
        
# define objective function: returns the array to be minimized
def fcn2min(params, x, data, single = False, return_plot = False, no_of_x_pts = 100):

    a = params['a'].value
    b = params['b'].value
    c = params['c'].value
    
    if single == False:
        print('HERE')
        d = params['d'].value
        e = params['e'].value

    if return_plot == True:
        # changing x to give more values
        x = np.linspace(np.min(x), np.max(x), no_of_x_pts)
        
    if single:
        model = a*np.exp(-x/b) + c   
    else:
        print('HERE2')
        model = a*np.exp(-x/b) + c + d*np.exp(-x/e)
        #model = a*np.exp(-x/b) + c + (data[0]-a)*np.exp(-x/e)   
     
    if return_plot == False:
        return model - data
    else:
        return (x, model)
        
def fitexp(x,y,single = False):
    
    # create a set of Parameters
    params = Parameters()
    params.add('a', value= 111.0, min=100.0,max=120.0)
    params.add('b', value= 0.78, min=0.3,max=1.0,vary=True)
    params.add('c', value= 11.9, min=9.0,max=14.0)
    if single == False:
        print('HERE')
        params.add('d', value= 56.0, min=30.0,max=70.0)
        params.add('e', value= 1.8, min=1.5,max=3.0,vary=True)
    
    result = minimize(fcn2min, params, args=(x, y))
    (x_fit, y_fit) = fcn2min(result.params, x, y, single, return_plot = True, no_of_x_pts = 100)
    
    report_fit(result.params)
    plt.plot(x_fit, y_fit, 'b')
    return (result.params['a'].value,result.params['b'].value,result.params['c'].value,result.params['d'].value,result.params['e'].value)

plt.close("all")

Ignore_no_first_points = 6
Pixel_size = 0.0820836e-06 #as given by ImageJ; corresponds to mag 7.5kX
total_exp_time = 10e-6
No_frames = 200 #max 240
No_decaypts = No_frames
time_detail = 0.05 # in microsec; 1/expt clock rate
No_specimen = 3

hdf5_file_name = ['Place1.hdf5','Place2.hdf5','Place3.hdf5']
plot_title = 'Yellow Y2O3:Eu film transient fluorescence (2kV, 30$\mu$m, 7.5kX, time detail = 0.05$\mu$s)'

mean_grana_patch_blue = np.zeros([No_specimen,No_decaypts])
sum_grana_blue = np.zeros([No_decaypts])

fig2 = plt.figure(figsize=(8, 6), dpi=80)
fig2.suptitle(plot_title,  fontsize=16)

#START OF LOOP

for k in range(No_specimen): 
    
    ##########################
    # Reading data 

    file    = h5py.File(hdf5_file_name[k], 'r')  
    se_dset   = file['/data/Analog channel 1 : SE2/data']
    #red_dset   = file['/data/Counter channel 1 : PMT red/PMT red/data']
    blue_dset   = file['/data/Counter channel 2 : PMT blue/PMT blue time-resolved TRANSIENT/data']

    ##########################
    # Mean fluo
    
    for kk in range(No_decaypts):

        mean_grana_patch_blue[k,kk] = np.mean(blue_dset[:,kk, :, :])
    
    x_array = np.array(range(No_decaypts))
    
    sum_grana_blue += mean_grana_patch_blue[k]
   
    ax1 = plt.subplot2grid((3, 4), (k, 0), colspan=1, rowspan=1)
    if k == 0:
        ax1.set_title('SE2')
        
    plt.imshow(se_dset[0,:,:], cmap = cm.Greys_r)
    scalebar = ScaleBar(Pixel_size, frameon = True, box_alpha = 0.001, location = 'lower left') # 1 pixel = Pixel_size in meter
    plt.gca().add_artist(scalebar)
    plt.axis('off')
    
     #plot
    ax2 = plt.subplot2grid((3, 4), (0, 1), colspan=3, rowspan=3)
    plt.hold(True)
    for kkk in range(No_specimen):
        plt.plot(x_array[:-Ignore_no_first_points]*time_detail,mean_grana_patch_blue[kkk,Ignore_no_first_points:]/1000.0,'bo',markersize=(kkk+1),label='Decay region ' + str(kkk+1)) 
        #plt.semilogx(x_array[:-1]*time_detail,mean_grana_patch_blue[kkk,1:]/1000.0,'bo',markersize=(kkk+1),label='Decay region ' + str(kkk+1))         
        plt.hold(True)    
    
    

# dofitting
# do fit, here with leastsq model
(a,b,c,d,e) = fitexp(x_array[:-Ignore_no_first_points]*time_detail,sum_grana_blue[Ignore_no_first_points:]/1000.0/No_specimen,single=False)  
    
plt.plot(x_array[:-Ignore_no_first_points]*time_detail,(sum_grana_blue[Ignore_no_first_points:])/1000.0/No_specimen,'bo',label='Average decay, $ \\tau_1 $ = ' + str("{0:.2f}".format(b)) + '$\mu$s; $ \\tau_2 $ = ' + str("{0:.2f}".format(e)) + '$\mu$s',markersize=(kkk+1)+2)     
#plt.semilogx(x_array[:-1]*time_detail,sum_grana_blue[1:]/1000.0/No_specimen,'bo',label='Average decay, $\\tau$ = ' + str("{0:.2f}".format(1.0/b)) + '$\mu$s',markersize=(kkk+1)+2)     
plt.xlabel('Time after blanking the eletron beam ($\mu$s)',  fontsize=14)
plt.ylabel('Fluorescence rate (kcps)',  fontsize=14)
plt.legend(loc='best')
    

plt.show()