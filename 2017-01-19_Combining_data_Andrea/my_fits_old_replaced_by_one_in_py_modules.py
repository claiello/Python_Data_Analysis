from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import numpy as np

def linear_fit(x, y):

    # define objective function: returns the array to be minimized
    def fcn2min(params, x, data):
        """ model decaying sine wave, subtract data"""
        a = params['a'].value
        b = params['b'].value

        model = a * x + b
    
        return model - data

    # create a set of Parameters
    params = Parameters()
    params.add('a', value = 0.1, vary = True)
    params.add('b', value = 0.1, vary = True)

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min, params, fcn_args=(x, y))
    result = minner.minimize()

    print(result.params)

    return (result.params['a'], result.params['b'], result)
    
def linear_fit_with_error(x, y, y_err):

    # define objective function: returns the array to be minimized
    def fcn2min(params, x, data):
        """ model decaying sine wave, subtract data"""
        a = params['a'].value
        b = params['b'].value

        model = a * x + b
        
        #print('beware doing poisson')
        #return 2*(data*np.log(data) - data*np.log(model) - (data-model)) #Poisson 
        
        return (model - data)**2/y_err**2

    # create a set of Parameters
    params = Parameters()
    params.add('a', value = 0.1, vary = True)
    params.add('b', value = 0.1, vary = True)

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min, params, fcn_args=(x, y))
    result = minner.minimize()

    print(result.params)

    return (result.params['a'], result.params['b'], result)



def fit_with_plot(ax_name, x, y, yerr = None, my_color = 'r', my_edgecolor='#ff3232', my_facecolor='#ff6666', my_linestyle = 'solid'):

    from FluoDecay import d_line
    # fit    
    if yerr is None:
        (a, b, result) = linear_fit(x, y)        
    else:
        (a, b, result) = linear_fit_with_error(x, y, yerr)
    
    ax_name.plot(x, a*x+b, color=my_color,ls=my_linestyle,lw=3,label=r'$\tau$' + ' $\sim $ ' + str("{0:.2f}".format(a.value))+ r'$\times$T($^{\circ}$C) + ' + str("{0:.2f}".format(b.value)))
    label1=r'$\tau$' + ' $\sim $ ' + str("{0:.2f}".format(a.value))+ r'$\times$T($^{\circ}$C) + ' + str("{0:.2f}".format(b.value))    
    
    sigma_dev = np.sqrt( [result.covar[0,0],result.covar[1,1]] )
    values = np.array([])
    my_hlp = np.array([])
    for s1 in [-1, +1]:
        for s2 in [-1, +1]:
                my_hlp = d_line( x, result.params['a'].value + s1*sigma_dev[0], result.params['b'].value + s2*sigma_dev[1])
                values = np.vstack((values, my_hlp)) if values.size else my_hlp
    fitError = np.std(values, axis=(0,1))     
    
    xn = np.sort(x)
    ax_name.fill_between(xn, (a*xn+b)-1.0*fitError, (a*xn+b)+1.0*fitError, alpha = 0.5, edgecolor = my_edgecolor, facecolor = my_facecolor)
    
    return label1
    
def fit_with_plot_small(ax_name, x, y, yerr = None, my_color = 'r', my_edgecolor='#ff3232', my_facecolor='#ff6666', my_linestyle = 'solid'):

    from FluoDecay import d_line
    # fit    
    if yerr is None:
        (a, b, result) = linear_fit(x, y)        
    else:
        (a, b, result) = linear_fit_with_error(x, y, yerr)
    
    ax_name.plot(x, a*x+b, color=my_color,ls=my_linestyle,lw=3,label=r'$\tau$' + ' $\sim $ ' + str("{0:.4f}".format(a.value))+ r'$\times$T($^{\circ}$C) + ' + str("{0:.4f}".format(b.value)))
    label1=r'$\tau$' + ' $\sim $ ' + str("{0:.4f}".format(a.value))+ r'$\times$T($^{\circ}$C) + ' + str("{0:.4f}".format(b.value))    
    
    sigma_dev = np.sqrt( [result.covar[0,0],result.covar[1,1]] )
    values = np.array([])
    my_hlp = np.array([])
    for s1 in [-1, +1]:
        for s2 in [-1, +1]:
                my_hlp = d_line( x, result.params['a'].value + s1*sigma_dev[0], result.params['b'].value + s2*sigma_dev[1])
                values = np.vstack((values, my_hlp)) if values.size else my_hlp
    fitError = np.std(values, axis=(0,1))     
    
    #xn = np.sort(x)
    #ax_name.fill_between(xn, (a*xn+b)-1.0*fitError, (a*xn+b)+1.0*fitError, alpha = 0.5, edgecolor = my_edgecolor, facecolor = my_facecolor)
    
    return label1
    