from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import numpy as np

def print_result(result):
    
    for k in result.var_names:
        print(k + " = " + str(result.params[k].value))
        
def choose_func_poly(x, params, choice):
    tau = params['tau'].value
    A = params['A'].value
    tau2 = params['tau2'].value
    A2 = params['A2'].value
    tau3 = params['tau3'].value
    A3 = params['A3'].value
                
    c = params['c'].value
    P = params['P'].value
    
    if choice is 'singleexp':
        # single exp
        return c*x + A*(1.- np.exp(-x/tau))
    elif choice is 'doubleexp':
        #double exp
        return c*x + A*(1.- np.exp(-x/tau)) + A2*(1- np.exp(-x/tau2))
    elif choice is 'tripleexp':
        # triple exponential    
        return c*x + A*(1.- np.exp(-x/tau)) + A2*(1- np.exp(-x/tau2)) + A3*(1- np.exp(-x/tau3)) 
    elif choice is 'weirdexp':
        return c*x + A*(1.- np.exp(-x/tau)) + P*(x**(3./2.))
    else:
        print('model not recognized')
        klklk
        
def choose_poly_fit(x,y,choice,y_err,doPoisson):
    
    # define objective function: returns the array to be minimized
    def fcn2min(params, x, data):
        """ model decaying sine wave, subtract data"""

        model = choose_func_poly(x, params, choice)
        
        if doPoisson:
            print('doing Poisson')
            return 2*(data*np.log(data) - data*np.log(model) - (data-model)) #Poisson            
        else:
            if y_err is not None:
                return (model - data)/y_err**2
            else:
                return (model - data) 
      
    initC, init_b, doofdummy = linear_fit(x[-500:], y[-500:])

    #(result.params['tau'], result.params['A'], result.params['tau2'], result.params['A2'], result.params['c'], result)

    #(h1, h2, h3, h4, h5, result_hlp) = cumu_fit(x, y)    
    
    # create a set of Parameters
    params = Parameters()
    #params.add('tau', value = 2000.0, min = 1.0e-6, max = 5000.0, vary = True)
    params.add('tau', value = 250.0, min = 1.0e-3,  vary = True)
    params.add('A', value = 100.0,  min = 0.0, vary = True)
  
    params.add('tau2', value = 25.0, min = 1.0e-3, vary = True)        
    #params.add('tau2', value = 50.0, min = 10.0, max = 100, vary = True)
    params.add('A2', value = 100.0, min = 0.0, vary = True)
    
    params.add('tau3', value = 1.0, min = 1.0e-3,  vary = True) 
    #params.add('tau3', value = 100.0, min = 100.0, max = 1000.0, vary = True)
    params.add('A3', value = 100.0, min = 0.0, vary = True)
 
    #initC = 1e3
    params.add('c', value = initC, min = 1.0e-5, vary = True)
    params.add('P', value = 0.01, min = 1.0e-6, vary = True)

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min, params, fcn_args=(x, y))
    result = minner.minimize()

#    print(result.params)
#    print(result.params['tau'].value)
#    print(result.params['tau2'].value)
#    print(result.params['tau3'].value)
#    return (result.params['tau'], result.params['A'], result.params['tau2'], result.params['A2'], result.params['tau3'], result.params['A3'], result.params['c'],  result.params['P'],result)
 
    return result
    
def func_poly(x, params):
    tau = params['tau'].value
    A = params['A'].value
    tau2 = params['tau2'].value
    A2 = params['A2'].value
    tau3 = params['tau3'].value
    A3 = params['A3'].value
                
    c = params['c'].value
    
    # single exp
    return c*x + tau*A*(1- np.exp(-x/tau))
    #double exp
    #return c*x + tau*A*(1- np.exp(-x/tau)) + tau2*A2*(1- np.exp(-x/tau2))

    # triple exponential    
    #return c*x + tau*A*(1- np.exp(-x/tau)) + tau2*A2*(1- np.exp(-x/tau2)) + tau3*A3*(1- np.exp(-x/tau3)) 
    
    # arctan    
    #return c*x + A * np.arctan(x**tau2/tau)
            
        
def poly_fit(x,y):
    
    # define objective function: returns the array to be minimized
    def fcn2min(params, x, data):
        """ model decaying sine wave, subtract data"""

        model = func_poly(x, params)
        
        return (model - data) #/y_err**2
        
        #print(data)
        #print(model)
        #print('beware doing poisson')
        #return 2*(data*np.log(data) - data*np.log(model) - (data-model)) #Poisson 


    initC, init_b, doofdummy = linear_fit(x[-500:], y[-500:])


    #(result.params['tau'], result.params['A'], result.params['tau2'], result.params['A2'], result.params['c'], result)

    #(h1, h2, h3, h4, h5, result_hlp) = cumu_fit(x, y)    
    
    # create a set of Parameters
    params = Parameters()
    params.add('tau', value = 0.1, min = 1.0e-3, vary = True)
    #params.add('tau', value = 1.0, min = 1.0, max = 50.0, vary = True)
    params.add('A', value = 0.1,  vary = True)
  
    params.add('tau2', value = 0.1, min = 1.0e-3, vary = True)        
    #params.add('tau2', value = 50.0, min = 10.0, max = 100, vary = True)
    params.add('A2', value = 0.1, vary = True)
    
    params.add('tau3', value = 0.1, min = 1.0e-3,  vary = True) 
    #params.add('tau3', value = 100.0, min = 100.0, max = 1000.0, vary = True)
    params.add('A3', value = 0.1, vary = True)
 
    params.add('c', value = initC, min = 1.0e-3, vary = True)

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min, params, fcn_args=(x, y))
    result = minner.minimize()

    print(result.params)
    print(result.params['tau'].value)
    print(result.params['tau2'].value)
    print(result.params['tau3'].value)
    return (result.params['tau'], result.params['A'], result.params['tau2'], result.params['A2'], result.params['c'], result)
    

def cumu_fit(x,y):
    
    print(np.sum(np.isnan(x)))
    print(np.sum(np.isnan(y)))
    
    # define objective function: returns the array to be minimized
    def fcn2min(params, x, data):
        """ model decaying sine wave, subtract data"""
        tau = params['tau'].value
        A = params['A'].value
        tau2 = params['tau2'].value
        A2 = params['A2'].value
                
        c = params['c'].value

        model = c*x + tau*A*(1- np.exp(-x/tau)) + tau2*A2*(1- np.exp(-x/tau2))
        
        return (model - data) #/y_err**2


    initC, init_b, doofdummy = linear_fit(x[-500:], y[-500:])
    # create a set of Parameters
    params = Parameters()
    params.add('tau', value = 0.1, min = 1.0e-3, vary = True)
    params.add('A', value = 0.1, vary = True)
    params.add('tau2', value = 0.1, min = 1.0e-3, vary = True)
    params.add('A2', value = 0.1, vary = True)
        
    params.add('c', value = initC, vary = True)

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min, params, fcn_args=(x, y))
    result = minner.minimize()

    print(result.params)

    return (result.params['tau'], result.params['A'], result.params['tau2'], result.params['A2'], result.params['c'], result)

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

    #print(result.params)

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

    #print(result.params)

    return (result.params['a'], result.params['b'], result)
    
def Tminus1_fit_with_error(x, y, y_err):

    # define objective function: returns the array to be minimized
    def fcn2min(params, x, data):
        """ model decaying sine wave, subtract data"""
        a = params['a'].value
        b = params['b'].value

        model = 1/(a * x + b)
        
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

    #print(result.params)

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
    
def fit_ratio_visib_improved(x, y, normtemp,yerr):
    
    # define objective function: returns the array to be minimized
    def fcn2min(params, x, data):
       
        c = params['c'].value  ####### 
        DeltaE = params['DeltaE'].value  ####### 
        d = params['d'].value
        
        k = 8.6173303*1.0e-5 #eV/K

        #model = 1.0 * np.tanh(-x*DeltaE/(2.0*k) + c)  #x is 1/T
        
        #smalldelta = params['smalldelta'].value  ####### 
        #model = k/x/smalldelta * (np.log(np.cosh(c + ((-DeltaE+smalldelta)/(2*k/x)))) - np.log(np.cosh(c - ( ((DeltaE+smalldelta)/(2*k/x))))))

        #model = (np.exp(2.*c) + (d-1) * np.exp(x*DeltaE/k))/(np.exp(2.0*c) + (d+1.0) * np.exp(x*DeltaE/k))
        
        #model = model / ((np.exp(2.*c) + (d-1.0) * np.exp(1.0/normtemp*DeltaE/k))/(np.exp(2.0*c) + (d+1.0) * np.exp(1.0/normtemp*DeltaE/k)))
        
        
        model = (1. + (d-1.) * np.exp(x*DeltaE/k - 2.*c))/(1. + (d+1.0) * np.exp(x*DeltaE/k - 2.*c))
        
        model = model / (1. + (d-1.) * np.exp((1./normtemp)*DeltaE/k - 2.*c))/(1. + (d+1.0) * np.exp((1./normtemp)*DeltaE/k - 2.*c))
        
        
        return np.abs(model - data)**2
    
        #print('doing err')
        #return (model - data)**2.0/yerr**2.0
        
        #print('beware doing poisson')
        #return 2*(data*np.log(data) - data*np.log(model) - (data-model)) #Poisson 


#    import matplotlib.pyplot as plt
#
#    x = np.linspace(1./500, 1./300, 20)
# 
#    DeltaE = 0.0
#    k = 8 * 1e-5
#    c = 2.5
#    smalldelta = 0.0001
#    
#    model1 = 1.0 * np.tanh(-x*DeltaE/(2.0*k) + c)  #x is 1/T
#        
#    model2 = k/x/smalldelta * (np.log(np.cosh(c + ((-DeltaE+smalldelta)/(2*k/x)))) - np.log(np.cosh(c - ( ((DeltaE+smalldelta)/(2*k/x))))))
#     
#    plt.plot(x, model1)
#    plt.plot(x, model2, 'ro-')
#    plt.show()

    ind = np.argsort(x)
    print(ind[-1])
    y = y/y[ind[-1]]
    
    print(x)
    print(y)
    print('below yerr')
    print(yerr/y[ind[-1]])
    
    # create a set of Parameters
    params = Parameters()
    params.add('DeltaE', value = 0.4, min = 0.05, max = 0.7, vary = False)
    params.add('c', value = 6.7, min = 5.0, max = 8.0, vary = True)
    params.add('d', value = 2.2,  min = 0.5, max = 3.5, vary = True)

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min, params, fcn_args=(x, y))
    result = minner.minimize()

    #print(x)
    #print(y)
    
    
    
#    N1 = 50
#    N2 = 50
#    dE_arr = np.linspace(-5, 5, N1)
#    dc_arr = np.linspace(2, 10, N2)
#    
#    result = np.zeros([N1, N2])
#    for dE_ind in range(len(dE_arr)):
#        for dc_ind in range(len(dc_arr)):
#                params = Parameters()
#                params.add('DeltaE', value = 0.4)
#                params.add('d', value = dE_arr[dE_ind])
#                params.add('c', value = dc_arr[dc_ind])
#
#                #print(fcn2min(params, x, y))
#                result[dE_ind, dc_ind] = np.sum(fcn2min(params, x, y))
#    import matplotlib.pyplot as plt
#    plt.figure()    
#    plt.pcolor(dc_arr, dE_arr, np.log(result)  , vmin = -5, vmax = 15   )
#    plt.colorbar()    
#    plt.show()
#
#    print(result.params['c'])
#    print(result.params['DeltaE'])
#    print(result.params['d'])

    return (result.params['c'], result.params['DeltaE'], result.params['d'], result)
   
      

def fit_with_plot_small_ratio_visib(ax_name, x, y, yerr = None, my_color = 'r', my_edgecolor='#ff3232', my_facecolor='#ff6666', my_linestyle = 'solid', normtemp=None, axnew=None,mso=None,my_hatch=None):
   
    print(x)
    print(y)
   
    if normtemp is None:
        print('Normtemp assumed 25C')
        normtemp = 25 + 273.15
    print(normtemp - 273.15)
   
    from FluoDecay import d_line
    # fit    
    if True: # yerr is None:
        print('here')
        (c, DeltaE, d, result) = fit_ratio_visib_improved(x, y,normtemp,yerr) 
        print(DeltaE)
        print(c)
        print(d)
    else:
        print('not programmed yet')
    
    ###
#    sigma_dev = np.sqrt( [result.covar[0,0],result.covar[1,1]] )
#    values = np.array([])
#    my_hlp = np.array([])
#    for s1 in [-1, +1]:
#        for s2 in [-1, +1]:
#                my_hlp = d_line( x, result.params['DeltaE'].value + s1*sigma_dev[0], result.params['c'].value + s2*sigma_dev[1])
#                values = np.vstack((values, my_hlp)) if values.size else my_hlp
#    fitError = np.std(values, axis=(0,1))     
    ###

    k = 8.6173303*1.0e-5 #eV/K  
    if axnew is not None:
        #axnew.plot(x*1.0e3, (1.0 * np.tanh(-x*DeltaE.value/(2.0*k) + c.value)/(1.0 * np.tanh(-DeltaE.value/(2.0*k)/normtemp + c.value))), color='k',ls='-',lw=3)
        #
        print("Normtemp: " + str(1/normtemp))
        f1 = (np.exp(2.*c.value) + (d.value-1.) * np.exp(x*DeltaE.value/k))/(np.exp(2.*c.value) + (d.value+1.) * np.exp(x*DeltaE.value/k))
        fnorm = (np.exp(2.*c.value) + (d.value-1.) * np.exp(1./normtemp*DeltaE.value/k))/(np.exp(2.*c.value) + (d.value+1.) * np.exp(1./normtemp*DeltaE.value/k))
        
        axnew.plot(x*1.0e3, f1/fnorm, color='k',ls='-',lw=3)

        #axnew.fill_between(x*1.0e3,  
#                           (1.0 * np.tanh(-x*DeltaE.value/(2.0*k) + c.value)/(1.0 * np.tanh(-DeltaE.value/(2.0*k)/normtemp + c.value)))-1.0*fitError,  
#                           (1.0 * np.tanh(-x*DeltaE.value/(2.0*k) + c.value)/(1.0 * np.tanh(-DeltaE.value/(2.0*k)/normtemp + c.value)))+1.0*fitError, 
#                           color = 'none',
#                           hatch=my_hatch,
#                           edgecolor='k',
#                           facecolor=[168/256,175/256,175/256],
#                           alpha=0.5,
#                           linewidth=0.0)

        label1= '' #'$\Delta E$ = ' + str("{0:.4f}".format(DeltaE.value))+ ' $\pm$ ' + str("{0:.4f}".format(sigma_dev[0])) + r', c = ' + str("{0:.4f}".format(c.value)) + ' $\pm$ ' +  str("{0:.4f}".format(sigma_dev[1])) 
        
    
    return label1,result
    
def fit_with_plot_small_ratio(ax_name, x, y, yerr = None, my_color = 'r', my_edgecolor='#ff3232', my_facecolor='#ff6666', my_linestyle = 'solid', normtemp=None):
   
    if normtemp is None:
        print('Normtemp assumed 25C')
        normtemp = 25 + 273.15
    #print(normtemp - 273.15)
   
    from FluoDecay import d_line
    # fit    
    if yerr is None:
        (a,b,result) = linear_fit(x, y)
        
    else:
        print('not programmed yet')
    
    k = 8.6173303*1.0e-5 #eV/K  
    #print(a.value*k)

    ax_name.plot(x*1.0e3,a*x+b , color=my_color,ls=my_linestyle,lw=3)

    
    label1= r'$\Delta E$ = ' + str("{0:.4f}".format(DeltaE.value))+ r' c = ' + str("{0:.4f}".format(c.value))
    
    return label1
