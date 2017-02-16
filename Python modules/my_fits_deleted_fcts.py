#def fit_with_plot_small_ratio_visib2(ax_name, x, y, yerr = None, my_color = 'r', my_edgecolor='#ff3232', my_facecolor='#ff6666', my_linestyle = 'solid', normtemp=None,result=None):
#   
#    if normtemp is None:
#        print('Normtemp assumed 25C')
#        normtemp = 25 + 273.15
#    #print(normtemp - 273.15)
#   
#    from FluoDecay import d_line
#    # fit    
#    if yerr is None:
#        #(a, C, result) = fit_ratio_visib(x, y)  
#        #print(a.value)
#        #print(C.value)      
#        #(C, DeltaE, result) = fit_ratio_visib_improved2(x, y,normtemp) 
#        #(c, DeltaE, result) = fit_ratio_visib_improved2(x, y,normtemp) 
#        #(C, DeltaE, Ko, result) = fit_ratio_visib_improved2(x, y,normtemp)  
#        (c, DeltaE, Ko, result) = fit_ratio_visib_improved2(x, y,normtemp,result) 
#        print(c.value)
#        print(DeltaE.value)
#        print(Ko.value)
#        #print()
#    else:
#        print('not programmed yet')
#    
#    #ax_name.plot(x*1.0e3, 1.0 - 2.0/(1.0 + C*np.exp(a*x)), color=my_color,ls=my_linestyle,lw=3) #,label=r'$\tau$' + ' $\sim $ ' + str("{0:.4f}".format(a.value))+ r'$\times$T($^{\circ}$C) + ' + str("{0:.4f}".format(C.value)))
#
#    k = 8.6173303*1.0e-5 #eV/K  
#    ax_name.plot(x*1.0e3, Ko.value/(np.tanh(-DeltaE.value/(2.0*k)/normtemp + c.value)) * np.tanh(-x*DeltaE.value/(2.0*k) + c.value), color='g',ls=my_linestyle,lw=3)
#    return label1  
    
#    def fit_ratio_visib_improved2(x, y, normtemp,resulto):
#
#    # define objective function: returns the array to be minimized
#    def fcn2min(params, x, data):
#       
#        #C = params['C'].value  ####### 
#        c = params['c'].value  ####### 
#        DeltaE = params['DeltaE'].value  ####### 
#        Ko = params['Ko'].value 
#        #####x is 1/T
#        
#        k = 8.6173303*1.0e-5 #eV/K
#
#        model = Ko/(np.tanh(-DeltaE/(2.0*k)/normtemp + c)) * np.tanh(-x*DeltaE/(2.0*k) + c)
#    
#        return model - data
#        
#        #print('beware doing poisson')
#        #return 2*(data*np.log(data) - data*np.log(model) - (data-model)) #Poisson 
#
#    # create a set of Parameters
#    params = Parameters()
#    params.add('DeltaE', value = resulto.params['DeltaE'].value, min = -1.0, max = +1.0, vary = True)
#    params.add('c', value = resulto.params['c'].value, vary = True)
#    #params.add('C', value = 0.1, vary = True)
#    params.add('Ko', value = 1.0, vary = False)
#
#    # do fit, here with leastsq model
#    minner = Minimizer(fcn2min, params, fcn_args=(x, y))
#    result = minner.minimize()
#
#    return (result.params['c'], result.params['DeltaE'],result.params['Ko'],result)
    
#    def fit_ratio_visib(x, y):
#
#    # define objective function: returns the array to be minimized
#    def fcn2min(params, x, data):
#        """ model decaying sine wave, subtract data"""
#        a = params['a'].value  ####### Delta E/k
#        C = params['C'].value   #####contstant
#        #####x is 1/T
#
#        model = 1.0 - 2.0/(1.0 + C*np.exp(a*x))
#    
#        return model - data
#
#    # create a set of Parameters
#    params = Parameters()
#    params.add('a', value = 0.1, vary = True)
#    params.add('C', value = 0.1, vary = True)
#
#    # do fit, here with leastsq model
#    minner = Minimizer(fcn2min, params, fcn_args=(x, y))
#    result = minner.minimize()
#
#    #print(result.params)
#
#    return (result.params['a'], result.params['C'], result)