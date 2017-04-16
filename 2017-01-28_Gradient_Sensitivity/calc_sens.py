import numpy as np
from uncertainties import unumpy
import os
import sys


from my_fits import linear_fit_with_error
import matplotlib.pyplot as plt

def find_nearest(array, value):
    # returns the nearest value in the matrix
    idx = (np.abs(array-value)).argmin()
    #return array[idx]
    return idx

def get_sens(S, Q, t, rho = None, use_poisson = False):
    # gets unumpy array
    # S has the form S[Qi, ti]
    # some checks
    if len(t) != S.shape[1]:
        print("Time array doesn't correspond to matrix size")
        return None
    if len(Q) != S.shape[0]:
        print("Quantity array doesn't correspond to matrix size")
        return None

    # get gradient

    # for constant time
    grad_Q_rho = np.array([])
    grad_Q_rho_err = np.array([])
    grad_Q_rho_err_of_sig = np.array([])
    err_S_rho = np.array([])
    for t_ind in range(len(t)):
        
        print(t_ind)

        x = Q 
        y = unumpy.nominal_values(S[:, t_ind])
        y_err = unumpy.std_devs(S[:, t_ind])
        #print(x)
        #print(y)
        #print(y_err)
        
        try:
            (a,b,result) = linear_fit_with_error(x, y, y_err, use_poisson = use_poisson)
        except:
            print('x, y fit did not converge')
            a = np.nan
            b = np.nan
            
        
        grad_Q_rho = np.append(grad_Q_rho, a)
        try:
            grad_Q_rho_err = np.append(grad_Q_rho_err, result.covar[0,0]**0.5)
        except:
            print('grad_Q fit did not converge')
            grad_Q_rho_err = np.append(grad_Q_rho_err, np.nan)
        grad_Q_rho_err_of_sig = np.append(grad_Q_rho_err_of_sig, unumpy.std_devs(np.mean(S[:,t_ind])))

        #plt.errorbar(x, y, yerr = y_err, color = 'r', marker = 'o', ls = None)
        #plt.plot(x, a.value*x+b.value, 'r--')

        #plt.xlabel('Q')
        #plt.ylabel('signal')

    #plt.figure()
    # for constant value in the matrix

    if rho is not None:
        grad_Q_time = np.array([])
        grad_Q_time_err = np.array([])
        grad_Q_time_err_of_sig = np.array([])
        err_S_time = np.array([])
        for rho_ind in range(len(rho)):
            
            print(rho_ind)
    
            x = Q
            hlp = unumpy.nominal_values(S)
            hlp_err = unumpy.std_devs(S)
            y = np.array([])
            y_err = np.array([])
           
            # first get the times for constant entries of the matrix, given by the array rho
            for ind_Q in range(len(x)):
                #idx = find_nearest(hlp[ind_Q, :], rho[rho_ind])
                
                idx = (np.abs(hlp[ind_Q, :] - rho[rho_ind])).argmin()
    
                y = np.append(y, t[idx])
                y_err = np.append(y_err, hlp_err[ind_Q, idx])
    
            try:
                (a,b,result) = linear_fit_with_error(x, y, y_err, use_poisson = use_poisson)
            except:
                print('x, y, fit did not converge')
                a = np.nan
                b = np.nan

    
            #plt.errorbar(x, y, yerr = y_err, color = 'r', marker = 'o', ls = None)
            #plt.plot(x, a.value*x+b.value, 'r--')
    
            grad_Q_time = np.append(grad_Q_time, a)
            try:
                grad_Q_time_err = np.append(grad_Q_time_err, result.covar[0,0]**0.5)
            except:
                print('grad_Q fit did not converge')
                grad_Q_time_err = np.append(grad_Q_time_err, np.inf)
              
            hlp = unumpy.uarray(y,y_err)
            grad_Q_time_err_of_sig = np.append(grad_Q_time_err_of_sig,unumpy.std_devs(np.mean(hlp)))
    
            #plt.errorbar(x, y, yerr = y_err, color = 'r', marker = 'o', ls = None)
            #plt.plot(x, a.value*x+b.value, 'r--')
    
            #plt.xlabel('Q')
            #plt.ylabel('time')
       # eta_time has length len(rho)
        eta_time = np.abs(grad_Q_time_err/grad_Q_time)
        eta_time_sig = np.abs(grad_Q_time_err_of_sig/grad_Q_time)

    # eta_rho has length len(t)
    eta_rho = np.abs(grad_Q_rho_err/grad_Q_rho)
    eta_rho_sig = np.abs(grad_Q_rho_err_of_sig/grad_Q_rho)
    
    

#    plt.figure()
#    plt.subplot(2,1,1)
#    plt.plot(t, grad_Q_rho, 'r')
#    plt.subplot(2,1,2)
#    plt.plot(rho, grad_Q_time, 'g')
#    plt.ylabel("gradient")
    if rho is not None:
        return (eta_rho, eta_time,eta_rho_sig, eta_time_sig)
    else: 
        return (eta_rho, eta_rho_sig)
    

def plot_S(S, Q, t):

    S_nom = unumpy.nominal_values(S)
    S_std = unumpy.std_devs(S)

    for Q_ind in range(len(Q)):
        plt.errorbar(t, S_nom[Q_ind, :], yerr = S_std[Q_ind, :], label = str(Q[Q_ind]))

    plt.xlabel('Time')
    plt.ylabel('Signal')



#NoQ = 5
#Qs = np.linspace(1,10,NoQ)
#
#Not = 100
#ts = np.linspace(0,50,Not)
#
#dS = np.random.rand(NoQ, Not)
##dS = np.array([Qs,Qs,Qs]).transpose()
#
#q1 = np.linspace(0,10.1,Not)
#q2 = np.linspace(2,12.2,Not)
#q3 = np.linspace(3,14.5,Not)
#q4 = np.linspace(4,16.2,Not)
#q5 = np.linspace(5,18.2,Not)
#
#S = np.array([q1,q2,q3,q4,q5])
#
#S = unumpy.uarray(S,dS)
#
##print S
#
#
#rhos = np.linspace(5,10,30)
#plot_S(S, Qs, ts)
#
#plt.figure()
#(eta_rho, eta_time)  = get_sens(S, Qs, ts, rhos)
#
#plt.figure()
#plt.subplot(2,1,1)
#plt.plot(ts, eta_rho)
#plt.xlabel('time')
#plt.ylabel('Sensitivity')
#plt.subplot(2,1,2)
#plt.plot(rhos, eta_time)
#plt.ylabel('Sensitivity')
#plt.xlabel('Signal')
#
#plt.show()
#
#

