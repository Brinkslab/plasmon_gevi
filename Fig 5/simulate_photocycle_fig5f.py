# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:50:22 2024

@author: mlocarno
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

global frequency, enh_mode
frequency = 5

#CVD-friendly qualitatively color cycle
OkabeIto=np.array([[0,0,0],
                   [0.9,0.6,0],
                   [0.35,0.7,0.9],
                   [0,0.6,0.5],
                   [0.95,0.9,0.25],
                   [0,0.45,0.7],
                   [0.8,0.4,0],
                   [0.8,0.6,0.7]])

# Set the default line width and font size
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['font.size'] = 18

def model(t, x, k1,k2m,k2q,k3,k4,k5, V):
    k2 = k2m * V(t) + k2q

    if k1<0:
        k1=-k1
    if k2<0:
        k2=-k2
    if k3<0:
        k3=-k3
    if k4<0:
        k4=-k4
    if k5<0:
        k5=-k5
    
    dxdt = np.zeros(4)
    dxdt[0] = -k1*x[0] + k5*x[2]
    dxdt[1] = k1*x[0] - k2*x[1]
    dxdt[2] = k2*x[1] - (k3+k5)*x[2] + k4*x[3]
    dxdt[3] = k3*x[2] - k4*x[3]
    return dxdt

# Input i(t)
def input_signal(t,f=frequency):
    if np.sin(2*np.pi*f*t) >= 0:
        return 0.03
    else:
        return -0.07

# Define exponential function to fit
def exp_func(t, k, A, B):
    return A*np.exp(-k*t) + B

def simulate(k1,k2m,k2q,k30,k3e,k4,k5,FE, Vstart=0.03, plot=False,control=True,enhanced=True):
    k2_init=k2m*Vstart+k2q
    x0 = [1/(1+k1/k5+k1/k2_init),  1/(1+k2_init/k5+k2_init/k1),  1/(1+k5/k2_init+k5/k1), 0.001]
    
    # Time points
    t_span = (0, 1)
    t_eval = np.arange(0, 1, 0.0001)
    
    if control:
        # Solve ODE for different input signals
        sol = solve_ivp(model, t_span, x0, method='LSODA', args=(k1,k2m,k2q,k30,k4,k5, input_signal),max_step=0.01,t_eval=t_eval)
        
        time=sol.t
        output=sol.y[3]
        Qh=max(output[np.where((time>2.25/frequency) & (time<2.5/frequency))])
        Ql=min(output[np.where((time>2.75/frequency) & (time<3/frequency))])
            
        # Define range to fit upswing
        start_time = 2/frequency
        end_time = 2.5/frequency
        mask = (time >= start_time) & (time < end_time)
        t_fit = time[mask]-start_time
        y_fit = output[mask]
        
        # Perform curve fit
        p0 = [200, y_fit[-1]-y_fit[0], y_fit[-1]] # initial guess for k, A, B
        popt, pcov = curve_fit(exp_func, t_fit, y_fit, p0=p0)
        
        # Print rate k
        k_up = popt[0]
    
        # Define range to fit downswing
        start_time = 2.5/frequency
        end_time = 3/frequency
        mask = (time >= start_time) & (time < end_time)
        t_fit = time[mask]-start_time
        y_fit = output[mask]
        
        # Perform curve fit
        p0 = [200, y_fit[-1]-y_fit[0], y_fit[-1]] # initial guess for k, A, B
        popt, pcov = curve_fit(exp_func, t_fit, y_fit, p0=p0)
        
        # Print rate k
        k_down = popt[0]
        Vsens=Qh/Ql-1

    #SIMULATION ENHANCED SIGNAL
    if enhanced:
        if enh_mode[0] == 'k':  
            sol = solve_ivp(model, t_span, x0, method='LSODA', args=(k1,k2m,k2q,k3e,k4,k5, input_signal),max_step=0.01,t_eval=t_eval)
        else:
            sol = solve_ivp(model, t_span, x0, method='LSODA', args=(k1,k2m,k2q,k30,k4,k5, input_signal),max_step=0.01,t_eval=t_eval)

        time=sol.t
        outputenh=sol.y[3]
        
        Qh_enh=max(outputenh[np.where((time>2.25/frequency) & (time<2.5/frequency))])
        Ql_enh=min(outputenh[np.where((time>2.75/frequency) & (time<3/frequency))])
        
        # Define range to fit
        start_time = 2/frequency
        end_time = 2.5/frequency
        mask = (time >= start_time) & (time < end_time)
        t_fit = time[mask]-start_time
        y_fit = outputenh[mask]
        
        # Perform curve fit
        p0 = [200, y_fit[-1]-y_fit[0], y_fit[-1]] # initial guess for k, A, B
        popt, pcov = curve_fit(exp_func, t_fit, y_fit, p0=p0)
        k_up_enh = popt[0]
        
        # Define range to fit
        start_time = 2.5/frequency
        end_time = 3/frequency
        mask = (time >= start_time) & (time < end_time)
        t_fit = time[mask]-start_time
        y_fit = outputenh[mask]
        
        # Perform curve fit
        p0 = [200, y_fit[-1]-y_fit[0], y_fit[-1]] # initial guess for k, A, B
        popt, pcov = curve_fit(exp_func, t_fit, y_fit, p0=p0)
        k_down_enh = popt[0]
        Vsens_enh=Qh_enh/Ql_enh-1
        
        if enh_mode[1] == 'Q':
            outputenh=outputenh*FE
        
    if control and enhanced:
        if enh_mode[1] == 'Q': 
            bright_enh=FE*Ql_enh/Ql-1
        else:
            bright_enh=Ql_enh/Ql-1

    if plot:
        # Plot results
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=OkabeIto)
        if control:
            plt.plot(time, output, '--', label='Normal')
        if enhanced:
            plt.plot(time, outputenh, label='Enhanced')
        plt.legend(loc='best',fontsize=16)
        plt.xlabel('Time [s]')
        plt.ylabel('Signal [a.u.]')
        if enh_mode=='k0':
            plt.title('N\u2192Q, Q*=1')
            plt.savefig('SimulatedTraces_PlasmEnh_NQ.png',dpi=300, bbox_inches='tight')
        elif enh_mode=='k1':
            plt.title('N\u2192Q, Q*=1')
            plt.savefig('SimulatedTraces_PlasmEnh_NQ1.png',dpi=300, bbox_inches='tight')
        elif enh_mode=='kQ':
            plt.title('N\u2192Q,    Q*')
            plt.savefig('SimulatedTraces_PlasmEnh_NQ_Q.png',dpi=300, bbox_inches='tight')
        elif enh_mode=='kQ1':
            plt.title('N\u2192Q,    Q*')
            plt.savefig('SimulatedTraces_PlasmEnh_NQ_Q1.png',dpi=300, bbox_inches='tight')
        elif enh_mode=='kQ2':
            plt.title('N\u2192Q,    Q*=1.69')
            plt.savefig('SimulatedTraces_PlasmEnh_NQ_Q2.png',dpi=300, bbox_inches='tight')
        elif enh_mode=='0Q':
            plt.title('Q*=1.69')
            plt.savefig('SimulatedTraces_PlasmEnh_Q.png',dpi=300, bbox_inches='tight') 
        plt.tight_layout()
        plt.show()

        print('\n\n')
        if control:
            print('dF/F (normal): {:.2f} %'.format(Vsens*100))
            print('Rate k_up: {:.0f} 1/s'.format(k_up))
            print('Rate k_down: {:.0f} 1/s'.format(k_down))
        if enhanced:
            print('dF/F (enhanced): {:.2f} %'.format(Vsens_enh*100))
            print('Enhanced rate k_up: {:.0f} 1/s'.format(k_up_enh))
            print('Enhanced rate k_down: {:.0f} 1/s'.format(k_down_enh))
        if control and enhanced:
            print('\nChange in brightness (low level): {:.2f} %'.format(bright_enh*100))
            print('Change in sensitivity: {:.2f} %'.format((Vsens_enh-Vsens)/Vsens*100))
            print('Change in speed up: {:.2f} %'.format((k_up_enh-k_up)/k_up*100))
            print('Change in speed down: {:.2f} %'.format((k_down_enh-k_down)/k_down*100))
    
    if control and enhanced:
        return bright_enh, Vsens, Vsens_enh, k_up, k_down, k_up_enh, k_down_enh
    if control and (not enhanced):
        return Vsens, k_up, k_down
    if enhanced and (not control):
        return Vsens_enh, k_up_enh, k_down_enh

def custom_loss(output, ideal_output, ideal_ranges):
    # Calculate squared relative differences
    relative_diff = np.square((output - ideal_output) / (ideal_output + 1e-6))

    # Calculate penalties for deviations from the desired ranges
    range_penalties = 0
    for i, (lower, upper) in enumerate(ideal_ranges):
        if lower is not None:
            range_penalties += np.maximum(0, (output[i] - upper) / (upper - lower))**2
        if upper is not None:
            range_penalties += np.maximum(0, (lower - output[i]) / (upper - lower))**2

    # Combine relative differences and range penalties
    combined_loss = np.mean(relative_diff) + 0.2 * np.mean(range_penalties)  # You can adjust the weight (0.1) based on your preference

    if np.any(np.array(output) < 0):
        return 1e6
    return combined_loss

def optimize_gaussian(control=True,enhanced=True):
    # Initial condition
    # N->Q, Q*
    if enh_mode=='kQ':
        k1 = 27947
        k2m, k2q = 40765, 4741
        k30 = 24
        k3e = 1131
        k4 = 164
        k5 = 1745
        FE = 0.15 #field enhancement
        sigmas=np.array([k1/4,k2m/4,k2q/4,k30/4,k3e/4,k4/4,k5/4,FE/4])
    #N->Q, Q*=1
    elif enh_mode=='k0':
        k1 = 15349
        k2m, k2q = 33068, 3716
        k30 = 283
        k3e = 1857
        k4 = 72
        k5 = 4380
        FE = 1 #field enhancement
        sigmas=np.array([k1/4,k2m/4,k2q/4,k30/4,k3e/4,k4/4,k5/4,0])
    #Q*=1.69
    elif enh_mode=='0Q':
        k1 = 1206
        k2m, k2q = 220, 106
        k30 = 18
        k3e = 18
        k4 = 231
        k5 = 345
        FE = 1.69 #field enhancement
        sigmas=np.array([k1/4,k2m/4,k2q/4,k30/4,k3e/4,k4/4,k5/4,0])
       
    
    x=np.array([k1,k2m,k2q,k30,k3e,k4,k5,FE])

    best_params=x
    best_output=simulate(*x,plot=False,control=control,enhanced=enhanced)
    if control and enhanced:
        ideal_output=np.array([0.69,0.38,0.11,143,167,1000,1000])
        ranges=[(0.69-0.19, 0.69+0.19), (0.38-0.07, 0.38+0.07), (0.11-0.04, 0.11+0.04), (111, 200), (125, 250), (None, None), (None, None)]
    if control and (not enhanced):
        ideal_output=np.array([0.38,143,167])
        ranges=[(0.38-0.07, 0.38+0.07), (111, 200), (125, 250)]
    if enhanced and (not control):
        ideal_output=np.array([0.11,1000,1000])
        ranges=[(0.11-0.04, 0.11+0.04), (None, None), (None, None)]

    best_loss=custom_loss(best_output, ideal_output, ranges)

    for j in range(0):
        improvement=False
        for i in range(2000):
            try:
                # Create an array of random Gaussian values with the specified sigmas
                rand_vals = np.random.normal(0, sigmas, size=len(sigmas))
                params = x+rand_vals
                output=simulate(*params,plot=False,control=control,enhanced=enhanced)
                loss = custom_loss(output, ideal_output, ranges)
            except:
                print("Optimization failed. Skipping to next iteration.")
                params=x
                loss=1e6
            if loss<best_loss:
                best_params=params
                best_output=output
                best_loss=custom_loss(best_output, ideal_output, ranges)
                improvement=True

        simulate(*best_params,plot=False,control=control,enhanced=enhanced)
        x=best_params
        if (not improvement):
            print('Plateau reached\n-------------------------------------------')
            break
    return x

if __name__ == "__main__":
    ideal_output=np.array([0.69,0.38,0.11,143,167,1000,1000])
    ranges=np.array([(0.69-0.19, 0.69+0.19), (0.38-0.07, 0.38+0.07), (0.11-0.04, 0.11+0.04), (111, 200), (125, 250), (500, 2000), (500, 2000)])
    err=(ranges[:,1]-ranges[:,0])/2

    enh_mode='0Q'
    x0Q=optimize_gaussian()
    best_output=simulate(*x0Q,plot=True)
    chi_squared=np.sum((ideal_output-best_output)**2/err**2)
    print('Chi-squared '+enh_mode+' = '+format(chi_squared))
    
    enh_mode='k0'
    xk0=optimize_gaussian()
    best_output=simulate(*xk0,plot=True)
    chi_squared=np.sum((ideal_output-best_output)**2/err**2)
    print('Chi-squared '+enh_mode+' = '+format(chi_squared))
    
    enh_mode='kQ'
    xkQ=optimize_gaussian()
    best_output=simulate(*xkQ,plot=True)
    chi_squared=np.sum((ideal_output-best_output)**2/err**2)
    print('Chi-squared '+enh_mode+' = '+format(chi_squared))
    
    
