# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:50:22 2024

@author: mlocarno
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution

# Global variables and parameters
global frequency, enh_mode, ideal_output, ranges
frequency = 5 #Hz
# Plotting settings
OkabeIto = np.array([[0, 0, 0], [0.9, 0.6, 0], [0.35, 0.7, 0.9], [0, 0.6, 0.5], [0.95, 0.9, 0.25], [0, 0.45, 0.7], [0.8, 0.4, 0], [0.8, 0.6, 0.7]])
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['font.size'] = 18

# Photocycle model (k values strictly positive)
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

# Square wave in voltage, from -70mV to 30mV
def input_signal(t, f=frequency):
    return 0.03 if np.sin(2 * np.pi * f * t) >= 0 else -0.07

# Exponential function for fitting
def exp_func(t, k, A, B):
    return A * np.exp(-k * t) + B

# Simulation of the model given the input parameters (transition rates and field enhancement)
def simulate(k1, k2m, k2q, k30, k3e, k4, k5, FE, Vstart=0.03, plot=False):
    # Initialization
    k2_init = k2m * Vstart + k2q
    x0 = [1 / (1 + k1 / k5 + k1 / k2_init), 1 / (1 + k2_init / k5 + k2_init / k1), 1 / (1 + k5 / k2_init + k5 / k1), 0.001]
    t_span = (0, 1)
    t_eval = np.arange(0, 1, 0.0001)
    
    # Fit returning the effective transition speed k
    def get_curve_fit_params(time, output, start_time, end_time):
        mask = (time >= start_time) & (time < end_time)
        t_fit = time[mask] - start_time
        y_fit = output[mask]
        p0 = [200, y_fit[-1] - y_fit[0], y_fit[-1]]
        popt, _ = curve_fit(exp_func, t_fit, y_fit, p0=p0)
        return popt[0]
    
    # Solution of the ODE
    def solve_ode(k3_val):
        sol = solve_ivp(model, t_span, x0, method='LSODA', args=(k1, k2m, k2q, k3_val, k4, k5, input_signal), max_step=0.01, t_eval=t_eval)
        return sol.t, sol.y[3]
    
    # Impose specific requirements based on which mode is used
    if enh_mode[0] != 'k':      # For models where k is not enhanced
        k3e = k30
    elif enh_mode[1] != 'Q':    # For models where Q is not enhanced
        FE = 1
        
    # Normal mode features calculation
    time, output = solve_ode(k30)
    Qh = max(output[np.where((time > 2.25 / frequency) & (time < 2.5 / frequency))])
    Ql = min(output[np.where((time > 2.75 / frequency) & (time < 3 / frequency))])
    k_up = get_curve_fit_params(time, output, 2 / frequency, 2.5 / frequency)
    k_down = get_curve_fit_params(time, output, 2.5 / frequency, 3 / frequency)
    Vsens = Qh / Ql - 1
    
    # Enhanced mode features calculation
    time, outputenh = solve_ode(k3e)
    Qh_enh = max(outputenh[np.where((time > 2.25 / frequency) & (time < 2.5 / frequency))])
    Ql_enh = min(outputenh[np.where((time > 2.75 / frequency) & (time < 3 / frequency))])
    k_up_enh = get_curve_fit_params(time, outputenh, 2 / frequency, 2.5 / frequency)
    k_down_enh = get_curve_fit_params(time, outputenh, 2.5 / frequency, 3 / frequency)
    Vsens_enh = Qh_enh / Ql_enh - 1
    outputenh *= FE
    
    # Plot results only when the optimization has ended
    if plot:
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=OkabeIto)
        plt.plot(time, output, '--', label='Normal')
        plt.plot(time, outputenh, label='Enhanced')
        plt.legend(loc='best',fontsize=16)
        plt.xlabel('Time [s]')
        plt.ylabel('Signal [a.u.]')
        if enh_mode=='k0':
            plt.title('N\u2192Q, Q*=1')
            plt.savefig('SimulatedTraces_PlasmEnh_NQ.png',dpi=300, bbox_inches='tight')
        elif enh_mode=='0Q':
            plt.title('Q*=1.69')
            plt.savefig('SimulatedTraces_PlasmEnh_Q.png',dpi=300, bbox_inches='tight') 
        elif enh_mode=='kQ':
            plt.title('N\u2192Q,    Q*')
            plt.savefig('SimulatedTraces_PlasmEnh_NQ_Q.png',dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
    
        print('\n\n')
        print('dF/F (normal): {:.2f} %'.format(Vsens*100))
        print('Rate k_up: {:.0f} 1/s'.format(k_up))
        print('Rate k_down: {:.0f} 1/s'.format(k_down))
        print('dF/F (enhanced): {:.2f} %'.format(Vsens_enh*100))
        print('Enhanced rate k_up: {:.0f} 1/s'.format(k_up_enh))
        print('Enhanced rate k_down: {:.0f} 1/s'.format(k_down_enh))
        print('\nChange in brightness (low level): {:.2f} %'.format((FE*Ql_enh / Ql - 1)*100))
        print('Change in sensitivity: {:.2f} %'.format((Vsens_enh-Vsens)/Vsens*100))
        print('Change in speed up: {:.2f} %'.format((k_up_enh-k_up)/k_up*100))
        print('Change in speed down: {:.2f} %'.format((k_down_enh-k_down)/k_down*100))
    return [FE*Ql_enh / Ql - 1, Vsens, Vsens_enh, k_up, k_down, k_up_enh, k_down_enh]

# Huber loss
def huber_loss(output):
    err = (ranges[:, 1] - ranges[:, 0]) / 2
    diff = ideal_output - output
    loss = np.where(np.abs(diff) <= err, 0.5 * err**2, err * (np.abs(diff) - 0.5 * err))
    return np.sum(loss/err**2) 

# Routine to optimize the parameters with differential evolution; the bounds are chosen such that k1>k2>k3>k4
def optimize_params():
    bounds = [
        (10000, 100000), 
        (1000, 300000), 
        (1000, 50000), 
        (100, 2000), 
        (100, 20000), 
        (1, 1000), 
        (1, 10000), 
        (0.5, 10)
    ]

    if enh_mode == '0Q':
        bounds[7]= (1.69,1.69)
    elif enh_mode == 'k0':
        bounds[7]= (1,1)

    def objective_function(x):
        output = simulate(*x, plot=False)
        loss = huber_loss(output)
        if enh_mode[0] == 'k':
            ratio = x[4] / x[3]
            FE = x[7]
            if ratio < 0.5 * FE or ratio > 2 * FE:
                loss += 1e6
        return loss

    result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=1000, popsize=15, tol=1e-6, disp=False)
    return result.x

if __name__ == "__main__":
    # Experimental values to match
    ideal_output = np.array([0.69, 0.38, 0.11, 170, 220, 1300, 930])
    ranges = np.array([(0.69 - 0.19, 0.69 + 0.19), (0.38 - 0.07, 0.38 + 0.07), (0.11 - 0.04, 0.11 + 0.04), (170 - 90, 170 + 90), (220 - 110, 220 + 110), (1300 - 400, 1300 + 400), (930 - 100, 930 + 100)])

    # Simulation of the model where only Q is enhanced
    enh_mode = '0Q'  
    QQ = optimize_params()
    best_output = simulate(*QQ, plot=True)
    loss = huber_loss(best_output)
    print(f'Huber loss {enh_mode} = {loss}')
    print('Best parameters:', QQ)

    # Simulation of the model where only k is enhanced
    enh_mode = 'k0'  
    kk = optimize_params()
    best_output = simulate(*kk, plot=True)
    loss = huber_loss(best_output)
    print(f'Huber loss {enh_mode} = {loss}')
    print('Best parameters:', kk)
    
    # Simulation of the model where both Q and k are enhanced
    enh_mode = 'kQ' 
    kQ = optimize_params()
    best_output = simulate(*kQ, plot=True)
    loss = huber_loss(best_output)
    print(f'Huber loss {enh_mode} = {loss}')
    print('Best parameters:', kQ)
