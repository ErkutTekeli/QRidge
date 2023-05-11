#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:20:35 2023

@author: erkuttekeli
"""

       
### RUN for Simulation######
### > sim(100)
###
### RUN for Diabetes dataset######
### > diabetes()
###
### RUN for watermelon dataset######
### > watermelon()


import numpy as np
from numpy.linalg import eig
from numpy.linalg import inv
#import time
import math
import csv
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

def kappa(kor):
    w,_ = eig(kor)
    if max(w)==0:
        kp = 0
    else:
        kp = np.sqrt(max(w)/min(w))
    return kp

def vif(kort):
    return max(np.diagonal(kort))
    

def artificial_data(beta, rho, sigma, n, p1, p2):
    p = p1+p2
    omega = np.zeros((n,p))
    for i in range(p):
        omega[:,i] = np.random.normal(size=(n,))
    omega_cor = np.random.normal(size=(n,)) 
    adata = omega
    for i in range(p1):
       #adata[:,i] = np.sqrt(1-rho*rho)*omega[:,i] + rho*omega_cor
       adata[:,i] = (1-rho)*omega[:,i] + rho*omega_cor
    err = np.random.normal(scale=sigma, size=(n,))
    y = np.matmul(adata, np.transpose(beta)) + err
    return( adata, y )
        

#from Qsun.Qcircuit import *
#from Qsun.Qgates import *

# QWave
class Wavefunction(object):
    """a wavefunction representing a quantum state"""

    def __init__(self, states, amplitude_vector):
        self.state = states
        self.amplitude = amplitude_vector
        self.visual = []
    
    def probabilities(self):
        """returns a dictionary of associated probabilities."""
        return np.abs(self.amplitude) ** 2

    def print_state(self):
        """represent a quantum state in bra-ket notations"""
        states = self.state
        string = str(self.amplitude[0]) + '|' + states[0] + '>'
        for i in range(1, len(states)):
            string += ' + ' + str(self.amplitude[i]) + '|' + states[i] + '>'
        return string
    
    def visual_circuit(self):
        """Visualization of a ciruict"""
        n = len((self.state)[0])
        a = self.visual
        b = [[]]*(2*n)
        for i in range(2*n):
            b[i] = [0]*len(a)

        for i in range(n):
            for j in range(len(a)):
                if i in a[j]:    
                    if ('RX' in a[j]) or ('RY' in a[j]) or ('RZ' in a[j]):
                        b[2*i][j] = 1.5
                    elif ('CRX' in a[j]) or ('CRY' in a[j]) or ('CRZ' in a[j]):
                        b[2*i][j] = 2.5
                    elif ('CX' in a[j]) or ('SWAP' in a[j]):
                        b[2*i][j] = 3
                    elif ('CP' in a[j]):
                        b[2*i][j] = 3.5
                    elif ('CCX' in a[j]):
                        b[2*i][j] = 4
                    else:
                        b[2*i][j] = 1

        for j in range(len(a)):
            if ('CX' in a[j]) or ('CCX' in a[j]) or ('SWAP' in a[j]):
                for i in range(2*min(a[j][:-1])+1, 2*max(a[j][:-1]), 2):
                    b[i][j] = 2
            if ('CP' in a[j]) or ('CRX' in a[j]):
                for i in range(2*min(a[j][:-2])+1, 2*max(a[j][:-2]), 2):
                    b[i][j] = 2

        string_out = [[]]*(2*n)
        for i in range(2*n):
            string_out[i] = []

        for i in range(n):
            out = ''
            if i < 10:
                out += '|Q_'+str(i)+'> : '
            else:
                out += '|Q_'+str(i)+'>: '
            space = ' '*len(out)
            string_out[2*i].append(out)
            string_out[2*i+1].append(space)

            out = ''
            space = ''
            for j in range(len(a)):

                if b[2*i][j] == 0:
                    out += '---'

                if b[2*i][j] == 1:
                    out += a[j][-1] + '--'

                if b[2*i][j] == 1.5:
                    out += a[j][-2] + '-'

                if b[2*i][j] == 2.5:
                    if i == a[j][0]:
                        out += 'o--'
                    elif i == a[j][1]:
                        out += a[j][-2][1:] + '-'

                if b[2*i][j] == 3:
                    if i == a[j][0]:
                        out += 'o--'
                    elif i == a[j][1]:
                        out += 'x--'

                if b[2*i][j] == 3.5:
                    if i == a[j][0]:
                        out += 'o--'
                    elif i == a[j][1]:
                        out += a[j][-2][1] + '--'

                if b[2*i][j] == 4:
                    if i == a[j][0] or i == a[j][1]:
                        out += 'o--'
                    elif i == a[j][2]:
                        out += 'x--'


                if b[2*i+1][j] == 2:
                    space += '|  '
                if b[2*i+1][j] == 0:
                    space += '   '

            string_out[2*i].append(out+'-M')
            string_out[2*i+1].append(space+'  ')

        for i in string_out:
            print(i[0]+i[1])

# Qcircuit
import itertools
#import numpy as np

# https://stackoverflow.com/questions/4928297/all-permutations-of-a-binary-sequence-x-bits-long
def Qubit(qubit_num):
    """create a quantum circuit"""
    states = ["".join(seq) for seq in itertools.product("01", repeat=qubit_num)]
    amplitude_vector = np.zeros(2**qubit_num, dtype = complex)
    amplitude_vector[0] = 1.0
    return Wavefunction(np.array(states), amplitude_vector)

def Walk_Qubit(qubit_num=1, dim=1):
    """create a initial quantum state for hadamard coin"""
    if dim != 1 and dim != 2:
        raise TypeError('The dimension of the quantum walk must be 1 or 2')
    else:
        qubit_num += 1
        if dim == 1:
            #initial state: (|0> - i|1>)x|n=0>/(sqrt(2))
            states = ['0' + str(i) for i in range(2*qubit_num-1)]
            states += ['1' + str(i) for i in range(2*qubit_num-1)]
        
            amplitude_vector = np.zeros(4*qubit_num-2, dtype = complex)
            amplitude_vector[qubit_num-1] = 2**-0.5
            amplitude_vector[3*qubit_num-2] = (-2)**-0.5
            return Wavefunction(np.array(states), amplitude_vector)
        else:
            #initial state: ((|0> + i|1>)/sqrt(2))x((|0> + i|1>)/sqrt(2))x|n=0>x|n=0>
            states = ['0' + str(i) for i in range(0, (2*qubit_num-1)**2)]
            states += ['1' + str(i) for i in range(0, (2*qubit_num-1)**2)]
            states += ['2' + str(i) for i in range(0, (2*qubit_num-1)**2)]
            states += ['3' + str(i) for i in range(0, (2*qubit_num-1)**2)]
            
            amplitude_vector = np.zeros(4*(2*qubit_num-1)**2, dtype = complex)
            index = int(((2*qubit_num-1)**2-1)/2)
            amplitude_vector[index] = 1/2
            amplitude_vector[index+(2*qubit_num-1)**2] = 0.5j
            amplitude_vector[index+2*(2*qubit_num-1)**2] = 0.5j
            amplitude_vector[index+3*(2*qubit_num-1)**2] = -1/2
            return Wavefunction(np.array(states), amplitude_vector)

# QGates
#import numpy as np
import cmath

def H(wavefunction, n):
    """Hadamard gate: math:`\frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}`"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i] += amplitude[i]/2**0.5
            new_amplitude[i+cut] += amplitude[i]/2**0.5
        else:
            new_amplitude[i] -= amplitude[i]/2**0.5
            new_amplitude[i-cut] += amplitude[i]/2**0.5  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'H'])
    
def X(wavefunction, n):
    """Pauli-X: math:`\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}`"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i+cut] += amplitude[i]
        else:
            new_amplitude[i-cut] += amplitude[i]  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'X'])
    
def Y(wavefunction, n):
    """Pauli-Y: math:`\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}`"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i+cut] += 1.0j*amplitude[i]
        else:
            new_amplitude[i-cut] -= 1.0j*amplitude[i]  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'Y'])
    
def Z(wavefunction, n):
    """Pauli-Z: math:`\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}`"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i] += amplitude[i]
        else:
            new_amplitude[i] -= amplitude[i]  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'Z'])
    
def RX(wavefunction, n, phi=0):
    """PHASE gate: math:`\begin{pmatrix} cos(phi/2) & -sin(phi/2) \\ sin(phi/2) & cos(phi/2) \end{pmatrix}`"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states)
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for idx, i in enumerate(np.nonzero(amplitude)[0]):
        if states[idx] == '0':
            new_amplitude[i] += cmath.cos(phi/2)*amplitude[i]
            new_amplitude[i+cut] -= 1j*cmath.sin(phi/2)*amplitude[i]
        else:
            new_amplitude[i] += cmath.cos(phi/2)*amplitude[i]
            new_amplitude[i-cut] -= 1j*cmath.sin(phi/2)*amplitude[i] 
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'RX', '0'])
    
def RY(wavefunction, n, phi=0):
    """PHASE gate: math:`\begin{pmatrix} cos(phi/2) & -sin(phi/2) \\ sin(phi/2) & cos(phi/2) \end{pmatrix}`"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states)
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for idx, i in enumerate(np.nonzero(amplitude)[0]):
        if states[idx] == '0':
            new_amplitude[i] += cmath.cos(phi/2)*amplitude[i]
            new_amplitude[i+cut] += cmath.sin(phi/2)*amplitude[i]
        else:
            new_amplitude[i] += cmath.cos(phi/2)*amplitude[i]
            new_amplitude[i-cut] -= cmath.sin(phi/2)*amplitude[i] 
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'RY', '0'])

def RZ(wavefunction, n, phi=0):
    """PHASE gate: math:`\begin{pmatrix} 1 & 0 \\ 0 & e^{i \phi} \end{pmatrix}`"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i] += cmath.exp(-1j*phi/2)*amplitude[i]
        else:
            new_amplitude[i] += cmath.exp(1j*phi/2)*amplitude[i]  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'RZ', '0'])

def Phase(wavefunction, n, phi=0):
    """PHASE gate: math:`\begin{pmatrix} 1 & 0 \\ 0 & e^{i \phi} \end{pmatrix}`"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i] += amplitude[i]
        else:
            new_amplitude[i] += cmath.exp(1j*phi)*amplitude[i]  
    wavefunction.amplitude = new_amplitude
#     (wavefunction.visual).append([n, 'P', phi])
    
def S(wavefunction, n):
    """Phase(pi/2): math:`\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}`"""
    Phase(wavefunction, n , cmath.pi/2)
    (wavefunction.visual).append([n, 'S'])
    
def T(wavefunction, n):
    """Phase(pi/4): math:`\begin{pmatrix} 1 & 0 \\ 0 & e^{i \pi / 4} \end{pmatrix}`"""
    Phase(wavefunction, n , cmath.pi/4)
    (wavefunction.visual).append([n, 'T'])
    
def Xsquare(wavefunction, n):
    """a square root of the NOT gate."""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        new_amplitude[i] += (1+1j)*amplitude[i]/2
        if states[i][n] == '0':
            new_amplitude[i+cut] += (1-1j)*amplitude[i]/2
        else:
            new_amplitude[i-cut] += (1-1j)*amplitude[i]/2  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'XS'])
    
def CNOT(wavefunction, control, target):
    """Flip target if control is |1>: 
    math:`P_0 \otimes I + P_1 \otimes X = \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\
                                            0&0&0&1 \\ 0&0&1&0 \end{pmatrix}`"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    if control < target or control > target:
        cut = 2**(qubit_num-target-1)
    else:
        raise TypeError("Control qubit and target qubit must be distinct")
    for i in np.nonzero(amplitude)[0]:
        if states[i][control] == '1':
            if states[i][target] == '0':
                new_amplitude[i+cut] += amplitude[i]
            else:
                new_amplitude[i-cut] += amplitude[i]
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([control, target, 'CX'])
    
def CPhase(wavefunction, control, target, phi=0):
    """Controlled PHASE gate: math:`\text{diag}(1, 1, 1, e^{i \phi})`"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    if control == target:
        raise TypeError("Control qubit and target qubit must be distinct")
    for i in np.nonzero(amplitude)[0]:
        if states[i][control] == '1':
            if states[i][target] == '0':
                new_amplitude[i] += amplitude[i]
            else:
                new_amplitude[i] += cmath.exp(1j*phi)*amplitude[i] 
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([control, target, 'CP', '0'])
    
def CCNOT(wavefunction, control_1, control_2, target):
    """CCNOT - double-controlled-X
    :math:`P_0 \otimes P_0 \otimes I + P_0 \otimes P_1 \otimes I + P_1 \otimes P_0 \otimes I
                                     + P_1 \otimes P_1 \otimes X`"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-target-1)
    if control_1 == target or control_2 == target or control_1 == control_2:
        raise TypeError("Control qubit and target qubit must be distinct")
    for i in np.nonzero(amplitude)[0]:
        if states[i][control_1] == '1' and states[i][control_2] == '1':
            if states[i][target] == '0':
                new_amplitude[i+cut] += amplitude[i]
            else:
                new_amplitude[i-cut] += amplitude[i]
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([control_1, control_2, target, 'CCX'])
    
def OR(wavefunction, control_1, control_2, target):
    """CCNOT - double-controlled-X
    :math:`P_0 \otimes P_0 \otimes I + P_0 \otimes P_1 \otimes I + P_1 \otimes P_0 \otimes I
                                     + P_1 \otimes P_1 \otimes X`"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-target-1)
    if control_1 == target or control_2 == target or control_1 == control_2:
        raise TypeError("Control qubit and target qubit must be distinct")
    for i in np.nonzero(amplitude)[0]:
        if states[i][control_1] == '1' or states[i][control_2] == '1':
            if states[i][target] == '0':
                new_amplitude[i+cut] += amplitude[i]
            else:
                new_amplitude[i-cut] += amplitude[i]
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    
def SWAP(wavefunction, target_1, target_2):
    """Swap gate: math:`\begin{pmatrix} 1&0&0&0 \\ 0&0&1&0 \\ 0&1&0&0 \\ 0&0&0&1 \end{pmatrix}`"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    minimum = target_2 ^ ((target_1 ^ target_2) & -(target_1 < target_2))
    maximum = target_1 ^ ((target_1 ^ target_2) & -(target_1 < target_2)) 
    cut = 2**(qubit_num-minimum-1) - 2**(qubit_num-maximum-1)
    if target_1 == target_2:
        raise TypeError("Target qubits must be distinct")
    for i in range(2**qubit_num):
        if states[i][target_1] != states[i][target_2]:
            if int(states[i][maximum]) > int(states[i][minimum]):
                new_amplitude[i+cut] += amplitude[i]                              
            else:
                new_amplitude[i-cut] += amplitude[i]
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([target_1, target_2, 'SWAP'])
    
def E(wavefunction, p, n):
    """Quantum depolarizing channel"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = np.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in np.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i+cut] += (p/2)*abs(amplitude[i])**2
            new_amplitude[i] += (1-p/2)*abs(amplitude[i])**2
        else:
            new_amplitude[i-cut] += (p/2)*abs(amplitude[i])**2
            new_amplitude[i] += (1-p/2)*abs(amplitude[i])**2
    #     wavefunction.wave.iloc[0, :] = np.sqrt(new_amplitude)
    for i in range(2**qubit_num):
        if amplitude[i] < 0:
            new_amplitude[i] = - np.sqrt(new_amplitude[i])
        else:
            new_amplitude[i] = np.sqrt(new_amplitude[i])
    wavefunction.amplitude = new_amplitude

def E_all(wavefunction, p_noise, qubit_num):
    if p_noise > 0:
        for i in range(qubit_num):
            E(wavefunction, p_noise, i)
            
# Continue QWM
def circuit(params):
    c = Qubit(1)
    RX(c, 0, params[0])
    RY(c, 0, params[1])
    return c

def output(params):
    c = circuit(params)
    prob = c.probabilities()
    return 1*prob[0] - 1*prob[1]

def outputM(params):
    ou = np.zeros(nc)
    for j in range(nc):
        ou[j] = output(params[j,:])
    return ou


def predict(x_true, coef_params):
    return np.matmul( x_true, coef_params )

def errors(x_true, y_true, coef_params, intercept_params, boundary):
    return mean_squared_error(y_true, predict(x_true, coef_params, intercept_params, boundary))

def grad(x_true, y_true, k_params, shift, eta, boundary, output_k, output_coef, output_kor, output_pred, kp ):
    
    k_diff = np.zeros((2,))
  
    for i in range(2):
            k_params_1 = k_params.copy()
            k_params_2 = k_params.copy()
            k_params_1[i] += shift
            k_params_2[i] -= shift
            output_1 = output(k_params_1)
            output_2 = output(k_params_2)
            if kp > 10:
                drc = 1
            else:
                drc = -1
            k_diff[i] -= drc*(2*np.matmul(np.transpose(np.matmul(np.matmul(x_true,inv(output_kor)),output_coef)), (y_true-output_pred))+np.matmul(np.transpose(output_coef),output_coef)-2*output_k*np.matmul(np.matmul(np.transpose(output_coef),inv(output_kor)),output_coef))*(output_1-output_2)/(2*np.sin(shift))
                        
            #for x, y in zip(x_true, y_true):
            #    coef_diff[j,i] -= (x*(y-predict(x, output_coef, output_intercept, boundary))*(output_1-output_2)/(2*np.sin(shift)))[0]
                         
    k_diff = k_diff*boundary*2/len(y_true)
            
    for i in range(len(k_params)):
        k_params[i] = k_params[i] - eta*k_diff[i]
        
    return k_params           


def OLS(xTrain, yTrain, xTest, yTest):
    # Create linear regression object
    regr = linear_model.LinearRegression(fit_intercept=False)

    # Train the model using the training sets
    regr.fit(xTrain, yTrain)

    yt_pred = regr.predict(xTrain)

    # Make predictions using the testing set
    y_pred = regr.predict(xTest)

    mseTrain = mean_squared_error(yTrain, yt_pred)
    mseTest = mean_squared_error(yTest, y_pred)
    kor = np.matmul(np.transpose(xTrain), xTrain)
    VIF = vif(inv(kor))
    KAPPA = kappa(kor)
    return 0, regr.coef_, regr.intercept_, mseTrain, mseTest, VIF, KAPPA


def khk(beta, sigSqrHat):
    return (sigSqrHat/np.matmul( beta, np.transpose(beta)))    

def khkb(p, beta, sigSqrHat):
    return (p*sigSqrHat/np.matmul( beta, np.transpose(beta)))    
    
def klw(p, beta, Lambda, sigSqrHat):
    return (p*sigSqrHat/np.matmul( Lambda*beta, np.transpose(beta)))    

def ridge_hk(beta, sigSqrHat, xTrain, yTrain, xTest, yTest):
    nr, nc = xTrain.shape
    k = khk(beta, sigSqrHat)
    kor = np.matmul(np.transpose(xTrain), xTrain) + k*np.eye(nc)
    coef = np.matmul(np.matmul(inv(kor),np.transpose(xTrain)),yTrain)
    ytrain_pred_qvm = predict(xTrain, coef)
    ytest_pred_qvm = predict(xTest, coef)

    mseTrain=mean_squared_error(yTrain, ytrain_pred_qvm)
    mseTest=mean_squared_error(yTest, ytest_pred_qvm)
    VIF=vif(inv(kor))
    KAPPA=kappa(kor)
    return k, coef, mseTrain, mseTest, VIF, KAPPA

def ridge_hkb(beta, sigSqrHat, xTrain, yTrain, xTest, yTest):
    nr, nc = xTrain.shape
    k = khkb(nc, beta, sigSqrHat)
    kor = np.matmul(np.transpose(xTrain), xTrain) + k*np.eye(nc)
    coef = np.matmul(np.matmul(inv(kor),np.transpose(xTrain)),yTrain)
    ytrain_pred_qvm = predict(xTrain, coef)
    ytest_pred_qvm = predict(xTest, coef)

    mseTrain=mean_squared_error(yTrain, ytrain_pred_qvm)
    mseTest=mean_squared_error(yTest, ytest_pred_qvm)
    VIF=vif(inv(kor))
    KAPPA=kappa(kor)
    return k, coef, mseTrain, mseTest, VIF, KAPPA

def ridge_lw(beta, Lambda, sigSqrHat, xTrain, yTrain, xTest, yTest):
    nr, nc = xTrain.shape
    k = klw(nc, beta, Lambda, sigSqrHat)
    kor = np.matmul(np.transpose(xTrain), xTrain) + k*np.eye(nc)
    coef = np.matmul(np.matmul(inv(kor),np.transpose(xTrain)),yTrain)
    ytrain_pred_qvm = predict(xTrain, coef)
    ytest_pred_qvm = predict(xTest, coef)

    mseTrain=mean_squared_error(yTrain, ytrain_pred_qvm)
    mseTest=mean_squared_error(yTest, ytest_pred_qvm)
    VIF=vif(inv(kor))
    KAPPA=kappa(kor)
    return k, coef, mseTrain, mseTest, VIF, KAPPA

def QRidge(xTrain, yTrain, xTest, yTest, rSeed=123, boun=2, verbose=True):
    np.random.seed(rSeed)
    nr, nc = xTrain.shape
    steps = 10000
    eta1 = 0.00001

    k_params = np.random.uniform(0,1,size=(2,))
    output_k = output(k_params)*boun
    kor = np.matmul(np.transpose(xTrain), xTrain) + output_k*np.eye(nc)
    coef = np.matmul(np.matmul(kor,np.transpose(xTrain)), yTrain)
    KAPPA=kappa(kor)

    ytrain_pred_qvm = predict(xTrain, coef)
    ytest_pred_qvm = predict(xTest, coef)

    rep = 0
    for i in range(steps):
        k_params = grad(xTrain, yTrain, k_params, np.pi/20, eta=eta1, boundary=boun, output_k=output_k, output_coef=coef, output_kor=kor, output_pred = ytrain_pred_qvm, kp=KAPPA)
        old_k = output_k
        output_k = output(k_params)*boun
        kor = np.matmul(np.transpose(xTrain), xTrain) + output_k*np.eye(nc)
        coef = np.matmul(np.matmul(inv(kor),np.transpose(xTrain)),yTrain)
        ytrain_pred_qvm = predict(xTrain, coef)
        ytest_pred_qvm = predict(xTest, coef)

        mseTrain=mean_squared_error(yTrain, ytrain_pred_qvm)
        mseTest=mean_squared_error(yTest, ytest_pred_qvm)
        VIF=vif(inv(kor))
        #old_kappa = KAPPA
        KAPPA=kappa(kor)
        if verbose and (i % 50 == 0):
            print("i= %4d  , k=%.5f   , MSE train= %.5f  , MSE test= %.5f, kappa= %.5f   , VIF= %.5f" % (i , output_k, mseTrain , mseTest, KAPPA, VIF))
        if old_k == output_k: 
            rep = rep+1
        else:
            rep = 0
        if rep > 5:
            break
        # if (i>0) and ((10-old_kappa)*(10-KAPPA)<=0):
        #     break

    return output_k, coef, mseTrain, mseTest, VIF, KAPPA

#simParam_n = np.array( [100, 500, 1000] )
#simParam_rho = np.array( [0.7, 0.8, 0.9, 0.95 ])
#simParam_p = np.array( [(2,1), (3,2), (3,4)])
#simParam_sigma = np.array( [5, 1, 0.5, 0.1])

simParam_n = np.array( [100, 500, 5000] )
simParam_rho = np.array( [0.7, 0.8, 0.9, 0.95, 0.99])
simParam_p = np.array( [(3,2)])
simParam_sigma = np.array( [2, 1, 0.5, 0.2])
simBoun = np.array( [1,2,3,20,50])

def sim(cs):
   ncase = len(simParam_n)*len(simParam_rho)* len(simParam_p)*len(simParam_sigma)
   simR = np.zeros((cs , 11 ))
   simCases = np.zeros((ncase , 9 ))
   j=0
   for n in simParam_n:
       for p in simParam_p:
           for ind,rho in enumerate(simParam_rho):
               for sigma in simParam_sigma:
                   p1 = p[0]
                   p2 = p[1]
                   #beta = [1] * (p1+p2)
                   beta = [0.1,0.5,-0.3,-0.1,0.8]
                   for i in range(cs):
                       x,y = artificial_data(beta, rho, sigma, n, p1, p2)
                    
                       nt = math.ceil(n*0.8)
                       xTrain = x[:nt,:]
                       yTrain = y[:nt]
                       xTest = x[nt:,:]
                       yTest = y[nt:]

                       kor = np.matmul(np.transpose(xTrain), xTrain)
                       Lambda,_ = eig(inv(kor))
                       e = yTrain - np.matmul(xTrain * np.matmul(Lambda, np.transpose(Lambda)),beta)
                       sigSqrHat = np.matmul(e, np.transpose(e))/(nt-(p1+p2))

                       kOLS,coef,intercept,mseTrain,mseTest,vif,kappa = OLS(xTrain, yTrain, xTest, yTest)
                       diffBeta = coef - beta    
                       mseBetaOLS = np.matmul( diffBeta, np.transpose(diffBeta))                         

                       kHK,coef,mseTrain,mseTest,vif,kappa = ridge_hk(beta, sigSqrHat, xTrain, yTrain, xTest, yTest)
                       diffBeta = coef - beta    
                       mseBetaKHK = np.matmul( diffBeta, np.transpose(diffBeta))                         

                       kHKB,coef,mseTrain,mseTest,vif,kappa = ridge_hkb(beta, sigSqrHat, xTrain, yTrain, xTest, yTest)
                       diffBeta = coef - beta    
                       mseBetaKHKB = np.matmul( diffBeta, np.transpose(diffBeta))                         

                       kLW,coef,mseTrain,mseTest,vif,kappa = ridge_lw(beta, Lambda, sigSqrHat, xTrain, yTrain, xTest, yTest)
                       diffBeta = coef - beta    
                       mseBetaKLW = np.matmul( diffBeta, np.transpose(diffBeta))                         

                       kQR,coef,mseTrain,mseTest,vif,kappa = QRidge(xTrain, yTrain, xTest, yTest, rSeed=123+i, boun=simBoun[ind], verbose=False)
                       diffBeta = coef - beta    
                       mseBeta = np.matmul( diffBeta, np.transpose(diffBeta))                         
                       simR[i] = np.array([i,n,rho,sigma,kQR,mseBetaOLS, mseBetaKHK,mseBetaKHKB,mseBetaKLW, mseBeta,kappa])
                       print("%4d, n= %4d, rho= %.2f, sigma= %.1f, kHK= %.5f, kQR= %.5f, mseBetaOLS= %.5f, mseBeta= %.5f" % (i, n, rho, sigma, kHK, kQR, mseBetaOLS, mseBeta))
                   simCases[j] = np.array([n,rho,sigma,kQR,np.average(simR[:,5]),np.average(simR[:,6]),np.average(simR[:,7]),np.average(simR[:,8]),np.average(simR[:,9])])
                   print(">>>>>  OLS= %.5f,  HK= %.5f,  HKB= %.5f,  LW= %.5f,  QRidge= %.5f" % (simCases[j,4], simCases[j,5], simCases[j,6], simCases[j,7], simCases[j,8]))
                   j=j+1

   header = ['n', 'rho', 'sigma', 'kQRidge', 'OLS', 'HK', 'HKB', 'LW', 'QRidge' ]
   with open('simulation.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(simCases)
   return simCases

def diabetes():
   X, y = datasets.load_diabetes(return_X_y=True)
   y = (y - np.min(y)) / (np.max(y) - np.min(y))
# Use only one feature
#X = X[:, np.newaxis, 2]

   nr, nc = X.shape
   for i in range(nc):
       X[:,i] = (X[:,i] - np.min(X[:,i])) / (np.max(X[:,i]) - np.min(X[:,i]))

   # Split the data into training/testing sets
   xTrain = X[:354,2:]    # Use only feature 2 and 3
   xTest = X[354:,2:]
   nr, nc = xTrain.shape

   # Split the targets into training/testing sets
   yTrain = y[:354]
   yTest = y[354:]    

   print("Method", "k", "beta","intercept","mse Train","mes Test", "VIF", "kappa")
   kOLS,coefOLS,interceptOLS,mseTrainOLS,mseTestOLS,vifOLS,kappaOLS = OLS(xTrain, yTrain, xTest, yTest)
   print("OLS    ",kOLS,coefOLS,interceptOLS,mseTrainOLS,mseTestOLS,vifOLS,kappaOLS)
   
   kor = np.matmul(np.transpose(xTrain), xTrain)
   Lambda,_ = eig(inv(kor))
   e = yTrain - np.matmul(xTrain * np.matmul(Lambda, np.transpose(Lambda)),coefOLS)
   sigSqrHat = np.matmul(e, np.transpose(e))/(nr-nc)
   
   kHK,coefHK,mseTrainHK,mseTestHK,vifHK,kappaHK = ridge_hk(coefOLS, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("HK     ",kHK,coefHK,mseTrainHK,mseTestHK,vifHK,kappaHK)

   kHKB,coefHKB,mseTrainHKB,mseTestHKB,vifHKB,kappaHKB = ridge_hkb(coefOLS, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("HKB    ",kHKB,coefHKB,mseTrainHKB,mseTestHKB,vifHKB,kappaHKB)

   kLW,coefLW,mseTrainLW,mseTestLW,vifLW,kappaLW = ridge_lw(coefOLS, Lambda, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("LW     ",kLW,coefLW,mseTrainLW,mseTestLW,vifLW,kappaLW )

   kQR,coefQR,mseTrainQR,mseTestQR,vifQR,kappaQR = QRidge(xTrain, yTrain, xTest, yTest, rSeed=123, boun=20, verbose=False)
   print("QRidge ",kQR,coefQR,mseTrainQR,mseTestQR,vifQR,kappaQR) 
  
   return 0                      


def watermelon():

   wm = np.loadtxt(open("watermelon.csv", "rb"), delimiter=",", skiprows=1)
   X = wm[:,1:]
   y = wm[:,0]

   nr, nc = X.shape
   for i in range(nc):
       X[:,i] = (X[:,i] - np.min(X[:,i])) / (np.max(X[:,i]) - np.min(X[:,i]))
   y = (y - np.min(y)) / (np.max(y) - np.min(y))

   # Split the data into training/testing sets
   xTrain = X[:18,:]    # Use only feature 2 and 3
   xTest = X[18:,:]
   nr, nc = xTrain.shape

   # Split the targets into training/testing sets
   yTrain = y[:18]
   yTest = y[18:]    

   print("Method", "k", "beta","intercept","mse Train","mes Test", "VIF", "kappa")
   kOLS,coefOLS,interceptOLS,mseTrainOLS,mseTestOLS,vifOLS,kappaOLS = OLS(xTrain, yTrain, xTest, yTest)
   print("OLS    ",kOLS,coefOLS,interceptOLS,mseTrainOLS,mseTestOLS,vifOLS,kappaOLS)
   
   kor = np.matmul(np.transpose(xTrain), xTrain)/nr
   Lambda,_ = eig(kor)
   e = yTrain - np.matmul(xTrain * np.matmul(Lambda, np.transpose(Lambda)),coefOLS)
   sigSqrHat = np.matmul(e, np.transpose(e))/(nr-nc)
   
   kHK,coefHK,mseTrainHK,mseTestHK,vifHK,kappaHK = ridge_hk(coefOLS, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("HK     ",kHK,coefHK,mseTrainHK,mseTestHK,vifHK,kappaHK)

   kHKB,coefHKB,mseTrainHKB,mseTestHKB,vifHKB,kappaHKB = ridge_hkb(coefOLS, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("HKB    ",kHKB,coefHKB,mseTrainHKB,mseTestHKB,vifHKB,kappaHKB)

   kLW,coefLW,mseTrainLW,mseTestLW,vifLW,kappaLW = ridge_lw(coefOLS, Lambda, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("LW     ",kLW,coefLW,mseTrainLW,mseTestLW,vifLW,kappaLW )

   kQR,coefQR,mseTrainQR,mseTestQR,vifQR,kappaQR = QRidge(xTrain, yTrain, xTest, yTest, rSeed=123, boun=1, verbose=False)
   print("QRidge ",kQR,coefQR,mseTrainQR,mseTestQR,vifQR,kappaQR) 
  
   return 0      


def gp():

   gpd = np.loadtxt(open("gp.csv", "rb"), delimiter=";")
   X = gpd[:,1:]
   y = gpd[:,0]

   nr, nc = X.shape
#   for i in range(nc):
#       X[:,i] = (X[:,i] - np.min(X[:,i])) / (np.max(X[:,i]) - np.min(X[:,i]))
   y = (y - np.min(y)) / (np.max(y) - np.min(y))

   # Split the data into training/testing sets
   xTrain = X[:350,:]    # Use only feature 2 and 3
   xTest = X[350:,:]
   nr, nc = xTrain.shape

   # Split the targets into training/testing sets
   yTrain = y[:350]
   yTest = y[350:]    

   print("Method", "k", "beta","intercept","mse Train","mes Test", "VIF", "kappa")
   kOLS,coefOLS,interceptOLS,mseTrainOLS,mseTestOLS,vifOLS,kappaOLS = OLS(xTrain, yTrain, xTest, yTest)
   print("OLS    ",kOLS,coefOLS,interceptOLS,mseTrainOLS,mseTestOLS,vifOLS,kappaOLS)
   
   kor = np.matmul(np.transpose(xTrain), xTrain)
   Lambda,_ = eig(inv(kor))
   e = yTrain - np.matmul(xTrain * np.matmul(Lambda, np.transpose(Lambda)),coefOLS)
   sigSqrHat = np.matmul(e, np.transpose(e))/(nr-nc)
   
   kHK,coefHK,mseTrainHK,mseTestHK,vifHK,kappaHK = ridge_hk(coefOLS, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("HK     ",kHK,coefHK,mseTrainHK,mseTestHK,vifHK,kappaHK)

   kHKB,coefHKB,mseTrainHKB,mseTestHKB,vifHKB,kappaHKB = ridge_hkb(coefOLS, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("HKB    ",kHKB,coefHKB,mseTrainHKB,mseTestHKB,vifHKB,kappaHKB)

   kLW,coefLW,mseTrainLW,mseTestLW,vifLW,kappaLW = ridge_lw(coefOLS, Lambda, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("LW     ",kLW,coefLW,mseTrainLW,mseTestLW,vifLW,kappaLW )

   kQR,coefQR,mseTrainQR,mseTestQR,vifQR,kappaQR = QRidge(xTrain, yTrain, xTest, yTest, rSeed=123, boun=50, verbose=False)
   print("QRidge ",kQR,coefQR,mseTrainQR,mseTestQR,vifQR,kappaQR) 
  
   return 0   


                    




                   



