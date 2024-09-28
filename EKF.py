#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:57:33 2024

@author: rahelmizrahi
"""
import pandas as pd
import numpy as np
from numpy import array, cos, sin, tan
from numpy import identity, zeros, ones, matmul, eye
from numpy.linalg import inv
import matplotlib.pyplot as plt
import re as re
import DataParser as d
from DataParser import GpsDataParser
from pprint import pprint
from cmath import sqrt
from Signals import sawtooth_wave, triangle_wave
test = False

G = 6.673 * 10**(-11 ) #N•m2/kg2
Me = 5.97219 *10**24 #kg

I3 = eye(3)
I6 = eye(6)
MIN2SEC = 60
u = 4736881359.18932 #estimae of u from the satelltie data

def printMat(M):
    # Iterate over each row in the matrix
    for row in M:
        # Create a formatted row string
        formatted_row = ','.join([f"{elem:,.2f}" for elem in row])
        print(formatted_row)  # Print the row with formatted elements
def printArr(L):
    # Iterate over each row in the matrix
        # Create a formatted row string
        formatted_row = ', '.join([f"{elem:,.2f}" for elem in L])
        print(formatted_row)  # Print the row with formatted elements

def padZero(input_):
    """ -0.1-----0----0.1"""
    clipedVal = .001
    if input_ < 0:
        if input_ > -.000001:
            input_ = -clippedVal
    elif input_ == 0:
        input_ = clippedVal
    
    elif input_ > 0:
        if input_ < .000001:
            input_ = clippedVal
    return input_


            
        
class EKF:
    def __init__(self, X0, A, B, Q, P, R, H, v, w, numStates, numMeasurements, numControlInputs, dt, data = None, f_model=None):
        """
        Parameters
        ----------
        A (array-like). nxn
        s0: state at t = 0
        
        Returns
        -------
        """
        self.n = numStates
        self.q = numMeasurements #same as the number of outputs
        self.i = numControlInputs
        self.A  = A
        self.B = B
        self.Q = Q
        self.R = R
        self.H = H
        self.v = v #model uncertainty
        self.P = P0
        self.X = X0 #the model estimates
        self.w = w#uncertainnity in sensor meas
        self.dt = dt
        #note, x, the state vector, looks like this: 
            #x = [x, x_dot, y, y_dot, z, z_dot] 
        self.y = np.array((self.q,1)) # measurements estimate
        self.K = np.eye(self.n)
        if f_model is not None:
            assert(f_model.size == (self.n)) #F is a vector containing the expected behavior of the states
            self.f_model = f_model
        
    def run(self,Xm, k):
        """
        algorithm from wikipedia was used 
        in generating this code
        """
        
        """
        predict
        """
        X = self.estimateStateFromModel(k) #predicted estimate of the states from our model, x[k|k-1]
        
        P = self.estimateCovarianceMatrix()
        
        y = self.estimateMeasurements(Xm) # y[k]
        
        K = self.computeKalmanGain()
        
        """
        Update
        P[k | k] = (I - Kk Hk ) P[k|k-1]
        
        x[k|k] = x[k | k-1] + Kk yk

        """
        self.X = X + matmul(K, y)
        KH = matmul(K, self.H)
        
        self.P =  matmul(( eye(self.n) - KH), P) #P[k | k] = (I - Kk Hk ) P[k|k-1]
        assert(self.P.shape == (self.n, self.n))
        
        
        
        
    
  
    def getF(self):
        A = np.zeros((6,6))
        x = self.X[0]
        y = self.X[2]
        z = self.X[4]
        R = sqrt(x*x + y*y + z*z )
        R3 = R*R*R
        
        
        a11 = 1 -( u*dt*x)/R3
        a12 = 0
        a21 = -(u*dt)/R3
        a22 = 1
        Ax = np.array([[a11, a12], [a21, a22]])
        
        
        a11 = 1 -( u*dt*y)/R3
        a12 = 0
        a21 = -(u*dt)/R3
        a22 = 1
        Ay = np.array([[a11, a12], [a21, a22]])
        
        a11 = 1 -( u*dt*z)/R3
        a12 = 0
        a21 = -(u*dt)/R3
        a22 = 1
        Az = np.array([[a11, a12], [a21, a22]])
        
        A[0:2, 0:2] = Ax
        A[2:4, 2:4] = Ay
        A[4:6, 4:6] = Az
        return A
        
    def estimateStateFromModel(self, k):
        """
        compute x[ k | k-1] =  f(x,u)
        """
        X = zeros((6,))
        for i in range(0, self.n):
            X[i] = self.f_model[i](k)
        
        assert(X.shape  == (self.n,))
        return X
    # def estimateStateFromModel(self):
    #     """
    #     compute x[ k | k-1] =  f(x,u)
    #     """
    #     X = zeros((6,))
    #     x  = self.X[0]
    #     x_dot = self.X[0 + 1]
    #     y = self.X[2]
    #     y_dot = self.X[2+1]
    #     z = self.X[4]
    #     z_dot = self.X[4+1]
        
    #     R = sqrt(x*x + y*y + z*z)
    #     R3 = R*R*R
    #     X[0] = x - (u*x*x*dt)/(2*R3)
    #     X[1] = x_dot - (x*u*self.dt)/R3
        
    #     X[2] = y - (u*y*y*dt)/(2*R3)
    #     X[3] = y_dot - (y*u*self.dt)/R3
        
    #     X[4] = z - (u*z*z*dt)/(2*R3)
    #     X[5] = z_dot -(z*u*self.dt)/R3
        
        
    #     assert(X.shape  == (self.n,))
    #     return X
    def estimateCovarianceMatrix(self):
        F = self.getF()
        tmp  = matmul(F, self.P)
        P  = matmul(tmp, F.T) + self.Q
        assert(P.shape == (self.n, self.n))
        return P
    def estimateMeasurements(self, Xm):
        zk =  np.array(Xm)
        h = matmul(self.H, self.X) 
        y = zk - h - self.w
        assert(y.shape == (self.q,))
        return y
    def computeInnovationCovariance(self):
        """
        note, when this called from run(), P hasn't yet 
        been updated, ie we are using 
        S[k] = H[k] P[k| k-1] H[k].T  + R[k]
        """
        HP = matmul(self.H, self.P)
        S =  matmul(HP, self.H.T) + self.R
        assert(S.shape == (self.n, self.n))
        return S
    def computeKalmanGain(self):
        """
        note, when this called from run(), P hasn't yet 
        been updated, ie we are using 
        K[k] = P[k| k-1] H.T inv(S)
        """
        S = self.computeInnovationCovariance()
        PHT = matmul(self.P, self.H.T)  
        K = matmul(PHT, inv(S))
        assert(K.shape == (self.n, self.n))
        
        return K
    
if __name__ == '__main__':
    p = GpsDataParser('Gps_meas.csv')

    run0 = False
    raw = False
    if run0:
        if raw:
            
            data = p.rawData 
            data.to_csv('GpsDataRaw.csv', index = False)
        else:
           
            dataFilt = p.removeOutliers()
            data = p.condenseData(dataFilt)
            data.to_csv('GpsDataFilt.csv', index = False)
        
    else:
        if raw:
            data = pd.read_csv('GpsDataRaw.csv')
        else:
            data = pd.read_csv('GpsDataFilt.csv')
        
    data = data[['x', 
    'x_dot', 
    'y', 
    'y_dot', 
    'z',
    'z_dot' ]]
    X0 = (data.iloc[0])
    x0 = X0['x']
    y0 = X0['y']
    z0 = X0['z']
    dt = 30*MIN2SEC
    R0 = sqrt(x0**2 + y0**2 + z0**2)
    X0 = array(data.iloc[0])
    velIndices = [1,3,5]
    posIndices = [0,2,4]

    v_model = array([(X0[vi] - (u*dt*X0[vi-1]**2/R0**3)) for vi in velIndices], dtype='float32')
    v_meas = array([X0[vi] for vi in velIndices], dtype='float32')
    diff = v_model - v_meas
    sf = v_model/v_meas
    print("scale factors:")
    printArr(sf)
    print(f"x_actual - x_model = {diff}")
    numStates = 6
    numMeasurements = numStates
    numControlInputs = 3
    dt = 30*MIN2SEC
    x0 = array(data.iloc[0])
    A = identity(numStates)
    B = zeros((numStates,numControlInputs))
    P0 = .001 * identity(numStates) #estimate of the accuracy of the predicated state, gets updated
    Q = .005 * identity(numStates)  #model uncertainty cov matrix
    R = .002 * identity(numStates)  #sensor noise cov matrix
    H = np.zeros((numMeasurements, numStates) ) #mapping btw sensor readings and predicted measurements
    H = I6 #H is the identiy bc theres is a direct 1:1 mapping between our
                  # model and measurements, ie our model gives us x,x', ..,z, z', and our measurements are
                  # those quantities exactly. 
    v = array([.1, .2, .1, 1, .2, .1])*100 #random noise
    w = v/3  #sensor noise 
    A = 6000
    T = 25
    ts = list(range(0,800))
    fx = lambda k: A*np.sin(np.deg2rad(k)/240.5)*np.sin( np.deg2rad(k)/ 16)
    fx_dot = lambda k: -A*np.sin(np.deg2rad(k/240.5))*np.sin( np.deg2rad(k / 16))
    fy = lambda k: A*np.cos(np.deg2rad(k /240.5)) *np.cos( np.deg2rad(k/ 16))
    fy_dot = lambda k: -A*np.cos(np.deg2rad(k /240.5)) *cos( np.deg2rad(k / 16))
    fz = lambda k: A*np.sin(np.deg2rad(16*k)*triangle_wave(k, (1/T) ))
    fz_dot = lambda k: -A*np.sin(np.deg2rad(16*k)*triangle_wave(k, (1/T) ))
    f_m = array([fx, fx_dot, fy, fy_dot, fz, fz_dot])
    
    ekf = EKF(x0, A, B, Q, P0, R, H, v, w, numStates, numMeasurements, numControlInputs, dt, data = data, f_model = f_m)

    
    N = data.shape[0]
    correctedData = pd.DataFrame(index = range(0, N), 
                                columns = data.columns)
    
    for k in range(0, N):
        print(f"run{k}")
        row = correctedData.iloc[k] 
        Xm = data.iloc[0]
        ekf.run(Xm, k)
        
        for i, ax in enumerate(list(['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot'])):
            correctedData.loc[k,ax] = ekf.X[i]
        
    
    plot = False
    if plot:
        plt.close()
        fig, axs = plt.subplots(len(list(correctedData.columns)), 1, figsize=(8, 12))
        t = list(range(0, correctedData.shape[0]))
        for i,ax in enumerate(list(correctedData.columns)):
            
            axs[i].plot(t,data[ax], linewidth = 0.5, color='purple', linestyle='--', label=f'{ax} raw')
            axs[i].plot(t,correctedData[ax], linewidth = 0.5, linestyle='dotted', label=f'{ax}')
            axs[i].legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()
    
    
    plot = True
    if plot:
        N = 44
        plt.close()
        fig, axs = plt.subplots(len(list(correctedData.columns)), 1, figsize=(8, 12))
        t = list(range(0, N))
        for i,ax in enumerate(list(correctedData.columns)):
            
            #axs[i].plot(t,data[ax].iloc[0:N], linewidth = 0.5, color='purple', linestyle='--', label=f'{ax} raw')
            axs[i].plot(t,correctedData[ax].iloc[0:N], linewidth = 0.5, linestyle='dotted', label=f'{ax}')
            axs[i].legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()
    
    
    
    
    
    
    
    
    
    