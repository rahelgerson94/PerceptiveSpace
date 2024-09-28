#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:37:15 2024

@author: rahelmizrahi
"""

import pandas as pd
import numpy as np
from numpy import array, cos, sin, tan
from numpy import identity, zeros, ones, matmul, eye
import matplotlib.pyplot as plt
import re as re
from Signals import triangle_wave, sawtooth_wave 
MIN2SEC = 60
w_en = 7.2921159 *10**-5 #rad/sec, earths velocity about the z axis
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
def split_filename_extension(path):
    # Use a regex to capture the filename and extension
    match = re.match(r"(.+)\.([^.]+)$", path)
    
    if match:
        # Return the filename and extension
        return match.group(1), match.group(2)
    else:
        # If no extension is found, return the filename and empty extension
        return path, ''

    
class GpsDataParser:
    def __init__(self, gpsData):
        self.dataRaw = pd.read_csv(gpsData)

        self.positionMeans = {'x': 0, 'y': 0, 'z': 0}
        self.positionStds =  {'x': 0, 'y': 0, 'z': 0}
        self.velocityMeans = {'x': 0, 'y': 0, 'z': 0}
        self.velocityStds =  {'x': 0, 'y': 0, 'z': 0}
        self.getVariabilityParams()
        self.outliers = pd.DataFrame(columns = self.dataRaw.columns)
                
  
    def reformatData(self, commutatedData):
        rows = int(len(commutatedData)/3)
        posDataReformatted = pd.DataFrame(index = list(range(rows)),  columns = ['time', 'clock', 'sv', 'x', 'y', 'z'])
        velDataReformatted = pd.DataFrame(index = list(range(rows)),  columns = ['time', 'clock', 'sv', 'x', 'y', 'z'])
        writeIndex = 0
        for i in range(0, len(commutatedData), 1):
            row = commutatedData.iloc[i]
            if i %3 == 0:
                posDataReformatted.loc[writeIndex,'time'] = row['time'] 
                posDataReformatted.loc[writeIndex,'clock'] = row['clock'] 
                posDataReformatted.loc[writeIndex,'sv'] = row['sv'] 
                
                velDataReformatted.loc[writeIndex,'time'] = row['time'] 
                velDataReformatted.loc[writeIndex,'clock'] = row['clock'] 
                velDataReformatted.loc[writeIndex,'sv'] = row['sv'] 
                writeIndex=writeIndex+1
                
        posDataReformatted.reset_index()
        velDataReformatted.reset_index()
        tmp = commutatedData[commutatedData['ECEF'] == 'x']
        posX = tmp['position'].dropna().reset_index(drop=True)
        
        tmpY = commutatedData[commutatedData['ECEF'] == 'y']
        posY = tmpY['position'].dropna().reset_index(drop=True)
        
        tmpZ = commutatedData[commutatedData['ECEF'] == 'z']
        posZ = tmpZ['position'].dropna().reset_index(drop=True)
        
        for idx, r in posDataReformatted.iterrows():                
            posDataReformatted.loc[idx, 'x']= posX[idx]
            posDataReformatted.loc[idx, 'y']= posY[idx]
            posDataReformatted.loc[idx, 'z']= posZ[idx]
        
        tmp = commutatedData[commutatedData['ECEF'] == 'x']
        velX = tmp['velocity'].reset_index(drop=True)
        
        tmpY = commutatedData[commutatedData['ECEF'] == 'y']
        velY = tmpY['velocity'].reset_index(drop=True)
        
        tmpZ = commutatedData[commutatedData['ECEF'] == 'z']
        velZ = tmpZ['velocity'].reset_index(drop=True)
        for idx, r in velDataReformatted.iterrows():                
            velDataReformatted.loc[idx, 'x']= velX[idx]
            velDataReformatted.loc[idx, 'y']= velY[idx]
            velDataReformatted.loc[idx, 'z']= velZ[idx]
        return posDataReformatted, velDataReformatted
            
    
    def condenseData(self, commutatedData):
        posDataReformatted, velDataReformatted = self.reformatData(commutatedData)

        velData = velDataReformatted.rename(columns= {'x': 'x_dot', 'y': 'y_dot', 'z': 'z_dot'})
        condensedData =  pd.concat([posDataReformatted['time'], 
                                            posDataReformatted['x'],
                                            velData['x_dot'],
                                            posDataReformatted['y'], 
                                            velData['y_dot'], 
                                            posDataReformatted['z'], 
                                            velData['z_dot'] ], axis = 1)
        
        
        
        
        return condensedData
        
    def getVariabilityParams(self):
        for ax in ['x', 'y', 'z']:
            df = self.dataRaw[self.dataRaw['ECEF'] == ax]
            self.positionStds[ax] = df['position'].std()
            self.positionMeans[ax] = df['position'].mean()
            self.velocityStds[ax] = df['velocity'].std()
            self.velocityMeans[ax] = df['velocity'].mean()
        
            
   
    def removeOutliers(self, write=False):
        dataFiltered = self.dataRaw.copy()
        for idx, row in dataFiltered.iterrows():
            
            if row['ECEF'] == 'x':
                posLowerBound = self.positionMeans['x']  - 3 * self.positionStds['x'] 
                posUpperBound = self.positionMeans['x']  + 3 * self.positionStds ['x']
                
                velLowerBound = self.velocityMeans['x']  - 3 * self.velocityStds['x'] 
                velUpperBound = self.velocityMeans['x']  + 3 * self.velocityStds ['x']
                
                if not ((row['velocity'] <= velUpperBound and  row['velocity'] >= velLowerBound) 
                    and (row['position'] <= posUpperBound and  row['position'] >= posLowerBound)):
                        
                
                    self.outliers = pd.concat([self.outliers, row])
                    dataFiltered = dataFiltered.drop([idx, idx+1, idx+2])
                        
            elif row['ECEF'] == 'y':
                
                posLowerBound = self.positionMeans['y']  - 3 * self.positionStds['y'] 
                posUpperBound = self.positionMeans['y']  + 3 * self.positionStds ['y']
                
                velLowerBound = self.velocityMeans['y']  - 3 * self.velocityStds['y'] 
                velUpperBound = self.velocityMeans['y']  + 3 * self.velocityStds ['y']
                
                if not ((row['velocity'] <= velUpperBound and  row['velocity'] >= velLowerBound) 
                    and (row['position'] <= posUpperBound and  row['position'] >= posLowerBound)):
                    
                    self.outliers = pd.concat([self.outliers, row])
                    dataFiltered = dataFiltered.drop([idx-1, idx, idx+1])
                    
                    
            elif row['ECEF'] == 'z':
                posLowerBound = self.positionMeans['z']  - 3 * self.positionStds['z'] 
                posUpperBound = self.positionMeans['z']  + 3 * self.positionStds ['z']

                velLowerBound = self.velocityMeans['z']  - 3 * self.velocityStds['z'] 
                velUpperBound = self.velocityMeans['z']  + 3 * self.velocityStds ['z']
                
                if not ((row['velocity'] <= velUpperBound and  row['velocity'] >= velLowerBound) 
                    and (row['position'] <= posUpperBound and  row['position'] >= posLowerBound)):
                    
                    self.outliers = pd.concat([self.outliers, row])
                    dataFiltered = dataFiltered.drop([ idx-1, idx-2, idx])
            if write:
                dataFiltered.to_csv('GPS_meas_filtered.csv')
        return dataFiltered
      
    def getStateFromRow(self, row):
        x = row['x']
        y = row['y']
        z = row['z']
        
        
        x_dot = row['x_dot']
        y_dot = row['y_dot']
        z_dot = row['z_dot']
        return x, x_dot, y, y_dot, z, z_dot
    def getPhi(self, x,y,z):
        x = padZero(x)
        y = padZero(y)
        z = padZero(z)
        
        # Conditional computation of phi based on x and y
        if x > 0:
            phi = np.arctan(y / x)
        elif x < 0 and y >= 0:
            phi = np.arctan(y / x) + np.pi
        elif x < 0 and y < 0:
            phi = np.arctan(y / x) - np.pi
        elif x == 0 and y > 0:
            phi = np.pi / 2
        elif x == 0 and y < 0:
            phi = -np.pi / 2
        else:
            phi = None  # undefined if x = 0 and y = 0
            
        # Return the computed values
        return phi
    
    def getTheta(self, x,y,z):
        # Compute r and other useful quantities
        x = padZero(x)
        y = padZero(y)
        z = padZero(z)
        
        r = np.sqrt(x**2 + y**2 + z**2)
        rho = np.sqrt(x**2 + y**2)  # sqrt(x^2 + y^2)
        
        # Conditional computation of theta
        if r == 0:  # if x = y = z = 0, theta is undefined
            theta = 0
        else:
            if z > 0:
                theta = np.arctan(rho / z)
            elif z < 0:
                theta = np.pi + np.arctan(rho / z)
            elif z == 0 and rho != 0:
                theta = np.pi / 2
            elif z == 0 and x == 0 and y == 0:
                theta = None  # undefined if x = y = z = 0
                print(f'getTeheta: x = y = z = 0')
        return theta
    def Ekf2Cyl(self, x,y,z):
        Phi = self.getPhi(x,y,z)
        theta = self.getTheta(x,y,z)
        r = (x*x + y*y + z*z)**.5
        return r, theta, Phi
        
        sphericalData = pd.DataFrame(index = list(range(posRows)),  columns = ['time', 'clock', 'sv',
                                                                          'r', 'r_dot',
                                                                            'theta',  'theta_dot', 
                                                                            'Phi',
                                                                            'Phi_dot'])
        
    
        for idx, row in sphericalData.iterrows():
            x = row['x']
            y = row['y']
            z = row['z']
            r, theta, Phi = self.Ekf2Cyl(x,y,z)
            sphericalData[idx,'time'] = row['time'] 
            sphericalData[idx,'clock'] = row['clock'] 
            sphericalData[idx,'sv'] = row['sv'] 
            sphericalData[idx, 'r'] = r
            sphericalData[idx, 'theta'] = theta
            sphericalData[idx, 'Phi'] = Phi
            
        for idx, row in self.velDataReformatted.iterrows():
            x = row['x']
            y = row['y']
            r, theta, Phi = self.Ekf2Cyl(x,y,z)
            velData.loc[idx,'time'] = row['time'] 
            velData.loc[idx,'clock'] = row['clock'] 
            velData.loc[idx,'sv'] = row['sv'] 
            velData.loc[idx, 'r_dot'] = r
            velData.loc[idx, 'theta_dot'] = theta
        velData = velData.rename(columns= {'x': 'x_dot', 'y': 'y_dot', 'z': 'z_dot'})
        data = pd.concat([posData['r'], velData['r_dot'], posData['theta'], velData['theta_dot'], posData['Phi'], velData['Phi_dot'] ], axis = 1)
        return data
    def convertToSphericalCoords(self, data):
        sphData = pd.DataFrame(index = list(range(data.shape[0])),  columns = ['time', 'r', 'r_dot', 'theta', 'theta_dot', 'Phi', 'Phi_dot'])

        for idx, row in data.iterrows():
            x = row['x']
            y = row['y']
            z = row['z']
            r, theta, Phi = self.Ekf2Cyl(x,y,z)
            
            x_dot = row['x_dot']
            y_dot = row['y_dot']
            z_dot = row['z_dot']
            r_dot, theta_dot, Phi_dot =  self.Ekf2Cyl(x_dot,y_dot,z_dot) 
            sphData.loc[idx, 'r'] = r
            sphData.loc[idx, 'theta'] = theta
            sphData.loc[idx, 'Phi'] = Phi
            sphData.loc[idx, 'r_dot'] = r_dot
            sphData.loc[idx, 'theta_dot'] = theta_dot
            sphData.loc[idx, 'Phi_dot'] = Phi_dot
        return sphData
    def data2Eci(self, data):
        dataEci =  pd.DataFrame(index = list(range(data.shape[0])),  columns = ['time', 
                                                                                'X', 
                                                                                'X_dot', 
                                                                                'Y', 
                                                                                'Y_dot', 
                                                                                'Z',
                                                                                'Z_dot'])
        dt = 30*MIN2SEC
        sideerealAngle0 = 0
        for idx, row in data.iterrows():
            x, x_dot, y, y_dot, z, z_dot = self.getStateFromRow(row)
            posEcef = array([x,y,z]).reshape((3,))
            velEcef = array([x_dot, y_dot, z_dot]).reshape((3,))
            longitudeAngle = self.getPhi(x,y,z)
            sideerealAngle = sideerealAngle0 + w_en *dt + longitudeAngle
            posEci, velEci = self.getEciVectors(posEcef, velEcef, longitudeAngle)
            dataEci.loc[idx, 'time'] = row['time']
            dataEci.loc[idx, 'X'] = posEci[0]
            dataEci.loc[idx, 'X_dot'] = velEci[0]
            dataEci.loc[idx, 'Y'] = posEci[1]
            dataEci.loc[idx, 'Y_dot'] = velEci[1]
            dataEci.loc[idx, 'Z'] = posEci[2]
            dataEci.loc[idx, 'Z_dot'] = velEci[2]
        return dataEci
        
    def getEciVectors(self, posEcef, velEcef, theta):
        dcm = np.array([
               [np.cos(theta),  np.sin(theta), 0],
               [-np.sin(theta), np.cos(theta), 0],
               [0,              0,             1]
               ])
        posEci = matmul(dcm, posEcef)
        velEci = velEcef  + np.cross(array([0,0, w_en]), posEcef)
        assert(posEci.shape == (3,))
        assert(velEci.shape == (3,))
        return posEci, velEci
        
if __name__ == '__main__':
    p = GpsDataParser('GPS_meas.csv')
    dataFilt = p.removeOutliers()
    dataFilt = p.condenseData(dataFilt)

    # Count the number of times a value is within the max ± threshold
    
    
    run0 = False
    raw = False
    if run0: 
        if raw:
            
            data = p.dataRaw
            data = p.condenseData(data)
            data.to_csv('GpsDataRaw.csv', drop_index = True)
        else:
            dataFilt = p.removeOutliers()
            dataFilt = p.condenseData(dataFilt)
            dataFilt.to_csv('GpsDataFilt.csv', drop_index = True)
    else:
        if raw:
            data = pd.read_csv('GpsDataRaw.csv')
        else:
            data = pd.read_csv('GpsDataFilt.csv')
        print(data.columns)
    
   
   
    plotSph = True
    plotSph = False
    sphData = p.convertToSphericalCoords(dataFilt)
    if plotSph:
        sphData = p.convertToSphericalCoords(dataFilt)
        cols = ['r', 'r_dot', 'theta', 'theta_dot', 'Phi', 'Phi_dot' ]
        fig, axs = plt.subplots(len(cols), 1, figsize=(8, 12))
        N = 100
        t = [i*30 for i in list(range(0, N))]
        
        for i,ax in enumerate(cols):
            
            axs[i].plot(t,sphData[ax].iloc[0:N], linewidth = 0.5, label=f'{ax}')
            axs[i].set_xlabel('t [mins]')
            axs[i].legend()
            plt.grid(True)
        plt.tight_layout()
        plt.show()
    Rmean = sphData['r'].mean()
    
    pltEcef = True
    interactive = True
    # if interactive == True:
    #     import mplcursors
    A = 6000
    fx = lambda k: A*np.sin(np.deg2rad(k*240.5))*np.sin( np.deg2rad(k*16))
    
    fx_dot = lambda k: -A*np.sin(np.deg2rad(k*240.5))*np.sin( np.deg2rad(k *16))
   
    fy = lambda k: A*np.cos(np.deg2rad(k *240.5)) *np.cos( np.deg2rad(k*16))
    fy_dot = lambda k: -A*np.cos(np.deg2rad(k *240.5)) *cos( np.deg2rad(k *16))
    fz = lambda k: A*np.sin(np.deg2rad(16*k)*triangle_wave(k, (1/240.5) ))
    fz_dot = lambda k: -A*np.sin(np.deg2rad(16*k)*triangle_wave(k, (1/240.5) ))
    f_m = array([fx, fx_dot, fy, fy_dot, fz, fz_dot])
    if pltEcef:
        N = 200
        plt.close()
        fig, axs = plt.subplots(len(['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot']), 1, figsize=(8, 12))
        t = list(range(0, N))
        
        for i,ax in enumerate(['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot']):
            axs[i].plot(t, data[ax][:N], linewidth = 0.5, label=f'({ax})')
            f = [f_m[i](t_i) for t_i in t]
            axs[i].plot(t, f, linewidth = 0.5, label=f'({ax})')
            axs[i].grid(True, which='both')  
            #mplcursors.cursor(axs[i], hover=True)
            axs[i].legend()
        plt.tight_layout()
        
        plt.show()



    posMax = p.dataFiltered['position'].max()
    thresh =.09*posMax
    numMaxes =  p.dataFiltered[(p.dataFiltered['position'] >= (posMax - thresh)) 
                               & (p.dataFiltered['position'] <= (posMax + thresh))].shape[0]