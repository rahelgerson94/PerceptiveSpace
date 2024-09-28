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
    
class Satellite:
    def __init__(self):
        self.position = (0,0,0)
        self.velocity = (0,0,0)
        self.axes = ['x', 'y', 'z']
        for idx, row in self.dataFiltered.iterrows():
            if row['ECEF'] == 'x':
                posLowerBound = self.positionMeans['x']  - 3 * self.positionStds['x'] 
                posUpperBound = self.positionMeans['x']  + 3 * self.positionStds ['x']
                
                velLowerBound = self.velocityMeans['x']  - 3 * self.velocityStds['x'] 
                velUpperBound = self.velocityMeans['x']  + 3 * self.velocityStds ['x']
                
                if not ((row['velocity'] <= velUpperBound and  row['velocity'] >= velLowerBound) 
                    and (row['position'] <= posUpperBound and  row['position'] >= posLowerBound)):
                        
                
                    self.outliers = pd.concat([self.outliers, row])
                    self.dataFiltered = self.dataFiltered.drop([idx, idx+1, idx+2])
                        
            elif row['ECEF'] == 'y':
                
                posLowerBound = self.positionMeans['y']  - 3 * self.positionStds['y'] 
                posUpperBound = self.positionMeans['y']  + 3 * self.positionStds ['y']
                
                velLowerBound = self.velocityMeans['y']  - 3 * self.velocityStds['y'] 
                velUpperBound = self.velocityMeans['y']  + 3 * self.velocityStds ['y']
                
                if not ((row['velocity'] <= velUpperBound and  row['velocity'] >= velLowerBound) 
                    and (row['position'] <= posUpperBound and  row['position'] >= posLowerBound)):
                    
                    self.outliers = pd.concat([self.outliers, row])
                    self.dataFiltered = self.dataFiltered.drop([idx-1, idx, idx+1])
                    
                    
            elif row['ECEF'] == 'z':
                posLowerBound = self.positionMeans['z']  - 3 * self.positionStds['z'] 
                posUpperBound = self.positionMeans['z']  + 3 * self.positionStds ['z']

                velLowerBound = self.velocityMeans['z']  - 3 * self.velocityStds['z'] 
                velUpperBound = self.velocityMeans['z']  + 3 * self.velocityStds ['z']
                
                if not ((row['velocity'] <= velUpperBound and  row['velocity'] >= velLowerBound) 
                    and (row['position'] <= posUpperBound and  row['position'] >= posLowerBound)):
                    
                    self.outliers = pd.concat([self.outliers, row])
                    self.dataFiltered = self.dataFiltered.drop([ idx-1, idx-2, idx])
            if write:
                self.dataFiltered.to_csv('GPS_meas_filtered.csv')
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
    def convertToCylindricalCoords(self):
        posRows = self.posDataReformatted.shape[0]
        posData = pd.DataFrame(index = list(range(posRows)),  columns = ['time', 'clock', 'sv', 'r', 'theta', 'Phi'])
        
        velRows = self.velDataReformatted.shape[0]
        velData = pd.DataFrame(index = list(range(velRows)),  columns = ['time', 'clock', 'sv', 'r_dot', 'theta_dot', 'Phi_dot'])
        for idx, row in self.posDataReformatted.iterrows():
            x = row['x']
            y = row['y']
            z = row['z']
            r, theta, Phi = self.Ekf2Cyl(x,y,z)
            posData.loc[idx,'time'] = row['time'] 
            posData.loc[idx,'clock'] = row['clock'] 
            posData.loc[idx,'sv'] = row['sv'] 
            posData.loc[idx, 'r'] = r
            posData.loc[idx, 'theta'] = theta
            posData.loc[idx, 'Phi'] = Phi
            
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
    def convertToCylindricalCoords2(self, data):
        cylData = pd.DataFrame(index = list(range(data.shape[0])),  columns = ['time', 'clock', 'sv', 'r', 'r_dot', 'theta', 'theta_dot', 'Phi', 'Phi_dot'])

        for idx, row in data.iterrows():
            x = row['x']
            y = row['y']
            z = row['z']
            r, theta, Phi = self.Ekf2Cyl(x,y,z)
            
            x_dot = row['x_dot']
            y_dot = row['y_dot']
            z_dot = row['z_dot']
            r_dot, theta_dot, Phi_dot =  self.Ekf2Cyl(x_dot,y_dot,z_dot) 
            cylData.loc[idx, 'r'] = r
            cylData.loc[idx, 'theta'] = theta
            cylData.loc[idx, 'Phi'] = Phi
            cylData.loc[idx, 'r_dot'] = r_dot
            cylData.loc[idx, 'theta_dot'] = theta_dot
            cylData.loc[idx, 'Phi_dot'] = Phi_dot
        return cylData
    def data2Eci(self, data):
        dataEci =  pd.DataFrame(index = list(range(data.shape[0])),  columns = ['time', 
                                                                                'X', 
                                                                                'X_dot', 
                                                                                'Y', 
                                                                                'Y_dot', 
                                                                                'Z',
                                                                                'Z_dot', ])
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
    s = Satellite('GPS_meas.csv')
    
    # Count the number of times a value is within the max ± threshold
    
    
    run0 = True
    raw = False
    if run0: 
        if raw:
            s.condensedDataRaw.to_csv('GpsDataRaw.csv', index = False)
            data = s.condensedDataRaw
        else:
            s.condensedDataFiltered.to_csv('GpsDataFilt.csv', index = False)
            data = s.condensedDataFiltered
    else:
        if raw:
            data = pd.read_csv('GpsDataRaw.csv')
        else:
            data = pd.read_csv('GpsDataFilt.csv')
        print(data.columns)
    
    dataEci = s.data2Eci(data)
    dataEci.to_csv('GpsDataFilteredEci_0.csv')
    #condensedDataFiltered, condensedDataRaw = s.condenseData()
    plotCyl = True
    cylData = s.convertToCylindricalCoords2(data)
    if plotCyl:
        
        cols = ['r', 'r_dot', 'theta', 'theta_dot', 'Phi', 'Phi_dot' ]
        fig, axs = plt.subplots(len(cols), 1, figsize=(8, 12))
        N = 100
        t = [i*30 for i in list(range(0, N))]
        
        for i,ax in enumerate(cols):
            
            axs[i].plot(t,cylData[ax].iloc[0:N], linewidth = 0.5, label=f'{ax}')
            axs[i].set_xlabel('t [mins]')
            axs[i].legend()
            plt.grid(True)
        plt.tight_layout()
        plt.show()
    Rmean = cylData['r'].mean()

    plotVel = False
    if plotVel:
        plt.close()
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))
        t = list(range(0, s.velDataReformatted.shape[0]))
        for i,ax in enumerate(s.axes):
            axs[i].plot(t,s.velDataReformatted[ax], linewidth = 0.5, label=f'Velocity ({ax})')
            axs[i].set_xlabel('t [mins]')
            axs[i].legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    
    plotPos = True
    if plotPos:
        plt.close()
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))
        t = list(range(0, s.posDataReformatted.shape[0]))
        for i,ax in enumerate(s.axes):
            axs[i].plot(t,s.posDataReformatted[ax], linewidth = 0.5, label=f'Position ({ax})')
            axs[i].legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()


    # if plotPos:
    #     fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    #     for i,ax in enumerate(s.axes):
    #         data = s.dataFiltered[s.dataFiltered['ECEF'] == ax][0:-1]
    #         t = list(range(0, len(data)))
    #         axs[i].plot(t,data['position'], linewidth = 0.5, label=f'Position ({ax})')
    #         axs[i].legend()
    #     plt.tight_layout()
    #     plt.show()
    #rDf = pd.DataFrame(columns = s.data.columns)
    
    # xPosData = s.dataFiltered[s.dataFiltered['ECEF'] == 'x']
    # yPosData = s.dataFiltered[s.dataFiltered['ECEF'] == 'y']
    # rowsInNewData = int( len(s.dataFiltered)/3) + 1
    # posData = pd.DataFrame(index = list(range(rowsInNewData)),  columns = ['time', 'clock', 'sv', 'x', 'y', 'z'])
    
    # velData = pd.DataFrame(index = list(range(rowsInNewData)),  columns = ['time', 'clock', 'sv', 'x', 'y', 'z'])
    
    # writeIndex = 0
    # for readIndex in range(0, len(s.dataFiltered), 1):
        
    #     row = s.dataFiltered.iloc[readIndex]
    #     if readIndex % 3 == 0:
    #         posData.loc[writeIndex,'x'] = row['position']
    #         velData.loc[writeIndex,'x'] = row['velocity']
            
    #     elif readIndex % 3 == 1:
    #         posData.loc[writeIndex,'y'] = row['position']
    #         velData.loc[writeIndex,'y'] = row['velocity']
    #     elif readIndex % 3 == 2:
    #         posData.loc[writeIndex,'z'] = row['position']
    #         velData.loc[writeIndex,'z'] = row['velocity']
    #         writeIndex = writeIndex + 1
    #     else:
    #         print('ERROR IN HERE')
    #     posData.loc[writeIndex,'time'] = row['time'] 
    #     posData.loc[writeIndex,'clock'] = row['clock'] 
    #     posData.loc[writeIndex,'sv'] = row['sv'] 
        
        
    
    
    # plt.show()
    # plt.plot(pCylind['r'])
    
    posMax = s.dataFiltered['position'].max()
    thresh =.09*posMax
    numMaxes =  s.dataFiltered[(s.dataFiltered['position'] >= (posMax - thresh)) 
                               & (s.dataFiltered['position'] <= (posMax + thresh))].shape[0]