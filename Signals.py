#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 12:59:23 2024

@author: rahelmizrahi
"""
import pandas as pd
import numpy as np
from numpy import array, cos, sin, tan
from numpy import identity, zeros, ones, matmul, eye
from numpy.linalg import inv
import matplotlib.pyplot as plt
import re as re

from cmath import sqrt
test = False
import numpy as np

def sawtooth_wave(t, amplitude=1, frequency=1, phase=0, slope=1):
    """
    Generate a sawtooth wave signal with adjustable slope.
    
    Parameters:
    - t: time or array of time values
    - amplitude: peak value of the wave (default is 1)
    - frequency: frequency of the wave in Hz (default is 1)
    - phase: phase shift of the wave in radians (default is 0)
    - slope: factor to adjust the slope of the wave (default is 1, higher value = less steep rise)
    
    Returns:
    - y: values of the sawtooth wave at time t
    """
    # Adjust time for phase and frequency
    t_shifted = t * frequency + phase / (2 * np.pi)
    
    # Generate sawtooth wave with adjustable slope
    y = amplitude * (2 * (t_shifted - np.floor(t_shifted + 0.5))) * slope
    
    return y


def triangle_wave(t, amplitude=1, frequency=1, phase=0):
    """
    Generate a triangle wave signal.
    
    Parameters:
    - t: time or array of time values
    - amplitude: peak value of the wave (default is 1)
    - frequency: frequency of the wave in Hz (default is 1)
    - phase: phase shift of the wave in radians (default is 0)
    
    Returns:
    - y: values of the triangle wave at time t
    """
    # Adjust time for phase and frequency
    t_shifted = t * frequency + phase / (2 * np.pi)
    
    # Generate triangle wave
    y = 2 * amplitude * np.abs(2 * (t_shifted - np.floor(t_shifted + 0.5))) - amplitude
    return y
if __name__ == '__main__':
    ts = list(range(70000))
    A = 6000
    T = 42300
    tri = [triangle_wave(t, 1, (1/T) ) for t  in ts ]
    sawtooth = [sawtooth_wave(t, 1, 1/T, 0, 1) for t in ts]
    sin = [A*np.sin(np.deg2rad(t)/16))  for t in ts]
    conv = np.convolve(tri, sin, mode='same')
    mult = [sin[t] * tri[t] for t in ts]
    
    
    plt.plot(ts, mult)
    plt.show()
