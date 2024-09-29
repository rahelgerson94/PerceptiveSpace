import csv
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
from scipy.fft import fft, fftfreq
import numpy as np
from  datetime import datetime as dt
from Signals import triangle_wave, sawtooth_wave
from scipy.optimize import curve_fit
TWO_WEEKS = 672 # 30 min increments


def get_data(file):
    data_dict = {}
    with open(file, 'r') as f:
        csv_reader = csv.DictReader(f)
    
        # Initialize lists for each column header
        for header in csv_reader.fieldnames:
            data_dict[header] = []
        
        # Populate the lists with data from each row
        for row in csv_reader:
            for header in csv_reader.fieldnames:
                data_dict[header].append(row[header]) if header == "time" else data_dict[header].append(float(row[header]))
    return data_dict

def moving_max(window, data):
    bucket = []
    new_data = []
    for ind, item in enumerate(data):
        if len(bucket) < window:
            bucket.append(item)
        else:
            bucket = bucket[1:]
            bucket.append(item)
            new_data.append(max(bucket))
    m = np.mean(new_data)
    new_data = [item - m for item in new_data]
    return new_data

def get_maxes(window, data, poly_window = 50, prop_forward:int = 0):
    maxes =  moving_max(window, data)
    pw_end = poly_window
    pw_start = 0
    new_data = []
    while pw_start < len(maxes):
        poly_coeffs = np.polyfit(range(pw_start,pw_end), maxes[pw_start:pw_end], 10)
        poly_fit = np.poly1d(poly_coeffs)
        new_data.extend(poly_fit(range(pw_start,pw_end)))
        pw_start = pw_end
        pw_end += min(poly_window, len(data) - pw_end - 5)
    
    if prop_forward:
        for num in range(prop_forward // poly_window):
            new_data.extend(new_data[-poly_window:])

    return new_data





        
def sine_wave(x, A, B, C):
    """
    A: Amplitude
    B: Frequency (radians per unit of x)
    C: Phase shift
    """
    return A * np.sin(B * x + C)


if __name__ == "__main__":
    import pandas as pd
    x = pd.Series([1,2,3,4,5])


    n1 = 0
    n2 = 500
    data = get_data("GpsDataFilt.csv")
    x_data = range(n1,n2)
    y_data = data["z"][n1:n2]

    #fitted_data = get_maxes(5, y_data, 50)


    # new_data = moving_max(5,y_data) #ema(.0005,data["z"])
    # new_data_ema = ema(0.1,y_data)
   
   
    ## Fit a polynomial of degree 3 (cubic)
    #degree = 60
    # poly_coeffs = np.polyfit(range(len(new_data)), new_data, 10)
    # poly_fit = np.poly1d(poly_coeffs)
    # fitted = poly_fit(range(len(new_data)))
    #print(poly_fit)
#
    ## Initial guess for parameters [Amplitude, Frequency, Phase]
#     initial_guess = [2602/2, 1/45, 0]  # These are rough estimates
# #
#     ## Perform curve fitting
#     popt, pcov = curve_fit(sine_wave, x_data, y_data, p0=initial_guess)
# #
#     ## Extract the optimal parameters (Amplitude, Frequency, Phase)
#     amplitude, frequency, phase_shift = popt
#     print(amplitude, frequency, phase_shift)
# #
#     ## Generate the fitted sine wave using the optimized parameters
#     y_fit = sine_wave(x_data, amplitude, frequency, phase_shift)
#     A = 2602/2
#     f = 1/30
#     sin1 = lambda k: A* np.cos(2*np.pi*f*k)
#     N = int(len(data['z'])/5)
#     plt.close()
    # plt.plot(data['z'][0:N])
    # plt.plot([sin1(k) for k in range(N)])

    N = 200
    plt.close()
    
    fig, axs = plt.subplots(len(['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot']), 1, figsize=(8, 12))
    t = list(range(0, N))
    window = 5
    
    for i,ax in enumerate(['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot']):
        axs[i].plot(data[ax][:N], linewidth = 0.5, label=f'({ax})')
        axs[i].plot([0]*window + get_maxes(window, data[ax][:N], 50),label=f'({ax}_Filtered)') # zero padding at window size
        axs[i].grid(True, which='both')  
        axs[i].legend(loc='upper right')
    plt.tight_layout()
    plt.show(block=False)

    N1 = 0
    NN1 = 200
    N2 = NN1 + TWO_WEEKS
    
    fig2, axs2 = plt.subplots(len(['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot']), 1, figsize=(8, 12))
    t = list(range(N1, N2))
    window = 5
    
    for i,ax in enumerate(['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot']):
        axs2[i].plot(data[ax][N1:N2], linewidth = 0.5, label=f'({ax})')
        axs2[i].plot([0]*window + get_maxes(window, data[ax][N1:NN1], 50,prop_forward=TWO_WEEKS),label=f'({ax}_Filtered)') # zero padding at window size
        axs2[i].axvline(NN1, color="r", linewidth=3)
        axs2[i].grid(True, which='both')  
        axs2[i].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()



