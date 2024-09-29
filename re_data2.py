import csv
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
from scipy.fft import fft, fftfreq
import numpy as np
from  datetime import datetime as dt
from Signals import triangle_wave, sawtooth_wave
from scipy.optimize import curve_fit


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
            new_data.append(max(bucket)) #max(data)
    return new_data


def ema(a,data):
    alg = lambda a,d, l: (1-a)*l + a*d
    last_val = 0
    new_data = []
    for item in data:
        new_data.append(alg(a,item,last_val))
        last_val = new_data[-1]
    return new_data

def ma(window, data):
    bucket = []
    new_data = []
    for ind, item in enumerate(data):
        if len(bucket) < window:
            bucket.append(item)
        else:
            bucket = bucket[1:]
            bucket.append(item)
            new_data.append(float(np.mean(bucket)))
    return new_data
        
def sine_wave(x, A, B, C):
    """
    A: Amplitude
    B: Frequency (radians per unit of x)
    C: Phase shift
    """
    return A * np.sin(B * x + C)


if __name__ == "__main__":
    n1 = 0
    n2 = 50
    data = get_data("GpsDataFilt.csv")
    x_data = range(n1,n2)
    y_data = data["z"][n1:n2]
    new_data = moving_max(5,y_data) #ema(.0005,data["z"])
    new_data_ema = ema(0.1,y_data)
   
   
    ## Fit a polynomial of degree 3 (cubic)
    #degree = 60
    poly_coeffs = np.polyfit(range(len(new_data)), new_data, 10)
    poly_fit = np.poly1d(poly_coeffs)
    fitted = poly_fit(range(len(new_data)))
    #print(poly_fit)
#
    ## Initial guess for parameters [Amplitude, Frequency, Phase]
    initial_guess = [2602/2, 1/45, 0]  # These are rough estimates
#
    ## Perform curve fitting
    popt, pcov = curve_fit(sine_wave, x_data, y_data, p0=initial_guess)
#
    ## Extract the optimal parameters (Amplitude, Frequency, Phase)
    amplitude, frequency, phase_shift = popt
    print(amplitude, frequency, phase_shift)
#
    ## Generate the fitted sine wave using the optimized parameters
    y_fit = sine_wave(x_data, amplitude, frequency, phase_shift)
    A = 2602/2
    f = 1/30
    sin1 = lambda k: A* np.cos(2*np.pi*f*k)
    N = int(len(data['z'])/5)
    plt.close()
    plt.plot(data['z'][0:N])
    plt.plot([sin1(k) for k in range(N)])

    # plt.plot(new_data, "r")
    # plt.plot(fitted,"g-")
    # plt.plot(y_fit)
    plt.legend()
    plt.show()

