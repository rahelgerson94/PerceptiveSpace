import csv
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
from scipy.fft import fft, fftfreq
import numpy as np
from  datetime import datetime as dt
from Signals import triangle_wave, sawtooth_wave 


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

def plot(data, col="z", range=None,cutoff = 2000, fs= 30, order=5):
    t = data["time"][range[0]:range[1]] if range else data["time"]
    d = data[col][range[0]:range[1]] if range else data[col]
    dlf = lowpass_filter(d, cutoff=3, fs=500)
    plt.plot(t,d)
    plt.plot(t,dlf)
    plt.xlabel("time")
    plt.ylabel(col)
    plt.legend([col, f"col_filtered"])
    plt.xticks(rotation=45)
    plt.show()
    plt.grid(True)
    plt.tight_layout()

def butter_lowpass(cutoff, fs, order=5):
    # Normalize the cutoff frequency by the Nyquist frequency
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    # Get the filter coefficients
    b, a = butter_lowpass(cutoff, fs, order=order)
    # Apply the filter using filtfilt (zero-phase filtering)
    y = filtfilt(b, a, data)
    return y

def gen_signal(duration_sec, sin_freq_l, sin_freq_h, amplitude_l, amplitude_h, number_points):
    arr = np.linspace(0.0, duration_sec, number_points, endpoint=False).tolist()
    make_sin = lambda f, A, point: -A * np.sin(2 * np.pi * f * point - 10)
    spoofed_data = [make_sin(sin_freq_l, amplitude_l, ii) * make_sin(sin_freq_h, amplitude_h, ii) for ii in arr]

    return spoofed_data


if __name__ == "__main__":
    data = get_data("/Users/eitangerson/Desktop/PerceptiveSpace/GpsDataFilt.csv")
    fs = 1/(30 * 60 * 20)
    n = len(data["z"])
    yf = fft(data["z"])
    freqs = [ii for ii in fftfreq(n,d=fs).tolist() if ii > 0]
    low = min(freqs)
    arb = freqs[int(len(freqs)/20)]
    high = max(freqs)






    #find_sampling_frequency(data["time"])
    #plot(data,"z", range=None,cutoff=1000000,fs = 30*60)
    #spoofed_h = gen_signal(1/(30*60), 30*60*len(data["z"]),high, max(data["z"]))
    #spoofed_l = gen_signal(1/(30*60), 30*60*len(data["z"]),low, max(data["z"]))

    #spoofed = gen_signal(5000, low, high, ma)

    #n = 100
    ##plt.plot(data["z"][0:n])
    #plt.plot(spoofed_h[0:n])
    ##plt.plot(spoofed_l[0:n])
    #plt.show()

    # A = 6000
    # arr = list(range(0,len(data["z"])))#np.linspace(0.0, 200, 10000, endpoint=False).tolist() 572
    # fz = lambda k: A*np.sin(np.deg2rad(270)*k - 10) # *triangle_wave(k, (1/low) )
    # fz_dot = lambda k: -A*np.sin(np.deg2rad(16*k)*triangle_wave(k, (1/240.5) ))
    # plt.plot([fz(ii) for ii in arr])
    # plt.plot(data["z"])
    # #plt.plot(yf)
    # plt.show()

    x = lowpass_filter(data["z"], 1/1003.3056510159993, 1)
    plt.plot(x)
    plt.plot(data["z"],"r:")
    plt.show()