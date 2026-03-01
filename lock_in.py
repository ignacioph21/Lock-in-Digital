import numpy as np 
import matplotlib.pyplot as plt
from pyroomacoustics.utilities import fractional_delay
from scipy.signal import convolve
from scipy.signal import butter, filtfilt, hilbert
from Formatter import read
from numba import njit

def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def fft_parameters(ts, signal):
    fs = 1/(ts[1] - ts[1-1])
    freqs = np.fft.rfftfreq(len(ts), d=1/fs)
    Y = 2 * np.fft.rfft(signal) / len(ts)   
    idx = np.argwhere(np.abs(Y) == np.max(np.abs(Y)))[0][0]
    fr = freqs[idx]
    Ar = np.abs(Y[idx])    
    return Ar, fr

def resample(ts, signal, t0=-0.5):
    fs = 1/(ts[1] - ts[1-1])
    h = fractional_delay(t0)
    delayed = convolve(signal, h, mode="same")
    new_ts = ts - t0/fs
    return new_ts, delayed

def shift_90(ts, signal, fr):
    fs = 1/(ts[1] - ts[1-1])
        
    Delta = -1/4 * fs/fr # 90° shift.
    h_cos = fractional_delay(Delta)
    cos_measured = convolve(signal, h_cos, mode="same")
    return cos_measured

@njit
def exponential_lowpass(X0, Y0, alpha, X1_prev, Y1_prev, X_prev, Y_prev):
    n = len(X0)
    X = np.empty(n)
    Y = np.empty(n)

    X1 = X1_prev
    Y1 = Y1_prev

    x_last = X_prev
    y_last = Y_prev

    for i in range(n):
        X1 += alpha * (X0[i] - X1)
        Y1 += alpha * (Y0[i] - Y1)

        x_last += alpha * (X1 - x_last)
        y_last += alpha * (Y1 - y_last)

        X[i] = x_last
        Y[i] = y_last

    return X1, Y1, X, Y



def low_pass_filter(X0, Y0, cut_off, fs, mode="butter", previous=None, return_previous=False):
    if mode == "butter":
        b, a = butter_lowpass(cut_off, fs)
        X = filtfilt(b, a, X0)
        Y = filtfilt(b, a, Y0)
        if return_previous:
            return X, Y, None
        return X, Y

    if mode == "exponential":
        if previous is None:
            X1_prev = X0[0] 
            Y1_prev = Y0[0] 
            X_prev = X0[0] 
            Y_prev = Y0[0] 
        else:
            X1_prev, Y1_prev, X_prev, Y_prev = previous

        gamma = 2 * np.pi * cut_off / fs
        cosg = np.cos(gamma)
        alpha = cosg - 1 + np.sqrt(cosg**2 - 4*cosg + 3)
        
        if not (0 < alpha < 1):
            raise ValueError(f"Alpha fuera de rango: {alpha}. Puede que cut_off={cut_off} sea demasiado alto para fs={fs}.")

        X1, Y1, X, Y = exponential_lowpass(X0, Y0, alpha, X1_prev, Y1_prev, X_prev, Y_prev)

        if return_previous:
            return X, Y, [X1, Y1, X[-1], Y[-1]]
        return X, Y


def reconstruct_reference_and_quadrature(ts_reference, reference, t0=-0.5, mode="artificial"):  
    fs = 1 / (ts_reference[1] - ts_reference[0])
    Ar, fr = fft_parameters(ts_reference, reference)
    A = np.max([np.max(reference), Ar])

    if mode == "FDF":
        ts_ref, sin_ref = resample(ts_reference, reference, t0)
        sin_ref /= A 
        cos_ref = -np.imag(hilbert(sin_ref)) # shift_90(ts_ref, sin_ref, fr)

    if mode == "artificial":
        ts_ref = ts_reference - t0/fs
        sin_ref = np.sin(2*np.pi*fr * ts_ref)
        cos_ref = np.cos(2*np.pi*fr * ts_ref)

    return sin_ref , cos_ref 

def get_amplitude_and_phase(ts_reference, reference, ts_signal, signal, cut_off=10, ref_mode="FDF", filter_mode="butter", t0=-0.5, previous=None, return_previous=False):
    sin_ref, cos_ref = reconstruct_reference_and_quadrature(ts_reference, reference, t0,  mode=ref_mode)
    
    X0 = 2 * cos_ref * signal 
    Y0 = 2 * sin_ref * signal

    fs = 1 / (ts_reference[1] - ts_reference[0])

    if return_previous:
        X, Y, previous = low_pass_filter(X0, Y0, cut_off, fs, filter_mode, previous=previous, return_previous=return_previous) 
        return np.sqrt(np.array(X)**2 + np.array(Y)**2), np.arctan2(X, Y), previous       

    X, Y = low_pass_filter(X0, Y0, cut_off, fs, filter_mode, previous=previous, return_previous=return_previous) 
    return np.sqrt(np.array(X)**2 + np.array(Y)**2), np.arctan2(X, Y)       

def trim(ts, X, cut_off):
    fs = 1/(ts[1] - ts[1-1])
    freqs = np.fft.rfftfreq(len(ts), d=1/fs)
    Y = np.fft.rfft(X)   

    f_mask = freqs < cut_off
    new_X = np.fft.irfft(Y[f_mask]) * np.sqrt(sum(f_mask)) / np.sqrt(len(Y))  
    new_ts = np.linspace(ts[0], ts[-1], len(new_X)) 
    
    return new_ts, new_X
    

def lock_in(times, volts, cut_off=10, ref_mode="FDF", filter_mode="butter", trim_output=False, plot_results=False, t0=-0.5, previous=None, return_previous=False):
    t_ref, volts_ref = times[0], volts[0]  # referencia: canal 1
    t_signal, volts_signal = times[1], volts[1]  # señal: canal 2

    if return_previous is False:
        A, phi = get_amplitude_and_phase(t_ref, volts_ref, t_signal, volts_signal, cut_off, ref_mode=ref_mode, filter_mode=filter_mode, t0=t0)

        if ref_mode == "artificial": 
            A0, phi0 = get_amplitude_and_phase(t_ref, volts_ref, t_ref, volts_ref, cut_off, ref_mode=ref_mode, filter_mode=filter_mode, t0=0)
            phi -= phi0
    else:
        A, phi, previous = get_amplitude_and_phase(t_ref, volts_ref, t_signal, volts_signal, cut_off, ref_mode=ref_mode, filter_mode=filter_mode, t0=t0, previous=previous, return_previous=return_previous)

        if ref_mode == "artificial":
            A0, phi0, previous0 = get_amplitude_and_phase(t_ref, volts_ref, t_ref, volts_ref, cut_off, ref_mode=ref_mode, filter_mode=filter_mode, t0=0, previous=previous, return_previous=return_previous)
            phi -= phi0
            previous = np.array(previous) - np.array(previous0)

    if trim_output:
        new_ts, A = trim(t_signal, A, cut_off)
        new_ts, phi = trim(t_signal, phi, cut_off)

    if plot_results:
        plt.subplot(211)
        plt.plot(t_signal, A) 
        plt.ylabel("Amplitud [V]")
        plt.subplot(212)
        plt.plot(t_signal, phi)  
        plt.ylabel("Fase [rad]")
        plt.xlabel("Tiempo [s]")
        plt.show()


    if trim_output is False and return_previous is False:
        return (A, phi)
    elif trim_output is False and return_previous:
        return (A, phi, previous)
    elif trim_output and return_previous is False: 
        return (A, phi, new_ts)
    else:
        return (A, phi, new_ts, previous)


if __name__ == "__main__":
    t, v = read("Test data/RLC_attenuation.bin", check_metadata=True)  
    A, phi = lock_in(t, v, cut_off=1000, filter_mode="exponential")
    
    plt.plot(t[0][::500], phi[::500])
    plt.show()
