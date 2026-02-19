import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
from numpy.fft import fft, ifft, fftfreq, rfft, irfft, rfftfreq
from numpy.random import randn
from numpy.exceptions import ComplexWarning
from scipy.special import sinc
from scipy.optimize import brent, curve_fit
import pywt
import warnings
import sys

def fft_roll(a, shift):
    '''
    Roll array by a given (possibly fractional) amount, in bins.
    Works by multiplying the FFT of the input array by exp(-2j*pi*shift*f)
    and Fourier transforming back. The sign convention matches that of 
    numpy.roll() -- positive shift is toward the end of the array.
    This is the reverse of the convention used by pypulse.utils.fftshift().
    If the array has more than one axis, the last axis is shifted.
    '''
    n = a.shape[-1]
    with warnings.catch_warnings():
        warnings.filterwarnings(action='error', category=ComplexWarning)
        try:
            phase = -2j*pi*shift*rfftfreq(n)
            return irfft(rfft(a)*np.exp(phase), n)
        except np.exceptions.ComplexWarning:
            phase = -2j*pi*shift*fftfreq(n)
            filtr = np.exp(phase)
            if n % 2 == 0:
                # Take real part of Nyquist frequency term
                # to match behavior when a is real
                filtr[n//2] = filtr[n//2].real
            return ifft(fft(a)*filtr, n)

def fft_roll_deriv(a, shift=0):
    '''
    Derivative of fft_roll(a, shift) with respect to the shift.
    '''
    n = a.shape[-1]
    phase = -2j*pi*rfftfreq(n)
    return irfft(phase*rfft(a)*np.exp(shift*phase), n)

def interp_ws(signal, ts = None):
    '''
    Calculate the Whittaker-Shannon interpolant of a signal.
    Returns a function computing the interpolant at a point `t`.
    `ts` is the array of sample points (assumed evenly-spaced).
    If `ts` is left unspecified, `arange(len(signal))` is used.
    '''
    if ts is None:
        ts = np.arange(len(signal))
    dt = ts[1] - ts[0]
    
    def interpolant(t):
        return np.sum(signal*sinc((t - ts)/dt))
    
    return interpolant

def eval_sin(t, amp, freq, phase, offset, cov=None):
    '''
    Evaluate a sine function with arbitrary amplitude, frequency,
    phase, and offset from zero. Primarily useful in conjunction with
    fit_sin() below. For example, to sample a sine curve fitted to
    a time series (t0, x0) at points t, can use 
    `eval_sin(t, **fit_sin(t0, x0))`.
    '''
    return amp * np.sin(2*pi*freq*t - phase) + offset

def fit_sin(t, x, return_cov=False, **kwargs):
    '''
    Fit a sine curve to a time series (t, x), using the frequency maximizing
    the FFT-based power spectrum as an initial guess for the frequency. 
    Returns the amplitude, frequency, phase, and offset of the optimal curve,
    along with the corresponding covariance matrix (if `return_cov` is `True`),
    in a dictionary, which can be used to sample it using eval_sin().
    '''
    freqs = np.fft.fftfreq(len(t), (t[1]-t[0]))   # assume uniform spacing
    fft = np.fft.fft(x)
    powspec = fft*np.conj(fft)
    guess_freq = freqs[1 + np.argmax(powspec[1:])]
    guess_amp = np.std(x) * np.sqrt(2)
    guess_offset = np.mean(x)
    guess_params = np.array([guess_amp, guess_freq, 0., guess_offset])

    (amp, freq, phase, offset), cov = curve_fit(eval_sin, t, x, p0=guess_params, **kwargs)
    params = {'amp': amp, 'freq': freq, 'phase': phase, 'offset': offset}
    if return_cov: params['cov'] = cov
    return params

def periodic_sinc(n, x):
    '''
    Calculate the "periodic sinc function": the Fourier transform of
    a windowed Dirac comb. Convolving this with a Nyquist sampled 
    periodic function gives back the original periodic function.
    The sample rate is `n` per period. This function should broadcast
    over array `x` values, and returns 0-dimensional arrays on scalars.
    '''
    if n % 2 == 0:
        return np.piecewise(x, [x % n == 0, x % n != 0], [1, lambda u: sin(pi*u)/(n*tan(pi*u/n))])
    else:
        return np.piecewise(x, [x % n == 0, x % n != 0], [1, lambda u: sin(pi*u)/(n*sin(pi*u/n))])

def interp_sinc(signal, ts = None):
    '''
    Calculate the periodic Whittaker-Shannon (sinc) interpolant of a signal.
    Returns a function computing the interpolant at a point `t`.
    `ts` is the array of sample points (assumed evenly-spaced).
    If `ts` is left unspecified, `arange(len(signal))` is used.
    '''
    n = len(signal)
    if ts is None:
        ts = np.arange(n)
    dt = ts[1] - ts[0]
    
    def interpolant(t):
        return np.sum(signal*periodic_sinc(n, (t - ts)/dt))
    
    return interpolant

def rolling_sum(arr, size):
    '''
    Calculate the sum of values in `arr` in a sliding window of length `size`,
    wrapping around at the end of the array.
    '''
    n = len(arr)
    s = np.cumsum(arr)
    return np.array([s[(i+size)%n]-s[i]+(i+size)//n*s[-1] for i in range(n)])

# Modified from code written by E. Fonseca for PulsePortraiture
def wavelet_smooth(prof, wavelet='db8', nlevel=5, threshtype='hard', fact=1.0):
    nbin = prof.shape[-1]
    # Translation-invariant (stationary) wavelet transform/denoising
    coeffs = np.array(pywt.swt(prof, wavelet, level=nlevel, start_level=0, axis=-1))
    # Get threshold value
    lopt = fact * (np.median(np.abs(coeffs[0])) / 0.6745) * np.sqrt(2 * np.log(nbin))
    # Do wavelet thresholding
    coeffs = pywt.threshold(coeffs, lopt, mode=threshtype, substitute=0.0)
    # Reconstruct data
    smooth_prof = pywt.iswt(list(map(tuple, coeffs)), wavelet)
    return smooth_prof
