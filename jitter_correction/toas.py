import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
from numpy.fft import fft, ifft, fftfreq, rfft, irfft, rfftfreq
from numpy.random import randn
from collections import namedtuple
from scipy.optimize import minimize_scalar
import sys

from .signal import fft_roll, rolling_sum, interp_ws

eps = sys.float_info.epsilon
if hasattr(np, "trapezoid"):
    # np.trapz was renamed to np.trapezoid in Numpy 2.0
    trapz = np.trapezoid
else:
    trapz = np.trapz

def offpulse_window(profile, size):
    '''
    Find the off-pulse window of a given profile, defined as the
    segment of pulse phase of length `size` (in phase bins)
    minimizing the integral of the pulse profile.
    '''
    bins = np.arange(len(profile))
    lower = np.argmin(rolling_sum(profile, size))
    upper = lower + size
    return np.logical_and(lower <= bins, bins < upper)

def offpulse_rms(profile, size):
    '''
    Calculate the off-pulse RMS of a profile (a measure of noise level).
    This is the RMS of `profile` in the segment of length `size`
    (in phase bins) minimizing the integral of `profile`.
    '''
    opw = offpulse_window(profile, size)
    return np.sqrt(np.mean(profile[opw]**2))

ToaResult = namedtuple('ToaResult', ['toa', 'error', 'ampl'])

def toa_ws(template, profile, ts = None, noise_level = None, tol = sqrt(eps)):
    '''
    Calculate a TOA by maximizing the Whittaker-Shannon interpolant of the 
    CCF between `template` and `profile`. Searches within the interval
    between the sample below and the sample above the argmax of the CCF.
    
    `ts`:  Evenly-spaced array of phase values corresponding to the profile.
           Sets the units of the TOA. If this is `None`, the TOA is reported
           in bins.
    `tol`: Relative tolerance for optimization.
    `noise_level`: Off-pulse noise, in the same units as the profile.
           Used in calculating error. If not supplied, noise level will be
           estimated as the standard deviation of the profile residual.
    '''
    n = len(profile)
    if ts is None:
        ts = np.arange(n)
    dt = ts[1] - ts[0]
    lags = np.arange(-len(ts) + 1, len(ts))*dt
    
    ccf = np.correlate(profile, template, mode = 'full')
    ccf_max = lags[np.argmax(ccf)]
    
    interpolant = interp_ws(ccf, lags)
    brack = (ccf_max - dt, ccf_max, ccf_max + dt)
    toa = minimize_scalar(lambda t: -interpolant(t),
                          method = 'Brent', bracket = brack, tol = tol).x
    
    assert brack[0] < toa < brack[-1]
    
    template_shifted = fft_roll(template, toa/dt)
    b = np.dot(template_shifted, profile)/np.dot(template, template)
    residual = profile - b*template_shifted
    ampl = b*np.max(template_shifted)
    if noise_level is None:
        noise_level = offpulse_rms(profile, profile.size//4)
    snr = ampl/noise_level
    
    w_eff = np.sqrt(n*dt/trapz(np.gradient(template, ts)**2, ts))
    error = w_eff/(snr*sqrt(n))
    
    return ToaResult(toa=toa, error=error, ampl=ampl)

def toa_fourier(template, profile, ts = None, noise_level = None, tol = sqrt(eps)):
    '''
    Calculate a TOA by maximizing the CCF of the template and the profile
    in the frequency domain. Searches within the interval between the sample
    below and the sample above the argmax of the circular CCF.
    
    `ts`:  Evenly-spaced array of phase values corresponding to the profile.
           Sets the units of the TOA. If this is `None`, the TOA is reported 
           in bins.
    `tol`: Relative tolerance for optimization (in bins).
    `noise_level`: Off-pulse noise, in the same units as the profile.
           Used in calculating error. If not supplied, noise level will be
           estimated as the standard deviation of the profile residual.
    '''
    n = len(profile)
    if ts is None:
        ts = np.arange(n)
    dt = float(ts[1] - ts[0])
    
    template_fft = fft(template)
    profile_fft = fft(profile)
    phase_per_bin = -2j*pi*fftfreq(n)
    
    circular_ccf = irfft(rfft(profile)*np.conj(rfft(template)), n)
    ccf_argmax = np.argmax(circular_ccf)
    if ccf_argmax > n/2:
        ccf_argmax -= n
    ccf_max = ccf_argmax*dt
    
    def ccf_fourier(tau):
        phase = phase_per_bin*tau/dt
        ccf = np.inner(profile_fft, exp(-phase)*np.conj(template_fft))/n
        return ccf.real
    
    brack = (ccf_max - dt, ccf_max, ccf_max + dt)
    toa = minimize_scalar(lambda tau: -ccf_fourier(tau),
                          method = 'Brent', bracket = brack, tol = tol*dt).x
    
    assert brack[0] < toa < brack[-1]
    
    template_shifted = fft_roll(template, toa/dt)
    b = np.dot(template_shifted, profile)/np.dot(template, template)
    residual = profile - b*template_shifted
    ampl = b*np.max(template_shifted)
    if noise_level is None:
        noise_level = offpulse_rms(profile, profile.size//4)
    snr = ampl/noise_level
    
    w_eff = np.sqrt(n*dt/trapz(np.gradient(template, ts)**2, ts))
    error = w_eff/(snr*sqrt(n))
    
    return ToaResult(toa=toa, error=error, ampl=ampl)

def test_toa_recovery(func, template, n, rms_toa, SNR=np.inf, ts=None,
                      tol=sqrt(eps)):
    '''
    Test function for `toa_ws()` and `toa_fourier()`.
    Attempts to recover `n` TOAs at a given SNR and returns the RMS error.
    
    `func`:     Function to test (`toa_ws` or `toa_fourier`).
    `template`: Template to use. Test profiles will be generated by
                shifting it.
    `n`:        Number of test profiles to generate.
    `rms_toa`:  RMS TOA for test profiles.
    `tol`:      Relative tolerance for optimization.
    '''
    if ts is None:
        ts = np.arange(len(template))
    dt = ts[1] - ts[0]
    dtoas = []
    toa_errs = []
    for i in range(n):
        true_toa = rms_toa*randn()
        profile = fft_roll(template, true_toa/dt)
        if np.isfinite(SNR):
            profile += randn(len(profile))/SNR
        result = func(template, profile, ts=ts, tol=tol)
        toa_estimate = result.toa
        dtoas.append(toa_estimate-true_toa)
        toa_errs.append(result.error)
    dtoas = np.array(dtoas)
    toa_errs = np.array(toa_errs)
    return np.sqrt(np.mean(dtoas**2)), np.sqrt(np.mean(toa_errs**2))
