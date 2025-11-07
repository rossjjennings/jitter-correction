import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
from numpy.fft import fft, ifft, fftfreq, rfft, irfft, rfftfreq
from numpy.random import randn
from typing import NamedTuple, Iterator, Self
from scipy.optimize import minimize_scalar
from collections.abc import Callable
from dataclasses import dataclass
import sys

from .signal import fft_roll, rolling_sum, interp_ws
from .profile_data import ProfileData
from .mixins import NpzSerializable, Hdf5Serializable

eps = np.finfo(np.float64).eps
if hasattr(np, "trapezoid"):
    # np.trapz was renamed to np.trapezoid in Numpy 2.0
    trapz = np.trapezoid
else:
    trapz = np.trapz

def offpulse_window(profile: np.ndarray, size: int | np.integer) -> np.ndarray:
    '''
    Find the off-pulse window of a given profile, defined as the
    segment of pulse phase of length `size` (in phase bins)
    minimizing the integral of the pulse profile.
    '''
    bins = np.arange(len(profile))
    lower = np.argmin(rolling_sum(profile, size))
    upper = lower + size
    return np.logical_and(lower <= bins, bins < upper)

def offpulse_rms(profile: np.ndarray, size: int | np.integer) -> np.floating:
    '''
    Calculate the off-pulse RMS of a profile (a measure of noise level).
    This is the RMS of `profile` in the segment of length `size`
    (in phase bins) minimizing the integral of `profile`.
    '''
    opw = offpulse_window(profile, size)
    return np.sqrt(np.mean(profile[opw]**2))

class ToaResult(NamedTuple):
    '''
    Represents the result of fitting for a TOA.
    '''
    toa: float | np.floating
    error: float | np.floating
    ampl: float | np.floating

@dataclass(slots=True)
class ToaResults(NpzSerializable, Hdf5Serializable):
    '''
    Represents the result of fitting for TOAs for several profiles.
    '''
    data: np.recarray

    def __init__(self, data: np.ndarray):
        self.data = np.rec.array(data)

    def __iter__(self) -> Iterator[ToaResult]:
        for rec in self.data:
            yield ToaResult(*rec)

    def __getitem__(self, key) -> ToaResult | Self:
        item = self.data[key]
        if item.shape == ():
            return ToaResult(*item)
        else:
            return ToaResults(item)

    def __getattr__(self, attr):
        return getattr(self.data, attr)

def toa_ws(
    template: np.ndarray,
    profile: np.ndarray,
    dt: float | np.floating = 1.,
    noise_level: float | np.floating | None = None,
    tol: float | np.floating = sqrt(eps),
) -> ToaResult:
    '''
    Calculate a TOA by maximizing the Whittaker-Shannon interpolant of the 
    CCF between `template` and `profile`. Searches within the interval
    between the sample below and the sample above the argmax of the CCF.

    `dt`:  The width of each phase bin in the profile. Sets the units of the TOA.
    `noise_level`: Off-pulse noise, in the same units as the profile.
           Used in calculating error. If not supplied, noise level will be
           estimated as the standard deviation of the profile residual.
    `tol`: Relative tolerance for optimization.
    '''
    n = len(profile)
    lags = np.arange(-len(profile) + 1, len(profile))*dt

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

    w_eff = np.sqrt(n*dt/trapz(np.gradient(template, dt)**2, dx=dt))
    error = w_eff/(snr*sqrt(n))

    return ToaResult(toa=toa, error=error, ampl=ampl)

def toa_fourier(
    template: np.ndarray,
    profile: np.ndarray,
    dt: float | np.floating = 1.,
    noise_level: float | np.floating | None = None,
    tol: float | np.floating = sqrt(eps),
) -> ToaResult:
    '''
    Calculate a TOA by maximizing the CCF of the template and the profile
    in the frequency domain. Searches within the interval between the sample
    below and the sample above the argmax of the circular CCF.

    `dt`:  The width of each phase bin in the profile. Sets the units of the TOA.
    `noise_level`: Off-pulse noise, in the same units as the profile.
           Used in calculating error. If not supplied, noise level will be
           estimated as the standard deviation of the profile residual.
    `tol`: Relative tolerance for optimization (in bins).
    '''
    n = len(profile)

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

    w_eff = np.sqrt(n*dt/trapz(np.gradient(template, dt)**2, dx=dt))
    error = w_eff/(snr*sqrt(n))

    return ToaResult(toa=toa, error=error, ampl=ampl)

def test_toa_recovery(
    func: Callable,
    template: np.ndarray,
    n: int | np.integer,
    rms_toa: float | np.floating,
    snr: float | np.floating = np.inf,
    dt: float | np.floating = 1.,
    tol: float | np.floating = sqrt(eps),
) -> tuple[np.floating, np.floating]:
    '''
    Test function for `toa_ws()` and `toa_fourier()`.
    Attempts to recover `n` TOAs at a given SNR and returns the RMS error.
    
    `func`:     Function to test (`toa_ws` or `toa_fourier`).
    `template`: Template to use. Test profiles will be generated by
                shifting it.
    `n`:        Number of test profiles to generate.
    `rms_toa`:  RMS TOA for test profiles.
    `snr`:      Signal-to-noise ratio of the test profiles
    `dt`:       The width of each phase bin in the profile.
                Sets the units of the TOA.
    `tol`:      Relative tolerance for optimization.
    '''
    dtoas = []
    toa_errs = []
    for i in range(n):
        true_toa = rms_toa*randn()
        profile = fft_roll(template, true_toa/dt)
        if np.isfinite(snr):
            profile += randn(len(profile))/snr
        result = func(template, profile, dt=dt, tol=tol)
        toa_estimate = result.toa
        dtoas.append(toa_estimate-true_toa)
        toa_errs.append(result.error)
    dtoas = np.array(dtoas)
    toa_errs = np.array(toa_errs)
    return np.sqrt(np.mean(dtoas**2)), np.sqrt(np.mean(toa_errs**2))

def get_toas(
    template: np.ndarray,
    data: ProfileData,
    method: str = 'fourier',
    noise_level: float | np.floating | None = None,
    dt: float | np.floating = 1.,
    tol: float | np.floating = sqrt(eps),
) -> np.recarray:
    '''
    Calculate TOAs for a set of profiles.

    `method`: Method used to calculate TOAs. Can be 'fourier' or 'ws'.
           For other methods, see the corresponding functions.
    `noise_level`: Off-pulse noise, in the same units as the profile.
           Used in calculating error. If not supplied, noise level will be
           estimated as the standard deviation of the profile residual.
    `dt`:  The width of each phase bin in the profile.
           Sets the units of the TOA.
    `tol`: Relative tolerance for optimization (in bins).
    '''
    methods = {'fourier': toa_fourier, 'ws': toa_ws}
    func = methods[method]
    results = [
        func(template, profile, dt, noise_level, tol)
        for profile in data.profiles
    ]
    records = np.rec.fromrecords(results, names=ToaResult._fields)
    return ToaResults(records)
