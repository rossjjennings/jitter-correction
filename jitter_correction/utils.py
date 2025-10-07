import numpy as np
from collections import namedtuple

from .toas import toa_fourier
from .signal import fft_roll
from .profile_data import gen_profiles

def gen_data(spec, n_profiles, npprof, n_bins, SNR, drift_bins):
    '''
    Generated simulated data based on a pulse specification.
    '''
    phase = np.linspace(-1/2, 1/2, n_bins, endpoint=False)
    profiles = gen_profiles(phase, spec=spec, n_profiles=n_profiles, npprof=npprof, SNR=SNR)
    
    shifts = drift_bins/n_profiles*np.arange(n_profiles)
    shifts -= np.mean(shifts)
    for i, profile in enumerate(profiles):
        profiles[i] = fft_roll(profile, shifts[i])
    
    return phase, profiles

def get_template(profiles, n_iter=1):
    '''
    Create a template by iteratively aligning and averaging profiles.
    Starts by averaging the middle 10% of profiles and iterates `n_iter` times,
    each time aligning the pulses using the previous template and averaging
    to create a new template.
    '''
    n = profiles.shape[0]
    n_5_percent = n//20
    sl = slice(n//2 - n_5_percent, n//2 + n_5_percent)
    template = np.mean(profiles[sl], axis=0)
    toas = np.empty(profiles.shape[0])
    
    for i in range(n_iter):
        for i, profile in enumerate(profiles):
            result = toa_fourier(template, profile)
            toas[i] = result.toa
        profiles_aligned = np.empty_like(profiles)
        for j, profile in enumerate(profiles):
            profiles_aligned[j] = fft_roll(profile, -toas[j])

        template = np.mean(profiles_aligned, axis=0)
    
    return template

def calc_dtoas(template, profiles, poly_degree=1):
    '''
    Calculate ΔTOAs from profiles with polynomial drift.
    '''
    n_profiles = profiles.shape[0]
    profile_number = np.arange(n_profiles)
    toas = np.empty(n_profiles)
    for i, profile in enumerate(profiles):
        result = toa_fourier(template, profile)
        toas[i] = result.toa
    timing_poly_coeffs = np.polyfit(profile_number, toas, poly_degree)
    timing_poly_vals = np.polyval(timing_poly_coeffs, profile_number)
    dtoas = toas - timing_poly_vals
    
    return dtoas
