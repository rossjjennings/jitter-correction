import numpy as np
from collections import namedtuple

from .toas import toa_fourier
from .signal import fft_roll
from .profile_data import ProfileData

def get_template(
    data: ProfileData,
    n_iter: int | np.integer = 1,
) -> np.ndarray:
    '''
    Create a template by iteratively aligning and averaging profiles.
    Starts by averaging the middle 10% of profiles and iterates `n_iter` times,
    each time aligning the pulses using the previous template and averaging
    to create a new template.
    '''
    n = data.n_profiles
    n_5_percent = n//20
    sl = slice(n//2 - n_5_percent, n//2 + n_5_percent)
    template = np.mean(data.profiles[sl], axis=0)
    toas = np.empty(data.n_profiles)

    for i in range(n_iter):
        for i, profile in enumerate(data.profiles):
            result = toa_fourier(template, profile)
            toas[i] = result.toa
        profiles_aligned = np.empty_like(data.profiles)
        for j, profile in enumerate(data.profiles):
            profiles_aligned[j] = fft_roll(profile, -toas[j])

        template = np.mean(profiles_aligned, axis=0)

    return template

def calc_dtoas(
    template: np.ndarray,
    data: ProfileData,
    poly_degree: int | np.integer = 1,
) -> np.ndarray:
    '''
    Calculate ΔTOAs from profiles with polynomial drift.
    '''
    profile_number = np.arange(data.n_profiles)
    toas = np.empty(data.n_profiles)
    for i, profile in enumerate(data.profiles):
        result = toa_fourier(template, profile)
        toas[i] = result.toa
    timing_poly_coeffs = np.polyfit(profile_number, toas, poly_degree)
    timing_poly_vals = np.polyval(timing_poly_coeffs, profile_number)
    dtoas = toas - timing_poly_vals

    return dtoas
