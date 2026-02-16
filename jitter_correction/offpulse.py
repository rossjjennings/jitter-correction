import numpy as np

from .signal import rolling_sum

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
