'''
PCA dot-product based timing (cf. Osłowski 2011)
'''
import numpy as np
from collections import namedtuple

from ..signal import fft_roll
from ..toas import toa_fourier

eps=np.finfo(np.float64).eps
ToaScoreResult = namedtuple('ToaResult', ['toa', 'ampl', 'scores'])

def toa_score(template, pcs, coeffs, profile, dt=1, tol=np.sqrt(eps)):
    '''
    Calculate a maximum-likelihood TOA given a template and a PCA model of pulse shape variations.
    Uses the dot-product based method of Osłowski (2011).

    `pcs`:    The principal components (unit vectors), as rows of an array.
    `coeffs`: Coefficients of principal component dot products to use in correcter.
    `dt`:  The width of each phase bin in the profile. Sets the units of the TOA.
    `tol`:    Relative tolerance for optimization (in bins).
    '''
    n = len(profile)
    k = len(pcs)

    result = toa_fourier(template, profile, dt=dt, tol=tol)
    initial_toa = result.toa
    ampl = result.ampl

    template_shifted = fft_roll(template, initial_toa/dt)
    pcs_shifted = fft_roll(pcs, initial_toa/dt)
    scores = np.dot(pcs_shifted, profile)
    correcter = np.dot(coeffs, scores)
    toa = initial_toa - correcter

    return ToaScoreResult(toa=toa, ampl=ampl, scores=scores)
