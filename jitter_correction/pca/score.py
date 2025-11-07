'''
PCA dot-product based timing (cf. Osłowski 2011)
'''
import numpy as np
from typing import NamedTuple, Iterator, Self
from dataclasses import dataclass

from ..signal import fft_roll
from ..toas import toa_fourier
from ..mixins import NpzSerializable, Hdf5Serializable

eps=np.finfo(np.float64).eps

class ToaScoreResult(NamedTuple):
    toa: float | np.floating
    ampl: float | np.floating
    scores: np.ndarray

@dataclass(slots=True)
class ToaScoreResults(NpzSerializable, Hdf5Serializable):
    '''
    Represents the result of fitting for TOAs for several profiles.
    '''
    data: np.recarray

    def __init__(self, data: np.ndarray):
        self.data = np.rec.array(data)

    def __iter__(self) -> Iterator[ToaScoreResult]:
        for rec in self.data:
            yield ToaScoreResult(*rec)

    def __getitem__(self, key) -> ToaScoreResult | Self:
        item = self.data[key]
        if item.shape == ():
            return ToaScoreResult(*item)
        else:
            return ToaScoreResults(item)

    def __getattr__(self, attr):
        return getattr(self.data, attr)

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
