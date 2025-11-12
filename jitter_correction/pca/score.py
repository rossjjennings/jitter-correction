'''
PCA dot-product based timing (cf. Osłowski 2011)
'''
import numpy as np
from typing import NamedTuple, Iterator, Self
from dataclasses import dataclass

from ..signal import fft_roll
from ..toas import toa_fourier
from ..profile_data import ProfileData
from ..mixins import NpzSerializable, Hdf5Serializable, RecordContainer
from .pcs import PrincipalComponentModel

eps=np.finfo(np.float64).eps

class ToaScoreResult(NamedTuple):
    toa: float | np.floating
    ampl: float | np.floating
    scores: np.ndarray

@dataclass(slots=True)
class ToaScoreResults(NpzSerializable, Hdf5Serializable, RecordContainer[ToaScoreResult]):
    '''
    Represents the result of fitting for TOAs for several profiles.
    '''
    pass

def toa_score(
    model: PrincipalComponentModel,
    coeffs: np.ndarray,
    profile: np.ndarray,
    dt: float | np.floating = 1.,
    tol: float | np.floating = np.sqrt(eps),
) -> ToaScoreResult:
    '''
    Calculate a maximum-likelihood TOA given a template and a PCA model of pulse shape variations.
    Uses the dot-product based method of Osłowski (2011).

    `model`:  The principal components model, including template and PCs.
    `coeffs`: Coefficients of principal component dot products to use in correction.
    `dt`:     The width of each phase bin in the profile. Sets the units of the TOA.
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

def get_toas_score(
    model: PrincipalComponentModel,
    coeffs: np.ndarray,
    data: ProfileData,
    dt: float | np.floating = 1.,
    tol: float | np.floating = np.sqrt(eps),
) -> ToaScoreResults:
    '''
    Calculate TOAs for a set of profiles using the PCA score method.

    `model`:  The principal components model, including template and PCs.
    `coeffs`: Coefficients of principal component dot products to use in correction.
    `data`:   Profile data from which to compute TOAs.
    `dt`:     The width of each phase bin in the profile. Sets the units of the TOA.
    `tol`:    Relative tolerance for optimization (in bins).
    '''
    n_pcs = len(model.pcs)
    results = [
        toa_score(model, coeffs, profile, dt, tol)
        for profile in data.profiles
    ]
    records = np.rec.fromrecords(
        results,
        dtype = [
            ('toa', np.float64),
            ('ampl', np.float64),
            ('scores', np.float64, (n_pcs,)),
        ],
    )
    return ToaScoreResults(records)
