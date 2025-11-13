import numpy as np
from typing import NamedTuple
from dataclasses import dataclass

from .profile_data import ProfileData
from .toas import get_toas
from .mixins import NpzSerializable, Hdf5Serializable, RecordContainer

eps=np.finfo(np.float64).eps

def skewness_function(profile: np.ndarray) -> np.ndarray:
    '''
    Calculates the skewness function
       K(tau) = <I(t)**2 I(t+tau) - I(t) I(t+tau)**2> / <I(t)**3>.
    K(tau) is antisymmetric.
    Size of K(tau) is 2*size(x)-1.
    Normalization: scaled by third moment, so skewness function is scale free
    (independent of multiplication by scale factor)
    '''
    third_moment = np.sum(profile**3)
    T_plus = np.correlate(profile, profile**2, mode='full')/third_moment
    T_minus= np.correlate(profile**2, profile, mode='full')/third_moment
    skewness = T_plus - T_minus
    return skewness

def skewness_coeff(
    lags: np.ndarray,
    skewness: np.ndarray,
    nlags: int | np.integer = 16,
) -> np.floating:
    '''
    Approximate the coefficient of tau**3 in the expansion of the skewness
    function around the origin by fitting a fifth-degree polynomial to the
    region of width `2*nlags + 1` around the zero-lag bin and taking the
    coefficient of the cubic term.
    '''
    inds, = np.where(lags == 0)
    zero_lag_bin = inds[0]
    sl = slice(zero_lag_bin-nlags, zero_lag_bin+nlags+1)
    coeffs = np.polyfit(lags[sl], skewness[sl], 5)
    return coeffs[2]

def calc_skewness_coeffs(data: ProfileData) -> np.ndarray:
    '''
    Calculate skewness coefficients for a set of profiles.
    '''
    n_profiles, n_bins = data.profiles.shape
    lags = np.empty(2*n_bins-1)
    lags[n_bins-1:] = np.linspace(0, 1, n_bins)
    lags[:n_bins-1] = -np.linspace(0, 1, n_bins)[:0:-1]

    skewness_fns = np.empty((n_profiles, 2*n_bins-1))
    for i, profile in enumerate(data.profiles):
        skewness_fn = skewness_function(profile)
        skewness_fns[i] = skewness_fn
    skewness_coeffs = np.array([
        skewness_coeff(lags, skewness_fn, nlags=129) for skewness_fn in skewness_fns
    ])

    return skewness_coeffs

class ToaSkewnessResult(NamedTuple):
    toa: float | np.floating
    ampl: float | np.floating
    skewness_coeff: float | np.floating

@dataclass(slots=True)
class ToaSkewnessResults(
    NpzSerializable,
    Hdf5Serializable,
    RecordContainer[ToaSkewnessResult],
):
    pass

def get_toas_skewness(
    template: np.ndarray,
    predictor_coeffs: np.ndarray,
    data: ProfileData,
    dt: float | np.floating = 1.,
    tol: float | np.floating = np.sqrt(eps),
) -> ToaSkewnessResults:
    '''
    Calculate a corrected TOA using the skewness model.

    Inputs
    ------
    `template`: The profile model to use for fitting
    `predictor_coeffs`: Coefficients of the skewness to use in correction
    `data`:   Profiles for which to calculate TOAs, as a `ProfileData` object
    `dt`:     The width of each phase bin in the profile. Sets the units of the TOA.
    `tol`:    Relative tolerance for optimization (in bins).
    '''
    initial_results = get_toas(template, data)
    skewness_coeffs = calc_skewness_coeffs(data)
    toa_corrections = np.polyval(predictor_coeffs, skewness_coeffs)
    toas_skewness = initial_results.toa - toa_corrections

    records = np.rec.fromarrays( # type: ignore # TODO
        [
            toas_skewness,
            initial_results.ampl,
            skewness_coeffs,
        ],
        names=ToaSkewnessResult._fields,
    )
    return records
