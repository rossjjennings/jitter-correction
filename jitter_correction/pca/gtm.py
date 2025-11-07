'''
Generalized template matching
'''
import numpy as np
from numpy.fft import fft, rfft, irfft, fftfreq
from scipy.optimize import minimize_scalar
from typing import NamedTuple, Iterator, Self
from dataclasses import dataclass

from ..signal import fft_roll
from ..toas import toa_fourier
from ..profile_data import ProfileData
from ..mixins import NpzSerializable, Hdf5Serializable, RecordContainer
from .pcs import PrincipalComponentModel

eps=np.finfo(np.float64).eps

class ToaGtmResult(NamedTuple):
    '''
    The result of fitting for a TOA using generalized template matching,
    including the principal component scores.
    '''
    toa: float | np.floating
    ampl: float | np.floating
    scores: np.ndarray

@dataclass(slots=True)
class ToaGtmResults(NpzSerializable, Hdf5Serializable, metaclass=RecordContainer):
    '''
    Represents the result of fitting for TOAs for several profiles.
    '''
    data: np.recarray
    record_type = ToaGtmResult

def toa_gtm(
    model: PrincipalComponentModel,
    profile: np.ndarray,
    dt: float | np.floating = 1.,
    tol: float | np.floating = np.sqrt(eps),
) -> ToaGtmResult:
    '''
    Calculate a maximum-likelihood TOA given a template and a PCA model of pulse shape variations.

    `model`: The principal model to use for fitting the TOA.
    `dt`:  The width of each phase bin in the profile. Sets the units of the TOA.
    `tol`: Relative tolerance for optimization (in bins).
    '''
    n = len(profile)
    k = len(model.pcs)

    template_fft = fft(model.template)
    profile_fft = fft(profile)
    pcs_fft = fft(model.pcs)
    phase_per_bin = -2j*np.pi*fftfreq(n)

    circular_ccf = irfft(rfft(profile)*np.conj(rfft(model.template)), n)*dt
    sq_ccf = circular_ccf**2 / (np.sum(model.template**2)*dt)
    for i in range(k):
        pcfft = irfft(rfft(profile)*np.conj(rfft(model.pcs[i])))*np.sqrt(dt)
        sq_ccf += pcfft**2

    ccf_argmax = np.argmax(sq_ccf)
    ccf_max_val = sq_ccf[ccf_argmax]
    ccf_max = ccf_argmax*dt
    if ccf_argmax > n/2:
        ccf_max -= n*dt

    def modified_squared_ccf(tau):
        phase = phase_per_bin*tau/dt
        ccf = np.inner(profile_fft, np.exp(-phase)*np.conj(template_fft))*dt/n
        sq_ccf = ccf.real**2 / (np.sum(model.template**2)*dt)

        for i in range(k):
            pc_fft = pcs_fft[i]
            pccf = np.inner(profile_fft, np.exp(-phase)*np.conj(pc_fft))*np.sqrt(dt)/n
            sq_ccf += pccf.real**2

        return sq_ccf

    brack = (ccf_max - dt, ccf_max, ccf_max + dt)
    toa = minimize_scalar(lambda tau: -modified_squared_ccf(tau),
                          method = 'Brent', bracket = brack, tol = tol*dt).x

    assert brack[0] < toa < brack[-1]

    template_shifted = fft_roll(model.template, toa/dt)
    b = np.dot(template_shifted, profile)/np.dot(model.template, model.template)
    ampl = b*np.max(template_shifted)

    pcs_shifted = fft_roll(model.pcs, toa/dt)
    scores = np.dot(pcs_shifted, profile)

    return ToaGtmResult(toa=toa, ampl=ampl, scores=scores)

def toa_gtm_prior(
    model: PrincipalComponentModel,
    weights: np.ndarray,
    profile: np.ndarray,
    tol: float | np.floating = np.sqrt(eps),
) -> ToaGtmResult:
    '''
    Calculate a maximum-likelihood TOA given a template and a PCA model of pulse shape variations.

    `pcs`: The principal components (unit vectors), as rows of an array.
    `dt`:  The width of each phase bin in the profile. Sets the units of the TOA.
    `tol`: Relative tolerance for optimization (in bins).
    '''
    n = len(profile)
    k = len(model.pcs)

    template_fft = fft(model.template)
    profile_fft = fft(profile)
    pcs_fft = fft(model.pcs)
    phase_per_bin = -2j*np.pi*fftfreq(n)

    circular_ccf = irfft(rfft(profile)*np.conj(rfft(model.template)), n)*dt
    sq_ccf = circular_ccf**2 / (np.sum(model.template**2)*dt)
    for i in range(k):
        pcfft = irfft(rfft(profile)*np.conj(rfft(model.pcs[i])))*np.sqrt(dt)
        sq_ccf += pcfft**2 / (1 + weights[i])

    ccf_argmax = np.argmax(sq_ccf)
    ccf_max_val = sq_ccf[ccf_argmax]
    ccf_max = ccf_argmax*dt
    if ccf_argmax > n/2:
        ccf_max -= n*dt

    def modified_squared_ccf(tau):
        phase = phase_per_bin*tau/dt
        ccf = np.inner(profile_fft, np.exp(-phase)*np.conj(template_fft))*dt/n
        sq_ccf = ccf.real**2 / (np.sum(model.template**2)*dt)

        for i in range(k):
            pc_fft = pcs_fft[i]
            weight = weights[i]
            pccf = np.inner(profile_fft, np.exp(-phase)*np.conj(pc_fft))*np.sqrt(dt)/n
            sq_ccf += pccf.real**2 / (1 + weight)

        return sq_ccf

    brack = (ccf_max - dt, ccf_max, ccf_max + dt)
    toa = minimize_scalar(lambda tau: -modified_squared_ccf(tau),
                          method = 'Brent', bracket = brack, tol = tol*dt).x

    assert brack[0] < toa < brack[-1]

    template_shifted = fft_roll(model.template, toa/dt)
    b = np.dot(template_shifted, profile)/np.dot(model.template, model.template)
    ampl = b*np.max(template_shifted)

    pcs_shifted = fft_roll(model.pcs, toa/dt)
    scores = np.dot(pcs_shifted, profile)

    return ToaGtmResult(toa=toa, ampl=ampl, scores=scores)

def get_toas_gtm(
    model: PrincipalComponentModel,
    data: ProfileData,
    dt: float | np.floating = 1.,
    tol: float | np.floating = np.sqrt(eps),
) -> ToaGtmResults:
    '''
    Calculate TOAs for a set of profiles using generalized template matching.

    `model`:  The principal components model, including template and PCs.
    `data`:   Profile data from which to compute TOAs.
    `dt`:     The width of each phase bin in the profile. Sets the units of the TOA.
    `tol`:    Relative tolerance for optimization (in bins).
    '''
    n_pcs = len(model.pcs)
    results = [
        toa_gtm(model, profile, dt, tol)
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
    return ToaGtmResults(records)
