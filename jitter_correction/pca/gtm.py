'''
Generalized template matching
'''
import numpy as np
from numpy.fft import fft, rfft, irfft, fftfreq
from scipy.optimize import minimize_scalar
from collections import namedtuple

from ..signal import fft_roll
from ..toas import toa_fourier

eps=np.finfo(np.float64).eps
ToaGtmResult = namedtuple('ToaResult', ['toa', 'ampl', 'scores'])

def toa_gtm(template, pcs, profile, dt=1, tol=np.sqrt(eps)):
    '''
    Calculate a maximum-likelihood TOA given a template and a PCA model of pulse shape variations.

    `pcs`: The principal components (unit vectors), as rows of an array.
    `dt`:  The width of each phase bin in the profile. Sets the units of the TOA.
    `tol`: Relative tolerance for optimization (in bins).
    '''
    n = len(profile)
    k = len(pcs)

    template_fft = fft(template)
    profile_fft = fft(profile)
    pcs_fft = fft(pcs)
    phase_per_bin = -2j*np.pi*fftfreq(n)

    circular_ccf = irfft(rfft(profile)*np.conj(rfft(template)), n)*dt
    sq_ccf = circular_ccf**2 / (np.sum(template**2)*dt)
    for i in range(k):
        pcfft = irfft(rfft(profile)*np.conj(rfft(pcs[i])))*np.sqrt(dt)
        sq_ccf += pcfft**2

    ccf_argmax = np.argmax(sq_ccf)
    ccf_max_val = sq_ccf[ccf_argmax]
    ccf_max = ccf_argmax*dt
    if ccf_argmax > n/2:
        ccf_max -= n*dt

    def modified_squared_ccf(tau):
        phase = phase_per_bin*tau/dt
        ccf = np.inner(profile_fft, np.exp(-phase)*np.conj(template_fft))*dt/n
        sq_ccf = ccf.real**2 / (np.sum(template**2)*dt)

        for i in range(k):
            pc_fft = pcs_fft[i]
            pccf = np.inner(profile_fft, np.exp(-phase)*np.conj(pc_fft))*np.sqrt(dt)/n
            sq_ccf += pccf.real**2

        return sq_ccf

    brack = (ccf_max - dt, ccf_max, ccf_max + dt)
    toa = minimize_scalar(lambda tau: -modified_squared_ccf(tau),
                          method = 'Brent', bracket = brack, tol = tol*dt).x

    assert brack[0] < toa < brack[-1]

    template_shifted = fft_roll(template, toa/dt)
    b = np.dot(template_shifted, profile)/np.dot(template, template)
    ampl = b*np.max(template_shifted)

    pcs_shifted = fft_roll(pcs, toa/dt)
    scores = np.dot(pcs_shifted, profile)

    return ToaGtmResult(toa=toa, ampl=ampl, scores=scores)

def toa_gtm_prior(template, pcs, weights, profile, dt=1, tol=np.sqrt(eps)):
    '''
    Calculate a maximum-likelihood TOA given a template and a PCA model of pulse shape variations.

    `pcs`: The principal components (unit vectors), as rows of an array.
    `dt`:  The width of each phase bin in the profile. Sets the units of the TOA.
    `tol`: Relative tolerance for optimization (in bins).
    '''
    n = len(profile)
    k = len(pcs)

    template_fft = fft(template)
    profile_fft = fft(profile)
    pcs_fft = fft(pcs)
    phase_per_bin = -2j*np.pi*fftfreq(n)

    circular_ccf = irfft(rfft(profile)*np.conj(rfft(template)), n)*dt
    sq_ccf = circular_ccf**2 / (np.sum(template**2)*dt)
    for i in range(k):
        pcfft = irfft(rfft(profile)*np.conj(rfft(pcs[i])))*np.sqrt(dt)
        sq_ccf += pcfft**2 / (1 + weights[i])

    ccf_argmax = np.argmax(sq_ccf)
    ccf_max_val = sq_ccf[ccf_argmax]
    ccf_max = ccf_argmax*dt
    if ccf_argmax > n/2:
        ccf_max -= n*dt

    def modified_squared_ccf(tau):
        phase = phase_per_bin*tau/dt
        ccf = np.inner(profile_fft, np.exp(-phase)*np.conj(template_fft))*dt/n
        sq_ccf = ccf.real**2 / (np.sum(template**2)*dt)

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

    template_shifted = fft_roll(template, toa/dt)
    b = np.dot(template_shifted, profile)/np.dot(template, template)
    ampl = b*np.max(template_shifted)

    pcs_shifted = fft_roll(pcs, toa/dt)
    scores = np.dot(pcs_shifted, profile)

    return ToaPcaResult(toa=toa, ampl=ampl, scores=scores)
