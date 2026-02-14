import numpy as np
import numba as nb
from scipy.optimize import minimize, minimize_scalar
from loguru import logger
from dataclasses import dataclass
from typing import NamedTuple
from functools import partial

from ..mixins import NpzSerializable, Hdf5Serializable, RecordContainer

class ToaGtmResult(NamedTuple):
    toa: np.floating
    ampl: np.floating
    offset: np.floating
    scores: np.array
    sigma: np.floating
    toa_error: np.floating
    ampl_error: np.floating
    offset_error: np.floating
    score_errors: np.array

@dataclass
class ToaGtmResults(NpzSerializable, Hdf5Serializable, RecordContainer[ToaGtmResult]):
    '''
    Represents the result of fitting for TOAs for several profiles.
    '''
    pass

class GtmEstimator:
    def __init__(self, model):
        n = model.template.shape[0]
        k = model.pcs.shape[0]

        template_fft = np.fft.rfft(model.template)
        self.template_fft = template_fft
        pcs_fft = np.fft.rfft(model.pcs)
        self.pcs_fft = pcs_fft
        phase_gradient = -2j*np.pi*np.fft.rfftfreq(n)

        template_sum = template_fft[0].real
        self.template_sum = template_sum
        template_sqsum = 2*np.real(np.trapezoid(np.abs(template_fft)**2))/n
        self.template_sqsum = template_sqsum

        @nb.njit
        def objective_function(profile_fft, tau):
            profile_sum = profile_fft[0].real
            phase = phase_gradient*tau
            ccf_fft = np.conj(np.exp(phase)*template_fft)*profile_fft
            ccf = 2*np.real(np.trapezoid(ccf_fft))/n
            obj = (ccf - profile_sum*template_sum/n)**2
            obj /= (template_sqsum - template_sum**2/n)

            for pc_fft in pcs_fft:
                pccf_fft = np.conj(np.exp(phase)*pc_fft)*profile_fft
                pccf = 2*np.real(np.trapezoid(pccf_fft))/n
                obj += pccf**2

            return obj

        self.objective_function = objective_function

    def get_objective_function(self, profile, vectorize=True):
        profile_fft = np.fft.rfft(profile)
        objective_function = self.objective_function

        def objective_for_profile(tau):
            return objective_function(profile_fft, tau)

        if vectorize:
            objective_for_profile = nb.vectorize(objective_for_profile)

        return objective_for_profile

    def sample_objective_function(self, profile):
        n = profile.shape[0]
        k = self.pcs_fft.shape[0]

        profile_fft = np.fft.rfft(profile)
        profile_sum = profile_fft[0].real

        ccf = np.fft.irfft(profile_fft*np.conj(self.template_fft), n)
        obj = (ccf - profile_sum*self.template_sum/n)**2
        obj /= (self.template_sqsum - self.template_sum**2/n)

        for i in range(k):
            pccf = np.fft.irfft(profile_fft*np.conj(self.pcs_fft[i]), n)
            obj += pccf**2

        return obj

    def maximize_objective_function(self, profile, tol):
        n = profile.shape[0]

        objective_fn = self.get_objective_function(profile, vectorize=False)
        objective_fn_samples = self.sample_objective_function(profile)

        sample_argmax = np.argmax(objective_fn_samples)
        if sample_argmax > n/2:
            sample_argmax -= n
        bracket = (sample_argmax - 1, sample_argmax, sample_argmax + 1)

        result = minimize_scalar(
            lambda tau: -objective_fn(tau),
            method = 'Brent',
            bracket = bracket,
            tol = tol,
        )
        if not result.success:
            logger.warning(result.message)
        return result.x

    def build_toa_result(self, profile, tauhat):
        n = profile.shape[0]

        # calculate best-fit values of a, b, and x_i
        profile_fft = np.fft.rfft(profile)
        profile_sum = profile_fft[0].real

        phase = -2j*np.pi*np.fft.rfftfreq(n)
        ccf_fft = np.conj(np.exp(phase)*self.template_fft)*profile_fft
        ccf = 2*np.real(np.trapezoid(ccf_fft))/n

        ahat = (ccf - profile_sum*self.template_sum/n)
        ahat /= (self.template_sqsum - self.template_sum**2/n)
        bhat = (profile_sum - ahat*self.template_sum)/n

        xhats = []
        for pc_fft in self.pcs_fft:
            pccf_fft = np.conj(np.exp(phase)*pc_fft)*profile_fft
            pccf = 2*np.real(np.trapezoid(pccf_fft))/n
            xhats.append(pccf/ahat)
        xhats = np.array(xhats)

        # estimate noise level from upper 1/4 of profile FFT
        sigma2hat = np.mean(np.abs(profile_fft[-n//8-1:-1])**2)/n
        sigmahat = np.sqrt(sigma2hat)

        # calculate errors in tau (by finite difference), a, and b
        h = np.finfo(np.float64).eps**(1/4)
        obj0 = self.objective_function(profile_fft, tauhat)
        obj1 = self.objective_function(profile_fft, tauhat - h)
        obj2 = self.objective_function(profile_fft, tauhat + h)
        obj_dderiv = (obj1 + obj2 - 2*obj0)/h**2
        tau_error = sigmahat*np.sqrt(-2/obj_dderiv)
        a_error = sigmahat/np.sqrt(self.template_sqsum - self.template_sum**2/n)
        b_error = sigmahat/np.sqrt(n)
        x_errors = sigmahat/ahat*np.ones_like(xhats)

        return ToaGtmResult(
            toa=tauhat,
            ampl=ahat,
            offset=bhat,
            scores=xhats,
            sigma=sigmahat,
            toa_error=tau_error,
            ampl_error=a_error,
            offset_error=b_error,
            score_errors=x_errors,
        )

    def estimate_toa(self, profile, tol=np.sqrt(np.finfo(np.float64).eps)):
        tauhat = self.maximize_objective_function(profile, tol)
        result = self.build_toa_result(profile, tauhat)

        return result

    def estimate_toas(self, data, tol=np.sqrt(np.finfo(np.float64).eps)):
        results = []
        for profile in data.profiles:
            tauhat = self.maximize_objective_function(profile, tol)
            result = self.build_toa_result(profile, tauhat)
            results.append(result)

        n_pcs = self.pcs_fft.shape[0]
        records = np.rec.fromrecords(
            results,
            dtype=[
                ('toa', np.float64),
                ('ampl', np.float64),
                ('offset', np.float64),
                ('scores', np.float64, (n_pcs,)),
                ('sigma', np.float64),
                ('toa_error', np.float64),
                ('ampl_error', np.float64),
                ('offset_error', np.float64),
                ('score_errors', np.float64, (n_pcs,)),
            ],
        )

        return ToaGtmResults(records)

class MapEstimator:
    def __init__(self, model):
        n = model.template.shape[0]
        k = model.pcs.shape[0]

        template_fft = np.fft.rfft(model.template)
        self.template_fft = template_fft
        pcs_fft = np.fft.rfft(model.pcs)
        self.pcs_fft = pcs_fft
        eigvals = model.eigvals
        self.eigvals = eigvals
        phase_gradient = -2j*np.pi*np.fft.rfftfreq(n)

        template_sum = template_fft[0].real
        self.template_sum = template_sum
        template_sqsum = 2*np.real(np.trapezoid(np.abs(template_fft)**2))/n
        self.template_sqsum = template_sqsum

        @nb.njit
        def ahat_gtm(profile_fft, tau):
            profile_sum = profile_fft[0].real
            phase = phase_gradient*tau
            ccf_fft = np.conj(np.exp(phase)*template_fft)*profile_fft
            ccf = 2*np.real(np.trapezoid(ccf_fft))/n

            ahat = (ccf - profile_sum*template_sum/n)
            ahat /= (template_sqsum - template_sum**2/n)

            return ahat

        self.ahat_gtm = ahat_gtm

        @nb.njit
        def objective_function(profile_fft, sigma, a, tau):
            profile_sum = profile_fft[0].real
            phase = phase_gradient*tau
            ccf_fft = np.conj(np.exp(phase)*template_fft)*profile_fft
            ccf = 2*np.real(np.trapezoid(ccf_fft))/n
            obj = (ccf - profile_sum*template_sum/n)**2
            obj /= (template_sqsum - template_sum**2/n)

            for pc_fft, eigval in zip(pcs_fft, eigvals):
                pccf_fft = np.conj(np.exp(phase)*pc_fft)*profile_fft
                pccf = 2*np.real(np.trapezoid(pccf_fft))/n
                shrinkage_factor = 1/(1 + sigma**2/(eigval*a))
                obj += shrinkage_factor*pccf**2
                obj += sigma**2/n*np.log(2*np.pi*eigval*a**2/n)

            ahat = (ccf - profile_sum*template_sum/n)
            ahat /= (template_sqsum - template_sum**2/n)
            obj -= (template_sqsum - template_sum**2/n)*(a - ahat)**2

            return obj

        self.objective_function = objective_function

    def get_objective_function(self, profile, sigma=None, vectorize=True):
        n = profile.shape[0]

        profile_fft = np.fft.rfft(profile)
        objective_function = self.objective_function

        if sigma is None:
            sigma2 = np.mean(np.abs(profile_fft[-n//8-1:-1])**2)/n
            sigma = np.sqrt(sigma2)

        def objective_for_profile(a, tau):
            return objective_function(profile_fft, sigma, a, tau)

        if vectorize:
            objective_for_profile = nb.vectorize(objective_for_profile)

        return objective_for_profile

    def get_1d_objective_function(self, profile, sigma=None):
        objective_fn = self.get_objective_function(profile, sigma)

        @partial(np.frompyfunc, nin=1, nout=1)
        def oned_objective(tau):
            result = minimize_scalar(
                lambda a: -objective_fn(a, tau),
                method='golden',
            )
            if not result.success:
                logger.error(result.message)
                return np.nan
            return -result.fun

        return oned_objective

    def sample_gtm_objective(self, profile):
        n = profile.shape[0]
        k = self.pcs_fft.shape[0]

        profile_fft = np.fft.rfft(profile)
        profile_sum = profile_fft[0].real

        ccf = np.fft.irfft(profile_fft*np.conj(self.template_fft), n)
        obj = (ccf - profile_sum*self.template_sum/n)**2
        obj /= (self.template_sqsum - self.template_sum**2/n)

        for i in range(k):
            pccf = np.fft.irfft(profile_fft*np.conj(self.pcs_fft[i]), n)
            obj += pccf**2

        return obj

    def maximize_objective_function(self, profile, tol, sigma=None):
        n = profile.shape[0]

        objective_fn = self.get_objective_function(profile, sigma, vectorize=False)
        gtm_objective_samples = self.sample_gtm_objective(profile)

        sample_argmax = np.argmax(gtm_objective_samples)
        if sample_argmax > n/2:
            sample_argmax -= n

        tau_guess = sample_argmax
        a_guess = self.ahat_gtm(tau_guess)

        result = minimize_scalar(
            lambda x: -objective_fn(*x),
            method = 'Powell',
            x0 = (a_guess, tau_guess),
            bounds = ((0, 2*a_guess), (sample_argmax - 1, sample_argmax + 1)),
            tol = tol,
        )
        if not result.success:
            logger.warning(result.message)
        ahat, tauhat = result.x
        return ahat, tauhat
