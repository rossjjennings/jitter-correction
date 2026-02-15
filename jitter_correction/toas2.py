import numpy as np
import numba as nb
from scipy.optimize import minimize_scalar
from loguru import logger
from dataclasses import dataclass
from typing import NamedTuple

from .mixins import NpzSerializable, Hdf5Serializable, RecordContainer, RecordType

@dataclass(slots=True, repr=False)
class ToaResult(RecordType):
    toa: np.floating
    ampl: np.floating
    offset: np.floating
    sigma: np.floating
    toa_error: np.floating
    ampl_error: np.floating
    offset_error: np.floating

@dataclass
class ToaResults(NpzSerializable, Hdf5Serializable, RecordContainer[ToaResult]):
    '''
    Represents the result of fitting for TOAs for several profiles.
    '''
    pass
 
class FourierEstimator:
    def __init__(self, template):
        n = template.shape[0]

        template_fft = np.fft.rfft(template)
        self.template_fft = template_fft
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

        profile_fft = np.fft.rfft(profile)
        profile_sum = profile_fft[0].real

        ccf = np.fft.irfft(profile_fft*np.conj(self.template_fft), n)
        obj = (ccf - profile_sum*self.template_sum/n)**2
        obj /= (self.template_sqsum - self.template_sum**2/n)
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

        # calculate best-fit values of a and b
        profile_fft = np.fft.rfft(profile)
        profile_sum = profile_fft[0].real
        profile_sqsum = 2*np.real(np.trapezoid(np.abs(profile_fft)**2))/n

        phase = -2j*np.pi*np.fft.rfftfreq(n)
        ccf_tauhat_fft = np.conj(np.exp(phase)*self.template_fft)*profile_fft
        ccf_tauhat = 2*np.real(np.trapezoid(ccf_tauhat_fft))/n

        ahat = (ccf_tauhat - profile_sum*self.template_sum/n)
        ahat /= (self.template_sqsum - self.template_sum**2/n)
        bhat = (profile_sum - ahat*self.template_sum)/n

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

        return ToaResult(tauhat, ahat, bhat, sigmahat, tau_error, a_error, b_error)

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
    
        records = np.rec.array(np.array(
            [result.as_record() for result in results]
        ))

        return ToaResults(records)
