import numpy as np
import numba as nb
from scipy.optimize import minimize_scalar
from loguru import logger
from dataclasses import dataclass
from typing import NamedTuple
from collections.abc import Callable

from .mixins import RecordType, RecordContainer
from .profile_data import ProfileData

if hasattr(np, "trapezoid"):
    # np.trapz was renamed to np.trapezoid in Numpy 2.0
    trapezoid = np.trapezoid
else:
    trapezoid = np.trapz # type: ignore

@dataclass(slots=True, repr=False)
class ToaResult(RecordType):
    '''
    Represents the result of fitting for a TOA.
    '''
    toa: np.floating
    ampl: np.floating
    offset: np.floating
    noise_level: np.floating
    toa_error: np.floating
    ampl_error: np.floating
    offset_error: np.floating

@dataclass
class ToaResults(RecordContainer[ToaResult]):
    '''
    Represents the result of fitting for TOAs for several profiles.
    '''
    pass
 
class TemplateMatchingEstimator:
    '''
    A TOA estimator based on matched filtering with a template profile.
    '''
    def __init__(self, template: np.ndarray):
        '''
        Construct the estimator and objective function from a template.
        Pre-computes an FFT of the template, and constructs a Numba JIT
        function to compute the objective given a profile FFT.
        '''
        n = template.shape[0]

        template_fft = np.fft.rfft(template)
        self.template_fft = template_fft
        phase_gradient = -2j*np.pi*np.fft.rfftfreq(n)

        template_sum = template_fft[0].real
        self.template_sum = template_sum
        template_sqsum = 2*np.real(trapezoid(np.abs(template_fft)**2))/n
        self.template_sqsum = template_sqsum

        @nb.njit
        def objective_function(
            profile_fft: np.ndarray,
            tau: float | np.floating,
        ) -> np.floating:
            '''
            Compute the objective function given the FFT of a profile.

            Parameters
            ----------
            profile_fft: "Real" FFT (e.g., `np.fft.rfft()`) of the profile.
            tau: Proposed phase shift.

            Returns
            -------
            obj: Value of the objective function.
            '''
            profile_sum = profile_fft[0].real
            phase = phase_gradient*tau
            ccf_fft = np.conj(np.exp(phase)*template_fft)*profile_fft
            ccf = 2*np.real(trapezoid(ccf_fft))/n
            obj = (ccf - profile_sum*template_sum/n)**2
            obj /= (template_sqsum - template_sum**2/n)
            return obj

        self.objective_function = objective_function

    def get_objective_function(
        self,
        profile: np.ndarray,
        vectorize: bool = True,
    ) -> Callable[[float | np.floating], np.floating]:
        '''
        Return a callable objective function specialized to a profile.

        Parameters
        ----------
        profile: Profile for which to compute the objective function.
        vectorize: If `True`, return a ufunc created with `numba.vectorize`.
            Setting this to `False` reduces JIT compilation overhead
            when there are relatively few function calls per profile.

        Returns
        -------
        objective_for_profile: The objective function for this profile.
            Accepts a proposed phase shift as input, and returns the
            value of the objective function.
        '''
        profile_fft = np.fft.rfft(profile)
        objective_function = self.objective_function

        def objective_for_profile(tau):
            return objective_function(profile_fft, tau)

        if vectorize:
            objective_for_profile = nb.vectorize(objective_for_profile)

        return objective_for_profile

    def sample_objective_function(self, profile: np.ndarray) -> np.ndarray:
        '''
        Calculate values of the objective function at an evenly spaced series
        of sample points, using an FFT. This can be done more efficiently than
        evaluating the objective function at an arbitrary set of points.

        Parameters
        ----------
        profile: Profile for which to compute the objective function.

        Returns
        -------
        obj: Samples values of the objective function, at a series of points
            with the same spacing as the profile samples. The first sample
            corresponds to a phase shift of 0.
        '''
        n = profile.shape[0]

        profile_fft = np.fft.rfft(profile)
        profile_sum = profile_fft[0].real

        ccf = np.fft.irfft(profile_fft*np.conj(self.template_fft), n)
        obj = (ccf - profile_sum*self.template_sum/n)**2
        obj /= (self.template_sqsum - self.template_sum**2/n)
        return obj

    def maximize_objective_function(
        self,
        profile: np.ndarray,
        tol: float | np.floating,
    ) -> np.floating:
        '''
        Maximize the objective function for a specific profile and return
        the best-fit value of the phase shift.

        Parameters
        ----------
        profile: Profile for which to compute the objective function.
        tol: Numerical tolerance used in optimization (in bins).

        Returns
        -------
        tauhat: Best-fit value of the phase shift.
        '''
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

    def build_toa_result(
        self,
        profile: np.ndarray,
        tauhat: float | np.floating,
        noise_level: float | np.floating | None = None,
    ) -> ToaResult:
        '''
        Given a profile and the corresponding best-fit phase shift, determine
        other values of interest and their uncertainties, and construct a
        `ToaResult` object.

        Parameters
        ----------
        profile: Profile for which to compute the objective function.
        tauhat: Best-fit value of the phase shift.

        Returns
        -------
        result: `ToaResult` object containing the complete results of the fit,
            including parameters and their uncertainties.
        '''
        n = profile.shape[0]

        # calculate best-fit values of a and b
        profile_fft = np.fft.rfft(profile)
        profile_sum = profile_fft[0].real
        profile_sqsum = 2*np.real(trapezoid(np.abs(profile_fft)**2))/n

        phase = -2j*np.pi*np.fft.rfftfreq(n)
        ccf_tauhat_fft = np.conj(np.exp(phase)*self.template_fft)*profile_fft
        ccf_tauhat = 2*np.real(trapezoid(ccf_tauhat_fft))/n

        ahat = (ccf_tauhat - profile_sum*self.template_sum/n)
        ahat /= (self.template_sqsum - self.template_sum**2/n)
        bhat = (profile_sum - ahat*self.template_sum)/n

        if noise_level is None:
            # estimate noise level from upper 1/4 of profile FFT
            sigma2hat = np.mean(np.abs(profile_fft[-n//8-1:-1])**2)/n
            sigmahat = np.sqrt(sigma2hat)
            noise_level = sigmahat

        # calculate errors in tau (by finite difference), a, and b
        h = np.finfo(np.float64).eps**(1/4)
        obj0 = self.objective_function(profile_fft, tauhat)
        obj1 = self.objective_function(profile_fft, tauhat - h)
        obj2 = self.objective_function(profile_fft, tauhat + h)
        obj_dderiv = (obj1 + obj2 - 2*obj0)/h**2
        tau_error = noise_level*np.sqrt(-2/obj_dderiv)
        a_error = noise_level/np.sqrt(self.template_sqsum - self.template_sum**2/n)
        b_error = noise_level/np.sqrt(n)

        return ToaResult(
            toa=tauhat,
            ampl=ahat,
            offset=bhat,
            noise_level=noise_level,
            toa_error=tau_error,
            ampl_error=a_error,
            offset_error=b_error,
        )

    def estimate_toa(
        self,
        profile: np.ndarray,
        tol: float | np.floating = np.sqrt(np.finfo(np.float64).eps),
        noise_level: float | np.floating | None = None,
    ) -> ToaResult:
        '''
        Given a profile, perform a fit and return the best-fit values and
        uncertainties of the phase shift and supporting parameters, in the form
        of a `ToaResult` object.

        Parameters
        ----------
        profile: Profile for which to estimate the TOA.
        tol: Numerical tolerance used in optimization (in bins).
        noise_level: Estimate of the off-pulse noise level in the profiles.
            If `None`, it will be estimated from the highest 1/4 of
            frequencies in the FFT of the profile.

        Returns
        -------
        result: `ToaResult` object containing the complete results of the fit,
            including parameter values and their uncertainties.
        '''
        tauhat = self.maximize_objective_function(profile, tol)
        result = self.build_toa_result(profile, tauhat, noise_level)

        return result

    def estimate_toas(
        self,
        data: ProfileData,
        tol: float | np.floating = np.sqrt(np.finfo(np.float64).eps),
        noise_level: float | np.floating | None = None,
    ) -> ToaResults:
        '''
        Given a collection of profiles, perform a fit for each of them and
        return the best-fit values and uncertainties of the phase shift and
        supporting parameters, in the form of a `ToaResults` object.

        Parameters
        ----------
        data: `ProfileData` object containing the profiles to fit.
        tol: Numerical tolerance used in optimization (in bins).
        noise_level: Estimate of the off-pulse noise level in the profiles.
            If `None`, it will be estimated from the highest 1/4 of
            frequencies in the FFT of the profile.

        Returns
        -------
        result: `ToaResults` object containing the complete results of the fit
            for each profile, including parameter values and uncertainties.
        '''
        results = []
        for profile in data.profiles:
            tauhat = self.maximize_objective_function(profile, tol)
            result = self.build_toa_result(profile, tauhat, noise_level)
            results.append(result)
    
        records = np.rec.array(np.array(
            [result.as_record() for result in results]
        ))

        return ToaResults(records)

def toa_fourier(
    template: np.ndarray,
    profile: np.ndarray,
    tol: float | np.floating = sqrt(eps),
    noise_level: float | np.floating | None = None,
) -> ToaResult:
    '''
    Calculate a TOA using Fourier-domain matched filtering with a template.
    The TOA is always reported in units of phase bins.

    This convenience function constructs a `TemplateMatchingEstimator` and
    uses it to fit for the TOA. For better performance, especially when
    fitting multiple TOAs using the same template, constructing and using a
    `TemplateMatchingEstimator` directly is preferable.

    Parameters
    ----------
    template: Template profile shape to use for matched filtering.
    profile: Profile for which to estimate the TOA.
    tol: Numerical tolerance used in optimization (in bins).
    noise_level: Estimate of the off-pulse noise level in the profiles.
        If `None`, it will be estimated from the highest 1/4 of
        frequencies in the FFT of the profile.

    Returns
    -------
    result: `ToaResult` object containing the complete results of the fit,
            including parameter values and their uncertainties.
    '''
    estimator = TemplateMatchingEstimator(template)
    return estimator.estimate_toa(profile, tol=tol)

