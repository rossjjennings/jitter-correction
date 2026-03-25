'''
Principal component matching with a prior
'''
import numpy as np
import numba as nb
from scipy.optimize import minimize, minimize_scalar
from loguru import logger
from functools import partial
from collections.abc import Callable

from ..profile_data import ProfileData
from .pcs import PrincipalComponentModel
from .results import ToaPcaResult, ToaPcaResults

if hasattr(np, "trapezoid"):
    # np.trapz was renamed to np.trapezoid in Numpy 2.0
    trapezoid = np.trapezoid
else:
    trapezoid = np.trapz # type: ignore

class PCBayesianEstimator:
    '''
    A TOA estimator based on matched filtering with a flexible profile model
    which can be described as the sum of a template profile and scaled copies
    of several principal components, similar to `PcMatchingEstimator`. Unlike
    in that case, a prior is imposed on the principal component scores, and
    TOA estimation is based on maximizing the posterior density.
    '''
    def __init__(self, model: PrincipalComponentModel):
        '''
        Construct the estimator from a principal component model. Pre-computes
        FFTs of the template and each principal component, and constructs
        Numba JIT functions to compute the objective given a profile FFT, and
        to estimate the best-fit value of the template amplitude, `ahat`.
        '''
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
        template_sqsum = 2*np.real(trapezoid(np.abs(template_fft)**2))/n
        self.template_sqsum = template_sqsum

        @nb.njit
        def ahat_ml(
            profile_fft: np.ndarray,
            tau: float | np.floating,
        ) -> np.floating:
            '''
            Compute an estimate of the best-fit template amplitude.
            Does not include a correction due to the prior.
            This is used in fitting to find an appropriate initial guess.

            Parameters
            ----------
            profile_fft: "Real" FFT (e.g., `np.fft.rfft()`) of the profile.
            tau: Proposed phase shift.

            Returns
            -------
            ahat: Estimated value of the template amplitude.
            '''
            profile_sum = profile_fft[0].real
            phase = phase_gradient*tau
            ccf_fft = np.conj(np.exp(phase)*template_fft)*profile_fft
            ccf = 2*np.real(trapezoid(ccf_fft))/n

            ahat = (ccf - profile_sum*template_sum/n)
            ahat /= (template_sqsum - template_sum**2/n)

            return ahat

        self.ahat_ml = ahat_ml

        @nb.njit
        def objective_function(
            profile_fft: np.ndarray,
            noise_level: float | np.floating,
            a: float | np.floating,
            tau: float | np.floating,
        ) -> np.floating:
            '''
            Compute the objective function given the FFT of a profile.

            Parameters
            ----------
            profile_fft: "Real" FFT (e.g., `np.fft.rfft()`) of the profile.
            noise_level: Estimate of the off-pulse noise level in the profile.
            a: Proposed template amplitude.
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

            for pc_fft, eigval in zip(pcs_fft, eigvals):
                pccf_fft = np.conj(np.exp(phase)*pc_fft)*profile_fft
                pccf = 2*np.real(trapezoid(pccf_fft))/n
                shrinkage_factor = 1/(1 + noise_level**2/(eigval*a))
                obj += shrinkage_factor*pccf**2
                obj += noise_level**2/n*np.log(2*np.pi*eigval*a**2/n)

            ahat = (ccf - profile_sum*template_sum/n)
            ahat /= (template_sqsum - template_sum**2/n)
            obj -= (template_sqsum - template_sum**2/n)*(a - ahat)**2

            return obj

        self.objective_function = objective_function

    def get_objective_function(
        self,
        profile: np.ndarray,
        noise_level: float | np.floating | None = None,
        vectorize: bool = True,
    ) -> Callable[[float | np.floating, float | np.floating], np.floating]:
        '''
        Return a callable objective function specialized to a profile.

        Parameters
        ----------
        profile: Profile for which to compute the objective function.
        noise_level: Estimate of the off-pulse noise level in the profile.
            If `None`, it will be estimated from the highest 1/4 of
            frequencies in the FFT of the profile.
        vectorize: If `True`, return a ufunc created with `numba.vectorize`.
            Setting this to `False` reduces JIT compilation overhead
            when there are relatively few function calls per profile.

        Returns
        -------
        objective_for_profile: The objective function for this profile.
            Accepts a proposed template amplitude and phase shift as input,
            and returns the value of the objective function.
        '''
        n = profile.shape[0]

        profile_fft = np.fft.rfft(profile)
        objective_function = self.objective_function

        if noise_level is None:
            sigma2 = np.mean(np.abs(profile_fft[-n//8-1:-1])**2)/n
            noise_level = np.sqrt(sigma2)

        def objective_for_profile(a, tau):
            return objective_function(profile_fft, noise_level, a, tau)

        if vectorize:
            objective_for_profile = nb.vectorize(objective_for_profile)

        return objective_for_profile

    def get_1d_objective_function(
        self,
        profile: np.ndarray,
        noise_level: float | np.floating | None = None,
    ) -> Callable[[float | np.floating], np.floating]:
        '''
        Return a callable version of the 1-dimensional objective function,
        specialized to a profile and optimized over the amplitude, `a`.

        Parameters
        ----------
        profile: Profile for which to compute the objective function.
        noise_level: Estimate of the off-pulse noise level in the profile.
            If `None`, it will be estimated from the highest 1/4 of
            frequencies in the FFT of the profile.
        vectorize: If `True`, return a ufunc created with `numba.vectorize`.
            Setting this to `False` reduces JIT compilation overhead
            when there are relatively few function calls per profile.

        Returns
        -------
        oned_objective: The 1-dimensional objective function for this profile.
            Accepts a proposed phased shift as input, and returns the value of
            the objective function.
        '''
        objective_fn = self.get_objective_function(profile, noise_level)

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

    def sample_ml_objective(self, profile: np.ndarray):
        '''
        Calculate values of the objective function (not including a correction
        due to the prior) at an evenly spaced series of sample points, using
        an FFT. This can be done more efficiently than evaluating the full
        objective function at an arbitrary set of points.

        Unlike values returned by the `sample_objective_function()` method of
        `TemplateMatchingEstimator` and `PcMatchingEstimator` objects, values
        returned by this function should not be expected to match the output
        of the objective function exactly. They are used in fitting to obtain
        a reasonable initial guess for the phase shift, which is then refined.

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

    def maximize_objective_function(
        self,
        profile: np.ndarray,
        tol: float | np.floating,
        noise_level: float | np.floating | None = None,
    ) -> tuple[np.floating, np.floating]:
        '''
        Maximize the objective function for a specific profile and return
        the best-fit value of the phase shift.

        Parameters
        ----------
        profile: Profile for which to compute the objective function.
        noise_level: Estimate of the off-pulse noise level in the profile.
            If `None`, it will be estimated from the highest 1/4 of
            frequencies in the FFT of the profile.
        tol: Numerical tolerance used in optimization.

        Returns
        -------
        tauhat: Best-fit value of the phase shift.
        '''
        n = profile.shape[0]

        objective_fn = self.get_objective_function(
            profile,
            noise_level,
            vectorize=False,
        )
        ml_objective_samples = self.sample_ml_objective(profile)

        sample_argmax = np.argmax(ml_objective_samples)
        if sample_argmax > n/2:
            sample_argmax -= n

        tau_guess = sample_argmax
        a_guess = self.ahat_ml(np.fft.rfft(profile), tau_guess)

        result = minimize(
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

    def build_toa_result(
        self,
        profile: np.ndarray,
        ahat: float | np.floating,
        tauhat: float | np.floating,
        noise_level: float | np.floating | None = None,
    ) -> ToaPcaResult:
        '''
        Given a profile and the corresponding best-fit phase shift, determine
        other values of interest and their uncertainties, and construct a
        `ToaPcaResult` object.

        Parameters
        ----------
        profile: Profile for which to compute the objective function.
        ahat: Best-fit value of the template amplitude.
        tauhat: Best-fit value of the phase shift.
        noise_level: Estimate of the off-pulse noise level in the profile.
            If `None`, it will be estimated from the highest 1/4 of
            frequencies in the FFT of the profile.

        Returns
        -------
        result: `ToaPcaResult` object containing the complete fit results,
            including parameters and their uncertainties.
        '''
        n = profile.shape[0]
        ahat = np.array([ahat])[0]
        tauhat = np.array([tauhat])[0]
        noise_level = np.array([noise_level])[0]

        # calculate best-fit values of a, b, and x_i
        profile_fft = np.fft.rfft(profile)
        profile_sum = profile_fft[0].real
        phase = -2j*np.pi*tauhat*np.fft.rfftfreq(n)

        if noise_level is None:
            # estimate noise level from upper 1/4 of profile FFT
            sigma2hat = np.mean(np.abs(profile_fft[-n//8-1:-1])**2)/n
            sigmahat = np.sqrt(sigma2hat)
            noise_level = sigmahat

        bhat = (profile_sum - ahat*self.template_sum)/n
        shrinkage_factors = 1/(1 + noise_level**2/(self.eigvals*ahat))

        xhats = []
        for pc_fft, shrinkage_factor in zip(self.pcs_fft, shrinkage_factors):
            pccf_fft = np.conj(np.exp(phase)*pc_fft)*profile_fft
            pccf = 2*np.real(trapezoid(pccf_fft))/n
            xhats.append(shrinkage_factor*pccf/ahat)
        xhats = np.array(xhats)

        # calculate errors in tau, a (by finite difference), and b
        h = np.finfo(np.float64).eps**(1/5)
        w1 = np.sqrt(5) - 1
        w2 = np.sqrt(5) + 1
        w3 = np.sqrt(10 + 2*np.sqrt(5))
        w4 = np.sqrt(10 - 2*np.sqrt(5))
        w5 = 2 + w2
        w6 = 2 - w1

        u1 = w1/4
        u2 = w2/4
        v1 = w3/4
        v2 = w4/4

        objective_fn = self.get_objective_function(
            profile,
            noise_level,
            vectorize=False,
        )
        f0 = objective_fn(ahat, tauhat)
        f1 = objective_fn(ahat + h, tauhat)
        f2 = objective_fn(ahat + u1*h, tauhat + v1*h)
        f3 = objective_fn(ahat - u2*h, tauhat + v2*h)
        f4 = objective_fn(ahat - u2*h, tauhat - v2*h)
        f5 = objective_fn(ahat + u1*h, tauhat - v1*h)

        dd_obj_da2 = (-10*f0 + 6*f1 - w1*f2 + w2*f3 + w2*f4 - w1*f5)/(5*h**2)
        dd_obj_da_dtau = (w4*f2 - w3*f3 + w3*f4 - w4*f5)/(5*h**2)
        dd_obj_dtau2 = (-10*f0 - 2*f1 + w5*f2 + w6*f3 + w6*f4 + w5*f5)/(5*h**2)

        hessdet = dd_obj_da2*dd_obj_dtau2 - dd_obj_da_dtau**2
        tau_error = noise_level*np.sqrt(-2*dd_obj_da2/hessdet)
        a_error = noise_level*np.sqrt(-2*dd_obj_dtau2/hessdet)
        a_tau_cov = 2*noise_level**2*dd_obj_da_dtau/hessdet
        a_tau_corr = a_tau_cov/(a_error*tau_error)
        b_error = noise_level/np.sqrt(n)
        x_errors = noise_level/ahat*shrinkage_factors

        return ToaPcaResult(
            toa=tauhat,
            ampl=ahat,
            offset=bhat,
            scores=xhats,
            noise_level=noise_level,
            toa_error=tau_error,
            ampl_error=a_error,
            toa_ampl_corr=a_tau_corr,
            offset_error=b_error,
            score_errors=x_errors,
        )

    def estimate_toa(
        self,
        profile: np.ndarray,
        tol: float | np.floating = np.sqrt(np.finfo(np.float64).eps),
        noise_level: float | np.floating | None =None,
    ) -> ToaPcaResult:
        '''
        Given a profile, perform a fit and return the best-fit values and
        uncertainties of the phase shift and supporting parameters, in the form
        of a `ToaPcaResult` object.

        Parameters
        ----------
        profile: Profile for which to estimate the TOA.
        tol: Numerical tolerance used in optimization.
        noise_level: Estimate of the off-pulse noise level in the profile.
            If `None`, it will be estimated from the highest 1/4 of
            frequencies in the FFT of the profile.

        Returns
        -------
        result: `ToaPcaResult` object containing the complete fit results,
            including parameter values and their uncertainties.
        '''
        ahat, tauhat = self.maximize_objective_function(
            profile,
            tol=tol,
            noise_level=noise_level,
        )
        result = self.build_toa_result(profile, ahat, tauhat, noise_level)

        return result

    def estimate_toas(
        self,
        data: ProfileData,
        tol: float | np.floating = np.sqrt(np.finfo(np.float64).eps),
        noise_level: float | np.floating | None = None,
    ) -> ToaPcaResults:
        '''
        Given a collection of profiles, perform a fit for each of them and
        return the best-fit values and uncertainties of the phase shift and
        supporting parameters, in the form of a `ToaPcaResults` object.

        Parameters
        ----------
        data: `ProfileData` object containing the profiles to fit.
        tol: Numerical tolerance used in optimization.
        noise_level: Estimate of the off-pulse noise level in the profiles.
            If `None`, it will be estimated from the highest 1/4 of
            frequencies in the FFT of the profile.

        Returns
        -------
        results: `ToaPcaResults` object containing the complete fit results
            for each profile, including parameter values and uncertainties.
        '''
        results = []
        for profile in data.profiles:
            ahat, tauhat = self.maximize_objective_function(
                profile,
                tol=tol,
                noise_level=noise_level,
            )
            result = self.build_toa_result(profile, ahat, tauhat, noise_level)
            results.append(result)

        records = np.rec.array(np.array(
            [result.as_record() for result in results]
        ))

        return ToaPcaResults(records)
