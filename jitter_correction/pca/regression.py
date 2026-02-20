'''
PCA dot-product based timing (cf. Osłowski 2011)
'''
import numpy as np
from typing import NamedTuple, Iterator, Self
from dataclasses import dataclass

from ..signal import fft_roll
from ..toas import TemplateMatchingEstimator, ToaResult, toa_fourier
from ..profile_data import ProfileData
from ..mixins import RecordContainer
from .pcs import PrincipalComponentModel
from .results import ToaPcaResult, ToaPcaResults

class PCRegressionEstimator:
    '''
    A TOA estimator based on multiple regression of the TOA error on the
    coefficients of principal components. A correction term is estimated
    from the data and added to an initial estimate calculated by template
    matching.
    '''
    def __init__(self, model: PrincipalComponentModel, coeffs: np.ndarray):
        '''
        Construct the estimator from a principal component model and a set of
        regression coefficients.
        '''
        self.tm_estimator = TemplateMatchingEstimator(model.template)
        self.model = model
        self.coeffs = coeffs

    def build_toa_result(
        self,
        profile: np.ndarray,
        initial_result: ToaResult,
        toa_estimate: np.floating,
        scores: np.ndarray,
        noise_level: float | np.floating | None,
    ) -> ToaPcaResult:
        '''
        Given an initial TOA result, principal component scores, and a
        corrected TOA estimate, determine other values of interest and their
        uncertainties, and construct a `ToaPcaResult` object.

        Parameters
        ----------
        profile: Profile for which a TOA was estimated.
        initial_result: `ToaResult` from template matching on this profile.
        toa_estimate: TOA estimate for this profile.
        scores: Principal component scores for this profile.
        noise_level: Estimate of the off-pulse noise level in the profiles.
            If `None`, it will be estimated from the highest 1/4 of
            frequencies in the FFT of the profile.

        Returns
        -------
        result: `ToaPcaResult` object containing the complete results of
            TOA estimation,  including parameters and their uncertainties.
        '''
        n, = profile.shape
        profile_fft = np.fft.rfft(profile)

        if noise_level is None:
            # estimate noise level from upper 1/4 of profile FFT
            sigma2hat = np.mean(np.abs(profile_fft[-n//8-1:-1])**2)/n
            sigmahat = np.sqrt(sigma2hat)
            noise_level = sigmahat
        noise_level = np.array([noise_level])[0]

        return ToaPcaResult(
            toa=toa_estimate,
            ampl=initial_result.ampl,
            offset=initial_result.offset,
            scores=scores,
            noise_level=noise_level,
            toa_error=initial_result.toa_error,
            ampl_error=initial_result.ampl_error,
            toa_ampl_corr=initial_result.ampl_error.dtype.type(0),
            offset_error=initial_result.offset_error,
            score_errors=np.zeros_like(scores),
        )

    def estimate_toa(
        self,
        profile: np.ndarray,
        tol: float | np.floating = np.sqrt(np.finfo(np.float64).eps),
        noise_level: float | np.floating | None = None,
    ) -> ToaPcaResult:
        '''
        Calculated a corrected TOA using the principal component regression
        model.

        Parameters
        ----------
        profile: Profile for which to estimate the TOA.
        tol: Numerical tolerance used in optimization.
        noise_level: Estimate of the off-pulse noise level in the profiles.
            If `None`, it will be estimated from the highest 1/4 of
            frequencies in the FFT of the profile.

        Returns
        -------
        result: `ToaPcaResult` object containing the complete fit results,
            including parameter values and their uncertainties.
        '''
        initial_result = self.tm_estimator.estimate_toa(
            profile,
            tol=tol,
            noise_level=noise_level,
        )

        template_shifted = fft_roll(self.model.template, initial_result.toa)
        pcs_shifted = fft_roll(self.model.pcs, initial_result.toa)
        scores = np.dot(pcs_shifted, profile)
        correction = np.dot(self.coeffs, scores)
        toa_estimate = initial_result.toa - correction

        return self.build_toa_result(
            profile,
            initial_result,
            toa_estimate,
            scores,
            noise_level,
        )

    def estimate_toas(
        self,
        data: ProfileData,
        tol: float | np.floating = np.sqrt(np.finfo(np.float64).eps),
        noise_level: float | np.floating | None = None,
    ) -> ToaPcaResults:
        '''
        Calculate corrected TOAs for each of several profiles using the
        principal component regression model.

        Parameters
        ----------
        data: Profiles for which to estimate TOAs, as a `ProfileData` object.
        tol: Numerical tolerance used in optimization.
        noise_level: Estimate of the off-pulse noise level in the profiles.
            If `None`, it will be estimated from the highest 1/4 of
            frequencies in the FFT of the profile.

        Returns
        -------
        results: `ToaPcaResults` object containing the complete fit results
            for each profile, including parameter values and uncertainties.
        '''
        initial_results = self.tm_estimator.estimate_toas(
            data,
            tol=tol,
            noise_level=noise_level,
        )

        results = []
        for profile, initial_result in zip(data.profiles, initial_results):
            template_shifted = fft_roll(self.model.template, initial_result.toa)
            pcs_shifted = fft_roll(self.model.pcs, initial_result.toa)
            scores = np.dot(pcs_shifted, profile)
            correction = np.dot(self.coeffs, scores)
            toa_estimate = initial_result.toa - correction

            result = self.build_toa_result(
                profile,
                initial_result,
                toa_estimate,
                scores,
                noise_level,
            )
            results.append(result)

        records = np.rec.array(np.array(
            [result.as_record() for result in results]
        ))
        return ToaPcaResults(records)
