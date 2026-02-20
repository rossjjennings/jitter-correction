import numpy as np
from typing import NamedTuple
from dataclasses import dataclass

from .profile_data import ProfileData
from .toas import TemplateMatchingEstimator, ToaResult
from .mixins import RecordContainer

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

def calc_skewness_coeff(
    profile: np.ndarray,
    nlags: int | np.integer | None = None,
) -> np.floating:
    '''
    Approximate the coefficient of tau**3 in the expansion of the skewness
    function around the origin by fitting a fifth-degree polynomial to the
    region of width `2*nlags + 1` around the zero-lag bin and taking the
    coefficient of the cubic term.

    Parameters
    ----------
    profile: Profile for which to calculate the skewness coefficient.
    nlags: Number of lags to use for fitting the polynomial.
        If not provided, will be taken as the greater of `n_bins//16` or 2.

    Returns
    -------
    coeff: Value of the skewness coefficient.
    '''
    n_bins, = profile.shape
    if nlags is None:
        nlags = max(n_bins//16, 2)
    skewness_fn = skewness_function(profile)

    lags = np.linspace(-n_bins + 1, n_bins - 1, 2*n_bins + 1)/n_bins
    zero_lag_bin = n_bins - 1
    sl = slice(zero_lag_bin - nlags, zero_lag_bin + nlags + 1)
    coeffs = np.polyfit(lags[sl], skewness_fn[sl], 5)
    return coeffs[2]

def calc_skewness_coeffs(
    data: ProfileData,
    nlags: int | np.integer | None = None,
) -> np.ndarray:
    '''
    Calculate skewness coefficients for a set of profiles.

    Parameters
    ----------
    data: `ProfileData` object containing the set of profiles.
    nlags: Number of lags to use for fitting the polynomial.
        If not provided, will be taken as the greater of `n_bins//16` or 2.

    Returns
    -------
    coeffs: Value of the skewness coefficient for each profile.
    '''
    skewness_coeffs = np.empty(data.n_profiles)
    for i, profile in enumerate(data.profiles):
        skewness_coeffs[i] = calc_skewness_coeff(profile, nlags)

    return skewness_coeffs

@dataclass(slots=True, repr=False)
class ToaSkewnessResult(ToaResult):
    toa: np.floating
    ampl: np.floating
    offset: np.floating
    noise_level: np.floating
    toa_error: np.floating
    ampl_error: np.floating
    offset_error: np.floating
    skewness_coeff: np.floating

@dataclass(slots=True)
class ToaSkewnessResults(RecordContainer[ToaSkewnessResult]):
    pass

class SkewnessRegressionEstimator:
    '''
    A TOA estimator based on the skewness coefficient
    '''
    def __init__(
        self,
        template: np.ndarray,
        predictor_coeffs: np.ndarray,
        nlags: int | np.integer | None = None,
    ):
        '''
        Construct the estimator from a template and predictor coefficients.

        Parameters
        ----------
        template: Template profile shape to use.
        predictor_coeffs: Coefficients relating the skewness coefficient
            to the TOA correction.
        nlags: Number of lags to use for fitting the polynomial to determine
            the skewness coefficient. If not provided, will be taken as the
            greater of `n_bins//16` or 2.
        '''
        self.tm_estimator = TemplateMatchingEstimator(template)
        self.predictor_coeffs = predictor_coeffs
        self.nlags = nlags

    def build_toa_result(
        self,
        initial_result: ToaResult,
        skewness_coeff: np.floating,
        toa_estimate: np.floating,
    ) -> ToaSkewnessResult:
        '''
        Given an initial TOA result, skewness coefficient, and TOA estimate,
        determine other values of interest and their uncertainties, and
        construct a `ToaSkewnessResult` object.

        Parameters
        ----------
        initial_result: `ToaResult` from template matching on this profile.
        skewness_coeff: Skewness coefficient for this profile.
        toa_estimate: TOA estimate for this profile.

        Returns
        -------
        result: `ToaSkewnessResult` object containing the complete results
            of TOA estimation,  including parameters and their uncertainties.
        '''
        return ToaSkewnessResult(
            toa=toa_estimate,
            ampl=initial_result.ampl,
            offset=initial_result.offset,
            noise_level=initial_result.noise_level,
            toa_error=initial_result.toa_error,
            ampl_error=initial_result.ampl_error,
            offset_error=initial_result.offset_error,
            skewness_coeff=skewness_coeff,
        )

    def estimate_toa(
        self,
        profile: np.ndarray,
        tol: float | np.floating = np.sqrt(np.finfo(np.float64).eps),
        noise_level: float | np.floating | None = None,
    ) -> ToaSkewnessResult:
        '''
        Calculate a corrected TOA using the skewness regression model.

        Parameters
        ----------
        template: The profile model to use for fitting
        predictor_coeffs: Coefficients of the skewness to use in correction
        data: Profiles for which to calculate TOAs, as a `ProfileData` object
        tol: Relative tolerance for optimization (in bins).

        Returns
        -------
        result: `ToaSkewnessResult` object containing parameter values and
            their uncertainties.
        '''
        initial_result = self.tm_estimator.estimate_toa(
            profile,
            tol=tol,
            noise_level=noise_level,
        )
        skewness_coeff = calc_skewness_coeff(profile, self.nlags)
        toa_correction = np.polyval(self.predictor_coeffs, skewness_coeff)
        toa_estimate = initial_result.toa - toa_correction

        return self.build_toa_result(
            initial_result,
            skewness_coeff,
            toa_estimate,
        )

    def estimate_toas(
        self,
        data: ProfileData,
        tol: float | np.floating = np.sqrt(np.finfo(np.float64).eps),
        noise_level: float | np.floating | None = None,
    ) -> ToaSkewnessResults:
        '''
        Calculate corrected TOAs for several profiles using the skewness
        regression model.

        Parameters
        ----------
        template: The profile model to use for fitting
        predictor_coeffs: Coefficients of the skewness to use in correction
        data: Profiles for which to calculate TOAs, as a `ProfileData` object
        tol: Relative tolerance for optimization (in bins).

        Returns
        -------
        result: `ToaSkewnessResult` object containing parameter values and
            their uncertainties.
        '''
        initial_results = self.tm_estimator.estimate_toas(
            data,
            tol=tol,
            noise_level=noise_level,
        )
        skewness_coeffs = calc_skewness_coeffs(data, self.nlags)
        toa_corrections = np.polyval(self.predictor_coeffs, skewness_coeffs)
        toa_estimates = initial_results.toa - toa_corrections

        results = []
        zipped_info = zip(initial_results, toa_estimates, skewness_coeffs)
        for initial_result, toa_estimate, skewness_coeff in zipped_info:
            result = self.build_toa_result(
                initial_result,
                skewness_coeff,
                toa_estimate,
            )
            results.append(result)

        records = np.rec.array(np.array(
            [result.as_record() for result in results]
        ))
        return ToaSkewnessResults(records)
