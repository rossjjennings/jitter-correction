import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
from numpy.fft import fft, ifft, rfft, irfft, fftfreq, fftshift
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.optimize import minimize_scalar, brentq
from collections import namedtuple

from .toas import toa_fourier
from .signal import fft_roll
from .profile_data import gen_profiles

def gen_data(spec, n_profiles, npprof, n_bins, SNR, drift_bins):
    '''
    Generated simulated data based on a pulse specification.
    '''
    phase = np.linspace(-1/2, 1/2, n_bins, endpoint=False)
    profiles = gen_profiles(phase, spec=spec, n_profiles=n_profiles, npprof=npprof, SNR=SNR)
    
    shifts = drift_bins/n_profiles*np.arange(n_profiles)
    shifts -= np.mean(shifts)
    for i, profile in enumerate(profiles):
        profiles[i] = fft_roll(profile, shifts[i])
    
    return phase, profiles

def skewness_function(profile):
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

def skewness_coeff(lags, skewness, nlags=16):
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

def get_template(profiles, n_iter=1):
    '''
    Create a template by iteratively aligning and averaging profiles.
    Starts by averaging the middle 10% of profiles and iterates `n_iter` times,
    each time aligning the pulses using the previous template and averaging
    to create a new template.
    '''
    n = profiles.shape[0]
    n_5_percent = n//20
    sl = slice(n//2 - n_5_percent, n//2 + n_5_percent)
    template = np.mean(profiles[sl], axis=0)
    toas = np.empty(profiles.shape[0])
    
    for i in range(n_iter):
        for i, profile in enumerate(profiles):
            result = toa_fourier(template, profile)
            toas[i] = result.toa
        profiles_aligned = np.empty_like(profiles)
        for j, profile in enumerate(profiles):
            profiles_aligned[j] = fft_roll(profile, -toas[j])

        template = np.mean(profiles_aligned, axis=0)
    
    return template

def calc_skewness_coeffs(profiles):
    '''
    Calculate skewness coefficients for a set of profiles.
    '''
    n_profiles, n_bins = profiles.shape
    lags = np.empty(2*n_bins-1)
    lags[n_bins-1:] = np.linspace(0, 1, n_bins)
    lags[:n_bins-1] = -np.linspace(0, 1, n_bins)[:0:-1]
    
    skewness_fns = np.empty((n_profiles, 2*n_bins-1))
    for i, profile in enumerate(profiles):
        skewness_fn = skewness_function(profile)
        skewness_fns[i] = skewness_fn
    skewness_coeffs = np.array([
        skewness_coeff(lags, skewness_fn, nlags=129) for skewness_fn in skewness_fns
    ])
    
    return skewness_coeffs

def calc_dtoas(template, profiles, poly_degree=1):
    '''
    Calculate ΔTOAs from profiles with polynomial drift.
    '''
    n_profiles = profiles.shape[0]
    profile_number = np.arange(n_profiles)
    toas = np.empty(n_profiles)
    for i, profile in enumerate(profiles):
        result = toa_fourier(template, profile)
        toas[i] = result.toa
    timing_poly_coeffs = np.polyfit(profile_number, toas, poly_degree)
    timing_poly_vals = np.polyval(timing_poly_coeffs, profile_number)
    dtoas = toas - timing_poly_vals
    
    return dtoas

def extract_pcs(profiles, n_pcs, initial_template=None, return_all=True, use_trend=True):
    '''
    Extract a template and principal components from a set of profiles.
    An initial template can be supplied; if not, the default strategy is to average
    the middle 10 percent of profiles to get an initial template.
    
    Inputs
    ------
    profiles:   The profiles, as rows of a 2-D array.
    n_pcs:      The number of principal components to use in the model.
    n_iter:     The number of iterations to perform.
    initial_template: The initial template (see above).
    return_all: Return all principal components (instead of the first `n_pcs`).
    use_trend:  If `False`, align profiles using their individual TOAs,
                ignoring n_iter. If `True`, align using a linear trend (default).
    
    Outputs
    -------
    template: The final template
    pcs:      The final principal components
    sgvals:   The singular values (characteristic amplitudes) corresponding to the
              principal components.
    scores:   The PC scores of the training data.
    dtoas:    The differences between the TOAs and the linear trend.
    '''
    n_profiles = profiles.shape[0]
    profile_number = np.arange(n_profiles)
    if initial_template is None:
        initial_template = get_template(profiles, n_iter=0)
    
    toas = np.zeros(n_profiles)
    for i, profile in enumerate(profiles):
        result = toa_fourier(initial_template, profile)
        toas[i] = result.toa
    
    resids = np.empty_like(profiles)
    if use_trend:
        trend_coeffs = np.polyfit(profile_number, toas, 1)
        trend = np.polyval(trend_coeffs, profile_number)
        
        profiles_aligned = np.empty_like(profiles)
        for j, profile in enumerate(profiles):
            profiles_aligned[j] = fft_roll(profile, -trend[j])
        
        template = np.mean(profiles_aligned, axis=0)
        for j, profile in enumerate(profiles_aligned):
            ampl = np.dot(profile, template)/np.dot(template, template)
            resids[j] = profile - ampl*template
        u, s, pcs = svd(resids, full_matrices=return_all)
        sgvals = s/np.sqrt(resids.shape[0])
        
        scores = np.dot(pcs, profiles_aligned.T)
        dtoas = toas - trend
    else:
        profiles_aligned = np.empty_like(profiles)
        for j, profile in enumerate(profiles):
            profiles_aligned[j] = fft_roll(profile, -toas[j])
        
        template = np.mean(profiles_aligned, axis=0)
        for j, profile in enumerate(profiles_aligned):
            ampl = np.dot(profile, template)/np.dot(template, template)
            resids[j] = profile - ampl*template
        u, s, pcs = svd(resids, full_matrices=return_all)
        sgvals = s/np.sqrt(resids.shape[0])
        
        # Trend used only for computing ΔTOAs
        trend_coeffs = np.polyfit(profile_number, toas, 1)
        trend = np.polyval(trend_coeffs, profile_number)
        scores = np.dot(pcs, profiles_aligned.T)
        dtoas = toas - trend
    
    return template, pcs, sgvals, scores, dtoas

def plot_pcs(phase, template, pcs, eigvals, n_pcs):
    fig = plt.figure(figsize=(5.4, 4.8))
    (spec1, spec2, spec3, spec4) = mpl.gridspec.GridSpec(
        nrows=2, ncols=2, width_ratios=(1.0, 0.25), height_ratios=(0.35, 1.0)
    )
    ax_main = fig.add_subplot(spec3)
    ax_side = fig.add_subplot(spec4, sharey=ax_main)
    ax_side.tick_params(axis='y', which='both', labelleft=False)
    ax_top = fig.add_subplot(spec1, sharex=ax_main)
    ax_top.tick_params(axis='x', which='both', labelbottom=False)

    ax_top.plot(phase, template)
    ax_top.set_ylabel('Mean')

    ax_side.scatter(eigvals, np.arange(1, len(eigvals)+1))
    stemlines = [((0, i), (eigval, i)) for i, eigval in enumerate(eigvals)]
    eigvals_geom_center = np.sqrt(eigvals[0]*eigvals[n_pcs-1])
    eigvals_span = eigvals[0]/eigvals_geom_center
    xlim_low = eigvals_geom_center/eigvals_span**1.25
    xlim_high = eigvals_geom_center*eigvals_span**1.25
    ax_side.set_xlim(xlim_low, xlim_high)
    ax_side.set_xscale('log')
    ax_side.set_xlabel('Eigenvalue')
    ax_side.set_xticks([1e-2, 1e0])
    ax_side.set_xticklabels([r'$10^{-2}$', '1'])

    for i in range(n_pcs):
        ax_main.plot(phase, -4*pcs[i]+i+1)
    ax_main.set_ylabel('Principal components')
    ax_main.set_yticks(np.arange(1, n_pcs+1))
    ax_main.set_ylim(n_pcs + 0.75, 0.25)
    ax_main.set_xlabel('Phase (cycles)')

    plt.minorticks_on()
    plt.tight_layout()
    
    return fig, (ax_top, ax_main, ax_side)

ToaScoreResult = namedtuple('ToaResult', ['toa', 'ampl', 'scores'])

def toa_score(template, pcs, coeffs, profile, ts = None, tol = sqrt(np.finfo(np.float64).eps)):
    '''
    Calculate a maximum-likelihood TOA given a template and a PCA model of pulse shape variations.
    Uses the dot-product-based method of Osłowski (2011).

    `pcs`:    The principal components (unit vectors), as rows of an array.
    `coeffs`: Coefficients of principal component dot products to use in correcter.
    `ts`:     Evenly-spaced array of phase values corresponding to the profile.
              Sets the units of the TOA. If this is `None`, the TOA is reported in bins.
    `tol`:    Relative tolerance for optimization (in bins).
    '''
    n = len(profile)
    if ts is None:
        ts = np.arange(n)
    dt = float(ts[1] - ts[0])
    k = len(pcs)

    result = toa_fourier(template, profile, ts = ts, tol = tol)
    initial_toa = result.toa
    ampl = result.ampl

    template_shifted = fft_roll(template, initial_toa/dt)
    pcs_shifted = fft_roll(pcs, initial_toa/dt)
    scores = np.dot(pcs_shifted, profile)
    correcter = np.dot(coeffs, scores)
    toa = initial_toa - correcter

    return ToaScoreResult(toa=toa, ampl=ampl, scores=scores)

ToaPcaResult = namedtuple('ToaResult', ['toa', 'ampl', 'scores'])

def toa_pca(template, pcs, profile, ts = None, tol = sqrt(np.finfo(np.float64).eps), plot=False):
    '''
    Calculate a maximum-likelihood TOA given a template and a PCA model of pulse shape variations.
    
    `pcs`: The principal components (unit vectors), as rows of an array.
    `ts`:  Evenly-spaced array of phase values corresponding to the profile.
           Sets the units of the TOA. If this is `None`, the TOA is reported in bins.
    `tol`: Relative tolerance for optimization (in bins).
    '''
    n = len(profile)
    if ts is None:
        ts = np.arange(n)
    dt = float(ts[1] - ts[0])
    k = len(pcs)
    
    template_fft = fft(template)
    profile_fft = fft(profile)
    pcs_fft = fft(pcs)
    phase_per_bin = -2j*pi*fftfreq(n)
    
    circular_ccf = irfft(rfft(profile)*np.conj(rfft(template)), n)*dt
    sq_ccf = circular_ccf**2 / (np.sum(template**2)*dt)
    for i in range(k):
        pcfft = irfft(rfft(profile)*np.conj(rfft(pcs[i])))*sqrt(dt)
        sq_ccf += pcfft**2
    
    ccf_argmax = np.argmax(sq_ccf)
    ccf_max_val = sq_ccf[ccf_argmax]
    ccf_max = ccf_argmax*dt
    if ccf_argmax > n/2:
        ccf_max -= n*dt
    
    def modified_squared_ccf(tau):
        phase = phase_per_bin*tau/dt
        ccf = np.inner(profile_fft, exp(-phase)*np.conj(template_fft))*dt/n
        sq_ccf = ccf.real**2 / (np.sum(template**2)*dt)
        
        for i in range(k):
            pc_fft = pcs_fft[i]
            pccf = np.inner(profile_fft, exp(-phase)*np.conj(pc_fft))*sqrt(dt)/n
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
    
    return ToaPcaResult(toa=toa, ampl=ampl, scores=scores)

def toa_pca_prior(template, pcs, weights, profile, ts = None, tol = sqrt(np.finfo(np.float64).eps), plot=False):
    '''
    Calculate a maximum-likelihood TOA given a template and a PCA model of pulse shape variations.
    
    `pcs`: The principal components (unit vectors), as rows of an array.
    `ts`:  Evenly-spaced array of phase values corresponding to the profile.
           Sets the units of the TOA. If this is `None`, the TOA is reported in bins.
    `tol`: Relative tolerance for optimization (in bins).
    '''
    n = len(profile)
    if ts is None:
        ts = np.arange(n)
    dt = float(ts[1] - ts[0])
    k = len(pcs)
    
    template_fft = fft(template)
    profile_fft = fft(profile)
    pcs_fft = fft(pcs)
    phase_per_bin = -2j*pi*fftfreq(n)
    
    circular_ccf = irfft(rfft(profile)*np.conj(rfft(template)), n)*dt
    sq_ccf = circular_ccf**2 / (np.sum(template**2)*dt)
    for i in range(k):
        pcfft = irfft(rfft(profile)*np.conj(rfft(pcs[i])))*sqrt(dt)
        sq_ccf += pcfft**2 / (1 + weights[i])
    
    ccf_argmax = np.argmax(sq_ccf)
    ccf_max_val = sq_ccf[ccf_argmax]
    ccf_max = ccf_argmax*dt
    if ccf_argmax > n/2:
        ccf_max -= n*dt
    
    def modified_squared_ccf(tau):
        phase = phase_per_bin*tau/dt
        ccf = np.inner(profile_fft, exp(-phase)*np.conj(template_fft))*dt/n
        sq_ccf = ccf.real**2 / (np.sum(template**2)*dt)
        
        for i in range(k):
            pc_fft = pcs_fft[i]
            weight = weights[i]
            pccf = np.inner(profile_fft, exp(-phase)*np.conj(pc_fft))*sqrt(dt)/n
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

def marchenko_pastur_cdf(lmbda, x):
    '''
    Calculate the cumulative distribution function for the Marchenko-Pastur distribution.
    '''
    lmbda_plus = (1 + sqrt(lmbda))**2
    lmbda_minus = (1 - sqrt(lmbda))**2

    if x == lmbda_minus:
        return 0
    elif x == lmbda_plus:
        return 1
    r = np.sqrt((lmbda_plus - x)/(x - lmbda_minus))
    cdf = 1/2 + np.sqrt((lmbda_plus - x)*(x - lmbda_minus))/(2*pi*lmbda)
    cdf -= (1+lmbda)*np.arctan((r**2-1)/(2*r))/(2*pi*lmbda)
    cdf += (1-lmbda)*np.arctan((lmbda_minus*r**2-lmbda_plus)/(2*(1-lmbda)*r))/(2*pi*lmbda)
    if lmbda > 1:
        cdf = lmbda*cdf - (lmbda-1)/2

    return cdf

def marchenko_pastur_eigval(m, n, i):
    '''
    Use the Marchenko-Pastur distribution to predict the eigenvalue corresponding to the ith
    principal component of an m×n matrix of white Gaussian noise with unit variance.
    '''
    rank = min(m, n)
    lmbda = n/m
    lmbda_plus = (1 + sqrt(lmbda))**2
    lmbda_minus = (1 - sqrt(lmbda))**2
    
    def objective(x):
        cdf = marchenko_pastur_cdf(lmbda, x)
        return rank*(1 - cdf) - i

    result = brentq(objective, lmbda_minus, lmbda_plus)
    return result

def get_chisq(template, pcs, sgvals, noise_level, profile):
    '''
    Get the chi-squared value (-2 * log-likelihood) of a profile based on a set of
    basis vectors (`pcs`) and corresponding variances.
    
    `sgvals`: Singular value (scale) associated with each principal component.
    `noise_level`: Standard deviation of additive noise (in the same units as the profile).
    '''
    def chisq_fn(shift):
        shifted_profile = fft_roll(profile, -shift)
        pc_ampls = pcs @ shifted_profile
        chisq = shifted_profile @ shifted_profile
        chisq -= (template @ shifted_profile)**2 / (template @ template)
        chisq -= np.sum(pc_ampls**2)
        chisq /= noise_level**2
        chisq += np.sum(pc_ampls**2/(sgvals**2 + noise_level**2))
        return chisq
    return chisq_fn

def toa_map(template, pcs, sgvals, noise_level, profile, dt = 1.0,
            grid_size = 8, tol = np.sqrt(np.finfo(np.float64).eps)):
    '''
    Calculate a maximum a-posteriori TOA estimate using a template and principal components.
    
    `pcs`: The principal components (unit vectors), as rows of an array.
    `sgvals`: The singular value (scale) associated with each principal component.
    `dt`:  Width of each phase bin. Sets the units of the TOA.
    `grid_size`: The number of initial samples to use in optimization.
    `tol`: Relative tolerance for optimization (in bins).
    '''
    n = len(profile)
    chisq_fn = get_chisq(template, pcs, sgvals, noise_level, profile)
    
    grid_points = np.linspace(0, n, grid_size, endpoint=False)
    grid_chisqs = [chisq_fn(x) for x in grid_points]
    grid_argmin = grid_points[np.argmin(grid_chisqs)]
    dgrid = n/grid_size
    brack = (grid_argmin - dgrid, grid_argmin, grid_argmin + dgrid)

    result = minimize_scalar(chisq_fn, method = 'Brent', bracket = brack, tol = tol)
    return result.x * dt

def get_chisq_tdmarg(template, template_deriv, pcs, sgvals, coeffs, scatter, noise_level, profile):
    '''
    Get the chi-squared value (-2 * log-likelihood) of a profile based on a template,
    principal components (`pcs`) and corresponding statistical information. Uses the
    "marginalized template derivative" model, which explicitly accounts for degeneracy
    between the TOA and the coefficient of the template derivative.
    
    `template`: The template (average profile shape).
    `template_deriv`: The derivative of the template with respect to shifts.
    `pcs`: The principal components of the profile residuals, after subtracting the
           best-fit linear combination of `template` and `template_deriv`.
    `sgvals`: Singular values corresponding to the principal components.
    `coeffs`: Coefficients in the linear combination of PC amplitudes that provides the
              best fit to the amplitude of `template_deriv`.
    `scatter`: The standard deviation of the difference between the best-fit coefficient
               of `template_deriv` and the value predicted using `coeffs`.
    `noise_level`: Standard deviation of additive noise (in the same units as the profile).
    `profile`: The profile to fit.
    '''
    def chisq_fn(shift):
        shifted_profile = fft_roll(profile, -shift)
        t_ampl = template @ shifted_profile / np.sqrt(template @ template)
        td_ampl = template_deriv @ shifted_profile / np.sqrt(template_deriv @ template_deriv)
        pc_ampls = pcs @ shifted_profile
        
        pred_td_ampl = coeffs @ pc_ampls * np.sqrt(template_deriv @ template_deriv)
        chisq = (td_ampl - pred_td_ampl)**2 / (noise_level**2 + scatter**2 * (template_deriv @ template_deriv))
        chisq += np.sum(pc_ampls**2 / (sgvals**2 + noise_level**2))
        
        return chisq
    return chisq_fn

def get_chisq_dev(template, template_deriv, pcs, sgvals, coeffs, scatter, template_ampl, template_scatter, profile):
    def chisq_fn(shift):
        shifted_profile = fft_roll(profile, -shift)
        t_ampl = template @ shifted_profile / (template @ template)
        td_ampl = template_deriv @ shifted_profile / (template_deriv @ template_deriv)
        pc_ampls = pcs @ shifted_profile
        
        pred_td_ampl = coeffs @ pc_ampls
        chisq = (td_ampl - pred_td_ampl)**2 / (t_ampl * scatter)**2
        chisq += np.sum(pc_ampls**2 / (t_ampl * sgvals)**2)
        chisq += (t_ampl - template_ampl)**2 / template_scatter**2
        
        return chisq
    return chisq_fn
    
def toa_tdmarg(template, template_deriv, pcs, sgvals, coeffs, scatter, noise_level,
               profile, dt = 1.0, grid_size = 8, tol = np.sqrt(np.finfo(np.float64).eps)):
    '''
    Calculate the "marginalized template derivative" TOA estimate using a 
    template, principal components, and associated statistical information.
    
    `template`: The template (average profile shape).
    `template_deriv`: The derivative of the template with respect to shifts.
    `pcs`: The principal components of the profile residuals, after subtracting the
           best-fit linear combination of `template` and `template_deriv`.
    `sgvals`: Singular values corresponding to the principal components.
    `coeffs`: Coefficients in the linear combination of PC amplitudes that provides the
              best fit to the amplitude of `template_deriv`.
    `scatter`: The standard deviation of the difference between the best-fit coefficient
               of `template_deriv` and the value predicted using `coeffs`.
    `noise_level`: Standard deviation of additive noise (in the same units as the profile).
    `profile`: The profile to fit.
    `grid_size`: The number of initial samples to use in optimization.
    `tol`: Relative tolerance for optimization (in bins).
    '''
    n = len(profile)
    chisq_fn = get_chisq_tdmarg(template, template_deriv, pcs, sgvals, coeffs,
                                scatter, noise_level, profile)
    
    grid_points = np.linspace(0, n, grid_size, endpoint=False)
    grid_chisqs = [chisq_fn(x) for x in grid_points]
    grid_argmin = grid_points[np.argmin(grid_chisqs)]
    dgrid = n/grid_size
    brack = (grid_argmin - dgrid, grid_argmin, grid_argmin + dgrid)

    result = minimize_scalar(chisq_fn, method = 'Brent', bracket = brack, tol = tol)
    return result.x * dt
