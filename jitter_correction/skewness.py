import numpy as np

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
