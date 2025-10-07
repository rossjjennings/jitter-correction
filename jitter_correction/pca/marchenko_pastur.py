'''
Marchenko-Pastur distribution (null distribution of eigenvalues)
'''
import numpy as np
from scipy.optimize import brentq

def marchenko_pastur_cdf(lmbda, x):
    '''
    Calculate the cumulative distribution function for the Marchenko-Pastur distribution.
    '''
    lmbda_plus = (1 + np.sqrt(lmbda))**2
    lmbda_minus = (1 - np.sqrt(lmbda))**2

    if x == lmbda_minus:
        return 0
    elif x == lmbda_plus:
        return 1
    r = np.sqrt((lmbda_plus - x)/(x - lmbda_minus))
    cdf = 1/2 + np.sqrt((lmbda_plus - x)*(x - lmbda_minus))/(2*np.pi*lmbda)
    cdf -= (1+lmbda)*np.arctan((r**2-1)/(2*r))/(2*np.pi*lmbda)
    cdf += (1-lmbda)*np.arctan((lmbda_minus*r**2-lmbda_plus)/(2*(1-lmbda)*r))/(2*np.pi*lmbda)
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
    lmbda_plus = (1 + np.sqrt(lmbda))**2
    lmbda_minus = (1 - np.sqrt(lmbda))**2

    def objective(x):
        cdf = marchenko_pastur_cdf(lmbda, x)
        return rank*(1 - cdf) - i

    result = brentq(objective, lmbda_minus, lmbda_plus)
    return result
