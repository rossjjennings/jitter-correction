'''
Template-derivative-marginalized timing
'''
import numpy as np
from scipy.optimize import minimize_scalar

from ..signal import fft_roll

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
