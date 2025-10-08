'''
Methods for working with principal components
'''
import numpy as np
from numpy.typing import ArrayLike
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import svd
from dataclasses import dataclass

from ..signal import fft_roll
from ..toas import toa_fourier
from ..mixins import NpzSerializable, Hdf5Serializable
from ..utils import get_template

@dataclass(slots=True)
class PrincipalComponentModel(NpzSerializable):
    '''
    A model derived using principal component analysis.
    Includes the template, principal components, and eigenvalues.
    '''
    phase: ArrayLike
    template: ArrayLike
    pcs: ArrayLike
    eigvals: ArrayLike

@dataclass(slots=True)
class PrincipalComponentResults(Hdf5Serializable):
    '''
    Results of performing principal component analysis on a set of profiles.
    Includes the scores (i.e., principal component values) and TOA errors (dtoas)
    as well as the PrincipalComponentModel.
    '''
    model: PrincipalComponentModel
    scores: ArrayLike
    dtoas: ArrayLike

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
