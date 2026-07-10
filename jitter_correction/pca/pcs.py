'''
Methods for working with principal components
'''
import numpy as np
from numpy.typing import ArrayLike
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg
from dataclasses import dataclass
from loguru import logger

from ..signal import fft_roll
from ..toas import TemplateMatchingEstimator
from ..mixins import NpzSerializable, Hdf5Serializable
from ..utils import get_template
from ..profile_data import ProfileData

@dataclass(slots=True)
class PrincipalComponentModel(NpzSerializable, Hdf5Serializable):
    '''
    A model derived using principal component analysis.
    Includes the template, principal components, and eigenvalues.

    Data attributes
    ---------------
    phase:    Array of pulse phase values with shape `(n,)`, where `n` is the
              number of phase bins. Can be used as an x-axis for plotting.
    template: The best-fit model profile shape, as an array with shape `(n,)`.
    pcs:      Principal components, as an array with shape `(k, n)`, where
              `k` is the number of principal components.
    eigvals:  Eigenvalues corresponding to each principal component, as an
              array with shape `(k,)`.
    '''
    phase: np.ndarray
    template: np.ndarray
    pcs: np.ndarray
    eigvals: np.ndarray

    @property
    def n_pcs(self) -> int:
        return self.pcs.shape[0]

    def truncate(self, n_pcs: int | np.integer) -> PrincipalComponentModel:
        '''
        Return a new `PrincipalComponentModel` with a truncated list of
        principal components and eigenvalues.
        '''
        return PrincipalComponentModel(
            phase=self.phase,
            template=self.template,
            pcs=self.pcs[:n_pcs],
            eigvals=self.eigvals[:n_pcs],
        )

def extract_pcs(
        data: ProfileData,
        n_pcs: int | np.integer | None,
        initial_template: np.ndarray | None = None,
        return_all: bool = False,
        use_trend: bool | np.bool_ = True,
        trend_order: int | np.integer = 1,
        remove_baseline: bool | np.bool_ = True,
    ) -> tuple[PrincipalComponentModel, np.ndarray, np.ndarray]:
    '''
    Extract a template and principal components from a set of profiles.
    An initial template can be supplied; if not, the default strategy is to average
    the middle 10 percent of profiles to get an initial template.

    Inputs
    ------
    data: ProfileData object containing the profiles.
    n_pcs: The number of principal components to use in the model.
        If `None`, choose automatically based on Bayesian Information Criterion.
    n_iter: The number of iterations to perform.
    initial_template: The initial template (see above).
    return_all: Return all principal components (instead of the first `n_pcs`).
    use_trend: If `False`, align profiles using their individual TOAs,
        If `True` (the default), align using a polynomial trend.
    trend_order: Degree of the trend polynomial to be fit. The trend is always
        used to calculate the dtoas. If `use_trend` is `True`, it is also
        used to align the profiles.
    remove_baseline: If `True`, fit and subtract a constant offset, in addition
        to a component proportional to the (shifted) template, when calculating
        profile residuals.

    Outputs
    -------
    model: PrincipalComponentModel object containing the template,
        principal components, and eigenvalues extracted from the data.
    scores: Scores for each profile and each principal component.
        Shape is `(k, n)`, where `k` is the number of principal components
        and `n` is the number of profiles.
    dtoas: Differences between the estimated TOAs and the best-fit
        polynomial trend.
    '''
    if initial_template is None:
        initial_template = get_template(data, n_iter=0)

    # Compute basic template matching TOAs
    estimator = TemplateMatchingEstimator(initial_template)
    toas = estimator.estimate_toas(data).toa

    # Fit polynomial trend (if use_trend=False, this is only used for dtoas)
    trend_coeffs = np.polyfit(data.profile_number, toas, trend_order)
    trend = np.polyval(trend_coeffs, data.profile_number)

    # Align profiles
    profiles_aligned = np.empty_like(data.profiles)
    for j, profile in enumerate(data.profiles):
        if use_trend:
            profiles_aligned[j] = fft_roll(profile, -trend[j])
        else:
            profiles_aligned[j] = fft_roll(profile, -toas[j])

    # Compute profile residuals
    resids = np.empty_like(data.profiles)
    template = np.mean(profiles_aligned, axis=0)
    columns = [template]
    if remove_baseline:
        columns.append(np.ones_like(template))
    design_matrix = np.array(columns).T
    for j, profile in enumerate(profiles_aligned):
        params = linalg.solve(
            design_matrix.T @ design_matrix,
            design_matrix.T @ profile,
        )
        resids[j] = profile - design_matrix @ params

    # Find principal components using Scipy SVD
    u, s, pcs = linalg.svd(resids, full_matrices=return_all)
    logger.debug("pcs shape: {}", pcs.shape)
    eigvals = s**2/data.n_profiles
    logger.debug("eigvals shape: {}", eigvals.shape)
    if n_pcs is None:
        # Use BIC to determine n_pcs automatically
        bic_vals = []
        for k in np.arange(eigvals.shape[0] - 1):
            trailing_eigvals = eigvals[k:-1]
            log_geomean = np.mean(np.log(trailing_eigvals))
            arithmean = np.mean(trailing_eigvals)
            n = data.n_profiles
            p = eigvals.shape[0] - 1
            bic = -n*(p - k)*(log_geomean - np.log(arithmean))
            bic += k/2*(2*p - k + 1)*np.log(n)
            bic_vals.append(bic)
        bic_vals = np.array(bic_vals)
        n_pcs = np.argmin(bic_vals)
        logger.info("Found {} significant PCs using BIC", n_pcs)
    if return_all:
        n_pcs = eigvals.shape[0]
    eigvals, pcs = eigvals[:n_pcs], pcs[:n_pcs,:]
    logger.debug("pcs shape after truncation: {}", pcs.shape)
    logger.debug("eigvals shape after truncation: {}", eigvals.shape)

    # Compute return values
    scores = np.dot(pcs, profiles_aligned.T)
    dtoas = toas - trend
    model = PrincipalComponentModel(data.phase, template, pcs, eigvals)

    return model, scores, dtoas, bic_vals

def plot_pcs(
        model: PrincipalComponentModel,
        n_pcs: int | np.integer,
        fig: plt.Figure | None = None,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes]]:
    '''
    Plot principal components.
    '''
    if fig is None:
        fig = plt.figure(figsize=(5.4, 4.8))
    gs = mpl.gridspec.GridSpec(
        nrows=2, ncols=2, width_ratios=(1.0, 0.25), height_ratios=(0.35, 1.0)
    )
    spec1 = gs[0,0]
    spec2 = gs[0,1]
    spec3 = gs[1,0]
    spec4 = gs[1,1]
    ax_main = fig.add_subplot(spec3)
    ax_side = fig.add_subplot(spec4, sharey=ax_main)
    ax_side.tick_params(axis='y', which='both', labelleft=False)
    ax_top = fig.add_subplot(spec1, sharex=ax_main)
    ax_top.tick_params(axis='x', which='both', labelbottom=False)

    ax_top.plot(model.phase, model.template)
    ax_top.set_ylabel('Mean')

    ax_side.scatter(model.eigvals, np.arange(1, len(model.eigvals)+1))
    stemlines = [((0, i), (eigval, i)) for i, eigval in enumerate(model.eigvals)]
    eigvals_geom_center = np.sqrt(model.eigvals[0]*model.eigvals[n_pcs-1])
    eigvals_span = model.eigvals[0]/eigvals_geom_center
    xlim_low = eigvals_geom_center/eigvals_span**1.25
    xlim_high = eigvals_geom_center*eigvals_span**1.25
    ax_side.set_xlim(xlim_low, xlim_high)
    ax_side.set_xscale('log')
    ax_side.set_xlabel('Eigenvalue')
    ax_side.set_xticks([1e-2, 1e0])
    ax_side.set_xticklabels([r'$10^{-2}$', '1'])

    for i in range(n_pcs):
        ax_main.plot(model.phase, -4*model.pcs[i]+i+1)
    ax_main.set_ylabel('Principal components')
    ax_main.set_yticks(np.arange(1, n_pcs+1))
    ax_main.set_ylim(n_pcs + 0.75, 0.25)
    ax_main.set_xlabel('Phase (cycles)')

    plt.minorticks_on()
    plt.tight_layout()

    return fig, (ax_top, ax_main, ax_side)
