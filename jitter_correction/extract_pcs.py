import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import svd

from .toas import toa_fourier
from .correction_utils import toa_pca
from .signal import fft_roll

def extract_pcs(profiles, n_pcs=None, n_iter=2, initial_template=None, return_all=False, use_trend=True):
    '''
    Iteratively extract a template and principal components from a set of profiles.
    Each iteration uses the results of the previous iteration to improve alignment.
    An initial template can be supplied; if not, the default strategy is to average
    the middle 10 percent of profiles to get an initial template.
    
    Inputs
    ------
    profiles:   The profiles, as rows of a 2-D array.
    n_pcs:      The number of principal components to use in the model.
    n_iter:     The number of iterations to perform.
    initial_template: The initial template (see above).
    return_all: Return all principal components (instead of the first `n`).
    use_trend:  If `False`, align profiles using their individual TOAs,
                ignoring n_iter. If `True`, align using a linear trend (default).
    
    Outputs
    -------
    template: The final template
    pcs:      The final principal components
    eigvals:  The eigenvalues corresponding to the principal components.
    scores:   The PC scores of the training data.
    dtoas:    The differences between the TOAs and the linear trend.
    '''
    if initial_template is None:
        initial_template = get_initial_template(profiles)
    profile_number = np.arange(profiles.shape[0])
    
    toas = np.zeros(profiles.shape[0])
    for i, profile in enumerate(profiles):
        result = toa_fourier(initial_template, profile)
        toas[i] = result.toa
    
    resids = np.empty_like(profiles)
    if use_trend:
        for i in range(n_iter):
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
            eigvals = s**2
            
            scores = np.dot(pcs, profiles_aligned.T)
            dtoas = toas - trend

            for j, profile in enumerate(profiles):
                result = toa_pca(template, pcs[:n_pcs], profile)
                toas[j] = result.toa
    else:
        profiles_aligned = np.empty_like(profiles)
        for j, profile in enumerate(profiles):
            profiles_aligned[j] = fft_roll(profile, -toas[j])
        
        template = np.mean(profiles_aligned, axis=0)
        for j, profile in enumerate(profiles_aligned):
            ampl = np.dot(profile, template)/np.dot(template, template)
            resids[j] = profile - ampl*template
        u, s, pcs = svd(resids, full_matrices=return_all)
        eigvals = s**2
        
        # Trend used only for computing coefficients
        trend_coeffs = np.polyfit(profile_number, toas, 1)
        trend = np.polyval(trend_coeffs, profile_number)
        scores = np.dot(pcs, profiles_aligned.T)
        dtoas = toas - trend
    
    return template, pcs, eigvals, scores, dtoas

def get_initial_template(profiles):
    '''
    Get an initial template by averaging the middle 10% of profiles.
    '''
    n = profiles.shape[0]
    n_5_percent = n//20
    sl = slice(n//2 - n_5_percent, n//2 + n_5_percent)
    return np.mean(profiles[sl], axis=0)

def plot_pcs(template, pcs, eigvals, n_pcs):
    phase = np.linspace(-0.5, 0.5, pcs.shape[-1], endpoint=False)
    
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
    ax_top.set_ylabel('Template')

    ax_side.scatter(eigvals, np.arange(len(eigvals)))
    stemlines = [((0, i), (eigval, i)) for i, eigval in enumerate(eigvals)]
    #stemlines = mpl.collections.LineCollection(stemlines)
    #ax_side.axvline(0, color='C3', zorder=-1)
    #ax_side.add_collection(stemlines)
    ax_side.set_ylim(-0.75, n_pcs-0.25)
    #ax_side.set_xlim(-0.1*np.max(eigvals), 1.1*np.max(eigvals))
    eigvals_geom_center = np.sqrt(eigvals[0]*eigvals[n_pcs-1])
    eigvals_span = eigvals[0]/eigvals_geom_center
    xlim_low = eigvals_geom_center/eigvals_span**1.25
    xlim_high = eigvals_geom_center*eigvals_span**1.25
    ax_side.set_xlim(xlim_low, xlim_high)
    ax_side.set_xscale('log')
    #ax_side.invert_xaxis()
    ax_side.invert_yaxis()
    ax_side.set_xlabel('Eigenvalue')
    ax_side.set_xticks([1e-2, 1e0])
    ax_side.set_xticklabels([r'$10^{-2}$', '1'])

    for i in range(n_pcs):
        ax_main.plot(phase, -4*pcs[i]+i)
    ax_main.set_ylabel('Principal components')
    ax_main.set_xlabel('Phase (cycles)')

    plt.minorticks_on()
    plt.tight_layout()
    
    return fig, (ax_top, ax_main, ax_side)

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--npcs', nargs='?', type=int, default=6, help='Number of principal components to extract')
    parser.add_argument('-s', '--show', nargs='?', type=int, default=None,
                        help="Number of principal components to show in plot (default: don't plot)")
    parser.add_argument('-i', '--iter', nargs='?', type=int, default=2, help='Number of iterations to perform')
    parser.add_argument('-t', '--align-to-toas', action='store_true', help='Align profiles according to their individual TOAs, rather than to a linear trend fit to these')
    parser.add_argument('infile', type=str, help='Input file (.npz containing profiles)')
    parser.add_argument('outfile', type=str, help='Output file (.npz for principal components)')
    args = parser.parse_args()
    
    data = np.load(args.infile)
    profiles = data['profiles']
    return_all = True if args.show is None else args.show > args.npcs
    template, pcs, eigvals, scores, dtoas = extract_pcs(profiles, args.npcs, n_iter=args.iter, return_all=return_all, use_trend=(not args.align_to_toas))
    
    if args.show is not None:
        fig, axes = plot_pcs(template, pcs, eigvals, args.show)
        plt.show()
    
    np.savez(args.outfile, template=template, pcs=pcs, eigvals=eigvals, scores=scores, dtoas=dtoas)

if __name__ == '__main__':
    main()
