import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import svd

from .toas import toa_fourier
from .correction_utils import toa_pca, plot_pcs
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
