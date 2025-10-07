import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import svd
from collections import namedtuple

from ..toas import toa_fourier
from ..signal import fft_roll
from ..pca.pcs import plot_pcs

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


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--npcs', type=int, default=None, help='Number of principal components to use (maximum)')
    parser.add_argument('-m', '--nmin', type=int, default=None, help='Minimum number of principal components to try')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot the calculated TOAs')
    parser.add_argument('-t', '--true-toas', type=str, nargs='?', help='File with true TOAs (for comparison with retrieved values)')
    parser.add_argument('calfile', type=str, help='Calibration file (.npz containing principal components)')
    parser.add_argument('infile', type=str, help='Output file (.npz containing profiles)')
    args = parser.parse_args()

    cal = np.load(args.calfile)
    template = cal['template']
    pcs = cal['pcs']
    eigvals = cal['eigvals']
    
    n_pcs = pcs.shape[0] if args.npcs is None else args.npcs

    data = np.load(args.infile)
    profiles = data['profiles']

    toas_template_only = np.empty(profiles.shape[0])
    profiles_aligned = np.empty_like(profiles)
    for i, profile in enumerate(profiles):
        result = toa_fourier(template, profile)
        toas_template_only[i] = result.toa
        profiles_aligned[i] = fft_roll(profile, -result.toa)
    
    if args.nmin is not None:
        best_index = np.nan
        best_rms = np.inf
        all_n_pcs = np.arange(args.nmin, args.npcs + 1)
        all_rms = np.empty(args.npcs - args.nmin + 1)
        all_toas = np.empty((args.npcs - args.nmin + 1, profiles.shape[0]))
        for n in all_n_pcs:
            ## No more cheating!
            tsc = cal['scores'][:n]
            training_dtoas = cal['dtoas']
            coeffs = np.linalg.inv(tsc @ tsc.T) @ tsc @ training_dtoas
            for i, profile in enumerate(profiles):
                result = toa_score(template, pcs[:n], coeffs, profile)
                all_toas[n-args.nmin,i] = result.toa
            toas = all_toas[n-args.nmin]
            true_toas = np.load(args.true_toas)['true_toas']
            rms = np.std(toas - true_toas)
            all_rms[n-args.nmin] = rms
            if rms < best_rms:
                best_index = n-args.nmin
                best_rms = rms
        toas = all_toas[best_index]
        n_pcs = all_n_pcs[best_index]
    else:
        toas = np.empty(profiles.shape[0])
        tsc = cal['scores'][:n_pcs]
        training_dtoas = cal['dtoas']
        coeffs = np.linalg.inv(tsc @ tsc.T) @ tsc @ training_dtoas
        for i, profile in enumerate(profiles):
            result = toa_score(template, pcs[:n_pcs], coeffs, profile)
            toas[i] = result.toa

    if args.plot is not None:
        fig, axes = plot_pcs(template, pcs, eigvals, n_pcs)
        plt.show()
        
        profile_number = np.arange(profiles.shape[0])
        if args.true_toas is not None:
            true_toas = np.load(args.true_toas)['true_toas']
            dtoas_template_only = toas_template_only - true_toas
            dtoas_template_only -= np.mean(dtoas_template_only)
            dtoas = toas - true_toas
            dtoas -= np.mean(dtoas)
            rms_template_only = np.std(dtoas_template_only)
            rms_pca = np.std(dtoas)
            plt.scatter(
                profile_number,
                dtoas_template_only,
                marker='.',
                label=f'Template only (RMS: {rms_template_only:.2e})',
            )
            plt.scatter(
                profile_number,
                dtoas,
                marker='.',
                label=f'Template + {n_pcs} PCs (RMS: {rms_pca:.2e})'
            )
            plt.legend()
            plt.ylabel(r'$\Delta$TOA (bins)')
        else:
            plt.scatter(
                profile_number, toas_template_only, marker='.',
                label=f'Template only',
            )
            plt.scatter(
                profile_number, toas, marker='.',
                label=f'Template + {n_pcs} PCs',
            )
            plt.legend()
            plt.ylabel('TOA (bins)')
        plt.xlabel('Profile number')
        plt.show()
        
        if args.nmin is not None:
            plt.axhline(0, color='C3')
            plt.axhline(rms_template_only, color='gray', label='Template only')
            plt.scatter(all_n_pcs, all_rms)
            plt.xlabel('Number of principal components used')
            plt.ylabel(r'RMS $\Delta$TOA')
            plt.legend()
            plt.show()

if __name__ == '__main__':
    main()
