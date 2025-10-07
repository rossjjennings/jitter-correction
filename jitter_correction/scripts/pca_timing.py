import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import svd

from ..toas import toa_fourier
from ..correction_utils import toa_pca
from ..signal import fft_roll

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
    for i, profile in enumerate(profiles):
        result = toa_fourier(template, profile)
        toas_template_only[i] = result.toa
    
    if args.nmin is not None:
        best_index = np.nan
        best_rms = np.inf
        all_n_pcs = np.arange(args.nmin, args.npcs + 1)
        all_rms = np.empty(args.npcs - args.nmin + 1)
        all_toas = np.empty((args.npcs - args.nmin + 1, profiles.shape[0]))
        for n in all_n_pcs:
            for i, profile in enumerate(profiles):
                result = toa_pca(template, pcs[:n], profile)
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
        for i, profile in enumerate(profiles):
            result = toa_pca(template, pcs[:n_pcs], profile)
            toas[i] = result.toa

    if args.plot is not None:
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
        stemlines = mpl.collections.LineCollection(stemlines)
        ax_side.axvline(0, color='C3', zorder=-1)
        ax_side.add_collection(stemlines)
        ax_side.set_ylim(-0.75, n_pcs-0.25)
        ax_side.set_xlim(-0.1*np.max(eigvals), 1.1*np.max(eigvals))
        #ax_side.invert_xaxis()
        ax_side.invert_yaxis()
        ax_side.set_xlabel('Eigenvalue')

        for i in range(n_pcs):
            ax_main.plot(phase, -4*pcs[i]+i)
        ax_main.set_ylabel('Principal components')
        ax_main.set_xlabel('Phase (cycles)')

        plt.tight_layout()
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
