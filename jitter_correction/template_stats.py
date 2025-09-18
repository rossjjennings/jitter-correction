import numpy as np
from numpy.random import random, randn
import matplotlib as mpl
import matplotlib.pyplot as plt

from .toas import toa_fourier
from .signal import fft_roll
from .gen_pulses import PulseSpec, gen_pulses, gen_profiles, gen_pseudo_profiles

if hasattr(np, "trapezoid"):
    # np.trapz was renamed to np.trapezoid in Numpy 2.0
    trapz = np.trapezoid
else:
    trapz = np.trapz

def main():
    import argparse
    import toml

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.toml', help='Configuration file (.toml)')
    parser.add_argument('-p', '--plot', action='store_true', help='Show a plot of the template')
    args = parser.parse_args()
    config = toml.load(args.config)
    spec = PulseSpec.from_fwhms(**config['pulse_spec'])
    n_bins = config['data']['n_bins']
    n_profiles = config['data']['n_profiles']
    SNR = config['data']['SNR']
    phase = np.linspace(-1/2, 1/2, n_bins, endpoint=False)

    template = spec.template(phase)
    template_deriv = np.gradient(template, phase)
    w_eff = np.sqrt(1/trapz(template_deriv**2, phase))
    toa_err = w_eff/(SNR*np.sqrt(n_bins))
    err_bins = n_bins*toa_err
    print(f'Effective width: {w_eff:g}')
    print(f'Signal-to-noise ratio: {SNR:g}')
    print(f'Number of phase bins: {n_bins:g}')
    print(f'Expected TOA error: {toa_err:g} ({err_bins:g} bins)')

    if args.plot:
        plt.plot(phase, template)
        plt.xlabel('Phase (cycles)')
        plt.ylabel('Intensity (relative to peak)')
        plt.show(block=False)

    profiles = template + randn(n_profiles, n_bins)/SNR
    toas = np.empty(n_profiles)
    for i, profile in enumerate(profiles):
        result = toa_fourier(template, profile)
        toas[i] = result.toa
    toa_std = np.std(toas)
    print(f'Simulated TOA error: {toa_std} bins')

if __name__ == '__main__':
    main()
