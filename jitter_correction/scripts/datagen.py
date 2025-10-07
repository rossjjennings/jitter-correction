import numpy as np
from numpy.random import random, randn, poisson
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..signal import fft_roll
from ..pulse_spec import PulseSpec
from ..profile_data import gen_pulses, gen_profiles, gen_pseudo_profiles
from ..correction_utils import gen_data

def gen_data_from_config(config):
    """
    Generate profiles based on configuration data, which may be loaded from a
    TOML configuration file or passed in directly as a dictionary.
    """
    spec = PulseSpec.from_fwhms(**config['pulse_spec'])
    phase, profiles = gen_data(spec, **config['data'])
    if 'ripple' in config:
        ripple_freq = config['ripple']['freq']
        ripple_ampl = config['ripple']['amplitude']

        for i, profile in enumerate(profiles):
            ripple_phase = 2*np.pi*random()
            profiles[i] += ripple_ampl*np.cos(2*np.pi*ripple_freq*phase - ripple_phase)

    if 'rfi' in config:
        period = config['obs']['period']
        dm = config['obs']['dm']
        rfi_ampl = config['rfi']['amplitude']
        rfi_dur = config['rfi']['duration']*period
        rfi_freq = config['rfi']['center_freq']
        rfi_bw = config['rfi']['bandwidth']
        rfi_rate = config['rfi']['rate']
        dm_constant = 1/2.41e-4 # MHz**2 s cm**3 pc**-1

        min_lag = dm_constant*dm/(rfi_freq + rfi_bw/2)**2
        max_lag = dm_constant*dm/(rfi_freq - rfi_bw/2)**2
        dt = (phase[-1] - phase[-2])*period
        length = period + max_lag - min_lag + rfi_dur - dt
        #print(f'Length is {length}')
        #print(f'Max lag is {max_lag}')
        #print(f'Min lag is {min_lag}')
        #print(f'RFI duration is {rfi_dur}')
        for i, profile in enumerate(profiles):
            n_rfi = poisson(rfi_rate*length/period)
            #print(f'Profile {i} has {n_rfi} RFI instances')
            for j in range(n_rfi):
                t1 = random()*length + min_lag
                t0 = t1 - rfi_dur
                #print(f'  RFI {j} has t0={t0}, t1={t1}')
                time = (phase + 0.5)*period
                first_bin = np.min(np.where(time > t0 - max_lag))
                last_bin = np.max(np.where(time <= t1 - min_lag))
                time_slice = time[first_bin:last_bin+1]
                #print(f'  Freq: {rfi_freq} MHz')
                if dm == 0:
                    top = rfi_freq + rfi_bw/2
                else:
                    denom = ((t0 - time_slice < 0)*(dm_constant*dm)
                                /(rfi_freq + rfi_bw/2)**2
                            + (t0 - time_slice > 0)*(t0 - time_slice))
                    top = np.minimum(
                        np.sqrt(dm_constant*dm/denom),
                        rfi_freq + rfi_bw/2,
                    )
                bottom = np.maximum(
                    np.sqrt(dm_constant*dm/(t1 - time_slice)),
                    rfi_freq - rfi_bw/2,
                )
                #print(f'  Top: {top} MHz')
                #print(f'  Bottom: {bottom} MHz')
                profiles[i,first_bin:last_bin+1] += rfi_ampl*(top - bottom)/rfi_bw
    return phase, profiles

def main():
    import argparse
    import toml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.toml', help='Configuration file (.toml)')
    parser.add_argument('-p', '--plot', action='store_true', help='Display a plot of the data')
    parser.add_argument('-f', '--force-write', action='store_true', help='Overwrite datafile if it exists')
    parser.add_argument('datafile', type=str, help='Data file (.npz) to output or plot')
    args = parser.parse_args()
    config = toml.load(args.config)
    
    try:
        data = np.load(args.datafile)
        profiles = data['profiles']
        phase = data['phase']
    except FileNotFoundError:
        args.force_write = True
    
    if args.force_write:
        print(f'Writing output to {args.datafile}...')
        phase, profiles = gen_data_from_config(config)
        np.savez(args.datafile, phase=phase, profiles=profiles)
    
    if args.plot:
        for profile in profiles[:8]:
            plt.plot(phase, profile)
        plt.xlabel('Phase (cycles)')
        plt.ylabel('Intensity (rel. peak)')
        plt.show()
        
        pc = plt.pcolormesh(phase, np.arange(profiles.shape[0]), profiles)
        plt.colorbar(pc)
        plt.xlabel('Phase (cycles)')
        plt.ylabel('Profile number')
        plt.show()

if __name__ == '__main__':
    main()
