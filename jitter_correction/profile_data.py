import numpy as np
from numpy.random import random, randn, poisson
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass

from .pulse_spec import PulseSpec
from .mixins import NpzSerializable
from .signal import fft_roll

@dataclass(slots=True)
class ProfileData(NpzSerializable):
    '''
    A set of profiles and corresponding phase information.
    '''
    phase: np.typing.ArrayLike
    profiles: np.typing.ArrayLike

    @property
    def profile_num(self):
        '''
        Array enumerating profiles. Useful for plotting purposes.
        '''
        return np.arange(self.profiles.shape[0])

    def plot(self, ax: plt.Axes | None = None):
        '''
        Create a pcolor-style plot of the data, with axis labels.
        `ax` is a pyplot.Axes object on which to plot.
        '''
        if ax is None:
            ax = plt.gca()
        pc = plt.pcolormesh(self.phase, self.profile_num, self.profiles)
        ax.set_xlabel("Phase (cycles)")
        ax.set_ylabel("Profile number")
        return pc

    def __iter__(self):
        '''
        Allow unpacking like a tuple
        '''
        yield phase
        yield profiles

def gen_pulses(phase, n_pulses=5000, SNR=np.inf, ampl_dist='gamma', spec=PulseSpec()):
    '''
    Generate synthetic pulses from a model with several Gaussian components.

    Inputs
    ------
    phase    : Phase values (between -0.5 and 0.5)
    n_pulses : Number of pulses to generate.
    SNR      : Signal-to-noise ratio. If this is `np.inf`, no noise is added.
    ampl_dist  : Distribution of amplitudes to use. Can be 'gamma' or 'lognorm'.
    spec     : Pulse specification (see `PulseSpec` class)

    Output
    ------
    pulses : Pulses, as rows of a 2D array.
    '''
    n_phase = len(phase)
    profiles = np.zeros((n_pulses, n_phase))

    for c in spec.components():
        try:
            a = 1/c.modindex**2
        except ZeroDivisionError:
            amplitudes = np.full(n_pulses, c.amplitude)
        else:
            if ampl_dist == 'gamma':
                scale = c.amplitude*c.modindex**2
                amplitudes = stats.gamma.rvs(size = n_pulses, a = a, scale = scale)
            elif ampl_dist == 'lognorm':
                s = np.sqrt(np.log(1 + c.modindex**2))
                scale = c.amplitude/np.sqrt(1 + c.modindex**2)
                amplitudes = stats.lognorm.rvs(size = n_pulses, s = s, scale = scale)
            else:
                raise ValueError(f"Amplitude distribution '{ampl_dist}' not recognized")
        jitter_rms = c.fj*c.width
        locs = c.loc + jitter_rms*randn(n_pulses)

        args = (phase - locs[:,np.newaxis])**2/(2*c.width**2)
        profiles += amplitudes[:,np.newaxis] * np.exp(-args)

    if np.any(SNR != np.inf):
        if np.ndim(SNR) != 0:
            SNR = SNR[..., np.newaxis]
        profiles += randn(n_pulses, n_phase)/SNR
    return ProfileData(phase, profiles)

def gen_profiles(phase, n_profiles=10, npprof=1000, SNR=np.inf,
                 ampl_dist='gamma', spec=PulseSpec()):
    '''
    Generate average profiles from a model with several Gaussian components.
    Averages pulses in the time domain, generating the Gaussian shape for each.

    Inputs
    ------
    phase      : Phase values (between -0.5 and 0.5)
    n_profiles : Number of profiles to generate.
    npprof     : Number of pulses to average for each profile.
    SNR        : Signal-to-noise ratio. If this is `np.inf`, no noise is added.
    ampl_dist  : Distribution of amplitudes to use. Can be 'gamma' or 'lognorm'.
    spec       : Pulse specification (see `PulseSpec` class).

    Output
    ------
    profiles : Profiles, as rows of a 2D array.
    '''
    n_phase = len(phase)
    profiles = np.empty((n_profiles, n_phase))
    pulses = np.empty((npprof, n_phase))

    for i in range(n_profiles):
        pulses.fill(0)
        for c in spec.components():
            try:
                a = 1/c.modindex**2
            except ZeroDivisionError:
                amplitudes = np.full(npprof, c.amplitude)
            else:
                if ampl_dist == 'gamma':
                    scale = c.amplitude*c.modindex**2
                    amplitudes = stats.gamma.rvs(size = npprof, a = a, scale = scale)
                elif ampl_dist == 'lognorm':
                    s = np.sqrt(np.log(1 + c.modindex**2))
                    scale = c.amplitude/np.sqrt(1 + c.modindex**2)
                    amplitudes = stats.lognorm.rvs(size = npprof, s = s, scale = scale)
                else:
                    raise ValueError(f"Amplitude distribution '{ampl_dist}' not recognized")
            jitter_rms = c.fj*c.width
            locs = c.loc + jitter_rms*randn(npprof)

            args = (phase - locs[:,np.newaxis])**2/(2*c.width**2)
            pulses += amplitudes[:,np.newaxis] * np.exp(-args)
        profiles[i,:] = np.mean(pulses, axis=0)

    if np.any(SNR != np.inf):
        if np.ndim(SNR) != 0:
            SNR = SNR[..., np.newaxis]
        profiles += randn(n_profiles, n_phase)/SNR

    return ProfileData(phase, profiles)

def gen_pseudo_profiles(phase, n_profiles = 100, npprof = 10000,
                        SNR = np.inf, spec = PulseSpec()):
    '''
    Generate synthetic "average profiles" from a model with several Gaussian
    components. Does not actually average generated pulses, but instead
    generates each profile shape once, with statistics that approximate those
    of an average of many pulses having the specified parameters.

    Inputs
    ------
    phase      : Phase values (between -0.5 and 0.5)
    n_profiles : Number of profiles to generate.
    npprof     : Number of pulses to emulate averaging for each profile.
    SNR        : Signal-to-noise ratio. If this is `np.inf`, no noise is added.
    spec       : Pulse specification (see `PulseSpec` class).

    Output
    ------
    profiles : Profiles, as rows of a 2D array.
    '''
    n_phase = len(phase)

    widths_profile, fj_profile, modindex_profile = [], [], []
    for c in spec.components():
        smearing_factor = np.sqrt(1+(npprof-1)/(npprof+c.modindex**2)*c.fj**2)
        averaging_factor = np.sqrt((1+c.modindex**2)/(npprof+c.modindex**2))
        widths_profile.append(c.width*smearing_factor)
        fj_profile.append(c.fj*averaging_factor/smearing_factor)
        modindex_profile.append(c.modindex/np.sqrt(npprof))

    profile_spec = PulseSpec(spec.amplitudes, spec.locs,
                             widths_profile, fj_profile, modindex_profile)
    data = gen_pulses(phase, n_profiles, SNR, profile_spec)
    return data

def shift_template(phase, shifts, SNR = np.inf, spec = PulseSpec()):
    '''
    Generate synthetic profiles by shifting a template.

    Inputs
    ------
    phase  : Phase values (between -0.5 and 0.5)
    shifts : Number of profiles to generate.
    npprof : Number of pulses to emulate averaging for each profile.
    SNR    : Signal-to-noise ratio. If this is `np.inf`, no noise is added.
    spec   : Pulse specification (see `PulseSpec` class).

    Output
    ------
    profiles : Profiles, as rows of a 2D array.
    '''
    n_phase = len(phase)
    shifts = np.atleast_1d(shifts)
    profiles_shape = list(shifts.shape) + [n_phase]
    profiles = np.zeros(profiles_shape)

    for c in spec.template_components():
        loc = c.loc + shifts[..., np.newaxis]
        profiles += c.amplitude*np.exp(-(phase-loc)**2/(2*c.width**2))

    if np.any(SNR != np.inf):
        if np.ndim(SNR) != 0:
            SNR = SNR[..., np.newaxis]
        profiles += randn(*profiles_shape)/SNR

    return ProfileData(phase, profiles)

def gen_data(spec, n_profiles, npprof, n_bins, SNR, drift_bins):
    '''
    Generated simulated data based on a pulse specification.
    '''
    phase = np.linspace(-1/2, 1/2, n_bins, endpoint=False)
    data = gen_profiles(phase, spec=spec, n_profiles=n_profiles, npprof=npprof, SNR=SNR)
    profiles = data.profiles

    shifts = drift_bins/n_profiles*np.arange(n_profiles)
    shifts -= np.mean(shifts)
    for i, profile in enumerate(profiles):
        profiles[i] = fft_roll(profile, shifts[i])

    return ProfileData(phase, profiles)

def gen_data_from_config(config):
    """
    Generate profiles based on configuration data, which may be loaded from a
    TOML configuration file or passed in directly as a dictionary.
    """
    spec = PulseSpec.from_fwhms(**config['pulse_spec'])
    data = gen_data(spec, **config['data'])
    profiles = data.profiles
    if 'ripple' in config:
        ripple_freq = config['ripple']['freq']
        ripple_ampl = config['ripple']['amplitude']

        for i, profile in enumerate(profiles):
            ripple_phase = ripple_freq*data.phase - random()
            profiles[i] += ripple_ampl*np.cos(2*np.pi*ripple_phase)

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
        dt = (data.phase[-1] - data.phase[-2])*period
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
                time = (data.phase + 0.5)*period
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
    return ProfileData(data.phase, profiles)
