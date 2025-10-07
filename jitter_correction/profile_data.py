import numpy as np
from numpy.random import randn
from scipy import stats
from dataclasses import dataclass

from .pulse_spec import PulseSpec
from .data_structures import ArrayCollection

@dataclass(slots=True)
class ProfileData(ArrayCollection):
    '''
    A set of profiles and corresponding phase information.
    '''
    phase: np.typing.ArrayLike
    profiles: np.typing.ArrayLike

def gen_pulses(phase, n_pulses = 5000, SNR = np.inf, ampl_dist = 'gamma', spec = PulseSpec()):
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
    return profiles

def gen_profiles(phase, n_profiles = 10, npprof = 1000, SNR = np.inf,
                 ampl_dist = 'gamma', spec = PulseSpec()):
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

    return profiles

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
    profiles = gen_pulses(phase, n_profiles, SNR, profile_spec)
    return profiles

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

    return profiles
