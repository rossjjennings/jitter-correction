import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
from typing import Any
from collections.abc import Callable

from .pulse_spec import PulseSpec
from .mixins import Serializable
from .signal import fft_roll

@dataclass(slots=True)
class ProfileData(Serializable):
    '''
    A set of profiles and corresponding phase information.
    '''
    phase: np.ndarray
    profiles: np.ndarray

    @property
    def profile_number(self):
        '''
        Array enumerating profiles. Useful for plotting purposes.
        '''
        return np.arange(self.n_profiles)

    @property
    def n_profiles(self):
        '''
        The number of profiles.
        '''
        return self.profiles.shape[0]

    def plot(self, ax: plt.Axes | None = None):
        '''
        Create a pcolor-style plot of the data, with axis labels.
        `ax` is a pyplot.Axes object on which to plot.
        '''
        if ax is None:
            ax = plt.gca()
        pc = plt.pcolormesh(self.phase, self.profile_number, self.profiles)
        ax.set_xlabel("Phase (cycles)")
        ax.set_ylabel("Profile number")
        return pc

    def __iter__(self):
        '''
        Allow unpacking like a tuple
        '''
        yield self.phase
        yield self.profiles

@nb.njit
def _gamma_ampl(mean, modindex):
    '''
    A random amplitude drawn from a gamma distribution.

    Inputs
    ------
    mean:     The mean value
    modindex: The modulation index of the distribution
    '''
    if modindex == 0:
        result = mean
    else:
        shape = 1/modindex**2
        scale = mean*modindex**2
        result = np.random.gamma(shape, scale)
    return result

@nb.njit
def _lognorm_ampl(mean, modindex):
    '''
    A random amplitude drawn from a lognormal distribution.

    Inputs
    ------
    mean:     The mean value
    modindex: The modulation index of the distribution
    '''
    sigma = np.sqrt(np.log(1 + modindex**2))
    mu = np.log(mean) - 1/2*sigma**2
    result = np.random.lognormal(mu, sigma)
    return result

@nb.njit(parallel=True)
def _gen_pulses(
    spec_data: np.recarray,
    phase: np.ndarray,
    n_pulses: int | np.integer,
    snr: float | np.floating = np.inf,
    ampl_dist: Callable = _gamma_ampl,
) -> np.ndarray:
    '''
    Numba-compiled function to generate single pulses from a model with several
    Gaussian components. `gen_pulses()` is a nice object-oriented wrapper
    around this function.

    Inputs
    ------
    spec_data:  Record array containing pulse specification
                (see `PulseSpec` class).
    phase:      Phase values (between -0.5 and 0.5).
    n_pulses:   Number of pulses to generate.
    snr:        Signal-to-noise ratio. If this is `np.inf`, no noise is added.
    ampl_dist:  Distribution of amplitudes to use. Should be a Numba-compiled
                function accepting `mean` and `modindex` parameters, and
                returning a sample value.

    Output
    ------
    pulses:     Pulses, as rows of a 2D array.
    '''
    pulses = np.zeros((n_pulses, phase.shape[0]))
    for i in range(n_pulses):
        for c in spec_data:
            amplitude = ampl_dist(c.amplitude, c.modindex)
            loc = c.loc + c.fj*c.width*np.random.randn()
            chi = (phase - loc)/c.width
            pulses[i] += amplitude * np.exp(-chi**2/2)
    for k in range(phase.shape[0]):
        pulses[i,k] += np.random.randn()/snr
    return pulses

@nb.njit(parallel=True)
def _gen_profiles(
    spec_data: np.recarray,
    phase: np.ndarray,
    n_profiles: int | np.integer,
    npprof: int | np.integer,
    snr: float | np.floating = np.inf,
    ampl_dist: Callable = _gamma_ampl,
) -> np.ndarray:
    '''
    Numba-compiled function to generate profiles from a model with several
    Gaussian components. `gen_profiles()` is a nice object-oriented wrapper
    around this function.

    Inputs
    ------
    spec_data:  Record array containing pulse specification
                (see `PulseSpec` class).
    phase:      Phase values (between -0.5 and 0.5).
    n_profiles: Number of profiles to generate.
    npprof:     Number of pulses to average for each profile.
    snr:        Signal-to-noise ratio. If this is `np.inf`, no noise is added.
    ampl_dist:  Distribution of amplitudes to use. Should be a Numba-compiled
                function accepting `mean` and `modindex` parameters, and
                returning a sample value.

    Output
    ------
    profiles:   Profiles, as rows of a 2D array.
    '''
    profiles = np.zeros((n_profiles, phase.shape[0]))
    for i in nb.prange(n_profiles):
        for j in range(npprof):
            for c in spec_data:
                amplitude = ampl_dist(c.amplitude, c.modindex)
                loc = c.loc + c.fj*c.width*np.random.randn()
                chi = (phase - loc)/c.width
                profiles[i] += amplitude * np.exp(-chi**2/2)
        profiles[i] /= npprof
        for k in range(phase.shape[0]):
            profiles[i,k] += np.random.randn()/snr
    return profiles

def gen_pulses(
    spec: PulseSpec,
    phase: np.ndarray,
    n_pulses: int | np.integer = 5000,
    snr: float | np.floating = np.inf,
    ampl_dist: str = 'gamma',
) -> ProfileData:
    '''
    Generate synthetic pulses from a model with several Gaussian components.

    Inputs
    ------
    phase    : Phase values (between -0.5 and 0.5)
    n_pulses : Number of pulses to generate.
    snr      : Signal-to-noise ratio. If this is `np.inf`, no noise is added.
    ampl_dist  : Distribution of amplitudes to use. Can be 'gamma' or 'lognorm'.
    spec     : Pulse specification (see `PulseSpec` class)

    Output
    ------
    pulses : Pulses, as rows of a 2D array.
    '''
    ampl_dist_callables = {
        'gamma': _gamma_ampl,
        'lognorm': _lognorm_ampl,
    }
    try:
        ampl_dist = ampl_dist_callables[ampl_dist]
    except KeyError:
        raise ValueError(f"Amplitude distribution '{ampl_dist}' not recognized")

    pulses = _gen_pulses(
        spec_data = spec.data,
        phase = phase,
        n_pulses = n_pulses,
        snr = snr,
        ampl_dist = ampl_dist,
    )

    return ProfileData(phase, pulses)

def gen_profiles(
    spec: PulseSpec,
    phase: np.ndarray,
    n_profiles: int | np.integer = 10,
    npprof: int | np.integer = 1000,
    snr: float | np.floating = np.inf,
    ampl_dist: str = 'gamma',
) -> ProfileData:
    '''
    Generate average profiles from a model with several Gaussian components.
    Averages pulses in the time domain, generating the Gaussian shape for each.

    Inputs
    ------
    phase      : Phase values (between -0.5 and 0.5)
    n_profiles : Number of profiles to generate.
    npprof     : Number of pulses to average for each profile.
    snr        : Signal-to-noise ratio. If this is `np.inf`, no noise is added.
    ampl_dist  : Distribution of amplitudes to use. Can be 'gamma' or 'lognorm'.
    spec       : Pulse specification (see `PulseSpec` class).

    Output
    ------
    profiles : Profiles, as rows of a 2D array.
    '''
    ampl_dist_callables = {
        'gamma': _gamma_ampl,
        'lognorm': _lognorm_ampl,
    }
    try:
        ampl_dist = ampl_dist_callables[ampl_dist]
    except KeyError:
        raise ValueError(f"Amplitude distribution '{ampl_dist}' not recognized")

    profiles = _gen_profiles(
        spec.data,
        phase,
        n_profiles,
        npprof,
        snr,
        ampl_dist,
    )

    return ProfileData(phase, profiles)

def gen_pseudo_profiles(
    spec: PulseSpec,
    phase: np.ndarray,
    n_profiles: int | np.integer = 100,
    npprof: int | np.integer = 10000,
    snr: float | np.floating = np.inf,
) -> ProfileData:
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
    snr        : Signal-to-noise ratio. If this is `np.inf`, no noise is added.
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

    profile_spec = PulseSpec.new(
        spec.data.amplitude,
        spec.data.loc,
        widths_profile,
        fj_profile,
        modindex_profile
    )
    data = gen_pulses(spec, phase, n_profiles, snr)
    return data

def shift_template(
    spec: PulseSpec,
    phase: np.ndarray,
    shifts: float | np.floating | np.ndarray,
    snr: float | np.floating | np.ndarray = np.inf,
) -> ProfileData:
    '''
    Generate synthetic profiles by shifting a template.

    Inputs
    ------
    phase  : Phase values (between -0.5 and 0.5)
    shifts : Number of profiles to generate.
    npprof : Number of pulses to emulate averaging for each profile.
    snr    : Signal-to-noise ratio. If this is `np.inf`, no noise is added.
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

    snr = np.atleast_1d(snr)
    if np.any(snr != np.inf):
        if np.ndim(snr) != 0:
            snr = snr[..., np.newaxis]
        profiles += np.random.randn(*profiles_shape)/snr

    return ProfileData(phase, profiles)
