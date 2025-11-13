import numpy as np
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from loguru import logger
from typing import Any

from .pulse_spec import PulseSpec
from .profile_data import ProfileData, gen_profiles
from .signal import fft_roll

DM_CONST = 1/2.41e-4 # MHz**2 s cm**3 pc**-1

class RFI(metaclass=ABCMeta):
    '''
    Abstract base class for all RFI sources
    '''
    @abstractmethod
    def realize(self, phase: np.ndarray) -> np.ndarray:
        '''
        Return a realization of the RFI source.
        Should have the same shape as `phase`.
        '''
        pass

@dataclass(slots=True)
class RippleRFI(RFI):
    freq: float | np.floating # ripple cycles / pulse period
    amplitude: float | np.floating # rel. pulse peak

    def realize(self, phase: np.ndarray) -> np.ndarray:
        ripple_phase = self.freq*phase - np.random.random()
        return self.amplitude*np.cos(2*np.pi*ripple_phase)

@dataclass(slots=True)
class ImpulsiveRFI(RFI):
    amplitude: float | np.floating # rel. pulse peak
    duration: float | np.floating # in units of pulse period
    center_freq: float | np.floating # in MHz
    bandwidth: float | np.floating # in MHz
    rate: float | np.floating # in MHz

    dm: float | np.floating # in pc/cm**3
    period: float | np.floating # in s

    def realize(self, phase: np.ndarray) -> np.ndarray:
        min_lag = DM_CONST*self.dm/(self.center_freq + self.bandwidth/2)**2
        max_lag = DM_CONST*self.dm/(self.center_freq + self.bandwidth/2)**2
        dt = (phase[-1] - phase[-2])*self.period
        length = self.period + max_lag - min_lag + self.duration - dt
        logger.debug(f'Length is {length}')
        logger.debug(f'Max lag is {max_lag}')
        logger.debug(f'Min lag is {min_lag}')
        logger.debug(f'RFI duration is {self.duration}')

        rfi = np.zeros_like(phase)
        n_rfi = np.random.poisson(self.rate*length/self.period)
        logger.debug(f'Profile has {n_rfi} RFI instances')
        for j in range(n_rfi):
            t1 = np.random.random()*length + min_lag
            t0 = t1 - self.duration
            logger.debug(f'  RFI {j} has t0={t0}, t1={t1}')
            time = (phase + 0.5)*self.period
            first_bin = np.min(np.where(time > t0 - max_lag))
            last_bin = np.max(np.where(time <= t1 - min_lag))
            time_slice = time[first_bin:last_bin+1]
            logger.debug(f'  Freq: {self.center_freq} MHz')
            if self.dm == 0:
                top = self.center_freq + self.bandwidth/2
            else:
                denom = ((t0 - time_slice < 0)*(DM_CONST*self.dm)
                            /(self.center_freq + self.bandwidth/2)**2
                        + (t0 - time_slice > 0)*(t0 - time_slice))
                top = np.minimum(
                    np.sqrt(DM_CONST*self.dm/denom),
                    self.center_freq + self.bandwidth/2,
                )
            bottom = np.maximum(
                np.sqrt(DM_CONST*self.dm/(t1 - time_slice)),
                self.center_freq - self.bandwidth/2,
            )
            logger.debug(f'  Top: {top} MHz')
            logger.debug(f'  Bottom: {bottom} MHz')
            rfi[first_bin:last_bin+1] += (
                self.amplitude*(top - bottom)/self.bandwidth
            )
        return rfi

@dataclass(slots=True)
class ProfileModel:
    '''
    Encodes configuration needed to generate pulse profiles.
    '''
    spec: PulseSpec
    n_profiles: int | np.integer # number of profiles to generate
    npprof: int | np.integer # number of pulses per profile
    n_bins: int | np.integer # number of phase bins
    snr: float | np.floating # signal-to-noise ratio (determines noise level)
    ampl_dist: str # amplitude distribution ('gamma' or 'lognorm')
    drift_bins: float | np.floating # phase drift from beginning to end, in bins
    rfi: list[RFI] # RFI to add

    def __init__(
        self,
        spec: PulseSpec,
        n_profiles: int | np.integer,
        npprof: int | np.integer,
        n_bins: int | np.integer,
        snr: float | np.floating,
        ampl_dist: str = 'gamma',
        drift_bins: float | np.floating = 0.,
        rfi: list[RFI] | None = None,
    ):
        '''
        Allow leaving out the RFI list
        '''
        if rfi is None:
            rfi = []
        self.spec = spec
        self.n_profiles = n_profiles
        self.npprof = npprof
        self.n_bins = n_bins
        self.snr = snr
        self.ampl_dist = ampl_dist
        self.drift_bins = drift_bins
        self.rfi = rfi

    @property
    def phase(self) -> np.ndarray:
        return np.linspace(-1/2, 1/2, self.n_bins, endpoint=False)

    @property
    def shifts(self) -> np.ndarray:
        return np.linspace(
            -self.drift_bins/2,
            self.drift_bins/2,
            self.n_profiles,
            endpoint=False
        )

    def generate_data(self) -> ProfileData:
        '''
        Generate profile data based on this model.
        '''
        data = gen_profiles(
            spec=self.spec,
            phase=self.phase,
            n_profiles=self.n_profiles,
            npprof=self.npprof,
            snr=self.snr,
            ampl_dist=self.ampl_dist,
        )

        profiles = np.empty_like(data.profiles)
        for i, profile in enumerate(data.profiles):
            profiles[i] = fft_roll(profile, self.shifts[i])

        for source in self.rfi:
            for i in range(len(profiles)):
                profiles[i] += source.realize(self.phase)

        return ProfileData(self.phase, profiles)

def gen_data(
    spec: PulseSpec,
    n_profiles: int | np.integer,
    npprof: int | np.integer,
    n_bins: int | np.integer,
    snr: float | np.floating,
    drift_bins: float | np.floating,
) -> ProfileData:
    '''
    Generated simulated data based on a pulse specification.
    '''
    phase = np.linspace(-1/2, 1/2, n_bins, endpoint=False)
    data = gen_profiles(spec, phase, n_profiles=n_profiles, npprof=npprof, snr=snr)
    profiles = data.profiles

    shifts = drift_bins/n_profiles*np.arange(n_profiles)
    shifts -= np.mean(shifts)
    for i, profile in enumerate(profiles):
        profiles[i] = fft_roll(profile, shifts[i])

    return ProfileData(phase, profiles)

def gen_data_from_config(config: dict[str, Any]) -> ProfileData:
    """
    Generate profiles based on configuration data, which may be loaded from a
    TOML configuration file or passed in directly as a dictionary.
    """
    spec = PulseSpec.new(**config['pulse_spec'])
    data = gen_data(spec, **config['data'])
    profiles = data.profiles
    if 'ripple' in config:
        ripple_freq = config['ripple']['freq']
        ripple_ampl = config['ripple']['amplitude']

        for i, profile in enumerate(profiles):
            ripple_phase = ripple_freq*data.phase - np.random.random()
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
            n_rfi = np.random.poisson(rfi_rate*length/period)
            #print(f'Profile {i} has {n_rfi} RFI instances')
            for j in range(n_rfi):
                t1 = np.random.random()*length + min_lag
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
