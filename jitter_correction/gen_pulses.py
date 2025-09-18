import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
from numpy.random import random, randn
from scipy import stats

class Subpulse:
    '''
    One component of a pulse.
    
    Constructor parameters
    ----------------------
    amplitude : Amplitude of the component.
    loc       : Location of the component center, in phase units.
    width     : Width of the component, in phase units.
                Defined in an RMS sense.
    fj        : Jitter parameter (std. dev. of location over `width`).
    modindex  : Modulation index (std. dev. of amplitude over `amplitude`).
    '''
    def __init__(self, amplitude=1., loc=0., width=0.1, fj=0.1, modindex=1.):
        self.amplitude = amplitude
        self.loc = loc
        self.width = width
        self.fj = fj
        self.modindex = modindex

class PulseSpec:
    '''
    Specification of a multi-component Gaussian model for generating pulses.
    
    Constructor parameters
    ----------------------
    amplitudes : Amplitudes of the components.
    locs       : Locations of the component centers, in phase units.
    widths     : Widths of the components, in phase units.
                 Defined in an RMS sense. These should be the single-pulse
                 widths; template widths can be used with the factory
                 function `from_template_widths()`.
    fj         : Jitter parameter (std. dev. of location over `width`).
                 Per-component list of values.
    modindex   : Modulation index (std. dev. of amplitude over amplitude).
                 Per-component list of values.
    
    Methods
    -------
    components() : The components of the pulse as `Subpulse` objects.
    normalize()  : Normalize amplitudes to a maximum of 1.
    
    Class methods
    -------------
    from_template_widths(): Create a `PulseSpec` object using template
                            widths rather than single-pulse widths.
    '''
    def __init__(self, amplitudes=[1., 0.4], locs=[-0.06, 0.06],
                 widths=[0.05, 0.05], fj=[0.1, 0.1], modindex=[1., 1.]):
        
        if not (len(amplitudes) == len(locs) == len(widths) == len(fj) == len(modindex)):
            err_msg = """
            lengths of 'amplitudes', 'locs', 'widths', 'fj', and 'modindex' should match.
            """
            raise ValueError(err_msg)
        
        self.amplitudes = amplitudes
        self.locs = locs
        self.widths = widths
        self.fj = fj
        self.modindex = modindex
    
    def components(self):
        '''
        Return an iterator yielding the components of the pulse as `Subpulse` objects.
        '''
        for parameters in zip(self.amplitudes, self.locs, self.widths, self.fj, self.modindex):
            yield Subpulse(*parameters)
    
    def normalize(self):
        '''
        Normalize the amplitudes of the pulse components to unit maximum.
        '''
        max_amplitude = max(amplitudes)
        amplitudes = [amplitude/max_amplitude for amplitude in amplitudes]
    
    def template_components(self):
        '''
        Return an iterator yielding the components of the template as `Subpulse` objects.
        '''
        for c in self.components():
            c.amplitude = c.amplitude/np.sqrt(1+c.fj**2)
            c.width = c.width*np.sqrt(1+c.fj**2)
            yield c
    
    def template(self, phase):
        '''
        Return the template shape given by this pulse specification.
        
        Inputs
        ------
        phase : Array of phase values at which to evaluate the template.
        '''
        template = np.zeros_like(phase)
        for c in self.template_components():
            template += c.amplitude*exp(-(phase-c.loc)**2/(2*c.width**2))
        return template
    
    def template_deriv(self, phase):
        '''
        Return the derivative of the template given by this pulse specification.
        
        Inputs
        ------
        phase: Array of phase values at which to evaluate the template.
        '''
        template_deriv = np.zeros_like(phase)
        for c in self.template_components():
            template_deriv += (-c.amplitude*(phase-c.loc)/c.width**2
                               * exp(-(phase-c.loc)**2/(2*c.width**2)))
        return template_deriv
    
    def covmat(self, phase):
        '''
        Return the covariance matrix of the pulses given by this pulse specification.
        
        Inputs
        ------
        phase: Array of phase values at which to evaluate the covariance matrix.
        '''
        xx, yy = np.meshgrid(phase, phase)
        covmat = np.zeros_like(xx)
        for c in self.components():
            prefactor1 = (1 + c.modindex**2)*c.amplitude**2/(1 + 2*c.fj**2)
            prefactor2 = c.amplitude**2/(1 + c.fj**2)
            expt1 = (xx - c.loc)**2 + (yy - c.loc)**2 + c.fj**2*(xx - yy)**2
            expt1 /= -2*(1 + 2*c.fj**2)*c.width**2
            expt2 = (xx - c.loc)**2 + (yy - c.loc)**2
            expt2 /= -2*(1 + c.fj**2)*c.width**2
            covmat += prefactor1*exp(expt1) - prefactor2*exp(expt2)
        return covmat
    
    @classmethod
    def from_template(cls, amplitudes=[1., 0.4], widths=[0.05, 0.05], fj=[0.1, 0.1], **kwargs):
        '''
        Generate a `PulseSpec` object using template widths and amplitudes
        rather than the single pulse parameters.
        
        Inputs
        ------
        widths : The widths of the pulse components, in phase units.
                 Defined in an RMS sense. These should be the template widths.
        amplitudes, locs, fj, modindex : See class docstring.
        '''
        single_pulse_ampls = [ampl*np.sqrt(1+fj**2) for (ampl, fj) in zip (amplitudes, fj)]
        single_pulse_widths = [width/np.sqrt(1+fj**2) for (width, fj) in zip (widths, fj)]
        return cls(amplitudes=single_pulse_ampls, widths=single_pulse_widths, fj=fj, **kwargs)
    
    @classmethod
    def from_fwhms(cls, fwhms=[0.10, 0.10], **kwargs):
        '''
        Generate a `PulseSpec` object using the full width at half max (FWHM)
        of each component rather than the RMS width.
        
        Inputs
        ------
        fwhms : The full widths at half max of the pule components, in phase units.
                These should be the single-pulse widths.
        amplitudes, locs, fj, modindex : See class docstring.
        '''
        widths = [fwhm/(2*np.sqrt(2*np.log(2))) for fwhm in fwhms]
        return cls(widths=widths, **kwargs)
    
    @classmethod
    def from_template_fwhms(cls, amplitudes=[1., 0.4], fwhms=[0.10, 0.10], fj=[0.1, 0.1], **kwargs):
        '''
        Generate a `PulseSpec` object using the full width at half max (FWHM)
        of each component in the template, rather than the RMS width of the component
        in an individual pulse.
        
        Inputs
        ------
        fwhms : The full widths at half max of the pulse components, in phase units.
                These should be the template widths.
        amplitudes, locs, fj, modindex : See class docstring.
        '''
        single_pulse_ampls = [ampl*np.sqrt(1+fj**2) for (ampl, fj) in zip (amplitudes, fj)]
        single_pulse_widths = [fwhm/(2*np.sqrt(2*(1+fj**2)*np.log(2)))
                               for (fwhm, fj) in zip (fwhms, fj)]
        return cls(amplitudes=single_pulse_ampls, widths=single_pulse_widths, fj=fj, **kwargs)

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
        profiles += amplitudes[:,np.newaxis] * exp(-args) 
    
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
            pulses += amplitudes[:,np.newaxis] * exp(-args)
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
        smearing_factor = sqrt(1+(npprof-1)/(npprof+c.modindex**2)*c.fj**2)
        averaging_factor = sqrt((1+c.modindex**2)/(npprof+c.modindex**2))
        widths_profile.append(c.width*smearing_factor)
        fj_profile.append(c.fj*averaging_factor/smearing_factor)
        modindex_profile.append(c.modindex/sqrt(npprof))
    
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
        profiles += c.amplitude*exp(-(phase-loc)**2/(2*c.width**2))
    
    if np.any(SNR != np.inf):
        if np.ndim(SNR) != 0:
            SNR = SNR[..., np.newaxis]
        profiles += randn(*profiles_shape)/SNR
    
    return profiles
