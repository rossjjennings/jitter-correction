import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
from numpy.random import random, randn
from numpy.typing import ArrayLike
from scipy import stats
from dataclasses import dataclass

from .mixins import NpzSerializable

@dataclass(slots=True)
class Subpulse:
    '''
    One component of a pulse.
    
    Data attributes
    ---------------
    amplitude : Amplitude of the component.
    loc       : Location of the component center, in phase units.
    width     : Width of the component, in phase units.
                Defined in an RMS sense.
    fj        : Jitter parameter (std. dev. of location over `width`).
    modindex  : Modulation index (std. dev. of amplitude over `amplitude`).
    '''
    amplitude: float | np.floating
    loc: float | np.floating
    width: float | np.floating
    fj: float | np.floating
    modindex: float | np.floating

@dataclass(slots=True)
class PulseSpec(NpzSerializable):
    '''
    Specification of a multi-component Gaussian model for generating pulses.
    
    Data attributes
    ---------------
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
    amplitudes: ArrayLike
    locs: ArrayLike
    widths: ArrayLike
    fj: ArrayLike
    modindex: ArrayLike

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
