import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
from numpy.random import random, randn
from numpy.typing import ArrayLike
from scipy import stats
from dataclasses import dataclass
from typing import Self

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
    data: Record array containing columns:
      * amplitude: Amplitudes of the components.
      * loc:       Locations of the component centers, in phase units.
      * width:     Widths of the components, in phase units.
                   Defined in an RMS sense. These should be the single-pulse
                   widths; template widths can be used with the factory
                   function `from_template()`.
      * fj:        Jitter parameter (std. dev. of location over `width`).
      * modindex:  Modulation index (std. dev. of amplitude over amplitude).
    
    Methods
    -------
    components(): The components of the pulse as `Subpulse` objects.
    normalize():  Normalize amplitudes to a maximum of 1.
    
    Class methods
    -------------
    from_template_widths(): Create a `PulseSpec` object using template
                            widths rather than single-pulse widths.
    '''
    data: np.recarray

    @classmethod
    def new(
        cls,
        amplitude: ArrayLike,
        loc: ArrayLike,
        fj: ArrayLike,
        modindex: ArrayLike,
        width: ArrayLike | None = None,
        fwhm: ArrayLike | None = None,
    ) -> Self:
        '''
        Generate a PulseSpec object using per-component lists of values.

        Inputs
        ------
        amplitude: Amplitudes of the components.
        loc:       Locations of the component centers, in phase units.
        width:     Widths of the components, in phase units.
                   Defined in an RMS sense. These should be the single-pulse
                   widths; template widths can be used with the factory
                   function `from_template_widths()`.
        fwhm:      Full widths at half max of the pulse components,
                   in phase units. These should be the single-pulse widths.
                   Either this or `width` must be specified (not `None`).
        fj:        Jitter parameter (std. dev. of location over `width`).
                   Per-component list of values.
        modindex:  Modulation index (std. dev. of amplitude over amplitude).
                   Per-component list of values.
        '''
        if width is None:
            if fwhm is not None:
                width = np.array(fwhm)/(2*np.sqrt(2*np.log(2)))
            else:
                raise ValueError("either `width` or `fwhm` must be specified")

        data = np.rec.fromarrays(
            [amplitude, loc, width, fj, modindex],
            names=['amplitude', 'loc', 'width', 'fj', 'modindex'],
        )
        return cls(data)

    @classmethod
    def from_template(
        cls,
        amplitude: ArrayLike,
        loc: ArrayLike,
        fj: ArrayLike,
        modindex: ArrayLike,
        width: ArrayLike | None = None,
        fwhm: ArrayLike | None = None,
    ) -> Self:
        '''
        Generate a `PulseSpec` object using template widths and amplitudes
        rather than the single pulse parameters.

        Inputs
        ------
        amplitude: Amplitudes of the components.
        loc:       Locations of the component centers, in phase units.
        width:     The widths of the pulse components, in phase units.
                   Defined in an RMS sense. These should be the template widths.
        fwhm:      Full widths at half max of the pulse components,
                   in phase units. These should be the template widths.
                   Either this or `width` must be specified (not `None`).
        fj:        Jitter parameter (std. dev. of location over `width`).
                   Per-component list of values.
        modindex:  Modulation index (std. dev. of amplitude over amplitude).
                   Per-component list of values.
        '''
        if width is None:
            if fwhm is not None:
                width = np.array(fwhm)/(2*np.sqrt(2*np.log(2)))
            else:
                raise ValueError("either `width` or `fwhm` must be specified")

        single_pulse_ampl = np.array(amplitude)*np.sqrt(1+np.array(fj)**2)
        single_pulse_width = np.array(width)/np.sqrt(1+np.array(fj)**2)
        return cls(
            amplitude=single_pulse_ampl,
            locs=loc,
            widths=single_pulse_width,
            fj=fj,
            modindex=modindex,
        )

    def components(self):
        '''
        Return an iterator yielding the components of the pulse as `Subpulse` objects.
        '''
        for rec in self.data:
            yield Subpulse(
                rec.amplitude,
                rec.loc,
                rec.width,
                rec.fj,
                rec.modindex,
            )

    def normalize(self):
        '''
        Normalize the amplitudes of the pulse components to unit maximum.
        '''
        self.data.amplitude /= np.max(self.data.amplitude)

    def template_components(self):
        '''
        Return an iterator yielding the components of the template as `Subpulse` objects.
        '''
        for c in self.components():
            c.amplitude = c.amplitude/np.sqrt(1+c.fj**2)
            c.width = c.width*np.sqrt(1+c.fj**2)
            yield c

    def template(self, phase: np.ndarray):
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

    def template_deriv(self, phase: np.ndarray):
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

    def covmat(self, phase: np.ndarray):
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
