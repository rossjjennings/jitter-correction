import numpy as np
from dataclasses import dataclass

from ..mixins import RecordType, RecordContainer

@dataclass(slots=True, repr=False)
class ToaPcaResult(RecordType):
    '''
    Represents the result of fitting for a TOA and principal component scores.
    '''
    toa: np.floating
    ampl: np.floating
    offset: np.floating
    scores: np.ndarray
    sigma: np.floating
    toa_error: np.floating
    ampl_error: np.floating
    toa_ampl_corr: np.floating
    offset_error: np.floating
    score_errors: np.ndarray

@dataclass
class ToaPcaResults(RecordContainer[ToaPcaResult]):
    '''
    Represents the result of fitting for TOAs and principal component scores
    for several profiles.
    '''
    pass
