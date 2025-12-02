import numpy as np
from typing import TypeVar, Generic
from dataclasses import dataclass
from collections.abc import Callable
from abc import ABC, abstractmethod

from .mixins import Hdf5Serializable
from .pulse_spec import PulseSpec
from .profile_model import ProfileModel
from .profile_data import ProfileData
from .pca.pcs import PrincipalComponentModel
from .toas import get_toas, ToaResults
from .skewness import calc_skewness_coeffs, get_toas_skewness
from .pca.pcs import extract_pcs
from .pca.gtm import get_toas_gtm, ToaGtmResults
from .pca.score import get_toas_score, ToaScoreResults
from .utils import get_template, calc_dtoas

M = TypeVar("M", bound=Hdf5Serializable)
T = TypeVar("T", bound=Hdf5Serializable)

class Analysis(Generic[M, T], ABC):
    @abstractmethod
    def train(data: ProfileData) -> M:
        pass

    @abstractmethod
    def get_toas(model: M, data: ProfileData) -> T:
        pass

@dataclass
class AnalysisResult(Generic[M, T], Hdf5Serializable):
    model: M
    toa_results: T

@dataclass
class ComprehensiveResult(Hdf5Serializable):
    profile_model: ProfileModel
    training_data: ProfileData
    validation_data: ProfileData
    analysis_results: dict[str, AnalysisResult]

    def __getitem__(self, key):
        return self.analysis_results[key]

def run_analyses(
    profile_model: ProfileModel,
    analyses: dict[str, Analysis[M, T]],
) -> dict[str, AnalysisResult[M, T]]:
    training_data = profile_model.generate_data()
    trained_models = {}
    for name, analysis in analyses.items():
        trained_models[name] = analysis.train(training_data)

    validation_data = profile_model.generate_data()
    toa_results = {}
    for name, analysis in analyses.items():
        toa_results[name] = analysis.get_toas(
            trained_models[name],
            validation_data
        )

    analysis_results = {
        name: AnalysisResult(trained_models[name], toa_results[name])
        for name in analyses
    }

    return ComprehensiveResult(
        profile_model,
        training_data,
        validation_data,
        analysis_results,
    )

@dataclass
class TemplateOnlyModel(Hdf5Serializable):
    template: np.ndarray

class TemplateOnlyAnalysis(Analysis[TemplateOnlyModel, ToaResults]):
    def __init__(self, n_iter: int = 2):
        self.n_iter = n_iter

    def train(self, data: ProfileData) -> TemplateOnlyModel:
        template = get_template(data, n_iter=self.n_iter)
        return TemplateOnlyModel(template)

    def get_toas(self, model: TemplateOnlyModel, data: ProfileData) -> ToaResults:
        return get_toas(self, model.template, data)

class GtmAnalysis(Analysis[PrincipalComponentModel, ToaGtmResults]):
    def __init__(self, n_pcs: int):
        def train(training_data: ProfileData) -> PrincipalComponentModel:
            pca_model, scores, dtoas = extract_pcs(training_data, n_pcs=n_pcs)
            return pca_model
        self.train = train
        self.get_toas = get_toas_gtm

@dataclass
class PcaScoreModel(Hdf5Serializable):
    pca_model: PrincipalComponentModel
    coeffs: np.ndarray

    def __iter__(self):
        yield self.pca_model
        yield self.coeffs

class PcaScoreAnalysis(Analysis[PcaScoreModel, ToaGtmResults]):
    def __init__(self, n_pcs: int):
        self.n_pcs = n_pcs

    def train(self, data: ProfileData) -> PcaScoreModel:
        pca_model, scores, dtoas = extract_pcs(
            data,
            n_pcs=self.n_pcs,
            use_trend=False,
        )
        coeffs = np.linalg.solve(scores @ scores.T, scores @ dtoas)
        return PcaScoreModel(pca_model, coeffs)

    def get_toas(self, model: PcaScoreModel, data: ProfileData) -> ToaGtmResults:
        pca_model, coeffs = model
        return get_toas_score(pca_model, coeffs, data, n_pcs=self.n_pcs)

@dataclass
class SkewnessModel(Hdf5Serializable):
    template: np.ndarray
    predictor_coeffs: np.ndarray

    def __iter__(self):
        yield self.template
        yield self.predictor_coeffs

class SkewnessAnalysis(Analysis[SkewnessModel, ToaResults]):
    def __init__(self, n_iter: int = 2):
        self.n_iter = n_iter

    def train(self, data: ProfileData) -> SkewnessModel:
        template = get_template(data, n_iter=n_iter)
        dtoas = calc_dtoas(template, data)
        skewness_coeffs = calc_skewness_coeffs(data)
        predictor_coeffs = np.polyfit(skewness_coeffs, dtoas, 1)
        return SkewnessModel(template, predictor_coeffs)

    def get_toas(self, model: SkewnessModel, data: ProfileData) -> ToaResults:
        template, predictor_coeffs = model
        return get_toas_skewness(template, predictor_coeffs, data)
