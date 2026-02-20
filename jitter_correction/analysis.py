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
from .toas import TemplateMatchingEstimator, ToaResults
from .skewness import (
    SkewnessRegressionEstimator,
    ToaSkewnessResults,
    calc_skewness_coeffs,
)
from .pca.pcs import extract_pcs
from .pca.matching import PCMatchingEstimator
from .pca.results import ToaPcaResults
from .pca.regression import PCRegressionEstimator
from .utils import get_template, calc_dtoas

M = TypeVar("M")
T = TypeVar("T")

class Analysis(Generic[M, T], ABC):
    @abstractmethod
    def train(self, data: ProfileData) -> M:
        pass

    @abstractmethod
    def get_toas(self, model: M, data: ProfileData) -> T:
        pass

@dataclass
class AnalysisResult(Generic[M, T], Hdf5Serializable):
    model: M
    toa_results: T

@dataclass
class Report(Hdf5Serializable):
    profile_model: ProfileModel
    training_data: ProfileData
    validation_data: ProfileData
    analysis_results: dict[str, AnalysisResult]

    def __getitem__(self, key):
        return self.analysis_results[key]

def run_analyses(
    profile_model: ProfileModel,
    analyses: dict[str, Analysis[M, T]],
) -> Report:
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

    return Report(
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
        estimator = TemplateMatchingEstimator(model.template)
        return estimator.estimate_toas(data)

class GtmAnalysis(Analysis[PrincipalComponentModel, ToaPcaResults]):
    def __init__(self, n_pcs: int):
        self.n_pcs = n_pcs

    def train(self, training_data: ProfileData) -> PrincipalComponentModel:
        pca_model, scores, dtoas = extract_pcs(training_data, n_pcs=self.n_pcs)
        return pca_model

    def get_toas(
        self,
        model: PrincipalComponentModel,
        data: ProfileData,
    ) -> ToaPcaResults:
        estimator = PCMatchingEstimator(model)
        return estimator.estimate_toas(data)

@dataclass
class PcaScoreModel(Hdf5Serializable):
    pca_model: PrincipalComponentModel
    coeffs: np.ndarray

    def __iter__(self):
        yield self.pca_model
        yield self.coeffs

class PcaScoreAnalysis(Analysis[PcaScoreModel, ToaPcaResults]):
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

    def get_toas(self, model: PcaScoreModel, data: ProfileData) -> ToaPcaResults:
        pca_model, coeffs = model
        estimator = PCRegressionEstimator(pca_model, coeffs)
        return estimator.estimate_toas(data)

@dataclass
class SkewnessModel(Hdf5Serializable):
    template: np.ndarray
    predictor_coeffs: np.ndarray

    def __iter__(self):
        yield self.template
        yield self.predictor_coeffs

class SkewnessAnalysis(Analysis[SkewnessModel, ToaSkewnessResults]):
    def __init__(self, n_iter: int = 2):
        self.n_iter = n_iter

    def train(self, data: ProfileData) -> SkewnessModel:
        template = get_template(data, n_iter=self.n_iter)
        dtoas = calc_dtoas(template, data)
        skewness_coeffs = calc_skewness_coeffs(data)
        predictor_coeffs = np.polyfit(skewness_coeffs, dtoas, 1)
        return SkewnessModel(template, predictor_coeffs)

    def get_toas(self, model: SkewnessModel, data: ProfileData) -> ToaSkewnessResults:
        template, predictor_coeffs = model
        estimator = SkewnessRegressionEstimator(template, predictor_coeffs)
        return estimator.estimate_toas(data)
