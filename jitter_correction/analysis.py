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
from .pca.results import ToaPcaResults
from .pca.regression import PCRegressionEstimator
from .pca.matching import PCMatchingEstimator
from .pca.bayesian import PCBayesianEstimator
from .utils import get_template, calc_dtoas

M = TypeVar("M", bound=Hdf5Serializable)
T = TypeVar("T", bound=Hdf5Serializable)

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
class TemplateMatchingModel(Hdf5Serializable):
    template: np.ndarray

class TemplateMatchingAnalysis(Analysis):
    def __init__(self, n_iter: int = 2):
        self.n_iter = n_iter

    def train(self, data: ProfileData) -> TemplateMatchingModel:
        template = get_template(data, n_iter=self.n_iter)
        return TemplateMatchingModel(template)

    def get_toas(
        self,
        model: TemplateMatchingModel, data: ProfileData
    ) -> ToaResults:
        estimator = TemplateMatchingEstimator(model.template)
        return estimator.estimate_toas(data)

class PCMatchingAnalysis(Analysis):
    def __init__(
        self,
        n_pcs: int,
        use_trend: bool = True,
        trend_order: int = 1,
        remove_baseline: bool = True,
    ):
        self.n_pcs = n_pcs
        self.use_trend = use_trend
        self.trend_order = trend_order
        self.remove_baseline = remove_baseline

    def train(self, training_data: ProfileData) -> PrincipalComponentModel:
        pca_model, scores, dtoas = extract_pcs(
            training_data,
            n_pcs=self.n_pcs,
            use_trend=self.use_trend,
            trend_order=self.trend_order,
            remove_baseline=self.remove_baseline,
        )
        return pca_model

    def get_toas(
        self,
        model: PrincipalComponentModel,
        data: ProfileData,
    ) -> ToaPcaResults:
        estimator = PCMatchingEstimator(model)
        return estimator.estimate_toas(data)

class PCBayesianAnalysis(Analysis):
    def __init__(
        self,
        n_pcs: int,
        use_trend: bool = True,
        trend_order: int = 1,
        remove_baseline: bool = True,
    ):
        self.n_pcs = n_pcs
        self.use_trend = use_trend
        self.trend_order = trend_order
        self.remove_baseline = remove_baseline

    def train(self, training_data: ProfileData) -> PrincipalComponentModel:
        pca_model, scores, dtoas = extract_pcs(
            training_data,
            n_pcs=self.n_pcs,
            use_trend=self.use_trend,
            trend_order=self.trend_order,
            remove_baseline=self.remove_baseline,
        )
        return pca_model

    def get_toas(
        self,
        model: PrincipalComponentModel,
        data: ProfileData,
    ) -> ToaPcaResults:
        estimator = PCBayesianEstimator(model)
        return estimator.estimate_toas(data)

@dataclass
class PCRegressionModel(Hdf5Serializable):
    pca_model: PrincipalComponentModel
    coeffs: np.ndarray

    def __iter__(self):
        yield self.pca_model
        yield self.coeffs

class PCRegressionAnalysis(Analysis):
    def __init__(
        self,
        n_pcs: int,
        use_trend: bool = False,
        trend_order: int = 1,
        remove_baseline: bool = True,
    ):
        self.n_pcs = n_pcs
        self.use_trend = use_trend
        self.trend_order = trend_order
        self.remove_baseline = remove_baseline

    def train(self, data: ProfileData) -> PCRegressionModel:
        pca_model, scores, dtoas = extract_pcs(
            data,
            n_pcs=self.n_pcs,
            use_trend=self.use_trend,
            trend_order=self.trend_order,
            remove_baseline=self.remove_baseline,
        )
        coeffs = np.linalg.solve(scores @ scores.T, scores @ dtoas)
        return PCRegressionModel(pca_model, coeffs)

    def get_toas(
        self,
        model: PCRegressionModel, data: ProfileData
    ) -> ToaPcaResults:
        pca_model, coeffs = model
        estimator = PCRegressionEstimator(pca_model, coeffs)
        return estimator.estimate_toas(data)

@dataclass
class SkewnessRegressionModel(Hdf5Serializable):
    template: np.ndarray
    predictor_coeffs: np.ndarray

    def __iter__(self):
        yield self.template
        yield self.predictor_coeffs

class SkewnessRegressionAnalysis(Analysis):
    def __init__(self, n_iter: int = 2):
        self.n_iter = n_iter

    def train(self, data: ProfileData) -> SkewnessRegressionModel:
        template = get_template(data, n_iter=self.n_iter)
        dtoas = calc_dtoas(template, data)
        skewness_coeffs = calc_skewness_coeffs(data)
        predictor_coeffs = np.polyfit(skewness_coeffs, dtoas, 1)
        return SkewnessRegressionModel(template, predictor_coeffs)

    def get_toas(
        self, model: SkewnessRegressionModel,
        data: ProfileData,
    ) -> ToaSkewnessResults:
        template, predictor_coeffs = model
        estimator = SkewnessRegressionEstimator(template, predictor_coeffs)
        return estimator.estimate_toas(data)
