"""Model helpers."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class SeedAverage(ClassifierMixin, BaseEstimator):
    """Train the same model several times with different random seeds and average
    the predicted probabilities.

    Tree boosting uses randomness (e.g. which columns each tree sees), so two runs
    with different seeds give slightly different models. Averaging them cancels
    part of that noise and usually gives a small, reliable improvement.
    """

    def __init__(self, estimator, seeds=(42, 1, 2, 3, 4)):
        self.estimator = estimator
        self.seeds = seeds

    def fit(self, X, y):
        self.models_ = [clone(self.estimator).set_params(random_state=s).fit(X, y) for s in self.seeds]
        self.classes_ = self.models_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)
