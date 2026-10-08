"""Probability calibrators.

These live in their own module (never run as a script) so that pickled models
always reference `src.calibration.<Class>`, regardless of how training was launched.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class IdentityCalibrator:
    """No-op: use the raw model probabilities."""

    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.asarray(p, float)


class PlattScaler:
    """Sigmoid calibration: logistic regression on the logit of the raw score."""

    def fit(self, p, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self.lr_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Monotone step-function calibration. Flexible, but needs a few thousand rows."""

    def fit(self, p, y):
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def predict(self, p):
        return self.iso_.predict(np.asarray(p, float))


CALIBRATORS = {"none": IdentityCalibrator, "platt": PlattScaler, "isotonic": IsotonicCalibrator}