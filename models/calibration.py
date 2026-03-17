"""Probability calibration for model predictions.

Ensures predicted probabilities are well-calibrated (a predicted 70%
should win ~70% of the time). Uses isotonic regression and clips
extreme predictions to avoid catastrophic log-loss.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

from config import PROBABILITY_CEILING, PROBABILITY_FLOOR


class ProbabilityCalibrator:
    """Calibrates raw model probabilities using isotonic regression."""

    def __init__(self):
        self.calibrator = IsotonicRegression(
            y_min=PROBABILITY_FLOOR,
            y_max=PROBABILITY_CEILING,
            out_of_bounds="clip",
        )
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, true_labels: np.ndarray):
        """Fit the calibrator on validation predictions and true outcomes."""
        self.calibrator.fit(raw_probs, true_labels)
        self.is_fitted = True
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Calibrate predicted probabilities.

        If not fitted, just clips to floor/ceiling.
        """
        if self.is_fitted:
            calibrated = self.calibrator.predict(probs)
        else:
            calibrated = probs

        return np.clip(calibrated, PROBABILITY_FLOOR, PROBABILITY_CEILING)


def clip_probability(prob: float) -> float:
    """Clip a single probability to the configured range."""
    return np.clip(prob, PROBABILITY_FLOOR, PROBABILITY_CEILING)
