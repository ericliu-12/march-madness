"""Logistic regression model for game prediction.

Simple, interpretable baseline. L2-regularized, with feature standardization.
Trains in <1 second on historical data.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED


class LogisticModel:
    """L2-regularized logistic regression for tournament game prediction."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model on training data."""
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X.values)
        self.model.fit(X_scaled, y.values)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict P(team A wins) for each matchup."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self) -> dict[str, float]:
        """Get absolute coefficient values as feature importance."""
        if self.feature_names is None:
            return {}
        coefs = np.abs(self.model.coef_[0])
        return dict(sorted(
            zip(self.feature_names, coefs),
            key=lambda x: x[1],
            reverse=True,
        ))
