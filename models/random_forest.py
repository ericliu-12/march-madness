"""Random forest model for game prediction.

Handles nonlinear interactions naturally. Moderate training time (~2-5 seconds).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_SEED


class RandomForestModel:
    """Random forest classifier for tournament game prediction."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            min_samples_split=20,
            max_features="sqrt",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model on training data."""
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y.values)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict P(team A wins) for each matchup."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> dict[str, float]:
        """Get Gini importance for each feature."""
        if self.feature_names is None:
            return {}
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        ))
