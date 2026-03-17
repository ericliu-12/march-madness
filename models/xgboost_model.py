"""XGBoost model for game prediction.

Typically the strongest individual model on tabular data.
Training time: ~5-10 seconds with early stopping.
"""

import numpy as np
import pandas as pd

from config import RANDOM_SEED

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, OSError):
    HAS_XGBOOST = False


class XGBoostModel:
    """Gradient boosted trees for tournament game prediction."""

    def __init__(self):
        if not HAS_XGBOOST:
            raise ImportError(
                "xgboost is required. Install with: pip install xgboost"
            )

        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            verbosity=0,
        )
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model with early stopping on a validation split."""
        self.feature_names = list(X.columns)

        # Use last 20% as validation for early stopping
        n = len(X)
        split = int(n * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        self.model.fit(
            X_train.values, y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            verbose=False,
        )
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict P(team A wins) for each matchup."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from XGBoost."""
        if self.feature_names is None:
            return {}
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        ))
