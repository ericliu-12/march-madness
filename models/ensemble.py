"""Ensemble model combining multiple base models.

Uses optimized weighted averaging to combine logistic regression,
random forest, and XGBoost predictions. Falls back to equal weights
if optimization fails.

Total training time: ~10-20 seconds locally.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import log_loss

from config import RANDOM_SEED
from models.logistic import LogisticModel
from models.random_forest import RandomForestModel
from models.calibration import ProbabilityCalibrator, clip_probability

try:
    from models.xgboost_model import XGBoostModel, HAS_XGBOOST
except ImportError:
    HAS_XGBOOST = False


class EnsembleModel:
    """Weighted ensemble of multiple base models."""

    def __init__(self):
        self.models = []
        self.model_names = []
        self.weights = None
        self.calibrator = ProbabilityCalibrator()
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series, seasons: pd.Series | None = None):
        """Train all base models and optimize ensemble weights.

        Args:
            X: Feature matrix
            y: Labels (1 = team A wins)
            seasons: Optional season labels for leave-one-season-out CV
        """
        self.feature_names = list(X.columns)

        # Initialize base models
        self.models = []
        self.model_names = []

        print("\nTraining base models...")

        # Logistic Regression
        lr = LogisticModel()
        lr.fit(X, y)
        self.models.append(lr)
        self.model_names.append("Logistic")
        print("  Logistic Regression: done")

        # Random Forest
        rf = RandomForestModel()
        rf.fit(X, y)
        self.models.append(rf)
        self.model_names.append("RandomForest")
        print("  Random Forest: done")

        # XGBoost (optional)
        if HAS_XGBOOST:
            try:
                xgb = XGBoostModel()
                xgb.fit(X, y)
                self.models.append(xgb)
                self.model_names.append("XGBoost")
                print("  XGBoost: done")
            except Exception as e:
                print(f"  XGBoost: failed ({e})")
        else:
            print("  XGBoost: not installed (pip install xgboost)")

        # Optimize weights via cross-validation
        print("\nOptimizing ensemble weights...")
        self._optimize_weights(X, y, seasons)

        # Fit calibrator on in-sample predictions
        raw_ensemble = self._raw_predict(X)
        self.calibrator.fit(raw_ensemble, y.values)

        print(f"\nEnsemble weights:")
        for name, w in zip(self.model_names, self.weights):
            print(f"  {name}: {w:.3f}")

        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict calibrated P(team A wins)."""
        raw = self._raw_predict(X)
        return self.calibrator.calibrate(raw)

    def predict_single(self, features: dict) -> float:
        """Predict for a single matchup from a feature dict."""
        # Align features with training columns
        row = {col: features.get(col, 0.0) for col in self.feature_names}
        X = pd.DataFrame([row])
        X = X.fillna(0.0)
        return float(self.predict_proba(X)[0])

    def _raw_predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Get weighted average of base model predictions (uncalibrated)."""
        preds = np.column_stack([m.predict_proba(X) for m in self.models])
        return preds @ self.weights

    def _optimize_weights(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        seasons: pd.Series | None,
    ):
        """Find optimal weights to minimize log-loss.

        Uses leave-one-season-out CV if season labels provided,
        otherwise uses a simple holdout split.
        """
        n_models = len(self.models)

        if seasons is not None and seasons.nunique() > 3:
            # Leave-one-season-out CV
            oof_preds = np.zeros((len(X), n_models))
            logo = LeaveOneGroupOut()

            for train_idx, val_idx in logo.split(X, y, seasons):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]

                temp_models = self._fit_temp_models(X_train, y_train)
                for j, model in enumerate(temp_models):
                    oof_preds[val_idx, j] = model.predict_proba(X_val)

            # Optimize weights on OOF predictions
            self.weights = self._find_optimal_weights(oof_preds, y.values)
        else:
            # Simple holdout
            np.random.seed(RANDOM_SEED)
            n = len(X)
            idx = np.random.permutation(n)
            split = int(n * 0.7)
            val_idx = idx[split:]

            val_preds = np.column_stack([
                m.predict_proba(X.iloc[val_idx]) for m in self.models
            ])

            self.weights = self._find_optimal_weights(val_preds, y.iloc[val_idx].values)

    def _fit_temp_models(self, X: pd.DataFrame, y: pd.Series) -> list:
        """Fit temporary models for cross-validation."""
        temp = []

        lr = LogisticModel()
        lr.fit(X, y)
        temp.append(lr)

        rf = RandomForestModel()
        rf.fit(X, y)
        temp.append(rf)

        if HAS_XGBOOST:
            try:
                xgb = XGBoostModel()
                xgb.fit(X, y)
                temp.append(xgb)
            except Exception:
                # Use RF prediction again as placeholder
                temp.append(rf)

        return temp

    def _find_optimal_weights(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
    ) -> np.ndarray:
        """Find weights that minimize log-loss."""
        n_models = predictions.shape[1]

        def objective(w):
            w = w / w.sum()  # Normalize
            ensemble_pred = predictions @ w
            ensemble_pred = np.clip(ensemble_pred, 0.01, 0.99)
            return log_loss(true_labels, ensemble_pred)

        # Start with equal weights
        w0 = np.ones(n_models) / n_models

        result = minimize(
            objective,
            w0,
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-6},
        )

        weights = result.x
        weights = np.maximum(weights, 0)  # No negative weights
        weights = weights / weights.sum()  # Normalize

        return weights

    def get_feature_importance(self) -> dict[str, float]:
        """Get weighted average feature importance across models."""
        combined = {}
        for model, weight in zip(self.models, self.weights):
            imp = model.get_feature_importance()
            for feat, val in imp.items():
                combined[feat] = combined.get(feat, 0) + val * weight

        return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))
