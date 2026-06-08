"""ML training and metric helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def smape(y_true, y_pred) -> float:
    """Symmetric mean absolute percentage error in percent."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    errors = np.where(denominator == 0, 0, np.abs(y_true - y_pred) / denominator)
    return float(np.mean(errors) * 100)


def score_predictions(y_true, y_pred) -> dict[str, float]:
    """Return MAE, RMSE and SMAPE for a forecast."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
    }


def build_random_forest(
    quick_mode: bool = False,
    random_state: int = 42,
    n_estimators: int | None = None,
) -> RandomForestRegressor:
    """Create the final RandomForest forecaster used by the pipeline."""
    n_estimators = (100 if quick_mode else 400) if n_estimators is None else n_estimators
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=12 if quick_mode else None,
        random_state=random_state,
        n_jobs=-1,
    )


def score_model_table(rows: list[dict]) -> pd.DataFrame:
    """Create a SMAPE-sorted dataframe from metric rows."""
    return pd.DataFrame(rows).sort_values("SMAPE").reset_index(drop=True)
