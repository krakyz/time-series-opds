"""Statistical forecasting helpers used by the notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_columns(columns) -> list[str]:
    """Return model forecast columns from a statsforecast result dataframe."""
    ignored = {"unique_id", "ds", "cutoff", "y"}
    return [
        col
        for col in columns
        if col not in ignored and "-lo-" not in col and "-hi-" not in col
    ]


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    errors = np.where(denominator == 0, 0, np.abs(y_true - y_pred) / denominator)
    return float(np.mean(errors) * 100)


def evaluate_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Evaluate each model forecast column with MAE, RMSE and SMAPE."""
    rows = []
    y_true = forecasts["y"].to_numpy()

    for model in forecast_columns(forecasts.columns):
        y_pred = forecasts[model].to_numpy()
        rows.append({"model": model, "MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "SMAPE": smape(y_true, y_pred)})

    return pd.DataFrame(rows).sort_values("SMAPE").reset_index(drop=True)


def build_statsforecast_models(include_auto_models: bool = True, include_auto_arima: bool = False):
    """Create the statistical model set used in the final report."""
    from statsforecast.models import ARIMA, AutoARIMA, AutoETS, AutoTheta, HoltWinters, HistoricAverage, Naive, SeasonalNaive, Theta

    baseline_models = [
        HistoricAverage(alias="HistoricAverage"),
        Naive(alias="Naive"),
        SeasonalNaive(season_length=24, alias="SeasonalNaive_24h"),
        SeasonalNaive(season_length=168, alias="SeasonalNaive_168h"),
    ]
    manual_models = [
        ARIMA(order=(1, 1, 1), season_length=24, alias="ARIMA_111_24h"),
        HoltWinters(season_length=24, error_type="A", alias="HoltWinters_AAA_24h"),
        Theta(season_length=24, alias="Theta_24h"),
    ]
    auto_models = [
        AutoETS(season_length=24, alias="AutoETS_24h"),
        AutoTheta(season_length=24, alias="AutoTheta_24h"),
    ]
    auto_arima_model = AutoARIMA(
        season_length=24,
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
        nmodels=20,
        approximation=True,
        alias="AutoARIMA_fast_24h",
    )

    models = baseline_models + manual_models
    if include_auto_models:
        models += auto_models
    if include_auto_arima:
        models += [auto_arima_model]
    return models
