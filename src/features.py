"""Feature engineering helpers for forecasting and anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COL = "Consumption"
WEATHER_FEATURES = [
    "temperature",
    "radiation_direct_horizontal",
    "radiation_diffuse_horizontal",
]
ENERGY_FEATURES = ["Wind", "Solar", "Wind+Solar"]
DEFAULT_LAGS = [1, 2, 3, 24, 48, 168]


def available_exogenous_features(data: pd.DataFrame) -> list[str]:
    """Return weather and energy exogenous features present in the dataframe."""
    return [col for col in WEATHER_FEATURES + ENERGY_FEATURES if col in data.columns]


def make_supervised_features(
    data: pd.DataFrame,
    target_col: str = TARGET_COL,
    horizon: int = 24,
    lags: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the t -> t+horizon supervised dataset used by ML models."""
    if horizon >= 168:
        raise ValueError("horizon must be lower than 168 hours for SeasonalNaive_168h baseline")

    features = data.copy()
    exogenous_features = available_exogenous_features(features)
    if exogenous_features:
        features[exogenous_features] = features[exogenous_features].ffill().bfill()

    lags = DEFAULT_LAGS if lags is None else lags
    features["hour"] = features["ds"].dt.hour
    features["day_of_week"] = features["ds"].dt.dayofweek
    features["month"] = features["ds"].dt.month
    features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

    for lag in lags:
        features[f"lag_{lag}"] = features[target_col].shift(lag)

    seasonal_naive_lag = 168 - horizon
    features["seasonal_naive_168h"] = features[target_col].shift(seasonal_naive_lag)
    features["rolling_mean_24"] = features[target_col].shift(1).rolling(24).mean()
    features["rolling_std_24"] = features[target_col].shift(1).rolling(24).std()
    features["rolling_mean_168"] = features[target_col].shift(1).rolling(168).mean()
    features["y"] = features[target_col].shift(-horizon)
    features["target_ds"] = features["ds"] + pd.Timedelta(hours=horizon)
    features = features.dropna().reset_index(drop=True)

    feature_cols = [
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "day_of_week",
        "is_weekend",
        *[f"lag_{lag}" for lag in lags],
        "rolling_mean_24",
        "rolling_std_24",
        "rolling_mean_168",
        *exogenous_features,
    ]
    return features, feature_cols


def make_anomaly_dataset(
    data: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_start: str = "2019-01-01",
    anomaly_history_days: int | None = 730,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the feature table used by unsupervised anomaly detectors."""
    cols = ["ds", target_col] + [col for col in WEATHER_FEATURES if col in data.columns]
    anomaly_data = data[cols].copy()

    if anomaly_history_days is not None:
        anomaly_start = pd.Timestamp(test_start) - pd.Timedelta(days=anomaly_history_days)
        anomaly_data = anomaly_data.loc[anomaly_data["ds"] >= anomaly_start].copy()

    present_weather = [col for col in WEATHER_FEATURES if col in anomaly_data.columns]
    if present_weather:
        anomaly_data[present_weather] = anomaly_data[present_weather].ffill().bfill()

    anomaly_data["hour"] = anomaly_data["ds"].dt.hour
    anomaly_data["day_of_week"] = anomaly_data["ds"].dt.dayofweek
    anomaly_data["rolling_mean_24"] = anomaly_data[target_col].rolling(24, min_periods=12).mean()
    anomaly_data["rolling_std_24"] = anomaly_data[target_col].rolling(24, min_periods=12).std()
    anomaly_data["z_score_24"] = (
        anomaly_data[target_col] - anomaly_data["rolling_mean_24"]
    ) / anomaly_data["rolling_std_24"]
    anomaly_data = anomaly_data.dropna().reset_index(drop=True)

    feature_cols = [target_col, *present_weather, "hour", "day_of_week"]
    return anomaly_data, feature_cols
