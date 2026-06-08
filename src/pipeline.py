"""End-to-end forecasting and anomaly detection pipeline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .data_preparation import load_power_weather_data
from .features import TARGET_COL, WEATHER_FEATURES, make_anomaly_dataset, make_supervised_features
from .train_ml import build_random_forest, score_model_table, score_predictions


@dataclass
class PipelineConfig:
    """Runtime configuration for the final pipeline."""

    quick_mode: bool = False
    forecast_horizon: int = 24
    test_start: str = "2019-01-01"
    target_col: str = TARGET_COL
    random_state: int = 42
    anomaly_contamination: float = 0.01
    processed_path: Path = Path("data/processed/de_hourly_power_and_weather_prepared.csv")
    raw_path: Path = Path("data/raw/de_hourly_power_and_weather.csv")
    reports_dir: Path = Path("reports")
    save_figures: bool = True
    run_synthetic_benchmark: bool = True

    @property
    def train_history_days(self) -> int | None:
        return 365 if self.quick_mode else None

    @property
    def n_estimators(self) -> int:
        return 100 if self.quick_mode else 400

    @property
    def anomaly_history_days(self) -> int | None:
        return 365 if self.quick_mode else 730


def split_train_test(feature_df: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split feature rows by time without shuffling."""
    train_full = feature_df.loc[feature_df["ds"] < config.test_start].copy()
    test = feature_df.loc[feature_df["ds"] >= config.test_start].copy()

    if config.train_history_days is None:
        train = train_full.copy()
    else:
        train_start = pd.Timestamp(config.test_start) - pd.Timedelta(days=config.train_history_days)
        train = train_full.loc[train_full["ds"] >= train_start].copy()

    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty partition")
    return train, test


def apply_anomaly_methods(
    anomaly_data: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.01,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply the anomaly detectors used by the final notebook."""
    result = anomaly_data.copy()
    x_anomaly = result[feature_cols]

    result["anomaly_zscore"] = result["z_score_24"].abs() > 3

    isolation_forest = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    result["anomaly_isolation_forest"] = isolation_forest.fit_predict(x_anomaly) == -1

    lof_neighbors = min(35, max(2, len(result) - 1))
    lof = LocalOutlierFactor(n_neighbors=lof_neighbors, contamination=contamination)
    result["anomaly_lof"] = lof.fit_predict(x_anomaly) == -1

    one_class_svm = make_pipeline(StandardScaler(), OneClassSVM(nu=contamination, gamma="scale"))
    result["anomaly_one_class_svm"] = one_class_svm.fit_predict(x_anomaly) == -1

    anomaly_cols = ["anomaly_zscore", "anomaly_isolation_forest", "anomaly_lof", "anomaly_one_class_svm"]
    result["anomaly_votes"] = result[anomaly_cols].sum(axis=1)
    result["is_anomaly"] = result["anomaly_votes"] >= 2
    return result, anomaly_cols


def synthetic_anomaly_benchmark(
    data: pd.DataFrame,
    config: PipelineConfig,
    anomaly_cols: list[str],
) -> pd.DataFrame:
    """Run a lightweight sanity-check on injected synthetic anomalies."""
    benchmark_days = 90
    synthetic_source = data.loc[data["ds"] >= config.test_start].head(24 * benchmark_days).copy()
    synthetic_source["synthetic_anomaly"] = False

    rng = np.random.default_rng(config.random_state)
    candidate_positions = np.arange(24, max(24, len(synthetic_source) - 24))
    n_synthetic = min(24, max(8, len(candidate_positions) // 100))

    if len(candidate_positions) >= n_synthetic and n_synthetic > 0:
        injected_positions = rng.choice(candidate_positions, size=n_synthetic, replace=False)
        shock_scale = synthetic_source[config.target_col].std() * 4
        shock_direction = rng.choice([-1, 1], size=n_synthetic)
        target_idx = synthetic_source.columns.get_loc(config.target_col)
        label_idx = synthetic_source.columns.get_loc("synthetic_anomaly")
        synthetic_source.iloc[injected_positions, target_idx] += shock_direction * shock_scale
        synthetic_source.iloc[injected_positions, label_idx] = True

    synthetic_labels = synthetic_source[["ds", "synthetic_anomaly"]].copy()
    synthetic_df, synthetic_feature_cols = make_anomaly_dataset(
        synthetic_source,
        target_col=config.target_col,
        test_start=config.test_start,
        anomaly_history_days=None,
    )
    synthetic_df, _ = apply_anomaly_methods(
        synthetic_df,
        synthetic_feature_cols,
        contamination=config.anomaly_contamination,
        random_state=config.random_state,
    )
    synthetic_df = synthetic_df.merge(synthetic_labels, on="ds", how="left")
    synthetic_df["synthetic_anomaly"] = synthetic_df["synthetic_anomaly"].fillna(False)

    rows = []
    for col in anomaly_cols + ["is_anomaly"]:
        rows.append(
            {
                "method": col,
                "precision": precision_score(synthetic_df["synthetic_anomaly"], synthetic_df[col], zero_division=0),
                "recall": recall_score(synthetic_df["synthetic_anomaly"], synthetic_df[col], zero_division=0),
                "f1": f1_score(synthetic_df["synthetic_anomaly"], synthetic_df[col], zero_division=0),
                "detected_count": int(synthetic_df[col].sum()),
                "synthetic_count": int(synthetic_df["synthetic_anomaly"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def save_forecast_figure(forecast_result: pd.DataFrame, output_path: Path, plot_n: int = 24 * 7) -> Path:
    """Save the forecast-vs-actual figure used in reports."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(forecast_result["target_ds"].head(plot_n), forecast_result["actual"].head(plot_n), label="Факт", linewidth=1.2)
    ax.plot(forecast_result["target_ds"].head(plot_n), forecast_result["prediction"].head(plot_n), label="Прогноз RandomForest", linewidth=1.2)
    ax.plot(forecast_result["target_ds"].head(plot_n), forecast_result["seasonal_naive_168h"].head(plot_n), label="Baseline SeasonalNaive_168h", linewidth=1.0, linestyle="--")
    ax.set_title("Pipeline-прогноз энергопотребления на первые 7 дней test")
    ax.set_xlabel("Время")
    ax.set_ylabel("Энергопотребление")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_anomaly_figure(anomaly_df: pd.DataFrame, config: PipelineConfig, output_path: Path) -> Path:
    """Save the anomaly-candidate figure used in reports."""
    import matplotlib.pyplot as plt

    plot_anomalies = anomaly_df.loc[anomaly_df["ds"] >= config.test_start].copy()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(plot_anomalies["ds"], plot_anomalies[config.target_col], label="Энергопотребление", linewidth=0.8)
    ax.scatter(
        plot_anomalies.loc[plot_anomalies["is_anomaly"], "ds"],
        plot_anomalies.loc[plot_anomalies["is_anomaly"], config.target_col],
        color="red",
        s=18,
        label="Кандидаты в аномалии, большинство методов",
    )
    ax.set_title("Pipeline-кандидаты в аномалии на test-периоде")
    ax.set_xlabel("Время")
    ax.set_ylabel("Энергопотребление")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_model_card(path_json: Path, path_md: Path, model_card: dict) -> None:
    """Write model card as JSON and Markdown."""
    path_json.write_text(json.dumps(model_card, ensure_ascii=False, indent=2), encoding="utf-8")
    best = next(row for row in model_card["metrics"] if row["model"] == "RandomForest")
    baseline = next(row for row in model_card["metrics"] if row["model"] == model_card["baseline"])
    markdown = "\n".join(
        [
            "# Model Card: RandomForest energy consumption forecaster",
            "",
            f"- Задача: прогноз энергопотребления на {model_card['forecast_horizon_hours']} часов вперед.",
            f"- Целевая переменная: `{model_card['target']}`.",
            "- Финальная модель: `RandomForestRegressor`.",
            f"- Baseline: `{model_card['baseline']}`.",
            f"- Train rows: {model_card['training_period']['rows']}.",
            f"- Test rows: {model_card['test_period']['rows']}.",
            f"- SMAPE финальной модели: {best['SMAPE']:.4f}%.",
            f"- SMAPE baseline: {baseline['SMAPE']:.4f}%.",
            "",
            "## Ограничения",
            "",
            "- Аномалии не имеют внешней разметки и трактуются как кандидаты.",
            "- Для production нужно обеспечить доступность тех же признаков, что использовались при обучении.",
            "- AutoARIMA исключена из финального протокола из-за высокой вычислительной стоимости на полном почасовом ряде.",
        ]
    )
    path_md.write_text(markdown + "\n", encoding="utf-8")


def run_pipeline(config: PipelineConfig | None = None) -> dict:
    """Run the final RandomForest forecasting and anomaly pipeline."""
    config = PipelineConfig() if config is None else config
    reports_dir = Path(config.reports_dir)
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    data = load_power_weather_data(config.processed_path, config.raw_path)
    feature_df, feature_cols = make_supervised_features(data, target_col=config.target_col, horizon=config.forecast_horizon)
    train, test = split_train_test(feature_df, config)

    x_train = train[feature_cols]
    y_train = train["y"]
    x_test = test[feature_cols]
    y_test = test["y"]

    model = build_random_forest(quick_mode=config.quick_mode, random_state=config.random_state, n_estimators=config.n_estimators)
    fit_start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    y_pred = model.predict(x_test)
    predict_seconds = time.perf_counter() - predict_start

    seasonal_naive_pred = test["seasonal_naive_168h"].to_numpy()
    forecast_scores = score_predictions(y_test, y_pred)
    seasonal_naive_scores = score_predictions(y_test, seasonal_naive_pred)
    forecast_scores_df = score_model_table(
        [
            {"model": "SeasonalNaive_168h", **seasonal_naive_scores},
            {"model": "RandomForest", **forecast_scores},
        ]
    )

    forecast_result = test[["ds", "target_ds", config.target_col, "y"]].copy()
    forecast_result = forecast_result.rename(columns={config.target_col: "consumption_at_feature_time", "y": "actual"})
    forecast_result["prediction"] = y_pred
    forecast_result["seasonal_naive_168h"] = seasonal_naive_pred
    forecast_result["residual"] = forecast_result["actual"] - forecast_result["prediction"]
    forecast_result["abs_error"] = forecast_result["residual"].abs()
    forecast_result["baseline_abs_error"] = (forecast_result["actual"] - forecast_result["seasonal_naive_168h"]).abs()

    anomaly_df, anomaly_feature_cols = make_anomaly_dataset(
        data,
        target_col=config.target_col,
        test_start=config.test_start,
        anomaly_history_days=config.anomaly_history_days,
    )
    anomaly_df, anomaly_cols = apply_anomaly_methods(
        anomaly_df,
        anomaly_feature_cols,
        contamination=config.anomaly_contamination,
        random_state=config.random_state,
    )
    anomaly_summary = anomaly_df[anomaly_cols + ["is_anomaly"]].sum().sort_values(ascending=False)

    synthetic_scores_df = pd.DataFrame()
    if config.run_synthetic_benchmark:
        synthetic_scores_df = synthetic_anomaly_benchmark(data, config, anomaly_cols)

    forecast_path = reports_dir / "04_forecast_predictions.csv"
    anomaly_path = reports_dir / "04_anomaly_candidates.csv"
    summary_path = reports_dir / "04_pipeline_summary.json"
    model_path = reports_dir / "04_random_forest_pipeline.joblib"
    model_card_json_path = reports_dir / "04_model_card.json"
    model_card_md_path = reports_dir / "04_model_card.md"
    final_summary_md_path = reports_dir / "final_summary.md"
    forecast_scores_table_path = tables_dir / "04_forecast_scores.csv"
    anomaly_counts_table_path = tables_dir / "04_anomaly_counts.csv"
    performance_table_path = tables_dir / "04_pipeline_performance.csv"
    synthetic_anomaly_table_path = tables_dir / "04_synthetic_anomaly_benchmark.csv"
    forecast_figure_path = figures_dir / "04_random_forest_forecast.png"
    anomaly_figure_path = figures_dir / "04_anomaly_candidates.png"

    forecast_result.to_csv(forecast_path, index=False)
    anomaly_df.loc[anomaly_df["is_anomaly"]].to_csv(anomaly_path, index=False)
    forecast_scores_df.to_csv(forecast_scores_table_path, index=False)
    anomaly_counts_to_save = anomaly_summary.rename("count").reset_index()
    anomaly_counts_to_save.columns = ["method", "count"]
    anomaly_counts_to_save.to_csv(anomaly_counts_table_path, index=False)
    performance_df = pd.DataFrame(
        [
            {"stage": "RandomForest fit", "seconds": fit_seconds, "rows": int(len(x_train)), "notes": "Final model fit"},
            {"stage": "RandomForest predict", "seconds": predict_seconds, "rows": int(len(x_test)), "notes": "Test-period inference"},
        ]
    )
    performance_df.to_csv(performance_table_path, index=False)
    if not synthetic_scores_df.empty:
        synthetic_scores_df.to_csv(synthetic_anomaly_table_path, index=False)

    if config.save_figures:
        save_forecast_figure(forecast_result, forecast_figure_path)
        save_anomaly_figure(anomaly_df, config, anomaly_figure_path)

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "forecast_horizon": config.forecast_horizon,
            "target_col": config.target_col,
            "baseline": "SeasonalNaive_168h",
            "created_from": "src.pipeline.run_pipeline",
        },
        model_path,
    )

    model_card = {
        "model_name": "RandomForest energy consumption forecaster",
        "model_type": "sklearn.ensemble.RandomForestRegressor",
        "task": "24-hour ahead hourly electricity consumption forecasting",
        "target": config.target_col,
        "forecast_horizon_hours": config.forecast_horizon,
        "training_period": {"start": str(train["ds"].min()), "end": str(train["ds"].max()), "rows": int(len(train))},
        "test_period": {"start": str(test["ds"].min()), "end": str(test["ds"].max()), "rows": int(len(test))},
        "features": feature_cols,
        "baseline": "SeasonalNaive_168h",
        "metrics": forecast_scores_df.to_dict(orient="records"),
        "known_limitations": [
            "Metrics are measured on a historical test period, not on future data.",
            "Anomalies are candidates for expert review because external labels are unavailable.",
            "Production use requires the same calendar, lag, rolling and weather features.",
        ],
    }
    write_model_card(model_card_json_path, model_card_md_path, model_card)

    final_summary_md_path.write_text(
        "# Итоговый отчет\n\n"
        f"Финальная модель `RandomForestRegressor` сравнивается с недельным baseline `SeasonalNaive_168h`. "
        f"SMAPE модели: {forecast_scores['SMAPE']:.4f}%, SMAPE baseline: {seasonal_naive_scores['SMAPE']:.4f}%.\n\n"
        "Результаты, таблицы и артефакты сохранены в `reports/`. Основной сдаваемый файл: `notebooks/05_final_report.ipynb`.\n",
        encoding="utf-8",
    )

    pipeline_summary = {
        "quick_mode": config.quick_mode,
        "forecast_horizon_hours": config.forecast_horizon,
        "final_model": "RandomForest",
        "baseline_model": "SeasonalNaive_168h",
        "forecast_scores": forecast_scores,
        "seasonal_naive_scores": seasonal_naive_scores,
        "forecast_scores_table": forecast_scores_df.to_dict(orient="records"),
        "performance_seconds": performance_df.to_dict(orient="records"),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "anomaly_contamination": config.anomaly_contamination,
        "anomaly_counts": {key: int(value) for key, value in anomaly_summary.to_dict().items()},
        "synthetic_anomaly_benchmark": synthetic_scores_df.to_dict(orient="records") if not synthetic_scores_df.empty else [],
        "forecast_path": str(forecast_path),
        "anomaly_path": str(anomaly_path),
        "summary_path": str(summary_path),
        "model_path": str(model_path),
        "model_card_json_path": str(model_card_json_path),
        "model_card_md_path": str(model_card_md_path),
        "forecast_scores_table_path": str(forecast_scores_table_path),
        "anomaly_counts_table_path": str(anomaly_counts_table_path),
        "performance_table_path": str(performance_table_path),
        "synthetic_anomaly_table_path": str(synthetic_anomaly_table_path) if not synthetic_scores_df.empty else None,
        "forecast_figure_path": str(forecast_figure_path),
        "anomaly_figure_path": str(anomaly_figure_path),
    }
    summary_path.write_text(json.dumps(pipeline_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return pipeline_summary
