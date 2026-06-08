"""Command-line entry point for the final time-series pipeline."""

from __future__ import annotations

import argparse
import json

from src.pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run energy forecasting and anomaly pipeline.")
    parser.add_argument("--quick", action="store_true", help="Use reduced quick-mode configuration.")
    parser.add_argument("--no-figures", action="store_true", help="Skip forecast and anomaly figure generation.")
    parser.add_argument("--no-synthetic-benchmark", action="store_true", help="Skip synthetic anomaly benchmark.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(
        PipelineConfig(
            quick_mode=args.quick,
            save_figures=not args.no_figures,
            run_synthetic_benchmark=not args.no_synthetic_benchmark,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
