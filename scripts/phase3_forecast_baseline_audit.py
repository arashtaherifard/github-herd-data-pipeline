from pathlib import Path

import numpy as np
import pandas as pd

from phase3_forecast_split_audit import (
    create_repository_aware_split,
)
from phase3_split_planning import build_forecast_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3" / "audit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "future_4week_stars"

CONFIGURATIONS = [
    {
        "name": "one_one_eight",
        "validation_samples": 1,
        "test_samples": 1,
        "minimum_train_samples": 8,
    },
    {
        "name": "two_two_eight",
        "validation_samples": 2,
        "test_samples": 2,
        "minimum_train_samples": 8,
    },
    {
        "name": "three_three_eight",
        "validation_samples": 3,
        "test_samples": 3,
        "minimum_train_samples": 8,
    },
]


def heading(title: str) -> str:
    return "\n" + "=" * 100 + f"\n{title}\n" + "=" * 100


def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict:
    y_true = np.asarray(actual, dtype=float)

    y_pred = np.clip(
        np.asarray(predicted, dtype=float),
        a_min=0.0,
        a_max=None,
    )

    errors = y_pred - y_true

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    rmsle = float(
        np.sqrt(
            np.mean(
                (
                    np.log1p(y_pred)
                    - np.log1p(y_true)
                )
                ** 2
            )
        )
    )

    denominator = (
        np.abs(y_true) + np.abs(y_pred)
    )

    smape = float(
        100
        * np.mean(
            np.where(
                denominator == 0,
                0,
                2 * np.abs(errors) / denominator,
            )
        )
    )

    bias = float(np.mean(errors))

    return {
        "mae": mae,
        "rmse": rmse,
        "rmsle": rmsle,
        "smape_percent": smape,
        "forecast_bias": bias,
    }


def create_baselines(frame: pd.DataFrame) -> dict:
    return {
        # Assume the next four weeks resemble the most recent week.
        "last_week_x4": (
            frame["weekly_new_stars"] * 4
        ),

        # Use the previous observed week rather than the current week.
        "previous_week_x4": (
            frame["previous_week_stars"] * 4
        ),

        # Project the recent three-week average across four weeks.
        "rolling_3week_mean_x4": (
            frame["rolling_3week_mean_stars"] * 4
        ),

        # Assume the next four-week total equals the previous four-week total.
        "previous_4week_total": (
            frame["rolling_4week_sum_stars"]
        ),
    }


def summarize_target(
    configuration_name: str,
    split_name: str,
    frame: pd.DataFrame,
) -> dict:
    target = frame[TARGET]

    return {
        "configuration": configuration_name,
        "split": split_name,
        "rows": len(frame),
        "repositories": frame["repo_id"].nunique(),
        "mean": float(target.mean()),
        "median": float(target.median()),
        "standard_deviation": float(target.std()),
        "minimum": float(target.min()),
        "q25": float(target.quantile(0.25)),
        "q75": float(target.quantile(0.75)),
        "q90": float(target.quantile(0.90)),
        "q95": float(target.quantile(0.95)),
        "maximum": float(target.max()),
        "zero_fraction": float((target == 0).mean()),
    }


def main() -> None:
    forecast_data = build_forecast_dataset()

    metric_rows = []
    distribution_rows = []
    shift_rows = []
    split_summary_rows = []

    for configuration in CONFIGURATIONS:
        split_data, split_summary = (
            create_repository_aware_split(
                forecast_data=forecast_data,
                validation_samples=configuration[
                    "validation_samples"
                ],
                test_samples=configuration[
                    "test_samples"
                ],
                minimum_train_samples=configuration[
                    "minimum_train_samples"
                ],
            )
        )

        configuration_name = configuration["name"]

        split_summary_rows.append(
            {
                "configuration": configuration_name,
                **split_summary,
            }
        )

        train = split_data[
            split_data["split"] == "train"
        ].copy()

        validation = split_data[
            split_data["split"] == "validation"
        ].copy()

        test = split_data[
            split_data["split"] == "test"
        ].copy()

        train_p95 = float(
            train[TARGET].quantile(0.95)
        )

        train_mean = float(train[TARGET].mean())

        for split_name, frame in [
            ("train", train),
            ("validation", validation),
            ("test", test),
        ]:
            distribution_rows.append(
                summarize_target(
                    configuration_name,
                    split_name,
                    frame,
                )
            )

        for split_name, frame in [
            ("validation", validation),
            ("test", test),
        ]:
            split_mean = float(frame[TARGET].mean())

            shift_rows.append(
                {
                    "configuration": configuration_name,
                    "split": split_name,
                    "target_mean_ratio_vs_train": (
                        split_mean / train_mean
                        if train_mean
                        else None
                    ),
                    "fraction_above_train_p95": float(
                        (frame[TARGET] > train_p95).mean()
                    ),
                    "train_p95": train_p95,
                }
            )

            baselines = create_baselines(frame)

            for baseline_name, predictions in baselines.items():
                metrics = calculate_metrics(
                    frame[TARGET],
                    predictions,
                )

                metric_rows.append(
                    {
                        "configuration": configuration_name,
                        "split": split_name,
                        "baseline": baseline_name,
                        "rows": len(frame),
                        "repositories": (
                            frame["repo_id"].nunique()
                        ),
                        **metrics,
                    }
                )

    split_summary_df = pd.DataFrame(
        split_summary_rows
    )

    distribution_df = pd.DataFrame(
        distribution_rows
    )

    shift_df = pd.DataFrame(shift_rows)

    metrics_df = pd.DataFrame(metric_rows)

    validation_ranking = (
        metrics_df[
            metrics_df["split"] == "validation"
        ]
        .sort_values(
            ["rmsle", "mae"],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    split_summary_df.to_csv(
        OUTPUT_DIR
        / "forecast_baseline_split_summary.csv",
        index=False,
    )

    distribution_df.to_csv(
        OUTPUT_DIR
        / "forecast_target_distributions.csv",
        index=False,
    )

    shift_df.to_csv(
        OUTPUT_DIR
        / "forecast_distribution_shift.csv",
        index=False,
    )

    metrics_df.to_csv(
        OUTPUT_DIR
        / "forecast_baseline_metrics.csv",
        index=False,
    )

    validation_ranking.to_csv(
        OUTPUT_DIR
        / "forecast_baseline_validation_ranking.csv",
        index=False,
    )

    report_lines = [
        heading("FORECASTING SPLIT AND BASELINE AUDIT"),
        "",
        "Split configurations:",
        split_summary_df[
            [
                "configuration",
                "eligible_repositories",
                "train_rows",
                "validation_rows",
                "test_rows",
                "purged_or_unused_rows",
            ]
        ].to_string(index=False),
        "",
        heading("TARGET DISTRIBUTIONS"),
        distribution_df.to_string(index=False),
        "",
        heading("TEMPORAL DISTRIBUTION SHIFT"),
        shift_df.to_string(index=False),
        "",
        heading("BASELINE METRICS"),
        metrics_df.to_string(index=False),
        "",
        heading("VALIDATION BASELINE RANKING"),
        validation_ranking.to_string(index=False),
        "",
        f"Outputs saved to: {OUTPUT_DIR}",
    ]

    report = "\n".join(report_lines)

    report_path = (
        OUTPUT_DIR
        / "forecast_baseline_audit.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print(report)


if __name__ == "__main__":
    main()
