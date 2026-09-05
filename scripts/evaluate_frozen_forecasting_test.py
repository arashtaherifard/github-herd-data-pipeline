import hashlib
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRETEST_TAG = (
    "forecasting-pretest-freeze-v1"
)

PRETEST_MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "forecasting"
    / "pretest_freeze"
    / "pretest_freeze_manifest.json"
)

PRETEST_METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
    / "selected_forecaster_pretest_metadata.json"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "forecasting"
    / "forecast_primary_test.csv"
)

SCRIPT_PATH = Path(__file__).resolve()

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "forecasting"
    / "final_test_evaluation"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "final_test_metrics.json"
)

METRICS_TABLE_PATH = (
    OUTPUT_DIR
    / "final_test_model_comparison.csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "final_test_predictions.csv"
)

PER_REPOSITORY_PATH = (
    OUTPUT_DIR
    / "final_test_per_repository_metrics.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "final_test_report.txt"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "final_test_evaluation_manifest.json"
)

EXPECTED_ARTIFACT_VERSION = (
    "forecasting_pretest_v1"
)

EXPECTED_APPROACH_FAMILY = (
    "additive_residual_extra_trees"
)

EXPECTED_MODE = "additive_residual"

EXPECTED_MODEL_KIND = "extra_trees"

EXPECTED_TARGET = "future_4week_stars"

EXPECTED_HORIZON_DAYS = 28

IDENTIFIER_COLUMNS = [
    "repo_id",
    "repo_full_name",
    "cutoff_week",
    "target_end_week",
]

RANDOM_STATE = 42


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def write_json(
    path: Path,
    value: dict,
) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_output(
    arguments: list[str],
) -> str:
    result = subprocess.run(
        [
            "git",
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def current_git_commit() -> str:
    return git_output(
        [
            "rev-parse",
            "HEAD",
        ]
    )


def tag_target(tag_name: str) -> str:
    return git_output(
        [
            "rev-list",
            "-n",
            "1",
            tag_name,
        ]
    )


def rmsle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    true_values = np.asarray(
        y_true,
        dtype=float,
    )

    predictions = np.clip(
        np.asarray(
            y_pred,
            dtype=float,
        ),
        a_min=0.0,
        a_max=None,
    )

    return float(
        np.sqrt(
            mean_squared_log_error(
                true_values,
                predictions,
            )
        )
    )


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    true_values = np.asarray(
        y_true,
        dtype=float,
    )

    predictions = np.asarray(
        y_pred,
        dtype=float,
    )

    denominator = (
        np.abs(true_values)
        + np.abs(predictions)
    )

    numerator = (
        2.0
        * np.abs(
            predictions
            - true_values
        )
    )

    values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(
            numerator,
            dtype=float,
        ),
        where=denominator != 0,
    )

    return float(
        np.mean(values)
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    true_values = np.asarray(
        y_true,
        dtype=float,
    )

    predictions = np.clip(
        np.asarray(
            y_pred,
            dtype=float,
        ),
        a_min=0.0,
        a_max=None,
    )

    return {
        "rmsle": rmsle(
            true_values,
            predictions,
        ),
        "mae": float(
            mean_absolute_error(
                true_values,
                predictions,
            )
        ),
        "rmse": rmse(
            true_values,
            predictions,
        ),
        "smape": smape(
            true_values,
            predictions,
        ),
        "forecast_bias": float(
            np.mean(
                predictions
                - true_values
            )
        ),
        "median_absolute_error": float(
            np.median(
                np.abs(
                    predictions
                    - true_values
                )
            )
        ),
        "maximum_absolute_error": float(
            np.max(
                np.abs(
                    predictions
                    - true_values
                )
            )
        ),
    }


def verify_pretest_freeze(
) -> tuple[
    dict,
    dict,
    str,
]:
    manifest = load_json(
        PRETEST_MANIFEST_PATH
    )

    metadata = load_json(
        PRETEST_METADATA_PATH
    )

    current_commit = (
        current_git_commit()
    )

    frozen_commit = tag_target(
        PRETEST_TAG
    )

    if current_commit != frozen_commit:
        raise RuntimeError(
            "HEAD is not the frozen pretest "
            "commit. Expected "
            f"{frozen_commit}, found "
            f"{current_commit}."
        )

    if (
        manifest["freeze_name"]
        != PRETEST_TAG
    ):
        raise RuntimeError(
            "The freeze manifest name does "
            "not match the required pretest "
            "tag."
        )

    checks = {
        manifest["model_path"]: (
            manifest["model_sha256"]
        ),
        manifest["metadata_path"]: (
            manifest["metadata_sha256"]
        ),
        manifest[
            "freeze_metadata_path"
        ]: manifest[
            "freeze_metadata_sha256"
        ],
        manifest["reproduction_path"]: (
            manifest[
                "reproduction_sha256"
            ]
        ),
        **manifest["input_hashes"],
    }

    mismatches = []

    for relative_path, expected_hash in (
        checks.items()
    ):
        path = (
            PROJECT_ROOT
            / relative_path
        )

        if not path.exists():
            mismatches.append(
                {
                    "path": relative_path,
                    "reason": "missing",
                }
            )

            continue

        actual_hash = sha256_file(
            path
        )

        if actual_hash != expected_hash:
            mismatches.append(
                {
                    "path": relative_path,
                    "reason": (
                        "hash_mismatch"
                    ),
                    "expected": (
                        expected_hash
                    ),
                    "actual": actual_hash,
                }
            )

    if mismatches:
        raise RuntimeError(
            "Pretest freeze verification "
            "failed:\n"
            + json.dumps(
                mismatches,
                indent=2,
            )
        )

    if (
        metadata["freeze_status"]
        != "forecasting_pretest_frozen"
    ):
        raise RuntimeError(
            "The saved metadata is not a "
            "forecasting pretest freeze."
        )

    if (
        metadata["test_data_status"]
        != "not_loaded_or_evaluated"
    ):
        raise RuntimeError(
            "The pretest metadata does not "
            "show an unopened test set."
        )

    return (
        manifest,
        metadata,
        current_commit,
    )


def validate_artifact(
    artifact: dict[str, Any],
) -> None:
    checks = {
        "artifact_version": (
            EXPECTED_ARTIFACT_VERSION
        ),
        "approach_family": (
            EXPECTED_APPROACH_FAMILY
        ),
        "mode": EXPECTED_MODE,
        "model_kind": (
            EXPECTED_MODEL_KIND
        ),
        "target": EXPECTED_TARGET,
    }

    for key, expected in (
        checks.items()
    ):
        actual = artifact.get(key)

        if actual != expected:
            raise RuntimeError(
                f"Frozen artifact field "
                f"{key!r} changed. "
                f"Expected {expected!r}, "
                f"found {actual!r}."
            )

    required_keys = {
        "baseline_column",
        "baseline_multiplier",
        "feature_columns",
        "prediction_minimum",
        "residual_estimator",
    }

    missing = (
        required_keys
        - set(artifact)
    )

    if missing:
        raise RuntimeError(
            "Frozen model artifact is "
            f"missing: {sorted(missing)}"
        )


def prepare_test_frame(
    raw_test: pd.DataFrame,
    artifact: dict[str, Any],
) -> pd.DataFrame:
    feature_columns = artifact[
        "feature_columns"
    ]

    baseline_column = artifact[
        "baseline_column"
    ]

    required = set(
        IDENTIFIER_COLUMNS
        + feature_columns
        + [
            baseline_column,
            EXPECTED_TARGET,
        ]
    )

    missing = (
        required
        - set(raw_test.columns)
    )

    if missing:
        raise ValueError(
            "Forecast test data is missing "
            f"columns: {sorted(missing)}"
        )

    test = raw_test.copy()

    for date_column in [
        "cutoff_week",
        "target_end_week",
    ]:
        test[
            date_column
        ] = pd.to_datetime(
            test[date_column],
            utc=True,
            errors="raise",
        )

    numeric_columns = sorted(
        set(
            feature_columns
            + [
                baseline_column,
                EXPECTED_TARGET,
            ]
        )
    )

    test[
        numeric_columns
    ] = test[
        numeric_columns
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    if test[
        numeric_columns
    ].isna().any().any():
        raise ValueError(
            "Forecast test data contains "
            "missing numeric values."
        )

    if not np.isfinite(
        test[
            numeric_columns
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Forecast test data contains "
            "non-finite numeric values."
        )

    if (
        test[EXPECTED_TARGET] < 0
    ).any():
        raise ValueError(
            "Forecast test data contains "
            "negative target values."
        )

    duplicate_keys = int(
        test.duplicated(
            subset=[
                "repo_id",
                "cutoff_week",
            ]
        ).sum()
    )

    if duplicate_keys:
        raise ValueError(
            "Forecast test data contains "
            f"{duplicate_keys} duplicate "
            "repository/cutoff keys."
        )

    horizons = (
        test["target_end_week"]
        - test["cutoff_week"]
    ).dt.days

    invalid_horizons = int(
        (
            horizons
            != EXPECTED_HORIZON_DAYS
        ).sum()
    )

    if invalid_horizons:
        raise ValueError(
            "Forecast test data contains "
            f"{invalid_horizons} rows with "
            "a horizon other than 28 days."
        )

    if "split" in test.columns:
        split_values = set(
            test["split"]
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
            .tolist()
        )

        if split_values not in [
            {"test"},
            {"primary_test"},
        ]:
            raise ValueError(
                "Unexpected test split labels: "
                f"{sorted(split_values)}"
            )

    repository_sample_counts = (
        test.groupby(
            "repo_id"
        )
        .size()
    )

    if (
        repository_sample_counts
        .nunique()
        != 1
    ):
        raise ValueError(
            "Repositories do not have an "
            "equal number of test samples."
        )

    return (
        test.sort_values(
            [
                "repo_id",
                "cutoff_week",
            ]
        )
        .reset_index(drop=True)
    )


def frozen_model_predictions(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    baseline = (
        float(
            artifact[
                "baseline_multiplier"
            ]
        )
        * frame[
            artifact[
                "baseline_column"
            ]
        ].to_numpy(dtype=float)
    )

    correction = artifact[
        "residual_estimator"
    ].predict(
        frame[
            artifact[
                "feature_columns"
            ]
        ]
    )

    final_prediction = np.clip(
        baseline + correction,
        a_min=float(
            artifact[
                "prediction_minimum"
            ]
        ),
        a_max=None,
    )

    return (
        baseline,
        correction,
        final_prediction,
    )


def benchmark_predictions(
    frame: pd.DataFrame,
    frozen_baseline: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "frozen_additive_residual_extra_trees": (
            np.array([], dtype=float)
        ),
        "raw_rolling_3week_mean_x4": (
            frozen_baseline
        ),
        "scaled_rolling_baseline_1_2": (
            1.2
            * frozen_baseline
        ),
        "previous_4week_total": (
            frame[
                "rolling_4week_sum_stars"
            ].to_numpy(dtype=float)
        ),
        "last_week_x4": (
            4.0
            * frame[
                "weekly_new_stars"
            ].to_numpy(dtype=float)
        ),
        "previous_week_x4": (
            4.0
            * frame[
                "previous_week_stars"
            ].to_numpy(dtype=float)
        ),
    }


def per_repository_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        repo_id,
        repo_full_name,
    ), group in predictions.groupby(
        [
            "repo_id",
            "repo_full_name",
        ],
        sort=True,
    ):
        metrics = evaluate_predictions(
            group[
                EXPECTED_TARGET
            ].to_numpy(dtype=float),
            group[
                "predicted_future_4week_stars"
            ].to_numpy(dtype=float),
        )

        rows.append(
            {
                "repo_id": repo_id,
                "repo_full_name": (
                    repo_full_name
                ),
                "test_rows": int(
                    len(group)
                ),
                "target_mean": float(
                    group[
                        EXPECTED_TARGET
                    ].mean()
                ),
                "prediction_mean": float(
                    group[
                        "predicted_future_4week_stars"
                    ].mean()
                ),
                **metrics,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "rmsle",
                "mae",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def save_plots(
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    per_repo: pd.DataFrame,
) -> None:
    actual = predictions[
        EXPECTED_TARGET
    ].to_numpy(dtype=float)

    predicted = predictions[
        "predicted_future_4week_stars"
    ].to_numpy(dtype=float)

    maximum = float(
        max(
            actual.max(),
            predicted.max(),
        )
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.scatter(
        actual,
        predicted,
        alpha=0.8,
    )

    plt.plot(
        [0.0, maximum],
        [0.0, maximum],
        linestyle="--",
    )

    plt.xlabel(
        "Observed Future Four-Week Stars"
    )

    plt.ylabel(
        "Frozen-Model Prediction"
    )

    plt.title(
        "Final Forecast Test: "
        "Observed vs Predicted"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "final_test_observed_vs_predicted.png",
        dpi=300,
    )

    plt.close()

    residuals = (
        predicted - actual
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.scatter(
        predicted,
        residuals,
        alpha=0.8,
    )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel(
        "Frozen-Model Prediction"
    )

    plt.ylabel(
        "Forecast Error "
        "(Predicted - Observed)"
    )

    plt.title(
        "Final Forecast Test Residuals"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "final_test_residuals.png",
        dpi=300,
    )

    plt.close()

    plot_comparison = (
        comparison.sort_values(
            "rmsle",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        plot_comparison[
            "approach"
        ],
        plot_comparison[
            "rmsle"
        ],
    )

    plt.xlabel(
        "Final-Test RMSLE"
    )

    plt.ylabel("Approach")

    plt.title(
        "Frozen Forecaster and "
        "Descriptive Test Benchmarks"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "final_test_rmsle_comparison.png",
        dpi=300,
    )

    plt.close()

    repo_plot = (
        per_repo.sort_values(
            "mae",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(11, 9)
    )

    plt.barh(
        repo_plot[
            "repo_full_name"
        ],
        repo_plot["mae"],
    )

    plt.xlabel(
        "Per-Repository Test MAE"
    )

    plt.ylabel("Repository")

    plt.title(
        "Frozen Forecaster Error "
        "by Repository"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "final_test_per_repository_mae.png",
        dpi=300,
    )

    plt.close()


def clean_report_text(
    text: str,
) -> str:
    return "\n".join(
        line.rstrip(" \t")
        for line in text.splitlines()
    ).rstrip() + "\n"


def main() -> None:
    if OUTPUT_DIR.exists():
        raise RuntimeError(
            "Final forecasting test outputs "
            "already exist. Refusing to "
            "overwrite the one-time "
            "evaluation directory:\n"
            f"{OUTPUT_DIR}"
        )

    (
        pretest_manifest,
        pretest_metadata,
        frozen_commit,
    ) = verify_pretest_freeze()

    model_path = (
        PROJECT_ROOT
        / pretest_manifest[
            "model_path"
        ]
    )

    artifact = joblib.load(
        model_path
    )

    validate_artifact(
        artifact
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    test_hash = sha256_file(
        TEST_PATH
    )

    raw_test = pd.read_csv(
        TEST_PATH
    )

    test = prepare_test_frame(
        raw_test,
        artifact,
    )

    (
        raw_rolling_baseline,
        residual_correction,
        frozen_prediction,
    ) = frozen_model_predictions(
        artifact,
        test,
    )

    y_true = test[
        EXPECTED_TARGET
    ].to_numpy(dtype=float)

    benchmarks = benchmark_predictions(
        test,
        raw_rolling_baseline,
    )

    benchmarks[
        "frozen_additive_residual_extra_trees"
    ] = frozen_prediction

    comparison_rows = []

    for approach, prediction in (
        benchmarks.items()
    ):
        metrics = evaluate_predictions(
            y_true,
            prediction,
        )

        comparison_rows.append(
            {
                "approach": approach,
                "evaluation_role": (
                    "frozen_primary_model"
                    if approach
                    == (
                        "frozen_additive_"
                        "residual_extra_trees"
                    )
                    else (
                        "descriptive_benchmark"
                    )
                ),
                **metrics,
            }
        )

    comparison = (
        pd.DataFrame(
            comparison_rows
        )
        .sort_values(
            [
                "rmsle",
                "mae",
            ]
        )
        .reset_index(drop=True)
    )

    frozen_metrics = (
        comparison[
            comparison[
                "evaluation_role"
            ]
            == "frozen_primary_model"
        ]
        .iloc[0]
        .to_dict()
    )

    best_descriptive_benchmark = (
        comparison[
            comparison[
                "evaluation_role"
            ]
            == "descriptive_benchmark"
        ]
        .sort_values(
            [
                "rmsle",
                "mae",
            ]
        )
        .iloc[0]
        .to_dict()
    )

    predictions = test[
        IDENTIFIER_COLUMNS
        + [
            EXPECTED_TARGET,
        ]
    ].copy()

    predictions[
        "raw_rolling_baseline"
    ] = raw_rolling_baseline

    predictions[
        "residual_correction"
    ] = residual_correction

    predictions[
        "predicted_future_4week_stars"
    ] = frozen_prediction

    predictions[
        "forecast_error"
    ] = (
        frozen_prediction
        - y_true
    )

    predictions[
        "absolute_error"
    ] = np.abs(
        predictions[
            "forecast_error"
        ]
    )

    predictions[
        "squared_error"
    ] = np.square(
        predictions[
            "forecast_error"
        ]
    )

    predictions[
        "absolute_log_error"
    ] = np.abs(
        np.log1p(
            frozen_prediction
        )
        - np.log1p(
            y_true
        )
    )

    per_repo = (
        per_repository_metrics(
            predictions
        )
    )

    comparison.to_csv(
        METRICS_TABLE_PATH,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    per_repo.to_csv(
        PER_REPOSITORY_PATH,
        index=False,
    )

    target_summary = {
        "rows": int(
            len(test)
        ),
        "repositories": int(
            test[
                "repo_id"
            ].nunique()
        ),
        "samples_per_repository": int(
            test.groupby(
                "repo_id"
            )
            .size()
            .iloc[0]
        ),
        "cutoff_week_minimum": (
            test[
                "cutoff_week"
            ].min().isoformat()
        ),
        "cutoff_week_maximum": (
            test[
                "cutoff_week"
            ].max().isoformat()
        ),
        "target_end_week_minimum": (
            test[
                "target_end_week"
            ].min().isoformat()
        ),
        "target_end_week_maximum": (
            test[
                "target_end_week"
            ].max().isoformat()
        ),
        "target_minimum": float(
            test[
                EXPECTED_TARGET
            ].min()
        ),
        "target_median": float(
            test[
                EXPECTED_TARGET
            ].median()
        ),
        "target_mean": float(
            test[
                EXPECTED_TARGET
            ].mean()
        ),
        "target_maximum": float(
            test[
                EXPECTED_TARGET
            ].max()
        ),
        "prediction_minimum": float(
            frozen_prediction.min()
        ),
        "prediction_median": float(
            np.median(
                frozen_prediction
            )
        ),
        "prediction_mean": float(
            frozen_prediction.mean()
        ),
        "prediction_maximum": float(
            frozen_prediction.max()
        ),
        "residual_correction_mean": float(
            residual_correction.mean()
        ),
        "residual_correction_median": float(
            np.median(
                residual_correction
            )
        ),
    }

    evaluation = {
        "evaluation_status": (
            "forecasting_final_test_complete"
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "one_time_evaluation": True,
        "frozen_pretest_tag": (
            PRETEST_TAG
        ),
        "frozen_pretest_commit": (
            frozen_commit
        ),
        "frozen_model_path": str(
            model_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "frozen_model_sha256": (
            pretest_manifest[
                "model_sha256"
            ]
        ),
        "artifact_version": (
            artifact[
                "artifact_version"
            ]
        ),
        "approach_family": (
            artifact[
                "approach_family"
            ]
        ),
        "mode": artifact["mode"],
        "model_kind": (
            artifact[
                "model_kind"
            ]
        ),
        "baseline_formula": (
            f"{artifact['baseline_multiplier']} "
            f"* "
            f"{artifact['baseline_column']}"
        ),
        "feature_columns": (
            artifact[
                "feature_columns"
            ]
        ),
        "residual_estimator_parameters": (
            artifact[
                "residual_estimator_parameters"
            ]
        ),
        "test_dataset": {
            "path": str(
                TEST_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": test_hash,
            **target_summary,
        },
        "frozen_model_test_metrics": {
            key: float(value)
            for key, value in (
                frozen_metrics.items()
            )
            if key
            not in [
                "approach",
                "evaluation_role",
            ]
        },
        "best_descriptive_test_benchmark": {
            "approach": (
                best_descriptive_benchmark[
                    "approach"
                ]
            ),
            **{
                key: float(value)
                for key, value in (
                    best_descriptive_benchmark.items()
                )
                if key
                not in [
                    "approach",
                    "evaluation_role",
                ]
            },
        },
        "comparison_interpretation_rule": (
            "The frozen model was selected "
            "before opening the test set. "
            "All other test approaches are "
            "reported only as descriptive "
            "benchmarks and cannot replace, "
            "retune, or modify the frozen "
            "model."
        ),
        "pretest_validation_metrics": (
            pretest_metadata[
                "selected_validation_metrics"
            ]
        ),
        "pretest_cv_metrics": (
            pretest_metadata[
                "selected_cv_metrics"
            ]
        ),
        "outputs": {
            "metrics_table": str(
                METRICS_TABLE_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "predictions": str(
                PREDICTIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "per_repository_metrics": str(
                PER_REPOSITORY_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "report": str(
                REPORT_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
        "software": {
            "python": (
                platform.python_version()
            ),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": (
                sklearn.__version__
            ),
            "joblib": (
                joblib.__version__
            ),
        },
        "test_data_status": (
            "loaded_once_for_final_evaluation"
        ),
        "post_test_model_changes": False,
    }

    write_json(
        METRICS_PATH,
        evaluation,
    )

    save_plots(
        predictions=predictions,
        comparison=comparison,
        per_repo=per_repo,
    )

    report_lines = [
        "=" * 112,
        (
            "FOUR-WEEK FORECASTING "
            "FINAL TEST REPORT"
        ),
        "=" * 112,
        "",
        (
            "Frozen pretest tag: "
            f"{PRETEST_TAG}"
        ),
        (
            "Frozen pretest commit: "
            f"{frozen_commit}"
        ),
        (
            "Frozen approach: "
            f"{EXPECTED_APPROACH_FAMILY}"
        ),
        (
            "Test rows: "
            f"{len(test)}"
        ),
        (
            "Test repositories: "
            f"{test['repo_id'].nunique()}"
        ),
        (
            "Test target mean: "
            f"{test[EXPECTED_TARGET].mean():.6f}"
        ),
        "",
        "FROZEN MODEL FINAL-TEST METRICS",
        "-" * 112,
        json.dumps(
            evaluation[
                "frozen_model_test_metrics"
            ],
            indent=2,
        ),
        "",
        "DESCRIPTIVE MODEL COMPARISON",
        "-" * 112,
        comparison.to_string(
            index=False
        ),
        "",
        "PER-REPOSITORY METRICS",
        "-" * 112,
        per_repo.to_string(
            index=False
        ),
        "",
        (
            "The test set was used once "
            "after the model, features, "
            "baseline, hyperparameters, and "
            "training procedure were frozen."
        ),
        (
            "No model selection, retuning, "
            "threshold adjustment, or "
            "post-test model change was "
            "performed."
        ),
    ]

    REPORT_PATH.write_text(
        clean_report_text(
            "\n".join(
                report_lines
            )
        ),
        encoding="utf-8",
    )

    output_hashes = {
        str(
            path.relative_to(
                PROJECT_ROOT
            )
        ): sha256_file(path)
        for path in [
            METRICS_PATH,
            METRICS_TABLE_PATH,
            PREDICTIONS_PATH,
            PER_REPOSITORY_PATH,
            REPORT_PATH,
            OUTPUT_DIR
            / (
                "final_test_"
                "observed_vs_predicted.png"
            ),
            OUTPUT_DIR
            / "final_test_residuals.png",
            OUTPUT_DIR
            / (
                "final_test_"
                "rmsle_comparison.png"
            ),
            OUTPUT_DIR
            / (
                "final_test_"
                "per_repository_mae.png"
            ),
        ]
    }

    final_manifest = {
        "evaluation_name": (
            "forecasting-final-test-v1"
        ),
        "created_at_utc": (
            evaluation[
                "created_at_utc"
            ]
        ),
        "pretest_tag": PRETEST_TAG,
        "pretest_commit": (
            frozen_commit
        ),
        "pretest_manifest_path": str(
            PRETEST_MANIFEST_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "pretest_manifest_sha256": (
            sha256_file(
                PRETEST_MANIFEST_PATH
            )
        ),
        "evaluation_script_path": str(
            SCRIPT_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "evaluation_script_sha256": (
            sha256_file(
                SCRIPT_PATH
            )
        ),
        "test_dataset_path": str(
            TEST_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "test_dataset_sha256": (
            test_hash
        ),
        "frozen_model_path": (
            pretest_manifest[
                "model_path"
            ]
        ),
        "frozen_model_sha256": (
            pretest_manifest[
                "model_sha256"
            ]
        ),
        "output_hashes": output_hashes,
        "post_test_model_changes": (
            False
        ),
    }

    write_json(
        MANIFEST_PATH,
        final_manifest,
    )

    print("=" * 112)
    print(
        "FORECASTING FINAL TEST "
        "EVALUATION"
    )
    print("=" * 112)

    print(
        json.dumps(
            evaluation,
            indent=2,
            default=str,
        )
    )

    print()
    print("=" * 112)
    print(
        "FINAL TEST MODEL COMPARISON"
    )
    print("=" * 112)

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print("=" * 112)
    print(
        "FINAL TEST PER-REPOSITORY "
        "METRICS"
    )
    print("=" * 112)

    print(
        per_repo.to_string(
            index=False
        )
    )

    print()
    print(
        "The frozen forecasting model "
        "was not changed after viewing "
        "test results."
    )


if __name__ == "__main__":
    main()
