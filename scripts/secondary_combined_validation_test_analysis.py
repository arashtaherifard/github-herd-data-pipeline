from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "phase3"
    / "secondary_combined_evaluation"
)


# ============================================================
# Input files
# ============================================================

CLASSIFICATION_METADATA = (
    ROOT
    / "models"
    / "classification"
    / "corrected"
    / "selected_classifier_deployment_metadata.json"
)

CLASSIFICATION_VALIDATION = (
    ROOT
    / "outputs"
    / "phase3"
    / "classification"
    / "probability_calibration"
    / "calibration_validation_predictions.csv"
)

CLASSIFICATION_TEST = (
    ROOT
    / "outputs"
    / "phase3"
    / "classification"
    / "final_test_evaluation"
    / "final_test_predictions.csv"
)

FORECAST_VALIDATION = (
    ROOT
    / "outputs"
    / "phase3"
    / "forecasting"
    / "pretest_freeze"
    / "train_only_validation_reproduction.csv"
)

FORECAST_TEST = (
    ROOT
    / "outputs"
    / "phase3"
    / "forecasting"
    / "final_test_evaluation"
    / "final_test_predictions.csv"
)


IDENTIFIERS = [
    "repo_id",
    "repo_full_name",
    "cutoff_week",
    "target_end_week",
]

CLASSIFICATION_TARGET = "future_growth_surge"

CLASSIFICATION_VALIDATION_PROBABILITY = (
    "sigmoid_temporal_cv_probability"
)

CLASSIFICATION_TEST_PROBABILITY = (
    "predicted_probability"
)

FORECAST_TARGET = "future_4week_stars"

FORECAST_VALIDATION_PREDICTION = (
    "reproduced_prediction"
)

FORECAST_TEST_PREDICTION = (
    "predicted_future_4week_stars"
)


# ============================================================
# General validation functions
# ============================================================

def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = sorted(
        set(columns)
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{label} is missing columns: "
            f"{missing}"
        )


def to_numeric_array(
    series: pd.Series,
    label: str,
) -> np.ndarray:
    numeric = pd.to_numeric(
        series,
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError(
            f"{label} contains non-finite values."
        )

    return numeric


def validate_unique_keys(
    frame: pd.DataFrame,
    label: str,
) -> None:
    duplicate_count = int(
        frame.duplicated(
            [
                "repo_id",
                "cutoff_week",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{label} contains "
            f"{duplicate_count} duplicate "
            "repository/cutoff keys."
        )


# ============================================================
# Classification metrics
# ============================================================

def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 6,
) -> tuple[float, float]:
    edges = np.unique(
        np.quantile(
            probabilities,
            np.linspace(
                0.0,
                1.0,
                n_bins + 1,
            ),
        )
    )

    if len(edges) < 2:
        return 0.0, 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    bin_ids = np.digitize(
        probabilities,
        edges[1:-1],
        right=True,
    )

    ece = 0.0
    maximum_error = 0.0

    for bin_id in np.unique(bin_ids):
        mask = bin_ids == bin_id

        observed = float(
            y_true[mask].mean()
        )

        predicted = float(
            probabilities[mask].mean()
        )

        error = abs(
            observed - predicted
        )

        ece += (
            float(mask.mean())
            * error
        )

        maximum_error = max(
            maximum_error,
            error,
        )

    return (
        float(ece),
        float(maximum_error),
    )


def classification_metrics(
    frame: pd.DataFrame,
    threshold: float,
    scope: str,
) -> dict:
    y_true = to_numeric_array(
        frame["y_true"],
        f"{scope} classification target",
    ).astype(int)

    probabilities = to_numeric_array(
        frame["predicted_probability"],
        f"{scope} classification probabilities",
    )

    if set(
        np.unique(y_true).tolist()
    ) != {0, 1}:
        raise ValueError(
            f"{scope} classification data "
            "must contain both classes."
        )

    if np.any(
        (probabilities < 0.0)
        | (probabilities > 1.0)
    ):
        raise ValueError(
            f"{scope} probabilities must "
            "be between 0 and 1."
        )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    (
        ece,
        maximum_calibration_error,
    ) = expected_calibration_error(
        y_true,
        probabilities,
        n_bins=6,
    )

    return {
        "scope": scope,
        "rows": int(len(frame)),
        "repositories": int(
            frame["repo_id"].nunique()
        ),
        "positive_cases": int(
            y_true.sum()
        ),
        "positive_rate": float(
            y_true.mean()
        ),
        "threshold": float(threshold),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                predictions,
            )
        ),
        "cohen_kappa": float(
            cohen_kappa_score(
                y_true,
                predictions,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "expected_calibration_error": (
            ece
        ),
        "maximum_calibration_error": (
            maximum_calibration_error
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# Forecasting metrics
# ============================================================

def smape(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> float:
    denominator = (
        np.abs(y_true)
        + np.abs(predictions)
    )

    numerator = (
        2.0
        * np.abs(
            predictions - y_true
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
        values.mean()
    )


def forecasting_metrics(
    frame: pd.DataFrame,
    scope: str,
) -> dict:
    y_true = to_numeric_array(
        frame["y_true"],
        f"{scope} forecasting target",
    )

    predictions = to_numeric_array(
        frame[
            "predicted_future_4week_stars"
        ],
        f"{scope} forecasting predictions",
    )

    if (
        np.any(y_true < 0.0)
        or np.any(predictions < 0.0)
    ):
        raise ValueError(
            f"{scope} targets and predictions "
            "must be non-negative."
        )

    errors = (
        predictions - y_true
    )

    absolute_errors = np.abs(
        errors
    )

    return {
        "scope": scope,
        "rows": int(len(frame)),
        "repositories": int(
            frame["repo_id"].nunique()
        ),
        "target_mean": float(
            y_true.mean()
        ),
        "prediction_mean": float(
            predictions.mean()
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                predictions,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    predictions,
                )
            )
        ),
        "rmsle": float(
            np.sqrt(
                mean_squared_log_error(
                    y_true,
                    predictions,
                )
            )
        ),
        "smape": smape(
            y_true,
            predictions,
        ),
        "forecast_bias": float(
            errors.mean()
        ),
        "median_absolute_error": float(
            np.median(
                absolute_errors
            )
        ),
        "maximum_absolute_error": float(
            absolute_errors.max()
        ),
    }


# ============================================================
# Plotting
# ============================================================

def save_grouped_bar_chart(
    comparison: pd.DataFrame,
    metrics: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    labels = [
        "Validation",
        "Test",
        "Validation + Test",
    ]

    scopes = [
        "validation",
        "test",
        "validation_plus_test_secondary",
    ]

    x = np.arange(
        len(metrics),
        dtype=float,
    )

    width = 0.24

    plt.figure(
        figsize=(12, 7)
    )

    for index, (
        scope,
        label,
    ) in enumerate(
        zip(
            scopes,
            labels,
            strict=True,
        )
    ):
        row = comparison.loc[
            comparison["scope"] == scope
        ].iloc[0]

        offset = (
            index - 1
        ) * width

        plt.bar(
            x + offset,
            [
                float(row[metric])
                for metric in metrics
            ],
            width,
            label=label,
        )

    plt.xticks(
        x,
        [
            metric
            .replace("_", " ")
            .upper()
            for metric in metrics
        ],
        rotation=20,
    )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


# ============================================================
# Load and standardize classification predictions
# ============================================================

def prepare_classification(
    threshold: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    validation_raw = pd.read_csv(
        CLASSIFICATION_VALIDATION
    )

    test_raw = pd.read_csv(
        CLASSIFICATION_TEST
    )

    require_columns(
        validation_raw,
        IDENTIFIERS
        + [
            CLASSIFICATION_TARGET,
            CLASSIFICATION_VALIDATION_PROBABILITY,
            (
                "sigmoid_temporal_cv_"
                "selected_threshold"
            ),
        ],
        (
            "classification validation "
            "predictions"
        ),
    )

    require_columns(
        test_raw,
        IDENTIFIERS
        + [
            CLASSIFICATION_TARGET,
            CLASSIFICATION_TEST_PROBABILITY,
        ],
        (
            "classification test "
            "predictions"
        ),
    )

    saved_validation_thresholds = (
        pd.to_numeric(
            validation_raw[
                (
                    "sigmoid_temporal_cv_"
                    "selected_threshold"
                )
            ],
            errors="raise",
        )
        .to_numpy(dtype=float)
    )

    if not np.allclose(
        saved_validation_thresholds,
        threshold,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Saved validation thresholds "
            "do not match frozen metadata."
        )

    if (
        "frozen_threshold"
        in test_raw.columns
    ):
        saved_test_thresholds = (
            pd.to_numeric(
                test_raw[
                    "frozen_threshold"
                ],
                errors="raise",
            )
            .to_numpy(dtype=float)
        )

        if not np.allclose(
            saved_test_thresholds,
            threshold,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Saved test thresholds do "
                "not match frozen metadata."
            )

    validation = validation_raw[
        IDENTIFIERS
        + [
            CLASSIFICATION_TARGET,
            CLASSIFICATION_VALIDATION_PROBABILITY,
        ]
    ].rename(
        columns={
            CLASSIFICATION_TARGET: (
                "y_true"
            ),
            CLASSIFICATION_VALIDATION_PROBABILITY: (
                "predicted_probability"
            ),
        }
    )

    validation.insert(
        0,
        "evaluation_split",
        "validation",
    )

    test = test_raw[
        IDENTIFIERS
        + [
            CLASSIFICATION_TARGET,
            CLASSIFICATION_TEST_PROBABILITY,
        ]
    ].rename(
        columns={
            CLASSIFICATION_TARGET: (
                "y_true"
            ),
            CLASSIFICATION_TEST_PROBABILITY: (
                "predicted_probability"
            ),
        }
    )

    test.insert(
        0,
        "evaluation_split",
        "test",
    )

    validate_unique_keys(
        validation,
        "classification validation",
    )

    validate_unique_keys(
        test,
        "classification test",
    )

    pooled = pd.concat(
        [
            validation,
            test,
        ],
        ignore_index=True,
    )

    validate_unique_keys(
        pooled,
        "pooled classification",
    )

    pooled[
        "frozen_threshold"
    ] = float(threshold)

    pooled[
        "predicted_class"
    ] = (
        pd.to_numeric(
            pooled[
                "predicted_probability"
            ],
            errors="raise",
        )
        >= threshold
    ).astype(int)

    return (
        validation,
        test,
        pooled,
    )


# ============================================================
# Load and standardize forecasting predictions
# ============================================================

def prepare_forecasting(
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    validation_raw = pd.read_csv(
        FORECAST_VALIDATION
    )

    test_raw = pd.read_csv(
        FORECAST_TEST
    )

    require_columns(
        validation_raw,
        IDENTIFIERS
        + [
            FORECAST_TARGET,
            FORECAST_VALIDATION_PREDICTION,
        ],
        (
            "forecasting validation "
            "reproduction"
        ),
    )

    require_columns(
        test_raw,
        IDENTIFIERS
        + [
            FORECAST_TARGET,
            FORECAST_TEST_PREDICTION,
        ],
        "forecasting test predictions",
    )

    if (
        "absolute_difference"
        in validation_raw.columns
    ):
        maximum_reproduction_difference = (
            float(
                pd.to_numeric(
                    validation_raw[
                        "absolute_difference"
                    ],
                    errors="raise",
                ).max()
            )
        )

        if (
            maximum_reproduction_difference
            > 1e-10
        ):
            raise ValueError(
                "Forecast validation "
                "reproduction differs from "
                "the saved audit predictions: "
                f"{maximum_reproduction_difference}"
            )

    validation = validation_raw[
        IDENTIFIERS
        + [
            FORECAST_TARGET,
            FORECAST_VALIDATION_PREDICTION,
        ]
    ].rename(
        columns={
            FORECAST_TARGET: (
                "y_true"
            ),
            FORECAST_VALIDATION_PREDICTION: (
                "predicted_future_4week_stars"
            ),
        }
    )

    validation.insert(
        0,
        "evaluation_split",
        "validation",
    )

    validation[
        "prediction_source"
    ] = (
        "train_only_validation_reproduction"
    )

    test = test_raw[
        IDENTIFIERS
        + [
            FORECAST_TARGET,
            FORECAST_TEST_PREDICTION,
        ]
    ].rename(
        columns={
            FORECAST_TARGET: (
                "y_true"
            ),
            FORECAST_TEST_PREDICTION: (
                "predicted_future_4week_stars"
            ),
        }
    )

    test.insert(
        0,
        "evaluation_split",
        "test",
    )

    test[
        "prediction_source"
    ] = (
        "frozen_train_plus_validation_model"
    )

    validate_unique_keys(
        validation,
        "forecasting validation",
    )

    validate_unique_keys(
        test,
        "forecasting test",
    )

    pooled = pd.concat(
        [
            validation,
            test,
        ],
        ignore_index=True,
    )

    validate_unique_keys(
        pooled,
        "pooled forecasting",
    )

    return (
        validation,
        test,
        pooled,
    )


# ============================================================
# Main analysis
# ============================================================

def main() -> None:
    required_files = [
        CLASSIFICATION_METADATA,
        CLASSIFICATION_VALIDATION,
        CLASSIFICATION_TEST,
        FORECAST_VALIDATION,
        FORECAST_TEST,
    ]

    for path in required_files:
        require_file(path)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = json.loads(
        CLASSIFICATION_METADATA.read_text(
            encoding="utf-8"
        )
    )

    threshold = float(
        metadata[
            "selected_threshold"
        ]
    )

    (
        classification_validation,
        classification_test,
        classification_pooled,
    ) = prepare_classification(
        threshold
    )

    classification_comparison = (
        pd.DataFrame(
            [
                classification_metrics(
                    classification_validation,
                    threshold,
                    "validation",
                ),
                classification_metrics(
                    classification_test,
                    threshold,
                    "test",
                ),
                classification_metrics(
                    classification_pooled,
                    threshold,
                    (
                        "validation_plus_"
                        "test_secondary"
                    ),
                ),
            ]
        )
    )

    classification_comparison.to_csv(
        OUTPUT_DIR
        / "classification_split_comparison.csv",
        index=False,
    )

    classification_pooled.to_csv(
        OUTPUT_DIR
        / "classification_pooled_predictions.csv",
        index=False,
    )

    save_grouped_bar_chart(
        classification_comparison,
        [
            "roc_auc",
            "pr_auc",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
        ],
        (
            "Classification: Validation vs "
            "Test vs Secondary Pooled Evaluation"
        ),
        "Metric value",
        OUTPUT_DIR
        / "classification_metrics_comparison.png",
    )

    (
        forecast_validation,
        forecast_test,
        forecast_pooled,
    ) = prepare_forecasting()

    forecasting_comparison = (
        pd.DataFrame(
            [
                forecasting_metrics(
                    forecast_validation,
                    "validation",
                ),
                forecasting_metrics(
                    forecast_test,
                    "test",
                ),
                forecasting_metrics(
                    forecast_pooled,
                    (
                        "validation_plus_"
                        "test_secondary"
                    ),
                ),
            ]
        )
    )

    forecasting_comparison.to_csv(
        OUTPUT_DIR
        / "forecasting_split_comparison.csv",
        index=False,
    )

    forecast_pooled.to_csv(
        OUTPUT_DIR
        / "forecasting_pooled_predictions.csv",
        index=False,
    )

    save_grouped_bar_chart(
        forecasting_comparison,
        [
            "mae",
            "rmse",
            "median_absolute_error",
        ],
        (
            "Forecasting: Absolute Error "
            "Comparison"
        ),
        "Stars",
        OUTPUT_DIR
        / (
            "forecasting_absolute_"
            "error_comparison.png"
        ),
    )

    save_grouped_bar_chart(
        forecasting_comparison,
        [
            "rmsle",
            "smape",
        ],
        (
            "Forecasting: Relative and "
            "Log-Scale Error Comparison"
        ),
        "Metric value (lower is better)",
        OUTPUT_DIR
        / (
            "forecasting_relative_"
            "error_comparison.png"
        ),
    )

    summary = {
        "analysis_status": (
            "secondary_combined_validation_"
            "test_evaluation_complete"
        ),
        "scientific_status": (
            "secondary_descriptive_not_a_"
            "replacement_for_final_test"
        ),
        "classification": {
            "frozen_threshold": (
                threshold
            ),
            "comparison": (
                classification_comparison
                .to_dict(
                    orient="records"
                )
            ),
            "note": (
                "The same frozen threshold "
                "was used for validation, "
                "test, and pooled metrics. "
                "No threshold was selected "
                "from the pooled labels. "
                "Validation participated in "
                "model and threshold selection, "
                "so the pooled result is not "
                "an independent test."
            ),
        },
        "forecasting": {
            "comparison": (
                forecasting_comparison
                .to_dict(
                    orient="records"
                )
            ),
            "note": (
                "Validation uses the saved "
                "train-only out-of-sample "
                "reproduction, while test uses "
                "the frozen model fitted on "
                "train plus validation. The "
                "pooled result is therefore a "
                "sequential out-of-sample "
                "late-period summary."
            ),
        },
        "primary_reporting_rule": (
            "Keep test-only metrics as the "
            "primary final results. Report "
            "pooled metrics only as a "
            "secondary descriptive stability "
            "analysis."
        ),
    }

    (
        OUTPUT_DIR
        / (
            "secondary_combined_"
            "evaluation_summary.json"
        )
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    report_lines = [
        (
            "SECONDARY VALIDATION + "
            "TEST ANALYSIS"
        ),
        "=" * 100,
        "",
        (
            "This analysis reads previously "
            "saved out-of-sample predictions only."
        ),
        (
            "It does not retrain models, rerun "
            "final-test scripts, or optimize "
            "a new threshold."
        ),
        "",
        "CLASSIFICATION",
        "-" * 100,
        classification_comparison.to_string(
            index=False
        ),
        "",
        "FORECASTING",
        "-" * 100,
        forecasting_comparison.to_string(
            index=False
        ),
        "",
        "INTERPRETATION RULE",
        "-" * 100,
        (
            "Test-only remains the primary "
            "unbiased final estimate."
        ),
        (
            "Validation plus test is a "
            "secondary descriptive stability "
            "analysis."
        ),
        (
            "Classification pooling includes "
            "data used for model and threshold "
            "selection."
        ),
        (
            "Forecast pooling combines "
            "sequential out-of-sample "
            "predictions from different "
            "training windows."
        ),
        "",
    ]

    (
        OUTPUT_DIR
        / (
            "secondary_combined_"
            "evaluation_report.txt"
        )
    ).write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    print(
        "=" * 110
    )

    print(
        "CLASSIFICATION: VALIDATION VS "
        "TEST VS SECONDARY POOLED"
    )

    print(
        "=" * 110
    )

    print(
        classification_comparison[
            [
                "scope",
                "rows",
                "positive_cases",
                "roc_auc",
                "pr_auc",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "mcc",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "FORECASTING: VALIDATION VS "
        "TEST VS SECONDARY POOLED"
    )

    print(
        "=" * 110
    )

    print(
        forecasting_comparison[
            [
                "scope",
                "rows",
                "mae",
                "rmse",
                "rmsle",
                "smape",
                "median_absolute_error",
                "maximum_absolute_error",
                "forecast_bias",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nOutputs written to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nPrimary reporting rule: "
        "test-only remains the final "
        "unbiased result."
    )

    print(
        "The combined result is "
        "secondary descriptive evidence."
    )


if __name__ == "__main__":
    main()
