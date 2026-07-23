import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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
    precision_score,
    recall_score,
    roc_auc_score,
)

from classification_temporal_cv import (
    build_repository_aware_temporal_cv,
    load_classification_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "classification"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "classification"
    / "corrected"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "classification"
    / "probability_calibration"
)

TRAIN_PATH = DATA_DIR / "classification_train.csv"

VALIDATION_PATH = (
    DATA_DIR / "classification_validation.csv"
)

SCHEMA_PATH = (
    DATA_DIR / "classification_schema.json"
)

BASE_MODEL_PATH = (
    MODEL_DIR / "selected_classifier.joblib"
)

BASE_METADATA_PATH = (
    MODEL_DIR
    / "selected_classifier_metadata.json"
)

DEPLOYMENT_MODEL_PATH = (
    MODEL_DIR
    / "selected_classifier_deployment.joblib"
)

DEPLOYMENT_METADATA_PATH = (
    MODEL_DIR
    / "selected_classifier_deployment_metadata.json"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_numeric_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    split_name: str,
) -> None:
    missing = (
        set(feature_columns)
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{split_name} is missing features: "
            f"{sorted(missing)}"
        )

    numeric = frame[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    if numeric.isna().any().any():
        raise ValueError(
            f"{split_name} contains missing "
            "feature values."
        )

    if not np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{split_name} contains non-finite "
            "feature values."
        )


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": float(
            roc_auc_score(
                y,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y,
                predictions,
            )
        ),
        "cohen_kappa": float(
            cohen_kappa_score(
                y,
                predictions,
            )
        ),
        "log_loss": float(
            log_loss(
                y,
                probabilities,
                labels=[0, 1],
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y,
                probabilities,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 6,
) -> tuple[float, float]:
    y = np.asarray(
        y_true,
        dtype=float,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    quantile_edges = np.unique(
        np.quantile(
            probabilities,
            np.linspace(
                0.0,
                1.0,
                n_bins + 1,
            ),
        )
    )

    if len(quantile_edges) < 2:
        return 0.0, 0.0

    quantile_edges[0] = -np.inf
    quantile_edges[-1] = np.inf

    bin_ids = np.digitize(
        probabilities,
        quantile_edges[1:-1],
        right=True,
    )

    weighted_error = 0.0
    maximum_error = 0.0

    for bin_id in np.unique(bin_ids):
        mask = bin_ids == bin_id

        if not mask.any():
            continue

        observed = float(
            y[mask].mean()
        )

        predicted = float(
            probabilities[mask].mean()
        )

        absolute_error = abs(
            observed - predicted
        )

        weighted_error += (
            mask.mean()
            * absolute_error
        )

        maximum_error = max(
            maximum_error,
            absolute_error,
        )

    return (
        float(weighted_error),
        float(maximum_error),
    )


def choose_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(
                    0.02,
                    0.98,
                    193,
                ),
                np.asarray(
                    probabilities,
                    dtype=float,
                ),
            ]
        )
    )

    rows = []

    for threshold in candidates:
        metrics = calculate_metrics(
            y_true,
            probabilities,
            float(threshold),
        )

        rows.append(metrics)

    results = pd.DataFrame(rows)

    results[
        "distance_from_0_5"
    ] = (
        results["threshold"] - 0.5
    ).abs()

    selected = results.sort_values(
        [
            "mcc",
            "balanced_accuracy",
            "f1",
            "distance_from_0_5",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    ).iloc[0]

    return (
        float(selected["threshold"]),
        results,
    )


def main() -> None:
    settings = load_classification_config()
    schema = load_json(SCHEMA_PATH)
    base_metadata = load_json(
        BASE_METADATA_PATH
    )

    target = settings["target"]

    feature_columns = schema[
        "feature_columns"
    ]

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(
        VALIDATION_PATH
    )

    validate_numeric_features(
        train,
        feature_columns,
        "training",
    )

    validate_numeric_features(
        validation,
        feature_columns,
        "validation",
    )

    (
        prepared_train,
        folds,
        fold_summary,
        _,
    ) = build_repository_aware_temporal_cv(
        frame=train,
        target_column=target,
        n_splits=int(
            settings[
                "cross_validation_folds"
            ]
        ),
        validation_samples_per_repository=int(
            settings[
                "cross_validation_"
                "validation_samples_per_repository"
            ]
        ),
        gap_samples=int(
            settings["purge_weeks"]
        ),
        require_all_repositories=bool(
            settings[
                "cross_validation_"
                "require_all_repositories"
            ]
        ),
    )

    X_train = prepared_train[
        feature_columns
    ]

    y_train = prepared_train[target].astype(
        int
    )

    X_validation = validation[
        feature_columns
    ]

    y_validation = validation[
        target
    ].astype(int)

    base_model = joblib.load(
        BASE_MODEL_PATH
    )

    uncalibrated_probabilities = (
        base_model.predict_proba(
            X_validation
        )[:, 1]
    )

    sigmoid_model = CalibratedClassifierCV(
        estimator=clone(base_model),
        method="sigmoid",
        cv=folds,
        n_jobs=-1,
        ensemble=True,
    )

    sigmoid_model.fit(
        X_train,
        y_train,
    )

    sigmoid_probabilities = (
        sigmoid_model.predict_proba(
            X_validation
        )[:, 1]
    )

    # Isotonic calibration is included only as
    # an exploratory diagnostic. It is not
    # eligible for deployment because each
    # temporal calibration fold is small.
    isotonic_model = CalibratedClassifierCV(
        estimator=clone(base_model),
        method="isotonic",
        cv=folds,
        n_jobs=-1,
        ensemble=True,
    )

    isotonic_model.fit(
        X_train,
        y_train,
    )

    isotonic_probabilities = (
        isotonic_model.predict_proba(
            X_validation
        )[:, 1]
    )

    probability_sets = {
        "uncalibrated": (
            uncalibrated_probabilities
        ),
        "sigmoid_temporal_cv": (
            sigmoid_probabilities
        ),
        "isotonic_exploratory": (
            isotonic_probabilities
        ),
    }

    model_objects = {
        "uncalibrated": base_model,
        "sigmoid_temporal_cv": (
            sigmoid_model
        ),
        "isotonic_exploratory": (
            isotonic_model
        ),
    }

    comparison_rows = []
    prediction_frame = validation[
        [
            "repo_id",
            "repo_full_name",
            "cutoff_week",
            "target_end_week",
            target,
        ]
    ].copy()

    threshold_frames = []

    for method, probabilities in (
        probability_sets.items()
    ):
        threshold, threshold_search = (
            choose_threshold(
                y_validation,
                probabilities,
            )
        )

        metrics_at_half = (
            calculate_metrics(
                y_validation,
                probabilities,
                threshold=0.5,
            )
        )

        metrics_at_selected = (
            calculate_metrics(
                y_validation,
                probabilities,
                threshold=threshold,
            )
        )

        ece, maximum_calibration_error = (
            expected_calibration_error(
                y_validation,
                probabilities,
                n_bins=6,
            )
        )

        row = {
            "method": method,
            "deployment_eligible": (
                method
                in {
                    "uncalibrated",
                    "sigmoid_temporal_cv",
                }
            ),
            "selected_threshold": (
                threshold
            ),
            "roc_auc": (
                metrics_at_selected[
                    "roc_auc"
                ]
            ),
            "pr_auc": (
                metrics_at_selected[
                    "pr_auc"
                ]
            ),
            "log_loss": (
                metrics_at_selected[
                    "log_loss"
                ]
            ),
            "brier_score": (
                metrics_at_selected[
                    "brier_score"
                ]
            ),
            "expected_calibration_error": (
                ece
            ),
            "maximum_calibration_error": (
                maximum_calibration_error
            ),
            "accuracy_at_0_5": (
                metrics_at_half[
                    "accuracy"
                ]
            ),
            "balanced_accuracy_at_0_5": (
                metrics_at_half[
                    "balanced_accuracy"
                ]
            ),
            "f1_at_0_5": (
                metrics_at_half["f1"]
            ),
            "mcc_at_0_5": (
                metrics_at_half["mcc"]
            ),
            "accuracy_at_selected_threshold": (
                metrics_at_selected[
                    "accuracy"
                ]
            ),
            "balanced_accuracy_at_selected_threshold": (
                metrics_at_selected[
                    "balanced_accuracy"
                ]
            ),
            "precision_at_selected_threshold": (
                metrics_at_selected[
                    "precision"
                ]
            ),
            "recall_at_selected_threshold": (
                metrics_at_selected[
                    "recall"
                ]
            ),
            "f1_at_selected_threshold": (
                metrics_at_selected["f1"]
            ),
            "mcc_at_selected_threshold": (
                metrics_at_selected["mcc"]
            ),
            "tn_at_selected_threshold": (
                metrics_at_selected["tn"]
            ),
            "fp_at_selected_threshold": (
                metrics_at_selected["fp"]
            ),
            "fn_at_selected_threshold": (
                metrics_at_selected["fn"]
            ),
            "tp_at_selected_threshold": (
                metrics_at_selected["tp"]
            ),
        }

        comparison_rows.append(row)

        threshold_search.insert(
            0,
            "method",
            method,
        )

        threshold_frames.append(
            threshold_search
        )

        prediction_frame[
            f"{method}_probability"
        ] = probabilities

        prediction_frame[
            f"{method}_prediction_0_5"
        ] = (
            probabilities >= 0.5
        ).astype(int)

        prediction_frame[
            f"{method}_selected_threshold"
        ] = threshold

        prediction_frame[
            f"{method}_prediction_selected"
        ] = (
            probabilities >= threshold
        ).astype(int)

    comparison = pd.DataFrame(
        comparison_rows
    )

    base_roc_auc = float(
        comparison.loc[
            comparison["method"]
            == "uncalibrated",
            "roc_auc",
        ].iloc[0]
    )

    eligible = comparison[
        comparison[
            "deployment_eligible"
        ]
    ].copy()

    eligible["roc_auc_drop"] = (
        base_roc_auc - eligible["roc_auc"]
    )

    acceptable = eligible[
        eligible["roc_auc_drop"] <= 0.02
    ].copy()

    if acceptable.empty:
        acceptable = eligible[
            eligible["method"]
            == "uncalibrated"
        ].copy()

    selected_row = acceptable.sort_values(
        [
            "log_loss",
            "brier_score",
            "expected_calibration_error",
            "roc_auc",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
    ).iloc[0]

    selected_method = str(
        selected_row["method"]
    )

    selected_threshold = float(
        selected_row[
            "selected_threshold"
        ]
    )

    selected_model = model_objects[
        selected_method
    ]

    joblib.dump(
        selected_model,
        DEPLOYMENT_MODEL_PATH,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "calibration_method_comparison.csv",
        index=False,
    )

    pd.concat(
        threshold_frames,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "calibration_threshold_search.csv",
        index=False,
    )

    prediction_frame.to_csv(
        OUTPUT_DIR
        / "calibration_validation_predictions.csv",
        index=False,
    )

    fold_summary.to_csv(
        OUTPUT_DIR
        / "calibration_temporal_folds.csv",
        index=False,
    )

    plt.figure(
        figsize=(9, 7)
    )

    for method, probabilities in (
        probability_sets.items()
    ):
        observed, predicted = (
            calibration_curve(
                y_validation,
                probabilities,
                n_bins=6,
                strategy="quantile",
            )
        )

        plt.plot(
            predicted,
            observed,
            marker="o",
            label=method,
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="perfect calibration",
    )

    plt.xlabel(
        "Mean predicted probability"
    )

    plt.ylabel(
        "Observed positive fraction"
    )

    plt.title(
        "Validation Probability Calibration"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "calibration_method_comparison.png",
        dpi=300,
    )

    plt.close()

    metadata = {
        "design_status": (
            "classification_probability_model_frozen"
        ),
        "target": target,
        "base_selected_model": (
            base_metadata[
                "selected_model"
            ]
        ),
        "selected_probability_method": (
            selected_method
        ),
        "selection_rule": (
            "Among the uncalibrated and "
            "training-only temporal-CV sigmoid "
            "models, select the lowest validation "
            "log loss, then Brier score, then "
            "expected calibration error, provided "
            "validation ROC-AUC does not fall by "
            "more than 0.02 relative to the "
            "uncalibrated model."
        ),
        "selected_threshold": (
            selected_threshold
        ),
        "selected_threshold_rule": (
            "Maximum validation MCC, then "
            "balanced accuracy, then F1."
        ),
        "feature_columns": (
            feature_columns
        ),
        "training_rows": int(
            len(prepared_train)
        ),
        "validation_rows": int(
            len(validation)
        ),
        "repositories": int(
            prepared_train[
                "repo_id"
            ].nunique()
        ),
        "temporal_cv_folds": int(
            len(folds)
        ),
        "validation_metrics": {
            key: (
                bool(value)
                if isinstance(
                    value,
                    (np.bool_, bool),
                )
                else int(value)
                if isinstance(
                    value,
                    (np.integer,),
                )
                else float(value)
                if isinstance(
                    value,
                    (np.floating,),
                )
                else value
            )
            for key, value
            in selected_row.to_dict().items()
        },
        "isotonic_status": (
            "exploratory_only_not_deployment_eligible_"
            "because_temporal_calibration_folds_are_small"
        ),
        "deployment_model_path": str(
            DEPLOYMENT_MODEL_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "test_status": (
            "not_loaded_or_evaluated"
        ),
    }

    DEPLOYMENT_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "=" * 105,
        "CLASSIFICATION PROBABILITY CALIBRATION AUDIT",
        "=" * 105,
        "",
        f"Training rows: {len(prepared_train)}",
        f"Validation rows: {len(validation)}",
        f"Repositories: {prepared_train['repo_id'].nunique()}",
        f"Temporal folds: {len(folds)}",
        "Test data: not loaded",
        "",
        "METHOD COMPARISON",
        "-" * 105,
        comparison.to_string(index=False),
        "",
        f"Selected probability method: {selected_method}",
        f"Selected threshold: {selected_threshold:.6f}",
        "",
        (
            "Isotonic calibration was evaluated only "
            "as a diagnostic and was not eligible for "
            "deployment because the temporal "
            "calibration folds are small."
        ),
        "",
        "Test data was not loaded or evaluated.",
    ]

    report = "\n".join(
        report_lines
    )

    (
        OUTPUT_DIR
        / "probability_calibration_report.txt"
    ).write_text(
        report,
        encoding="utf-8",
    )

    print(report)

    print()
    print(
        "Classification probability calibration "
        "audit completed successfully."
    )


if __name__ == "__main__":
    main()
