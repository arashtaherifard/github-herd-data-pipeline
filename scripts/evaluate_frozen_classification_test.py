import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# Run this external script from the project root.
# Keeping it outside the repository preserves the exact
# pre-test freeze commit while the test is opened once.
PROJECT_ROOT = Path.cwd().resolve()

FREEZE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "classification"
    / "pre_test_freeze_manifest.json"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "classification"
    / "classification_test.csv"
)

SCHEMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "classification"
    / "classification_schema.json"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classification"
    / "corrected"
    / "selected_classifier_deployment.joblib"
)

MODEL_METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "classification"
    / "corrected"
    / "selected_classifier_deployment_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "classification"
    / "final_test_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FROZEN_TAG = "classification-pretest-freeze-v1"


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def verify_frozen_artifacts(
    freeze_manifest: dict,
) -> dict:
    results = {}

    for label, information in (
        freeze_manifest["frozen_files"].items()
    ):
        path = (
            PROJECT_ROOT
            / information["relative_path"]
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Frozen file is missing: {path}"
            )

        actual_hash = sha256_file(path)
        expected_hash = information["sha256"]

        match = actual_hash == expected_hash

        results[label] = {
            "relative_path": information[
                "relative_path"
            ],
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "match": match,
        }

        if not match:
            raise RuntimeError(
                "Frozen artifact hash mismatch for "
                f"{label}: {path}"
            )

    return results


def git_output(arguments: list[str]) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def verify_git_freeze() -> dict:
    current_head = git_output(
        ["rev-parse", "HEAD"]
    )

    tag_target = git_output(
        [
            "rev-list",
            "-n",
            "1",
            FROZEN_TAG,
        ]
    )

    working_tree_status = git_output(
        ["status", "--porcelain"]
    )

    if current_head != tag_target:
        raise RuntimeError(
            "Current HEAD does not match the "
            "pre-test freeze tag."
        )

    if working_tree_status:
        raise RuntimeError(
            "Working tree is not clean before "
            "one-time test evaluation:\n"
            f"{working_tree_status}"
        )

    return {
        "current_head": current_head,
        "freeze_tag": FROZEN_TAG,
        "freeze_tag_target": tag_target,
        "head_matches_tag": True,
        "working_tree_clean": True,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 6,
) -> tuple[float, float]:
    y_true = np.asarray(
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

    ece = 0.0
    maximum_error = 0.0

    for bin_id in np.unique(bin_ids):
        mask = bin_ids == bin_id

        if not mask.any():
            continue

        observed = float(
            y_true[mask].mean()
        )

        predicted = float(
            probabilities[mask].mean()
        )

        absolute_error = abs(
            observed - predicted
        )

        ece += (
            mask.mean()
            * absolute_error
        )

        maximum_error = max(
            maximum_error,
            absolute_error,
        )

    return float(ece), float(maximum_error)


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    y_true = np.asarray(
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
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    ece, maximum_calibration_error = (
        expected_calibration_error(
            y_true,
            probabilities,
            n_bins=6,
        )
    )

    return {
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
        "expected_calibration_error": ece,
        "maximum_calibration_error": (
            maximum_calibration_error
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def validate_test_data(
    test: pd.DataFrame,
    schema: dict,
    target: str,
) -> list[str]:
    feature_columns = schema[
        "feature_columns"
    ]

    required_columns = set(
        feature_columns
    ) | {
        "repo_id",
        "repo_full_name",
        "cutoff_week",
        "target_end_week",
        target,
    }

    missing = (
        required_columns
        - set(test.columns)
    )

    if missing:
        raise ValueError(
            "Test dataset is missing columns: "
            f"{sorted(missing)}"
        )

    numeric_features = test[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    if numeric_features.isna().any().any():
        raise ValueError(
            "Test dataset contains missing "
            "feature values."
        )

    if not np.isfinite(
        numeric_features.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Test dataset contains non-finite "
            "feature values."
        )

    target_values = set(
        pd.to_numeric(
            test[target],
            errors="raise",
        )
        .astype(int)
        .unique()
        .tolist()
    )

    if target_values != {0, 1}:
        raise ValueError(
            "Test target must contain both "
            f"classes, found: {target_values}"
        )

    return feature_columns


def save_plots(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> None:
    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"Frozen model (AUC={roc_auc:.3f})",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Chance",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        "Final Classification Test ROC Curve"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "test_roc_curve.png",
        dpi=300,
    )
    plt.close()

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    prevalence = float(
        np.mean(y_true)
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        label=f"Frozen model (AP={pr_auc:.3f})",
    )
    plt.axhline(
        prevalence,
        linestyle="--",
        label=(
            "Positive prevalence "
            f"({prevalence:.3f})"
        ),
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        "Final Classification Test "
        "Precision–Recall Curve"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR
        / "test_precision_recall_curve.png",
        dpi=300,
    )
    plt.close()

    observed, predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=6,
        strategy="quantile",
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        predicted,
        observed,
        marker="o",
        label="Frozen calibrated model",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )
    plt.xlabel(
        "Mean Predicted Probability"
    )
    plt.ylabel(
        "Observed Positive Fraction"
    )
    plt.title(
        "Final Classification Test "
        "Calibration"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR
        / "test_calibration_curve.png",
        dpi=300,
    )
    plt.close()

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ),
        display_labels=[
            "No surge",
            "Growth surge",
        ],
    )

    display.plot(
        values_format="d",
    )
    plt.title(
        "Final Test Confusion Matrix "
        "at Frozen Threshold"
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR
        / "test_confusion_matrix.png",
        dpi=300,
    )
    plt.close()


def main() -> None:
    if any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            "Final test-evaluation directory is "
            "not empty. Refusing to rerun the "
            "one-time test evaluation."
        )

    freeze_manifest = load_json(
        FREEZE_MANIFEST_PATH
    )

    git_verification = verify_git_freeze()

    hash_verification = (
        verify_frozen_artifacts(
            freeze_manifest
        )
    )

    schema = load_json(SCHEMA_PATH)
    model_metadata = load_json(
        MODEL_METADATA_PATH
    )

    target = model_metadata["target"]

    frozen_threshold = float(
        model_metadata[
            "selected_threshold"
        ]
    )

    # The test CSV is deliberately parsed only
    # after the Git and frozen-file checks pass.
    test = pd.read_csv(TEST_PATH)

    feature_columns = validate_test_data(
        test,
        schema,
        target,
    )

    model = joblib.load(MODEL_PATH)

    X_test = test[feature_columns]

    y_test = (
        pd.to_numeric(
            test[target],
            errors="raise",
        )
        .astype(int)
        .to_numpy()
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    frozen_predictions = (
        probabilities >= frozen_threshold
    ).astype(int)

    metrics_at_frozen_threshold = (
        calculate_metrics(
            y_test,
            probabilities,
            frozen_threshold,
        )
    )

    metrics_at_default_threshold = (
        calculate_metrics(
            y_test,
            probabilities,
            0.5,
        )
    )

    predictions = test[
        [
            "repo_id",
            "repo_full_name",
            "cutoff_week",
            "target_end_week",
            target,
        ]
    ].copy()

    predictions[
        "predicted_probability"
    ] = probabilities

    predictions[
        "frozen_threshold"
    ] = frozen_threshold

    predictions[
        "predicted_class_frozen_threshold"
    ] = frozen_predictions

    predictions[
        "predicted_class_default_0_5"
    ] = (
        probabilities >= 0.5
    ).astype(int)

    predictions[
        "correct_frozen_threshold"
    ] = (
        predictions[
            "predicted_class_frozen_threshold"
        ]
        == predictions[target]
    ).astype(int)

    predictions.to_csv(
        OUTPUT_DIR
        / "final_test_predictions.csv",
        index=False,
    )

    metrics_table = pd.DataFrame(
        [
            {
                "operating_point": (
                    "frozen_validation_selected"
                ),
                **metrics_at_frozen_threshold,
            },
            {
                "operating_point": (
                    "default_0_5"
                ),
                **metrics_at_default_threshold,
            },
        ]
    )

    metrics_table.to_csv(
        OUTPUT_DIR
        / "final_test_metrics.csv",
        index=False,
    )

    save_plots(
        y_test,
        probabilities,
        frozen_predictions,
    )

    validation_metrics = model_metadata[
        "validation_metrics"
    ]

    comparison = {
        "roc_auc_change_test_minus_validation": (
            metrics_at_frozen_threshold[
                "roc_auc"
            ]
            - float(
                validation_metrics["roc_auc"]
            )
        ),
        "pr_auc_change_test_minus_validation": (
            metrics_at_frozen_threshold[
                "pr_auc"
            ]
            - float(
                validation_metrics["pr_auc"]
            )
        ),
        "log_loss_change_test_minus_validation": (
            metrics_at_frozen_threshold[
                "log_loss"
            ]
            - float(
                validation_metrics["log_loss"]
            )
        ),
        "brier_change_test_minus_validation": (
            metrics_at_frozen_threshold[
                "brier_score"
            ]
            - float(
                validation_metrics["brier_score"]
            )
        ),
        "balanced_accuracy_change_test_minus_validation": (
            metrics_at_frozen_threshold[
                "balanced_accuracy"
            ]
            - float(
                validation_metrics[
                    "balanced_accuracy_at_selected_threshold"
                ]
            )
        ),
        "f1_change_test_minus_validation": (
            metrics_at_frozen_threshold[
                "f1"
            ]
            - float(
                validation_metrics[
                    "f1_at_selected_threshold"
                ]
            )
        ),
        "mcc_change_test_minus_validation": (
            metrics_at_frozen_threshold[
                "mcc"
            ]
            - float(
                validation_metrics[
                    "mcc_at_selected_threshold"
                ]
            )
        ),
    }

    evaluation_record = {
        "evaluation_status": (
            "one_time_final_classification_"
            "test_evaluation_complete"
        ),
        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "freeze_commit": (
            git_verification[
                "current_head"
            ]
        ),
        "freeze_tag": FROZEN_TAG,
        "target": target,
        "model": model_metadata[
            "base_selected_model"
        ],
        "probability_method": (
            model_metadata[
                "selected_probability_method"
            ]
        ),
        "test_rows": int(len(test)),
        "test_repositories": int(
            test["repo_id"].nunique()
        ),
        "test_class_counts": {
            str(key): int(value)
            for key, value
            in (
                test[target]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "test_positive_rate": float(
            test[target].mean()
        ),
        "frozen_threshold": (
            frozen_threshold
        ),
        "metrics_at_frozen_threshold": (
            metrics_at_frozen_threshold
        ),
        "metrics_at_default_threshold_0_5": (
            metrics_at_default_threshold
        ),
        "validation_metrics_from_frozen_metadata": (
            validation_metrics
        ),
        "test_minus_validation_changes": (
            comparison
        ),
        "git_verification": (
            git_verification
        ),
        "frozen_hash_verification": (
            hash_verification
        ),
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": (
                sklearn.__version__
            ),
        },
        "post_test_rule": (
            "No model, feature, calibration, or "
            "threshold changes are permitted in "
            "response to these test results."
        ),
    }

    (
        OUTPUT_DIR
        / "final_test_evaluation.json"
    ).write_text(
        json.dumps(
            evaluation_record,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "=" * 105,
        "ONE-TIME FINAL CLASSIFICATION TEST EVALUATION",
        "=" * 105,
        "",
        (
            "Freeze commit: "
            f"{git_verification['current_head']}"
        ),
        f"Freeze tag: {FROZEN_TAG}",
        f"Test rows: {len(test)}",
        (
            "Test repositories: "
            f"{test['repo_id'].nunique()}"
        ),
        (
            "Test class counts: "
            f"{evaluation_record['test_class_counts']}"
        ),
        (
            "Test positive rate: "
            f"{evaluation_record['test_positive_rate']:.6f}"
        ),
        (
            "Frozen threshold: "
            f"{frozen_threshold:.6f}"
        ),
        "",
        "METRICS AT FROZEN THRESHOLD",
        "-" * 105,
        json.dumps(
            metrics_at_frozen_threshold,
            indent=2,
        ),
        "",
        "METRICS AT DEFAULT THRESHOLD 0.5",
        "-" * 105,
        json.dumps(
            metrics_at_default_threshold,
            indent=2,
        ),
        "",
        "TEST MINUS VALIDATION CHANGES",
        "-" * 105,
        json.dumps(
            comparison,
            indent=2,
        ),
        "",
        (
            "No model, feature, calibration, "
            "or threshold changes are permitted "
            "after this evaluation."
        ),
    ]

    report = "\n".join(
        report_lines
    )

    (
        OUTPUT_DIR
        / "final_test_report.txt"
    ).write_text(
        report,
        encoding="utf-8",
    )

    print(report)

    print()
    print(
        "One-time final classification test "
        "evaluation completed successfully."
    )


if __name__ == "__main__":
    main()
