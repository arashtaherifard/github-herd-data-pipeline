import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORECAST_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "forecasting"
)

TRAIN_PATH = (
    FORECAST_DIR / "forecast_primary_train.csv"
)

VALIDATION_PATH = (
    FORECAST_DIR
    / "forecast_primary_validation.csv"
)

TEST_PATH = (
    FORECAST_DIR / "forecast_primary_test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "classification"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "future_4week_stars"
RECENT_SUM = "rolling_4week_sum_stars"


def best_threshold_accuracy(
    feature_values: pd.Series,
    target_values: pd.Series,
) -> float:
    x = np.asarray(feature_values, dtype=float)
    y = np.asarray(target_values, dtype=int)

    unique_values = np.unique(x)

    if len(unique_values) <= 1:
        return float(max(y.mean(), 1 - y.mean()))

    if len(unique_values) > 500:
        thresholds = np.unique(
            np.quantile(
                x,
                np.linspace(0.01, 0.99, 199),
            )
        )
    else:
        thresholds = (
            unique_values[:-1]
            + unique_values[1:]
        ) / 2

    best_accuracy = 0.0

    for threshold in thresholds:
        higher_prediction = (
            x >= threshold
        ).astype(int)

        lower_prediction = (
            x < threshold
        ).astype(int)

        best_accuracy = max(
            best_accuracy,
            float(
                np.mean(
                    higher_prediction == y
                )
            ),
            float(
                np.mean(
                    lower_prediction == y
                )
            ),
        )

    return best_accuracy


def get_numeric_historical_features(
    frame: pd.DataFrame,
) -> list[str]:
    excluded_columns = {
        "repo_id",
        "repo_full_name",
        "cutoff_week",
        "target_start_week",
        "target_end_week",
        TARGET,
        "split",
        "split_order",
        "repository_sample_index",
    }

    return [
        column
        for column in frame.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(
            frame[column]
        )
        and frame[column].nunique(
            dropna=False
        ) > 1
    ]


def create_candidate_labels(
    frame: pd.DataFrame,
    thresholds: dict,
) -> dict[str, pd.Series]:
    future = frame[TARGET].astype(float)

    recent = (
        frame[RECENT_SUM]
        .astype(float)
        .clip(lower=1.0)
    )

    future_to_recent_ratio = future / recent

    return {
        "absolute_q75": (
            future
            >= thresholds["absolute_q75"]
        ).astype(int),

        "absolute_q80": (
            future
            >= thresholds["absolute_q80"]
        ).astype(int),

        "absolute_q85": (
            future
            >= thresholds["absolute_q85"]
        ).astype(int),

        "absolute_q90": (
            future
            >= thresholds["absolute_q90"]
        ).astype(int),

        "relative_growth_1_25": (
            future_to_recent_ratio >= 1.25
        ).astype(int),

        "relative_growth_1_50": (
            future_to_recent_ratio >= 1.50
        ).astype(int),

        "hybrid_q75_relative_1_10": (
            (
                future
                >= thresholds["absolute_q75"]
            )
            & (
                future_to_recent_ratio
                >= 1.10
            )
        ).astype(int),

        "hybrid_q80_relative_1_10": (
            (
                future
                >= thresholds["absolute_q80"]
            )
            & (
                future_to_recent_ratio
                >= 1.10
            )
        ).astype(int),

        "hybrid_q80_relative_1_25": (
            (
                future
                >= thresholds["absolute_q80"]
            )
            & (
                future_to_recent_ratio
                >= 1.25
            )
        ).astype(int),
    }


def summarize_candidate(
    frame: pd.DataFrame,
    labels: pd.Series,
    candidate: str,
    split_name: str,
) -> dict:
    positive_frame = frame.loc[
        labels == 1
    ]

    repository_positive_counts = (
        positive_frame.groupby("repo_id")
        .size()
    )

    return {
        "candidate": candidate,
        "split": split_name,
        "rows": int(len(frame)),
        "positive_rows": int(labels.sum()),
        "negative_rows": int(
            len(labels) - labels.sum()
        ),
        "positive_rate": float(
            labels.mean()
        ),
        "repositories": int(
            frame["repo_id"].nunique()
        ),
        "repositories_with_positive": int(
            positive_frame[
                "repo_id"
            ].nunique()
        ),
        "median_positive_rows_per_positive_repo": (
            float(
                repository_positive_counts
                .median()
            )
            if len(
                repository_positive_counts
            )
            else 0.0
        ),
        "majority_baseline_accuracy": float(
            max(
                labels.mean(),
                1 - labels.mean(),
            )
        ),
        "future_target_mean_class_0": float(
            frame.loc[
                labels == 0,
                TARGET,
            ].mean()
        ),
        "future_target_mean_class_1": float(
            frame.loc[
                labels == 1,
                TARGET,
            ].mean()
        ),
    }


def run_univariate_audit(
    frame: pd.DataFrame,
    candidate_labels: dict,
    feature_columns: list[str],
    split_name: str,
) -> pd.DataFrame:
    rows = []

    for candidate, labels in (
        candidate_labels.items()
    ):
        if labels.nunique() < 2:
            continue

        for feature in feature_columns:
            values = frame[feature].astype(
                float
            )

            direct_auc = float(
                roc_auc_score(
                    labels,
                    values,
                )
            )

            predictive_auc = max(
                direct_auc,
                1.0 - direct_auc,
            )

            rows.append(
                {
                    "candidate": candidate,
                    "split": split_name,
                    "feature": feature,
                    "direct_auc": direct_auc,
                    "predictive_auc": (
                        predictive_auc
                    ),
                    "best_single_threshold_accuracy": (
                        best_threshold_accuracy(
                            values,
                            labels,
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(
        VALIDATION_PATH
    )

    # Read only the test header. Test values and
    # labels remain completely untouched.
    test_columns = pd.read_csv(
        TEST_PATH,
        nrows=0,
    ).columns.tolist()

    required_columns = {
        "repo_id",
        TARGET,
        RECENT_SUM,
    }

    for split_name, frame in [
        ("train", train),
        ("validation", validation),
    ]:
        missing = (
            required_columns
            - set(frame.columns)
        )

        if missing:
            raise ValueError(
                f"{split_name} is missing: "
                f"{sorted(missing)}"
            )

    thresholds = {
        "absolute_q75": float(
            np.ceil(
                train[TARGET].quantile(
                    0.75
                )
            )
        ),
        "absolute_q80": float(
            np.ceil(
                train[TARGET].quantile(
                    0.80
                )
            )
        ),
        "absolute_q85": float(
            np.ceil(
                train[TARGET].quantile(
                    0.85
                )
            )
        ),
        "absolute_q90": float(
            np.ceil(
                train[TARGET].quantile(
                    0.90
                )
            )
        ),
    }

    train_labels = create_candidate_labels(
        train,
        thresholds,
    )

    validation_labels = (
        create_candidate_labels(
            validation,
            thresholds,
        )
    )

    summary_rows = []

    for candidate, labels in (
        train_labels.items()
    ):
        summary_rows.append(
            summarize_candidate(
                train,
                labels,
                candidate,
                "train",
            )
        )

    for candidate, labels in (
        validation_labels.items()
    ):
        summary_rows.append(
            summarize_candidate(
                validation,
                labels,
                candidate,
                "validation",
            )
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        ["candidate", "split"]
    )

    feature_columns = (
        get_numeric_historical_features(
            train
        )
    )

    train_univariate = (
        run_univariate_audit(
            train,
            train_labels,
            feature_columns,
            "train",
        )
    )

    validation_univariate = (
        run_univariate_audit(
            validation,
            validation_labels,
            feature_columns,
            "validation",
        )
    )

    univariate = pd.concat(
        [
            train_univariate,
            validation_univariate,
        ],
        ignore_index=True,
    )

    best_univariate = (
        univariate.sort_values(
            [
                "candidate",
                "split",
                "predictive_auc",
                "best_single_threshold_accuracy",
            ],
            ascending=[
                True,
                True,
                False,
                False,
            ],
        )
        .groupby(
            ["candidate", "split"],
            as_index=False,
        )
        .first()
    )

    comparison = summary.merge(
        best_univariate,
        on=["candidate", "split"],
        how="left",
    )

    train_recent = (
        train[RECENT_SUM]
        .astype(float)
        .clip(lower=1.0)
    )

    validation_recent = (
        validation[RECENT_SUM]
        .astype(float)
        .clip(lower=1.0)
    )

    distribution_summary = {
        "training_rows": int(len(train)),
        "validation_rows": int(
            len(validation)
        ),
        "training_repositories": int(
            train["repo_id"].nunique()
        ),
        "validation_repositories": int(
            validation[
                "repo_id"
            ].nunique()
        ),
        "training_thresholds": thresholds,
        "training_future_target": {
            "mean": float(
                train[TARGET].mean()
            ),
            "median": float(
                train[TARGET].median()
            ),
            "maximum": float(
                train[TARGET].max()
            ),
        },
        "validation_future_target": {
            "mean": float(
                validation[TARGET].mean()
            ),
            "median": float(
                validation[
                    TARGET
                ].median()
            ),
            "maximum": float(
                validation[TARGET].max()
            ),
        },
        "training_future_to_recent_ratio": {
            "median": float(
                (
                    train[TARGET]
                    / train_recent
                ).median()
            ),
            "q75": float(
                (
                    train[TARGET]
                    / train_recent
                ).quantile(0.75)
            ),
            "q90": float(
                (
                    train[TARGET]
                    / train_recent
                ).quantile(0.90)
            ),
        },
        "validation_future_to_recent_ratio": {
            "median": float(
                (
                    validation[TARGET]
                    / validation_recent
                ).median()
            ),
            "q75": float(
                (
                    validation[TARGET]
                    / validation_recent
                ).quantile(0.75)
            ),
            "q90": float(
                (
                    validation[TARGET]
                    / validation_recent
                ).quantile(0.90)
            ),
        },
        "historical_feature_columns": (
            feature_columns
        ),
        "test_file_status": (
            "header_only_values_not_loaded"
        ),
        "test_column_count": len(
            test_columns
        ),
    }

    summary.to_csv(
        OUTPUT_DIR
        / "corrected_target_candidate_summary.csv",
        index=False,
    )

    univariate.to_csv(
        OUTPUT_DIR
        / "corrected_target_univariate_audit.csv",
        index=False,
    )

    best_univariate.to_csv(
        OUTPUT_DIR
        / "corrected_target_best_univariate.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "corrected_target_candidate_comparison.csv",
        index=False,
    )

    (
        OUTPUT_DIR
        / "corrected_target_distribution_summary.json"
    ).write_text(
        json.dumps(
            distribution_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "=" * 110,
        "CORRECTED HIGH-GROWTH TARGET AUDIT",
        "=" * 110,
        "",
        "Thresholds derived only from training data:",
        json.dumps(
            thresholds,
            indent=2,
        ),
        "",
        (
            "Test set status: header read only; "
            "test values and targets were not loaded."
        ),
        "",
        "CANDIDATE COMPARISON:",
        comparison.to_string(index=False),
        "",
        "Interpretation:",
        (
            "- A valid candidate needs enough positive "
            "examples and repository coverage."
        ),
        (
            "- A candidate with perfect univariate "
            "prediction should be examined carefully."
        ),
        (
            "- Validation positive rate may differ from "
            "training because the validation period has "
            "stronger future growth."
        ),
    ]

    report = "\n".join(report_lines)

    report_path = (
        OUTPUT_DIR
        / "corrected_target_audit_report.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print(report)
    print()
    print(
        "Corrected target audit completed "
        "successfully."
    )


if __name__ == "__main__":
    main()
