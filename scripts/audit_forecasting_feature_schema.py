import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "forecasting"
)

TRAIN_PATH = (
    DATA_DIR
    / "forecast_primary_train.csv"
)

VALIDATION_PATH = (
    DATA_DIR
    / "forecast_primary_validation.csv"
)

TEST_PATH = (
    DATA_DIR
    / "forecast_primary_test.csv"
)

CLASSIFICATION_SCHEMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "classification"
    / "classification_schema.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "forecasting"
    / "feature_schema_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TARGET = "future_4week_stars"

IDENTIFIER_COLUMNS = [
    "repo_id",
    "repo_full_name",
    "cutoff_week",
    "target_end_week",
]


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_dates(
    frame: pd.DataFrame,
    split_name: str,
) -> dict:
    cutoff = pd.to_datetime(
        frame["cutoff_week"],
        utc=True,
        errors="raise",
    )

    target_end = pd.to_datetime(
        frame["target_end_week"],
        utc=True,
        errors="raise",
    )

    horizon_days = (
        target_end - cutoff
    ).dt.total_seconds() / 86400.0

    invalid_order = int(
        (target_end <= cutoff).sum()
    )

    return {
        "split": split_name,
        "invalid_date_order_rows": invalid_order,
        "minimum_horizon_days": float(
            horizon_days.min()
        ),
        "maximum_horizon_days": float(
            horizon_days.max()
        ),
        "unique_horizon_days": sorted(
            float(value)
            for value in horizon_days.unique()
        ),
    }


def numeric_summary(
    frame: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    rows = []

    target_values = pd.to_numeric(
        frame[TARGET],
        errors="raise",
    )

    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(
            frame[column]
        ):
            continue

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        finite_mask = np.isfinite(
            values.to_numpy(dtype=float)
        )

        finite_values = values[
            finite_mask
        ]

        correlation = np.nan

        if (
            column != TARGET
            and finite_values.nunique() > 1
            and target_values.nunique() > 1
            and values.notna().all()
        ):
            correlation = float(
                values.corr(target_values)
            )

        rows.append(
            {
                "split": split_name,
                "column": column,
                "dtype": str(
                    frame[column].dtype
                ),
                "rows": int(len(frame)),
                "missing_rows": int(
                    values.isna().sum()
                ),
                "non_finite_rows": int(
                    (~finite_mask).sum()
                ),
                "unique_values": int(
                    values.nunique(
                        dropna=True
                    )
                ),
                "minimum": (
                    float(
                        finite_values.min()
                    )
                    if len(finite_values)
                    else np.nan
                ),
                "q01": (
                    float(
                        finite_values.quantile(
                            0.01
                        )
                    )
                    if len(finite_values)
                    else np.nan
                ),
                "median": (
                    float(
                        finite_values.median()
                    )
                    if len(finite_values)
                    else np.nan
                ),
                "mean": (
                    float(
                        finite_values.mean()
                    )
                    if len(finite_values)
                    else np.nan
                ),
                "q99": (
                    float(
                        finite_values.quantile(
                            0.99
                        )
                    )
                    if len(finite_values)
                    else np.nan
                ),
                "maximum": (
                    float(
                        finite_values.max()
                    )
                    if len(finite_values)
                    else np.nan
                ),
                "zero_rows": int(
                    (values == 0).sum()
                ),
                "negative_rows": int(
                    (values < 0).sum()
                ),
                "correlation_with_target": (
                    correlation
                ),
            }
        )

    return pd.DataFrame(rows)


def distribution_shift(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows = []

    for feature in features:
        train_values = pd.to_numeric(
            train[feature],
            errors="raise",
        ).astype(float)

        validation_values = pd.to_numeric(
            validation[feature],
            errors="raise",
        ).astype(float)

        train_std = float(
            train_values.std(ddof=0)
        )

        standardized_mean_shift = (
            (
                float(
                    validation_values.mean()
                )
                - float(
                    train_values.mean()
                )
            )
            / train_std
            if train_std > 0
            else np.nan
        )

        rows.append(
            {
                "feature": feature,
                "train_mean": float(
                    train_values.mean()
                ),
                "validation_mean": float(
                    validation_values.mean()
                ),
                "train_median": float(
                    train_values.median()
                ),
                "validation_median": float(
                    validation_values.median()
                ),
                "train_std": train_std,
                "standardized_mean_shift": (
                    standardized_mean_shift
                ),
                "train_minimum": float(
                    train_values.min()
                ),
                "validation_minimum": float(
                    validation_values.min()
                ),
                "train_maximum": float(
                    train_values.max()
                ),
                "validation_maximum": float(
                    validation_values.max()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "standardized_mean_shift",
        key=lambda series: series.abs(),
        ascending=False,
    )


def high_correlations(
    frame: pd.DataFrame,
    features: list[str],
    threshold: float = 0.95,
) -> pd.DataFrame:
    numeric = frame[
        features
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    correlation_matrix = numeric.corr()

    rows = []

    for first_index, first_feature in enumerate(
        features
    ):
        for second_feature in features[
            first_index + 1:
        ]:
            correlation = float(
                correlation_matrix.loc[
                    first_feature,
                    second_feature,
                ]
            )

            if abs(correlation) >= threshold:
                rows.append(
                    {
                        "feature_1": first_feature,
                        "feature_2": second_feature,
                        "correlation": correlation,
                        "absolute_correlation": abs(
                            correlation
                        ),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "correlation",
                "absolute_correlation",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        "absolute_correlation",
        ascending=False,
    )


def exact_duplicate_features(
    frame: pd.DataFrame,
    features: list[str],
) -> list[dict]:
    duplicates = []

    for first_index, first_feature in enumerate(
        features
    ):
        first_values = frame[
            first_feature
        ].reset_index(drop=True)

        for second_feature in features[
            first_index + 1:
        ]:
            second_values = frame[
                second_feature
            ].reset_index(drop=True)

            if first_values.equals(
                second_values
            ):
                duplicates.append(
                    {
                        "feature_1": first_feature,
                        "feature_2": second_feature,
                    }
                )

    return duplicates


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    # Read only the test header. No test values
    # or test target outcomes are loaded.
    test_columns = pd.read_csv(
        TEST_PATH,
        nrows=0,
    ).columns.tolist()

    classification_schema = load_json(
        CLASSIFICATION_SCHEMA_PATH
    )

    candidate_features = classification_schema[
        "feature_columns"
    ]

    required_columns = set(
        IDENTIFIER_COLUMNS
        + [TARGET]
        + candidate_features
    )

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
                f"{split_name} is missing "
                f"columns: {sorted(missing)}"
            )

    train_feature_values = train[
        candidate_features
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    validation_feature_values = validation[
        candidate_features
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    if train_feature_values.isna().any().any():
        raise ValueError(
            "Training candidate features "
            "contain missing values."
        )

    if (
        validation_feature_values
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Validation candidate features "
            "contain missing values."
        )

    if not np.isfinite(
        train_feature_values.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Training candidate features "
            "contain non-finite values."
        )

    if not np.isfinite(
        validation_feature_values.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Validation candidate features "
            "contain non-finite values."
        )

    target_train = pd.to_numeric(
        train[TARGET],
        errors="raise",
    )

    target_validation = pd.to_numeric(
        validation[TARGET],
        errors="raise",
    )

    if (target_train < 0).any():
        raise ValueError(
            "Training target contains "
            "negative values."
        )

    if (target_validation < 0).any():
        raise ValueError(
            "Validation target contains "
            "negative values."
        )

    train_duplicates = int(
        train.duplicated(
            subset=[
                "repo_id",
                "cutoff_week",
            ]
        ).sum()
    )

    validation_duplicates = int(
        validation.duplicated(
            subset=[
                "repo_id",
                "cutoff_week",
            ]
        ).sum()
    )

    train_date_audit = validate_dates(
        train,
        "train",
    )

    validation_date_audit = validate_dates(
        validation,
        "validation",
    )

    column_summary = pd.concat(
        [
            numeric_summary(
                train,
                "train",
            ),
            numeric_summary(
                validation,
                "validation",
            ),
        ],
        ignore_index=True,
    )

    shift_summary = distribution_shift(
        train,
        validation,
        candidate_features,
    )

    correlated_features = high_correlations(
        train,
        candidate_features,
        threshold=0.95,
    )

    exact_duplicates = (
        exact_duplicate_features(
            train,
            candidate_features,
        )
    )

    constant_features = [
        feature
        for feature in candidate_features
        if train[feature].nunique(
            dropna=False
        ) <= 1
    ]

    non_model_columns = [
        column
        for column in train.columns
        if column
        not in set(
            candidate_features
            + IDENTIFIER_COLUMNS
            + [TARGET]
        )
    ]

    column_summary.to_csv(
        OUTPUT_DIR
        / "forecasting_numeric_column_summary.csv",
        index=False,
    )

    shift_summary.to_csv(
        OUTPUT_DIR
        / "forecasting_feature_distribution_shift.csv",
        index=False,
    )

    correlated_features.to_csv(
        OUTPUT_DIR
        / "forecasting_high_feature_correlations.csv",
        index=False,
    )

    audit_record = {
        "training_rows": int(len(train)),
        "validation_rows": int(
            len(validation)
        ),
        "training_repositories": int(
            train["repo_id"].nunique()
        ),
        "validation_repositories": int(
            validation["repo_id"].nunique()
        ),
        "target": TARGET,
        "candidate_feature_count": len(
            candidate_features
        ),
        "candidate_features": (
            candidate_features
        ),
        "identifier_columns": (
            IDENTIFIER_COLUMNS
        ),
        "non_model_columns": (
            non_model_columns
        ),
        "constant_candidate_features": (
            constant_features
        ),
        "exact_duplicate_candidate_features": (
            exact_duplicates
        ),
        "train_duplicate_keys": (
            train_duplicates
        ),
        "validation_duplicate_keys": (
            validation_duplicates
        ),
        "train_target_summary": {
            "minimum": float(
                target_train.min()
            ),
            "median": float(
                target_train.median()
            ),
            "mean": float(
                target_train.mean()
            ),
            "maximum": float(
                target_train.max()
            ),
            "zero_rows": int(
                (target_train == 0).sum()
            ),
        },
        "validation_target_summary": {
            "minimum": float(
                target_validation.min()
            ),
            "median": float(
                target_validation.median()
            ),
            "mean": float(
                target_validation.mean()
            ),
            "maximum": float(
                target_validation.max()
            ),
            "zero_rows": int(
                (
                    target_validation == 0
                ).sum()
            ),
        },
        "train_date_audit": (
            train_date_audit
        ),
        "validation_date_audit": (
            validation_date_audit
        ),
        "test_column_count": len(
            test_columns
        ),
        "test_columns": test_columns,
        "test_values_loaded": False,
    }

    (
        OUTPUT_DIR
        / "forecasting_feature_schema_audit.json"
    ).write_text(
        json.dumps(
            audit_record,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 105)
    print("FORECASTING FEATURE-SCHEMA AUDIT")
    print("=" * 105)

    print(
        json.dumps(
            audit_record,
            indent=2,
        )
    )

    print()
    print("=" * 105)
    print("TRAINING NUMERIC COLUMN SUMMARY")
    print("=" * 105)

    print(
        column_summary[
            column_summary["split"]
            == "train"
        ].to_string(index=False)
    )

    print()
    print("=" * 105)
    print("TRAIN–VALIDATION FEATURE SHIFT")
    print("=" * 105)

    print(
        shift_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 105)
    print("HIGH FEATURE CORRELATIONS")
    print("=" * 105)

    if correlated_features.empty:
        print(
            "No absolute feature correlations "
            "at or above 0.95."
        )
    else:
        print(
            correlated_features.to_string(
                index=False
            )
        )

    print()
    print(
        "Forecast test values were not "
        "loaded or evaluated."
    )


if __name__ == "__main__":
    main()
