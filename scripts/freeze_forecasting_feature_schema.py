import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "phase3_config.json"
)

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

SCHEMA_PATH = (
    DATA_DIR
    / "forecasting_schema.json"
)

TARGET = "future_4week_stars"

FEATURE_COLUMNS = [
    "clean_history_weeks",
    "weekly_new_stars",
    "cumulative_stars",
    "previous_week_stars",
    "weekly_growth_rate",
    "herd_momentum_score",
    "lag_2_week_stars",
    "rolling_3week_mean_stars",
    "rolling_4week_sum_stars",
    "weekly_growth_acceleration",
]

IDENTIFIER_COLUMNS = [
    "repo_id",
    "repo_full_name",
    "cutoff_week",
    "target_end_week",
]

EXCLUDED_REDUNDANT_FEATURES = {
    "lag_1_week_stars": (
        "Exactly identical to "
        "previous_week_stars."
    ),
    "cumulative_growth_lag_1": (
        "Exactly reconstructible as "
        "cumulative_stars minus "
        "weekly_new_stars."
    ),
}

DETERMINISTIC_RELATIONSHIPS = {
    "lag_1_week_stars": (
        "previous_week_stars"
    ),
    "cumulative_growth_lag_1": (
        "cumulative_stars - "
        "weekly_new_stars"
    ),
    "weekly_growth_acceleration": (
        "weekly_new_stars - "
        "previous_week_stars"
    ),
    "weekly_growth_rate": (
        "weekly_growth_acceleration / "
        "previous_week_stars"
    ),
    "herd_momentum_score": (
        "previous_week_stars * "
        "cumulative_stars"
    ),
}


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def maximum_absolute_difference(
    first: pd.Series,
    second: pd.Series,
) -> float:
    first = pd.to_numeric(
        first,
        errors="raise",
    ).astype(float)

    second = pd.to_numeric(
        second,
        errors="raise",
    ).astype(float)

    return float(
        np.abs(
            first - second
        ).max()
    )


def validate_split(
    frame: pd.DataFrame,
    split_name: str,
) -> dict:
    required_columns = set(
        IDENTIFIER_COLUMNS
        + FEATURE_COLUMNS
        + list(
            EXCLUDED_REDUNDANT_FEATURES
        )
        + [TARGET]
    )

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{split_name} is missing: "
            f"{sorted(missing_columns)}"
        )

    numeric_features = frame[
        FEATURE_COLUMNS
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    if numeric_features.isna().any().any():
        raise ValueError(
            f"{split_name} contains missing "
            "feature values."
        )

    if not np.isfinite(
        numeric_features.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            f"{split_name} contains non-finite "
            "feature values."
        )

    target = pd.to_numeric(
        frame[TARGET],
        errors="raise",
    )

    if target.isna().any():
        raise ValueError(
            f"{split_name} contains missing "
            "target values."
        )

    if (target < 0).any():
        raise ValueError(
            f"{split_name} contains negative "
            "target values."
        )

    relationships = {
        "lag_1_equals_previous_week": (
            maximum_absolute_difference(
                frame["lag_1_week_stars"],
                frame["previous_week_stars"],
            )
        ),
        "cumulative_lag_reconstruction": (
            maximum_absolute_difference(
                frame[
                    "cumulative_growth_lag_1"
                ],
                (
                    frame["cumulative_stars"]
                    - frame[
                        "weekly_new_stars"
                    ]
                ),
            )
        ),
        "growth_acceleration_formula": (
            maximum_absolute_difference(
                frame[
                    "weekly_growth_acceleration"
                ],
                (
                    frame["weekly_new_stars"]
                    - frame[
                        "previous_week_stars"
                    ]
                ),
            )
        ),
        "growth_rate_formula": (
            maximum_absolute_difference(
                frame["weekly_growth_rate"],
                (
                    frame[
                        "weekly_growth_acceleration"
                    ]
                    / frame[
                        "previous_week_stars"
                    ]
                ),
            )
        ),
        "herd_momentum_formula": (
            maximum_absolute_difference(
                frame[
                    "herd_momentum_score"
                ],
                (
                    frame[
                        "previous_week_stars"
                    ]
                    * frame[
                        "cumulative_stars"
                    ]
                ),
            )
        ),
    }

    tolerance = 1e-10

    failed_relationships = {
        name: difference
        for name, difference
        in relationships.items()
        if difference > tolerance
    }

    if failed_relationships:
        raise ValueError(
            f"{split_name} failed formula "
            f"checks: {failed_relationships}"
        )

    return {
        "split": split_name,
        "rows": int(len(frame)),
        "repositories": int(
            frame["repo_id"].nunique()
        ),
        "target_minimum": float(
            target.min()
        ),
        "target_median": float(
            target.median()
        ),
        "target_mean": float(
            target.mean()
        ),
        "target_maximum": float(
            target.max()
        ),
        "formula_maximum_absolute_differences": (
            relationships
        ),
    }


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    # Only the test header is read.
    test_columns = pd.read_csv(
        TEST_PATH,
        nrows=0,
    ).columns.tolist()

    required_test_columns = set(
        IDENTIFIER_COLUMNS
        + FEATURE_COLUMNS
        + [TARGET]
    )

    missing_test_columns = (
        required_test_columns
        - set(test_columns)
    )

    if missing_test_columns:
        raise ValueError(
            "Forecast test header is missing: "
            f"{sorted(missing_test_columns)}"
        )

    train_validation = validate_split(
        train,
        "train",
    )

    validation_validation = validate_split(
        validation,
        "validation",
    )

    schema = {
        "task": (
            "four_week_repository_star_"
            "count_forecasting"
        ),
        "target": TARGET,
        "forecast_horizon_weeks": 4,
        "primary_metric": "rmsle",
        "secondary_metrics": [
            "mae",
            "rmse",
            "smape",
            "forecast_bias",
        ],
        "identifier_columns": (
            IDENTIFIER_COLUMNS
        ),
        "feature_count": len(
            FEATURE_COLUMNS
        ),
        "feature_columns": (
            FEATURE_COLUMNS
        ),
        "feature_groups": {
            "history_and_maturity": [
                "clean_history_weeks",
                "cumulative_stars",
            ],
            "recent_activity": [
                "weekly_new_stars",
                "previous_week_stars",
                "lag_2_week_stars",
                "rolling_3week_mean_stars",
                "rolling_4week_sum_stars",
            ],
            "growth_dynamics": [
                "weekly_growth_rate",
                "weekly_growth_acceleration",
            ],
            "herd_interaction": [
                "herd_momentum_score",
            ],
        },
        "excluded_redundant_features": (
            EXCLUDED_REDUNDANT_FEATURES
        ),
        "deterministic_relationships": (
            DETERMINISTIC_RELATIONSHIPS
        ),
        "modeling_rules": {
            "negative_predictions": (
                "Clip to zero before metrics "
                "or persistence."
            ),
            "linear_and_distance_models": (
                "Use feature scaling inside "
                "the training pipeline."
            ),
            "tree_models": (
                "Scaling is not required."
            ),
            "target_transform_candidates": [
                "raw",
                "log1p",
            ],
            "rolling_feature_ablation": (
                "Compare the full schema with "
                "an ablation removing one of "
                "the two highly correlated "
                "rolling features."
            ),
            "herd_feature_ablation": (
                "Compare the full schema with "
                "a version excluding "
                "herd_momentum_score."
            ),
        },
        "train_validation_checks": [
            train_validation,
            validation_validation,
        ],
        "test_columns": test_columns,
        "test_values_loaded": False,
        "selection_rationale": (
            "The schema removes exact duplicate "
            "and exactly reconstructible lag "
            "features while preserving recent "
            "activity, repository maturity, "
            "growth dynamics, rolling history, "
            "and the explicit herd-momentum "
            "interaction."
        ),
    }

    SCHEMA_PATH.write_text(
        json.dumps(
            schema,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_json(CONFIG_PATH)

    forecasting = config["forecasting"]

    forecasting[
        "feature_schema_path"
    ] = str(
        SCHEMA_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    forecasting[
        "feature_columns"
    ] = FEATURE_COLUMNS

    forecasting[
        "excluded_redundant_features"
    ] = EXCLUDED_REDUNDANT_FEATURES

    forecasting[
        "target_transform_candidates"
    ] = [
        "raw",
        "log1p",
    ]

    forecasting[
        "prediction_minimum"
    ] = 0.0

    CONFIG_PATH.write_text(
        json.dumps(
            config,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 105)
    print(
        "FROZEN FORECASTING FEATURE SCHEMA"
    )
    print("=" * 105)

    print(
        json.dumps(
            schema,
            indent=2,
        )
    )

    print()
    print(
        "Forecast test values were not loaded."
    )


if __name__ == "__main__":
    main()
