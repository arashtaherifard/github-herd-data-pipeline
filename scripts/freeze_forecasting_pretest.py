import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "phase3_config.json"
)

SCHEMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "forecasting"
    / "forecasting_schema.json"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "forecasting"
    / "forecast_primary_train.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "forecasting"
    / "forecast_primary_validation.csv"
)

AUDIT_METADATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "forecasting"
    / "baseline_residual_audit"
    / "audit_metadata.json"
)

AUDIT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "forecasting"
    / "baseline_residual_audit"
    / "validation_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "forecasting"
    / "pretest_freeze"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
)

MODEL_PATH = (
    MODEL_DIR
    / "selected_forecaster_pretest.joblib"
)

MODEL_METADATA_PATH = (
    MODEL_DIR
    / "selected_forecaster_pretest_metadata.json"
)

FREEZE_METADATA_PATH = (
    OUTPUT_DIR
    / "pretest_freeze_metadata.json"
)

FREEZE_MANIFEST_PATH = (
    OUTPUT_DIR
    / "pretest_freeze_manifest.json"
)

REPRODUCTION_PATH = (
    OUTPUT_DIR
    / "train_only_validation_reproduction.csv"
)

RANDOM_STATE = 42
N_ESTIMATORS = 400

EXPECTED_FAMILY = (
    "additive_residual_extra_trees"
)

EXPECTED_MODE = "additive_residual"
EXPECTED_MODEL_KIND = "extra_trees"
EXPECTED_FEATURE_SET = "recent_only"

EXPECTED_PARAMETERS = {
    "max_depth": 4,
    "max_features": "sqrt",
    "min_samples_leaf": 10,
}

FEATURES = [
    "weekly_new_stars",
    "previous_week_stars",
    "weekly_growth_rate",
    "lag_2_week_stars",
    "rolling_3week_mean_stars",
    "rolling_4week_sum_stars",
    "weekly_growth_acceleration",
]

TARGET = "future_4week_stars"

IDENTIFIERS = [
    "repo_id",
    "repo_full_name",
    "cutoff_week",
    "target_end_week",
]

BASELINE_COLUMN = (
    "rolling_3week_mean_stars"
)

BASELINE_MULTIPLIER = 4.0

AUDIT_PREDICTION_COLUMN = (
    "prediction_additive_residual_extra_trees"
)

REPRODUCTION_TOLERANCE = 1e-10


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


def current_git_commit() -> str:
    result = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def current_git_branch() -> str:
    result = subprocess.run(
        [
            "git",
            "branch",
            "--show-current",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def baseline_prediction(
    frame: pd.DataFrame,
) -> np.ndarray:
    return (
        BASELINE_MULTIPLIER
        * frame[
            BASELINE_COLUMN
        ].to_numpy(dtype=float)
    )


def build_residual_estimator(
) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=int(
            EXPECTED_PARAMETERS[
                "max_depth"
            ]
        ),
        max_features=(
            EXPECTED_PARAMETERS[
                "max_features"
            ]
        ),
        min_samples_leaf=int(
            EXPECTED_PARAMETERS[
                "min_samples_leaf"
            ]
        ),
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def validate_frame(
    frame: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    required = set(
        IDENTIFIERS
        + FEATURES
        + [
            TARGET,
            BASELINE_COLUMN,
        ]
    )

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{split_name} is missing: "
            f"{sorted(missing)}"
        )

    prepared = frame.copy()

    prepared["cutoff_week"] = (
        pd.to_datetime(
            prepared["cutoff_week"],
            utc=True,
            errors="raise",
        )
    )

    prepared["target_end_week"] = (
        pd.to_datetime(
            prepared[
                "target_end_week"
            ],
            utc=True,
            errors="raise",
        )
    )

    numeric_columns = sorted(
        set(
            FEATURES
            + [
                TARGET,
                BASELINE_COLUMN,
            ]
        )
    )

    prepared[
        numeric_columns
    ] = prepared[
        numeric_columns
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    if prepared[
        numeric_columns
    ].isna().any().any():
        raise ValueError(
            f"{split_name} contains "
            "missing numeric values."
        )

    if not np.isfinite(
        prepared[
            numeric_columns
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{split_name} contains "
            "non-finite numeric values."
        )

    if (
        prepared[TARGET] < 0
    ).any():
        raise ValueError(
            f"{split_name} contains "
            "negative targets."
        )

    duplicate_keys = int(
        prepared.duplicated(
            subset=[
                "repo_id",
                "cutoff_week",
            ]
        ).sum()
    )

    if duplicate_keys:
        raise ValueError(
            f"{split_name} contains "
            f"{duplicate_keys} duplicate "
            "repository/cutoff keys."
        )

    return prepared


def assert_selected_approach(
    audit_metadata: dict,
) -> dict:
    selected = audit_metadata[
        "selected_development_approach"
    ]

    checks = {
        "family": EXPECTED_FAMILY,
        "mode": EXPECTED_MODE,
        "model_kind": (
            EXPECTED_MODEL_KIND
        ),
        "feature_set": (
            EXPECTED_FEATURE_SET
        ),
    }

    for key, expected in (
        checks.items()
    ):
        actual = selected[key]

        if actual != expected:
            raise ValueError(
                f"Selected {key} changed: "
                f"expected {expected!r}, "
                f"found {actual!r}."
            )

    parameters = json.loads(
        selected["parameters"]
    )

    if parameters != (
        EXPECTED_PARAMETERS
    ):
        raise ValueError(
            "Selected parameters changed: "
            f"expected "
            f"{EXPECTED_PARAMETERS}, "
            f"found {parameters}."
        )

    return selected


def predict_artifact(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
) -> np.ndarray:
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

    return np.clip(
        baseline + correction,
        a_min=float(
            artifact[
                "prediction_minimum"
            ]
        ),
        a_max=None,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    protected_paths = [
        MODEL_PATH,
        MODEL_METADATA_PATH,
        FREEZE_METADATA_PATH,
        FREEZE_MANIFEST_PATH,
        REPRODUCTION_PATH,
    ]

    existing = [
        path
        for path in protected_paths
        if path.exists()
    ]

    if existing:
        raise RuntimeError(
            "Pretest-freeze artifacts "
            "already exist. Refusing to "
            "overwrite:\n"
            + "\n".join(
                str(path)
                for path in existing
            )
        )

    config = load_json(
        CONFIG_PATH
    )

    schema = load_json(
        SCHEMA_PATH
    )

    audit_metadata = load_json(
        AUDIT_METADATA_PATH
    )

    selected = (
        assert_selected_approach(
            audit_metadata
        )
    )

    if schema["target"] != TARGET:
        raise ValueError(
            "Forecast target changed."
        )

    if not set(FEATURES).issubset(
        set(
            schema[
                "feature_columns"
            ]
        )
    ):
        raise ValueError(
            "Frozen feature schema no "
            "longer contains the selected "
            "recent-only feature set."
        )

    train = validate_frame(
        pd.read_csv(TRAIN_PATH),
        "train",
    )

    validation = validate_frame(
        pd.read_csv(
            VALIDATION_PATH
        ),
        "validation",
    )

    audit_predictions = pd.read_csv(
        AUDIT_PREDICTIONS_PATH
    )

    required_audit_columns = set(
        IDENTIFIERS
        + [
            TARGET,
            AUDIT_PREDICTION_COLUMN,
        ]
    )

    missing_audit_columns = (
        required_audit_columns
        - set(
            audit_predictions.columns
        )
    )

    if missing_audit_columns:
        raise ValueError(
            "Audit predictions are "
            f"missing: "
            f"{sorted(missing_audit_columns)}"
        )

    verification_model = (
        build_residual_estimator()
    )

    train_baseline = (
        baseline_prediction(train)
    )

    train_residual_target = (
        train[TARGET].to_numpy(
            dtype=float
        )
        - train_baseline
    )

    verification_model.fit(
        train[FEATURES],
        train_residual_target,
    )

    verification_artifact = {
        "baseline_column": (
            BASELINE_COLUMN
        ),
        "baseline_multiplier": (
            BASELINE_MULTIPLIER
        ),
        "feature_columns": (
            FEATURES
        ),
        "prediction_minimum": 0.0,
        "residual_estimator": (
            verification_model
        ),
    }

    reproduced_predictions = (
        predict_artifact(
            verification_artifact,
            validation,
        )
    )

    for date_column in [
        "cutoff_week",
        "target_end_week",
    ]:
        audit_predictions[
            date_column
        ] = pd.to_datetime(
            audit_predictions[
                date_column
            ],
            utc=True,
            errors="raise",
        )

    comparison = (
        validation[
            IDENTIFIERS
            + [TARGET]
        ]
        .merge(
            audit_predictions[
                IDENTIFIERS
                + [
                    AUDIT_PREDICTION_COLUMN
                ]
            ],
            on=IDENTIFIERS,
            how="left",
            validate="one_to_one",
        )
    )

    if comparison[
        AUDIT_PREDICTION_COLUMN
    ].isna().any():
        raise ValueError(
            "Could not align all saved "
            "audit predictions."
        )

    comparison[
        "reproduced_prediction"
    ] = reproduced_predictions

    comparison[
        "absolute_difference"
    ] = np.abs(
        comparison[
            "reproduced_prediction"
        ]
        - comparison[
            AUDIT_PREDICTION_COLUMN
        ]
    )

    maximum_difference = float(
        comparison[
            "absolute_difference"
        ].max()
    )

    if (
        maximum_difference
        > REPRODUCTION_TOLERANCE
    ):
        raise ValueError(
            "Selected implementation did "
            "not reproduce the saved "
            "validation predictions. "
            f"Maximum difference: "
            f"{maximum_difference}"
        )

    comparison.to_csv(
        REPRODUCTION_PATH,
        index=False,
    )

    development = (
        pd.concat(
            [
                train.assign(
                    development_source=(
                        "train"
                    )
                ),
                validation.assign(
                    development_source=(
                        "validation"
                    )
                ),
            ],
            ignore_index=True,
        )
        .sort_values(
            [
                "repo_id",
                "cutoff_week",
            ]
        )
        .reset_index(drop=True)
    )

    development_duplicate_keys = int(
        development.duplicated(
            subset=[
                "repo_id",
                "cutoff_week",
            ]
        ).sum()
    )

    if development_duplicate_keys:
        raise ValueError(
            "Combined development data "
            "contains duplicate keys."
        )

    final_model = (
        build_residual_estimator()
    )

    development_baseline = (
        baseline_prediction(
            development
        )
    )

    development_residual_target = (
        development[
            TARGET
        ].to_numpy(dtype=float)
        - development_baseline
    )

    final_model.fit(
        development[FEATURES],
        development_residual_target,
    )

    artifact = {
        "artifact_version": (
            "forecasting_pretest_v1"
        ),
        "task": (
            "four_week_repository_"
            "star_count_forecasting"
        ),
        "approach_family": (
            EXPECTED_FAMILY
        ),
        "mode": EXPECTED_MODE,
        "model_kind": (
            EXPECTED_MODEL_KIND
        ),
        "baseline_column": (
            BASELINE_COLUMN
        ),
        "baseline_multiplier": (
            BASELINE_MULTIPLIER
        ),
        "feature_set_name": (
            EXPECTED_FEATURE_SET
        ),
        "feature_columns": (
            FEATURES
        ),
        "target": TARGET,
        "forecast_horizon_weeks": 4,
        "prediction_minimum": 0.0,
        "residual_estimator_parameters": {
            **EXPECTED_PARAMETERS,
            "n_estimators": (
                N_ESTIMATORS
            ),
            "random_state": (
                RANDOM_STATE
            ),
        },
        "residual_estimator": (
            final_model
        ),
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    model_hash = sha256_file(
        MODEL_PATH
    )

    input_hashes = {
        str(
            path.relative_to(
                PROJECT_ROOT
            )
        ): sha256_file(path)
        for path in [
            CONFIG_PATH,
            SCHEMA_PATH,
            TRAIN_PATH,
            VALIDATION_PATH,
            AUDIT_METADATA_PATH,
            AUDIT_PREDICTIONS_PATH,
        ]
    }

    metadata = {
        "freeze_status": (
            "forecasting_pretest_frozen"
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "git_branch": (
            current_git_branch()
        ),
        "git_commit_before_freeze": (
            current_git_commit()
        ),
        "selection_source": str(
            AUDIT_METADATA_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "selected_development_approach": (
            selected
        ),
        "selection_rule": (
            audit_metadata[
                "selection_rule"
            ]
        ),
        "development_training": {
            "rows": int(
                len(development)
            ),
            "repositories": int(
                development[
                    "repo_id"
                ].nunique()
            ),
            "train_rows": int(
                len(train)
            ),
            "validation_rows": int(
                len(validation)
            ),
            "target_minimum": float(
                development[
                    TARGET
                ].min()
            ),
            "target_median": float(
                development[
                    TARGET
                ].median()
            ),
            "target_mean": float(
                development[
                    TARGET
                ].mean()
            ),
            "target_maximum": float(
                development[
                    TARGET
                ].max()
            ),
            "residual_target_mean": float(
                development_residual_target.mean()
            ),
            "residual_target_median": float(
                np.median(
                    development_residual_target
                )
            ),
        },
        "implementation_reproduction": {
            "rows": int(
                len(comparison)
            ),
            "maximum_absolute_difference": (
                maximum_difference
            ),
            "tolerance": (
                REPRODUCTION_TOLERANCE
            ),
            "passed": True,
            "reproduction_path": str(
                REPRODUCTION_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
        "artifact": {
            "path": str(
                MODEL_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": model_hash,
            "artifact_version": (
                artifact[
                    "artifact_version"
                ]
            ),
            "baseline_formula": (
                "4 * "
                "rolling_3week_mean_stars"
            ),
            "residual_model": (
                "ExtraTreesRegressor"
            ),
            "feature_columns": (
                FEATURES
            ),
            "parameters": (
                artifact[
                    "residual_estimator_parameters"
                ]
            ),
        },
        "selected_validation_metrics": {
            "rmsle": float(
                selected[
                    "validation_rmsle"
                ]
            ),
            "mae": float(
                selected[
                    "validation_mae"
                ]
            ),
            "rmse": float(
                selected[
                    "validation_rmse"
                ]
            ),
            "smape": float(
                selected[
                    "validation_smape"
                ]
            ),
            "forecast_bias": float(
                selected[
                    "validation_forecast_bias"
                ]
            ),
        },
        "selected_cv_metrics": {
            "mean_rmsle": float(
                selected[
                    "cv_mean_rmsle"
                ]
            ),
            "std_rmsle": float(
                selected[
                    "cv_std_rmsle"
                ]
            ),
            "mean_mae": float(
                selected[
                    "cv_mean_mae"
                ]
            ),
        },
        "input_hashes": input_hashes,
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
            "not_loaded_or_evaluated"
        ),
    }

    write_json(
        MODEL_METADATA_PATH,
        metadata,
    )

    write_json(
        FREEZE_METADATA_PATH,
        metadata,
    )

    manifest = {
        "freeze_name": (
            "forecasting-pretest-freeze-v1"
        ),
        "git_commit_before_freeze": (
            metadata[
                "git_commit_before_freeze"
            ]
        ),
        "model_path": (
            metadata[
                "artifact"
            ]["path"]
        ),
        "model_sha256": model_hash,
        "metadata_path": str(
            MODEL_METADATA_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "metadata_sha256": sha256_file(
            MODEL_METADATA_PATH
        ),
        "freeze_metadata_path": str(
            FREEZE_METADATA_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "freeze_metadata_sha256": (
            sha256_file(
                FREEZE_METADATA_PATH
            )
        ),
        "reproduction_path": str(
            REPRODUCTION_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "reproduction_sha256": (
            sha256_file(
                REPRODUCTION_PATH
            )
        ),
        "input_hashes": input_hashes,
        "test_data_status": (
            "not_loaded_or_evaluated"
        ),
    }

    write_json(
        FREEZE_MANIFEST_PATH,
        manifest,
    )

    config_forecasting = (
        config["forecasting"]
    )

    config_forecasting[
        "selected_approach_family"
    ] = EXPECTED_FAMILY

    config_forecasting[
        "selected_approach_mode"
    ] = EXPECTED_MODE

    config_forecasting[
        "selected_feature_set_name"
    ] = EXPECTED_FEATURE_SET

    config_forecasting[
        "selected_feature_columns"
    ] = FEATURES

    config_forecasting[
        "selected_baseline_formula"
    ] = (
        "4 * "
        "rolling_3week_mean_stars"
    )

    config_forecasting[
        "selected_residual_model"
    ] = (
        "ExtraTreesRegressor"
    )

    config_forecasting[
        "selected_residual_model_parameters"
    ] = (
        artifact[
            "residual_estimator_parameters"
        ]
    )

    config_forecasting[
        "pretest_model_path"
    ] = str(
        MODEL_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    config_forecasting[
        "pretest_model_metadata_path"
    ] = str(
        MODEL_METADATA_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    config_forecasting[
        "pretest_freeze_manifest_path"
    ] = str(
        FREEZE_MANIFEST_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    config_forecasting[
        "test_data_status"
    ] = (
        "not_loaded_or_evaluated"
    )

    write_json(
        CONFIG_PATH,
        config,
    )

    # CONFIG_PATH is both an input and an output
    # of this freeze script. Refresh its hash
    # after writing the final frozen settings,
    # then refresh all dependent metadata hashes.
    config_relative_path = str(
        CONFIG_PATH.relative_to(
            PROJECT_ROOT
        )
    )

    final_config_hash = sha256_file(
        CONFIG_PATH
    )

    metadata["input_hashes"][
        config_relative_path
    ] = final_config_hash

    write_json(
        MODEL_METADATA_PATH,
        metadata,
    )

    write_json(
        FREEZE_METADATA_PATH,
        metadata,
    )

    manifest["input_hashes"][
        config_relative_path
    ] = final_config_hash

    manifest["metadata_sha256"] = (
        sha256_file(
            MODEL_METADATA_PATH
        )
    )

    manifest[
        "freeze_metadata_sha256"
    ] = sha256_file(
        FREEZE_METADATA_PATH
    )

    write_json(
        FREEZE_MANIFEST_PATH,
        manifest,
    )

    print("=" * 108)
    print(
        "FORECASTING PRETEST FREEZE"
    )
    print("=" * 108)

    print(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        )
    )

    print()
    print("=" * 108)
    print(
        "FREEZE MANIFEST"
    )
    print("=" * 108)

    print(
        json.dumps(
            manifest,
            indent=2,
        )
    )

    print()
    print(
        "Forecast test data was not "
        "loaded or evaluated."
    )


if __name__ == "__main__":
    main()
