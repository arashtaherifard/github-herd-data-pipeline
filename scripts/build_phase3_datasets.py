import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase3_forecast_split_audit import create_repository_aware_split
from phase3_split_planning import (
    build_forecast_dataset,
    choose_date_cutoffs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "phase3_config.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELING_DIR = PROJECT_ROOT / "data" / "modeling"

CLASSIFICATION_DIR = MODELING_DIR / "classification"
FORECASTING_DIR = MODELING_DIR / "forecasting"
RECOMMENDATION_DIR = MODELING_DIR / "recommendation"

for directory in [
    MODELING_DIR,
    CLASSIFICATION_DIR,
    FORECASTING_DIR,
    RECOMMENDATION_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def save_dataframe(
    frame: pd.DataFrame,
    path: Path,
    manifest: dict,
) -> None:
    frame.to_csv(path, index=False)

    relative_path = str(path.relative_to(PROJECT_ROOT))

    manifest[relative_path] = {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "sha256": sha256_file(path),
    }


def build_classification_data(
    config: dict,
    manifest: dict,
) -> dict:
    settings = config["classification"]

    primary_target = settings["target"]
    secondary_target = settings["secondary_target"]
    target_base = settings["target_base"]
    recent_activity_column = settings[
        "recent_activity_column"
    ]

    relative_multiplier = float(
        settings["relative_growth_multiplier"]
    )

    absolute_quantile = float(
        settings["absolute_high_growth_quantile"]
    )

    source_paths = {
        "train": (
            FORECASTING_DIR
            / "forecast_primary_train.csv"
        ),
        "validation": (
            FORECASTING_DIR
            / "forecast_primary_validation.csv"
        ),
        "test": (
            FORECASTING_DIR
            / "forecast_primary_test.csv"
        ),
    }

    source_frames = {
        split_name: pd.read_csv(path)
        for split_name, path in source_paths.items()
    }

    train_source = source_frames["train"]

    required_columns = set(
        settings["feature_columns"]
    ) | {
        "repo_id",
        "repo_full_name",
        "cutoff_week",
        "target_end_week",
        target_base,
        recent_activity_column,
    }

    for split_name, frame in source_frames.items():
        missing_columns = (
            required_columns - set(frame.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{split_name} classification source "
                "is missing columns: "
                f"{sorted(missing_columns)}"
            )

    absolute_threshold = float(
        np.ceil(
            train_source[target_base].quantile(
                absolute_quantile
            )
        )
    )

    def add_targets(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        result = frame.copy()

        future_growth = pd.to_numeric(
            result[target_base],
            errors="raise",
        ).astype(float)

        recent_activity = pd.to_numeric(
            result[recent_activity_column],
            errors="raise",
        ).astype(float)

        if (recent_activity <= 0).any():
            raise ValueError(
                "Relative-growth target requires "
                "strictly positive historical "
                "four-week activity."
            )

        result[primary_target] = (
            future_growth
            >= (
                relative_multiplier
                * recent_activity
            )
        ).astype(int)

        result[secondary_target] = (
            future_growth
            >= absolute_threshold
        ).astype(int)

        return result.sort_values(
            ["repo_id", "cutoff_week"]
        ).reset_index(drop=True)

    classification_frames = {
        split_name: add_targets(frame)
        for split_name, frame
        in source_frames.items()
    }

    # Remove obsolete leakage-prone files from
    # the previous repository-level design.
    for legacy_name in [
        "classification_development.csv",
        "classification_test.csv",
    ]:
        legacy_path = (
            CLASSIFICATION_DIR / legacy_name
        )

        if legacy_path.exists():
            legacy_path.unlink()

    split_summary = {}

    for split_name, frame in (
        classification_frames.items()
    ):
        output_path = (
            CLASSIFICATION_DIR
            / f"classification_{split_name}.csv"
        )

        save_dataframe(
            frame,
            output_path,
            manifest,
        )

        split_summary[split_name] = {
            "rows": int(len(frame)),
            "repositories": int(
                frame["repo_id"].nunique()
            ),
        }

        # Do not inspect or expose the untouched
        # test-label distribution during model
        # development.
        if split_name != "test":
            split_summary[split_name][
                "primary_target_class_counts"
            ] = {
                str(key): int(value)
                for key, value in (
                    frame[primary_target]
                    .value_counts()
                    .sort_index()
                    .items()
                )
            }

            split_summary[split_name][
                "secondary_target_class_counts"
            ] = {
                str(key): int(value)
                for key, value in (
                    frame[secondary_target]
                    .value_counts()
                    .sort_index()
                    .items()
                )
            }
        else:
            split_summary[split_name][
                "target_distribution_status"
            ] = "not_summarized_during_development"

    schema = {
        "primary_target": primary_target,
        "secondary_target": secondary_target,
        "target_base": target_base,
        "identifier_columns": [
            "repo_id",
            "repo_full_name",
            "cutoff_week",
            "target_end_week",
        ],
        "feature_columns": settings[
            "feature_columns"
        ],
        "split_strategy": settings[
            "split_strategy"
        ],
        "cross_validation": {
            "strategy": settings[
                "cross_validation_strategy"
            ],
            "folds": int(
                settings["cross_validation_folds"]
            ),
            "validation_samples_per_repository": int(
                settings[
                    "cross_validation_"
                    "validation_samples_per_repository"
                ]
            ),
            "purge_weeks": int(
                settings["purge_weeks"]
            ),
            "require_all_repositories": bool(
                settings[
                    "cross_validation_"
                    "require_all_repositories"
                ]
            ),
            "selection_rationale": settings[
                "cross_validation_selection_rationale"
            ],
        },
        "primary_target_definition": {
            "description": (
                "The next four weeks receive at "
                "least 1.5 times as many stars as "
                "the previous four weeks."
            ),
            "future_column": target_base,
            "historical_column": (
                recent_activity_column
            ),
            "relative_growth_multiplier": (
                relative_multiplier
            ),
        },
        "secondary_target_definition": {
            "description": (
                "The next four-week star total is "
                "at or above the 80th percentile "
                "of the primary training period."
            ),
            "training_quantile": absolute_quantile,
            "training_derived_threshold": (
                absolute_threshold
            ),
        },
        "target_feature_ablation": settings[
            "target_feature_ablation"
        ],
        "leakage_controls": [
            (
                "Targets are created from future "
                "four-week outcomes."
            ),
            (
                "All predictors are observed at or "
                "before the cutoff week."
            ),
            (
                "The absolute target threshold is "
                "derived only from training data."
            ),
            (
                "Train, validation, and test use "
                "purged temporal repository-aware "
                "forecasting splits."
            ),
            (
                "The test target distribution is "
                "not summarized during development."
            ),
        ],
    }

    schema_path = (
        CLASSIFICATION_DIR
        / "classification_schema.json"
    )

    schema_path.write_text(
        json.dumps(
            schema,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest[
        str(schema_path.relative_to(PROJECT_ROOT))
    ] = {
        "type": "schema",
        "sha256": sha256_file(schema_path),
    }

    return {
        "design_status": (
            "corrected_after_collection_cap_"
            "leakage_audit"
        ),
        "primary_target": primary_target,
        "secondary_target": secondary_target,
        "relative_growth_multiplier": (
            relative_multiplier
        ),
        "absolute_training_quantile": (
            absolute_quantile
        ),
        "absolute_training_threshold": (
            absolute_threshold
        ),
        "feature_count": len(
            settings["feature_columns"]
        ),
        "splits": split_summary,
    }


def build_primary_forecast_split(
    config: dict,
    forecast_data: pd.DataFrame,
    manifest: dict,
) -> dict:
    settings = config["forecasting"]["primary_split"]

    split_data, split_summary = (
        create_repository_aware_split(
            forecast_data=forecast_data,
            validation_samples=settings[
                "validation_samples_per_repository"
            ],
            test_samples=settings[
                "test_samples_per_repository"
            ],
            minimum_train_samples=settings[
                "minimum_train_samples_per_repository"
            ],
        )
    )

    split_counts = {}

    for split_name in ["train", "validation", "test"]:
        frame = split_data[
            split_data["split"] == split_name
        ].copy()

        path = (
            FORECASTING_DIR
            / f"forecast_primary_{split_name}.csv"
        )

        save_dataframe(frame, path, manifest)

        split_counts[split_name] = {
            "rows": len(frame),
            "repositories": int(
                frame["repo_id"].nunique()
            ),
            "target_mean": float(
                frame["future_4week_stars"].mean()
            ),
            "target_median": float(
                frame["future_4week_stars"].median()
            ),
        }

    return {
        "split_summary": split_summary,
        "splits": split_counts,
    }


def build_cold_start_forecast_split(
    forecast_data: pd.DataFrame,
    manifest: dict,
) -> dict:
    train_end, validation_end = choose_date_cutoffs(
        forecast_data
    )

    train = forecast_data[
        forecast_data["target_end_week"] <= train_end
    ].copy()

    validation = forecast_data[
        (forecast_data["cutoff_week"] > train_end)
        & (
            forecast_data["target_end_week"]
            <= validation_end
        )
    ].copy()

    test = forecast_data[
        forecast_data["cutoff_week"] > validation_end
    ].copy()

    summary = {}

    for split_name, frame in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        frame = frame.copy()
        frame["split"] = split_name

        path = (
            FORECASTING_DIR
            / f"forecast_cold_start_{split_name}.csv"
        )

        save_dataframe(frame, path, manifest)

        summary[split_name] = {
            "rows": len(frame),
            "repositories": int(
                frame["repo_id"].nunique()
            ),
            "target_mean": float(
                frame["future_4week_stars"].mean()
            ),
            "target_median": float(
                frame["future_4week_stars"].median()
            ),
        }

    summary["training_boundary"] = str(train_end)
    summary["validation_boundary"] = str(
        validation_end
    )

    return summary


def build_forecasting_data(
    config: dict,
    manifest: dict,
) -> dict:
    forecast_data = build_forecast_dataset()

    complete_path = (
        FORECASTING_DIR
        / "forecast_supervised_complete.csv"
    )

    save_dataframe(
        forecast_data,
        complete_path,
        manifest,
    )

    primary_summary = build_primary_forecast_split(
        config=config,
        forecast_data=forecast_data,
        manifest=manifest,
    )

    cold_start_summary = (
        build_cold_start_forecast_split(
            forecast_data=forecast_data,
            manifest=manifest,
        )
    )

    return {
        "forecast_horizon_weeks": int(
            config["forecasting"]["forecast_horizon_weeks"]
        ),
        "initial_snapshot_rows": int(
            config["forecasting"]["initial_snapshot_rows"]
        ),
        "minimum_clean_history_weeks": int(
            config["forecasting"][
                "minimum_clean_history_weeks"
            ]
        ),
        "minimum_observed_rows": int(
            config["forecasting"]["minimum_observed_rows"]
        ),
        "complete_rows": len(forecast_data),
        "repositories": int(
            forecast_data["repo_id"].nunique()
        ),
        "primary_repository_aware": primary_summary,
        "secondary_cold_start": cold_start_summary,
    }


def build_recommendation_data(
    config: dict,
    manifest: dict,
) -> dict:
    settings = config["recommendation"]

    interactions = pd.read_csv(
        PROCESSED_DIR / "recommendation_features.csv"
    )

    interactions["starred_at"] = pd.to_datetime(
        interactions["starred_at"],
        errors="coerce",
        utc=True,
    )

    interactions = interactions.sort_values(
        ["user_id", "starred_at", "repo_id"]
    ).reset_index(drop=True)

    user_counts = interactions.groupby(
        "user_id"
    ).size()

    eligible_users = set(
        user_counts[
            user_counts
            >= settings[
                "minimum_interactions_for_evaluation"
            ]
        ].index
    )

    eligible = interactions[
        interactions["user_id"].isin(eligible_users)
    ].copy()

    ineligible = interactions[
        ~interactions["user_id"].isin(eligible_users)
    ].copy()

    eligible["reverse_order"] = (
        eligible.groupby("user_id")
        .cumcount(ascending=False)
    )

    test = eligible[
        eligible["reverse_order"] == 0
    ].copy()

    validation = eligible[
        eligible["reverse_order"] == 1
    ].copy()

    eligible_train = eligible[
        eligible["reverse_order"] >= 2
    ].copy()

    train = pd.concat(
        [eligible_train, ineligible],
        ignore_index=True,
    )

    for frame in [train, validation, test]:
        frame.drop(
            columns=["reverse_order"],
            errors="ignore",
            inplace=True,
        )

    train = train.sort_values(
        ["user_id", "starred_at", "repo_id"]
    ).reset_index(drop=True)

    validation = validation.sort_values(
        ["user_id", "starred_at", "repo_id"]
    ).reset_index(drop=True)

    test = test.sort_values(
        ["user_id", "starred_at", "repo_id"]
    ).reset_index(drop=True)

    for split_name, frame in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        path = (
            RECOMMENDATION_DIR
            / f"recommendation_{split_name}.csv"
        )

        save_dataframe(frame, path, manifest)

    train_repositories = set(train["repo_id"])

    return {
        "eligible_evaluation_users": len(
            eligible_users
        ),
        "train": {
            "rows": len(train),
            "users": int(train["user_id"].nunique()),
            "repositories": int(
                train["repo_id"].nunique()
            ),
        },
        "validation": {
            "rows": len(validation),
            "users": int(
                validation["user_id"].nunique()
            ),
            "repositories": int(
                validation["repo_id"].nunique()
            ),
            "cold_start_repositories": len(
                set(validation["repo_id"])
                - train_repositories
            ),
        },
        "test": {
            "rows": len(test),
            "users": int(test["user_id"].nunique()),
            "repositories": int(
                test["repo_id"].nunique()
            ),
            "cold_start_repositories": len(
                set(test["repo_id"])
                - train_repositories
            ),
        },
    }


def main() -> None:
    config = load_config()
    manifest = {}

    forecasting_summary = build_forecasting_data(
        config=config,
        manifest=manifest,
    )

    classification_summary = (
        build_classification_data(
            config=config,
            manifest=manifest,
        )
    )

    recommendation_summary = (
        build_recommendation_data(
            config=config,
            manifest=manifest,
        )
    )

    final_manifest = {
        "project": config["project_name"],
        "random_state": config["random_state"],
        "classification": classification_summary,
        "forecasting": forecasting_summary,
        "recommendation": recommendation_summary,
        "generated_files": manifest,
    }

    manifest_path = (
        MODELING_DIR / "phase3_dataset_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            final_manifest,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            final_manifest,
            indent=2,
            default=str,
        )
    )

    print(
        "\nPhase 3 modeling datasets created successfully."
    )

    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
