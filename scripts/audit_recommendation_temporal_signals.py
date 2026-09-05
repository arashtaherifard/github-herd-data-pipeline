import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "recommendation"
    / "recommendation_train.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "recommendation"
    / "recommendation_validation.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "recommendation"
    / "recommendation_test.csv"
)

WEEKLY_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weekly_timeseries_features.csv"
)

FORECAST_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
    / "selected_forecaster_pretest.joblib"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "database"
    / "github_herd.db"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "recommendation"
    / "temporal_signal_audit"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "recommendation_temporal_signal_audit.json"
)

VIOLATIONS_PATH = (
    OUTPUT_DIR
    / "recommendation_temporal_violations.csv"
)

FORECAST_REPOSITORY_PATH = (
    OUTPUT_DIR
    / "recommendation_forecast_repository_availability.csv"
)

FORECAST_USER_PATH = (
    OUTPUT_DIR
    / "recommendation_forecast_validation_coverage.csv"
)

TOPIC_PATH = (
    OUTPUT_DIR
    / "recommendation_topic_profile.csv"
)

INTERNAL_TUNING_PATH = (
    OUTPUT_DIR
    / "recommendation_internal_tuning_eligibility.csv"
)


def write_json(
    path: Path,
    value: dict[str, Any],
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


def quantile_summary(
    values: pd.Series,
) -> dict[str, float]:
    numeric = pd.to_numeric(
        values,
        errors="raise",
    )

    return {
        "minimum": float(numeric.min()),
        "q25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "mean": float(numeric.mean()),
        "q75": float(numeric.quantile(0.75)),
        "q90": float(numeric.quantile(0.90)),
        "q95": float(numeric.quantile(0.95)),
        "q99": float(numeric.quantile(0.99)),
        "maximum": float(numeric.max()),
    }


def load_interactions(
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(path)

    required = {
        "user_id",
        "username",
        "repo_id",
        "repo_full_name",
        "starred_at",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: "
            f"{sorted(missing)}"
        )

    frame["starred_at"] = pd.to_datetime(
        frame["starred_at"],
        utc=True,
        errors="raise",
    )

    frame["user_id"] = pd.to_numeric(
        frame["user_id"],
        errors="raise",
    ).astype("int64")

    frame["repo_id"] = pd.to_numeric(
        frame["repo_id"],
        errors="raise",
    ).astype("int64")

    return frame


def diagnose_validation_order(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validation_users = set(
        validation["user_id"].tolist()
    )

    relevant_train = train[
        train["user_id"].isin(
            validation_users
        )
    ].copy()

    latest_time = (
        relevant_train.groupby(
            "user_id"
        )["starred_at"]
        .max()
        .rename("latest_train_time")
    )

    validation_rows = (
        validation[
            [
                "user_id",
                "username",
                "repo_id",
                "repo_full_name",
                "starred_at",
            ]
        ]
        .rename(
            columns={
                "repo_id": (
                    "validation_repo_id"
                ),
                "repo_full_name": (
                    "validation_repo_full_name"
                ),
                "starred_at": (
                    "validation_time"
                ),
            }
        )
        .set_index("user_id")
    )

    comparison = validation_rows.join(
        latest_time,
        how="left",
    )

    comparison[
        "delta_seconds"
    ] = (
        comparison["validation_time"]
        - comparison["latest_train_time"]
    ).dt.total_seconds()

    comparison[
        "order_status"
    ] = np.select(
        [
            comparison[
                "latest_train_time"
            ]
            < comparison[
                "validation_time"
            ],
            comparison[
                "latest_train_time"
            ]
            == comparison[
                "validation_time"
            ],
            comparison[
                "latest_train_time"
            ]
            > comparison[
                "validation_time"
            ],
        ],
        [
            "strictly_later",
            "equal_timestamp",
            "train_after_validation",
        ],
        default="missing",
    )

    problematic = comparison[
        comparison[
            "order_status"
        ]
        != "strictly_later"
    ].copy()

    latest_rows = relevant_train.merge(
        latest_time.reset_index(),
        left_on=[
            "user_id",
            "starred_at",
        ],
        right_on=[
            "user_id",
            "latest_train_time",
        ],
        how="inner",
    )

    latest_repo_lists = (
        latest_rows.groupby(
            "user_id"
        )
        .agg(
            latest_train_repo_ids=(
                "repo_id",
                lambda values: "|".join(
                    str(int(value))
                    for value in sorted(
                        set(values)
                    )
                ),
            ),
            latest_train_repo_names=(
                "repo_full_name",
                lambda values: "|".join(
                    sorted(
                        set(
                            str(value)
                            for value in values
                        )
                    )
                ),
            ),
            latest_train_rows=(
                "repo_id",
                "size",
            ),
        )
    )

    problematic = (
        problematic.join(
            latest_repo_lists,
            how="left",
        )
        .reset_index()
        .sort_values(
            [
                "order_status",
                "user_id",
            ]
        )
        .reset_index(drop=True)
    )

    status_counts = (
        comparison[
            "order_status"
        ]
        .value_counts()
        .to_dict()
    )

    summary = {
        "validation_users": int(
            len(comparison)
        ),
        "strictly_later": int(
            status_counts.get(
                "strictly_later",
                0,
            )
        ),
        "equal_timestamp": int(
            status_counts.get(
                "equal_timestamp",
                0,
            )
        ),
        "train_after_validation": int(
            status_counts.get(
                "train_after_validation",
                0,
            )
        ),
        "missing": int(
            status_counts.get(
                "missing",
                0,
            )
        ),
        "problematic_rows": int(
            len(problematic)
        ),
    }

    return problematic, summary


def internal_tuning_eligibility(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    grouped = train.groupby(
        "user_id",
        sort=True,
    )

    history_count = grouped.size().rename(
        "train_history_count"
    )

    latest_time = grouped[
        "starred_at"
    ].max().rename(
        "latest_train_time"
    )

    profile = pd.concat(
        [
            history_count,
            latest_time,
        ],
        axis=1,
    ).reset_index()

    latest_rows = train.merge(
        latest_time.reset_index(),
        left_on=[
            "user_id",
            "starred_at",
        ],
        right_on=[
            "user_id",
            "latest_train_time",
        ],
        how="inner",
    )

    latest_multiplicity = (
        latest_rows.groupby(
            "user_id"
        )
        .size()
        .rename(
            "latest_timestamp_rows"
        )
    )

    profile = profile.merge(
        latest_multiplicity.reset_index(),
        on="user_id",
        how="left",
        validate="one_to_one",
    )

    profile[
        "latest_timestamp_rows"
    ] = profile[
        "latest_timestamp_rows"
    ].fillna(0).astype(int)

    profile[
        "eligible_unique_latest"
    ] = (
        (
            profile[
                "train_history_count"
            ]
            >= 2
        )
        & (
            profile[
                "latest_timestamp_rows"
            ]
            == 1
        )
    )

    eligible = profile[
        profile[
            "train_history_count"
        ]
        >= 2
    ].copy()

    summary = {
        "all_train_users": int(
            len(profile)
        ),
        "users_with_at_least_two_train_interactions": int(
            len(eligible)
        ),
        "users_with_unique_latest_timestamp": int(
            eligible[
                "eligible_unique_latest"
            ].sum()
        ),
        "users_with_tied_latest_timestamp": int(
            (
                ~eligible[
                    "eligible_unique_latest"
                ]
            ).sum()
        ),
        "eligible_history_count": (
            quantile_summary(
                eligible.loc[
                    eligible[
                        "eligible_unique_latest"
                    ],
                    "train_history_count",
                ]
            )
        ),
    }

    return profile, summary


def parse_topics(
    raw: Any,
) -> list[str]:
    if pd.isna(raw):
        return []

    text = str(raw).strip()

    if not text:
        return []

    return sorted(
        {
            token.strip().lower()
            for token in text.split("|")
            if token.strip()
        }
    )


def content_metadata_audit(
    catalog_ids: list[int],
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    placeholders = ",".join(
        "?"
        for _ in catalog_ids
    )

    repositories = pd.read_sql_query(
        f"""
        SELECT
            repo_id,
            full_name,
            description,
            language,
            topics,
            collected_at
        FROM repositories
        WHERE repo_id IN ({placeholders})
        """,
        connection,
        params=catalog_ids,
    )

    connection.close()

    repositories[
        "repo_id"
    ] = pd.to_numeric(
        repositories["repo_id"],
        errors="raise",
    ).astype("int64")

    repositories = (
        repositories.sort_values(
            [
                "repo_id",
                "collected_at",
            ]
        )
        .drop_duplicates(
            "repo_id",
            keep="last",
        )
        .sort_values(
            "repo_id"
        )
        .reset_index(drop=True)
    )

    repositories[
        "topic_list"
    ] = repositories[
        "topics"
    ].apply(
        parse_topics
    )

    repositories[
        "topic_identity_count"
    ] = repositories[
        "topic_list"
    ].map(len)

    repositories[
        "description_length"
    ] = (
        repositories[
            "description"
        ]
        .fillna("")
        .astype(str)
        .str.len()
    )

    topic_counts: dict[str, int] = {}

    for topic_list in repositories[
        "topic_list"
    ]:
        for topic in topic_list:
            topic_counts[topic] = (
                topic_counts.get(
                    topic,
                    0,
                )
                + 1
            )

    topic_profile = (
        pd.DataFrame(
            [
                {
                    "topic": topic,
                    "repository_count": count,
                    "catalog_share": (
                        count
                        / len(repositories)
                    ),
                }
                for topic, count in (
                    topic_counts.items()
                )
            ]
        )
        .sort_values(
            [
                "repository_count",
                "topic",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    collected_at = pd.to_datetime(
        repositories[
            "collected_at"
        ],
        utc=True,
        errors="coerce",
    )

    nonempty_description = (
        repositories[
            "description"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    summary = {
        "catalog_repositories": int(
            len(catalog_ids)
        ),
        "matched_repositories": int(
            repositories[
                "repo_id"
            ].nunique()
        ),
        "repositories_with_description": int(
            nonempty_description.sum()
        ),
        "repositories_with_topic_identities": int(
            (
                repositories[
                    "topic_identity_count"
                ]
                > 0
            ).sum()
        ),
        "unique_topics": int(
            len(topic_profile)
        ),
        "topic_count_per_repository": (
            quantile_summary(
                repositories[
                    "topic_identity_count"
                ]
            )
        ),
        "description_length": (
            quantile_summary(
                repositories[
                    "description_length"
                ]
            )
        ),
        "metadata_collected_at_minimum": (
            collected_at.min().isoformat()
        ),
        "metadata_collected_at_maximum": (
            collected_at.max().isoformat()
        ),
        "temporal_status": (
            "static_snapshot_not_strictly_"
            "time_safe_for_historical_"
            "validation"
        ),
    }

    return topic_profile, summary


def infer_week_column(
    frame: pd.DataFrame,
) -> str:
    for candidate in [
        "week",
        "date",
        "cutoff_week",
    ]:
        if candidate in frame.columns:
            return candidate

    raise ValueError(
        "Could not identify the weekly "
        "timestamp column."
    )


def parse_week_availability(
    values: pd.Series,
) -> pd.Series:
    text = (
        values.astype(str)
        .str.strip()
    )

    period_mask = text.str.contains(
        "/",
        regex=False,
    )

    interval_end = text.where(
        ~period_mask,
        text.str.rsplit(
            "/",
            n=1,
        ).str[-1],
    )

    parsed = pd.to_datetime(
        interval_end,
        format="mixed",
        utc=True,
        errors="raise",
    )

    parsed.loc[
        period_mask
    ] = (
        parsed.loc[
            period_mask
        ]
        + pd.Timedelta(days=1)
    )

    return parsed


def forecast_signal_audit(
    validation: pd.DataFrame,
    catalog_ids: list[int],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    artifact = joblib.load(
        FORECAST_MODEL_PATH
    )

    feature_columns = list(
        artifact[
            "feature_columns"
        ]
    )

    weekly = pd.read_csv(
        WEEKLY_FEATURES_PATH
    )

    week_column = infer_week_column(
        weekly
    )

    required = {
        "repo_id",
        week_column,
        *feature_columns,
    }

    missing = required - set(
        weekly.columns
    )

    if missing:
        raise ValueError(
            "Weekly forecasting features "
            f"are missing: {sorted(missing)}"
        )

    weekly[
        "repo_id"
    ] = pd.to_numeric(
        weekly["repo_id"],
        errors="raise",
    ).astype("int64")

    availability_column = (
        "week_available_at"
    )

    weekly[
        availability_column
    ] = parse_week_availability(
        weekly[
            week_column
        ]
    )

    weekly[
        feature_columns
    ] = weekly[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    weekly = weekly[
        weekly[
            "repo_id"
        ].isin(
            catalog_ids
        )
    ].copy()

    complete_mask = (
        weekly[
            feature_columns
        ]
        .notna()
        .all(axis=1)
        & np.isfinite(
            weekly[
                feature_columns
            ].to_numpy(
                dtype=float
            )
        ).all(axis=1)
    )

    complete = weekly[
        complete_mask
    ].copy()

    availability = (
        complete.groupby(
            "repo_id"
        )
        .agg(
            first_complete_week=(
                availability_column,
                "min",
            ),
            latest_complete_week=(
                availability_column,
                "max",
            ),
            complete_feature_rows=(
                availability_column,
                "size",
            ),
        )
        .reset_index()
    )

    catalog_availability = (
        pd.DataFrame(
            {
                "repo_id": catalog_ids,
            }
        )
        .merge(
            availability,
            on="repo_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            "repo_id"
        )
        .reset_index(drop=True)
    )

    first_week_map = (
        catalog_availability.set_index(
            "repo_id"
        )[
            "first_complete_week"
        ]
        .to_dict()
    )

    first_complete_weeks = (
        catalog_availability[
            "first_complete_week"
        ]
        .dropna()
        .sort_values()
        .reset_index(
            drop=True
        )
    )

    coverage_rows = []

    for row in validation[
        [
            "user_id",
            "repo_id",
            "starred_at",
        ]
    ].itertuples(
        index=False
    ):
        cutoff = pd.Timestamp(
            row.starred_at
        )

        available_count = int(
            (
                first_complete_weeks
                <= cutoff
            ).sum()
        )

        positive_first_week = (
            first_week_map.get(
                int(row.repo_id)
            )
        )

        positive_available = bool(
            pd.notna(
                positive_first_week
            )
            and (
                positive_first_week
                <= cutoff
            )
        )

        coverage_rows.append(
            {
                "user_id": int(
                    row.user_id
                ),
                "validation_repo_id": int(
                    row.repo_id
                ),
                "validation_time": cutoff,
                "forecast_available_repositories": (
                    available_count
                ),
                "forecast_catalog_coverage": (
                    available_count
                    / len(catalog_ids)
                ),
                "validation_positive_has_forecast_history": (
                    positive_available
                ),
            }
        )

    coverage = pd.DataFrame(
        coverage_rows
    )

    current_eligible = (
        catalog_availability[
            "complete_feature_rows"
        ]
        .notna()
    )

    summary = {
        "frozen_forecast_artifact_version": (
            artifact[
                "artifact_version"
            ]
        ),
        "feature_columns": (
            feature_columns
        ),
        "weekly_period_availability_rule": (
            "For an interval such as "
            "YYYY-MM-DD/YYYY-MM-DD, the "
            "weekly feature row becomes "
            "available at 00:00 UTC on the "
            "day after the interval ends."
        ),
        "catalog_repositories": int(
            len(catalog_ids)
        ),
        "repositories_with_any_complete_forecast_history": int(
            current_eligible.sum()
        ),
        "repositories_without_complete_forecast_history": int(
            (
                ~current_eligible
            ).sum()
        ),
        "complete_feature_rows": (
            quantile_summary(
                catalog_availability.loc[
                    current_eligible,
                    "complete_feature_rows",
                ]
            )
        ),
        "validation_forecast_catalog_coverage": (
            quantile_summary(
                coverage[
                    "forecast_catalog_coverage"
                ]
            )
        ),
        "validation_available_repository_count": (
            quantile_summary(
                coverage[
                    "forecast_available_repositories"
                ]
            )
        ),
        "validation_positive_with_forecast_history": int(
            coverage[
                "validation_positive_has_forecast_history"
            ].sum()
        ),
        "validation_positive_forecast_coverage": float(
            coverage[
                "validation_positive_has_forecast_history"
            ].mean()
        ),
        "recommended_missing_signal_policy": (
            "Use a neutral rank score for "
            "repositories with no forecast "
            "history at the user's cutoff; "
            "do not impute using future rows."
        ),
    }

    return (
        catalog_availability,
        coverage,
        summary,
    )


def main() -> None:
    if OUTPUT_DIR.exists():
        raise RuntimeError(
            "Temporal-signal audit outputs "
            "already exist. Refusing to "
            "overwrite:\n"
            f"{OUTPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    train = load_interactions(
        TRAIN_PATH
    )

    validation = load_interactions(
        VALIDATION_PATH
    )

    test_header = pd.read_csv(
        TEST_PATH,
        nrows=0,
    )

    required_test_columns = {
        "user_id",
        "repo_id",
        "starred_at",
    }

    if (
        required_test_columns
        - set(test_header.columns)
    ):
        raise ValueError(
            "Recommendation test header "
            "does not match expectations."
        )

    catalog_ids = sorted(
        set(
            train[
                "repo_id"
            ].tolist()
        )
    )

    (
        temporal_violations,
        temporal_summary,
    ) = diagnose_validation_order(
        train=train,
        validation=validation,
    )

    (
        tuning_profile,
        tuning_summary,
    ) = internal_tuning_eligibility(
        train
    )

    (
        topic_profile,
        content_summary,
    ) = content_metadata_audit(
        catalog_ids
    )

    (
        forecast_repository,
        forecast_user,
        forecast_summary,
    ) = forecast_signal_audit(
        validation=validation,
        catalog_ids=catalog_ids,
    )

    temporal_violations.to_csv(
        VIOLATIONS_PATH,
        index=False,
    )

    tuning_profile.to_csv(
        INTERNAL_TUNING_PATH,
        index=False,
    )

    topic_profile.to_csv(
        TOPIC_PATH,
        index=False,
    )

    forecast_repository.to_csv(
        FORECAST_REPOSITORY_PATH,
        index=False,
    )

    forecast_user.to_csv(
        FORECAST_USER_PATH,
        index=False,
    )

    summary = {
        "audit_status": (
            "recommendation_temporal_"
            "signal_audit_complete"
        ),
        "test_data_status": (
            "header_only_not_loaded_or_"
            "evaluated"
        ),
        "validation_temporal_order": (
            temporal_summary
        ),
        "internal_tuning": (
            tuning_summary
        ),
        "static_content_metadata": (
            content_summary
        ),
        "forecast_signal": (
            forecast_summary
        ),
        "architecture_implications": {
            "primary_time_safe_pool": [
                "training-popularity baseline",
                "item-item cosine collaborative filtering",
                "TruncatedSVD latent-factor recommender",
                "co-occurrence graph recommender",
                "rank-normalized collaborative-graph hybrid",
            ],
            "static_content_pool": [
                "topic-description-language "
                "TF-IDF recommender",
                "static-content hybrid",
            ],
            "forecast_aware_pool": [
                "cutoff-aligned frozen-forecast "
                "reranker with neutral missing "
                "signal handling",
            ],
            "selection_policy": (
                "Select the primary frozen "
                "recommender from the time-safe "
                "pool. Report static-content "
                "models separately because the "
                "metadata was collected in June "
                "2026. Forecast-aware models are "
                "eligible only when their score "
                "uses the latest weekly features "
                "at or before each user's cutoff."
            ),
        },
        "output_paths": {
            "summary": str(
                SUMMARY_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "temporal_violations": str(
                VIOLATIONS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "forecast_repository_availability": str(
                FORECAST_REPOSITORY_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "forecast_validation_coverage": str(
                FORECAST_USER_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "topic_profile": str(
                TOPIC_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "internal_tuning_eligibility": str(
                INTERNAL_TUNING_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
    }

    write_json(
        SUMMARY_PATH,
        summary,
    )

    print("=" * 108)
    print(
        "RECOMMENDATION TEMPORAL AND "
        "SIGNAL AUDIT"
    )
    print("=" * 108)

    print(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
    )

    print()
    print("=" * 108)
    print(
        "NON-STRICT VALIDATION ORDER ROWS"
    )
    print("=" * 108)

    if temporal_violations.empty:
        print("None")
    else:
        print(
            temporal_violations.to_string(
                index=False
            )
        )

    print()
    print("=" * 108)
    print("TOP CONTENT TOPICS")
    print("=" * 108)

    print(
        topic_profile.head(
            30
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "Recommendation test data was "
        "not loaded; only its header was "
        "validated."
    )


if __name__ == "__main__":
    main()
