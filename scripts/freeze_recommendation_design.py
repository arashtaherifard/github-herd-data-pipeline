import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "phase3_config.json"
)

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

DESIGN_AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "recommendation"
    / "design_audit"
    / "recommendation_design_audit.json"
)

TEMPORAL_AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "recommendation"
    / "temporal_signal_audit"
    / "recommendation_temporal_signal_audit.json"
)

SCHEMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "recommendation"
    / "recommendation_schema.json"
)

INTERNAL_ASSIGNMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "recommendation"
    / "recommendation_internal_tuning_assignments.csv"
)

PRIMARY_VALIDATION_USERS_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "recommendation"
    / "recommendation_primary_validation_users.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "recommendation"
    / "design_freeze"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "recommendation_design_freeze_metadata.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "recommendation_design_freeze_manifest.json"
)

EXPECTED_COLUMNS = [
    "user_id",
    "username",
    "repo_id",
    "repo_full_name",
    "starred_at",
    "interaction",
    "user_activity_count",
    "repo_popularity_count",
    "language",
    "topic_count",
    "stars_count",
    "forks_count",
]

SCHEMA_VERSION = "recommendation_design_v1"

FREEZE_NAME = (
    "recommendation-design-freeze-v1"
)

RANDOM_STATE = 42


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


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


def load_interactions(
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(path)

    missing = (
        set(EXPECTED_COLUMNS)
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{path} is missing columns: "
            f"{sorted(missing)}"
        )

    frame = frame[
        EXPECTED_COLUMNS
    ].copy()

    frame["starred_at"] = (
        pd.to_datetime(
            frame["starred_at"],
            utc=True,
            errors="raise",
        )
    )

    frame["user_id"] = pd.to_numeric(
        frame["user_id"],
        errors="raise",
    ).astype("int64")

    frame["repo_id"] = pd.to_numeric(
        frame["repo_id"],
        errors="raise",
    ).astype("int64")

    if frame.isna().any().any():
        raise ValueError(
            f"{path} contains missing values."
        )

    duplicate_pairs = int(
        frame.duplicated(
            subset=[
                "user_id",
                "repo_id",
            ]
        ).sum()
    )

    if duplicate_pairs:
        raise ValueError(
            f"{path} contains "
            f"{duplicate_pairs} duplicate "
            "user-repository pairs."
        )

    return frame


def build_internal_assignments(
    train: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    history_counts = (
        train.groupby(
            "user_id"
        )
        .size()
        .rename(
            "train_history_count"
        )
    )

    eligible_users = (
        history_counts[
            history_counts >= 2
        ]
        .index
    )

    eligible_rows = train[
        train["user_id"].isin(
            eligible_users
        )
    ].copy()

    latest_times = (
        eligible_rows.groupby(
            "user_id"
        )["starred_at"]
        .max()
        .rename(
            "internal_holdout_time"
        )
    )

    latest_rows = (
        eligible_rows.merge(
            latest_times.reset_index(),
            left_on=[
                "user_id",
                "starred_at",
            ],
            right_on=[
                "user_id",
                "internal_holdout_time",
            ],
            how="inner",
        )
    )

    latest_multiplicity = (
        latest_rows.groupby(
            "user_id"
        )
        .size()
    )

    unique_latest_users = (
        latest_multiplicity[
            latest_multiplicity == 1
        ]
        .index
    )

    assignments = (
        latest_rows[
            latest_rows[
                "user_id"
            ].isin(
                unique_latest_users
            )
        ][
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
                    "internal_holdout_repo_id"
                ),
                "repo_full_name": (
                    "internal_holdout_repo_full_name"
                ),
                "starred_at": (
                    "internal_holdout_time"
                ),
            }
        )
        .merge(
            history_counts.reset_index(),
            on="user_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            "user_id"
        )
        .reset_index(drop=True)
    )

    assignment_keys = assignments[
        [
            "user_id",
            "internal_holdout_repo_id",
        ]
    ].rename(
        columns={
            "internal_holdout_repo_id": (
                "repo_id"
            )
        }
    )

    internal_train = (
        train.merge(
            assignment_keys.assign(
                _internal_holdout=True
            ),
            on=[
                "user_id",
                "repo_id",
            ],
            how="left",
        )
    )

    internal_train = internal_train[
        internal_train[
            "_internal_holdout"
        ].isna()
    ].drop(
        columns=[
            "_internal_holdout",
        ]
    )

    internal_latest = (
        internal_train[
            internal_train[
                "user_id"
            ].isin(
                assignments[
                    "user_id"
                ]
            )
        ]
        .groupby(
            "user_id"
        )["starred_at"]
        .max()
    )

    assignment_times = (
        assignments.set_index(
            "user_id"
        )[
            "internal_holdout_time"
        ]
    )

    temporal_violations = int(
        (
            internal_latest
            >= assignment_times
        ).sum()
    )

    overlap = (
        internal_train[
            [
                "user_id",
                "repo_id",
            ]
        ]
        .merge(
            assignment_keys,
            on=[
                "user_id",
                "repo_id",
            ],
            how="inner",
        )
    )

    original_catalog = set(
        train["repo_id"].tolist()
    )

    internal_catalog = set(
        internal_train[
            "repo_id"
        ].tolist()
    )

    summary = {
        "users_with_at_least_two_training_interactions": int(
            len(eligible_users)
        ),
        "users_with_unique_latest_timestamp": int(
            len(assignments)
        ),
        "users_excluded_for_latest_timestamp_tie": int(
            len(eligible_users)
            - len(assignments)
        ),
        "internal_training_rows": int(
            len(internal_train)
        ),
        "internal_holdout_rows": int(
            len(assignments)
        ),
        "minimum_remaining_history": int(
            (
                assignments[
                    "train_history_count"
                ]
                - 1
            ).min()
        ),
        "maximum_remaining_history": int(
            (
                assignments[
                    "train_history_count"
                ]
                - 1
            ).max()
        ),
        "train_holdout_pair_overlap": int(
            len(overlap)
        ),
        "temporal_order_violations": (
            temporal_violations
        ),
        "catalog_repositories_before_holdout": int(
            len(original_catalog)
        ),
        "catalog_repositories_after_holdout": int(
            len(internal_catalog)
        ),
        "cold_repositories_created": int(
            len(
                original_catalog
                - internal_catalog
            )
        ),
    }

    if (
        summary[
            "train_holdout_pair_overlap"
        ]
        != 0
    ):
        raise RuntimeError(
            "Internal tuning holdout still "
            "overlaps internal training."
        )

    if (
        summary[
            "temporal_order_violations"
        ]
        != 0
    ):
        raise RuntimeError(
            "Internal tuning split contains "
            "temporal-order violations."
        )

    if (
        summary[
            "cold_repositories_created"
        ]
        != 0
    ):
        raise RuntimeError(
            "Internal tuning holdout created "
            "cold repositories."
        )

    return assignments, summary


def build_primary_validation_users(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    latest_train = (
        train[
            train[
                "user_id"
            ].isin(
                validation[
                    "user_id"
                ]
            )
        ]
        .groupby(
            "user_id"
        )["starred_at"]
        .max()
        .rename(
            "latest_train_time"
        )
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
        .set_index(
            "user_id"
        )
        .join(
            latest_train,
            how="left",
        )
        .reset_index()
    )

    validation_rows[
        "strict_temporal_order"
    ] = (
        validation_rows[
            "latest_train_time"
        ]
        < validation_rows[
            "validation_time"
        ]
    )

    primary = (
        validation_rows[
            validation_rows[
                "strict_temporal_order"
            ]
        ]
        .sort_values(
            "user_id"
        )
        .reset_index(drop=True)
    )

    summary = {
        "all_validation_users": int(
            len(validation_rows)
        ),
        "primary_strict_validation_users": int(
            len(primary)
        ),
        "excluded_equal_timestamp_users": int(
            (
                validation_rows[
                    "latest_train_time"
                ]
                == validation_rows[
                    "validation_time"
                ]
            ).sum()
        ),
        "excluded_train_after_validation_users": int(
            (
                validation_rows[
                    "latest_train_time"
                ]
                > validation_rows[
                    "validation_time"
                ]
            ).sum()
        ),
        "missing_train_history_users": int(
            validation_rows[
                "latest_train_time"
            ].isna().sum()
        ),
    }

    if (
        summary[
            "excluded_train_after_validation_users"
        ]
        != 0
    ):
        raise RuntimeError(
            "External validation contains "
            "train-after-validation cases."
        )

    if (
        summary[
            "missing_train_history_users"
        ]
        != 0
    ):
        raise RuntimeError(
            "External validation contains "
            "users without training history."
        )

    return primary, summary


def main() -> None:
    if OUTPUT_DIR.exists():
        raise RuntimeError(
            "Recommendation design-freeze "
            "directory already exists. "
            "Refusing to overwrite:\n"
            f"{OUTPUT_DIR}"
        )

    if SCHEMA_PATH.exists():
        raise RuntimeError(
            "Recommendation schema already "
            "exists. Refusing to overwrite:\n"
            f"{SCHEMA_PATH}"
        )

    if (
        INTERNAL_ASSIGNMENTS_PATH.exists()
        or PRIMARY_VALIDATION_USERS_PATH.exists()
    ):
        raise RuntimeError(
            "Recommendation assignment files "
            "already exist. Refusing to "
            "overwrite them."
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

    missing_test_columns = (
        set(EXPECTED_COLUMNS)
        - set(test_header.columns)
    )

    if missing_test_columns:
        raise ValueError(
            "Recommendation test header is "
            f"missing: "
            f"{sorted(missing_test_columns)}"
        )

    design_audit = load_json(
        DESIGN_AUDIT_PATH
    )

    temporal_audit = load_json(
        TEMPORAL_AUDIT_PATH
    )

    if (
        design_audit[
            "test_data_status"
        ]
        != (
            "header_only_not_loaded_or_"
            "evaluated"
        )
    ):
        raise RuntimeError(
            "Design audit does not preserve "
            "the sealed test status."
        )

    if (
        temporal_audit[
            "test_data_status"
        ]
        != (
            "header_only_not_loaded_or_"
            "evaluated"
        )
    ):
        raise RuntimeError(
            "Temporal audit does not preserve "
            "the sealed test status."
        )

    (
        internal_assignments,
        internal_summary,
    ) = build_internal_assignments(
        train
    )

    (
        primary_validation_users,
        external_summary,
    ) = build_primary_validation_users(
        train=train,
        validation=validation,
    )

    internal_assignments.to_csv(
        INTERNAL_ASSIGNMENTS_PATH,
        index=False,
    )

    primary_validation_users.to_csv(
        PRIMARY_VALIDATION_USERS_PATH,
        index=False,
    )

    catalog_size = int(
        train[
            "repo_id"
        ].nunique()
    )

    if catalog_size != 145:
        raise RuntimeError(
            "Unexpected recommendation "
            f"catalog size: {catalog_size}"
        )

    recommendation_design = {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "git_branch": git_output(
            [
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            ]
        ),
        "git_commit_before_freeze": (
            git_output(
                [
                    "rev-parse",
                    "HEAD",
                ]
            )
        ),
        "random_state": RANDOM_STATE,
        "feedback_type": "implicit",
        "catalog": {
            "source": (
                "training catalog"
            ),
            "repositories": (
                catalog_size
            ),
            "candidate_strategy": (
                "exact_full_catalog"
            ),
            "validation_seen_item_exclusion": [
                "training history",
            ],
            "final_test_seen_item_exclusion": [
                "training history",
                "validation history",
            ],
            "score_tie_breaker": (
                "repo_id ascending"
            ),
        },
        "internal_tuning": {
            "strategy": (
                "leave_latest_training_"
                "interaction_out"
            ),
            "eligibility": (
                "at least two training "
                "interactions and a unique "
                "latest timestamp"
            ),
            "assignment_path": str(
                INTERNAL_ASSIGNMENTS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            **internal_summary,
        },
        "external_validation": {
            "primary_population": (
                "strictly later held-out "
                "validation timestamp"
            ),
            "primary_user_path": str(
                PRIMARY_VALIDATION_USERS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "full_population_role": (
                "sensitivity analysis only"
            ),
            **external_summary,
        },
        "ranking_metrics": {
            "primary": "ndcg_at_10",
            "selection_tie_breakers": [
                "mrr",
                "recall_at_10",
                "ndcg_at_5",
            ],
            "accuracy_metrics": [
                "precision_at_5",
                "precision_at_10",
                "recall_at_5",
                "recall_at_10",
                "hit_rate_at_5",
                "hit_rate_at_10",
                "ndcg_at_5",
                "ndcg_at_10",
                "mrr",
                "map_at_10",
            ],
            "beyond_accuracy_metrics": [
                "catalog_coverage_at_10",
                "novelty_at_10",
                "intra_list_diversity_at_10",
                "average_recommended_popularity_at_10",
                "long_tail_share_at_10",
            ],
            "one_positive_user_equivalences": {
                "recall_at_k": (
                    "equals hit_rate_at_k"
                ),
                "map_at_10": (
                    "equals reciprocal rank "
                    "when the positive rank "
                    "is at most 10"
                ),
            },
        },
        "primary_time_safe_model_pool": {
            "eligible_for_primary_selection": (
                True
            ),
            "families": {
                "random_sanity": {
                    "seeds": [
                        11,
                        29,
                        42,
                        71,
                        101,
                    ],
                },
                "training_popularity": {
                    "score": (
                        "internal-training "
                        "interaction count"
                    ),
                },
                "item_item_cosine": {
                    "aggregation": [
                        "sum",
                        "mean",
                    ],
                    "shrinkage": [
                        0.0,
                        10.0,
                        50.0,
                        100.0,
                    ],
                },
                "truncated_svd": {
                    "n_components": [
                        16,
                        32,
                        64,
                        96,
                    ],
                    "n_iter": [
                        7,
                        15,
                    ],
                    "regularization": (
                        "component-count "
                        "restriction"
                    ),
                },
                "graph_personalized_pagerank": {
                    "edge_weights": [
                        "cosine",
                        "ppmi",
                    ],
                    "restart_probability": [
                        0.15,
                        0.30,
                        0.50,
                    ],
                },
                "rank_normalized_hybrid": {
                    "components": [
                        "item_item_cosine",
                        "truncated_svd",
                        "graph_personalized_pagerank",
                        "training_popularity",
                    ],
                    "weight_grid": (
                        "convex combinations "
                        "in 0.25 increments"
                    ),
                },
                "herd_mitigation_reranker": {
                    "base": (
                        "best time-safe hybrid"
                    ),
                    "popularity_penalty": [
                        0.0,
                        0.05,
                        0.10,
                        0.20,
                        0.30,
                    ],
                    "purpose": (
                        "measure accuracy versus "
                        "popularity-bias tradeoff"
                    ),
                },
            },
        },
        "secondary_static_content_pool": {
            "eligible_for_primary_selection": (
                False
            ),
            "reason": (
                "Descriptions, topics, and "
                "languages were collected in "
                "June 2026 and are not strictly "
                "time-safe for historical "
                "validation."
            ),
            "families": {
                "tfidf_content": {
                    "fields": [
                        "topics",
                        "description",
                        "language",
                    ],
                    "ngram_range": [
                        1,
                        2,
                    ],
                    "topic_repeat": [
                        1,
                        2,
                        3,
                    ],
                    "language_repeat": [
                        1,
                        2,
                    ],
                },
                "static_content_hybrid": {
                    "base": (
                        "best time-safe hybrid"
                    ),
                    "content_weight": [
                        0.10,
                        0.20,
                        0.30,
                    ],
                },
            },
        },
        "secondary_forecast_aware_pool": {
            "eligible_for_primary_selection": (
                False
            ),
            "reason": (
                "Forecast-history coverage "
                "varies by validation cutoff; "
                "this branch is reported as a "
                "secondary herd-aware analysis."
            ),
            "frozen_forecast_artifact": (
                "models/forecasting/"
                "selected_forecaster_pretest."
                "joblib"
            ),
            "availability_rule": (
                "use the latest complete weekly "
                "feature row available no later "
                "than the user's cutoff"
            ),
            "missing_signal_policy": (
                "neutral rank-normalized score"
            ),
            "forecast_weight": [
                0.05,
                0.10,
                0.20,
                0.30,
            ],
            "validation_mean_catalog_coverage": (
                temporal_audit[
                    "forecast_signal"
                ][
                    "validation_forecast_"
                    "catalog_coverage"
                ][
                    "mean"
                ]
            ),
            "validation_positive_coverage": (
                temporal_audit[
                    "forecast_signal"
                ][
                    "validation_positive_"
                    "forecast_coverage"
                ]
            ),
        },
        "selection_protocol": {
            "stage_1": (
                "Tune each family using only "
                "the internal temporal holdout."
            ),
            "stage_2": (
                "Evaluate one internally chosen "
                "winner per primary time-safe "
                "family on strict external "
                "validation users."
            ),
            "final_selection": (
                "Select by external validation "
                "NDCG@10, then MRR, Recall@10, "
                "and NDCG@5."
            ),
            "test_rule": (
                "Do not load or evaluate "
                "recommendation test values "
                "until the recommender is "
                "frozen and tagged."
            ),
            "post_test_rule": (
                "No model, weights, features, "
                "or reranking parameter may "
                "change after final test "
                "evaluation."
            ),
        },
        "known_data_limitations": {
            "equal_timestamp_validation_users": (
                external_summary[
                    "excluded_equal_timestamp_users"
                ]
            ),
            "repo_popularity_count": (
                "Excluded because 144 of 145 "
                "repositories are capped at "
                "1500."
            ),
            "topic_count": (
                "Excluded because it is "
                "constant at one."
            ),
            "stars_count_and_forks_count": (
                "Excluded from the primary "
                "time-safe pool because they "
                "are static present-day "
                "snapshots."
            ),
            "content_metadata": (
                "Available for all 145 "
                "repositories but secondary "
                "only."
            ),
        },
        "test_data_status": (
            "header_only_not_loaded_or_"
            "evaluated"
        ),
    }

    write_json(
        SCHEMA_PATH,
        recommendation_design,
    )

    config = load_json(
        CONFIG_PATH
    )

    recommendation_config = dict(
        config.get(
            "recommendation",
            {},
        )
    )

    recommendation_config.update(
        {
            "schema_path": str(
                SCHEMA_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "internal_tuning_assignments_path": str(
                INTERNAL_ASSIGNMENTS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "primary_validation_users_path": str(
                PRIMARY_VALIDATION_USERS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "catalog_size": (
                catalog_size
            ),
            "primary_validation_users": (
                external_summary[
                    "primary_strict_validation_users"
                ]
            ),
            "validation_sensitivity_users": (
                external_summary[
                    "all_validation_users"
                ]
            ),
            "internal_tuning_users": (
                internal_summary[
                    "users_with_unique_latest_timestamp"
                ]
            ),
            "primary_model_pool": (
                "time_safe_interaction_models"
            ),
            "secondary_content_pool": (
                "static_snapshot_exploratory"
            ),
            "secondary_forecast_pool": (
                "cutoff_aligned_exploratory"
            ),
            "test_data_status": (
                "header_only_not_loaded_or_"
                "evaluated"
            ),
        }
    )

    config[
        "recommendation"
    ] = recommendation_config

    write_json(
        CONFIG_PATH,
        config,
    )

    input_hashes = {
        str(
            path.relative_to(
                PROJECT_ROOT
            )
        ): sha256_file(path)
        for path in [
            CONFIG_PATH,
            TRAIN_PATH,
            VALIDATION_PATH,
            TEST_PATH,
            DESIGN_AUDIT_PATH,
            TEMPORAL_AUDIT_PATH,
        ]
    }

    artifact_hashes = {
        str(
            path.relative_to(
                PROJECT_ROOT
            )
        ): sha256_file(path)
        for path in [
            SCHEMA_PATH,
            INTERNAL_ASSIGNMENTS_PATH,
            PRIMARY_VALIDATION_USERS_PATH,
        ]
    }

    metadata = {
        "freeze_status": (
            "recommendation_design_frozen"
        ),
        "freeze_name": FREEZE_NAME,
        "created_at_utc": (
            recommendation_design[
                "created_at_utc"
            ]
        ),
        "git_branch": (
            recommendation_design[
                "git_branch"
            ]
        ),
        "git_commit_before_freeze": (
            recommendation_design[
                "git_commit_before_freeze"
            ]
        ),
        "schema_version": (
            SCHEMA_VERSION
        ),
        "catalog_size": (
            catalog_size
        ),
        "internal_tuning": (
            internal_summary
        ),
        "external_validation": (
            external_summary
        ),
        "primary_metric": (
            "ndcg_at_10"
        ),
        "primary_selection_pool": (
            list(
                recommendation_design[
                    "primary_time_safe_"
                    "model_pool"
                ][
                    "families"
                ].keys()
            )
        ),
        "secondary_static_content_pool": (
            list(
                recommendation_design[
                    "secondary_static_"
                    "content_pool"
                ][
                    "families"
                ].keys()
            )
        ),
        "secondary_forecast_pool": [
            "cutoff_aligned_frozen_"
            "forecast_reranker",
        ],
        "input_hashes": (
            input_hashes
        ),
        "artifact_hashes": (
            artifact_hashes
        ),
        "test_data_status": (
            "header_only_not_loaded_or_"
            "evaluated"
        ),
    }

    write_json(
        METADATA_PATH,
        metadata,
    )

    manifest = {
        "freeze_name": FREEZE_NAME,
        "schema_version": (
            SCHEMA_VERSION
        ),
        "git_commit_before_freeze": (
            recommendation_design[
                "git_commit_before_freeze"
            ]
        ),
        "metadata_path": str(
            METADATA_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "metadata_sha256": (
            sha256_file(
                METADATA_PATH
            )
        ),
        "input_hashes": (
            input_hashes
        ),
        "artifact_hashes": (
            artifact_hashes
        ),
        "test_data_status": (
            "header_only_not_loaded_or_"
            "evaluated"
        ),
    }

    write_json(
        MANIFEST_PATH,
        manifest,
    )

    print("=" * 108)
    print(
        "RECOMMENDATION DESIGN FREEZE"
    )
    print("=" * 108)

    print(
        json.dumps(
            metadata,
            indent=2,
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
        "Recommendation test values were "
        "not loaded or evaluated."
    )


if __name__ == "__main__":
    main()
