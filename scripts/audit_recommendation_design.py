import json
import math
import sqlite3
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

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
    / "design_audit"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "recommendation_design_audit.json"
)

CATALOG_PATH = (
    OUTPUT_DIR
    / "recommendation_catalog_profile.csv"
)

USER_HISTORY_PATH = (
    OUTPUT_DIR
    / "recommendation_user_history_profile.csv"
)

CANDIDATE_SIZE_PATH = (
    OUTPUT_DIR
    / "recommendation_candidate_size_profile.csv"
)

ITEM_POPULARITY_PATH = (
    OUTPUT_DIR
    / "recommendation_item_popularity_profile.csv"
)

COOCCURRENCE_PATH = (
    OUTPUT_DIR
    / "recommendation_top_cooccurrence_edges.csv"
)

DATABASE_SCHEMA_PATH = (
    OUTPUT_DIR
    / "recommendation_database_schema.json"
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

METADATA_COLUMNS = [
    "repo_full_name",
    "language",
    "topic_count",
    "stars_count",
    "forks_count",
    "repo_popularity_count",
]


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


def gini(values: np.ndarray) -> float:
    array = np.asarray(
        values,
        dtype=float,
    )

    if array.size == 0:
        return float("nan")

    if np.any(array < 0):
        raise ValueError(
            "Gini input cannot contain "
            "negative values."
        )

    total = float(
        array.sum()
    )

    if total == 0:
        return 0.0

    sorted_values = np.sort(
        array
    )

    n = sorted_values.size

    index = np.arange(
        1,
        n + 1,
        dtype=float,
    )

    return float(
        (
            2.0
            * np.sum(
                index
                * sorted_values
            )
            / (
                n
                * total
            )
        )
        - (
            n + 1.0
        )
        / n
    )


def quantile_summary(
    values: pd.Series,
) -> dict[str, float]:
    numeric = pd.to_numeric(
        values,
        errors="raise",
    )

    return {
        "minimum": float(
            numeric.min()
        ),
        "q25": float(
            numeric.quantile(0.25)
        ),
        "median": float(
            numeric.median()
        ),
        "mean": float(
            numeric.mean()
        ),
        "q75": float(
            numeric.quantile(0.75)
        ),
        "q90": float(
            numeric.quantile(0.90)
        ),
        "q95": float(
            numeric.quantile(0.95)
        ),
        "maximum": float(
            numeric.max()
        ),
    }


def load_split(
    path: Path,
    split_name: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)

    missing = (
        set(EXPECTED_COLUMNS)
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{split_name} is missing "
            f"columns: {sorted(missing)}"
        )

    prepared = frame[
        EXPECTED_COLUMNS
    ].copy()

    prepared["starred_at"] = (
        pd.to_datetime(
            prepared["starred_at"],
            utc=True,
            errors="raise",
        )
    )

    integer_columns = [
        "user_id",
        "repo_id",
        "interaction",
        "user_activity_count",
        "repo_popularity_count",
        "topic_count",
        "stars_count",
        "forks_count",
    ]

    prepared[
        integer_columns
    ] = prepared[
        integer_columns
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    if prepared.isna().any().any():
        raise ValueError(
            f"{split_name} contains "
            "missing values."
        )

    if not (
        prepared["interaction"]
        == 1
    ).all():
        raise ValueError(
            f"{split_name} contains "
            "non-positive implicit "
            "interactions."
        )

    duplicate_pairs = int(
        prepared.duplicated(
            subset=[
                "user_id",
                "repo_id",
            ]
        ).sum()
    )

    if duplicate_pairs:
        raise ValueError(
            f"{split_name} contains "
            f"{duplicate_pairs} duplicate "
            "user-repository pairs."
        )

    return prepared


def validate_metadata_consistency(
    combined: pd.DataFrame,
) -> dict[str, Any]:
    inconsistent = {}

    for column in (
        METADATA_COLUMNS
    ):
        counts = (
            combined.groupby(
                "repo_id"
            )[column]
            .nunique(
                dropna=False
            )
        )

        bad = counts[
            counts > 1
        ]

        inconsistent[column] = {
            "inconsistent_repositories": int(
                len(bad)
            ),
            "maximum_unique_values": int(
                counts.max()
            ),
        }

    return inconsistent


def inspect_database(
) -> tuple[
    dict[str, Any],
    dict[str, pd.DataFrame],
]:
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    tables = [
        row[0]
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()
    ]

    schemas = {}
    frames = {}

    for table in tables:
        columns = (
            connection.execute(
                f'PRAGMA table_info("{table}")'
            ).fetchall()
        )

        row_count = (
            connection.execute(
                f'SELECT COUNT(*) '
                f'FROM "{table}"'
            ).fetchone()[0]
        )

        schemas[table] = {
            "rows": int(
                row_count
            ),
            "columns": [
                {
                    "name": column[1],
                    "type": column[2],
                    "not_null": bool(
                        column[3]
                    ),
                    "primary_key": bool(
                        column[5]
                    ),
                }
                for column in columns
            ],
        }

        if table in [
            "repositories",
            "herd_modeling",
        ]:
            frames[table] = (
                pd.read_sql_query(
                    f'SELECT * '
                    f'FROM "{table}"',
                    connection,
                )
            )

    connection.close()

    return schemas, frames


def table_join_profile(
    catalog: pd.DataFrame,
    frame: pd.DataFrame,
    table_name: str,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "table": table_name,
            "available": False,
        }

    if "repo_id" not in frame.columns:
        return {
            "table": table_name,
            "available": True,
            "join_key_available": False,
            "columns": list(
                frame.columns
            ),
        }

    table_repo_ids = set(
        pd.to_numeric(
            frame["repo_id"],
            errors="coerce",
        )
        .dropna()
        .astype("int64")
        .tolist()
    )

    catalog_repo_ids = set(
        catalog[
            "repo_id"
        ].astype("int64")
        .tolist()
    )

    overlap = (
        catalog_repo_ids
        & table_repo_ids
    )

    return {
        "table": table_name,
        "available": True,
        "join_key_available": True,
        "rows": int(
            len(frame)
        ),
        "unique_repo_ids": int(
            len(table_repo_ids)
        ),
        "catalog_repositories": int(
            len(catalog_repo_ids)
        ),
        "catalog_overlap": int(
            len(overlap)
        ),
        "catalog_coverage": float(
            len(overlap)
            / len(catalog_repo_ids)
        ),
        "columns": list(
            frame.columns
        ),
    }


def build_cooccurrence_edges(
    train: pd.DataFrame,
) -> pd.DataFrame:
    pair_counts = Counter()

    histories = (
        train.groupby(
            "user_id"
        )["repo_id"]
        .apply(
            lambda values: sorted(
                set(
                    int(value)
                    for value in values
                )
            )
        )
    )

    eligible_histories = histories[
        histories.map(len) >= 2
    ]

    for repositories in (
        eligible_histories
    ):
        for left, right in combinations(
            repositories,
            2,
        ):
            pair_counts[
                (
                    left,
                    right,
                )
            ] += 1

    catalog_names = (
        train[
            [
                "repo_id",
                "repo_full_name",
            ]
        ]
        .drop_duplicates(
            "repo_id"
        )
        .set_index(
            "repo_id"
        )[
            "repo_full_name"
        ]
        .to_dict()
    )

    rows = [
        {
            "repo_id_left": left,
            "repo_name_left": (
                catalog_names.get(
                    left,
                    ""
                )
            ),
            "repo_id_right": right,
            "repo_name_right": (
                catalog_names.get(
                    right,
                    ""
                )
            ),
            "cooccurrence_count": count,
        }
        for (
            left,
            right,
        ), count in pair_counts.items()
    ]

    if not rows:
        return pd.DataFrame(
            columns=[
                "repo_id_left",
                "repo_name_left",
                "repo_id_right",
                "repo_name_right",
                "cooccurrence_count",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "cooccurrence_count",
                "repo_id_left",
                "repo_id_right",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def main() -> None:
    if OUTPUT_DIR.exists():
        raise RuntimeError(
            "Recommendation design-audit "
            "outputs already exist. "
            "Refusing to overwrite:\n"
            f"{OUTPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    train = load_split(
        TRAIN_PATH,
        "train",
    )

    validation = load_split(
        VALIDATION_PATH,
        "validation",
    )

    test_header = pd.read_csv(
        TEST_PATH,
        nrows=0,
    )

    missing_test_columns = (
        set(EXPECTED_COLUMNS)
        - set(
            test_header.columns
        )
    )

    if missing_test_columns:
        raise ValueError(
            "Test header is missing "
            f"columns: "
            f"{sorted(missing_test_columns)}"
        )

    combined_development = (
        pd.concat(
            [
                train.assign(
                    split="train"
                ),
                validation.assign(
                    split="validation"
                ),
            ],
            ignore_index=True,
        )
    )

    train_users = set(
        train["user_id"].tolist()
    )

    validation_users = set(
        validation[
            "user_id"
        ].tolist()
    )

    train_catalog = set(
        train["repo_id"].tolist()
    )

    validation_catalog = set(
        validation[
            "repo_id"
        ].tolist()
    )

    validation_counts = (
        validation.groupby(
            "user_id"
        ).size()
    )

    validation_user_train_counts = (
        train[
            train[
                "user_id"
            ].isin(
                validation_users
            )
        ]
        .groupby(
            "user_id"
        )
        .size()
        .reindex(
            sorted(
                validation_users
            ),
            fill_value=0,
        )
    )

    train_latest = (
        train[
            train[
                "user_id"
            ].isin(
                validation_users
            )
        ]
        .groupby(
            "user_id"
        )[
            "starred_at"
        ]
        .max()
    )

    validation_time = (
        validation.set_index(
            "user_id"
        )[
            "starred_at"
        ]
    )

    temporal_comparison = (
        pd.DataFrame(
            {
                "train_latest": (
                    train_latest
                ),
                "validation_time": (
                    validation_time
                ),
            }
        )
        .dropna()
    )

    temporal_violations = int(
        (
            temporal_comparison[
                "train_latest"
            ]
            >= temporal_comparison[
                "validation_time"
            ]
        ).sum()
    )

    overlap_pairs = (
        train[
            [
                "user_id",
                "repo_id",
            ]
        ]
        .merge(
            validation[
                [
                    "user_id",
                    "repo_id",
                ]
            ],
            on=[
                "user_id",
                "repo_id",
            ],
            how="inner",
        )
    )

    catalog = (
        combined_development[
            [
                "repo_id",
                "repo_full_name",
                "language",
                "topic_count",
                "stars_count",
                "forks_count",
                "repo_popularity_count",
            ]
        ]
        .drop_duplicates(
            "repo_id",
            keep="last",
        )
        .sort_values(
            "repo_id"
        )
        .reset_index(drop=True)
    )

    train_item_counts = (
        train.groupby(
            [
                "repo_id",
                "repo_full_name",
            ]
        )
        .size()
        .rename(
            "train_interaction_count"
        )
        .reset_index()
    )

    validation_item_counts = (
        validation.groupby(
            "repo_id"
        )
        .size()
        .rename(
            "validation_interaction_count"
        )
        .reset_index()
    )

    item_profile = (
        catalog.merge(
            train_item_counts,
            on=[
                "repo_id",
                "repo_full_name",
            ],
            how="left",
            validate="one_to_one",
        )
        .merge(
            validation_item_counts,
            on="repo_id",
            how="left",
            validate="one_to_one",
        )
    )

    item_profile[
        [
            "train_interaction_count",
            "validation_interaction_count",
        ]
    ] = item_profile[
        [
            "train_interaction_count",
            "validation_interaction_count",
        ]
    ].fillna(0).astype(int)

    total_train_interactions = float(
        len(train)
    )

    item_profile[
        "train_popularity_share"
    ] = (
        item_profile[
            "train_interaction_count"
        ]
        / total_train_interactions
    )

    item_profile[
        "train_popularity_rank"
    ] = (
        item_profile[
            "train_interaction_count"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    item_profile = (
        item_profile.sort_values(
            [
                "train_interaction_count",
                "repo_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    evaluation_history = (
        validation_user_train_counts
        .rename(
            "train_history_count"
        )
        .reset_index()
    )

    evaluation_history[
        "candidate_count_validation"
    ] = (
        len(train_catalog)
        - evaluation_history[
            "train_history_count"
        ]
    )

    validation_positive_seen = (
        validation[
            [
                "user_id",
                "repo_id",
            ]
        ]
        .merge(
            train[
                [
                    "user_id",
                    "repo_id",
                ]
            ],
            on=[
                "user_id",
                "repo_id",
            ],
            how="left",
            indicator=True,
        )
    )

    seen_positive_count = int(
        (
            validation_positive_seen[
                "_merge"
            ]
            == "both"
        ).sum()
    )

    user_profile = (
        train.groupby(
            "user_id"
        )
        .agg(
            train_history_count=(
                "repo_id",
                "size",
            ),
            unique_languages=(
                "language",
                "nunique",
            ),
            first_interaction=(
                "starred_at",
                "min",
            ),
            latest_train_interaction=(
                "starred_at",
                "max",
            ),
        )
        .reset_index()
    )

    user_profile[
        "is_validation_user"
    ] = user_profile[
        "user_id"
    ].isin(
        validation_users
    )

    language_counts = (
        catalog["language"]
        .value_counts(
            dropna=False
        )
    )

    metadata_consistency = (
        validate_metadata_consistency(
            combined_development
        )
    )

    repo_popularity_values = (
        catalog[
            "repo_popularity_count"
        ]
    )

    repo_popularity_cap = int(
        repo_popularity_values.max()
    )

    repositories_at_cap = int(
        (
            repo_popularity_values
            == repo_popularity_cap
        ).sum()
    )

    schemas, database_frames = (
        inspect_database()
    )

    database_join_profiles = {
        table_name: (
            table_join_profile(
                catalog=catalog,
                frame=frame,
                table_name=table_name,
            )
        )
        for (
            table_name,
            frame,
        ) in database_frames.items()
    }

    cooccurrence = (
        build_cooccurrence_edges(
            train
        )
    )

    possible_item_pairs = (
        len(train_catalog)
        * (
            len(train_catalog)
            - 1
        )
        / 2
    )

    observed_item_pairs = int(
        len(cooccurrence)
    )

    candidate_profile = (
        evaluation_history[
            [
                "user_id",
                "train_history_count",
                "candidate_count_validation",
            ]
        ]
        .sort_values(
            [
                "train_history_count",
                "user_id",
            ]
        )
        .reset_index(drop=True)
    )

    catalog.to_csv(
        CATALOG_PATH,
        index=False,
    )

    user_profile.to_csv(
        USER_HISTORY_PATH,
        index=False,
    )

    candidate_profile.to_csv(
        CANDIDATE_SIZE_PATH,
        index=False,
    )

    item_profile.to_csv(
        ITEM_POPULARITY_PATH,
        index=False,
    )

    cooccurrence.head(
        500
    ).to_csv(
        COOCCURRENCE_PATH,
        index=False,
    )

    write_json(
        DATABASE_SCHEMA_PATH,
        schemas,
    )

    summary = {
        "audit_status": (
            "recommendation_design_audit_complete"
        ),
        "test_data_status": (
            "header_only_not_loaded_or_evaluated"
        ),
        "split_protocol": {
            "feedback_type": "implicit",
            "train_rows": int(
                len(train)
            ),
            "validation_rows": int(
                len(validation)
            ),
            "train_users": int(
                len(train_users)
            ),
            "validation_users": int(
                len(validation_users)
            ),
            "validation_rows_per_user": (
                quantile_summary(
                    validation_counts
                )
            ),
            "validation_users_missing_from_train": int(
                len(
                    validation_users
                    - train_users
                )
            ),
            "train_validation_pair_overlap": int(
                len(overlap_pairs)
            ),
            "validation_positive_already_seen_in_train": (
                seen_positive_count
            ),
            "temporal_order_violations": (
                temporal_violations
            ),
            "minimum_train_history_for_validation_user": int(
                validation_user_train_counts.min()
            ),
            "train_history_for_validation_users": (
                quantile_summary(
                    validation_user_train_counts
                )
            ),
        },
        "catalog": {
            "train_repositories": int(
                len(train_catalog)
            ),
            "validation_repositories": int(
                len(validation_catalog)
            ),
            "validation_cold_repositories": int(
                len(
                    validation_catalog
                    - train_catalog
                )
            ),
            "matrix_density_all_train_users": float(
                len(train)
                / (
                    len(train_users)
                    * len(train_catalog)
                )
            ),
            "languages": int(
                catalog[
                    "language"
                ].nunique()
            ),
            "language_distribution": {
                str(key): int(value)
                for key, value in (
                    language_counts.items()
                )
            },
            "topic_count_distribution": (
                quantile_summary(
                    catalog[
                        "topic_count"
                    ]
                )
            ),
        },
        "candidate_generation": {
            "strategy": "full_catalog",
            "validation_exclusion_rule": (
                "Exclude repositories already "
                "seen in training; retain the "
                "single validation positive."
            ),
            "validation_candidate_count": (
                quantile_summary(
                    candidate_profile[
                        "candidate_count_validation"
                    ]
                )
            ),
            "all_validation_positives_eligible": bool(
                seen_positive_count == 0
            ),
        },
        "popularity": {
            "training_interaction_count": (
                quantile_summary(
                    item_profile[
                        "train_interaction_count"
                    ]
                )
            ),
            "training_popularity_gini": (
                gini(
                    item_profile[
                        "train_interaction_count"
                    ].to_numpy(
                        dtype=float
                    )
                )
            ),
            "repo_popularity_count_maximum": (
                repo_popularity_cap
            ),
            "repositories_at_maximum_repo_popularity_count": (
                repositories_at_cap
            ),
            "fraction_at_maximum_repo_popularity_count": float(
                repositories_at_cap
                / len(catalog)
            ),
            "repo_popularity_count_warning": (
                "The collected repository "
                "popularity count is capped "
                "for nearly the full catalog "
                "and must not be treated as a "
                "faithful external popularity "
                "target."
            ),
        },
        "metadata": {
            "consistency": (
                metadata_consistency
            ),
            "temporal_leakage_warning": (
                "stars_count, forks_count, "
                "language, topic_count, and "
                "repo_popularity_count are "
                "static repository snapshots. "
                "Use stars_count and forks_count "
                "only in a clearly labeled "
                "non-temporal content baseline "
                "unless historical snapshot "
                "timestamps can be established."
            ),
            "topic_limitation": (
                "Only topic_count is present in "
                "the recommendation files; actual "
                "topic identities are not present."
            ),
        },
        "collaborative_graph": {
            "users_with_at_least_two_train_interactions": int(
                (
                    user_profile[
                        "train_history_count"
                    ]
                    >= 2
                ).sum()
            ),
            "observed_item_pairs": (
                observed_item_pairs
            ),
            "possible_item_pairs": int(
                possible_item_pairs
            ),
            "pair_graph_density": float(
                observed_item_pairs
                / possible_item_pairs
            ),
            "maximum_pair_cooccurrence": int(
                cooccurrence[
                    "cooccurrence_count"
                ].max()
                if not cooccurrence.empty
                else 0
            ),
        },
        "database_join_profiles": (
            database_join_profiles
        ),
        "recommended_validation_protocol": {
            "catalog": (
                "Use the 145-item training "
                "catalog."
            ),
            "validation_candidates": (
                "For each evaluation user, "
                "rank every catalog item except "
                "items observed in that user's "
                "training history."
            ),
            "test_candidates_after_freeze": (
                "For final test evaluation, "
                "exclude both training and "
                "validation histories, then "
                "rank the remaining full "
                "catalog."
            ),
            "negative_sampling": (
                "Do not use sampled negatives "
                "for final ranking metrics. "
                "The catalog is small enough "
                "for exact full-catalog ranking. "
                "Negative sampling may be used "
                "only inside model training."
            ),
            "primary_metric": "ndcg_at_10",
            "selection_rule": (
                "Tune each model family on "
                "training data and select using "
                "validation NDCG@10, with MRR "
                "and Recall@10 as tie-breakers. "
                "Do not inspect test ranking "
                "metrics until the recommender "
                "is frozen."
            ),
        },
        "eligible_model_families": [
            "random ranking sanity baseline",
            "training-popularity baseline",
            "language-affinity content baseline",
            "item-item cosine collaborative filtering",
            "TruncatedSVD latent-factor recommender",
            "co-occurrence graph recommender",
            "rank-normalized hybrid recommender",
            "herd-aware reranker using time-safe training popularity",
        ],
        "excluded_or_restricted_features": {
            "repo_popularity_count": (
                "Excluded as a meaningful "
                "external popularity feature "
                "because of the collection cap."
            ),
            "stars_count": (
                "Restricted because it is a "
                "static present-day snapshot "
                "and may leak future popularity "
                "into historical recommendations."
            ),
            "forks_count": (
                "Restricted for the same "
                "temporal-snapshot reason."
            ),
            "topic_count": (
                "Permitted only as a weak scalar "
                "content feature; it does not "
                "encode topic identity."
            ),
        },
        "output_paths": {
            "summary": str(
                SUMMARY_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "catalog_profile": str(
                CATALOG_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "user_history_profile": str(
                USER_HISTORY_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "candidate_size_profile": str(
                CANDIDATE_SIZE_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "item_popularity_profile": str(
                ITEM_POPULARITY_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "top_cooccurrence_edges": str(
                COOCCURRENCE_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "database_schema": str(
                DATABASE_SCHEMA_PATH.relative_to(
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
        "RECOMMENDATION DESIGN AUDIT"
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
        "TOP TRAINING POPULARITY ITEMS"
    )
    print("=" * 108)

    print(
        item_profile.head(
            20
        ).to_string(
            index=False
        )
    )

    print()
    print("=" * 108)
    print(
        "TOP ITEM COOCCURRENCE EDGES"
    )
    print("=" * 108)

    print(
        cooccurrence.head(
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
