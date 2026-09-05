import hashlib
import importlib.util
import json
import math
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "train_recommendation_models.py"
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

DATABASE_PATH = (
    PROJECT_ROOT
    / "database"
    / "github_herd.db"
)

WEEKLY_FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weekly_timeseries_features.csv"
)

FORECAST_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
    / "selected_forecaster_pretest.joblib"
)

FORECAST_METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
    / "selected_forecaster_pretest_metadata.json"
)

PRIMARY_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
    / "selected_recommender_development.joblib"
)

PRIMARY_METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
    / "selected_recommender_development_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "recommendation"
    / "secondary_branches"
)

INTERNAL_RESULTS_PATH = (
    OUTPUT_DIR
    / "secondary_internal_tuning_results.csv"
)

INTERNAL_WINNERS_PATH = (
    OUTPUT_DIR
    / "secondary_internal_family_winners.csv"
)

EXTERNAL_RESULTS_PATH = (
    OUTPUT_DIR
    / "secondary_external_validation_results.csv"
)

BOOTSTRAP_PATH = (
    OUTPUT_DIR
    / "secondary_paired_bootstrap.csv"
)

FORECAST_COVERAGE_PATH = (
    OUTPUT_DIR
    / "forecast_signal_coverage_summary.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "secondary_recommendation_summary.json"
)

NDCG_PLOT_PATH = (
    OUTPUT_DIR
    / "secondary_external_validation_ndcg_at_10.png"
)

COVERAGE_PLOT_PATH = (
    OUTPUT_DIR
    / "secondary_external_validation_coverage_at_10.png"
)

RANDOM_STATE = 42
BOOTSTRAP_REPLICATIONS = 2000
CATALOG_SIZE = 145
TOLERANCE = 1e-12


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
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


def git_output(arguments: list[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def load_training_module():
    spec = importlib.util.spec_from_file_location(
        "recommendation_training_module_secondary",
        TRAINING_SCRIPT_PATH,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Could not load the primary "
            "recommendation training module."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def verify_test_header_only() -> list[str]:
    header = pd.read_csv(
        TEST_PATH,
        nrows=0,
    )

    required = {
        "user_id",
        "repo_id",
        "starred_at",
    }

    missing = required - set(
        header.columns
    )

    if missing:
        raise RuntimeError(
            "Recommendation test header is "
            f"missing columns: {sorted(missing)}"
        )

    return list(
        header.columns
    )


def construct_internal_fit(
    train: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    holdout_keys = (
        assignments[
            [
                "user_id",
                "internal_holdout_repo_id",
            ]
        ]
        .rename(
            columns={
                "internal_holdout_repo_id": (
                    "repo_id"
                )
            }
        )
        .assign(
            _internal_holdout=True
        )
    )

    merged = train.merge(
        holdout_keys,
        on=[
            "user_id",
            "repo_id",
        ],
        how="left",
    )

    internal_fit = merged[
        merged[
            "_internal_holdout"
        ].isna()
    ].drop(
        columns=[
            "_internal_holdout",
        ]
    )

    if (
        len(train)
        - len(internal_fit)
        != len(assignments)
    ):
        raise RuntimeError(
            "Unexpected number of internal "
            "holdout rows removed."
        )

    return internal_fit


def ppmi_matrix(
    context,
) -> np.ndarray:
    matrix = (
        context.fit_matrix_all_users
    )

    cooccurrence = (
        matrix.T
        @ matrix
    ).toarray().astype(
        np.float64
    )

    number_of_users = float(
        matrix.shape[0]
    )

    item_probability = (
        np.diag(
            cooccurrence
        )
        / number_of_users
    )

    joint_probability = (
        cooccurrence
        / number_of_users
    )

    expected = np.outer(
        item_probability,
        item_probability,
    )

    ppmi = np.zeros_like(
        cooccurrence,
        dtype=np.float64,
    )

    valid = (
        joint_probability > 0
    ) & (
        expected > 0
    )

    ppmi[valid] = np.maximum(
        np.log(
            joint_probability[valid]
            / expected[valid]
        ),
        0.0,
    )

    np.fill_diagonal(
        ppmi,
        0.0,
    )

    return ppmi


def rebuild_primary_scores(
    context,
    artifact: dict[str, Any],
    module,
) -> np.ndarray:
    winners = artifact[
        "internal_family_winners"
    ]

    item_parameters = winners[
        "item_item_cosine"
    ][
        "parameters"
    ]

    item_similarity = (
        module.build_item_similarity(
            context=context,
            shrinkage=float(
                item_parameters[
                    "shrinkage"
                ]
            ),
        )
    )

    item_scores = (
        module.item_item_scores(
            context=context,
            similarity=item_similarity,
            aggregation=item_parameters[
                "aggregation"
            ],
        )
    )

    svd_parameters = winners[
        "truncated_svd"
    ][
        "parameters"
    ]

    svd_score_matrix, _ = (
        module.svd_scores(
            context=context,
            n_components=int(
                svd_parameters[
                    "n_components"
                ]
            ),
            n_iter=int(
                svd_parameters[
                    "n_iter"
                ]
            ),
        )
    )

    graph_parameters = winners[
        "graph_personalized_pagerank"
    ][
        "parameters"
    ]

    edge_weight = graph_parameters[
        "edge_weight"
    ]

    if edge_weight == "cosine":
        edge_matrix = (
            context.interaction_cosine
        )
    elif edge_weight == "ppmi":
        edge_matrix = ppmi_matrix(
            context
        )
    else:
        raise ValueError(
            f"Unknown graph edge weight: "
            f"{edge_weight}"
        )

    propagation = (
        module.graph_propagation_matrix(
            edge_matrix=edge_matrix,
            restart_probability=float(
                graph_parameters[
                    "restart_probability"
                ]
            ),
        )
    )

    graph_score_matrix = (
        module.graph_scores(
            context=context,
            propagation=propagation,
        )
    )

    popularity_score_matrix = (
        module.popularity_scores(
            context
        )
    )

    component_scores = {
        "item_item_cosine": (
            item_scores
        ),
        "truncated_svd": (
            svd_score_matrix
        ),
        "graph_personalized_pagerank": (
            graph_score_matrix
        ),
        "training_popularity": (
            popularity_score_matrix
        ),
    }

    normalized = {
        name: (
            module.rank_normalize_scores(
                scores=scores,
                seen_mask=(
                    context.seen_mask
                ),
            )
        )
        for name, scores in (
            component_scores.items()
        )
    }

    hybrid_scores = np.zeros_like(
        popularity_score_matrix,
        dtype=np.float64,
    )

    for name, weight in (
        artifact[
            "hybrid_parameters"
        ][
            "weights"
        ].items()
    ):
        hybrid_scores += (
            float(weight)
            * normalized[name]
        )

    hybrid_rank = (
        module.rank_normalize_scores(
            scores=hybrid_scores,
            seen_mask=(
                context.seen_mask
            ),
        )
    )

    penalty = float(
        artifact[
            "herd_parameters"
        ][
            "popularity_penalty"
        ]
    )

    return (
        hybrid_rank
        - penalty
        * normalized[
            "training_popularity"
        ]
    )


def artifact_primary_scores(
    context,
    artifact: dict[str, Any],
    module,
) -> np.ndarray:
    component_artifacts = artifact[
        "component_artifacts"
    ]

    winners = artifact[
        "internal_family_winners"
    ]

    item_parameters = winners[
        "item_item_cosine"
    ][
        "parameters"
    ]

    item_scores = (
        module.item_item_scores(
            context=context,
            similarity=np.asarray(
                component_artifacts[
                    "item_similarity"
                ],
                dtype=np.float64,
            ),
            aggregation=item_parameters[
                "aggregation"
            ],
        )
    )

    components = np.asarray(
        component_artifacts[
            "svd_components"
        ],
        dtype=np.float64,
    )

    user_latent = (
        context.history_matrix
        @ components.T
    )

    svd_score_matrix = np.asarray(
        user_latent
        @ components,
        dtype=np.float64,
    )

    graph_score_matrix = (
        module.graph_scores(
            context=context,
            propagation=np.asarray(
                component_artifacts[
                    "graph_propagation"
                ],
                dtype=np.float64,
            ),
        )
    )

    popularity_score_matrix = (
        module.popularity_scores(
            context
        )
    )

    component_scores = {
        "item_item_cosine": (
            item_scores
        ),
        "truncated_svd": (
            svd_score_matrix
        ),
        "graph_personalized_pagerank": (
            graph_score_matrix
        ),
        "training_popularity": (
            popularity_score_matrix
        ),
    }

    normalized = {
        name: (
            module.rank_normalize_scores(
                scores=scores,
                seen_mask=(
                    context.seen_mask
                ),
            )
        )
        for name, scores in (
            component_scores.items()
        )
    }

    hybrid_scores = np.zeros_like(
        popularity_score_matrix,
        dtype=np.float64,
    )

    for name, weight in (
        artifact[
            "hybrid_parameters"
        ][
            "weights"
        ].items()
    ):
        hybrid_scores += (
            float(weight)
            * normalized[name]
        )

    hybrid_rank = (
        module.rank_normalize_scores(
            scores=hybrid_scores,
            seen_mask=(
                context.seen_mask
            ),
        )
    )

    penalty = float(
        artifact[
            "herd_parameters"
        ][
            "popularity_penalty"
        ]
    )

    return (
        hybrid_rank
        - penalty
        * normalized[
            "training_popularity"
        ]
    )


def static_metadata(
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    repositories = pd.read_sql_query(
        """
        SELECT
            repo_id,
            full_name,
            owner,
            repo_name,
            description,
            language,
            topics,
            collected_at
        FROM repositories
        """,
        connection,
    )

    connection.close()

    if (
        repositories[
            "repo_id"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Repository metadata contains "
            "duplicate repository IDs."
        )

    merged = catalog.merge(
        repositories,
        on="repo_id",
        how="left",
        validate="one_to_one",
    )

    if merged[
        "full_name"
    ].isna().any():
        raise RuntimeError(
            "Static metadata does not cover "
            "the complete recommendation "
            "catalog."
        )

    return merged


def sanitize_token(value: str) -> str:
    token = str(value).strip().lower()

    for character in [
        " ",
        "-",
        "/",
        ".",
        "+",
        "#",
    ]:
        token = token.replace(
            character,
            "_",
        )

    return (
        "".join(
            character
            for character in token
            if (
                character.isalnum()
                or character == "_"
            )
        )
    )


def content_documents(
    metadata: pd.DataFrame,
    topic_repeat: int,
    language_repeat: int,
) -> list[str]:
    documents = []

    for row in metadata.itertuples(
        index=False
    ):
        topics = []

        if pd.notna(
            row.topics
        ):
            topics = [
                sanitize_token(topic)
                for topic in str(
                    row.topics
                ).split("|")
                if str(topic).strip()
            ]

        topic_tokens = [
            f"topic_{topic}"
            for topic in topics
            if topic
        ]

        language_tokens = []

        if (
            pd.notna(
                row.language
            )
            and str(
                row.language
            ).strip()
        ):
            language_tokens = [
                "language_"
                + sanitize_token(
                    row.language
                )
            ]

        base_fields = [
            str(
                row.full_name
            ),
            str(
                row.owner
            ),
            str(
                row.repo_name
            ),
            (
                ""
                if pd.isna(
                    row.description
                )
                else str(
                    row.description
                )
            ),
        ]

        tokens = (
            base_fields
            + topic_tokens
            * topic_repeat
            + language_tokens
            * language_repeat
        )

        documents.append(
            " ".join(tokens)
        )

    return documents


def content_item_matrix(
    metadata: pd.DataFrame,
    ngram_range: tuple[int, int],
    topic_repeat: int,
    language_repeat: int,
) -> tuple[
    sparse.csr_matrix,
    TfidfVectorizer,
]:
    documents = content_documents(
        metadata=metadata,
        topic_repeat=topic_repeat,
        language_repeat=language_repeat,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=ngram_range,
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        norm="l2",
        token_pattern=(
            r"(?u)\b[\w_]{2,}\b"
        ),
    )

    matrix = vectorizer.fit_transform(
        documents
    ).tocsr()

    return matrix, vectorizer


def content_scores(
    context,
    item_matrix: sparse.csr_matrix,
) -> np.ndarray:
    profiles = (
        context.history_matrix
        @ item_matrix
    )

    profiles = normalize(
        profiles,
        norm="l2",
        axis=1,
        copy=False,
    )

    scores = (
        profiles
        @ item_matrix.T
    )

    return scores.toarray().astype(
        np.float64
    )


def forecast_records(
    catalog: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    artifact = joblib.load(
        FORECAST_ARTIFACT_PATH
    )

    metadata = load_json(
        FORECAST_METADATA_PATH
    )

    expected_hash = metadata[
        "artifact"
    ][
        "sha256"
    ]

    actual_hash = sha256_file(
        FORECAST_ARTIFACT_PATH
    )

    if actual_hash != expected_hash:
        raise RuntimeError(
            "Frozen forecast artifact hash "
            "does not match metadata."
        )

    if (
        artifact[
            "artifact_version"
        ]
        != "forecasting_pretest_v1"
    ):
        raise RuntimeError(
            "Unexpected forecast artifact "
            "version."
        )

    weekly = pd.read_csv(
        WEEKLY_FEATURE_PATH
    )

    weekly = weekly[
        weekly[
            "repo_id"
        ].isin(
            catalog[
                "repo_id"
            ]
        )
    ].copy()

    weekly[
        "week_start"
    ] = pd.to_datetime(
        weekly[
            "week_start"
        ],
        utc=True,
        errors="raise",
    )

    weekly[
        "available_at"
    ] = (
        weekly[
            "week_start"
        ]
        + pd.Timedelta(
            days=7
        )
    )

    feature_columns = artifact[
        "feature_columns"
    ]

    missing_features = (
        set(
            feature_columns
        )
        - set(
            weekly.columns
        )
    )

    if missing_features:
        raise RuntimeError(
            "Weekly feature table is missing "
            f"features: {sorted(missing_features)}"
        )

    feature_frame = (
        weekly[
            feature_columns
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    complete = (
        feature_frame.notna()
        .all(
            axis=1
        )
    )

    if not complete.all():
        raise RuntimeError(
            "Feature-only weekly table "
            "contains incomplete frozen "
            "forecast feature rows."
        )

    baseline = (
        float(
            artifact[
                "baseline_multiplier"
            ]
        )
        * weekly[
            artifact[
                "baseline_column"
            ]
        ].astype(float)
    )

    residual = artifact[
        "residual_estimator"
    ].predict(
        feature_frame.astype(float)
    )

    predictions = np.maximum(
        float(
            artifact[
                "prediction_minimum"
            ]
        ),
        baseline.to_numpy()
        + residual,
    )

    if not np.isfinite(
        predictions
    ).all():
        raise RuntimeError(
            "Frozen forecast predictions "
            "contain nonfinite values."
        )

    weekly[
        "predicted_4week_stars"
    ] = predictions

    represented = set(
        weekly[
            "repo_id"
        ]
    )

    missing_repositories = sorted(
        set(
            catalog[
                "repo_id"
            ]
        )
        - represented
    )

    if missing_repositories:
        raise RuntimeError(
            "Feature-only forecast table "
            "does not cover all catalog "
            f"repositories: "
            f"{missing_repositories}"
        )

    summary = {
        "artifact_version": (
            artifact[
                "artifact_version"
            ]
        ),
        "artifact_sha256": (
            actual_hash
        ),
        "feature_columns": (
            feature_columns
        ),
        "weekly_rows": int(
            len(weekly)
        ),
        "repositories": int(
            weekly[
                "repo_id"
            ].nunique()
        ),
        "availability_rule": (
            "week_start plus seven days; "
            "latest row with available_at "
            "less than or equal to user "
            "cutoff"
        ),
        "prediction_minimum": float(
            predictions.min()
        ),
        "prediction_median": float(
            np.median(
                predictions
            )
        ),
        "prediction_mean": float(
            np.mean(
                predictions
            )
        ),
        "prediction_maximum": float(
            predictions.max()
        ),
    }

    return (
        weekly[
            [
                "repo_id",
                "available_at",
                "predicted_4week_stars",
            ]
        ]
        .sort_values(
            [
                "repo_id",
                "available_at",
            ]
        )
        .reset_index(drop=True),
        summary,
    )


def cutoff_aligned_prediction_matrix(
    context,
    cutoffs: pd.DataFrame,
    cutoff_column: str,
    weekly_predictions: pd.DataFrame,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
]:
    cutoff_frame = (
        cutoffs[
            [
                "user_id",
                cutoff_column,
            ]
        ]
        .copy()
        .sort_values(
            "user_id"
        )
        .reset_index(drop=True)
    )

    cutoff_frame[
        cutoff_column
    ] = pd.to_datetime(
        cutoff_frame[
            cutoff_column
        ],
        utc=True,
        errors="raise",
    )

    if not np.array_equal(
        cutoff_frame[
            "user_id"
        ].to_numpy(
            dtype=np.int64
        ),
        context.eval_user_ids,
    ):
        raise RuntimeError(
            "Forecast cutoff users do not "
            "align with evaluation context."
        )

    cutoff_ns = (
        cutoff_frame[
            cutoff_column
        ]
        .astype("int64")
        .to_numpy()
    )

    prediction_matrix = np.full(
        (
            len(
                context.eval_user_ids
            ),
            len(
                context.repo_ids
            ),
        ),
        np.nan,
        dtype=np.float64,
    )

    for repo_index, repo_id in enumerate(
        context.repo_ids
    ):
        repository_rows = (
            weekly_predictions[
                weekly_predictions[
                    "repo_id"
                ]
                == repo_id
            ]
            .sort_values(
                "available_at"
            )
        )

        available_ns = (
            repository_rows[
                "available_at"
            ]
            .astype("int64")
            .to_numpy()
        )

        repository_predictions = (
            repository_rows[
                "predicted_4week_stars"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        positions = (
            np.searchsorted(
                available_ns,
                cutoff_ns,
                side="right",
            )
            - 1
        )

        valid = positions >= 0

        prediction_matrix[
            valid,
            repo_index,
        ] = repository_predictions[
            positions[
                valid
            ]
        ]

    candidate_mask = (
        ~context.seen_mask
    )

    candidate_available = (
        np.isfinite(
            prediction_matrix
        )
        & candidate_mask
    )

    available_candidate_count = (
        candidate_available.sum(
            axis=1
        )
    )

    candidate_count = (
        candidate_mask.sum(
            axis=1
        )
    )

    positive_available = np.isfinite(
        prediction_matrix[
            np.arange(
                len(
                    context.eval_user_ids
                )
            ),
            context.positive_indices,
        ]
    )

    coverage = pd.DataFrame(
        {
            "user_id": (
                context.eval_user_ids
            ),
            "candidate_repositories": (
                candidate_count
            ),
            "available_candidate_repositories": (
                available_candidate_count
            ),
            "candidate_signal_fraction": (
                available_candidate_count
                / candidate_count
            ),
            "positive_has_signal": (
                positive_available
            ),
        }
    )

    return (
        prediction_matrix,
        coverage,
    )


def neutral_forecast_signal(
    prediction_matrix: np.ndarray,
    seen_mask: np.ndarray,
) -> np.ndarray:
    signal = np.full(
        prediction_matrix.shape,
        0.5,
        dtype=np.float64,
    )

    for row_index in range(
        prediction_matrix.shape[0]
    ):
        available = (
            np.isfinite(
                prediction_matrix[
                    row_index
                ]
            )
            & (
                ~seen_mask[
                    row_index
                ]
            )
        )

        indices = np.flatnonzero(
            available
        )

        if len(indices) < 2:
            continue

        values = prediction_matrix[
            row_index,
            indices,
        ]

        order = np.argsort(
            values,
            kind="stable",
        )

        ranks = np.empty(
            len(indices),
            dtype=np.float64,
        )

        ranks[order] = np.arange(
            len(indices),
            dtype=np.float64,
        )

        normalized = (
            ranks
            / (
                len(indices)
                - 1
            )
        )

        signal[
            row_index,
            indices,
        ] = normalized

    signal[
        seen_mask
    ] = 0.0

    return signal


def evaluate(
    module,
    scores: np.ndarray,
    context,
    family: str,
    name: str,
    parameters: dict[str, Any],
    external: bool,
) -> tuple[
    dict[str, Any],
    pd.DataFrame | None,
]:
    return module.evaluate_scores(
        scores=scores,
        context=context,
        model_family=family,
        model_name=name,
        parameters=parameters,
        return_user_ranks=external,
        compute_beyond_accuracy=external,
    )


def select_best(
    frame: pd.DataFrame,
) -> pd.Series:
    ordered = frame.sort_values(
        [
            "ndcg_at_10",
            "mrr",
            "recall_at_10",
            "ndcg_at_5",
            "model_name",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
        kind="stable",
    )

    return ordered.iloc[0]


def parse_parameters(
    row: pd.Series,
) -> dict[str, Any]:
    return json.loads(
        row[
            "parameters_json"
        ]
    )


def bootstrap_comparisons(
    rank_frames: list[pd.DataFrame],
    primary_family: str,
) -> pd.DataFrame:
    combined = pd.concat(
        rank_frames,
        ignore_index=True,
    )

    primary = (
        combined[
            combined[
                "model_family"
            ]
            == primary_family
        ]
        .sort_values(
            "user_id"
        )
        .reset_index(drop=True)
    )

    other_families = sorted(
        set(
            combined[
                "model_family"
            ]
        )
        - {
            primary_family,
        }
    )

    metrics = {
        "ndcg_at_10": (
            "ndcg_at_10"
        ),
        "mrr": (
            "reciprocal_rank"
        ),
        "recall_at_10": (
            "hit_at_10"
        ),
    }

    generator = np.random.default_rng(
        RANDOM_STATE
    )

    rows = []

    for family in other_families:
        comparator = (
            combined[
                combined[
                    "model_family"
                ]
                == family
            ]
            .sort_values(
                "user_id"
            )
            .reset_index(drop=True)
        )

        if not np.array_equal(
            primary[
                "user_id"
            ].to_numpy(),
            comparator[
                "user_id"
            ].to_numpy(),
        ):
            raise RuntimeError(
                "Secondary per-user rank tables "
                "are not aligned."
            )

        for metric, column in (
            metrics.items()
        ):
            differences = (
                comparator[
                    column
                ].to_numpy(
                    dtype=float
                )
                - primary[
                    column
                ].to_numpy(
                    dtype=float
                )
            )

            bootstrap_means = np.empty(
                BOOTSTRAP_REPLICATIONS,
                dtype=np.float64,
            )

            for replicate in range(
                BOOTSTRAP_REPLICATIONS
            ):
                sample = (
                    generator.integers(
                        0,
                        len(
                            differences
                        ),
                        size=len(
                            differences
                        ),
                    )
                )

                bootstrap_means[
                    replicate
                ] = float(
                    differences[
                        sample
                    ].mean()
                )

            rows.append(
                {
                    "secondary_family": (
                        family
                    ),
                    "reference_family": (
                        primary_family
                    ),
                    "metric": metric,
                    "users": int(
                        len(
                            differences
                        )
                    ),
                    "observed_secondary_minus_primary": float(
                        differences.mean()
                    ),
                    "ci_2_5": float(
                        np.quantile(
                            bootstrap_means,
                            0.025,
                        )
                    ),
                    "ci_97_5": float(
                        np.quantile(
                            bootstrap_means,
                            0.975,
                        )
                    ),
                    "probability_secondary_better": float(
                        np.mean(
                            bootstrap_means
                            > 0
                        )
                    ),
                    "bootstrap_replications": (
                        BOOTSTRAP_REPLICATIONS
                    ),
                    "random_state": (
                        RANDOM_STATE
                    ),
                }
            )

    return pd.DataFrame(rows)


def save_bar_plot(
    frame: pd.DataFrame,
    metric: str,
    path: Path,
    title: str,
) -> None:
    ordered = frame.sort_values(
        metric,
        ascending=True,
    )

    labels = (
        ordered[
            "model_family"
        ]
        .astype(str)
        .tolist()
    )

    values = ordered[
        metric
    ].to_numpy(
        dtype=float
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.barh(
        labels,
        values,
    )

    axis.set_xlabel(
        metric
    )

    axis.set_title(
        title
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main() -> None:
    if OUTPUT_DIR.exists():
        raise RuntimeError(
            "Secondary recommendation output "
            "directory already exists. "
            "Refusing to overwrite:\n"
            f"{OUTPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    test_header_columns = (
        verify_test_header_only()
    )

    module = load_training_module()

    train = module.load_interactions(
        TRAIN_PATH
    )

    validation = module.load_interactions(
        VALIDATION_PATH
    )

    assignments = pd.read_csv(
        INTERNAL_ASSIGNMENTS_PATH,
        parse_dates=[
            "internal_holdout_time",
        ],
    )

    assignments[
        "user_id"
    ] = pd.to_numeric(
        assignments[
            "user_id"
        ],
        errors="raise",
    ).astype("int64")

    assignments[
        "internal_holdout_repo_id"
    ] = pd.to_numeric(
        assignments[
            "internal_holdout_repo_id"
        ],
        errors="raise",
    ).astype("int64")

    primary_validation = pd.read_csv(
        PRIMARY_VALIDATION_USERS_PATH,
        parse_dates=[
            "validation_time",
            "latest_train_time",
        ],
    )

    primary_validation[
        "user_id"
    ] = pd.to_numeric(
        primary_validation[
            "user_id"
        ],
        errors="raise",
    ).astype("int64")

    primary_validation[
        "validation_repo_id"
    ] = pd.to_numeric(
        primary_validation[
            "validation_repo_id"
        ],
        errors="raise",
    ).astype("int64")

    primary_artifact = joblib.load(
        PRIMARY_ARTIFACT_PATH
    )

    primary_metadata = load_json(
        PRIMARY_METADATA_PATH
    )

    primary_artifact_hash = (
        sha256_file(
            PRIMARY_ARTIFACT_PATH
        )
    )

    if (
        primary_artifact_hash
        != primary_metadata[
            "development_artifact_sha256"
        ]
    ):
        raise RuntimeError(
            "Primary recommendation artifact "
            "hash does not match metadata."
        )

    if (
        primary_artifact[
            "test_data_status"
        ]
        != (
            "header_only_not_loaded_or_"
            "evaluated"
        )
    ):
        raise RuntimeError(
            "Primary recommendation artifact "
            "does not preserve the sealed "
            "test status."
        )

    catalog = module.build_catalog(
        train
    )

    if len(catalog) != CATALOG_SIZE:
        raise RuntimeError(
            "Unexpected recommendation "
            f"catalog size: {len(catalog)}"
        )

    internal_fit = (
        construct_internal_fit(
            train=train,
            assignments=assignments,
        )
    )

    internal_context = (
        module.build_context(
            fit_interactions=(
                internal_fit
            ),
            holdout=assignments,
            catalog=catalog,
            holdout_repo_column=(
                "internal_holdout_repo_id"
            ),
        )
    )

    external_context = (
        module.build_context(
            fit_interactions=train,
            holdout=primary_validation,
            catalog=catalog,
            holdout_repo_column=(
                "validation_repo_id"
            ),
        )
    )

    internal_primary_scores = (
        rebuild_primary_scores(
            context=internal_context,
            artifact=primary_artifact,
            module=module,
        )
    )

    external_primary_scores = (
        artifact_primary_scores(
            context=external_context,
            artifact=primary_artifact,
            module=module,
        )
    )

    reproduced_primary, _ = evaluate(
        module=module,
        scores=external_primary_scores,
        context=external_context,
        family=(
            "primary_herd_mitigation"
        ),
        name=(
            primary_artifact[
                "selected_model_name"
            ]
        ),
        parameters=(
            primary_artifact[
                "selected_parameters"
            ]
        ),
        external=True,
    )

    stored_primary = (
        primary_metadata[
            "selected_external_validation_metrics"
        ]
    )

    primary_metric_differences = {
        metric: abs(
            float(
                reproduced_primary[
                    metric
                ]
            )
            - float(
                value
            )
        )
        for metric, value in (
            stored_primary.items()
        )
    }

    if not all(
        np.isfinite(
            difference
        )
        for difference in (
            primary_metric_differences.values()
        )
    ):
        raise RuntimeError(
            "Primary metric reproduction "
            "produced nonfinite differences."
        )

    if (
        max(
            primary_metric_differences.values()
        )
        > TOLERANCE
    ):
        raise RuntimeError(
            "External primary scores did not "
            "reproduce the stored metrics."
        )

    metadata = static_metadata(
        catalog
    )

    weekly_predictions, forecast_summary = (
        forecast_records(
            catalog
        )
    )

    internal_forecast_matrix, (
        internal_forecast_coverage
    ) = cutoff_aligned_prediction_matrix(
        context=internal_context,
        cutoffs=assignments,
        cutoff_column=(
            "internal_holdout_time"
        ),
        weekly_predictions=(
            weekly_predictions
        ),
    )

    external_forecast_matrix, (
        external_forecast_coverage
    ) = cutoff_aligned_prediction_matrix(
        context=external_context,
        cutoffs=primary_validation,
        cutoff_column=(
            "validation_time"
        ),
        weekly_predictions=(
            weekly_predictions
        ),
    )

    internal_forecast_signal = (
        neutral_forecast_signal(
            prediction_matrix=(
                internal_forecast_matrix
            ),
            seen_mask=(
                internal_context.seen_mask
            ),
        )
    )

    external_forecast_signal = (
        neutral_forecast_signal(
            prediction_matrix=(
                external_forecast_matrix
            ),
            seen_mask=(
                external_context.seen_mask
            ),
        )
    )

    internal_primary_rank = (
        module.rank_normalize_scores(
            scores=(
                internal_primary_scores
            ),
            seen_mask=(
                internal_context.seen_mask
            ),
        )
    )

    external_primary_rank = (
        module.rank_normalize_scores(
            scores=(
                external_primary_scores
            ),
            seen_mask=(
                external_context.seen_mask
            ),
        )
    )

    internal_results = []
    content_cache = {}

    content_configurations = []

    for ngram_range in [
        (1, 1),
        (1, 2),
    ]:
        for topic_repeat in [
            1,
            2,
        ]:
            for language_repeat in [
                1,
                2,
            ]:
                content_configurations.append(
                    {
                        "ngram_range": (
                            list(
                                ngram_range
                            )
                        ),
                        "topic_repeat": (
                            topic_repeat
                        ),
                        "language_repeat": (
                            language_repeat
                        ),
                    }
                )

    for configuration in (
        content_configurations
    ):
        ngram_range = tuple(
            configuration[
                "ngram_range"
            ]
        )

        item_matrix, vectorizer = (
            content_item_matrix(
                metadata=metadata,
                ngram_range=(
                    ngram_range
                ),
                topic_repeat=int(
                    configuration[
                        "topic_repeat"
                    ]
                ),
                language_repeat=int(
                    configuration[
                        "language_repeat"
                    ]
                ),
            )
        )

        scores = content_scores(
            context=internal_context,
            item_matrix=item_matrix,
        )

        name = (
            "tfidf_"
            f"ngram_{ngram_range[0]}_"
            f"{ngram_range[1]}_"
            "topic_"
            f"{configuration['topic_repeat']}_"
            "language_"
            f"{configuration['language_repeat']}"
        )

        parameters = {
            **configuration,
            "vocabulary_size": int(
                len(
                    vectorizer.vocabulary_
                )
            ),
            "metadata_snapshot": (
                "2026-06-13"
            ),
            "historically_time_safe": (
                False
            ),
        }

        result, _ = evaluate(
            module=module,
            scores=scores,
            context=internal_context,
            family=(
                "secondary_tfidf_content"
            ),
            name=name,
            parameters=parameters,
            external=False,
        )

        internal_results.append(
            result
        )

        content_cache[name] = {
            "item_matrix": (
                item_matrix
            ),
            "parameters": (
                parameters
            ),
        }

    internal_frame = pd.DataFrame(
        internal_results
    )

    content_winner = select_best(
        internal_frame[
            internal_frame[
                "model_family"
            ]
            == (
                "secondary_tfidf_content"
            )
        ]
    )

    content_winner_name = str(
        content_winner[
            "model_name"
        ]
    )

    winner_item_matrix = (
        content_cache[
            content_winner_name
        ][
            "item_matrix"
        ]
    )

    winner_content_internal_scores = (
        content_scores(
            context=internal_context,
            item_matrix=(
                winner_item_matrix
            ),
        )
    )

    winner_content_internal_rank = (
        module.rank_normalize_scores(
            scores=(
                winner_content_internal_scores
            ),
            seen_mask=(
                internal_context.seen_mask
            ),
        )
    )

    for content_weight in [
        0.10,
        0.25,
        0.50,
        0.75,
    ]:
        hybrid_scores = (
            (
                1.0
                - content_weight
            )
            * internal_primary_rank
            + content_weight
            * winner_content_internal_rank
        )

        name = (
            "static_content_hybrid_"
            f"content_{content_weight:.2f}"
        )

        result, _ = evaluate(
            module=module,
            scores=hybrid_scores,
            context=internal_context,
            family=(
                "secondary_static_content_"
                "hybrid"
            ),
            name=name,
            parameters={
                "content_model": (
                    content_winner_name
                ),
                "content_weight": (
                    content_weight
                ),
                "primary_weight": (
                    1.0
                    - content_weight
                ),
                "normalization": (
                    "candidate_rank"
                ),
                "metadata_snapshot": (
                    "2026-06-13"
                ),
                "historically_time_safe": (
                    False
                ),
            },
            external=False,
        )

        internal_results.append(
            result
        )

    for forecast_weight in [
        0.00,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
    ]:
        reranked_scores = (
            internal_primary_rank
            + forecast_weight
            * (
                internal_forecast_signal
                - 0.5
            )
        )

        name = (
            "forecast_reranker_"
            f"weight_{forecast_weight:.2f}"
        )

        result, _ = evaluate(
            module=module,
            scores=reranked_scores,
            context=internal_context,
            family=(
                "secondary_cutoff_aligned_"
                "forecast_reranker"
            ),
            name=name,
            parameters={
                "forecast_weight": (
                    forecast_weight
                ),
                "missing_signal_value": (
                    0.5
                ),
                "forecast_signal": (
                    "within-user rank of "
                    "latest cutoff-available "
                    "frozen four-week forecast"
                ),
                "forecast_artifact_version": (
                    forecast_summary[
                        "artifact_version"
                    ]
                ),
                "historically_time_safe": (
                    True
                ),
            },
            external=False,
        )

        internal_results.append(
            result
        )

    internal_frame = pd.DataFrame(
        internal_results
    )

    internal_frame.to_csv(
        INTERNAL_RESULTS_PATH,
        index=False,
    )

    secondary_families = [
        "secondary_tfidf_content",
        "secondary_static_content_hybrid",
        (
            "secondary_cutoff_aligned_"
            "forecast_reranker"
        ),
    ]

    winner_rows = []

    for family in secondary_families:
        winner_rows.append(
            select_best(
                internal_frame[
                    internal_frame[
                        "model_family"
                    ]
                    == family
                ]
            )
        )

    winners = pd.DataFrame(
        winner_rows
    )

    winners.to_csv(
        INTERNAL_WINNERS_PATH,
        index=False,
    )

    winner_configuration = {
        row[
            "model_family"
        ]: {
            "model_name": str(
                row[
                    "model_name"
                ]
            ),
            "parameters": (
                parse_parameters(
                    row
                )
            ),
            "internal_metrics": {
                metric: float(
                    row[metric]
                )
                for metric in [
                    "ndcg_at_10",
                    "mrr",
                    "recall_at_10",
                    "ndcg_at_5",
                ]
            },
        }
        for _, row in (
            winners.iterrows()
        )
    }

    external_results = []
    rank_frames = []

    primary_result, primary_ranks = (
        evaluate(
            module=module,
            scores=(
                external_primary_scores
            ),
            context=external_context,
            family=(
                "primary_herd_mitigation"
            ),
            name=(
                primary_artifact[
                    "selected_model_name"
                ]
            ),
            parameters=(
                primary_artifact[
                    "selected_parameters"
                ]
            ),
            external=True,
        )
    )

    external_results.append(
        primary_result
    )

    rank_frames.append(
        primary_ranks
    )

    content_external_scores = (
        content_scores(
            context=external_context,
            item_matrix=(
                winner_item_matrix
            ),
        )
    )

    content_result, content_ranks = (
        evaluate(
            module=module,
            scores=content_external_scores,
            context=external_context,
            family=(
                "secondary_tfidf_content"
            ),
            name=(
                winner_configuration[
                    "secondary_tfidf_content"
                ][
                    "model_name"
                ]
            ),
            parameters=(
                winner_configuration[
                    "secondary_tfidf_content"
                ][
                    "parameters"
                ]
            ),
            external=True,
        )
    )

    external_results.append(
        content_result
    )

    rank_frames.append(
        content_ranks
    )

    content_external_rank = (
        module.rank_normalize_scores(
            scores=(
                content_external_scores
            ),
            seen_mask=(
                external_context.seen_mask
            ),
        )
    )

    static_hybrid_parameters = (
        winner_configuration[
            "secondary_static_content_hybrid"
        ][
            "parameters"
        ]
    )

    content_weight = float(
        static_hybrid_parameters[
            "content_weight"
        ]
    )

    static_hybrid_scores = (
        (
            1.0
            - content_weight
        )
        * external_primary_rank
        + content_weight
        * content_external_rank
    )

    static_result, static_ranks = (
        evaluate(
            module=module,
            scores=static_hybrid_scores,
            context=external_context,
            family=(
                "secondary_static_content_"
                "hybrid"
            ),
            name=(
                winner_configuration[
                    "secondary_static_content_hybrid"
                ][
                    "model_name"
                ]
            ),
            parameters=(
                static_hybrid_parameters
            ),
            external=True,
        )
    )

    external_results.append(
        static_result
    )

    rank_frames.append(
        static_ranks
    )

    forecast_parameters = (
        winner_configuration[
            "secondary_cutoff_aligned_forecast_reranker"
        ][
            "parameters"
        ]
    )

    forecast_weight = float(
        forecast_parameters[
            "forecast_weight"
        ]
    )

    forecast_scores = (
        external_primary_rank
        + forecast_weight
        * (
            external_forecast_signal
            - 0.5
        )
    )

    forecast_result, forecast_ranks = (
        evaluate(
            module=module,
            scores=forecast_scores,
            context=external_context,
            family=(
                "secondary_cutoff_aligned_"
                "forecast_reranker"
            ),
            name=(
                winner_configuration[
                    "secondary_cutoff_aligned_forecast_reranker"
                ][
                    "model_name"
                ]
            ),
            parameters=(
                forecast_parameters
            ),
            external=True,
        )
    )

    external_results.append(
        forecast_result
    )

    rank_frames.append(
        forecast_ranks
    )

    external_frame = pd.DataFrame(
        external_results
    )

    external_frame.to_csv(
        EXTERNAL_RESULTS_PATH,
        index=False,
    )

    bootstrap = bootstrap_comparisons(
        rank_frames=rank_frames,
        primary_family=(
            "primary_herd_mitigation"
        ),
    )

    bootstrap.to_csv(
        BOOTSTRAP_PATH,
        index=False,
    )

    internal_coverage_summary = {
        "stage": (
            "internal_tuning"
        ),
        "users": int(
            len(
                internal_forecast_coverage
            )
        ),
        "positive_signal_fraction": float(
            internal_forecast_coverage[
                "positive_has_signal"
            ].mean()
        ),
        "mean_candidate_signal_fraction": float(
            internal_forecast_coverage[
                "candidate_signal_fraction"
            ].mean()
        ),
        "median_candidate_signal_fraction": float(
            internal_forecast_coverage[
                "candidate_signal_fraction"
            ].median()
        ),
        "zero_candidate_signal_users": int(
            (
                internal_forecast_coverage[
                    "available_candidate_repositories"
                ]
                == 0
            ).sum()
        ),
        "complete_candidate_signal_users": int(
            (
                internal_forecast_coverage[
                    "available_candidate_repositories"
                ]
                == internal_forecast_coverage[
                    "candidate_repositories"
                ]
            ).sum()
        ),
    }

    external_coverage_summary = {
        "stage": (
            "strict_external_validation"
        ),
        "users": int(
            len(
                external_forecast_coverage
            )
        ),
        "positive_signal_fraction": float(
            external_forecast_coverage[
                "positive_has_signal"
            ].mean()
        ),
        "mean_candidate_signal_fraction": float(
            external_forecast_coverage[
                "candidate_signal_fraction"
            ].mean()
        ),
        "median_candidate_signal_fraction": float(
            external_forecast_coverage[
                "candidate_signal_fraction"
            ].median()
        ),
        "zero_candidate_signal_users": int(
            (
                external_forecast_coverage[
                    "available_candidate_repositories"
                ]
                == 0
            ).sum()
        ),
        "complete_candidate_signal_users": int(
            (
                external_forecast_coverage[
                    "available_candidate_repositories"
                ]
                == external_forecast_coverage[
                    "candidate_repositories"
                ]
            ).sum()
        ),
    }

    pd.DataFrame(
        [
            internal_coverage_summary,
            external_coverage_summary,
        ]
    ).to_csv(
        FORECAST_COVERAGE_PATH,
        index=False,
    )

    primary_row = external_frame[
        external_frame[
            "model_family"
        ]
        == (
            "primary_herd_mitigation"
        )
    ].iloc[0]

    branch_comparisons = {}

    for _, row in (
        external_frame[
            external_frame[
                "model_family"
            ]
            != (
                "primary_herd_mitigation"
            )
        ].iterrows()
    ):
        branch_comparisons[
            row[
                "model_family"
            ]
        ] = {
            "model_name": (
                row[
                    "model_name"
                ]
            ),
            "ndcg_at_10": float(
                row[
                    "ndcg_at_10"
                ]
            ),
            "mrr": float(
                row[
                    "mrr"
                ]
            ),
            "recall_at_10": float(
                row[
                    "recall_at_10"
                ]
            ),
            "ndcg_at_10_minus_primary": float(
                row[
                    "ndcg_at_10"
                ]
                - primary_row[
                    "ndcg_at_10"
                ]
            ),
            "mrr_minus_primary": float(
                row[
                    "mrr"
                ]
                - primary_row[
                    "mrr"
                ]
            ),
            "recall_at_10_minus_primary": float(
                row[
                    "recall_at_10"
                ]
                - primary_row[
                    "recall_at_10"
                ]
            ),
            "catalog_coverage_at_10": float(
                row[
                    "catalog_coverage_at_10"
                ]
            ),
            "long_tail_share_at_10": float(
                row[
                    "long_tail_share_at_10"
                ]
            ),
        }

    summary = {
        "status": (
            "secondary_recommendation_"
            "branches_complete"
        ),
        "created_at_commit": git_output(
            [
                "rev-parse",
                "HEAD",
            ]
        ),
        "test_data_status": (
            "header_only_not_loaded_or_"
            "evaluated"
        ),
        "test_header_columns": (
            test_header_columns
        ),
        "primary_model": {
            "family": (
                primary_artifact[
                    "selected_family"
                ]
            ),
            "model_name": (
                primary_artifact[
                    "selected_model_name"
                ]
            ),
            "artifact_sha256": (
                primary_artifact_hash
            ),
            "external_metric_reproduction_"
            "maximum_absolute_difference": (
                max(
                    primary_metric_differences.values()
                )
            ),
            "eligibility": (
                "frozen primary model; "
                "secondary branches do not "
                "replace it"
            ),
        },
        "static_content": {
            "catalog_metadata_rows": int(
                len(metadata)
            ),
            "metadata_collection_minimum": str(
                pd.to_datetime(
                    metadata[
                        "collected_at"
                    ],
                    utc=True,
                    errors="coerce",
                ).min()
            ),
            "metadata_collection_maximum": str(
                pd.to_datetime(
                    metadata[
                        "collected_at"
                    ],
                    utc=True,
                    errors="coerce",
                ).max()
            ),
            "historically_time_safe": (
                False
            ),
            "interpretation": (
                "exploratory secondary "
                "analysis only because the "
                "metadata is a June 2026 "
                "snapshot"
            ),
        },
        "forecast_signal": {
            **forecast_summary,
            "historically_time_safe": (
                True
            ),
            "missing_signal_policy": (
                "neutral value 0.5, producing "
                "zero reranking adjustment"
            ),
            "internal_coverage": (
                internal_coverage_summary
            ),
            "external_coverage": (
                external_coverage_summary
            ),
        },
        "internal_family_winners": (
            winner_configuration
        ),
        "external_results": (
            external_frame.to_dict(
                orient="records"
            )
        ),
        "branch_comparisons_to_primary": (
            branch_comparisons
        ),
        "paired_bootstrap": (
            bootstrap.to_dict(
                orient="records"
            )
        ),
        "selection_policy": (
            "Tune each secondary family on "
            "the frozen internal temporal "
            "holdout. Evaluate one winner per "
            "family on strict external "
            "validation. Secondary results are "
            "reported but do not alter the "
            "frozen primary winner."
        ),
        "output_paths": {
            "internal_results": str(
                INTERNAL_RESULTS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "internal_winners": str(
                INTERNAL_WINNERS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "external_results": str(
                EXTERNAL_RESULTS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "bootstrap": str(
                BOOTSTRAP_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "forecast_coverage": str(
                FORECAST_COVERAGE_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
    }

    write_json(
        SUMMARY_PATH,
        summary,
    )

    save_bar_plot(
        frame=external_frame,
        metric="ndcg_at_10",
        path=NDCG_PLOT_PATH,
        title=(
            "Primary and Secondary "
            "Recommendation NDCG@10"
        ),
    )

    save_bar_plot(
        frame=external_frame,
        metric=(
            "catalog_coverage_at_10"
        ),
        path=COVERAGE_PLOT_PATH,
        title=(
            "Primary and Secondary "
            "Recommendation Catalog "
            "Coverage@10"
        ),
    )

    print("=" * 112)
    print(
        "SECONDARY RECOMMENDATION "
        "BRANCHES"
    )
    print("=" * 112)

    print()
    print(
        "Internal family winners:"
    )

    print(
        winners[
            [
                "model_family",
                "model_name",
                "ndcg_at_10",
                "mrr",
                "recall_at_10",
                "ndcg_at_5",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 112)
    print(
        "STRICT EXTERNAL VALIDATION"
    )
    print("=" * 112)

    print(
        external_frame[
            [
                "model_family",
                "model_name",
                "ndcg_at_10",
                "mrr",
                "recall_at_10",
                "ndcg_at_5",
                "catalog_coverage_at_10",
                "novelty_at_10",
                "intra_list_diversity_at_10",
                "average_recommended_popularity_at_10",
                "long_tail_share_at_10",
            ]
        ]
        .sort_values(
            [
                "ndcg_at_10",
                "mrr",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 112)
    print(
        "FORECAST SIGNAL COVERAGE"
    )
    print("=" * 112)

    print(
        pd.DataFrame(
            [
                internal_coverage_summary,
                external_coverage_summary,
            ]
        ).to_string(
            index=False
        )
    )

    print()
    print("=" * 112)
    print(
        "PAIRED BOOTSTRAP: "
        "SECONDARY MINUS PRIMARY"
    )
    print("=" * 112)

    print(
        bootstrap.to_string(
            index=False
        )
    )

    print()
    print(
        "Primary model remains frozen. "
        "Static content is exploratory and "
        "not historically time-safe."
    )

    print(
        "Recommendation test values were "
        "not loaded or evaluated."
    )


if __name__ == "__main__":
    main()
