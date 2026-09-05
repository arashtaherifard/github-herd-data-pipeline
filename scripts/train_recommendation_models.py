import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


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

SCHEMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "recommendation"
    / "recommendation_schema.json"
)

DESIGN_MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "recommendation"
    / "design_freeze"
    / "recommendation_design_freeze_manifest.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "recommendation"
    / "model_search"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
)

INTERNAL_RESULTS_PATH = (
    OUTPUT_DIR
    / "internal_tuning_results.csv"
)

INTERNAL_WINNERS_PATH = (
    OUTPUT_DIR
    / "internal_family_winners.csv"
)

EXTERNAL_RESULTS_PATH = (
    OUTPUT_DIR
    / "external_validation_family_results.csv"
)

EXTERNAL_RANKS_PATH = (
    OUTPUT_DIR
    / "external_validation_per_user_ranks.csv"
)

SELECTION_SUMMARY_PATH = (
    OUTPUT_DIR
    / "recommendation_model_selection_summary.json"
)

DEVELOPMENT_METADATA_PATH = (
    MODEL_DIR
    / "selected_recommender_development_metadata.json"
)

DEVELOPMENT_ARTIFACT_PATH = (
    MODEL_DIR
    / "selected_recommender_development.joblib"
)

RANDOM_STATE = 42
TOP_K = 10
EPSILON = 1e-12


@dataclass
class EvaluationContext:
    fit_interactions: pd.DataFrame
    holdout: pd.DataFrame
    catalog: pd.DataFrame
    repo_ids: np.ndarray
    repo_names: np.ndarray
    repo_to_index: dict[int, int]
    eval_user_ids: np.ndarray
    eval_user_to_row: dict[int, int]
    history_matrix: sparse.csr_matrix
    seen_mask: np.ndarray
    positive_indices: np.ndarray
    item_counts: np.ndarray
    popularity_share: np.ndarray
    popularity_novelty: np.ndarray
    long_tail_mask: np.ndarray
    interaction_cosine: np.ndarray
    fit_matrix_all_users: sparse.csr_matrix
    fit_user_ids: np.ndarray


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


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def verify_design_freeze() -> dict:
    manifest = load_json(
        DESIGN_MANIFEST_PATH
    )

    checks = {
        manifest["metadata_path"]: (
            manifest["metadata_sha256"]
        ),
        **manifest["input_hashes"],
        **manifest["artifact_hashes"],
    }

    mismatches = []

    for relative_path, expected in checks.items():
        path = (
            PROJECT_ROOT
            / relative_path
        )

        actual = sha256_file(path)

        if actual != expected:
            mismatches.append(
                {
                    "path": relative_path,
                    "expected": expected,
                    "actual": actual,
                }
            )

    if mismatches:
        raise RuntimeError(
            "Recommendation design-freeze "
            "verification failed:\n"
            + json.dumps(
                mismatches,
                indent=2,
            )
        )

    if (
        manifest[
            "test_data_status"
        ]
        != (
            "header_only_not_loaded_or_"
            "evaluated"
        )
    ):
        raise RuntimeError(
            "Recommendation test status is "
            "not sealed."
        )

    return manifest


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
        "interaction",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"{path} is missing columns: "
            f"{sorted(missing)}"
        )

    frame["user_id"] = pd.to_numeric(
        frame["user_id"],
        errors="raise",
    ).astype("int64")

    frame["repo_id"] = pd.to_numeric(
        frame["repo_id"],
        errors="raise",
    ).astype("int64")

    frame["starred_at"] = pd.to_datetime(
        frame["starred_at"],
        utc=True,
        errors="raise",
    )

    frame["interaction"] = pd.to_numeric(
        frame["interaction"],
        errors="raise",
    )

    if not (
        frame["interaction"]
        == 1
    ).all():
        raise ValueError(
            f"{path} contains non-positive "
            "implicit interactions."
        )

    return frame


def build_catalog(
    train: pd.DataFrame,
) -> pd.DataFrame:
    catalog = (
        train[
            [
                "repo_id",
                "repo_full_name",
            ]
        ]
        .drop_duplicates(
            "repo_id"
        )
        .sort_values(
            "repo_id"
        )
        .reset_index(drop=True)
    )

    if len(catalog) != 145:
        raise RuntimeError(
            "Unexpected recommendation "
            f"catalog size: {len(catalog)}"
        )

    return catalog


def build_sparse_matrix(
    interactions: pd.DataFrame,
    user_ids: np.ndarray,
    repo_to_index: dict[int, int],
) -> sparse.csr_matrix:
    user_to_row = {
        int(user_id): index
        for index, user_id in enumerate(
            user_ids
        )
    }

    selected = interactions[
        interactions[
            "user_id"
        ].isin(
            user_to_row
        )
    ]

    row_indices = (
        selected[
            "user_id"
        ]
        .map(
            user_to_row
        )
        .to_numpy(
            dtype=np.int64
        )
    )

    column_indices = (
        selected[
            "repo_id"
        ]
        .map(
            repo_to_index
        )
        .to_numpy(
            dtype=np.int64
        )
    )

    values = np.ones(
        len(selected),
        dtype=np.float64,
    )

    matrix = sparse.csr_matrix(
        (
            values,
            (
                row_indices,
                column_indices,
            ),
        ),
        shape=(
            len(user_ids),
            len(repo_to_index),
        ),
        dtype=np.float64,
    )

    matrix.data[:] = 1.0

    return matrix


def build_context(
    fit_interactions: pd.DataFrame,
    holdout: pd.DataFrame,
    catalog: pd.DataFrame,
    holdout_repo_column: str,
) -> EvaluationContext:
    repo_ids = catalog[
        "repo_id"
    ].to_numpy(
        dtype=np.int64
    )

    repo_names = catalog[
        "repo_full_name"
    ].astype(str).to_numpy()

    repo_to_index = {
        int(repo_id): index
        for index, repo_id in enumerate(
            repo_ids
        )
    }

    eval_user_ids = (
        holdout[
            "user_id"
        ]
        .astype("int64")
        .sort_values()
        .to_numpy()
    )

    eval_user_to_row = {
        int(user_id): index
        for index, user_id in enumerate(
            eval_user_ids
        )
    }

    holdout_sorted = (
        holdout.set_index(
            "user_id"
        )
        .loc[
            eval_user_ids
        ]
        .reset_index()
    )

    positive_indices = (
        holdout_sorted[
            holdout_repo_column
        ]
        .map(
            repo_to_index
        )
        .to_numpy(
            dtype=np.int64
        )
    )

    if np.any(
        pd.isna(
            positive_indices
        )
    ):
        raise RuntimeError(
            "At least one held-out repository "
            "is absent from the catalog."
        )

    history_matrix = build_sparse_matrix(
        interactions=fit_interactions,
        user_ids=eval_user_ids,
        repo_to_index=repo_to_index,
    )

    seen_mask = (
        history_matrix.toarray()
        > 0
    )

    if np.any(
        seen_mask[
            np.arange(
                len(eval_user_ids)
            ),
            positive_indices,
        ]
    ):
        raise RuntimeError(
            "Held-out positives overlap "
            "fitting histories."
        )

    fit_user_ids = np.sort(
        fit_interactions[
            "user_id"
        ]
        .astype("int64")
        .unique()
    )

    fit_matrix_all_users = (
        build_sparse_matrix(
            interactions=fit_interactions,
            user_ids=fit_user_ids,
            repo_to_index=repo_to_index,
        )
    )

    item_counts = np.asarray(
        fit_matrix_all_users.sum(
            axis=0
        )
    ).ravel()

    smoothed_share = (
        item_counts
        + 1.0
    ) / (
        item_counts.sum()
        + len(item_counts)
    )

    popularity_novelty = (
        -np.log2(
            smoothed_share
        )
    )

    popularity_order = (
        np.argsort(
            -item_counts,
            kind="stable",
        )
    )

    top_head_count = max(
        1,
        int(
            math.ceil(
                0.20
                * len(item_counts)
            )
        ),
    )

    long_tail_mask = np.ones(
        len(item_counts),
        dtype=bool,
    )

    long_tail_mask[
        popularity_order[
            :top_head_count
        ]
    ] = False

    cooccurrence = (
        fit_matrix_all_users.T
        @ fit_matrix_all_users
    ).toarray().astype(
        np.float64
    )

    norm = np.sqrt(
        np.maximum(
            np.diag(
                cooccurrence
            ),
            EPSILON,
        )
    )

    interaction_cosine = (
        cooccurrence
        / np.outer(
            norm,
            norm,
        )
    )

    np.fill_diagonal(
        interaction_cosine,
        0.0,
    )

    return EvaluationContext(
        fit_interactions=(
            fit_interactions
        ),
        holdout=holdout_sorted,
        catalog=catalog,
        repo_ids=repo_ids,
        repo_names=repo_names,
        repo_to_index=repo_to_index,
        eval_user_ids=eval_user_ids,
        eval_user_to_row=(
            eval_user_to_row
        ),
        history_matrix=history_matrix,
        seen_mask=seen_mask,
        positive_indices=(
            positive_indices
        ),
        item_counts=item_counts,
        popularity_share=(
            smoothed_share
        ),
        popularity_novelty=(
            popularity_novelty
        ),
        long_tail_mask=(
            long_tail_mask
        ),
        interaction_cosine=(
            interaction_cosine
        ),
        fit_matrix_all_users=(
            fit_matrix_all_users
        ),
        fit_user_ids=fit_user_ids,
    )


def apply_candidate_mask(
    scores: np.ndarray,
    seen_mask: np.ndarray,
) -> np.ndarray:
    masked = np.asarray(
        scores,
        dtype=np.float64,
    ).copy()

    masked[
        ~np.isfinite(masked)
    ] = -np.inf

    masked[
        seen_mask
    ] = -np.inf

    return masked


def stable_top_order(
    masked_scores: np.ndarray,
) -> np.ndarray:
    return np.argsort(
        -masked_scores,
        axis=1,
        kind="stable",
    )


def rank_normalize_scores(
    scores: np.ndarray,
    seen_mask: np.ndarray,
) -> np.ndarray:
    masked = apply_candidate_mask(
        scores,
        seen_mask,
    )

    order = stable_top_order(
        masked
    )

    ranks = np.empty_like(
        order,
        dtype=np.int32,
    )

    row_indices = np.arange(
        len(order)
    )[:, None]

    ranks[
        row_indices,
        order,
    ] = np.arange(
        order.shape[1],
        dtype=np.int32,
    )[None, :]

    candidate_counts = (
        (~seen_mask).sum(
            axis=1
        )
    ).astype(
        np.float64
    )

    denominator = np.maximum(
        candidate_counts
        - 1.0,
        1.0,
    )

    normalized = (
        1.0
        - (
            ranks.astype(
                np.float64
            )
            / denominator[:, None]
        )
    )

    normalized[
        seen_mask
    ] = 0.0

    return normalized


def mean_intra_list_diversity(
    top_indices: np.ndarray,
    similarity: np.ndarray,
) -> float:
    if top_indices.shape[1] < 2:
        return 0.0

    triangle = np.triu_indices(
        top_indices.shape[1],
        k=1,
    )

    values = []

    for row in top_indices:
        pair_similarity = similarity[
            np.ix_(
                row,
                row,
            )
        ][triangle]

        values.append(
            float(
                np.mean(
                    1.0
                    - pair_similarity
                )
            )
        )

    return float(
        np.mean(values)
    )


def evaluate_scores(
    scores: np.ndarray,
    context: EvaluationContext,
    model_family: str,
    model_name: str,
    parameters: dict[str, Any],
    return_user_ranks: bool = False,
    compute_beyond_accuracy: bool = False,
) -> tuple[
    dict[str, Any],
    pd.DataFrame | None,
]:
    masked = apply_candidate_mask(
        scores=scores,
        seen_mask=context.seen_mask,
    )

    order = stable_top_order(
        masked
    )

    ranks_zero = np.empty_like(
        order,
        dtype=np.int32,
    )

    row_indices = np.arange(
        len(order)
    )[:, None]

    ranks_zero[
        row_indices,
        order,
    ] = np.arange(
        order.shape[1],
        dtype=np.int32,
    )[None, :]

    positive_ranks = (
        ranks_zero[
            np.arange(
                len(
                    context.eval_user_ids
                )
            ),
            context.positive_indices,
        ]
        + 1
    )

    top5 = order[
        :,
        :5,
    ]

    top10 = order[
        :,
        :10,
    ]

    hit5 = positive_ranks <= 5
    hit10 = positive_ranks <= 10

    ndcg5 = np.where(
        hit5,
        1.0
        / np.log2(
            positive_ranks
            + 1.0
        ),
        0.0,
    )

    ndcg10 = np.where(
        hit10,
        1.0
        / np.log2(
            positive_ranks
            + 1.0
        ),
        0.0,
    )

    reciprocal_rank = (
        1.0
        / positive_ranks
    )

    map10 = np.where(
        hit10,
        reciprocal_rank,
        0.0,
    )

    if compute_beyond_accuracy:
        unique_top10 = np.unique(
            top10
        )

        catalog_coverage = (
            len(unique_top10)
            / len(
                context.repo_ids
            )
        )

        novelty = float(
            np.mean(
                context.popularity_novelty[
                    top10
                ]
            )
        )

        average_popularity = float(
            np.mean(
                context.item_counts[
                    top10
                ]
            )
        )

        long_tail_share = float(
            np.mean(
                context.long_tail_mask[
                    top10
                ]
            )
        )

        diversity = (
            mean_intra_list_diversity(
                top_indices=top10,
                similarity=(
                    context.interaction_cosine
                ),
            )
        )
    else:
        catalog_coverage = float("nan")
        novelty = float("nan")
        average_popularity = float("nan")
        long_tail_share = float("nan")
        diversity = float("nan")

    result = {
        "model_family": model_family,
        "model_name": model_name,
        "parameters_json": json.dumps(
            parameters,
            sort_keys=True,
        ),
        "evaluation_users": int(
            len(
                context.eval_user_ids
            )
        ),
        "precision_at_5": float(
            np.mean(
                hit5
            )
            / 5.0
        ),
        "precision_at_10": float(
            np.mean(
                hit10
            )
            / 10.0
        ),
        "recall_at_5": float(
            np.mean(
                hit5
            )
        ),
        "recall_at_10": float(
            np.mean(
                hit10
            )
        ),
        "hit_rate_at_5": float(
            np.mean(
                hit5
            )
        ),
        "hit_rate_at_10": float(
            np.mean(
                hit10
            )
        ),
        "ndcg_at_5": float(
            np.mean(
                ndcg5
            )
        ),
        "ndcg_at_10": float(
            np.mean(
                ndcg10
            )
        ),
        "mrr": float(
            np.mean(
                reciprocal_rank
            )
        ),
        "map_at_10": float(
            np.mean(
                map10
            )
        ),
        "median_positive_rank": float(
            np.median(
                positive_ranks
            )
        ),
        "mean_positive_rank": float(
            np.mean(
                positive_ranks
            )
        ),
        "catalog_coverage_at_10": float(
            catalog_coverage
        ),
        "novelty_at_10": novelty,
        "intra_list_diversity_at_10": (
            diversity
        ),
        "average_recommended_popularity_at_10": (
            average_popularity
        ),
        "long_tail_share_at_10": (
            long_tail_share
        ),
    }

    user_ranks = None

    if return_user_ranks:
        user_ranks = pd.DataFrame(
            {
                "user_id": (
                    context.eval_user_ids
                ),
                "positive_repo_id": (
                    context.repo_ids[
                        context.positive_indices
                    ]
                ),
                "positive_rank": (
                    positive_ranks
                ),
                "hit_at_5": (
                    hit5.astype(int)
                ),
                "hit_at_10": (
                    hit10.astype(int)
                ),
                "ndcg_at_10": (
                    ndcg10
                ),
                "reciprocal_rank": (
                    reciprocal_rank
                ),
                "model_family": (
                    model_family
                ),
                "model_name": (
                    model_name
                ),
            }
        )

    return result, user_ranks


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


def build_item_similarity(
    context: EvaluationContext,
    shrinkage: float,
) -> np.ndarray:
    cooccurrence = (
        context.fit_matrix_all_users.T
        @ context.fit_matrix_all_users
    ).toarray().astype(
        np.float64
    )

    diagonal = np.diag(
        cooccurrence
    )

    denominator = np.sqrt(
        np.maximum(
            diagonal,
            EPSILON,
        )[:, None]
        * np.maximum(
            diagonal,
            EPSILON,
        )[None, :]
    )

    cosine = (
        cooccurrence
        / denominator
    )

    if shrinkage > 0:
        cosine *= (
            cooccurrence
            / (
                cooccurrence
                + shrinkage
            )
        )

    np.fill_diagonal(
        cosine,
        0.0,
    )

    return cosine


def item_item_scores(
    context: EvaluationContext,
    similarity: np.ndarray,
    aggregation: str,
) -> np.ndarray:
    scores = (
        context.history_matrix
        @ similarity
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if aggregation == "mean":
        history_length = np.asarray(
            context.history_matrix.sum(
                axis=1
            )
        ).ravel()

        scores = (
            scores
            / np.maximum(
                history_length,
                1.0,
            )[:, None]
        )
    elif aggregation != "sum":
        raise ValueError(
            f"Unknown aggregation: "
            f"{aggregation}"
        )

    return scores


def build_ppmi_matrix(
    context: EvaluationContext,
) -> np.ndarray:
    x = context.fit_matrix_all_users

    cooccurrence = (
        x.T
        @ x
    ).toarray().astype(
        np.float64
    )

    n_users = float(
        x.shape[0]
    )

    item_probability = (
        np.diag(
            cooccurrence
        )
        / n_users
    )

    joint_probability = (
        cooccurrence
        / n_users
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


def graph_propagation_matrix(
    edge_matrix: np.ndarray,
    restart_probability: float,
) -> np.ndarray:
    row_sum = edge_matrix.sum(
        axis=1
    )

    transition = np.divide(
        edge_matrix,
        row_sum[:, None],
        out=np.zeros_like(
            edge_matrix,
            dtype=np.float64,
        ),
        where=(
            row_sum[:, None] > 0
        ),
    )

    identity = np.eye(
        transition.shape[0],
        dtype=np.float64,
    )

    propagation = (
        restart_probability
        * np.linalg.inv(
            identity
            - (
                1.0
                - restart_probability
            )
            * transition
        )
    )

    return propagation


def graph_scores(
    context: EvaluationContext,
    propagation: np.ndarray,
) -> np.ndarray:
    seed = context.history_matrix.copy()

    row_sum = np.asarray(
        seed.sum(
            axis=1
        )
    ).ravel()

    inverse = np.divide(
        1.0,
        row_sum,
        out=np.zeros_like(
            row_sum,
            dtype=np.float64,
        ),
        where=(
            row_sum > 0
        ),
    )

    seed = sparse.diags(
        inverse
    ) @ seed

    return np.asarray(
        seed
        @ propagation,
        dtype=np.float64,
    )


def svd_scores(
    context: EvaluationContext,
    n_components: int,
    n_iter: int,
) -> tuple[
    np.ndarray,
    TruncatedSVD,
]:
    model = TruncatedSVD(
        n_components=n_components,
        n_iter=n_iter,
        random_state=RANDOM_STATE,
    )

    model.fit(
        context.fit_matrix_all_users
    )

    user_latent = model.transform(
        context.history_matrix
    )

    scores = (
        user_latent
        @ model.components_
    )

    return (
        np.asarray(
            scores,
            dtype=np.float64,
        ),
        model,
    )


def random_scores(
    context: EvaluationContext,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(
        seed
    )

    return generator.random(
        (
            len(
                context.eval_user_ids
            ),
            len(
                context.repo_ids
            ),
        )
    )


def popularity_scores(
    context: EvaluationContext,
) -> np.ndarray:
    return np.broadcast_to(
        context.item_counts[
            None,
            :,
        ],
        (
            len(
                context.eval_user_ids
            ),
            len(
                context.repo_ids
            ),
        ),
    ).astype(
        np.float64,
        copy=True,
    )


def convex_weight_grid(
    step: float = 0.25,
) -> list[tuple[float, float, float, float]]:
    units = int(
        round(
            1.0
            / step
        )
    )

    combinations = []

    for a in range(
        units + 1
    ):
        for b in range(
            units + 1
            - a
        ):
            for c in range(
                units + 1
                - a
                - b
            ):
                d = (
                    units
                    - a
                    - b
                    - c
                )

                weights = (
                    a / units,
                    b / units,
                    c / units,
                    d / units,
                )

                nonzero = sum(
                    weight > 0
                    for weight in weights
                )

                if nonzero >= 2:
                    combinations.append(
                        weights
                    )

    return combinations


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
            "Unexpected number of removed "
            "internal holdout rows."
        )

    return internal_fit


def evaluate_random_family(
    context: EvaluationContext,
    seeds: list[int],
    stage: str,
    compute_beyond_accuracy: bool,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    rows = []

    for seed in seeds:
        scores = random_scores(
            context=context,
            seed=seed,
        )

        result, _ = evaluate_scores(
            scores=scores,
            context=context,
            model_family=(
                "random_sanity"
            ),
            model_name=(
                f"random_seed_{seed}"
            ),
            parameters={
                "seed": seed,
                "stage": stage,
            },
            compute_beyond_accuracy=(
                compute_beyond_accuracy
            ),
        )

        rows.append(result)

    frame = pd.DataFrame(rows)

    numeric_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(
            frame[column]
        )
    ]

    averaged = {
        "model_family": (
            "random_sanity"
        ),
        "model_name": (
            "random_seed_average"
        ),
        "parameters_json": json.dumps(
            {
                "seeds": seeds,
                "aggregation": (
                    "mean_metrics"
                ),
                "stage": stage,
            },
            sort_keys=True,
        ),
    }

    for column in numeric_columns:
        averaged[column] = float(
            frame[column].mean()
        )

    averaged[
        "random_ndcg_at_10_sd"
    ] = float(
        frame[
            "ndcg_at_10"
        ].std(
            ddof=1
        )
    )

    return averaged, rows


def component_score_bundle(
    context: EvaluationContext,
    winners: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
]:
    artifacts = {}

    item_parameters = winners[
        "item_item_cosine"
    ][
        "parameters"
    ]

    item_similarity = (
        build_item_similarity(
            context=context,
            shrinkage=float(
                item_parameters[
                    "shrinkage"
                ]
            ),
        )
    )

    item_scores = item_item_scores(
        context=context,
        similarity=item_similarity,
        aggregation=item_parameters[
            "aggregation"
        ],
    )

    svd_parameters = winners[
        "truncated_svd"
    ][
        "parameters"
    ]

    svd_score_matrix, svd_model = (
        svd_scores(
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

    if (
        graph_parameters[
            "edge_weight"
        ]
        == "cosine"
    ):
        edge_matrix = (
            context.interaction_cosine
        )
    elif (
        graph_parameters[
            "edge_weight"
        ]
        == "ppmi"
    ):
        edge_matrix = (
            build_ppmi_matrix(
                context
            )
        )
    else:
        raise ValueError(
            "Unknown graph edge weight."
        )

    propagation = (
        graph_propagation_matrix(
            edge_matrix=edge_matrix,
            restart_probability=float(
                graph_parameters[
                    "restart_probability"
                ]
            ),
        )
    )

    graph_score_matrix = (
        graph_scores(
            context=context,
            propagation=propagation,
        )
    )

    popularity_score_matrix = (
        popularity_scores(
            context
        )
    )

    score_bundle = {
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

    artifacts.update(
        {
            "item_similarity": (
                item_similarity
            ),
            "svd_components": (
                svd_model.components_
            ),
            "svd_explained_variance_ratio": (
                svd_model.explained_variance_ratio_
            ),
            "graph_propagation": (
                propagation
            ),
        }
    )

    return score_bundle, artifacts


def parse_parameters(
    row: pd.Series,
) -> dict[str, Any]:
    return json.loads(
        row[
            "parameters_json"
        ]
    )


def save_metric_plot(
    results: pd.DataFrame,
    metric: str,
    path: Path,
    title: str,
) -> None:
    ordered = results.sort_values(
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
            "Recommendation model-search "
            "outputs already exist. "
            "Refusing to overwrite:\n"
            f"{OUTPUT_DIR}"
        )

    if DEVELOPMENT_ARTIFACT_PATH.exists():
        raise RuntimeError(
            "Development recommender artifact "
            "already exists. Refusing to "
            "overwrite it."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    design_manifest = (
        verify_design_freeze()
    )

    schema = load_json(
        SCHEMA_PATH
    )

    if (
        schema[
            "test_data_status"
        ]
        != (
            "header_only_not_loaded_or_"
            "evaluated"
        )
    ):
        raise RuntimeError(
            "Recommendation schema does not "
            "preserve the sealed test state."
        )

    test_header = pd.read_csv(
        TEST_PATH,
        nrows=0,
    )

    expected_test_columns = {
        "user_id",
        "repo_id",
        "starred_at",
    }

    if (
        expected_test_columns
        - set(
            test_header.columns
        )
    ):
        raise RuntimeError(
            "Recommendation test header "
            "does not match expectations."
        )

    train = load_interactions(
        TRAIN_PATH
    )

    validation = load_interactions(
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
        assignments["user_id"],
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
        primary_validation["user_id"],
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

    catalog = build_catalog(
        train
    )

    internal_fit = (
        construct_internal_fit(
            train=train,
            assignments=assignments,
        )
    )

    internal_context = (
        build_context(
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

    internal_results = []
    internal_score_cache = {}
    internal_artifact_cache = {}

    random_average, random_rows = (
        evaluate_random_family(
            context=internal_context,
            seeds=[
                11,
                29,
                42,
                71,
                101,
            ],
            stage="internal_tuning",
            compute_beyond_accuracy=False,
        )
    )

    internal_results.extend(
        random_rows
    )

    internal_results.append(
        random_average
    )

    popularity_score_matrix = (
        popularity_scores(
            internal_context
        )
    )

    popularity_result, _ = (
        evaluate_scores(
            scores=(
                popularity_score_matrix
            ),
            context=internal_context,
            model_family=(
                "training_popularity"
            ),
            model_name=(
                "training_popularity"
            ),
            parameters={
                "score": (
                    "internal_training_"
                    "interaction_count"
                )
            },
        )
    )

    internal_results.append(
        popularity_result
    )

    for aggregation in [
        "sum",
        "mean",
    ]:
        for shrinkage in [
            0.0,
            10.0,
            50.0,
            100.0,
        ]:
            similarity = (
                build_item_similarity(
                    context=(
                        internal_context
                    ),
                    shrinkage=shrinkage,
                )
            )

            scores = item_item_scores(
                context=internal_context,
                similarity=similarity,
                aggregation=aggregation,
            )

            name = (
                f"item_cosine_"
                f"{aggregation}_"
                f"shrink_{shrinkage:g}"
            )

            result, _ = evaluate_scores(
                scores=scores,
                context=internal_context,
                model_family=(
                    "item_item_cosine"
                ),
                model_name=name,
                parameters={
                    "aggregation": (
                        aggregation
                    ),
                    "shrinkage": (
                        shrinkage
                    ),
                },
            )

            internal_results.append(
                result
            )

            internal_score_cache[
                name
            ] = scores

            internal_artifact_cache[
                name
            ] = {
                "item_similarity": (
                    similarity
                )
            }

    for n_components in [
        16,
        32,
        64,
        96,
    ]:
        for n_iter in [
            7,
            15,
        ]:
            scores, model = (
                svd_scores(
                    context=(
                        internal_context
                    ),
                    n_components=(
                        n_components
                    ),
                    n_iter=n_iter,
                )
            )

            name = (
                f"svd_{n_components}_"
                f"iter_{n_iter}"
            )

            result, _ = evaluate_scores(
                scores=scores,
                context=internal_context,
                model_family=(
                    "truncated_svd"
                ),
                model_name=name,
                parameters={
                    "n_components": (
                        n_components
                    ),
                    "n_iter": (
                        n_iter
                    ),
                },
            )

            internal_results.append(
                result
            )

            internal_score_cache[
                name
            ] = scores

            internal_artifact_cache[
                name
            ] = {
                "svd_components": (
                    model.components_
                ),
                "svd_explained_"
                "variance_ratio": (
                    model.explained_variance_ratio_
                ),
            }

    graph_edges = {
        "cosine": (
            internal_context[
                "interaction_cosine"
            ]
            if isinstance(
                internal_context,
                dict,
            )
            else (
                internal_context
                .interaction_cosine
            )
        ),
        "ppmi": build_ppmi_matrix(
            internal_context
        ),
    }

    for (
        edge_weight,
        edge_matrix,
    ) in graph_edges.items():
        for restart_probability in [
            0.15,
            0.30,
            0.50,
        ]:
            propagation = (
                graph_propagation_matrix(
                    edge_matrix=edge_matrix,
                    restart_probability=(
                        restart_probability
                    ),
                )
            )

            scores = graph_scores(
                context=internal_context,
                propagation=propagation,
            )

            name = (
                f"ppr_{edge_weight}_"
                f"restart_"
                f"{restart_probability:.2f}"
            )

            result, _ = evaluate_scores(
                scores=scores,
                context=internal_context,
                model_family=(
                    "graph_personalized_"
                    "pagerank"
                ),
                model_name=name,
                parameters={
                    "edge_weight": (
                        edge_weight
                    ),
                    "restart_probability": (
                        restart_probability
                    ),
                },
            )

            internal_results.append(
                result
            )

            internal_score_cache[
                name
            ] = scores

            internal_artifact_cache[
                name
            ] = {
                "graph_propagation": (
                    propagation
                )
            }

    internal_frame = pd.DataFrame(
        internal_results
    )

    base_family_names = [
        "training_popularity",
        "item_item_cosine",
        "truncated_svd",
        "graph_personalized_pagerank",
    ]

    base_winner_rows = {}

    for family in base_family_names:
        family_frame = internal_frame[
            internal_frame[
                "model_family"
            ]
            == family
        ]

        winner = select_best(
            family_frame
        )

        base_winner_rows[
            family
        ] = winner

    popularity_winner_name = (
        base_winner_rows[
            "training_popularity"
        ][
            "model_name"
        ]
    )

    item_winner_name = (
        base_winner_rows[
            "item_item_cosine"
        ][
            "model_name"
        ]
    )

    svd_winner_name = (
        base_winner_rows[
            "truncated_svd"
        ][
            "model_name"
        ]
    )

    graph_winner_name = (
        base_winner_rows[
            "graph_personalized_pagerank"
        ][
            "model_name"
        ]
    )

    component_raw_scores = {
        "item_item_cosine": (
            internal_score_cache[
                item_winner_name
            ]
        ),
        "truncated_svd": (
            internal_score_cache[
                svd_winner_name
            ]
        ),
        "graph_personalized_pagerank": (
            internal_score_cache[
                graph_winner_name
            ]
        ),
        "training_popularity": (
            popularity_score_matrix
        ),
    }

    component_rank_scores = {
        name: rank_normalize_scores(
            scores=scores,
            seen_mask=(
                internal_context
                .seen_mask
            ),
        )
        for name, scores in (
            component_raw_scores.items()
        )
    }

    hybrid_results = []

    component_order = [
        "item_item_cosine",
        "truncated_svd",
        "graph_personalized_pagerank",
        "training_popularity",
    ]

    for weights in convex_weight_grid(
        step=0.25
    ):
        scores = np.zeros_like(
            popularity_score_matrix,
            dtype=np.float64,
        )

        weight_dict = {}

        for component, weight in zip(
            component_order,
            weights,
        ):
            scores += (
                weight
                * component_rank_scores[
                    component
                ]
            )

            weight_dict[
                component
            ] = weight

        name = (
            "hybrid_"
            + "_".join(
                f"{component}={weight:.2f}"
                for component, weight in zip(
                    component_order,
                    weights,
                )
            )
        )

        result, _ = evaluate_scores(
            scores=scores,
            context=internal_context,
            model_family=(
                "rank_normalized_hybrid"
            ),
            model_name=name,
            parameters={
                "weights": weight_dict,
                "normalization": (
                    "candidate_rank"
                ),
            },
        )

        hybrid_results.append(
            result
        )

        internal_score_cache[
            name
        ] = scores

    internal_results.extend(
        hybrid_results
    )

    internal_frame = pd.DataFrame(
        internal_results
    )

    hybrid_winner = select_best(
        internal_frame[
            internal_frame[
                "model_family"
            ]
            == (
                "rank_normalized_hybrid"
            )
        ]
    )

    hybrid_winner_scores = (
        internal_score_cache[
            hybrid_winner[
                "model_name"
            ]
        ]
    )

    hybrid_rank = rank_normalize_scores(
        scores=hybrid_winner_scores,
        seen_mask=(
            internal_context.seen_mask
        ),
    )

    popularity_rank = (
        rank_normalize_scores(
            scores=(
                popularity_score_matrix
            ),
            seen_mask=(
                internal_context.seen_mask
            ),
        )
    )

    herd_results = []

    for penalty in [
        0.0,
        0.05,
        0.10,
        0.20,
        0.30,
    ]:
        scores = (
            hybrid_rank
            - penalty
            * popularity_rank
        )

        name = (
            "herd_mitigation_"
            f"penalty_{penalty:.2f}"
        )

        result, _ = evaluate_scores(
            scores=scores,
            context=internal_context,
            model_family=(
                "herd_mitigation_"
                "reranker"
            ),
            model_name=name,
            parameters={
                "base_hybrid": (
                    hybrid_winner[
                        "model_name"
                    ]
                ),
                "popularity_penalty": (
                    penalty
                ),
            },
        )

        herd_results.append(
            result
        )

        internal_score_cache[
            name
        ] = scores

    internal_results.extend(
        herd_results
    )

    internal_frame = pd.DataFrame(
        internal_results
    )

    internal_frame.to_csv(
        INTERNAL_RESULTS_PATH,
        index=False,
    )

    family_order = [
        "random_sanity",
        "training_popularity",
        "item_item_cosine",
        "truncated_svd",
        "graph_personalized_pagerank",
        "rank_normalized_hybrid",
        "herd_mitigation_reranker",
    ]

    internal_winner_rows = []

    for family in family_order:
        family_frame = internal_frame[
            internal_frame[
                "model_family"
            ]
            == family
        ]

        if family == "random_sanity":
            winner = family_frame[
                family_frame[
                    "model_name"
                ]
                == (
                    "random_seed_average"
                )
            ].iloc[0]
        else:
            winner = select_best(
                family_frame
            )

        internal_winner_rows.append(
            winner
        )

    internal_winners = pd.DataFrame(
        internal_winner_rows
    )

    internal_winners.to_csv(
        INTERNAL_WINNERS_PATH,
        index=False,
    )

    winner_configuration = {
        row[
            "model_family"
        ]: {
            "model_name": (
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
                    "catalog_coverage_at_10",
                    "novelty_at_10",
                    "intra_list_diversity_at_10",
                    "average_recommended_popularity_at_10",
                    "long_tail_share_at_10",
                ]
                if metric in row
                and pd.notna(
                    row[metric]
                )
            },
        }
        for _, row in (
            internal_winners.iterrows()
        )
    }

    external_context = build_context(
        fit_interactions=train,
        holdout=primary_validation,
        catalog=catalog,
        holdout_repo_column=(
            "validation_repo_id"
        ),
    )

    external_results = []
    external_rank_frames = []

    random_average_external, _ = (
        evaluate_random_family(
            context=external_context,
            seeds=[
                11,
                29,
                42,
                71,
                101,
            ],
            stage=(
                "external_validation"
            ),
            compute_beyond_accuracy=True,
        )
    )

    external_results.append(
        random_average_external
    )

    popularity_external_scores = (
        popularity_scores(
            external_context
        )
    )

    result, ranks = evaluate_scores(
        scores=(
            popularity_external_scores
        ),
        context=external_context,
        model_family=(
            "training_popularity"
        ),
        model_name=(
            "training_popularity"
        ),
        parameters=(
            winner_configuration[
                "training_popularity"
            ][
                "parameters"
            ]
        ),
        return_user_ranks=True,
        compute_beyond_accuracy=True,
    )

    external_results.append(
        result
    )

    external_rank_frames.append(
        ranks
    )

    component_winner_config = {
        family: (
            winner_configuration[
                family
            ]
        )
        for family in [
            "item_item_cosine",
            "truncated_svd",
            "graph_personalized_pagerank",
        ]
    }

    component_scores_external, component_artifacts = (
        component_score_bundle(
            context=external_context,
            winners=(
                component_winner_config
            ),
        )
    )

    for family in [
        "item_item_cosine",
        "truncated_svd",
        "graph_personalized_pagerank",
    ]:
        result, ranks = (
            evaluate_scores(
                scores=(
                    component_scores_external[
                        family
                    ]
                ),
                context=external_context,
                model_family=family,
                model_name=(
                    winner_configuration[
                        family
                    ][
                        "model_name"
                    ]
                ),
                parameters=(
                    winner_configuration[
                        family
                    ][
                        "parameters"
                    ]
                ),
                return_user_ranks=True,
                compute_beyond_accuracy=True,
            )
        )

        external_results.append(
            result
        )

        external_rank_frames.append(
            ranks
        )

    component_scores_external[
        "training_popularity"
    ] = popularity_external_scores

    normalized_external = {
        family: rank_normalize_scores(
            scores=scores,
            seen_mask=(
                external_context.seen_mask
            ),
        )
        for family, scores in (
            component_scores_external.items()
        )
    }

    hybrid_parameters = (
        winner_configuration[
            "rank_normalized_hybrid"
        ][
            "parameters"
        ]
    )

    hybrid_external_scores = np.zeros_like(
        popularity_external_scores,
        dtype=np.float64,
    )

    for family, weight in (
        hybrid_parameters[
            "weights"
        ].items()
    ):
        hybrid_external_scores += (
            float(weight)
            * normalized_external[
                family
            ]
        )

    result, ranks = evaluate_scores(
        scores=hybrid_external_scores,
        context=external_context,
        model_family=(
            "rank_normalized_hybrid"
        ),
        model_name=(
            winner_configuration[
                "rank_normalized_hybrid"
            ][
                "model_name"
            ]
        ),
        parameters=(
            hybrid_parameters
        ),
        return_user_ranks=True,
        compute_beyond_accuracy=True,
    )

    external_results.append(
        result
    )

    external_rank_frames.append(
        ranks
    )

    herd_parameters = (
        winner_configuration[
            "herd_mitigation_reranker"
        ][
            "parameters"
        ]
    )

    herd_external_scores = (
        rank_normalize_scores(
            scores=hybrid_external_scores,
            seen_mask=(
                external_context.seen_mask
            ),
        )
        - float(
            herd_parameters[
                "popularity_penalty"
            ]
        )
        * normalized_external[
            "training_popularity"
        ]
    )

    result, ranks = evaluate_scores(
        scores=herd_external_scores,
        context=external_context,
        model_family=(
            "herd_mitigation_reranker"
        ),
        model_name=(
            winner_configuration[
                "herd_mitigation_reranker"
            ][
                "model_name"
            ]
        ),
        parameters=(
            herd_parameters
        ),
        return_user_ranks=True,
        compute_beyond_accuracy=True,
    )

    external_results.append(
        result
    )

    external_rank_frames.append(
        ranks
    )

    external_frame = pd.DataFrame(
        external_results
    )

    external_frame.to_csv(
        EXTERNAL_RESULTS_PATH,
        index=False,
    )

    external_ranks = pd.concat(
        [
            frame
            for frame in (
                external_rank_frames
            )
            if frame is not None
        ],
        ignore_index=True,
    )

    external_ranks.to_csv(
        EXTERNAL_RANKS_PATH,
        index=False,
    )

    selected_row = select_best(
        external_frame
    )

    selected_family = str(
        selected_row[
            "model_family"
        ]
    )

    selected_model_name = str(
        selected_row[
            "model_name"
        ]
    )

    selected_parameters = (
        parse_parameters(
            selected_row
        )
    )

    development_artifact = {
        "artifact_version": (
            "recommendation_development_v1"
        ),
        "selection_status": (
            "selected_on_external_"
            "validation_not_refit_for_test"
        ),
        "git_commit": git_output(
            [
                "rev-parse",
                "HEAD",
            ]
        ),
        "design_freeze_commit": (
            design_manifest[
                "git_commit_before_freeze"
            ]
        ),
        "catalog_repo_ids": (
            external_context.repo_ids
        ),
        "catalog_repo_names": (
            external_context.repo_names
        ),
        "item_counts": (
            external_context.item_counts
        ),
        "selected_family": (
            selected_family
        ),
        "selected_model_name": (
            selected_model_name
        ),
        "selected_parameters": (
            selected_parameters
        ),
        "internal_family_winners": (
            winner_configuration
        ),
        "component_artifacts": (
            component_artifacts
        ),
        "hybrid_parameters": (
            hybrid_parameters
        ),
        "herd_parameters": (
            herd_parameters
        ),
        "test_data_status": (
            "header_only_not_loaded_or_"
            "evaluated"
        ),
    }

    joblib.dump(
        development_artifact,
        DEVELOPMENT_ARTIFACT_PATH,
    )

    development_metadata = {
        "artifact_version": (
            development_artifact[
                "artifact_version"
            ]
        ),
        "selection_status": (
            development_artifact[
                "selection_status"
            ]
        ),
        "git_commit": (
            development_artifact[
                "git_commit"
            ]
        ),
        "selected_family": (
            selected_family
        ),
        "selected_model_name": (
            selected_model_name
        ),
        "selected_parameters": (
            selected_parameters
        ),
        "selected_external_validation_metrics": {
            metric: float(
                selected_row[metric]
            )
            for metric in [
                "precision_at_5",
                "precision_at_10",
                "recall_at_5",
                "recall_at_10",
                "ndcg_at_5",
                "ndcg_at_10",
                "mrr",
                "map_at_10",
                "catalog_coverage_at_10",
                "novelty_at_10",
                "intra_list_diversity_at_10",
                "average_recommended_popularity_at_10",
                "long_tail_share_at_10",
            ]
        },
        "development_artifact_path": str(
            DEVELOPMENT_ARTIFACT_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "development_artifact_sha256": (
            sha256_file(
                DEVELOPMENT_ARTIFACT_PATH
            )
        ),
        "test_data_status": (
            "header_only_not_loaded_or_"
            "evaluated"
        ),
    }

    write_json(
        DEVELOPMENT_METADATA_PATH,
        development_metadata,
    )

    selection_summary = {
        "search_status": (
            "recommendation_primary_"
            "model_search_complete"
        ),
        "git_commit": git_output(
            [
                "rev-parse",
                "HEAD",
            ]
        ),
        "design_freeze_tag": (
            "recommendation-design-"
            "freeze-v1"
        ),
        "internal_tuning_users": int(
            len(
                internal_context.eval_user_ids
            )
        ),
        "external_validation_users": int(
            len(
                external_context.eval_user_ids
            )
        ),
        "catalog_size": int(
            len(
                external_context.repo_ids
            )
        ),
        "internal_family_winners": (
            winner_configuration
        ),
        "external_validation_results": (
            external_frame.to_dict(
                orient="records"
            )
        ),
        "selected_family": (
            selected_family
        ),
        "selected_model_name": (
            selected_model_name
        ),
        "selected_parameters": (
            selected_parameters
        ),
        "selection_rule": (
            "External validation NDCG@10, "
            "then MRR, Recall@10, and "
            "NDCG@5."
        ),
        "random_baseline_policy": (
            "Mean metrics across five "
            "predefined seeds; no best-seed "
            "selection."
        ),
        "test_data_status": (
            "header_only_not_loaded_or_"
            "evaluated"
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
            "external_per_user_ranks": str(
                EXTERNAL_RANKS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "development_metadata": str(
                DEVELOPMENT_METADATA_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "development_artifact": str(
                DEVELOPMENT_ARTIFACT_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
    }

    write_json(
        SELECTION_SUMMARY_PATH,
        selection_summary,
    )

    save_metric_plot(
        results=external_frame,
        metric="ndcg_at_10",
        path=(
            OUTPUT_DIR
            / "external_validation_ndcg_at_10.png"
        ),
        title=(
            "Recommendation External "
            "Validation NDCG@10"
        ),
    )

    save_metric_plot(
        results=external_frame,
        metric="mrr",
        path=(
            OUTPUT_DIR
            / "external_validation_mrr.png"
        ),
        title=(
            "Recommendation External "
            "Validation MRR"
        ),
    )

    save_metric_plot(
        results=external_frame,
        metric=(
            "long_tail_share_at_10"
        ),
        path=(
            OUTPUT_DIR
            / "external_validation_"
            "long_tail_share_at_10.png"
        ),
        title=(
            "Recommendation External "
            "Validation Long-Tail Share@10"
        ),
    )

    print("=" * 112)
    print(
        "RECOMMENDATION PRIMARY "
        "MODEL SEARCH"
    )
    print("=" * 112)

    print()
    print(
        "Internal family winners:"
    )

    print(
        internal_winners[
            [
                "model_family",
                "model_name",
                "ndcg_at_10",
                "mrr",
                "recall_at_10",
                "catalog_coverage_at_10",
                "novelty_at_10",
                "long_tail_share_at_10",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 112)
    print(
        "EXTERNAL VALIDATION FAMILY "
        "RESULTS"
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
        "SELECTED DEVELOPMENT "
        "RECOMMENDER"
    )
    print("=" * 112)

    print(
        json.dumps(
            development_metadata,
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
