import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

PRIMARY_VALIDATION_USERS_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "recommendation"
    / "recommendation_primary_validation_users.csv"
)

EXTERNAL_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "recommendation"
    / "model_search"
    / "external_validation_family_results.csv"
)

EXTERNAL_RANKS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "recommendation"
    / "model_search"
    / "external_validation_per_user_ranks.csv"
)

INTERNAL_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "recommendation"
    / "model_search"
    / "internal_tuning_results.csv"
)

INTERNAL_WINNERS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "recommendation"
    / "model_search"
    / "internal_family_winners.csv"
)

SELECTION_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "recommendation"
    / "model_search"
    / "recommendation_model_selection_summary.json"
)

DEVELOPMENT_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
    / "selected_recommender_development.joblib"
)

DEVELOPMENT_METADATA_PATH = (
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
    / "model_search_audit"
)

AUDIT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "recommendation_model_search_audit.json"
)

BOOTSTRAP_PATH = (
    OUTPUT_DIR
    / "paired_bootstrap_differences.csv"
)

SUBGROUP_PATH = (
    OUTPUT_DIR
    / "history_subgroup_metrics.csv"
)

ABLATION_PATH = (
    OUTPUT_DIR
    / "external_validation_ablation.csv"
)

SENSITIVITY_PATH = (
    OUTPUT_DIR
    / "full_validation_sensitivity.csv"
)

BOOTSTRAP_PLOT_PATH = (
    OUTPUT_DIR
    / "paired_bootstrap_ndcg_at_10.png"
)

BOOTSTRAP_REPLICATIONS = 2000
RANDOM_STATE = 42
TOLERANCE = 1e-12


def load_json(path: Path) -> dict:
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


def load_training_module():
    spec = (
        importlib.util
        .spec_from_file_location(
            "recommendation_training_module",
            TRAINING_SCRIPT_PATH,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Could not load the recommendation "
            "training script as a module."
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def verify_test_header_only() -> list[str]:
    header = pd.read_csv(
        TEST_PATH,
        nrows=0,
    )

    required = [
        "user_id",
        "repo_id",
        "starred_at",
    ]

    missing = [
        column
        for column in required
        if column not in header.columns
    ]

    if missing:
        raise RuntimeError(
            "Recommendation test header is "
            f"missing columns: {missing}"
        )

    return list(
        header.columns
    )


def validate_selection_rule(
    external: pd.DataFrame,
    metadata: dict,
) -> dict[str, Any]:
    ordered = external.sort_values(
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

    expected = ordered.iloc[0]

    matched = bool(
        (
            expected[
                "model_family"
            ]
            == metadata[
                "selected_family"
            ]
        )
        and (
            expected[
                "model_name"
            ]
            == metadata[
                "selected_model_name"
            ]
        )
    )

    if not matched:
        raise RuntimeError(
            "Stored selected recommender does "
            "not match the frozen selection "
            "rule."
        )

    return {
        "selection_rule_matched": (
            matched
        ),
        "expected_family": str(
            expected[
                "model_family"
            ]
        ),
        "expected_model_name": str(
            expected[
                "model_name"
            ]
        ),
    }


def selected_scores_from_artifact(
    context,
    artifact: dict,
    module,
) -> np.ndarray:
    component_artifacts = artifact[
        "component_artifacts"
    ]

    winner_configuration = artifact[
        "internal_family_winners"
    ]

    item_parameters = (
        winner_configuration[
            "item_item_cosine"
        ][
            "parameters"
        ]
    )

    item_scores = (
        module.item_item_scores(
            context=context,
            similarity=(
                component_artifacts[
                    "item_similarity"
                ]
            ),
            aggregation=(
                item_parameters[
                    "aggregation"
                ]
            ),
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

    svd_scores = np.asarray(
        user_latent
        @ components,
        dtype=np.float64,
    )

    graph_scores = (
        module.graph_scores(
            context=context,
            propagation=(
                component_artifacts[
                    "graph_propagation"
                ]
            ),
        )
    )

    popularity_scores = (
        module.popularity_scores(
            context
        )
    )

    raw_components = {
        "item_item_cosine": (
            item_scores
        ),
        "truncated_svd": (
            svd_scores
        ),
        "graph_personalized_pagerank": (
            graph_scores
        ),
        "training_popularity": (
            popularity_scores
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
            raw_components.items()
        )
    }

    hybrid_scores = np.zeros_like(
        popularity_scores,
        dtype=np.float64,
    )

    hybrid_parameters = artifact[
        "hybrid_parameters"
    ]

    for name, weight in (
        hybrid_parameters[
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

    popularity_rank = normalized[
        "training_popularity"
    ]

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
        * popularity_rank
    )


def reproduce_selected_metrics(
    module,
    train: pd.DataFrame,
    primary_validation: pd.DataFrame,
    artifact: dict,
    metadata: dict,
) -> tuple[
    dict[str, Any],
    Any,
    np.ndarray,
]:
    catalog = module.build_catalog(
        train
    )

    artifact_repo_ids = np.asarray(
        artifact[
            "catalog_repo_ids"
        ],
        dtype=np.int64,
    )

    catalog_repo_ids = catalog[
        "repo_id"
    ].to_numpy(
        dtype=np.int64
    )

    if not np.array_equal(
        artifact_repo_ids,
        catalog_repo_ids,
    ):
        raise RuntimeError(
            "Development artifact catalog "
            "does not match the training "
            "catalog."
        )

    context = module.build_context(
        fit_interactions=train,
        holdout=primary_validation,
        catalog=catalog,
        holdout_repo_column=(
            "validation_repo_id"
        ),
    )

    scores = selected_scores_from_artifact(
        context=context,
        artifact=artifact,
        module=module,
    )

    reproduced, _ = (
        module.evaluate_scores(
            scores=scores,
            context=context,
            model_family=(
                artifact[
                    "selected_family"
                ]
            ),
            model_name=(
                artifact[
                    "selected_model_name"
                ]
            ),
            parameters=(
                artifact[
                    "selected_parameters"
                ]
            ),
            compute_beyond_accuracy=True,
        )
    )

    stored = metadata[
        "selected_external_validation_metrics"
    ]

    differences = {}

    for metric, expected in (
        stored.items()
    ):
        actual = float(
            reproduced[metric]
        )

        differences[metric] = {
            "stored": float(
                expected
            ),
            "reproduced": actual,
            "absolute_difference": float(
                abs(
                    actual
                    - float(expected)
                )
            ),
        }

    maximum_difference = max(
        value[
            "absolute_difference"
        ]
        for value in differences.values()
    )

    if maximum_difference > TOLERANCE:
        raise RuntimeError(
            "Selected development metrics "
            "did not reproduce within the "
            f"tolerance. Maximum difference: "
            f"{maximum_difference}"
        )

    summary = {
        "maximum_absolute_difference": (
            maximum_difference
        ),
        "tolerance": TOLERANCE,
        "all_metrics_reproduced": True,
        "metric_differences": (
            differences
        ),
    }

    return (
        summary,
        context,
        scores,
    )


def build_full_validation_holdout(
    validation: pd.DataFrame,
) -> pd.DataFrame:
    return (
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
        .sort_values(
            "user_id"
        )
        .reset_index(drop=True)
    )


def external_sensitivity(
    module,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    artifact: dict,
    strict_metrics: dict,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    catalog = module.build_catalog(
        train
    )

    full_holdout = (
        build_full_validation_holdout(
            validation
        )
    )

    context = module.build_context(
        fit_interactions=train,
        holdout=full_holdout,
        catalog=catalog,
        holdout_repo_column=(
            "validation_repo_id"
        ),
    )

    scores = selected_scores_from_artifact(
        context=context,
        artifact=artifact,
        module=module,
    )

    full_metrics, ranks = (
        module.evaluate_scores(
            scores=scores,
            context=context,
            model_family=(
                artifact[
                    "selected_family"
                ]
            ),
            model_name=(
                artifact[
                    "selected_model_name"
                ]
            ),
            parameters=(
                artifact[
                    "selected_parameters"
                ]
            ),
            return_user_ranks=True,
            compute_beyond_accuracy=True,
        )
    )

    comparison_metrics = [
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

    rows = []

    for metric in comparison_metrics:
        strict_value = float(
            strict_metrics[metric]
        )

        full_value = float(
            full_metrics[metric]
        )

        rows.append(
            {
                "metric": metric,
                "strict_validation_value": (
                    strict_value
                ),
                "full_validation_value": (
                    full_value
                ),
                "full_minus_strict": (
                    full_value
                    - strict_value
                ),
            }
        )

    frame = pd.DataFrame(rows)

    strict_users = 13295
    full_users = int(
        full_metrics[
            "evaluation_users"
        ]
    )

    summary = {
        "strict_validation_users": (
            strict_users
        ),
        "full_validation_users": (
            full_users
        ),
        "additional_equal_timestamp_users": (
            full_users
            - strict_users
        ),
        "full_validation_metrics": {
            metric: float(
                full_metrics[metric]
            )
            for metric in (
                comparison_metrics
            )
        },
        "maximum_absolute_metric_change": float(
            frame[
                "full_minus_strict"
            ].abs().max()
        ),
    }

    return frame, summary


def build_ablation_table(
    external: pd.DataFrame,
    selected_family: str,
) -> pd.DataFrame:
    selected = external[
        external[
            "model_family"
        ]
        == selected_family
    ].iloc[0]

    metrics = [
        "ndcg_at_10",
        "mrr",
        "recall_at_10",
        "ndcg_at_5",
        "catalog_coverage_at_10",
        "novelty_at_10",
        "average_recommended_popularity_at_10",
        "long_tail_share_at_10",
    ]

    frame = external.copy()

    for metric in metrics:
        frame[
            f"delta_selected_minus_{metric}"
        ] = (
            float(
                selected[metric]
            )
            - frame[metric]
        )

    return frame.sort_values(
        [
            "ndcg_at_10",
            "mrr",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )


def prepare_rank_frame(
    ranks: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:
    history = (
        train.groupby(
            "user_id"
        )
        .size()
        .rename(
            "train_history_count"
        )
        .reset_index()
    )

    prepared = ranks.merge(
        history,
        on="user_id",
        how="left",
        validate="many_to_one",
    )

    if (
        prepared[
            "train_history_count"
        ].isna().any()
    ):
        raise RuntimeError(
            "At least one ranked validation "
            "user is missing a training "
            "history count."
        )

    prepared[
        "history_group"
    ] = pd.cut(
        prepared[
            "train_history_count"
        ],
        bins=[
            0,
            1,
            2,
            4,
            9,
            np.inf,
        ],
        labels=[
            "1",
            "2",
            "3-4",
            "5-9",
            "10+",
        ],
        include_lowest=True,
        right=True,
    )

    prepared[
        "ndcg_at_5"
    ] = np.where(
        prepared[
            "positive_rank"
        ]
        <= 5,
        1.0
        / np.log2(
            prepared[
                "positive_rank"
            ]
            + 1.0
        ),
        0.0,
    )

    return prepared


def subgroup_metrics(
    prepared: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        model_family,
        model_name,
        history_group,
    ), group in prepared.groupby(
        [
            "model_family",
            "model_name",
            "history_group",
        ],
        observed=True,
        sort=True,
    ):
        rows.append(
            {
                "model_family": (
                    model_family
                ),
                "model_name": (
                    model_name
                ),
                "history_group": str(
                    history_group
                ),
                "users": int(
                    len(group)
                ),
                "mean_train_history_count": float(
                    group[
                        "train_history_count"
                    ].mean()
                ),
                "ndcg_at_10": float(
                    group[
                        "ndcg_at_10"
                    ].mean()
                ),
                "mrr": float(
                    group[
                        "reciprocal_rank"
                    ].mean()
                ),
                "recall_at_10": float(
                    group[
                        "hit_at_10"
                    ].mean()
                ),
                "ndcg_at_5": float(
                    group[
                        "ndcg_at_5"
                    ].mean()
                ),
                "median_positive_rank": float(
                    group[
                        "positive_rank"
                    ].median()
                ),
                "mean_positive_rank": float(
                    group[
                        "positive_rank"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def bootstrap_differences(
    prepared: pd.DataFrame,
    selected_family: str,
) -> pd.DataFrame:
    selected = (
        prepared[
            prepared[
                "model_family"
            ]
            == selected_family
        ]
        .sort_values(
            "user_id"
        )
        .reset_index(drop=True)
    )

    comparator_families = sorted(
        set(
            prepared[
                "model_family"
            ].unique()
        )
        - {
            selected_family,
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

    for comparator_family in (
        comparator_families
    ):
        comparator = (
            prepared[
                prepared[
                    "model_family"
                ]
                == comparator_family
            ]
            .sort_values(
                "user_id"
            )
            .reset_index(drop=True)
        )

        if not np.array_equal(
            selected[
                "user_id"
            ].to_numpy(),
            comparator[
                "user_id"
            ].to_numpy(),
        ):
            raise RuntimeError(
                "Per-user model rank tables "
                "are not aligned."
            )

        n_users = len(
            selected
        )

        for metric_name, column in (
            metrics.items()
        ):
            differences = (
                selected[column]
                .to_numpy(
                    dtype=float
                )
                - comparator[column]
                .to_numpy(
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
                sample_indices = (
                    generator.integers(
                        0,
                        n_users,
                        size=n_users,
                    )
                )

                bootstrap_means[
                    replicate
                ] = float(
                    differences[
                        sample_indices
                    ].mean()
                )

            rows.append(
                {
                    "selected_family": (
                        selected_family
                    ),
                    "comparator_family": (
                        comparator_family
                    ),
                    "metric": metric_name,
                    "users": n_users,
                    "observed_mean_difference": float(
                        differences.mean()
                    ),
                    "bootstrap_mean_difference": float(
                        bootstrap_means.mean()
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
                    "probability_selected_better": float(
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


def save_bootstrap_plot(
    frame: pd.DataFrame,
) -> None:
    ndcg = (
        frame[
            frame[
                "metric"
            ]
            == "ndcg_at_10"
        ]
        .sort_values(
            "observed_mean_difference"
        )
        .reset_index(drop=True)
    )

    labels = ndcg[
        "comparator_family"
    ].astype(str).tolist()

    centers = ndcg[
        "observed_mean_difference"
    ].to_numpy(
        dtype=float
    )

    lower = (
        centers
        - ndcg[
            "ci_2_5"
        ].to_numpy(
            dtype=float
        )
    )

    upper = (
        ndcg[
            "ci_97_5"
        ].to_numpy(
            dtype=float
        )
        - centers
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    positions = np.arange(
        len(labels)
    )

    axis.errorbar(
        centers,
        positions,
        xerr=np.vstack(
            [
                lower,
                upper,
            ]
        ),
        fmt="o",
        capsize=4,
    )

    axis.axvline(
        0.0,
        linewidth=1,
    )

    axis.set_yticks(
        positions
    )

    axis.set_yticklabels(
        labels
    )

    axis.set_xlabel(
        "Selected minus comparator "
        "NDCG@10"
    )

    axis.set_title(
        "Paired Bootstrap Differences "
        "on Strict Validation Users"
    )

    figure.tight_layout()

    figure.savefig(
        BOOTSTRAP_PLOT_PATH,
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
            "audit outputs already exist. "
            "Refusing to overwrite:\n"
            f"{OUTPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    test_columns = (
        verify_test_header_only()
    )

    module = load_training_module()

    train = module.load_interactions(
        TRAIN_PATH
    )

    validation = module.load_interactions(
        VALIDATION_PATH
    )

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

    external = pd.read_csv(
        EXTERNAL_RESULTS_PATH
    )

    ranks = pd.read_csv(
        EXTERNAL_RANKS_PATH
    )

    internal_results = pd.read_csv(
        INTERNAL_RESULTS_PATH
    )

    internal_winners = pd.read_csv(
        INTERNAL_WINNERS_PATH
    )

    selection_summary = load_json(
        SELECTION_SUMMARY_PATH
    )

    metadata = load_json(
        DEVELOPMENT_METADATA_PATH
    )

    artifact = joblib.load(
        DEVELOPMENT_ARTIFACT_PATH
    )

    artifact_hash = sha256_file(
        DEVELOPMENT_ARTIFACT_PATH
    )

    if (
        artifact_hash
        != metadata[
            "development_artifact_sha256"
        ]
    ):
        raise RuntimeError(
            "Development recommender artifact "
            "hash does not match metadata."
        )

    if (
        artifact[
            "test_data_status"
        ]
        != (
            "header_only_not_loaded_or_"
            "evaluated"
        )
    ):
        raise RuntimeError(
            "Development artifact test status "
            "is not sealed."
        )

    selection_check = (
        validate_selection_rule(
            external=external,
            metadata=metadata,
        )
    )

    (
        reproduction_summary,
        strict_context,
        strict_scores,
    ) = reproduce_selected_metrics(
        module=module,
        train=train,
        primary_validation=(
            primary_validation
        ),
        artifact=artifact,
        metadata=metadata,
    )

    sensitivity_frame, sensitivity = (
        external_sensitivity(
            module=module,
            train=train,
            validation=validation,
            artifact=artifact,
            strict_metrics=(
                metadata[
                    "selected_external_"
                    "validation_metrics"
                ]
            ),
        )
    )

    ablation = build_ablation_table(
        external=external,
        selected_family=(
            metadata[
                "selected_family"
            ]
        ),
    )

    prepared_ranks = prepare_rank_frame(
        ranks=ranks,
        train=train,
    )

    subgroup = subgroup_metrics(
        prepared_ranks
    )

    bootstrap = bootstrap_differences(
        prepared=prepared_ranks,
        selected_family=(
            metadata[
                "selected_family"
            ]
        ),
    )

    ablation.to_csv(
        ABLATION_PATH,
        index=False,
    )

    subgroup.to_csv(
        SUBGROUP_PATH,
        index=False,
    )

    bootstrap.to_csv(
        BOOTSTRAP_PATH,
        index=False,
    )

    sensitivity_frame.to_csv(
        SENSITIVITY_PATH,
        index=False,
    )

    save_bootstrap_plot(
        bootstrap
    )

    selected_row = external[
        external[
            "model_family"
        ]
        == metadata[
            "selected_family"
        ]
    ].iloc[0]

    best_nonselected = (
        external[
            external[
                "model_family"
            ]
            != metadata[
                "selected_family"
            ]
        ]
        .sort_values(
            [
                "ndcg_at_10",
                "mrr",
                "recall_at_10",
                "ndcg_at_5",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )

    summary = {
        "audit_status": (
            "recommendation_model_search_"
            "audit_complete"
        ),
        "test_data_status": (
            "header_only_not_loaded_or_"
            "evaluated"
        ),
        "test_header_columns": (
            test_columns
        ),
        "selected_family": (
            metadata[
                "selected_family"
            ]
        ),
        "selected_model_name": (
            metadata[
                "selected_model_name"
            ]
        ),
        "selected_parameters": (
            metadata[
                "selected_parameters"
            ]
        ),
        "selection_rule_check": (
            selection_check
        ),
        "development_artifact": {
            "path": str(
                DEVELOPMENT_ARTIFACT_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": artifact_hash,
            "metadata_hash_matched": True,
        },
        "metric_reproduction": (
            reproduction_summary
        ),
        "strict_external_validation": {
            "users": int(
                selected_row[
                    "evaluation_users"
                ]
            ),
            "metrics": {
                metric: float(
                    selected_row[metric]
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
            },
        },
        "best_nonselected_family": {
            "model_family": str(
                best_nonselected[
                    "model_family"
                ]
            ),
            "model_name": str(
                best_nonselected[
                    "model_name"
                ]
            ),
            "ndcg_at_10": float(
                best_nonselected[
                    "ndcg_at_10"
                ]
            ),
            "selected_minus_ndcg_at_10": float(
                selected_row[
                    "ndcg_at_10"
                ]
                - best_nonselected[
                    "ndcg_at_10"
                ]
            ),
            "relative_ndcg_at_10_improvement": float(
                (
                    selected_row[
                        "ndcg_at_10"
                    ]
                    - best_nonselected[
                        "ndcg_at_10"
                    ]
                )
                / best_nonselected[
                    "ndcg_at_10"
                ]
            ),
        },
        "full_validation_sensitivity": (
            sensitivity
        ),
        "bootstrap": {
            "replications": (
                BOOTSTRAP_REPLICATIONS
            ),
            "random_state": (
                RANDOM_STATE
            ),
            "comparisons": (
                bootstrap.to_dict(
                    orient="records"
                )
            ),
        },
        "history_subgroups": (
            subgroup.to_dict(
                orient="records"
            )
        ),
        "input_counts": {
            "internal_result_rows": int(
                len(
                    internal_results
                )
            ),
            "internal_winner_rows": int(
                len(
                    internal_winners
                )
            ),
            "external_family_rows": int(
                len(
                    external
                )
            ),
            "external_rank_rows": int(
                len(
                    ranks
                )
            ),
            "selection_summary_external_users": int(
                selection_summary[
                    "external_validation_users"
                ]
            ),
        },
        "output_paths": {
            "summary": str(
                AUDIT_SUMMARY_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "paired_bootstrap": str(
                BOOTSTRAP_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "history_subgroups": str(
                SUBGROUP_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "external_ablation": str(
                ABLATION_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "full_validation_sensitivity": str(
                SENSITIVITY_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "bootstrap_plot": str(
                BOOTSTRAP_PLOT_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
    }

    write_json(
        AUDIT_SUMMARY_PATH,
        summary,
    )

    print("=" * 112)
    print(
        "RECOMMENDATION MODEL-SEARCH "
        "AUDIT"
    )
    print("=" * 112)

    print(
        json.dumps(
            {
                key: value
                for key, value in (
                    summary.items()
                )
                if key not in [
                    "bootstrap",
                    "history_subgroups",
                ]
            },
            indent=2,
        )
    )

    print()
    print("=" * 112)
    print(
        "PAIRED BOOTSTRAP DIFFERENCES"
    )
    print("=" * 112)

    print(
        bootstrap.to_string(
            index=False
        )
    )

    print()
    print("=" * 112)
    print(
        "HISTORY SUBGROUP METRICS"
    )
    print("=" * 112)

    print(
        subgroup.to_string(
            index=False
        )
    )

    print()
    print(
        "Recommendation test values were "
        "not loaded or evaluated."
    )


if __name__ == "__main__":
    main()
