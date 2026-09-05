#!/usr/bin/env python3
"""Register already-frozen Phase 3 models and metrics in local MLflow.

This script never retrains a model and never re-runs a final test evaluation.
It verifies the frozen evidence, records historical validation/test metrics,
and packages the frozen inference contracts as MLflow PyFunc models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRACKING_ROOT = ROOT / ".mlflow_local"
DEFAULT_SUMMARY_PATH = (
    ROOT
    / "outputs"
    / "phase3"
    / "mlflow"
    / "frozen_registry_summary.json"
)
EXPERIMENT_NAME = "github-herd-phase3-frozen-models"

CLASSIFICATION_MODEL = (
    ROOT
    / "models"
    / "classification"
    / "corrected"
    / "selected_classifier_deployment.joblib"
)
CLASSIFICATION_METADATA = (
    ROOT
    / "models"
    / "classification"
    / "corrected"
    / "selected_classifier_deployment_metadata.json"
)
CLASSIFICATION_VALIDATION = (
    ROOT
    / "data"
    / "modeling"
    / "classification"
    / "classification_validation.csv"
)
CLASSIFICATION_PRETEST = (
    ROOT
    / "outputs"
    / "phase3"
    / "classification"
    / "pre_test_freeze_manifest.json"
)
CLASSIFICATION_FINAL = (
    ROOT
    / "outputs"
    / "phase3"
    / "classification"
    / "final_test_freeze_manifest.json"
)

FORECAST_MODEL = (
    ROOT
    / "models"
    / "forecasting"
    / "selected_forecaster_pretest.joblib"
)
FORECAST_METADATA = (
    ROOT
    / "models"
    / "forecasting"
    / "selected_forecaster_pretest_metadata.json"
)
FORECAST_VALIDATION = (
    ROOT
    / "data"
    / "modeling"
    / "forecasting"
    / "forecast_primary_validation.csv"
)
FORECAST_PRETEST = (
    ROOT
    / "outputs"
    / "phase3"
    / "forecasting"
    / "pretest_freeze"
    / "pretest_freeze_manifest.json"
)
FORECAST_FINAL_METRICS = (
    ROOT
    / "outputs"
    / "phase3"
    / "forecasting"
    / "final_test_evaluation"
    / "final_test_metrics.json"
)
FORECAST_FINAL_MANIFEST = (
    ROOT
    / "outputs"
    / "phase3"
    / "forecasting"
    / "final_test_evaluation"
    / "final_test_evaluation_manifest.json"
)

RECOMMENDATION_MODEL = (
    ROOT
    / "models"
    / "recommendation"
    / "selected_recommender_pretest.joblib"
)
RECOMMENDATION_METADATA = (
    ROOT
    / "models"
    / "recommendation"
    / "selected_recommender_pretest_metadata.json"
)
RECOMMENDATION_TRAIN = (
    ROOT
    / "data"
    / "modeling"
    / "recommendation"
    / "recommendation_train.csv"
)
RECOMMENDATION_VALIDATION = (
    ROOT
    / "data"
    / "modeling"
    / "recommendation"
    / "recommendation_validation.csv"
)
RECOMMENDATION_PRETEST = (
    ROOT
    / "outputs"
    / "phase3"
    / "recommendation"
    / "pretest_freeze"
    / "recommendation_pretest_freeze_manifest.json"
)
RECOMMENDATION_FINAL_SUMMARY = (
    ROOT
    / "outputs"
    / "phase3"
    / "recommendation"
    / "final_test"
    / "recommendation_final_test_summary.json"
)
RECOMMENDATION_FINAL_MANIFEST = (
    ROOT
    / "outputs"
    / "phase3"
    / "recommendation"
    / "final_test"
    / "recommendation_final_test_manifest.json"
)

OPERATIONAL_SUMMARY = (
    ROOT
    / "outputs"
    / "phase3"
    / "operational_predictions"
    / "operational_v1_1cafd65"
    / "operational_run_summary.json"
)
OPERATIONAL_MANIFEST = (
    ROOT
    / "outputs"
    / "phase3"
    / "operational_predictions"
    / "operational_v1_1cafd65"
    / "operational_run_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and register already-frozen Phase 3 models in MLflow "
            "without retraining or reopening final-test decisions."
        )
    )
    parser.add_argument(
        "--tracking-root",
        type=Path,
        default=DEFAULT_TRACKING_ROOT,
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Verify frozen evidence and sample inference without writing files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=str, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def require_files(paths: Iterable[Path]) -> None:
    missing = [relative(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required frozen files are missing:\n" + "\n".join(missing)
        )


def verify_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    actual = sha256_file(path)
    passed = actual == expected
    if not passed:
        raise RuntimeError(
            f"Frozen hash mismatch for {label}: {relative(path)}\n"
            f"expected={expected}\nactual={actual}"
        )
    return {
        "path": relative(path),
        "source": "current_worktree",
        "expected_sha256": expected,
        "actual_sha256": actual,
        "passed": True,
    }


def git_blob_sha256(reference: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{reference}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def verify_current_or_git_hash(
    path: Path,
    expected: str,
    label: str,
    reference: str,
) -> dict[str, Any]:
    current_actual = sha256_file(path) if path.is_file() else None
    if current_actual == expected:
        return {
            "path": relative(path),
            "source": "current_worktree",
            "reference": reference,
            "expected_sha256": expected,
            "actual_sha256": current_actual,
            "passed": True,
        }

    relative_path = relative(path)
    historical_actual = git_blob_sha256(reference, relative_path)
    if historical_actual != expected:
        raise RuntimeError(
            f"Frozen hash mismatch for {label}: {relative_path}\n"
            f"reference={reference}\nexpected={expected}\n"
            f"current={current_actual}\nhistorical={historical_actual}"
        )

    return {
        "path": relative_path,
        "source": "git_frozen_snapshot",
        "reference": reference,
        "expected_sha256": expected,
        "current_sha256": current_actual,
        "actual_sha256": historical_actual,
        "passed": True,
    }


def verify_relative_hash_mapping(
    mapping: dict[str, str],
    label: str,
    historical_reference: str | None = None,
) -> list[dict[str, Any]]:
    results = []
    for relative_path, expected in sorted(mapping.items()):
        path = ROOT / relative_path
        if historical_reference is None:
            result = verify_hash(
                path,
                str(expected),
                f"{label}:{relative_path}",
            )
        else:
            result = verify_current_or_git_hash(
                path,
                str(expected),
                f"{label}:{relative_path}",
                historical_reference,
            )
        results.append(result)
    return results


def verify_classification() -> dict[str, Any]:
    pretest = load_json(CLASSIFICATION_PRETEST)
    final = load_json(CLASSIFICATION_FINAL)

    results = []
    for name, information in sorted(pretest["frozen_files"].items()):
        results.append(
            verify_current_or_git_hash(
                ROOT / information["relative_path"],
                information["sha256"],
                f"classification_pretest:{name}",
                "classification-pretest-freeze-v1",
            )
        )

    for name, information in sorted(final["artifacts"].items()):
        results.append(
            verify_hash(
                ROOT / information["relative_path"],
                information["sha256"],
                f"classification_final:{name}",
            )
        )

    metadata = load_json(CLASSIFICATION_METADATA)
    model = joblib.load(CLASSIFICATION_MODEL)
    features = list(metadata["feature_columns"])
    validation = pd.read_csv(CLASSIFICATION_VALIDATION)
    sample = validation[features].head(3).copy()
    probabilities = model.predict_proba(sample)[:, 1]
    threshold = float(metadata["selected_threshold"])
    predictions = (probabilities >= threshold).astype(int)

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Classification validation produced non-finite probabilities.")

    return {
        "hash_checks": results,
        "pretest": pretest,
        "final": final,
        "metadata": metadata,
        "sample_input": sample,
        "sample_output": pd.DataFrame(
            {
                "growth_surge_probability": probabilities,
                "growth_surge_prediction": predictions,
            }
        ),
    }


def forecast_predict(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
) -> np.ndarray:
    features = list(artifact["feature_columns"])
    numeric = frame[features].apply(pd.to_numeric, errors="raise")
    if numeric.isna().any().any():
        raise ValueError("Forecast input contains missing feature values.")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Forecast input contains non-finite feature values.")

    baseline = (
        float(artifact["baseline_multiplier"])
        * pd.to_numeric(
            frame[artifact["baseline_column"]],
            errors="raise",
        ).to_numpy(dtype=float)
    )
    residual = artifact["residual_estimator"].predict(numeric)
    prediction = np.maximum(
        float(artifact["prediction_minimum"]),
        baseline + np.asarray(residual, dtype=float),
    )
    return prediction


def verify_forecasting() -> dict[str, Any]:
    pretest = load_json(FORECAST_PRETEST)
    final_metrics = load_json(FORECAST_FINAL_METRICS)
    final_manifest = load_json(FORECAST_FINAL_MANIFEST)

    results = [
        verify_current_or_git_hash(
            ROOT / pretest["model_path"],
            pretest["model_sha256"],
            "forecast_pretest:model",
            "forecasting-pretest-freeze-v1",
        ),
        verify_current_or_git_hash(
            ROOT / pretest["metadata_path"],
            pretest["metadata_sha256"],
            "forecast_pretest:metadata",
            "forecasting-pretest-freeze-v1",
        ),
        verify_current_or_git_hash(
            ROOT / pretest["reproduction_path"],
            pretest["reproduction_sha256"],
            "forecast_pretest:reproduction",
            "forecasting-pretest-freeze-v1",
        ),
        verify_current_or_git_hash(
            ROOT / pretest["freeze_metadata_path"],
            pretest["freeze_metadata_sha256"],
            "forecast_pretest:freeze_metadata",
            "forecasting-pretest-freeze-v1",
        ),
    ]
    results.extend(
        verify_relative_hash_mapping(
            pretest["input_hashes"],
            "forecast_pretest_input",
            historical_reference="forecasting-pretest-freeze-v1",
        )
    )
    results.extend(
        verify_relative_hash_mapping(
            final_manifest["output_hashes"],
            "forecast_final_output",
        )
    )
    results.extend(
        [
            verify_hash(
                ROOT / final_manifest["evaluation_script_path"],
                final_manifest["evaluation_script_sha256"],
                "forecast_final:evaluation_script",
            ),
            verify_hash(
                ROOT / final_manifest["test_dataset_path"],
                final_manifest["test_dataset_sha256"],
                "forecast_final:test_dataset",
            ),
            verify_hash(
                ROOT / final_manifest["frozen_model_path"],
                final_manifest["frozen_model_sha256"],
                "forecast_final:frozen_model",
            ),
            verify_hash(
                ROOT / final_manifest["pretest_manifest_path"],
                final_manifest["pretest_manifest_sha256"],
                "forecast_final:pretest_manifest",
            ),
        ]
    )

    metadata = load_json(FORECAST_METADATA)
    artifact = joblib.load(FORECAST_MODEL)
    validation = pd.read_csv(FORECAST_VALIDATION)
    feature_columns = list(artifact["feature_columns"])
    required = list(dict.fromkeys(feature_columns + [artifact["baseline_column"]]))
    sample = validation[required].head(3).copy()
    predictions = forecast_predict(artifact, sample)

    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise RuntimeError("Forecast validation produced invalid predictions.")

    return {
        "hash_checks": results,
        "pretest": pretest,
        "final_metrics": final_metrics,
        "final_manifest": final_manifest,
        "metadata": metadata,
        "artifact": artifact,
        "sample_input": sample,
        "sample_output": pd.DataFrame(
            {"predicted_future_4week_stars": predictions}
        ),
    }


def mask_scores(scores: np.ndarray, seen: np.ndarray) -> np.ndarray:
    prepared = np.asarray(scores, dtype=float).copy()
    prepared[~np.isfinite(prepared)] = -np.inf
    prepared[seen] = -np.inf
    return prepared


def descending_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, axis=1, kind="stable")


def rank_normalize(scores: np.ndarray, seen: np.ndarray) -> np.ndarray:
    order = descending_order(mask_scores(scores, seen))
    ranks = np.empty_like(order, dtype=np.int32)
    rows = np.arange(len(order))[:, None]
    ranks[rows, order] = np.arange(order.shape[1], dtype=np.int32)[None, :]
    candidate_counts = (~seen).sum(axis=1).astype(float)
    normalized = (
        1.0
        - ranks.astype(float)
        / np.maximum(candidate_counts - 1.0, 1.0)[:, None]
    )
    normalized[seen] = 0.0
    return normalized


def build_recommendation_history(
    artifact: dict[str, Any],
    train_path: Path,
    validation_path: Path,
) -> sparse.csr_matrix:
    interactions = pd.concat(
        [
            pd.read_csv(train_path, usecols=["user_id", "repo_id"]),
            pd.read_csv(validation_path, usecols=["user_id", "repo_id"]),
        ],
        ignore_index=True,
    )
    if interactions.duplicated(["user_id", "repo_id"]).any():
        raise RuntimeError("Recommendation fit interactions contain duplicate pairs.")

    user_ids = np.asarray(artifact["fit_user_ids"], dtype=np.int64)
    repo_ids = np.asarray(artifact["catalog_repo_ids"], dtype=np.int64)

    expected_users = np.sort(
        interactions["user_id"].astype(np.int64).unique()
    )
    if not np.array_equal(user_ids, expected_users):
        raise RuntimeError("Recommendation fit-user IDs do not match the frozen artifact.")

    user_to_row = {int(value): index for index, value in enumerate(user_ids)}
    repo_to_column = {int(value): index for index, value in enumerate(repo_ids)}

    rows = interactions["user_id"].map(user_to_row)
    columns = interactions["repo_id"].map(repo_to_column)
    if rows.isna().any() or columns.isna().any():
        raise RuntimeError("Recommendation interactions include unknown users or repositories.")

    history = sparse.csr_matrix(
        (
            np.ones(len(interactions), dtype=float),
            (
                rows.to_numpy(dtype=np.int64),
                columns.to_numpy(dtype=np.int64),
            ),
        ),
        shape=(len(user_ids), len(repo_ids)),
        dtype=float,
    )

    observed_counts = np.asarray(history.sum(axis=0)).ravel()
    frozen_counts = np.asarray(artifact["item_counts"], dtype=float)
    if not np.array_equal(observed_counts, frozen_counts):
        raise RuntimeError("Recommendation item counts do not match the frozen artifact.")

    return history


def recommendation_score_rows(
    artifact: dict[str, Any],
    history: sparse.csr_matrix,
    row_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected_history = history[row_indices]
    seen = selected_history.toarray() > 0
    top_k = int(artifact["scoring_contract"]["top_k"])
    candidate_counts = (~seen).sum(axis=1)
    if np.any(candidate_counts < top_k):
        raise RuntimeError("At least one user has fewer candidates than top-k.")

    item_scores = np.asarray(
        selected_history
        @ np.asarray(
            artifact["component_artifacts"]["item_similarity"],
            dtype=float,
        ),
        dtype=float,
    )

    history_lengths = np.asarray(selected_history.sum(axis=1)).ravel()
    aggregation = artifact["internal_family_winners"]["item_item_cosine"][
        "parameters"
    ]["aggregation"]
    if aggregation == "mean":
        item_scores /= np.maximum(history_lengths, 1.0)[:, None]
    elif aggregation != "sum":
        raise RuntimeError(f"Unexpected item-item aggregation: {aggregation}")

    inverse_lengths = np.divide(
        1.0,
        history_lengths,
        out=np.zeros_like(history_lengths),
        where=history_lengths > 0,
    )
    graph_scores = np.asarray(
        (
            sparse.diags(inverse_lengths)
            @ selected_history
        )
        @ np.asarray(
            artifact["component_artifacts"]["graph_propagation"],
            dtype=float,
        ),
        dtype=float,
    )

    popularity_scores = np.broadcast_to(
        np.asarray(artifact["item_counts"], dtype=float)[None, :],
        item_scores.shape,
    ).copy()

    item_rank = rank_normalize(item_scores, seen)
    graph_rank = rank_normalize(graph_scores, seen)
    popularity_rank = rank_normalize(popularity_scores, seen)

    weights = artifact["hybrid_parameters"]["weights"]
    if (
        float(weights["truncated_svd"]) != 0.0
        or float(weights["training_popularity"]) != 0.0
    ):
        raise RuntimeError("Unexpected nonzero frozen recommendation weight.")

    hybrid = (
        float(weights["item_item_cosine"]) * item_rank
        + float(weights["graph_personalized_pagerank"]) * graph_rank
    )
    final_scores = (
        rank_normalize(hybrid, seen)
        - float(artifact["herd_parameters"]["popularity_penalty"])
        * popularity_rank
    )
    ordered = descending_order(mask_scores(final_scores, seen))
    top_indices = ordered[:, :top_k]
    return top_indices, final_scores


def recommendation_output(
    artifact: dict[str, Any],
    history: sparse.csr_matrix,
    user_ids: np.ndarray,
) -> pd.DataFrame:
    frozen_user_ids = np.asarray(artifact["fit_user_ids"], dtype=np.int64)
    user_to_row = {
        int(value): index for index, value in enumerate(frozen_user_ids)
    }
    unknown = [int(value) for value in user_ids if int(value) not in user_to_row]
    if unknown:
        raise ValueError(f"Unknown frozen recommendation users: {unknown[:10]}")

    row_indices = np.asarray(
        [user_to_row[int(value)] for value in user_ids],
        dtype=np.int64,
    )
    top_indices, scores = recommendation_score_rows(
        artifact,
        history,
        row_indices,
    )

    catalog_ids = np.asarray(artifact["catalog_repo_ids"], dtype=np.int64)
    catalog_names = np.asarray(artifact["catalog_repo_names"], dtype=object)
    rows = np.arange(len(user_ids))[:, None]

    result_rows = []
    for position, user_id in enumerate(user_ids):
        indices = top_indices[position]
        result_rows.append(
            {
                "user_id": int(user_id),
                "recommended_repo_ids_json": json.dumps(
                    catalog_ids[indices].astype(int).tolist()
                ),
                "recommended_repo_names_json": json.dumps(
                    catalog_names[indices].astype(str).tolist()
                ),
                "scores_json": json.dumps(
                    scores[rows[position], indices].ravel().astype(float).tolist()
                ),
            }
        )
    return pd.DataFrame(result_rows)


def verify_recommendation() -> dict[str, Any]:
    pretest = load_json(RECOMMENDATION_PRETEST)
    final_summary = load_json(RECOMMENDATION_FINAL_SUMMARY)
    final_manifest = load_json(RECOMMENDATION_FINAL_MANIFEST)

    results = [
        verify_current_or_git_hash(
            ROOT / pretest["artifact_path"],
            pretest["artifact_sha256"],
            "recommendation_pretest:artifact",
            "recommendation-pretest-freeze-v1",
        ),
        verify_current_or_git_hash(
            ROOT / pretest["metadata_path"],
            pretest["metadata_sha256"],
            "recommendation_pretest:metadata",
            "recommendation-pretest-freeze-v1",
        ),
        verify_current_or_git_hash(
            ROOT / pretest["refit_summary_path"],
            pretest["refit_summary_sha256"],
            "recommendation_pretest:refit_summary",
            "recommendation-pretest-freeze-v1",
        ),
    ]
    results.extend(
        verify_relative_hash_mapping(
            pretest["input_hashes"],
            "recommendation_pretest_input",
            historical_reference="recommendation-pretest-freeze-v1",
        )
    )
    results.extend(
        verify_relative_hash_mapping(
            final_manifest["file_hashes"],
            "recommendation_final_output",
        )
    )
    results.extend(
        [
            verify_hash(
                RECOMMENDATION_MODEL,
                final_manifest["artifact_sha256"],
                "recommendation_final:artifact",
            ),
            verify_hash(
                ROOT / "data/modeling/recommendation/recommendation_test.csv",
                final_manifest["test_file_sha256"],
                "recommendation_final:test_dataset",
            ),
        ]
    )

    metadata = load_json(RECOMMENDATION_METADATA)
    artifact = joblib.load(RECOMMENDATION_MODEL)
    if artifact["artifact_version"] != "recommendation_pretest_v1":
        raise RuntimeError("Unexpected recommendation artifact version.")
    if artifact["freeze_status"] != "recommendation_pretest_frozen":
        raise RuntimeError("Recommendation artifact is not frozen.")

    contract = artifact["scoring_contract"]
    if float(contract["forecast_weight"]) != 0.0:
        raise RuntimeError("Frozen primary recommender unexpectedly uses forecast weight.")
    if bool(contract["static_content_in_primary"]):
        raise RuntimeError("Frozen primary recommender unexpectedly uses static content.")

    history = build_recommendation_history(
        artifact,
        RECOMMENDATION_TRAIN,
        RECOMMENDATION_VALIDATION,
    )
    sample_user_ids = np.asarray(artifact["fit_user_ids"][:3], dtype=np.int64)
    sample_input = pd.DataFrame({"user_id": sample_user_ids})
    sample_output = recommendation_output(
        artifact,
        history,
        sample_user_ids,
    )

    if len(sample_output) != len(sample_input):
        raise RuntimeError("Recommendation sample output row count mismatch.")

    return {
        "hash_checks": results,
        "pretest": pretest,
        "final_summary": final_summary,
        "final_manifest": final_manifest,
        "metadata": metadata,
        "artifact": artifact,
        "history": history,
        "sample_input": sample_input,
        "sample_output": sample_output,
    }


def verify_operational() -> dict[str, Any]:
    summary = load_json(OPERATIONAL_SUMMARY)
    manifest = load_json(OPERATIONAL_MANIFEST)

    results = []
    results.extend(
        verify_relative_hash_mapping(
            manifest["input_hashes"],
            "operational_input",
        )
    )

    run_directory = OPERATIONAL_MANIFEST.parent
    for filename, expected in sorted(manifest["output_hashes"].items()):
        results.append(
            verify_hash(
                run_directory / filename,
                expected,
                f"operational_output:{filename}",
            )
        )

    database_path = ROOT / manifest["database_path"]
    database_verification: dict[str, Any]
    if database_path.is_file():
        database_verification = verify_hash(
            database_path,
            manifest["database_sha256"],
            "operational_database",
        )
    else:
        database_verification = {
            "path": relative(database_path),
            "expected_sha256": manifest["database_sha256"],
            "actual_sha256": None,
            "passed": False,
            "note": (
                "The generated database is intentionally ignored by Git and "
                "was not found. Compact operational evidence is still verified."
            ),
        }

    return {
        "hash_checks": results,
        "database_verification": database_verification,
        "summary": summary,
        "manifest": manifest,
    }


class FrozenClassifierPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        self.model = joblib.load(context.artifacts["model"])
        self.metadata = load_json(Path(context.artifacts["metadata"]))

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        features = list(self.metadata["feature_columns"])
        frame = model_input[features].apply(pd.to_numeric, errors="raise")
        probabilities = self.model.predict_proba(frame)[:, 1]
        threshold = float(self.metadata["selected_threshold"])
        return pd.DataFrame(
            {
                "growth_surge_probability": probabilities,
                "growth_surge_prediction": (
                    probabilities >= threshold
                ).astype(int),
            }
        )


class FrozenForecastPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        self.artifact = joblib.load(context.artifacts["model"])

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        predictions = forecast_predict(self.artifact, model_input)
        return pd.DataFrame(
            {"predicted_future_4week_stars": predictions}
        )


class FrozenRecommenderPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        self.artifact = joblib.load(context.artifacts["model"])
        self.history = build_recommendation_history(
            self.artifact,
            Path(context.artifacts["train"]),
            Path(context.artifacts["validation"]),
        )

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        user_ids = pd.to_numeric(
            model_input["user_id"],
            errors="raise",
        ).to_numpy(dtype=np.int64)
        return recommendation_output(
            self.artifact,
            self.history,
            user_ids,
        )


def flatten_numeric_metrics(
    value: dict[str, Any],
    prefix: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, item in value.items():
        metric_name = f"{prefix}_{key}"
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float, np.integer, np.floating)):
            number = float(item)
            if math.isfinite(number):
                result[metric_name] = number
    return result


def safe_log_params(parameters: dict[str, Any]) -> None:
    prepared: dict[str, str] = {}
    for key, value in parameters.items():
        text = json.dumps(value, sort_keys=True, default=str)
        if len(text) > 500:
            text = text[:497] + "..."
        prepared[str(key)] = text
    mlflow.log_params(prepared)


def model_version_for_run(
    client: MlflowClient,
    registered_model_name: str,
    run_id: str,
) -> str:
    versions = client.search_model_versions(
        f"name='{registered_model_name}'"
    )
    matching = [
        version
        for version in versions
        if version.run_id == run_id
    ]
    if not matching:
        raise RuntimeError(
            f"No registered version found for {registered_model_name} and run {run_id}."
        )
    return str(max(int(version.version) for version in matching))


def configure_mlflow(
    tracking_root: Path,
) -> tuple[MlflowClient, str, Path, Path]:
    tracking_root = tracking_root.resolve()
    database_path = tracking_root / "mlflow.db"
    artifact_root = tracking_root / "artifacts"
    tracking_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{database_path}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
    )
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=artifact_root.as_uri(),
            tags={
                "project": "github-herd-data-pipeline",
                "tracking_policy": "frozen_results_no_retraining",
            },
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)
    return client, experiment_id, database_path, artifact_root


def pyfunc_requirements() -> list[str]:
    return [
        "mlflow==3.14.0",
        "joblib==1.5.3",
        "numpy==2.5.1",
        "pandas==2.3.3",
        "scikit-learn==1.9.0",
        "scipy==1.18.0",
    ]


def log_evidence(paths: Iterable[Path]) -> None:
    for path in paths:
        mlflow.log_artifact(str(path), artifact_path="frozen_evidence")


def register_classification(
    client: MlflowClient,
    verification: dict[str, Any],
) -> dict[str, Any]:
    pretest = verification["pretest"]
    final = verification["final"]
    metadata = verification["metadata"]
    sample_input = verification["sample_input"]
    sample_output = verification["sample_output"]

    registered_name = "github-herd-growth-surge-classifier"
    with mlflow.start_run(run_name="classification_frozen_v1") as run:
        mlflow.set_tags(
            {
                "task": "growth_surge_classification",
                "model_status": "frozen",
                "registration_mode": "historical_frozen_evidence_no_retraining",
                "pretest_tag": "classification-pretest-freeze-v1",
                "final_test_tag": "classification-final-test-v1",
                "scientific_rule": final["scientific_rule"],
                "source_git_commit": final["pre_test_freeze_commit"],
            }
        )
        safe_log_params(
            {
                "base_model": pretest["base_model"],
                "target": pretest["target"],
                "probability_method": pretest["probability_method"],
                "selected_threshold": pretest["selected_threshold"],
                "threshold_rule": pretest["threshold_rule"],
                "feature_count": len(metadata["feature_columns"]),
            }
        )
        mlflow.log_metrics(
            flatten_numeric_metrics(
                pretest["validation_metrics"],
                "validation",
            )
        )
        mlflow.log_metrics(
            flatten_numeric_metrics(
                final["metrics_at_frozen_threshold"],
                "test",
            )
        )
        log_evidence(
            [
                CLASSIFICATION_METADATA,
                CLASSIFICATION_PRETEST,
                CLASSIFICATION_FINAL,
                ROOT
                / "outputs/phase3/classification/final_test_evaluation/final_test_evaluation.json",
                ROOT
                / "outputs/phase3/classification/final_test_evaluation/final_test_metrics.csv",
                ROOT
                / "outputs/phase3/classification/final_test_evaluation/final_test_report.txt",
            ]
        )

        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=FrozenClassifierPyFunc(),
            artifacts={
                "model": str(CLASSIFICATION_MODEL),
                "metadata": str(CLASSIFICATION_METADATA),
            },
            input_example=sample_input,
            signature=infer_signature(sample_input, sample_output),
            registered_model_name=registered_name,
            pip_requirements=pyfunc_requirements(),
            metadata={
                "frozen_model_sha256": sha256_file(CLASSIFICATION_MODEL),
                "frozen_threshold": float(metadata["selected_threshold"]),
            },
        )
        version = model_version_for_run(client, registered_name, run.info.run_id)
        client.set_registered_model_alias(
            registered_name,
            "champion",
            version,
        )
        client.set_model_version_tag(
            registered_name,
            version,
            "freeze_status",
            "classification_final_test_results_frozen",
        )
        return {
            "run_id": run.info.run_id,
            "registered_model_name": registered_name,
            "registered_model_version": version,
            "model_uri": model_info.model_uri,
            "artifact_sha256": sha256_file(CLASSIFICATION_MODEL),
        }


def register_forecasting(
    client: MlflowClient,
    verification: dict[str, Any],
) -> dict[str, Any]:
    metadata = verification["metadata"]
    final_metrics = verification["final_metrics"]
    sample_input = verification["sample_input"]
    sample_output = verification["sample_output"]
    artifact = verification["artifact"]

    registered_name = "github-herd-four-week-forecaster"
    with mlflow.start_run(run_name="forecasting_frozen_v1") as run:
        mlflow.set_tags(
            {
                "task": "four_week_popularity_forecasting",
                "model_status": "frozen",
                "registration_mode": "historical_frozen_evidence_no_retraining",
                "pretest_tag": final_metrics["frozen_pretest_tag"],
                "final_test_tag": "forecasting-final-test-v1",
                "post_test_model_changes": str(
                    final_metrics["post_test_model_changes"]
                ).lower(),
                "source_git_commit": final_metrics["frozen_pretest_commit"],
            }
        )
        safe_log_params(
            {
                "approach_family": final_metrics["approach_family"],
                "mode": final_metrics["mode"],
                "model_kind": final_metrics["model_kind"],
                "baseline_formula": final_metrics["baseline_formula"],
                "feature_count": len(final_metrics["feature_columns"]),
                "residual_estimator_parameters": final_metrics[
                    "residual_estimator_parameters"
                ],
            }
        )
        mlflow.log_metrics(
            flatten_numeric_metrics(
                metadata["selected_cv_metrics"],
                "cv",
            )
        )
        mlflow.log_metrics(
            flatten_numeric_metrics(
                metadata["selected_validation_metrics"],
                "validation",
            )
        )
        mlflow.log_metrics(
            flatten_numeric_metrics(
                final_metrics["frozen_model_test_metrics"],
                "test",
            )
        )
        log_evidence(
            [
                FORECAST_METADATA,
                FORECAST_PRETEST,
                FORECAST_FINAL_METRICS,
                FORECAST_FINAL_MANIFEST,
                ROOT
                / "outputs/phase3/forecasting/final_test_evaluation/final_test_report.txt",
                ROOT
                / "outputs/phase3/forecasting/final_test_evaluation/final_test_model_comparison.csv",
            ]
        )

        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=FrozenForecastPyFunc(),
            artifacts={"model": str(FORECAST_MODEL)},
            input_example=sample_input,
            signature=infer_signature(sample_input, sample_output),
            registered_model_name=registered_name,
            pip_requirements=pyfunc_requirements(),
            metadata={
                "frozen_model_sha256": sha256_file(FORECAST_MODEL),
                "artifact_version": artifact["artifact_version"],
            },
        )
        version = model_version_for_run(client, registered_name, run.info.run_id)
        client.set_registered_model_alias(
            registered_name,
            "champion",
            version,
        )
        client.set_model_version_tag(
            registered_name,
            version,
            "freeze_status",
            "forecasting_final_test_results_frozen",
        )
        return {
            "run_id": run.info.run_id,
            "registered_model_name": registered_name,
            "registered_model_version": version,
            "model_uri": model_info.model_uri,
            "artifact_sha256": sha256_file(FORECAST_MODEL),
        }


def register_recommendation(
    client: MlflowClient,
    verification: dict[str, Any],
) -> dict[str, Any]:
    metadata = verification["metadata"]
    final_summary = verification["final_summary"]
    sample_input = verification["sample_input"]
    sample_output = verification["sample_output"]

    registered_name = "github-herd-herd-aware-recommender"
    with mlflow.start_run(run_name="recommendation_frozen_v1") as run:
        mlflow.set_tags(
            {
                "task": "herd_aware_repository_recommendation",
                "model_status": "frozen",
                "registration_mode": "historical_frozen_evidence_no_retraining",
                "pretest_tag": final_summary["pretest_tag"],
                "final_test_tag": "recommendation-final-test-v1",
                "post_test_status": final_summary["post_test_status"],
                "source_git_commit": final_summary["pretest_commit"],
            }
        )
        safe_log_params(
            {
                "selected_family": metadata["selected_family"],
                "selected_model_name": metadata["selected_model_name"],
                "popularity_penalty": metadata["herd_parameters"][
                    "popularity_penalty"
                ],
                "item_item_weight": metadata["hybrid_parameters"]["weights"][
                    "item_item_cosine"
                ],
                "graph_weight": metadata["hybrid_parameters"]["weights"][
                    "graph_personalized_pagerank"
                ],
                "top_k": metadata["scoring_contract"]["top_k"],
                "candidate_exclusion": metadata["scoring_contract"][
                    "candidate_exclusion"
                ],
            }
        )
        mlflow.log_metrics(
            flatten_numeric_metrics(
                metadata["selection_evidence"][
                    "primary_validation_metrics"
                ],
                "validation",
            )
        )
        mlflow.log_metrics(
            flatten_numeric_metrics(
                final_summary["metrics"],
                "test",
            )
        )
        log_evidence(
            [
                RECOMMENDATION_METADATA,
                RECOMMENDATION_PRETEST,
                RECOMMENDATION_FINAL_SUMMARY,
                RECOMMENDATION_FINAL_MANIFEST,
                ROOT
                / "outputs/phase3/recommendation/final_test/recommendation_final_test_metrics.csv",
                ROOT
                / "outputs/phase3/recommendation/model_search/recommendation_model_selection_summary.json",
            ]
        )

        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=FrozenRecommenderPyFunc(),
            artifacts={
                "model": str(RECOMMENDATION_MODEL),
                "train": str(RECOMMENDATION_TRAIN),
                "validation": str(RECOMMENDATION_VALIDATION),
            },
            input_example=sample_input,
            signature=infer_signature(sample_input, sample_output),
            registered_model_name=registered_name,
            pip_requirements=pyfunc_requirements(),
            metadata={
                "frozen_model_sha256": sha256_file(RECOMMENDATION_MODEL),
                "artifact_version": metadata["artifact_version"],
                "top_k": int(metadata["scoring_contract"]["top_k"]),
            },
        )
        version = model_version_for_run(client, registered_name, run.info.run_id)
        client.set_registered_model_alias(
            registered_name,
            "champion",
            version,
        )
        client.set_model_version_tag(
            registered_name,
            version,
            "freeze_status",
            "recommendation_final_test_results_frozen",
        )
        return {
            "run_id": run.info.run_id,
            "registered_model_name": registered_name,
            "registered_model_version": version,
            "model_uri": model_info.model_uri,
            "artifact_sha256": sha256_file(RECOMMENDATION_MODEL),
        }


def log_operational_run(
    verification: dict[str, Any],
) -> dict[str, Any]:
    summary = verification["summary"]
    manifest = verification["manifest"]

    with mlflow.start_run(run_name="operational_predictions_v1") as run:
        mlflow.set_tags(
            {
                "task": "operational_inference",
                "run_status": summary["status"],
                "source_git_commit": summary["git_commit"],
                "tracked_phase2_database_modified": str(
                    summary["tracked_phase2_database_modified"]
                ).lower(),
                "registration_mode": "existing_operational_run_evidence",
            }
        )
        safe_log_params(
            {
                "operational_run_id": summary["run_id"],
                "classification_threshold": summary[
                    "repository_predictions"
                ]["classification_threshold"],
                "recommendation_top_k": summary["recommendations"]["top_k"],
                "recommendation_model": summary["recommendations"][
                    "selected_model"
                ],
            }
        )
        mlflow.log_metrics(
            {
                "repository_prediction_rows": float(
                    summary["repository_predictions"]["rows"]
                ),
                "predicted_growth_surges": float(
                    summary["repository_predictions"]["predicted_surges"]
                ),
                "forecast_mean": float(
                    summary["repository_predictions"]["forecast_mean"]
                ),
                "recommendation_users": float(
                    summary["recommendations"]["fit_users"]
                ),
                "recommendation_rows": float(
                    summary["recommendations"]["rows_written"]
                ),
                "duplicate_user_rank_groups": float(
                    summary["database_verification"][
                        "duplicate_user_rank_groups"
                    ]
                ),
            }
        )
        log_evidence(
            [
                OPERATIONAL_SUMMARY,
                OPERATIONAL_MANIFEST,
                OPERATIONAL_MANIFEST.parent / "repository_predictions.csv",
                OPERATIONAL_MANIFEST.parent
                / "recommendation_repository_frequency.csv",
                OPERATIONAL_MANIFEST.parent
                / "recommendation_sample_first_100_users.csv",
            ]
        )
        return {
            "run_id": run.info.run_id,
            "operational_run_id": manifest["run_id"],
            "database_sha256": manifest["database_sha256"],
        }


def validate_all() -> dict[str, Any]:
    required = [
        CLASSIFICATION_MODEL,
        CLASSIFICATION_METADATA,
        CLASSIFICATION_VALIDATION,
        CLASSIFICATION_PRETEST,
        CLASSIFICATION_FINAL,
        FORECAST_MODEL,
        FORECAST_METADATA,
        FORECAST_VALIDATION,
        FORECAST_PRETEST,
        FORECAST_FINAL_METRICS,
        FORECAST_FINAL_MANIFEST,
        RECOMMENDATION_MODEL,
        RECOMMENDATION_METADATA,
        RECOMMENDATION_TRAIN,
        RECOMMENDATION_VALIDATION,
        RECOMMENDATION_PRETEST,
        RECOMMENDATION_FINAL_SUMMARY,
        RECOMMENDATION_FINAL_MANIFEST,
        OPERATIONAL_SUMMARY,
        OPERATIONAL_MANIFEST,
    ]
    require_files(required)

    classification = verify_classification()
    forecasting = verify_forecasting()
    recommendation = verify_recommendation()
    operational = verify_operational()

    return {
        "classification": classification,
        "forecasting": forecasting,
        "recommendation": recommendation,
        "operational": operational,
    }


def print_validation_summary(
    verification: dict[str, Any],
) -> None:
    classification = verification["classification"]
    forecasting = verification["forecasting"]
    recommendation = verification["recommendation"]
    operational = verification["operational"]

    print("=" * 108)
    print("MLFLOW FROZEN-MODEL REGISTRATION VALIDATION")
    print("=" * 108)
    print(
        "Classification hash checks:",
        len(classification["hash_checks"]),
    )
    print(
        "Classification sample rows:",
        len(classification["sample_output"]),
    )
    print(
        "Classification sample probabilities:",
        classification["sample_output"][
            "growth_surge_probability"
        ].round(8).tolist(),
    )
    print(
        "Forecasting hash checks:",
        len(forecasting["hash_checks"]),
    )
    print(
        "Forecasting sample rows:",
        len(forecasting["sample_output"]),
    )
    print(
        "Forecasting sample predictions:",
        forecasting["sample_output"][
            "predicted_future_4week_stars"
        ].round(8).tolist(),
    )
    print(
        "Recommendation hash checks:",
        len(recommendation["hash_checks"]),
    )
    print(
        "Recommendation history shape:",
        recommendation["history"].shape,
    )
    print(
        "Recommendation history nonzero:",
        recommendation["history"].nnz,
    )
    print(
        "Recommendation sample users:",
        recommendation["sample_input"]["user_id"].astype(int).tolist(),
    )
    print(
        "Recommendation sample output rows:",
        len(recommendation["sample_output"]),
    )
    print(
        "Operational compact hash checks:",
        len(operational["hash_checks"]),
    )
    print(
        "Operational database found and verified:",
        operational["database_verification"]["passed"],
    )
    print(
        "Frozen final-test scripts were not executed:",
        True,
    )
    print(
        "Models were not retrained:",
        True,
    )
    print(
        "Validation status: passed_no_files_written",
    )


def main() -> None:
    arguments = parse_args()
    verification = validate_all()

    if arguments.validate_only:
        print_validation_summary(verification)
        return

    tracking_root = arguments.tracking_root.resolve()
    summary_path = arguments.summary_path.resolve()

    if summary_path.exists():
        raise RuntimeError(
            f"MLflow registry summary already exists; refusing duplicate registration: "
            f"{summary_path}"
        )
    if tracking_root.exists() and any(tracking_root.iterdir()):
        raise RuntimeError(
            f"MLflow tracking root is not empty; refusing an ambiguous first registration: "
            f"{tracking_root}"
        )

    client, experiment_id, database_path, artifact_root = configure_mlflow(
        tracking_root
    )

    classification_result = register_classification(
        client,
        verification["classification"],
    )
    forecasting_result = register_forecasting(
        client,
        verification["forecasting"],
    )
    recommendation_result = register_recommendation(
        client,
        verification["recommendation"],
    )
    operational_result = log_operational_run(
        verification["operational"],
    )

    registered_models = {}
    for name in [
        classification_result["registered_model_name"],
        forecasting_result["registered_model_name"],
        recommendation_result["registered_model_name"],
    ]:
        alias = client.get_model_version_by_alias(name, "champion")
        registered_models[name] = {
            "champion_version": str(alias.version),
            "champion_run_id": alias.run_id,
            "source": alias.source,
        }

    summary = {
        "status": "mlflow_frozen_registration_complete",
        "policy": (
            "Existing frozen models and historical validation/final-test evidence "
            "were registered without retraining and without rerunning final tests."
        ),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "mlflow_version": mlflow.__version__,
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": experiment_id,
        "tracking_database": relative(database_path),
        "artifact_root": relative(artifact_root),
        "classification": classification_result,
        "forecasting": forecasting_result,
        "recommendation": recommendation_result,
        "operational": operational_result,
        "registered_models": registered_models,
        "frozen_hashes": {
            "classification": sha256_file(CLASSIFICATION_MODEL),
            "forecasting": sha256_file(FORECAST_MODEL),
            "recommendation": sha256_file(RECOMMENDATION_MODEL),
        },
        "operational_database_verification": verification["operational"][
            "database_verification"
        ],
    }
    write_json(summary_path, summary)

    print("=" * 108)
    print("MLFLOW FROZEN-MODEL REGISTRATION COMPLETE")
    print("=" * 108)
    print("Experiment:", EXPERIMENT_NAME)
    print("Experiment ID:", experiment_id)
    print("Tracking database:", database_path)
    print("Artifact root:", artifact_root)
    print("Classification run:", classification_result["run_id"])
    print(
        "Classification model version:",
        classification_result["registered_model_version"],
    )
    print("Forecasting run:", forecasting_result["run_id"])
    print(
        "Forecasting model version:",
        forecasting_result["registered_model_version"],
    )
    print("Recommendation run:", recommendation_result["run_id"])
    print(
        "Recommendation model version:",
        recommendation_result["registered_model_version"],
    )
    print("Operational run:", operational_result["run_id"])
    print("Summary:", summary_path)
    print("Models were not retrained.")
    print("Final-test scripts were not rerun.")


if __name__ == "__main__":
    main()
