#!/usr/bin/env python3
"""Freeze-aware Prefect orchestration for the GitHub Herd Phase 3 project.

The default flow is read-only. It verifies frozen artifacts, validates the
existing MLflow registration evidence, and validates operational inference.
Writing a new operational prediction run requires an explicit CLI flag and a
run ID beginning with ``prefect_``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("DO_NOT_TRACK", "1")

from prefect import flow, get_run_logger, task


ROOT = Path(__file__).resolve().parents[1]

MAIN_PYTHON = ROOT / "venv" / "bin" / "python"
BONUS_PYTHON = ROOT / "venv_bonus" / "bin" / "python"

OPERATIONAL_SCRIPT = (
    ROOT
    / "scripts"
    / "generate_operational_predictions.py"
)
MLFLOW_SCRIPT = (
    ROOT
    / "scripts"
    / "register_frozen_models_mlflow.py"
)

MLFLOW_SUMMARY = (
    ROOT
    / "outputs"
    / "phase3"
    / "mlflow"
    / "frozen_registry_summary.json"
)
MLFLOW_DATABASE = (
    ROOT
    / ".mlflow_local"
    / "mlflow.db"
)

CLASSIFICATION_MODEL = (
    ROOT
    / "models"
    / "classification"
    / "corrected"
    / "selected_classifier_deployment.joblib"
)
FORECASTING_MODEL = (
    ROOT
    / "models"
    / "forecasting"
    / "selected_forecaster_pretest.joblib"
)
RECOMMENDATION_MODEL = (
    ROOT
    / "models"
    / "recommendation"
    / "selected_recommender_pretest.joblib"
)

PHASE2_DATABASE = (
    ROOT
    / "database"
    / "github_herd.db"
)
DEFAULT_PREFECT_DATABASE = (
    ROOT
    / "database"
    / "generated"
    / "github_herd_phase3_prefect_predictions.db"
)
DEFAULT_OPERATIONAL_OUTPUT = (
    ROOT
    / "outputs"
    / "phase3"
    / "operational_predictions"
)
ALLOWED_SUMMARY_ROOT = (
    ROOT
    / "outputs"
    / "phase3"
    / "prefect"
)

EXPECTED_MLFLOW_STATUS = (
    "mlflow_frozen_registration_complete"
)
EXPECTED_FROZEN_HASH_KEYS = {
    "classification": CLASSIFICATION_MODEL,
    "forecasting": FORECASTING_MODEL,
    "recommendation": RECOMMENDATION_MODEL,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_output(
    *arguments: str,
) -> str:
    return subprocess.run(
        [
            "git",
            *arguments,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def relative_to_root(
    path: Path,
) -> str:
    return str(
        path.resolve().relative_to(
            ROOT.resolve()
        )
    )


def require_files(
    paths: Iterable[Path],
) -> None:
    missing = [
        str(path)
        for path in paths
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Required orchestration files are missing:\n"
            + "\n".join(missing)
        )


def ensure_inside(
    candidate: Path,
    allowed_root: Path,
    label: str,
) -> Path:
    resolved_candidate = candidate.resolve()
    resolved_root = allowed_root.resolve()

    try:
        resolved_candidate.relative_to(
            resolved_root
        )
    except ValueError as error:
        raise ValueError(
            f"{label} must be inside "
            f"{resolved_root}: "
            f"{resolved_candidate}"
        ) from error

    return resolved_candidate


def clean_command(
    command: list[str],
) -> list[str]:
    root_text = str(ROOT.resolve())
    return [
        item.replace(
            root_text,
            ".",
        )
        for item in command
    ]


def compact_output(
    text: str,
    maximum_lines: int = 20,
) -> str:
    lines = text.strip().splitlines()

    if len(lines) <= maximum_lines:
        return "\n".join(lines)

    return "\n".join(
        lines[-maximum_lines:]
    )


def subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "DO_NOT_TRACK": "1",
            "PREFECT_SERVER_ANALYTICS_ENABLED": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def run_checked_command(
    command: list[str],
    label: str,
) -> dict[str, Any]:
    logger = get_run_logger()
    started_at = utc_now()
    started = time.perf_counter()

    logger.info(
        "Starting %s",
        label,
    )
    logger.info(
        "Command: %s",
        " ".join(
            clean_command(command)
        ),
    )

    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    duration_seconds = (
        time.perf_counter()
        - started
    )
    completed_at = utc_now()

    if completed.stdout.strip():
        logger.info(
            "%s stdout:\n%s",
            label,
            completed.stdout.rstrip(),
        )

    if completed.stderr.strip():
        logger.warning(
            "%s stderr:\n%s",
            label,
            completed.stderr.rstrip(),
        )

    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code "
            f"{completed.returncode}.\n"
            f"stdout tail:\n"
            f"{compact_output(completed.stdout)}\n"
            f"stderr tail:\n"
            f"{compact_output(completed.stderr)}"
        )

    logger.info(
        "%s completed in %.3f seconds",
        label,
        duration_seconds,
    )

    return {
        "label": label,
        "status": "passed",
        "command": clean_command(command),
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "duration_seconds": duration_seconds,
        "stdout_tail": compact_output(
            completed.stdout
        ),
        "stderr_tail": compact_output(
            completed.stderr
        ),
        "exit_code": completed.returncode,
    }


@task(
    name="verify-frozen-phase3-inputs",
    description=(
        "Verify the committed frozen-model hashes, "
        "MLflow evidence, and orchestration executables."
    ),
    retries=0,
    persist_result=False,
)
def verify_frozen_inputs() -> dict[str, Any]:
    logger = get_run_logger()

    require_files(
        [
            MAIN_PYTHON,
            BONUS_PYTHON,
            OPERATIONAL_SCRIPT,
            MLFLOW_SCRIPT,
            MLFLOW_SUMMARY,
            MLFLOW_DATABASE,
            CLASSIFICATION_MODEL,
            FORECASTING_MODEL,
            RECOMMENDATION_MODEL,
            PHASE2_DATABASE,
        ]
    )

    summary = load_json(
        MLFLOW_SUMMARY
    )

    if (
        summary.get("status")
        != EXPECTED_MLFLOW_STATUS
    ):
        raise RuntimeError(
            "Unexpected MLflow summary status: "
            f"{summary.get('status')}"
        )

    recorded_hashes = summary.get(
        "frozen_hashes",
        {},
    )
    hash_results: dict[
        str,
        dict[str, Any],
    ] = {}

    for model_name, model_path in (
        EXPECTED_FROZEN_HASH_KEYS.items()
    ):
        expected = recorded_hashes.get(
            model_name
        )

        if not expected:
            raise RuntimeError(
                "Missing frozen hash in MLflow "
                f"summary: {model_name}"
            )

        actual = sha256_file(
            model_path
        )

        if actual != expected:
            raise RuntimeError(
                "Frozen model hash mismatch for "
                f"{model_name}:\n"
                f"expected={expected}\n"
                f"actual={actual}"
            )

        hash_results[model_name] = {
            "path": relative_to_root(
                model_path
            ),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "passed": True,
        }

    frozen_model_status = git_output(
        "status",
        "--porcelain",
        "--",
        relative_to_root(
            CLASSIFICATION_MODEL
        ),
        relative_to_root(
            FORECASTING_MODEL
        ),
        relative_to_root(
            RECOMMENDATION_MODEL
        ),
    )

    if frozen_model_status:
        raise RuntimeError(
            "Frozen model files have Git-visible "
            "changes:\n"
            f"{frozen_model_status}"
        )

    branch = git_output(
        "branch",
        "--show-current",
    )
    commit = git_output(
        "rev-parse",
        "HEAD",
    )
    repository_status = git_output(
        "status",
        "--short",
    )

    phase2_database_hash = sha256_file(
        PHASE2_DATABASE
    )

    logger.info(
        "Verified frozen artifacts at commit %s",
        commit,
    )
    logger.info(
        "Classification, forecasting, and "
        "recommendation hashes match the "
        "committed MLflow summary."
    )

    return {
        "status": "passed",
        "git_branch": branch,
        "git_commit": commit,
        "repository_clean": (
            repository_status == ""
        ),
        "repository_status": (
            repository_status
            if repository_status
            else "clean"
        ),
        "mlflow_summary_status": summary[
            "status"
        ],
        "mlflow_experiment_name": summary[
            "experiment_name"
        ],
        "mlflow_version": summary[
            "mlflow_version"
        ],
        "frozen_model_hashes": hash_results,
        "phase2_database": {
            "path": relative_to_root(
                PHASE2_DATABASE
            ),
            "sha256": phase2_database_hash,
        },
    }


@task(
    name="validate-frozen-mlflow-evidence",
    description=(
        "Run the existing MLflow integration in "
        "read-only validation mode."
    ),
    retries=0,
    persist_result=False,
)
def validate_mlflow_evidence() -> dict[str, Any]:
    result = run_checked_command(
        [
            str(BONUS_PYTHON),
            str(MLFLOW_SCRIPT),
            "--validate-only",
        ],
        "MLflow frozen-evidence validation",
    )

    expected_marker = (
        "Validation status: "
        "passed_no_files_written"
    )

    if (
        expected_marker
        not in result["stdout_tail"]
    ):
        raise RuntimeError(
            "MLflow validation completed but the "
            "read-only success marker was not found."
        )

    return result


@task(
    name="validate-operational-inference",
    description=(
        "Validate classification, forecasting, "
        "and recommendation inference without "
        "writing outputs."
    ),
    retries=0,
    persist_result=False,
)
def validate_operational_inference(
    batch_size: int,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    result = run_checked_command(
        [
            str(MAIN_PYTHON),
            str(OPERATIONAL_SCRIPT),
            "--batch-size",
            str(batch_size),
            "--validate-only",
        ],
        "Operational inference validation",
    )

    expected_marker = (
        "Validation status: "
        "passed_no_files_written"
    )

    if (
        expected_marker
        not in result["stdout_tail"]
    ):
        raise RuntimeError(
            "Operational validation completed but "
            "the read-only success marker was not "
            "found."
        )

    return result


@task(
    name="execute-explicit-operational-run",
    description=(
        "Execute a new operational prediction run "
        "only after explicit write authorization."
    ),
    retries=0,
    persist_result=False,
)
def execute_operational_run(
    run_id: str,
    database_path: str,
    output_root: str,
    batch_size: int,
) -> dict[str, Any]:
    if not run_id:
        raise ValueError(
            "An explicit run_id is required for "
            "a write-enabled operational run."
        )

    if not run_id.startswith(
        "prefect_"
    ):
        raise ValueError(
            "Write-enabled Prefect run IDs must "
            "begin with 'prefect_'."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    resolved_database = (
        Path(database_path)
        if Path(database_path).is_absolute()
        else ROOT / database_path
    ).resolve()

    resolved_output_root = (
        Path(output_root)
        if Path(output_root).is_absolute()
        else ROOT / output_root
    ).resolve()

    ensure_inside(
        resolved_database,
        ROOT / "database" / "generated",
        "database_path",
    )
    ensure_inside(
        resolved_output_root,
        DEFAULT_OPERATIONAL_OUTPUT,
        "output_root",
    )

    if (
        resolved_database
        == PHASE2_DATABASE.resolve()
    ):
        raise RuntimeError(
            "Refusing to modify the tracked "
            "Phase 2 database."
        )

    expected_output_directory = (
        resolved_output_root
        / run_id
    )

    if expected_output_directory.exists():
        raise RuntimeError(
            "Operational output directory already "
            f"exists: {expected_output_directory}"
        )

    phase2_hash_before = sha256_file(
        PHASE2_DATABASE
    )

    result = run_checked_command(
        [
            str(MAIN_PYTHON),
            str(OPERATIONAL_SCRIPT),
            "--database-path",
            str(resolved_database),
            "--output-root",
            str(resolved_output_root),
            "--run-id",
            run_id,
            "--batch-size",
            str(batch_size),
        ],
        "Explicit Prefect operational prediction run",
    )

    phase2_hash_after = sha256_file(
        PHASE2_DATABASE
    )

    if (
        phase2_hash_before
        != phase2_hash_after
    ):
        raise RuntimeError(
            "The tracked Phase 2 database changed "
            "during operational inference."
        )

    summary_path = (
        expected_output_directory
        / "operational_run_summary.json"
    )
    manifest_path = (
        expected_output_directory
        / "operational_run_manifest.json"
    )

    require_files(
        [
            summary_path,
            manifest_path,
        ]
    )

    operational_summary = load_json(
        summary_path
    )
    operational_manifest = load_json(
        manifest_path
    )

    if (
        operational_summary.get("run_id")
        != run_id
    ):
        raise RuntimeError(
            "Operational summary run ID mismatch."
        )

    if (
        operational_manifest.get("run_id")
        != run_id
    ):
        raise RuntimeError(
            "Operational manifest run ID mismatch."
        )

    result.update(
        {
            "run_id": run_id,
            "database_path": relative_to_root(
                resolved_database
            ),
            "output_directory": relative_to_root(
                expected_output_directory
            ),
            "repository_prediction_rows": (
                operational_manifest[
                    "repository_prediction_rows"
                ]
            ),
            "recommendation_rows": (
                operational_manifest[
                    "recommendation_rows"
                ]
            ),
            "recommendation_users": (
                operational_manifest[
                    "recommendation_users"
                ]
            ),
            "database_sha256": (
                operational_manifest[
                    "database_sha256"
                ]
            ),
            "phase2_database_unchanged": True,
        }
    )

    return result


@task(
    name="write-prefect-flow-summary",
    description=(
        "Write a compact, portable summary of a "
        "verified Prefect flow run."
    ),
    retries=0,
    persist_result=False,
)
def write_flow_summary(
    summary_path: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    candidate = (
        Path(summary_path)
        if Path(summary_path).is_absolute()
        else ROOT / summary_path
    ).resolve()

    ensure_inside(
        candidate,
        ALLOWED_SUMMARY_ROOT,
        "summary_path",
    )

    if candidate.exists():
        raise RuntimeError(
            "Refusing to overwrite an existing "
            f"Prefect summary: {candidate}"
        )

    portable_summary = dict(
        summary
    )
    portable_summary[
        "summary_path"
    ] = relative_to_root(
        candidate
    )

    write_json(
        candidate,
        portable_summary,
    )

    digest = sha256_file(
        candidate
    )

    return {
        "status": "written",
        "path": relative_to_root(
            candidate
        ),
        "sha256": digest,
    }


@flow(
    name="github-herd-phase3-safe-orchestration",
    version="1.0.0",
    description=(
        "Freeze-aware orchestration for validating "
        "MLflow evidence and operational inference. "
        "Operational writes are disabled by default."
    ),
    retries=0,
    persist_result=False,
    log_prints=True,
)
def phase3_prefect_flow(
    batch_size: int = 5000,
    execute_operational_predictions: bool = False,
    run_id: str | None = None,
    database_path: str = (
        "database/generated/"
        "github_herd_phase3_prefect_predictions.db"
    ),
    output_root: str = (
        "outputs/phase3/"
        "operational_predictions"
    ),
    summary_path: str | None = None,
) -> dict[str, Any]:
    logger = get_run_logger()
    flow_started_at = utc_now()
    flow_started = time.perf_counter()

    logger.info(
        "Starting freeze-aware Phase 3 Prefect flow."
    )
    logger.info(
        "Operational writes enabled: %s",
        execute_operational_predictions,
    )

    preflight = verify_frozen_inputs()
    mlflow_validation = (
        validate_mlflow_evidence()
    )
    operational_validation = (
        validate_operational_inference(
            batch_size=batch_size,
        )
    )

    operational_execution = None

    if execute_operational_predictions:
        if run_id is None:
            raise ValueError(
                "run_id is required when "
                "execute_operational_predictions "
                "is enabled."
            )

        operational_execution = (
            execute_operational_run(
                run_id=run_id,
                database_path=database_path,
                output_root=output_root,
                batch_size=batch_size,
            )
        )
    elif run_id is not None:
        logger.warning(
            "run_id was supplied, but operational "
            "writes are disabled. No prediction run "
            "was created."
        )

    flow_completed_at = utc_now()
    duration_seconds = (
        time.perf_counter()
        - flow_started
    )

    summary: dict[str, Any] = {
        "status": (
            "prefect_phase3_flow_complete"
        ),
        "policy": (
            "Frozen models and historical final-test "
            "results were validated without "
            "retraining or rerunning final tests."
        ),
        "flow_name": (
            "github-herd-phase3-"
            "safe-orchestration"
        ),
        "flow_version": "1.0.0",
        "started_at_utc": flow_started_at,
        "completed_at_utc": flow_completed_at,
        "duration_seconds": duration_seconds,
        "git_branch": preflight[
            "git_branch"
        ],
        "git_commit": preflight[
            "git_commit"
        ],
        "repository_clean_at_preflight": (
            preflight[
                "repository_clean"
            ]
        ),
        "write_enabled": (
            execute_operational_predictions
        ),
        "preflight": preflight,
        "mlflow_validation": (
            mlflow_validation
        ),
        "operational_validation": (
            operational_validation
        ),
        "operational_execution": (
            operational_execution
        ),
        "models_retrained": False,
        "final_test_scripts_rerun": False,
    }

    if summary_path:
        summary["summary_artifact"] = (
            write_flow_summary(
                summary_path=summary_path,
                summary=summary,
            )
        )
    else:
        summary[
            "summary_artifact"
        ] = None

    logger.info(
        "Prefect flow completed in %.3f seconds.",
        duration_seconds,
    )
    logger.info(
        "Models retrained: false"
    )
    logger.info(
        "Final-test scripts rerun: false"
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the freeze-aware Phase 3 Prefect "
            "orchestration flow. The default mode "
            "is read-only validation."
        )
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--execute-operational-predictions",
        action="store_true",
        help=(
            "Explicitly authorize a new operational "
            "prediction run. Disabled by default."
        ),
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Required for a write-enabled run and "
            "must begin with 'prefect_'."
        ),
    )
    parser.add_argument(
        "--database-path",
        default=relative_to_root(
            DEFAULT_PREFECT_DATABASE
        ),
    )
    parser.add_argument(
        "--output-root",
        default=relative_to_root(
            DEFAULT_OPERATIONAL_OUTPUT
        ),
    )
    parser.add_argument(
        "--summary-path",
        help=(
            "Optional compact flow summary path "
            "inside outputs/phase3/prefect."
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()

    result = phase3_prefect_flow(
        batch_size=arguments.batch_size,
        execute_operational_predictions=(
            arguments.execute_operational_predictions
        ),
        run_id=arguments.run_id,
        database_path=arguments.database_path,
        output_root=arguments.output_root,
        summary_path=arguments.summary_path,
    )

    print(
        "=" * 108
    )
    print(
        "PREFECT PHASE 3 FLOW RESULT"
    )
    print(
        "=" * 108
    )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
