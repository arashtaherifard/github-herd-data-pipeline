#!/usr/bin/env python3
"""Portable, read-only validation for the Phase 3 delivery environment.

This validator does not train models, rerun final-test scripts, register new
MLflow models, start a Prefect server, or create operational predictions.

It validates the committed MLflow/Prefect evidence, checks frozen hashes, and
runs the existing operational inference script with ``--validate-only``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

OPERATIONAL_SCRIPT = (
    ROOT
    / "scripts"
    / "generate_operational_predictions.py"
)

OPERATIONAL_RUN_DIRECTORY = (
    ROOT
    / "outputs"
    / "phase3"
    / "operational_predictions"
    / "operational_v1_1cafd65"
)

OPERATIONAL_SUMMARY = (
    OPERATIONAL_RUN_DIRECTORY
    / "operational_run_summary.json"
)

OPERATIONAL_MANIFEST = (
    OPERATIONAL_RUN_DIRECTORY
    / "operational_run_manifest.json"
)

MLFLOW_EVIDENCE = (
    ROOT
    / "outputs"
    / "phase3"
    / "mlflow"
    / "frozen_registry_summary.json"
)

PREFECT_READ_ONLY_EVIDENCE = (
    ROOT
    / "outputs"
    / "phase3"
    / "prefect"
    / "read_only_validation_v1.json"
)

PREFECT_REGISTRATION_EVIDENCE = (
    ROOT
    / "outputs"
    / "phase3"
    / "prefect"
    / "deployment_registration_validation_v1.json"
)

PREFECT_WORKER_EVIDENCE = (
    ROOT
    / "outputs"
    / "phase3"
    / "prefect"
    / "deployment_worker_run_validation_v1.json"
)

EXPECTED_HASHES = {
    "database/github_herd.db": (
        "559803caa16b7f43a55cf6947f4eb580"
        "e92bb5a0ed7cd68ad0b581b960f88ee4"
    ),
    (
        "models/classification/corrected/"
        "selected_classifier_deployment.joblib"
    ): (
        "7f66e845d0689c2c29583ec28d93218a"
        "602b951a3246a56767646d2cc48f5f23"
    ),
    (
        "models/forecasting/"
        "selected_forecaster_pretest.joblib"
    ): (
        "e728866f0b523a3393d1fe00cbf58b75"
        "00a02181866b337a4c1b78f8d2806294"
    ),
    (
        "models/recommendation/"
        "selected_recommender_pretest.joblib"
    ): (
        "f2b178d1c888a04a5dbf15253e0d99b"
        "1a920e9230c549c8213e4f529f4c5e878"
    ),
    (
        "outputs/phase3/mlflow/"
        "frozen_registry_summary.json"
    ): (
        "d9c8caa4d506dfa585aa489f5f9f3fd4"
        "40015f3dc740aefa57f91d95f801155b"
    ),
    (
        "outputs/phase3/prefect/"
        "read_only_validation_v1.json"
    ): (
        "2c0e449f9a80ae283be31c0f4a833734"
        "2aa2af6565c28c2a93a0dbd1fa6ce71a"
    ),
    (
        "outputs/phase3/prefect/"
        "deployment_registration_validation_v1.json"
    ): (
        "98c5485904fc21e6c2d407beac74fccc"
        "79d1da7da6f4bbf8f20cdd2deb8e13ea"
    ),
    (
        "outputs/phase3/prefect/"
        "deployment_worker_run_validation_v1.json"
    ): (
        "8fc7173c484bc4de2b3232ab47807a03"
        "4ceb1d70725a2194639e360d0d63d658"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen Phase 3 delivery without "
            "training models or writing predictions."
        )
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def recursive_values(
    value: Any,
    key: str,
) -> list[Any]:
    results: list[Any] = []

    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key:
                results.append(current_value)

            results.extend(
                recursive_values(
                    current_value,
                    key,
                )
            )

    elif isinstance(value, list):
        for item in value:
            results.extend(
                recursive_values(
                    item,
                    key,
                )
            )

    return results


def snapshot_tree(
    path: Path,
    *,
    hash_files: bool,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    snapshot: dict[str, dict[str, Any]] = {}

    for file_path in sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
    ):
        relative_path = str(
            file_path.relative_to(ROOT)
        )

        stat = file_path.stat()

        record: dict[str, Any] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

        if hash_files:
            record["sha256"] = sha256_file(
                file_path
            )

        snapshot[relative_path] = record

    return snapshot


def verify_expected_hashes() -> dict[str, str]:
    actual_hashes: dict[str, str] = {}

    for relative_path, expected_hash in (
        EXPECTED_HASHES.items()
    ):
        path = ROOT / relative_path

        if not path.is_file():
            raise FileNotFoundError(
                f"Required delivery file is missing: "
                f"{relative_path}"
            )

        actual_hash = sha256_file(path)

        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Frozen hash mismatch for "
                f"{relative_path}:\n"
                f"expected={expected_hash}\n"
                f"actual={actual_hash}"
            )

        actual_hashes[
            relative_path
        ] = actual_hash

    return actual_hashes


def verify_operational_evidence() -> dict[str, Any]:
    summary = load_json(
        OPERATIONAL_SUMMARY
    )

    manifest = load_json(
        OPERATIONAL_MANIFEST
    )

    if summary["status"] != (
        "operational_predictions_complete"
    ):
        raise RuntimeError(
            "Unexpected operational summary status."
        )

    if manifest["status"] != (
        "operational_predictions_frozen"
    ):
        raise RuntimeError(
            "Unexpected operational manifest status."
        )

    for relative_path, expected_hash in sorted(
        manifest["input_hashes"].items()
    ):
        path = ROOT / relative_path

        if not path.is_file():
            raise FileNotFoundError(
                f"Operational input is missing: "
                f"{relative_path}"
            )

        actual_hash = sha256_file(path)

        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Operational input hash mismatch: "
                f"{relative_path}"
            )

    for filename, expected_hash in sorted(
        manifest["output_hashes"].items()
    ):
        path = (
            OPERATIONAL_RUN_DIRECTORY
            / filename
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Operational compact output is "
                f"missing: {filename}"
            )

        actual_hash = sha256_file(path)

        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Operational output hash mismatch: "
                f"{filename}"
            )

    generated_database = (
        ROOT
        / manifest["database_path"]
    )

    database_present = (
        generated_database.is_file()
    )

    if database_present:
        actual_database_hash = sha256_file(
            generated_database
        )

        if actual_database_hash != (
            manifest["database_sha256"]
        ):
            raise RuntimeError(
                "Generated operational database "
                "hash mismatch."
            )
    else:
        actual_database_hash = None

    return {
        "run_id": manifest["run_id"],
        "repository_prediction_rows": (
            manifest[
                "repository_prediction_rows"
            ]
        ),
        "recommendation_rows": (
            manifest["recommendation_rows"]
        ),
        "recommendation_users": (
            manifest["recommendation_users"]
        ),
        "generated_database_present": (
            database_present
        ),
        "generated_database_hash": (
            actual_database_hash
        ),
    }


def verify_committed_evidence() -> dict[str, Any]:
    mlflow = load_json(
        MLFLOW_EVIDENCE
    )

    prefect_read_only = load_json(
        PREFECT_READ_ONLY_EVIDENCE
    )

    prefect_registration = load_json(
        PREFECT_REGISTRATION_EVIDENCE
    )

    prefect_worker = load_json(
        PREFECT_WORKER_EVIDENCE
    )

    if mlflow["status"] != (
        "mlflow_frozen_registration_complete"
    ):
        raise RuntimeError(
            "Unexpected MLflow evidence status."
        )

    expected_model_hashes = {
        "classification": (
            EXPECTED_HASHES[
                "models/classification/corrected/"
                "selected_classifier_deployment.joblib"
            ]
        ),
        "forecasting": (
            EXPECTED_HASHES[
                "models/forecasting/"
                "selected_forecaster_pretest.joblib"
            ]
        ),
        "recommendation": (
            EXPECTED_HASHES[
                "models/recommendation/"
                "selected_recommender_pretest.joblib"
            ]
        ),
    }

    if mlflow["frozen_hashes"] != (
        expected_model_hashes
    ):
        raise RuntimeError(
            "MLflow frozen model hashes do not "
            "match the selected models."
        )

    if prefect_read_only["status"] != (
        "prefect_phase3_flow_complete"
    ):
        raise RuntimeError(
            "Unexpected Prefect read-only status."
        )

    if True in recursive_values(
        prefect_read_only,
        "write_enabled",
    ):
        raise RuntimeError(
            "Prefect read-only evidence reports "
            "write mode."
        )

    if True in recursive_values(
        prefect_read_only,
        "models_retrained",
    ):
        raise RuntimeError(
            "Prefect evidence reports retraining."
        )

    if True in recursive_values(
        prefect_read_only,
        "final_test_scripts_rerun",
    ):
        raise RuntimeError(
            "Prefect evidence reports final-test "
            "execution."
        )

    registration_write_values = (
        recursive_values(
            prefect_registration,
            "execute_operational_predictions",
        )
    )

    if False not in registration_write_values:
        raise RuntimeError(
            "Safe Prefect deployment parameter "
            "was not found."
        )

    if prefect_worker["status"] != (
        "prefect_deployment_worker_run_"
        "validation_complete"
    ):
        raise RuntimeError(
            "Unexpected Prefect worker evidence "
            "status."
        )

    flow_run = prefect_worker["flow_run"]
    safety = prefect_worker["safety"]

    if flow_run["state_type"] != "COMPLETED":
        raise RuntimeError(
            "Prefect worker flow did not complete."
        )

    if flow_run["deployment_triggered"] is not True:
        raise RuntimeError(
            "Prefect flow was not deployment-triggered."
        )

    if safety["operational_writes_enabled"] is not False:
        raise RuntimeError(
            "Prefect worker evidence reports writes."
        )

    if safety["models_retrained"] is not False:
        raise RuntimeError(
            "Prefect worker evidence reports "
            "retraining."
        )

    if safety["final_test_scripts_rerun"] is not False:
        raise RuntimeError(
            "Prefect worker evidence reports "
            "final-test execution."
        )

    return {
        "mlflow_status": mlflow["status"],
        "prefect_read_only_status": (
            prefect_read_only["status"]
        ),
        "prefect_registration_status": (
            prefect_registration.get(
                "status",
                "verified_by_frozen_hash",
            )
        ),
        "prefect_worker_status": (
            prefect_worker["status"]
        ),
        "prefect_worker_final_state": (
            flow_run["state_type"]
        ),
    }


def run_operational_validation(
    batch_size: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(OPERATIONAL_SCRIPT),
        "--run-id",
        "delivery_validation",
        "--batch-size",
        str(batch_size),
        "--validate-only",
    ]

    environment = os.environ.copy()
    environment.update(
        {
            "DO_NOT_TRACK": "1",
            "PREFECT_SERVER_ANALYTICS_ENABLED": (
                "false"
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )

    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    if completed.stdout:
        print(completed.stdout.rstrip())

    if completed.returncode != 0:
        if completed.stderr:
            print(
                completed.stderr.rstrip(),
                file=sys.stderr,
            )

        raise RuntimeError(
            "Operational read-only validation "
            f"failed with status "
            f"{completed.returncode}."
        )

    required_markers = [
        "OPERATIONAL PREDICTION VALIDATION",
        "Validation status: "
        "passed_no_files_written",
    ]

    for marker in required_markers:
        if marker not in completed.stdout:
            raise RuntimeError(
                f"Operational validation marker "
                f"is missing: {marker}"
            )

    if (
        "OPERATIONAL PREDICTIONS COMPLETE"
        in completed.stdout
    ):
        raise RuntimeError(
            "The write-enabled operational path "
            "was unexpectedly executed."
        )

    return {
        "exit_status": completed.returncode,
        "batch_size": batch_size,
        "read_only_marker_found": True,
        "write_path_executed": False,
    }


def main() -> None:
    arguments = parse_args()

    if arguments.batch_size <= 0:
        raise ValueError(
            "batch-size must be positive."
        )

    protected_before = (
        verify_expected_hashes()
    )

    compact_outputs_before = snapshot_tree(
        ROOT
        / "outputs"
        / "phase3"
        / "operational_predictions",
        hash_files=True,
    )

    prefect_outputs_before = snapshot_tree(
        ROOT
        / "outputs"
        / "phase3"
        / "prefect",
        hash_files=True,
    )

    generated_database_before = snapshot_tree(
        ROOT / "database" / "generated",
        hash_files=False,
    )

    committed_evidence = (
        verify_committed_evidence()
    )

    operational_evidence = (
        verify_operational_evidence()
    )

    operational_validation = (
        run_operational_validation(
            arguments.batch_size
        )
    )

    protected_after = (
        verify_expected_hashes()
    )

    compact_outputs_after = snapshot_tree(
        ROOT
        / "outputs"
        / "phase3"
        / "operational_predictions",
        hash_files=True,
    )

    prefect_outputs_after = snapshot_tree(
        ROOT
        / "outputs"
        / "phase3"
        / "prefect",
        hash_files=True,
    )

    generated_database_after = snapshot_tree(
        ROOT / "database" / "generated",
        hash_files=False,
    )

    if protected_before != protected_after:
        raise RuntimeError(
            "Protected Phase 3 files changed "
            "during validation."
        )

    if (
        compact_outputs_before
        != compact_outputs_after
    ):
        raise RuntimeError(
            "Operational compact outputs changed "
            "during validation."
        )

    if (
        prefect_outputs_before
        != prefect_outputs_after
    ):
        raise RuntimeError(
            "Prefect evidence changed during "
            "validation."
        )

    if (
        generated_database_before
        != generated_database_after
    ):
        raise RuntimeError(
            "Generated database files changed "
            "during validation."
        )

    result = {
        "status": (
            "phase3_portable_delivery_"
            "validation_passed"
        ),
        "models_retrained": False,
        "final_test_scripts_rerun": False,
        "operational_writes_performed": False,
        "protected_files_unchanged": True,
        "operational_outputs_unchanged": True,
        "prefect_outputs_unchanged": True,
        "generated_database_unchanged": True,
        "committed_evidence": (
            committed_evidence
        ),
        "operational_evidence": (
            operational_evidence
        ),
        "operational_validation": (
            operational_validation
        ),
    }

    print()
    print("=" * 100)
    print("PHASE 3 PORTABLE DELIVERY VALIDATION")
    print("=" * 100)
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )
    print()
    print(
        "Delivery validation status:",
        "PASSED",
    )


if __name__ == "__main__":
    main()
