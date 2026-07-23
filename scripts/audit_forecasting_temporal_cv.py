import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
)
from sklearn.model_selection import TimeSeriesSplit


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
    / "forecasting"
    / "forecast_primary_train.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "forecasting"
    / "temporal_cv_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def rmsle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    y_pred = np.clip(
        y_pred,
        a_min=0.0,
        a_max=None,
    )

    return float(
        np.sqrt(
            mean_squared_log_error(
                y_true,
                y_pred,
            )
        )
    )


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    denominator = (
        np.abs(y_true)
        + np.abs(y_pred)
    )

    numerator = (
        2.0
        * np.abs(y_pred - y_true)
    )

    values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(
            numerator,
            dtype=float,
        ),
        where=denominator != 0,
    )

    return float(
        np.mean(values)
    )


def forecast_bias(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    return float(
        np.mean(
            np.asarray(
                y_pred,
                dtype=float,
            )
            - np.asarray(
                y_true,
                dtype=float,
            )
        )
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    y_pred = np.clip(
        np.asarray(
            y_pred,
            dtype=float,
        ),
        a_min=0.0,
        a_max=None,
    )

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    return {
        "rmsle": rmsle(
            y_true,
            y_pred,
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": rmse(
            y_true,
            y_pred,
        ),
        "smape": smape(
            y_true,
            y_pred,
        ),
        "forecast_bias": forecast_bias(
            y_true,
            y_pred,
        ),
    }


def baseline_predictions(
    frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    return {
        "last_week_x4": (
            4.0
            * frame[
                "weekly_new_stars"
            ].to_numpy(dtype=float)
        ),
        "previous_week_x4": (
            4.0
            * frame[
                "previous_week_stars"
            ].to_numpy(dtype=float)
        ),
        "rolling_3week_mean_x4": (
            4.0
            * frame[
                "rolling_3week_mean_stars"
            ].to_numpy(dtype=float)
        ),
        "previous_4week_total": (
            frame[
                "rolling_4week_sum_stars"
            ].to_numpy(dtype=float)
        ),
    }


def prepare_frame(
    frame: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    required = {
        "repo_id",
        "repo_full_name",
        "cutoff_week",
        "target_end_week",
        target,
        "weekly_new_stars",
        "previous_week_stars",
        "rolling_3week_mean_stars",
        "rolling_4week_sum_stars",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Forecast training data is missing "
            f"columns: {sorted(missing)}"
        )

    prepared = frame.copy()

    prepared["cutoff_week"] = (
        pd.to_datetime(
            prepared["cutoff_week"],
            utc=True,
            errors="raise",
        )
    )

    prepared["target_end_week"] = (
        pd.to_datetime(
            prepared["target_end_week"],
            utc=True,
            errors="raise",
        )
    )

    numeric_columns = [
        target,
        "weekly_new_stars",
        "previous_week_stars",
        "rolling_3week_mean_stars",
        "rolling_4week_sum_stars",
    ]

    prepared[numeric_columns] = (
        prepared[numeric_columns].apply(
            pd.to_numeric,
            errors="raise",
        )
    )

    if (
        prepared[numeric_columns]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Forecast training data contains "
            "missing numeric values."
        )

    if (
        prepared[target] < 0
    ).any():
        raise ValueError(
            "Forecast target contains negative "
            "values."
        )

    return prepared.sort_values(
        ["repo_id", "cutoff_week"]
    ).reset_index(drop=True)


def audit_configuration(
    frame: pd.DataFrame,
    target: str,
    n_splits: int,
    validation_samples_per_repo: int,
    gap_samples: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    fold_storage = {
        fold: {
            "train_positions": [],
            "validation_positions": [],
            "repositories": set(),
        }
        for fold in range(
            1,
            n_splits + 1,
        )
    }

    eligibility_rows = []

    for repo_id, group in frame.groupby(
        "repo_id",
        sort=True,
    ):
        positions = group.index.to_numpy()

        splitter = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=(
                validation_samples_per_repo
            ),
            gap=gap_samples,
        )

        try:
            repository_splits = list(
                splitter.split(positions)
            )
            eligible = True
            error = ""
        except ValueError as exception:
            repository_splits = []
            eligible = False
            error = str(exception)

        eligibility_rows.append(
            {
                "n_splits": n_splits,
                "validation_samples_per_repo": (
                    validation_samples_per_repo
                ),
                "gap_samples": gap_samples,
                "repo_id": repo_id,
                "repo_full_name": (
                    group[
                        "repo_full_name"
                    ].iloc[0]
                ),
                "repository_rows": int(
                    len(group)
                ),
                "eligible": eligible,
                "error": error,
            }
        )

        if not eligible:
            continue

        for fold_number, (
            local_train,
            local_validation,
        ) in enumerate(
            repository_splits,
            start=1,
        ):
            fold_storage[fold_number][
                "train_positions"
            ].extend(
                positions[
                    local_train
                ].tolist()
            )

            fold_storage[fold_number][
                "validation_positions"
            ].extend(
                positions[
                    local_validation
                ].tolist()
            )

            fold_storage[fold_number][
                "repositories"
            ].add(repo_id)

    fold_rows = []
    baseline_rows = []

    for fold_number, stored in (
        fold_storage.items()
    ):
        train_positions = np.asarray(
            sorted(
                stored["train_positions"]
            ),
            dtype=int,
        )

        validation_positions = (
            np.asarray(
                sorted(
                    stored[
                        "validation_positions"
                    ]
                ),
                dtype=int,
            )
        )

        train = frame.iloc[
            train_positions
        ].copy()

        validation = frame.iloc[
            validation_positions
        ].copy()

        overlap_violations = 0

        shared_repositories = (
            set(train["repo_id"])
            & set(validation["repo_id"])
        )

        for repo_id in shared_repositories:
            repo_train = train[
                train["repo_id"] == repo_id
            ]

            repo_validation = validation[
                validation[
                    "repo_id"
                ] == repo_id
            ]

            latest_training_target_end = (
                repo_train[
                    "target_end_week"
                ].max()
            )

            earliest_validation_cutoff = (
                repo_validation[
                    "cutoff_week"
                ].min()
            )

            if (
                latest_training_target_end
                >= earliest_validation_cutoff
            ):
                overlap_violations += 1

        fold_rows.append(
            {
                "n_splits": n_splits,
                "validation_samples_per_repo": (
                    validation_samples_per_repo
                ),
                "gap_samples": gap_samples,
                "fold": fold_number,
                "eligible_repositories": int(
                    len(
                        stored[
                            "repositories"
                        ]
                    )
                ),
                "train_rows": int(
                    len(train)
                ),
                "validation_rows": int(
                    len(validation)
                ),
                "train_target_mean": float(
                    train[target].mean()
                ),
                "validation_target_mean": float(
                    validation[
                        target
                    ].mean()
                ),
                "train_target_median": float(
                    train[target].median()
                ),
                "validation_target_median": float(
                    validation[
                        target
                    ].median()
                ),
                "target_window_overlap_violations": (
                    overlap_violations
                ),
            }
        )

        y_validation = validation[
            target
        ].to_numpy(dtype=float)

        for (
            baseline_name,
            predictions,
        ) in baseline_predictions(
            validation
        ).items():
            metrics = evaluate_predictions(
                y_validation,
                predictions,
            )

            baseline_rows.append(
                {
                    "n_splits": n_splits,
                    "validation_samples_per_repo": (
                        validation_samples_per_repo
                    ),
                    "gap_samples": gap_samples,
                    "fold": fold_number,
                    "baseline": baseline_name,
                    "validation_rows": int(
                        len(validation)
                    ),
                    **metrics,
                }
            )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(baseline_rows),
        pd.DataFrame(
            eligibility_rows
        ),
    )


def main() -> None:
    config = load_config()

    forecasting = config[
        "forecasting"
    ]

    target = forecasting["target"]

    gap_samples = int(
        forecasting[
            "forecast_horizon_weeks"
        ]
    )

    frame = prepare_frame(
        pd.read_csv(TRAIN_PATH),
        target,
    )

    candidate_configurations = [
        (3, 1),
        (3, 2),
        (4, 1),
        (4, 2),
        (5, 1),
        (5, 2),
    ]

    fold_frames = []
    baseline_frames = []
    eligibility_frames = []

    for (
        n_splits,
        validation_samples,
    ) in candidate_configurations:
        (
            fold_results,
            baseline_results,
            eligibility,
        ) = audit_configuration(
            frame=frame,
            target=target,
            n_splits=n_splits,
            validation_samples_per_repo=(
                validation_samples
            ),
            gap_samples=gap_samples,
        )

        fold_frames.append(
            fold_results
        )

        baseline_frames.append(
            baseline_results
        )

        eligibility_frames.append(
            eligibility
        )

    fold_results = pd.concat(
        fold_frames,
        ignore_index=True,
    )

    baseline_results = pd.concat(
        baseline_frames,
        ignore_index=True,
    )

    eligibility = pd.concat(
        eligibility_frames,
        ignore_index=True,
    )

    configuration_summary = (
        fold_results.groupby(
            [
                "n_splits",
                "validation_samples_per_repo",
                "gap_samples",
            ],
            as_index=False,
        )
        .agg(
            folds=(
                "fold",
                "count",
            ),
            minimum_eligible_repositories=(
                "eligible_repositories",
                "min",
            ),
            minimum_train_rows=(
                "train_rows",
                "min",
            ),
            minimum_validation_rows=(
                "validation_rows",
                "min",
            ),
            mean_validation_target=(
                "validation_target_mean",
                "mean",
            ),
            std_validation_target=(
                "validation_target_mean",
                "std",
            ),
            total_overlap_violations=(
                "target_window_overlap_violations",
                "sum",
            ),
        )
    )

    baseline_summary = (
        baseline_results.groupby(
            [
                "n_splits",
                "validation_samples_per_repo",
                "gap_samples",
                "baseline",
            ],
            as_index=False,
        )
        .agg(
            folds=(
                "fold",
                "count",
            ),
            mean_rmsle=(
                "rmsle",
                "mean",
            ),
            std_rmsle=(
                "rmsle",
                "std",
            ),
            mean_mae=(
                "mae",
                "mean",
            ),
            mean_rmse=(
                "rmse",
                "mean",
            ),
            mean_smape=(
                "smape",
                "mean",
            ),
            mean_forecast_bias=(
                "forecast_bias",
                "mean",
            ),
        )
    )

    best_baseline_by_configuration = (
        baseline_summary.sort_values(
            [
                "n_splits",
                "validation_samples_per_repo",
                "gap_samples",
                "mean_rmsle",
                "std_rmsle",
            ]
        )
        .groupby(
            [
                "n_splits",
                "validation_samples_per_repo",
                "gap_samples",
            ],
            as_index=False,
        )
        .first()
        .rename(
            columns={
                "baseline": (
                    "best_baseline"
                ),
                "mean_rmsle": (
                    "best_baseline_mean_rmsle"
                ),
                "std_rmsle": (
                    "best_baseline_std_rmsle"
                ),
                "mean_mae": (
                    "best_baseline_mean_mae"
                ),
                "mean_rmse": (
                    "best_baseline_mean_rmse"
                ),
                "mean_smape": (
                    "best_baseline_mean_smape"
                ),
                "mean_forecast_bias": (
                    "best_baseline_mean_bias"
                ),
            }
        )
    )

    final_summary = (
        configuration_summary.merge(
            best_baseline_by_configuration[
                [
                    "n_splits",
                    "validation_samples_per_repo",
                    "gap_samples",
                    "best_baseline",
                    "best_baseline_mean_rmsle",
                    "best_baseline_std_rmsle",
                    "best_baseline_mean_mae",
                    "best_baseline_mean_rmse",
                    "best_baseline_mean_smape",
                    "best_baseline_mean_bias",
                ]
            ],
            on=[
                "n_splits",
                "validation_samples_per_repo",
                "gap_samples",
            ],
            how="left",
        )
    )

    fold_results.to_csv(
        OUTPUT_DIR
        / "forecast_temporal_cv_fold_summary.csv",
        index=False,
    )

    baseline_results.to_csv(
        OUTPUT_DIR
        / "forecast_temporal_cv_baseline_fold_metrics.csv",
        index=False,
    )

    baseline_summary.to_csv(
        OUTPUT_DIR
        / "forecast_temporal_cv_baseline_summary.csv",
        index=False,
    )

    eligibility.to_csv(
        OUTPUT_DIR
        / "forecast_temporal_cv_repository_eligibility.csv",
        index=False,
    )

    final_summary.to_csv(
        OUTPUT_DIR
        / "forecast_temporal_cv_configuration_summary.csv",
        index=False,
    )

    print("=" * 110)
    print(
        "FORECASTING TEMPORAL "
        "CROSS-VALIDATION AUDIT"
    )
    print("=" * 110)

    print()
    print("Training rows:", len(frame))
    print(
        "Repositories:",
        frame["repo_id"].nunique(),
    )
    print(
        "Target:",
        target,
    )
    print(
        "Purge gap:",
        gap_samples,
        "weeks",
    )
    print(
        "Test data:",
        "not loaded",
    )

    print()
    print("=" * 110)
    print("CONFIGURATION SUMMARY")
    print("=" * 110)
    print(
        final_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 110)
    print("BASELINE SUMMARY")
    print("=" * 110)
    print(
        baseline_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 110)
    print("FOLD DETAILS")
    print("=" * 110)
    print(
        fold_results.to_string(
            index=False
        )
    )

    print()
    print(
        "Test data was not loaded or evaluated."
    )


if __name__ == "__main__":
    main()
