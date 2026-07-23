import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT / "config" / "phase3_config.json"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "classification"
    / "classification_train.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "classification"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def summarize_repositories(
    frame: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    rows = []

    for repo_id, group in frame.groupby(
        "repo_id",
        sort=True,
    ):
        group = group.sort_values(
            "cutoff_week"
        )

        rows.append(
            {
                "repo_id": repo_id,
                "repo_full_name": (
                    group[
                        "repo_full_name"
                    ].iloc[0]
                ),
                "rows": int(len(group)),
                "positive_rows": int(
                    group[target].sum()
                ),
                "negative_rows": int(
                    len(group)
                    - group[target].sum()
                ),
                "positive_rate": float(
                    group[target].mean()
                ),
                "first_cutoff_week": str(
                    group[
                        "cutoff_week"
                    ].min()
                ),
                "last_cutoff_week": str(
                    group[
                        "cutoff_week"
                    ].max()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["rows", "repo_id"]
    ).reset_index(drop=True)


def audit_configuration(
    frame: pd.DataFrame,
    target: str,
    n_splits: int,
    validation_samples_per_repo: int,
    gap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_indices = {
        fold: {
            "train": [],
            "validation": [],
            "repositories": set(),
        }
        for fold in range(1, n_splits + 1)
    }

    eligibility_rows = []

    for repo_id, group in frame.groupby(
        "repo_id",
        sort=True,
    ):
        group = group.sort_values(
            "cutoff_week"
        )

        group_indices = group.index.to_numpy()

        splitter = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=(
                validation_samples_per_repo
            ),
            gap=gap_samples,
        )

        try:
            splits = list(
                splitter.split(group_indices)
            )
            eligible = True
            error = ""
        except ValueError as exception:
            splits = []
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
        ) in enumerate(splits, start=1):
            fold_indices[fold_number][
                "train"
            ].extend(
                group_indices[
                    local_train
                ].tolist()
            )

            fold_indices[fold_number][
                "validation"
            ].extend(
                group_indices[
                    local_validation
                ].tolist()
            )

            fold_indices[fold_number][
                "repositories"
            ].add(repo_id)

    fold_rows = []

    for fold_number, values in (
        fold_indices.items()
    ):
        train = frame.loc[
            values["train"]
        ].copy()

        validation = frame.loc[
            values["validation"]
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
                "eligible_repositories": len(
                    values["repositories"]
                ),
                "train_rows": int(
                    len(train)
                ),
                "validation_rows": int(
                    len(validation)
                ),
                "train_positive_rows": int(
                    train[target].sum()
                ),
                "validation_positive_rows": int(
                    validation[target].sum()
                ),
                "train_positive_rate": (
                    float(train[target].mean())
                    if len(train)
                    else np.nan
                ),
                "validation_positive_rate": (
                    float(
                        validation[
                            target
                        ].mean()
                    )
                    if len(validation)
                    else np.nan
                ),
                "train_has_both_classes": bool(
                    train[target].nunique()
                    == 2
                ),
                "validation_has_both_classes": bool(
                    validation[
                        target
                    ].nunique()
                    == 2
                ),
                "target_window_overlap_violations": (
                    overlap_violations
                ),
            }
        )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(eligibility_rows),
    )


def main() -> None:
    config = load_config()

    target = config[
        "classification"
    ]["target"]

    gap_samples = int(
        config[
            "classification"
        ]["purge_weeks"]
    )

    frame = pd.read_csv(TRAIN_PATH)

    frame["cutoff_week"] = pd.to_datetime(
        frame["cutoff_week"],
        utc=True,
        errors="raise",
    )

    frame["target_end_week"] = pd.to_datetime(
        frame["target_end_week"],
        utc=True,
        errors="raise",
    )

    frame = frame.sort_values(
        ["repo_id", "cutoff_week"]
    ).reset_index(drop=True)

    repository_summary = (
        summarize_repositories(
            frame,
            target,
        )
    )

    candidate_configurations = [
        (3, 1),
        (4, 1),
        (5, 1),
        (3, 2),
        (4, 2),
        (5, 2),
    ]

    all_fold_results = []
    all_eligibility = []

    for (
        n_splits,
        validation_samples,
    ) in candidate_configurations:
        fold_results, eligibility = (
            audit_configuration(
                frame=frame,
                target=target,
                n_splits=n_splits,
                validation_samples_per_repo=(
                    validation_samples
                ),
                gap_samples=gap_samples,
            )
        )

        all_fold_results.append(
            fold_results
        )

        all_eligibility.append(
            eligibility
        )

    fold_results = pd.concat(
        all_fold_results,
        ignore_index=True,
    )

    eligibility = pd.concat(
        all_eligibility,
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
            folds=("fold", "count"),
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
            minimum_validation_positives=(
                "validation_positive_rows",
                "min",
            ),
            maximum_validation_positives=(
                "validation_positive_rows",
                "max",
            ),
            all_train_folds_have_both_classes=(
                "train_has_both_classes",
                "all",
            ),
            all_validation_folds_have_both_classes=(
                "validation_has_both_classes",
                "all",
            ),
            total_overlap_violations=(
                "target_window_overlap_violations",
                "sum",
            ),
        )
    )

    eligible_counts = (
        eligibility.groupby(
            [
                "n_splits",
                "validation_samples_per_repo",
                "gap_samples",
            ],
            as_index=False,
        )["eligible"]
        .sum()
        .rename(
            columns={
                "eligible": (
                    "eligible_repository_count"
                )
            }
        )
    )

    configuration_summary = (
        configuration_summary.merge(
            eligible_counts,
            on=[
                "n_splits",
                "validation_samples_per_repo",
                "gap_samples",
            ],
            how="left",
        )
    )

    repository_summary.to_csv(
        OUTPUT_DIR
        / "temporal_cv_repository_summary.csv",
        index=False,
    )

    fold_results.to_csv(
        OUTPUT_DIR
        / "temporal_cv_fold_results.csv",
        index=False,
    )

    eligibility.to_csv(
        OUTPUT_DIR
        / "temporal_cv_repository_eligibility.csv",
        index=False,
    )

    configuration_summary.to_csv(
        OUTPUT_DIR
        / "temporal_cv_configuration_summary.csv",
        index=False,
    )

    print("=" * 105)
    print(
        "CLASSIFICATION TEMPORAL "
        "CROSS-VALIDATION AUDIT"
    )
    print("=" * 105)

    print("\nTraining rows:", len(frame))
    print(
        "Repositories:",
        frame["repo_id"].nunique(),
    )
    print(
        "Target counts:",
        frame[target]
        .value_counts()
        .sort_index()
        .to_dict(),
    )
    print("Purge gap:", gap_samples, "weeks")

    print("\n" + "=" * 105)
    print("REPOSITORY SAMPLE SUMMARY")
    print("=" * 105)
    print(
        repository_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 105)
    print("CONFIGURATION SUMMARY")
    print("=" * 105)
    print(
        configuration_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 105)
    print("FOLD DETAILS")
    print("=" * 105)
    print(
        fold_results.to_string(
            index=False
        )
    )

    print(
        "\nTest data was not loaded or "
        "evaluated."
    )


if __name__ == "__main__":
    main()
