from pathlib import Path

import pandas as pd

from phase3_split_planning import build_forecast_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3" / "audit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def heading(title: str) -> str:
    return "\n" + "=" * 100 + f"\n{title}\n" + "=" * 100


def create_repository_aware_split(
    forecast_data: pd.DataFrame,
    validation_samples: int,
    test_samples: int,
    minimum_train_samples: int,
) -> tuple[pd.DataFrame, dict]:
    split_frames = []
    eligible_repository_count = 0
    overlap_violations = 0
    original_rows_for_eligible_repositories = 0

    for repo_id, group in forecast_data.groupby("repo_id"):
        group = group.sort_values("cutoff_week").reset_index(drop=True)

        minimum_raw_samples = (
            minimum_train_samples
            + validation_samples
            + test_samples
        )

        if len(group) < minimum_raw_samples:
            continue

        # Reserve the latest samples for final testing.
        test = group.tail(test_samples).copy()
        first_test_cutoff = test["cutoff_week"].min()

        # Validation targets must finish before testing begins.
        validation_pool = group[
            group["target_end_week"] < first_test_cutoff
        ].copy()

        if len(validation_pool) < validation_samples:
            continue

        validation = validation_pool.tail(
            validation_samples
        ).copy()

        first_validation_cutoff = validation[
            "cutoff_week"
        ].min()

        # Training targets must finish before validation begins.
        train = group[
            group["target_end_week"]
            < first_validation_cutoff
        ].copy()

        if len(train) < minimum_train_samples:
            continue

        train_validation_overlap = not (
            train["target_end_week"].max()
            < validation["cutoff_week"].min()
        )

        validation_test_overlap = not (
            validation["target_end_week"].max()
            < test["cutoff_week"].min()
        )

        if train_validation_overlap or validation_test_overlap:
            overlap_violations += 1
            continue

        eligible_repository_count += 1
        original_rows_for_eligible_repositories += len(group)

        train["split"] = "train"
        validation["split"] = "validation"
        test["split"] = "test"

        for frame in [train, validation, test]:
            frame["validation_samples_per_repo"] = (
                validation_samples
            )
            frame["test_samples_per_repo"] = test_samples
            frame["minimum_train_samples"] = (
                minimum_train_samples
            )

        split_frames.extend([train, validation, test])

    if split_frames:
        split_data = pd.concat(
            split_frames,
            ignore_index=True,
        )
    else:
        split_data = pd.DataFrame()

    if len(split_data):
        train_rows = int(
            (split_data["split"] == "train").sum()
        )
        validation_rows = int(
            (split_data["split"] == "validation").sum()
        )
        test_rows = int(
            (split_data["split"] == "test").sum()
        )

        train_target_mean = split_data.loc[
            split_data["split"] == "train",
            "future_4week_stars",
        ].mean()

        validation_target_mean = split_data.loc[
            split_data["split"] == "validation",
            "future_4week_stars",
        ].mean()

        test_target_mean = split_data.loc[
            split_data["split"] == "test",
            "future_4week_stars",
        ].mean()
    else:
        train_rows = 0
        validation_rows = 0
        test_rows = 0
        train_target_mean = None
        validation_target_mean = None
        test_target_mean = None

    retained_rows = len(split_data)

    summary = {
        "validation_samples_per_repo": validation_samples,
        "test_samples_per_repo": test_samples,
        "minimum_train_samples": minimum_train_samples,
        "eligible_repositories": eligible_repository_count,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "retained_rows": retained_rows,
        "purged_or_unused_rows": (
            original_rows_for_eligible_repositories
            - retained_rows
        ),
        "overlap_violations": overlap_violations,
        "train_target_mean": train_target_mean,
        "validation_target_mean": validation_target_mean,
        "test_target_mean": test_target_mean,
    }

    return split_data, summary


def verify_split(split_data: pd.DataFrame) -> dict:
    if split_data.empty:
        return {
            "repositories_in_all_splits": 0,
            "train_validation_overlap_violations": 0,
            "validation_test_overlap_violations": 0,
        }

    train = split_data[
        split_data["split"] == "train"
    ]

    validation = split_data[
        split_data["split"] == "validation"
    ]

    test = split_data[
        split_data["split"] == "test"
    ]

    common_repositories = (
        set(train["repo_id"])
        & set(validation["repo_id"])
        & set(test["repo_id"])
    )

    train_validation_violations = 0
    validation_test_violations = 0

    for repo_id in common_repositories:
        repo_train = train[train["repo_id"] == repo_id]
        repo_validation = validation[
            validation["repo_id"] == repo_id
        ]
        repo_test = test[test["repo_id"] == repo_id]

        if not (
            repo_train["target_end_week"].max()
            < repo_validation["cutoff_week"].min()
        ):
            train_validation_violations += 1

        if not (
            repo_validation["target_end_week"].max()
            < repo_test["cutoff_week"].min()
        ):
            validation_test_violations += 1

    return {
        "repositories_in_all_splits": len(
            common_repositories
        ),
        "train_validation_overlap_violations": (
            train_validation_violations
        ),
        "validation_test_overlap_violations": (
            validation_test_violations
        ),
    }


def main() -> None:
    forecast_data = build_forecast_dataset()

    configurations = []

    for validation_samples in [1, 2, 3]:
        for test_samples in [1, 2, 3]:
            if validation_samples != test_samples:
                continue

            for minimum_train_samples in [4, 8, 12]:
                configurations.append(
                    {
                        "validation_samples": (
                            validation_samples
                        ),
                        "test_samples": test_samples,
                        "minimum_train_samples": (
                            minimum_train_samples
                        ),
                    }
                )

    summaries = []
    split_results = {}

    for configuration in configurations:
        split_data, summary = (
            create_repository_aware_split(
                forecast_data=forecast_data,
                validation_samples=configuration[
                    "validation_samples"
                ],
                test_samples=configuration[
                    "test_samples"
                ],
                minimum_train_samples=configuration[
                    "minimum_train_samples"
                ],
            )
        )

        verification = verify_split(split_data)
        summary.update(verification)

        key = (
            configuration["validation_samples"],
            configuration["test_samples"],
            configuration["minimum_train_samples"],
        )

        split_results[key] = split_data
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    summary_df = summary_df.sort_values(
        [
            "minimum_train_samples",
            "validation_samples_per_repo",
        ]
    ).reset_index(drop=True)

    summary_path = (
        OUTPUT_DIR
        / "forecast_repository_aware_candidates.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    # Save the likely preferred candidate for closer inspection:
    # 2 validation observations, 2 test observations,
    # and at least 8 training observations per repository.
    preferred_key = (2, 2, 8)
    preferred_split = split_results[preferred_key]

    preferred_path = (
        OUTPUT_DIR
        / "forecast_repository_aware_candidate_2_2_8.csv"
    )

    preferred_split.to_csv(
        preferred_path,
        index=False,
    )

    preferred_summary = summary_df[
        (summary_df["validation_samples_per_repo"] == 2)
        & (summary_df["test_samples_per_repo"] == 2)
        & (summary_df["minimum_train_samples"] == 8)
    ]

    report_lines = [
        heading("REPOSITORY-AWARE FORECAST SPLIT AUDIT"),
        "Forecast target: future_4week_stars",
        "Target horizon: four consecutive weeks",
        "",
        "Each split is purged so that:",
        "1. Training target windows finish before validation starts.",
        "2. Validation target windows finish before testing starts.",
        "",
        "Candidate comparison:",
        summary_df.to_string(index=False),
        "",
        heading("PREFERRED CANDIDATE FOR REVIEW"),
        preferred_summary.to_string(index=False),
        "",
        f"Candidate split saved to: {preferred_path}",
        f"Summary saved to: {summary_path}",
    ]

    report = "\n".join(report_lines)

    report_path = (
        OUTPUT_DIR
        / "forecast_repository_aware_audit.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print(report)


if __name__ == "__main__":
    main()
