import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "phase3_config.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3" / "audit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    PHASE3_CONFIG = json.load(file)

FORECAST_SETTINGS = PHASE3_CONFIG["forecasting"]

FORECAST_HORIZON = int(
    FORECAST_SETTINGS["forecast_horizon_weeks"]
)

INITIAL_SNAPSHOT_ROWS = int(
    FORECAST_SETTINGS["initial_snapshot_rows"]
)

MINIMUM_CLEAN_HISTORY_WEEKS = int(
    FORECAST_SETTINGS["minimum_clean_history_weeks"]
)

MINIMUM_OBSERVED_ROWS = int(
    FORECAST_SETTINGS["minimum_observed_rows"]
)

RANDOM_STATE = int(
    PHASE3_CONFIG["random_state"]
)

if MINIMUM_OBSERVED_ROWS != (
    INITIAL_SNAPSHOT_ROWS
    + MINIMUM_CLEAN_HISTORY_WEEKS
):
    raise ValueError(
        "minimum_observed_rows must equal "
        "initial_snapshot_rows plus "
        "minimum_clean_history_weeks."
    )


def heading(title: str) -> str:
    return "\n" + "=" * 100 + f"\n{title}\n" + "=" * 100


def build_forecast_dataset() -> pd.DataFrame:
    weekly = pd.read_csv(
        PROCESSED_DIR / "weekly_timeseries_features.csv"
    )

    weekly["week_start"] = pd.to_datetime(
        weekly["week_start"],
        errors="coerce",
        utc=True,
    )

    weekly = weekly.sort_values(
        ["repo_id", "week_start"]
    ).reset_index(drop=True)

    feature_columns = [
        "weekly_new_stars",
        "cumulative_stars",
        "previous_week_stars",
        "weekly_growth_rate",
        "herd_momentum_score",
        "lag_1_week_stars",
        "lag_2_week_stars",
        "rolling_3week_mean_stars",
        "rolling_4week_sum_stars",
        "weekly_growth_acceleration",
        "cumulative_growth_lag_1",
    ]

    samples = []

    for repo_id, group in weekly.groupby("repo_id", sort=False):
        group = group.sort_values("week_start").reset_index(drop=True)

        for cutoff_index in range(len(group)):
            history_length = cutoff_index + 1

            if history_length < MINIMUM_OBSERVED_ROWS:
                continue

            final_index = cutoff_index + FORECAST_HORIZON

            if final_index >= len(group):
                continue

            cutoff_week = group.loc[cutoff_index, "week_start"]

            expected_future_dates = [
                cutoff_week + pd.Timedelta(weeks=offset)
                for offset in range(1, FORECAST_HORIZON + 1)
            ]

            actual_future_dates = [
                group.loc[cutoff_index + offset, "week_start"]
                for offset in range(1, FORECAST_HORIZON + 1)
            ]

            if expected_future_dates != actual_future_dates:
                continue

            future_rows = group.iloc[
                cutoff_index + 1 : final_index + 1
            ]

            sample = {
                "repo_id": repo_id,
                "repo_full_name": group.loc[
                    cutoff_index, "repo_full_name"
                ],
                "cutoff_week": cutoff_week,
                "target_end_week": group.loc[
                    final_index, "week_start"
                ],
                "history_length": history_length,
                "initial_snapshot_rows": INITIAL_SNAPSHOT_ROWS,
                "clean_history_weeks": (
                    history_length - INITIAL_SNAPSHOT_ROWS
                ),
                "future_4week_stars": float(
                    future_rows["weekly_new_stars"].sum()
                ),
            }

            for column in feature_columns:
                sample[column] = group.loc[cutoff_index, column]

            samples.append(sample)

    return pd.DataFrame(samples)


def choose_date_cutoffs(
    forecast_data: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique_dates = sorted(
        forecast_data["cutoff_week"].dropna().unique()
    )

    train_index = int((len(unique_dates) - 1) * 0.70)
    validation_index = int((len(unique_dates) - 1) * 0.85)

    train_end = pd.Timestamp(unique_dates[train_index])
    validation_end = pd.Timestamp(unique_dates[validation_index])

    return train_end, validation_end


def audit_forecast_splits() -> list[str]:
    forecast_data = build_forecast_dataset()

    train_end, validation_end = choose_date_cutoffs(
        forecast_data
    )

    train = forecast_data[
        forecast_data["target_end_week"] <= train_end
    ].copy()

    validation = forecast_data[
        (forecast_data["cutoff_week"] > train_end)
        & (
            forecast_data["target_end_week"]
            <= validation_end
        )
    ].copy()

    test = forecast_data[
        forecast_data["cutoff_week"] > validation_end
    ].copy()

    assigned_indices = (
        set(train.index)
        | set(validation.index)
        | set(test.index)
    )

    dropped_for_purging = (
        len(forecast_data) - len(assigned_indices)
    )

    train_repositories = set(train["repo_id"])
    validation_repositories = set(validation["repo_id"])
    test_repositories = set(test["repo_id"])

    cold_validation_repositories = (
        validation_repositories - train_repositories
    )

    cold_test_repositories = (
        test_repositories - train_repositories
    )

    split_frames = []

    for split_name, frame in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        temporary = frame.copy()
        temporary["split"] = split_name
        split_frames.append(temporary)

    split_data = pd.concat(
        split_frames,
        ignore_index=True,
    )

    split_data.to_csv(
        OUTPUT_DIR / "forecast_split_plan.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "split": split_name,
                "rows": len(frame),
                "repositories": frame["repo_id"].nunique(),
                "start_cutoff": (
                    frame["cutoff_week"].min()
                    if len(frame)
                    else None
                ),
                "end_cutoff": (
                    frame["cutoff_week"].max()
                    if len(frame)
                    else None
                ),
                "target_mean": (
                    frame["future_4week_stars"].mean()
                    if len(frame)
                    else None
                ),
                "target_median": (
                    frame["future_4week_stars"].median()
                    if len(frame)
                    else None
                ),
            }
            for split_name, frame in [
                ("train", train),
                ("validation", validation),
                ("test", test),
            ]
        ]
    )

    summary.to_csv(
        OUTPUT_DIR / "forecast_split_summary.csv",
        index=False,
    )

    samples_per_repo = (
        forecast_data.groupby("repo_id")
        .size()
        .rename("supervised_samples")
    )

    per_repo_feasibility = pd.DataFrame(
        {
            "minimum_samples": [5, 8, 10, 15, 20],
            "eligible_repositories": [
                int((samples_per_repo >= threshold).sum())
                for threshold in [5, 8, 10, 15, 20]
            ],
        }
    )

    per_repo_feasibility.to_csv(
        OUTPUT_DIR / "forecast_repo_split_feasibility.csv",
        index=False,
    )

    lines = [
        heading("FORECAST SPLIT PLAN"),
        f"Forecast horizon: {FORECAST_HORIZON} weeks",
        (
            "Minimum clean history: "
            f"{MINIMUM_CLEAN_HISTORY_WEEKS} weeks"
        ),
        (
            "Minimum observed rows including the initial snapshot: "
            f"{MINIMUM_OBSERVED_ROWS}"
        ),
        f"Total supervised samples: {len(forecast_data)}",
        f"Repositories: {forecast_data['repo_id'].nunique()}",
        f"Training boundary: {train_end}",
        f"Validation boundary: {validation_end}",
        f"Samples dropped by temporal purging: {dropped_for_purging}",
        "",
        "Split summary:",
        summary.to_string(index=False),
        "",
        "Repositories appearing in validation but not training:",
        str(len(cold_validation_repositories)),
        "Repositories appearing in test but not training:",
        str(len(cold_test_repositories)),
        "",
        "Per-repository split feasibility:",
        per_repo_feasibility.to_string(index=False),
    ]

    return lines


def audit_classification_split() -> list[str]:
    classification = pd.read_csv(
        PROCESSED_DIR / "herd_model_ready.csv"
    )

    development, test = train_test_split(
        classification,
        test_size=0.20,
        stratify=classification["became_high_growth"],
        random_state=RANDOM_STATE,
    )

    summary = pd.DataFrame(
        [
            {
                "split": "development",
                "rows": len(development),
                "class_0": int(
                    (development["became_high_growth"] == 0).sum()
                ),
                "class_1": int(
                    (development["became_high_growth"] == 1).sum()
                ),
            },
            {
                "split": "test",
                "rows": len(test),
                "class_0": int(
                    (test["became_high_growth"] == 0).sum()
                ),
                "class_1": int(
                    (test["became_high_growth"] == 1).sum()
                ),
            },
        ]
    )

    summary.to_csv(
        OUTPUT_DIR / "classification_split_summary.csv",
        index=False,
    )

    split_assignments = pd.DataFrame(
        {
            "repo_id": pd.concat(
                [
                    development["repo_id"],
                    test["repo_id"],
                ]
            ),
            "dataset_split": (
                ["development"] * len(development)
                + ["test"] * len(test)
            ),
        }
    )

    split_assignments.to_csv(
        OUTPUT_DIR / "classification_split_assignments.csv",
        index=False,
    )

    lines = [
        heading("CLASSIFICATION SPLIT PLAN"),
        "Development/test strategy: stratified 80/20 split",
        "Validation strategy: 5-fold stratified CV inside development data",
        f"Random state: {RANDOM_STATE}",
        "",
        summary.to_string(index=False),
    ]

    return lines


def audit_recommendation_split() -> list[str]:
    interactions = pd.read_csv(
        PROCESSED_DIR / "recommendation_features.csv"
    )

    interactions["starred_at"] = pd.to_datetime(
        interactions["starred_at"],
        errors="coerce",
        utc=True,
    )

    interactions = interactions.sort_values(
        ["user_id", "starred_at", "repo_id"]
    ).reset_index(drop=True)

    user_counts = interactions.groupby("user_id").size()

    eligible_users = set(
        user_counts[user_counts >= 3].index
    )

    eligible_mask = interactions["user_id"].isin(
        eligible_users
    )

    eligible = interactions[eligible_mask].copy()
    ineligible = interactions[~eligible_mask].copy()

    eligible["reverse_order"] = (
        eligible.groupby("user_id").cumcount(ascending=False)
    )

    test = eligible[
        eligible["reverse_order"] == 0
    ].copy()

    validation = eligible[
        eligible["reverse_order"] == 1
    ].copy()

    eligible_train = eligible[
        eligible["reverse_order"] >= 2
    ].copy()

    train = pd.concat(
        [eligible_train, ineligible],
        ignore_index=True,
    )

    train = train.drop(columns=["reverse_order"], errors="ignore")
    validation = validation.drop(
        columns=["reverse_order"],
        errors="ignore",
    )
    test = test.drop(
        columns=["reverse_order"],
        errors="ignore",
    )

    train_repositories = set(train["repo_id"])
    validation_repositories = set(validation["repo_id"])
    test_repositories = set(test["repo_id"])

    cold_validation_items = (
        validation_repositories - train_repositories
    )

    cold_test_items = (
        test_repositories - train_repositories
    )

    train.to_csv(
        OUTPUT_DIR / "recommendation_train.csv",
        index=False,
    )

    validation.to_csv(
        OUTPUT_DIR / "recommendation_validation.csv",
        index=False,
    )

    test.to_csv(
        OUTPUT_DIR / "recommendation_test.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "split": split_name,
                "rows": len(frame),
                "users": frame["user_id"].nunique(),
                "repositories": frame["repo_id"].nunique(),
                "earliest_timestamp": frame["starred_at"].min(),
                "latest_timestamp": frame["starred_at"].max(),
            }
            for split_name, frame in [
                ("train", train),
                ("validation", validation),
                ("test", test),
            ]
        ]
    )

    summary.to_csv(
        OUTPUT_DIR / "recommendation_split_summary.csv",
        index=False,
    )

    validation_order = validation[
        ["user_id", "starred_at"]
    ].rename(
        columns={"starred_at": "validation_time"}
    )

    test_order = test[
        ["user_id", "starred_at"]
    ].rename(
        columns={"starred_at": "test_time"}
    )

    ordering_check = validation_order.merge(
        test_order,
        on="user_id",
        how="inner",
    )

    time_order_violations = int(
        (
            ordering_check["validation_time"]
            > ordering_check["test_time"]
        ).sum()
    )

    lines = [
        heading("RECOMMENDATION SPLIT PLAN"),
        f"Users eligible for validation and test: {len(eligible_users)}",
        "Split strategy: leave-two-out by timestamp",
        "Second-most-recent interaction: validation",
        "Most recent interaction: test",
        "All earlier and non-evaluation-user interactions: training",
        "",
        summary.to_string(index=False),
        "",
        f"Cold validation repositories: {len(cold_validation_items)}",
        f"Cold test repositories: {len(cold_test_items)}",
        f"Validation/test ordering violations: {time_order_violations}",
    ]

    return lines


def main() -> None:
    report_lines = []

    report_lines.extend(audit_classification_split())
    report_lines.extend(audit_forecast_splits())
    report_lines.extend(audit_recommendation_split())

    report_lines.extend(
        [
            heading("SPLIT PLANNING COMPLETE"),
            f"Outputs saved to: {OUTPUT_DIR}",
        ]
    )

    report = "\n".join(report_lines)

    report_path = OUTPUT_DIR / "phase3_split_planning.txt"
    report_path.write_text(report, encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
