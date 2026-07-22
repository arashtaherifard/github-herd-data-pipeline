from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3" / "audit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def separator(title: str) -> str:
    return "\n" + "=" * 100 + f"\n{title}\n" + "=" * 100


def audit_classification() -> list[str]:
    path = PROCESSED_DIR / "herd_model_ready.csv"
    df = pd.read_csv(path)

    target = "became_high_growth"
    identifier_columns = ["repo_id"]

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in identifier_columns + [target] + constant_columns
    ]

    lines = [
        separator("CLASSIFICATION FEASIBILITY"),
        f"Rows: {len(df)}",
        f"Columns: {df.shape[1]}",
        f"Missing values: {int(df.isna().sum().sum())}",
        f"Duplicate rows: {int(df.duplicated().sum())}",
        f"Identifier columns to exclude: {identifier_columns}",
        f"Constant columns to exclude: {constant_columns}",
        f"Usable predictor count: {len(feature_columns)}",
        "",
        "Target counts:",
        df[target].value_counts().sort_index().to_string(),
    ]

    pd.DataFrame(
        {
            "feature": feature_columns,
            "dtype": [str(df[column].dtype) for column in feature_columns],
        }
    ).to_csv(OUTPUT_DIR / "classification_features.csv", index=False)

    return lines


def build_forecast_feasibility(weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for repo_id, group in weekly.groupby("repo_id", sort=False):
        group = group.sort_values("week_start").reset_index(drop=True)

        dates = group["week_start"]
        stars = group["weekly_new_stars"]

        for cutoff_index in range(len(group)):
            history_length = cutoff_index + 1

            for horizon in [1, 2, 4]:
                final_index = cutoff_index + horizon

                if final_index >= len(group):
                    continue

                expected_dates = [
                    dates.iloc[cutoff_index] + pd.Timedelta(weeks=offset)
                    for offset in range(1, horizon + 1)
                ]

                actual_dates = [
                    dates.iloc[cutoff_index + offset]
                    for offset in range(1, horizon + 1)
                ]

                consecutive = expected_dates == actual_dates

                if not consecutive:
                    continue

                future_stars = float(
                    stars.iloc[cutoff_index + 1 : final_index + 1].sum()
                )

                rows.append(
                    {
                        "repo_id": repo_id,
                        "cutoff_week": dates.iloc[cutoff_index],
                        "history_length": history_length,
                        "forecast_horizon_weeks": horizon,
                        "future_stars": future_stars,
                    }
                )

    return pd.DataFrame(rows)


def audit_time_series() -> tuple[list[str], pd.DataFrame]:
    path = PROCESSED_DIR / "weekly_timeseries_features.csv"
    weekly = pd.read_csv(path)

    weekly["week_start"] = pd.to_datetime(
        weekly["week_start"],
        errors="coerce",
        utc=True,
    )

    duplicate_repo_weeks = int(
        weekly.duplicated(subset=["repo_id", "week_start"]).sum()
    )

    weekly = weekly.sort_values(["repo_id", "week_start"]).reset_index(drop=True)

    weekly["week_gap_days"] = (
        weekly.groupby("repo_id")["week_start"]
        .diff()
        .dt.days
    )

    observed_gaps = weekly["week_gap_days"].dropna()
    exact_week_fraction = (
        float((observed_gaps == 7).mean())
        if len(observed_gaps)
        else 0.0
    )

    feasibility = build_forecast_feasibility(weekly)

    summary_rows = []

    for horizon in [1, 2, 4]:
        horizon_data = feasibility[
            feasibility["forecast_horizon_weeks"] == horizon
        ]

        for minimum_history in [1, 2, 4, 8]:
            eligible = horizon_data[
                horizon_data["history_length"] >= minimum_history
            ]

            summary_rows.append(
                {
                    "forecast_horizon_weeks": horizon,
                    "minimum_history_weeks": minimum_history,
                    "valid_samples": len(eligible),
                    "eligible_repositories": eligible["repo_id"].nunique(),
                    "mean_future_stars": (
                        eligible["future_stars"].mean()
                        if len(eligible)
                        else None
                    ),
                    "median_future_stars": (
                        eligible["future_stars"].median()
                        if len(eligible)
                        else None
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows)

    feasibility.to_csv(
        OUTPUT_DIR / "forecast_feasible_samples.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "forecast_horizon_summary.csv",
        index=False,
    )

    lines = [
        separator("TIME-SERIES FORECASTING FEASIBILITY"),
        f"Rows: {len(weekly)}",
        f"Repositories: {weekly['repo_id'].nunique()}",
        f"Duplicate repository-week pairs: {duplicate_repo_weeks}",
        f"Unparseable week dates: {int(weekly['week_start'].isna().sum())}",
        f"Observed week-to-week gaps: {len(observed_gaps)}",
        f"Fraction of observed gaps equal to exactly 7 days: {exact_week_fraction:.4f}",
        "",
        "Valid supervised forecasting samples:",
        summary.to_string(index=False),
    ]

    return lines, summary


def audit_recommendation() -> list[str]:
    path = PROCESSED_DIR / "recommendation_features.csv"
    interactions = pd.read_csv(path)

    interactions["starred_at"] = pd.to_datetime(
        interactions["starred_at"],
        errors="coerce",
        utc=True,
    )

    duplicate_pairs = int(
        interactions.duplicated(subset=["user_id", "repo_id"]).sum()
    )

    user_activity = interactions.groupby("user_id").size()
    repo_activity = interactions.groupby("repo_id").size()

    maximum_collected_count = int(repo_activity.max())
    repositories_at_maximum = int(
        (repo_activity == maximum_collected_count).sum()
    )

    eligible_two = int((user_activity >= 2).sum())
    eligible_three = int((user_activity >= 3).sum())
    eligible_five = int((user_activity >= 5).sum())

    matrix_size = (
        interactions["user_id"].nunique()
        * interactions["repo_id"].nunique()
    )

    density = (
        len(interactions) / matrix_size
        if matrix_size
        else 0.0
    )

    user_summary = (
        user_activity.value_counts()
        .sort_index()
        .rename_axis("interaction_count")
        .reset_index(name="number_of_users")
    )

    repo_summary = (
        repo_activity.rename("interaction_count")
        .reset_index()
        .sort_values("interaction_count", ascending=False)
    )

    user_summary.to_csv(
        OUTPUT_DIR / "recommendation_user_activity.csv",
        index=False,
    )
    repo_summary.to_csv(
        OUTPUT_DIR / "recommendation_repo_activity.csv",
        index=False,
    )

    lines = [
        separator("RECOMMENDATION FEASIBILITY"),
        f"Interactions: {len(interactions)}",
        f"Unique users: {interactions['user_id'].nunique()}",
        f"Unique repositories: {interactions['repo_id'].nunique()}",
        f"Duplicate user-repository pairs: {duplicate_pairs}",
        f"Unparseable timestamps: {int(interactions['starred_at'].isna().sum())}",
        f"Earliest interaction: {interactions['starred_at'].min()}",
        f"Latest interaction: {interactions['starred_at'].max()}",
        f"Interaction matrix density: {density:.8f}",
        f"Interaction matrix sparsity: {1 - density:.8f}",
        "",
        f"Users eligible for train/test splitting (>=2 interactions): {eligible_two}",
        f"Users eligible for train/validation/test splitting (>=3 interactions): {eligible_three}",
        f"Users with at least 5 interactions: {eligible_five}",
        "",
        f"Maximum collected interactions for a repository: {maximum_collected_count}",
        f"Repositories at the maximum collected count: {repositories_at_maximum}",
        f"Fraction of repositories at the maximum: "
        f"{repositories_at_maximum / len(repo_activity):.4f}",
    ]

    return lines


def main() -> None:
    report_lines = []

    report_lines.extend(audit_classification())

    time_series_lines, _ = audit_time_series()
    report_lines.extend(time_series_lines)

    report_lines.extend(audit_recommendation())

    report_lines.extend(
        [
            separator("AUDIT COMPLETE"),
            f"Outputs saved to: {OUTPUT_DIR}",
        ]
    )

    report = "\n".join(report_lines)

    report_path = OUTPUT_DIR / "phase3_feasibility_audit.txt"
    report_path.write_text(report, encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
