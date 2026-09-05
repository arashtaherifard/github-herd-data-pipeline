import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = (
    PROJECT_ROOT
    / "database"
    / "github_herd.db"
)

FORECAST_DIR = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "forecasting"
)

TRAIN_PATH = (
    FORECAST_DIR
    / "forecast_primary_train.csv"
)

VALIDATION_PATH = (
    FORECAST_DIR
    / "forecast_primary_validation.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "forecasting"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def heading(title: str) -> str:
    return (
        "\n"
        + "=" * 100
        + f"\n{title}\n"
        + "=" * 100
    )


def load_weekly_data() -> pd.DataFrame:
    with sqlite3.connect(DATABASE_PATH) as connection:
        weekly = pd.read_sql_query(
            """
            SELECT
                repo_id,
                repo_full_name,
                week,
                weekly_new_stars,
                cumulative_stars
            FROM weekly_timeseries
            """,
            connection,
        )

    # SQLite stores weeks as:
    # YYYY-MM-DD/YYYY-MM-DD
    weekly["week_start"] = pd.to_datetime(
        weekly["week"]
        .astype(str)
        .str.split("/", n=1)
        .str[0],
        errors="raise",
        utc=True,
    )

    for column in [
        "repo_id",
        "weekly_new_stars",
        "cumulative_stars",
    ]:
        weekly[column] = pd.to_numeric(
            weekly[column],
            errors="raise",
        )

    weekly = weekly.sort_values(
        ["repo_id", "week_start"]
    ).reset_index(drop=True)

    weekly["repository_row_index"] = (
        weekly.groupby("repo_id").cumcount()
    )

    weekly["cumulative_difference"] = (
        weekly.groupby("repo_id")[
            "cumulative_stars"
        ].diff()
    )

    # The first cumulative difference is unknown,
    # because observations before collection began
    # are not available.
    weekly["clean_weekly_new_stars"] = (
        weekly["cumulative_difference"]
    )

    weekly["clean_rolling_3week_mean"] = (
        weekly.groupby("repo_id")[
            "clean_weekly_new_stars"
        ].transform(
            lambda values: values.rolling(
                window=3,
                min_periods=3,
            ).mean()
        )
    )

    weekly["clean_rolling_4week_sum"] = (
        weekly.groupby("repo_id")[
            "clean_weekly_new_stars"
        ].transform(
            lambda values: values.rolling(
                window=4,
                min_periods=4,
            ).sum()
        )
    )

    return weekly


def audit_split(
    split_name: str,
    path: Path,
    weekly: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    frame = pd.read_csv(path)

    frame["repo_id"] = pd.to_numeric(
        frame["repo_id"],
        errors="raise",
    )

    frame["cutoff_week"] = pd.to_datetime(
        frame["cutoff_week"],
        errors="raise",
        utc=True,
    )

    source_columns = [
        "repo_id",
        "week_start",
        "repository_row_index",
        "weekly_new_stars",
        "cumulative_stars",
        "cumulative_difference",
        "clean_weekly_new_stars",
        "clean_rolling_3week_mean",
        "clean_rolling_4week_sum",
    ]

    merged = frame.merge(
        weekly[source_columns],
        left_on=[
            "repo_id",
            "cutoff_week",
        ],
        right_on=[
            "repo_id",
            "week_start",
        ],
        how="left",
        validate="many_to_one",
    )

    if merged["week_start"].isna().any():
        raise ValueError(
            f"{split_name}: some cutoff weeks "
            "could not be matched."
        )

    comparable = merged[
        merged[
            "clean_rolling_4week_sum"
        ].notna()
    ].copy()

    comparable[
        "rolling_4week_difference"
    ] = (
        comparable[
            "rolling_4week_sum_stars"
        ]
        - comparable[
            "clean_rolling_4week_sum"
        ]
    )

    mismatch_mask = ~np.isclose(
        comparable[
            "rolling_4week_sum_stars"
        ],
        comparable[
            "clean_rolling_4week_sum"
        ],
    )

    summary = {
        "split": split_name,
        "rows": int(len(merged)),
        "repositories": int(
            merged["repo_id"].nunique()
        ),
        "rows_without_four_clean_history_weeks": int(
            merged[
                "clean_rolling_4week_sum"
            ].isna().sum()
        ),
        "comparable_rows": int(
            len(comparable)
        ),
        "rolling_4week_mismatch_rows": int(
            mismatch_mask.sum()
        ),
        "rolling_4week_match_rows": int(
            (~mismatch_mask).sum()
        ),
        "maximum_absolute_difference": float(
            comparable[
                "rolling_4week_difference"
            ].abs().max()
        ),
    }

    return summary, merged


def main() -> None:
    weekly = load_weekly_data()

    first_rows = weekly[
        weekly["repository_row_index"] == 0
    ].copy()

    nonfirst_rows = weekly[
        weekly["repository_row_index"] > 0
    ].copy()

    first_row_matches_cumulative = np.isclose(
        first_rows["weekly_new_stars"],
        first_rows["cumulative_stars"],
    )

    nonfirst_flow_matches_difference = np.isclose(
        nonfirst_rows["weekly_new_stars"],
        nonfirst_rows[
            "cumulative_difference"
        ],
    )

    train_summary, train_merged = (
        audit_split(
            "train",
            TRAIN_PATH,
            weekly,
        )
    )

    validation_summary, validation_merged = (
        audit_split(
            "validation",
            VALIDATION_PATH,
            weekly,
        )
    )

    split_summary = pd.DataFrame(
        [
            train_summary,
            validation_summary,
        ]
    )

    split_summary.to_csv(
        OUTPUT_DIR
        / "weekly_snapshot_split_impact.csv",
        index=False,
    )

    weekly.to_csv(
        OUTPUT_DIR
        / "weekly_snapshot_recomputed_features.csv",
        index=False,
    )

    sample_repo_id = 858127

    sample_cutoff = pd.Timestamp(
        "2010-09-13",
        tz="UTC",
    )

    sample_history = weekly[
        (weekly["repo_id"] == sample_repo_id)
        & (
            weekly["week_start"]
            <= sample_cutoff
        )
    ].tail(8)

    sample_history.to_csv(
        OUTPUT_DIR
        / "weekly_snapshot_sample_history.csv",
        index=False,
    )

    train_comparable = train_merged[
        train_merged[
            "clean_rolling_4week_sum"
        ].notna()
    ].copy()

    train_comparable[
        "rolling_4week_difference"
    ] = (
        train_comparable[
            "rolling_4week_sum_stars"
        ]
        - train_comparable[
            "clean_rolling_4week_sum"
        ]
    )

    largest_differences = (
        train_comparable.sort_values(
            "rolling_4week_difference",
            key=lambda values: values.abs(),
            ascending=False,
        )
        .head(25)
    )

    largest_differences.to_csv(
        OUTPUT_DIR
        / "weekly_snapshot_largest_differences.csv",
        index=False,
    )

    report_lines = [
        heading(
            "WEEKLY INITIAL-SNAPSHOT AUDIT"
        ),
        f"Weekly rows: {len(weekly)}",
        (
            "Repositories: "
            f"{weekly['repo_id'].nunique()}"
        ),
        "",
        (
            "First rows where weekly_new_stars "
            "equals cumulative_stars: "
            f"{int(first_row_matches_cumulative.sum())}"
            f" / {len(first_rows)}"
        ),
        (
            "First-row equality fraction: "
            f"{first_row_matches_cumulative.mean():.6f}"
        ),
        "",
        (
            "Non-first rows where weekly_new_stars "
            "equals cumulative difference: "
            f"{int(nonfirst_flow_matches_difference.sum())}"
            f" / {len(nonfirst_rows)}"
        ),
        (
            "Non-first-row equality fraction: "
            f"{nonfirst_flow_matches_difference.mean():.6f}"
        ),
        heading(
            "IMPACT ON FORECASTING SPLITS"
        ),
        split_summary.to_string(index=False),
        heading(
            "PANDAS SAMPLE HISTORY"
        ),
        sample_history.to_string(index=False),
        heading(
            "LARGEST TRAINING DIFFERENCES"
        ),
        largest_differences[
            [
                "repo_id",
                "repo_full_name",
                "cutoff_week",
                "history_length",
                "rolling_4week_sum_stars",
                "clean_rolling_4week_sum",
                "rolling_4week_difference",
            ]
        ]
        .head(15)
        .to_string(index=False),
        "",
        (
            "Test split values were not loaded "
            "or evaluated."
        ),
    ]

    report = "\n".join(report_lines)

    report_path = (
        OUTPUT_DIR
        / "weekly_initial_snapshot_audit.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print(report)


if __name__ == "__main__":
    main()
