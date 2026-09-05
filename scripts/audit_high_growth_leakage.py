import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_READY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "herd_model_ready.csv"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "database"
    / "github_herd.db"
)

SCHEMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "modeling"
    / "classification"
    / "classification_schema.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "phase3"
    / "classification"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "became_high_growth"


def heading(title: str) -> str:
    return (
        "\n"
        + "=" * 100
        + f"\n{title}\n"
        + "=" * 100
    )


def best_threshold_accuracy(
    values: pd.Series,
    target: pd.Series,
) -> dict:
    x = np.asarray(values, dtype=float)
    y = np.asarray(target, dtype=int)

    unique_values = np.unique(x)

    if len(unique_values) == 1:
        return {
            "best_threshold_accuracy": 0.5,
            "best_threshold": float(unique_values[0]),
            "threshold_direction": "constant",
        }

    midpoints = (
        unique_values[:-1] + unique_values[1:]
    ) / 2.0

    thresholds = np.concatenate(
        [
            [unique_values[0] - 1e-9],
            midpoints,
            [unique_values[-1] + 1e-9],
        ]
    )

    best_accuracy = -1.0
    best_threshold = None
    best_direction = None

    for threshold in thresholds:
        predictions_high = (
            x >= threshold
        ).astype(int)

        predictions_low = (
            x < threshold
        ).astype(int)

        accuracy_high = float(
            np.mean(predictions_high == y)
        )

        accuracy_low = float(
            np.mean(predictions_low == y)
        )

        if accuracy_high > best_accuracy:
            best_accuracy = accuracy_high
            best_threshold = threshold
            best_direction = "class_1_if_value_at_least_threshold"

        if accuracy_low > best_accuracy:
            best_accuracy = accuracy_low
            best_threshold = threshold
            best_direction = "class_1_if_value_below_threshold"

    return {
        "best_threshold_accuracy": best_accuracy,
        "best_threshold": float(best_threshold),
        "threshold_direction": best_direction,
    }


def read_herd_modeling_table() -> pd.DataFrame:
    with sqlite3.connect(DATABASE_PATH) as connection:
        tables = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """,
            connection,
        )

        print(heading("DATABASE TABLES"))
        print(tables.to_string(index=False))

        table_info = pd.read_sql_query(
            "PRAGMA table_info(herd_modeling)",
            connection,
        )

        print(heading("HERD_MODELING TABLE COLUMNS"))
        print(table_info.to_string(index=False))

        return pd.read_sql_query(
            "SELECT * FROM herd_modeling",
            connection,
        )


def audit_raw_target_relationships(
    raw: pd.DataFrame,
) -> list[str]:
    lines = [heading("RAW TARGET RELATIONSHIPS")]

    lines.append(f"Rows: {len(raw)}")
    lines.append(
        "Columns: " + ", ".join(raw.columns)
    )

    required = {
        TARGET,
        "early_4week_stars",
        "later_stars",
    }

    if not required.issubset(raw.columns):
        missing = sorted(required - set(raw.columns))
        lines.append(
            "Could not run the main cap audit. "
            f"Missing columns: {missing}"
        )
        return lines

    raw = raw.copy()

    raw["collected_period_total"] = (
        raw["early_4week_stars"]
        + raw["later_stars"]
    )

    lines.extend(
        [
            "",
            "Class-wise early/later star summary:",
            raw.groupby(TARGET)[
                [
                    "early_4week_stars",
                    "later_stars",
                    "collected_period_total",
                ]
            ]
            .agg(
                [
                    "count",
                    "min",
                    "median",
                    "mean",
                    "max",
                ]
            )
            .to_string(),
            "",
            (
                "Correlation between early_4week_stars "
                "and later_stars: "
                f"{raw['early_4week_stars'].corr(raw['later_stars']):.6f}"
            ),
            (
                "Repositories where early + later = 1500: "
                f"{int((raw['collected_period_total'] == 1500).sum())}"
                f" / {len(raw)}"
            ),
            (
                "Fraction where early + later = 1500: "
                f"{float((raw['collected_period_total'] == 1500).mean()):.6f}"
            ),
            (
                "Unique collected-period totals: "
                f"{raw['collected_period_total'].nunique()}"
            ),
            (
                "Collected-period total minimum/median/maximum: "
                f"{raw['collected_period_total'].min()} / "
                f"{raw['collected_period_total'].median()} / "
                f"{raw['collected_period_total'].max()}"
            ),
        ]
    )

    target_zero = raw[raw[TARGET] == 0]
    target_one = raw[raw[TARGET] == 1]

    lines.extend(
        [
            "",
            (
                "Class 0 later_stars range: "
                f"{target_zero['later_stars'].min()} to "
                f"{target_zero['later_stars'].max()}"
            ),
            (
                "Class 1 later_stars range: "
                f"{target_one['later_stars'].min()} to "
                f"{target_one['later_stars'].max()}"
            ),
            (
                "Class 0 early_4week_stars range: "
                f"{target_zero['early_4week_stars'].min()} to "
                f"{target_zero['early_4week_stars'].max()}"
            ),
            (
                "Class 1 early_4week_stars range: "
                f"{target_one['early_4week_stars'].min()} to "
                f"{target_one['early_4week_stars'].max()}"
            ),
        ]
    )

    later_threshold = best_threshold_accuracy(
        raw["later_stars"],
        raw[TARGET],
    )

    early_threshold = best_threshold_accuracy(
        raw["early_4week_stars"],
        raw[TARGET],
    )

    lines.extend(
        [
            "",
            "Best single-threshold prediction using later_stars:",
            str(later_threshold),
            "",
            "Best single-threshold prediction using early_4week_stars:",
            str(early_threshold),
        ]
    )

    raw[
        [
            "repo_id",
            TARGET,
            "early_4week_stars",
            "later_stars",
            "collected_period_total",
        ]
    ].to_csv(
        OUTPUT_DIR / "target_cap_relationship.csv",
        index=False,
    )

    return lines


def audit_model_features(
    model_ready: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame]:
    lines = [heading("UNIVARIATE FEATURE AUDIT")]

    y = model_ready[TARGET]
    feature_rows = []

    excluded = {
        "repo_id",
        TARGET,
    }

    for feature in model_ready.columns:
        if feature in excluded:
            continue

        values = model_ready[feature]

        if not pd.api.types.is_numeric_dtype(values):
            continue

        unique_count = values.nunique(
            dropna=False
        )

        if unique_count <= 1:
            feature_rows.append(
                {
                    "feature": feature,
                    "unique_values": unique_count,
                    "direct_auc": None,
                    "predictive_auc": None,
                    "auc_direction": "constant",
                    "best_threshold_accuracy": None,
                    "best_threshold": None,
                    "threshold_direction": "constant",
                    "class_0_mean": float(
                        values[y == 0].mean()
                    ),
                    "class_1_mean": float(
                        values[y == 1].mean()
                    ),
                }
            )
            continue

        direct_auc = float(
            roc_auc_score(y, values)
        )

        if direct_auc >= 0.5:
            predictive_auc = direct_auc
            auc_direction = (
                "higher_values_predict_class_1"
            )
        else:
            predictive_auc = 1.0 - direct_auc
            auc_direction = (
                "lower_values_predict_class_1"
            )

        threshold_result = (
            best_threshold_accuracy(values, y)
        )

        feature_rows.append(
            {
                "feature": feature,
                "unique_values": unique_count,
                "direct_auc": direct_auc,
                "predictive_auc": predictive_auc,
                "auc_direction": auc_direction,
                **threshold_result,
                "class_0_mean": float(
                    values[y == 0].mean()
                ),
                "class_1_mean": float(
                    values[y == 1].mean()
                ),
            }
        )

    results = pd.DataFrame(feature_rows)

    results = results.sort_values(
        [
            "predictive_auc",
            "best_threshold_accuracy",
        ],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    results.insert(
        0,
        "rank",
        np.arange(1, len(results) + 1),
    )

    results.to_csv(
        OUTPUT_DIR
        / "classification_univariate_leakage_audit.csv",
        index=False,
    )

    lines.extend(
        [
            "",
            "Top univariate predictors:",
            results.head(15).to_string(index=False),
        ]
    )

    top_features = results.head(8)["feature"].tolist()

    class_summaries = []

    for feature in top_features:
        summary = (
            model_ready.groupby(TARGET)[feature]
            .agg(
                [
                    "count",
                    "min",
                    "median",
                    "mean",
                    "max",
                ]
            )
            .reset_index()
        )

        summary.insert(0, "feature", feature)
        class_summaries.append(summary)

    if class_summaries:
        class_summary = pd.concat(
            class_summaries,
            ignore_index=True,
        )

        class_summary.to_csv(
            OUTPUT_DIR
            / "top_feature_class_distributions.csv",
            index=False,
        )

        lines.extend(
            [
                "",
                "Top feature distributions by class:",
                class_summary.to_string(index=False),
            ]
        )

    return lines, results


def main() -> None:
    model_ready = pd.read_csv(
        MODEL_READY_PATH
    )

    raw = read_herd_modeling_table()

    report_lines = [
        heading("HIGH-GROWTH TARGET LEAKAGE AUDIT"),
        f"Model-ready rows: {len(model_ready)}",
        f"Model-ready columns: {model_ready.shape[1]}",
        (
            "Target counts: "
            + str(
                model_ready[TARGET]
                .value_counts()
                .sort_index()
                .to_dict()
            )
        ),
    ]

    report_lines.extend(
        audit_raw_target_relationships(raw)
    )

    feature_lines, _ = audit_model_features(
        model_ready
    )

    report_lines.extend(feature_lines)

    report_lines.extend(
        [
            heading("AUDIT OUTPUTS"),
            (
                "Results saved to: "
                f"{OUTPUT_DIR}"
            ),
            (
                "Important: do not evaluate the "
                "untouched test set until this audit "
                "has been reviewed."
            ),
        ]
    )

    report = "\n".join(report_lines)

    report_path = (
        OUTPUT_DIR
        / "classification_leakage_audit.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print(report)


if __name__ == "__main__":
    main()
