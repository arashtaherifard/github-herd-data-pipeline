from pathlib import Path
import ast
import numpy as np
import pandas as pd
from database_connection import get_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "section2"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

engine = get_engine()

def load_table(table_name):
    with engine.connect() as conn:
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)

def safe_to_datetime(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = df[col].dt.tz_convert(None)
    return df

def parse_week_start(value):
    if pd.isna(value):
        return pd.NaT
    text = str(value)
    if "/" in text:
        text = text.split("/")[0]
    return pd.to_datetime(text, errors="coerce")

def parse_topic_count(value):
    if pd.isna(value):
        return 0
    if isinstance(value, list):
        return len(value)
    text = str(value).strip()
    if text == "" or text == "[]":
        return 0
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return len(parsed)
    except Exception:
        pass
    if "," in text:
        return len([x for x in text.split(",") if x.strip()])
    return 1

def clean_text(value):
    if pd.isna(value):
        return "No Description"
    text = str(value).strip()
    if text == "":
        return "No Description"
    return text

repositories = load_table("repositories")
stars = load_table("stars")
interactions = load_table("user_repo_interactions")
daily = load_table("daily_timeseries")
weekly = load_table("weekly_timeseries")
herd = load_table("herd_modeling")

summary_lines = []
summary_lines.append("Section 2 Preprocessing Summary")
summary_lines.append("=" * 50)
summary_lines.append("")

repositories = safe_to_datetime(repositories, ["created_at", "updated_at", "pushed_at", "collected_at"])

if "language" in repositories.columns:
    repositories["language"] = repositories["language"].fillna("Unknown").replace("", "Unknown")

if "license" in repositories.columns:
    repositories["license"] = repositories["license"].fillna("No License").replace("", "No License")

if "homepage" in repositories.columns:
    repositories["has_homepage"] = repositories["homepage"].notna().astype(int)
    repositories["homepage"] = repositories["homepage"].fillna("No Homepage")

if "description" in repositories.columns:
    repositories["description"] = repositories["description"].apply(clean_text)
    repositories["description_length"] = repositories["description"].str.len()

if "topics" in repositories.columns:
    repositories["topics"] = repositories["topics"].fillna("[]")
    repositories["topic_count"] = repositories["topics"].apply(parse_topic_count)

reference_date = pd.Timestamp("2026-06-14")

if "created_at" in repositories.columns:
    repositories["repo_age_days"] = (reference_date - repositories["created_at"]).dt.days
    repositories["repo_age_days"] = repositories["repo_age_days"].clip(lower=0)
    repositories["created_year"] = repositories["created_at"].dt.year
    repositories["created_month"] = repositories["created_at"].dt.month
    repositories["created_weekday"] = repositories["created_at"].dt.weekday

if "updated_at" in repositories.columns:
    repositories["days_since_update"] = (reference_date - repositories["updated_at"]).dt.days
    repositories["days_since_update"] = repositories["days_since_update"].clip(lower=0)

if "pushed_at" in repositories.columns:
    repositories["days_since_push"] = (reference_date - repositories["pushed_at"]).dt.days
    repositories["days_since_push"] = repositories["days_since_push"].clip(lower=0)

numeric_repo_cols = [
    "stars_count",
    "forks_count",
    "watchers_count",
    "open_issues_count",
    "repo_age_days",
    "days_since_update",
    "days_since_push",
    "topic_count",
    "description_length"
]

for col in numeric_repo_cols:
    if col in repositories.columns:
        repositories[col] = pd.to_numeric(repositories[col], errors="coerce")

for col in ["stars_count", "forks_count", "watchers_count", "open_issues_count", "repo_age_days", "days_since_update", "days_since_push", "topic_count", "description_length"]:
    if col in repositories.columns:
        repositories[col] = repositories[col].fillna(repositories[col].median())

if "stars_count" in repositories.columns and "repo_age_days" in repositories.columns:
    repositories["stars_per_day"] = repositories["stars_count"] / (repositories["repo_age_days"] + 1)

if "forks_count" in repositories.columns and "stars_count" in repositories.columns:
    repositories["forks_per_star"] = repositories["forks_count"] / (repositories["stars_count"] + 1)

if "open_issues_count" in repositories.columns and "stars_count" in repositories.columns:
    repositories["issues_per_star"] = repositories["open_issues_count"] / (repositories["stars_count"] + 1)

stars = safe_to_datetime(stars, ["starred_at", "collected_at"])
interactions = safe_to_datetime(interactions, ["starred_at", "collected_at"])
daily = safe_to_datetime(daily, ["date"])
if "week" in weekly.columns:
    weekly["week_start"] = weekly["week"].apply(parse_week_start)

for df_name, df in [("stars", stars), ("user_repo_interactions", interactions)]:
    if "repo_id" in df.columns:
        df["repo_id"] = pd.to_numeric(df["repo_id"], errors="coerce")
    if "user_id" in df.columns:
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    if "username" in df.columns:
        df["username"] = df["username"].fillna("unknown_user")

if "interaction" in interactions.columns:
    interactions["interaction"] = pd.to_numeric(interactions["interaction"], errors="coerce").fillna(1).astype(int)

for col in ["daily_new_stars", "cumulative_stars", "daily_growth_rate"]:
    if col in daily.columns:
        daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0)

for col in ["weekly_new_stars", "cumulative_stars", "previous_week_stars", "weekly_growth_rate", "herd_momentum_score"]:
    if col in weekly.columns:
        weekly[col] = pd.to_numeric(weekly[col], errors="coerce").fillna(0)

if "repo_id" in herd.columns:
    herd["repo_id"] = pd.to_numeric(herd["repo_id"], errors="coerce")

if "language" in herd.columns:
    herd["language"] = herd["language"].fillna("Unknown").replace("", "Unknown")

if "license" in herd.columns:
    herd["license"] = herd["license"].fillna("No License").replace("", "No License")

if "topics" in herd.columns:
    herd["topics"] = herd["topics"].fillna("[]")
    herd["topic_count"] = herd["topics"].apply(parse_topic_count)

herd_numeric_cols = [
    "stars_count",
    "forks_count",
    "watchers_count",
    "open_issues_count",
    "early_4week_stars",
    "early_avg_weekly_stars",
    "early_max_weekly_stars",
    "later_stars",
    "later_avg_weekly_stars",
    "later_max_weekly_stars",
    "became_high_growth",
    "topic_count"
]

for col in herd_numeric_cols:
    if col in herd.columns:
        herd[col] = pd.to_numeric(herd[col], errors="coerce")

for col in herd_numeric_cols:
    if col in herd.columns:
        if col == "became_high_growth":
            herd[col] = herd[col].fillna(0).astype(int)
        else:
            herd[col] = herd[col].fillna(herd[col].median())

repositories.to_csv(PROCESSED_DIR / "repositories_preprocessed.csv", index=False)
stars.to_csv(PROCESSED_DIR / "stars_preprocessed.csv", index=False)
interactions.to_csv(PROCESSED_DIR / "user_repo_interactions_preprocessed.csv", index=False)
daily.to_csv(PROCESSED_DIR / "daily_timeseries_preprocessed.csv", index=False)
weekly.to_csv(PROCESSED_DIR / "weekly_timeseries_preprocessed.csv", index=False)
herd.to_csv(PROCESSED_DIR / "herd_modeling_preprocessed.csv", index=False)

processed_files = {
    "repositories_preprocessed.csv": repositories,
    "stars_preprocessed.csv": stars,
    "user_repo_interactions_preprocessed.csv": interactions,
    "daily_timeseries_preprocessed.csv": daily,
    "weekly_timeseries_preprocessed.csv": weekly,
    "herd_modeling_preprocessed.csv": herd
}

summary_lines.append("Generated Preprocessed Files")
summary_lines.append("-" * 50)
for name, df in processed_files.items():
    summary_lines.append(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")
summary_lines.append("")

summary_lines.append("Main Preprocessing Decisions")
summary_lines.append("-" * 50)
summary_lines.append("Missing programming languages were replaced with Unknown.")
summary_lines.append("Missing license values were replaced with No License.")
summary_lines.append("Missing homepage values were converted into the binary feature has_homepage.")
summary_lines.append("Repository timestamps were converted into datetime format.")
summary_lines.append("Repository age, update recency, push recency, topic count, and ratio-based repository features were created.")
summary_lines.append("Growth-rate missing values in time-series tables were filled with 0 when caused by unavailable previous time steps.")
summary_lines.append("Numeric columns were converted to numeric types and missing numeric values were filled using median values.")
summary_lines.append("User-repository interactions were kept as implicit feedback with interaction = 1.")
summary_lines.append("")

missing_after = []
for name, df in processed_files.items():
    total_missing = int(df.isna().sum().sum())
    missing_after.append({"file": name, "total_missing_values": total_missing})
    summary_lines.append(f"{name}: {total_missing} total missing values after preprocessing")

pd.DataFrame(missing_after).to_csv(OUTPUT_DIR / "missing_after_preprocessing.csv", index=False)

with open(OUTPUT_DIR / "preprocessing_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("Preprocessing completed successfully.")
print(f"Preprocessed files saved to: {PROCESSED_DIR}")
print(f"Preprocessing summary saved to: {OUTPUT_DIR / 'preprocessing_summary.txt'}")