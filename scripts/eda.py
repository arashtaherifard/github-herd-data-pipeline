from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from database_connection import get_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "section2"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

engine = get_engine()

def parse_week_start(value):
    if pd.isna(value):
        return pd.NaT
    text = str(value)
    if "/" in text:
        text = text.split("/")[0]
    return pd.to_datetime(text, errors="coerce")

tables = {
    "repositories": "SELECT * FROM repositories",
    "stars": "SELECT * FROM stars",
    "user_repo_interactions": "SELECT * FROM user_repo_interactions",
    "daily_timeseries": "SELECT * FROM daily_timeseries",
    "weekly_timeseries": "SELECT * FROM weekly_timeseries",
    "herd_modeling": "SELECT * FROM herd_modeling"
}

data = {}

with engine.connect() as conn:
    for name, query in tables.items():
        data[name] = pd.read_sql(query, conn)

repositories = data["repositories"]
stars = data["stars"]
interactions = data["user_repo_interactions"]
daily = data["daily_timeseries"]
weekly = data["weekly_timeseries"]
herd = data["herd_modeling"]

summary_lines = []

summary_lines.append("Section 2 EDA Summary")
summary_lines.append("=" * 50)
summary_lines.append("")

summary_lines.append("Dataset Shape Summary")
summary_lines.append("-" * 50)
for name, df in data.items():
    summary_lines.append(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")
summary_lines.append("")

summary_lines.append("Main Dataset Counts")
summary_lines.append("-" * 50)
summary_lines.append(f"Number of repositories: {repositories['repo_id'].nunique()}")
summary_lines.append(f"Number of star interactions: {len(stars)}")
summary_lines.append(f"Number of unique users: {stars['user_id'].nunique()}")
summary_lines.append(f"Number of recommendation interactions: {len(interactions)}")
summary_lines.append(f"Number of daily time-series rows: {len(daily)}")
summary_lines.append(f"Number of weekly time-series rows: {len(weekly)}")
summary_lines.append(f"Number of herd-modeling rows: {len(herd)}")
summary_lines.append("")

missing_frames = []
for name, df in data.items():
    missing = df.isna().sum().reset_index()
    missing.columns = ["column", "missing_count"]
    missing["table"] = name
    missing["missing_percent"] = (missing["missing_count"] / len(df) * 100).round(2)
    missing_frames.append(missing)

missing_summary = pd.concat(missing_frames, ignore_index=True)
missing_summary = missing_summary[["table", "column", "missing_count", "missing_percent"]]
missing_summary.to_csv(OUTPUT_DIR / "missing_values_summary.csv", index=False)

summary_lines.append("Missing Value Summary")
summary_lines.append("-" * 50)
important_missing = missing_summary[missing_summary["missing_count"] > 0]
if len(important_missing) == 0:
    summary_lines.append("No missing values found.")
else:
    for _, row in important_missing.iterrows():
        summary_lines.append(f"{row['table']}.{row['column']}: {row['missing_count']} missing values ({row['missing_percent']}%)")
summary_lines.append("")

if "language" in repositories.columns:
    language_counts = repositories["language"].fillna("Unknown").value_counts().head(10)
    language_counts.to_csv(OUTPUT_DIR / "top_languages.csv", header=["repo_count"])

    summary_lines.append("Top 10 Programming Languages")
    summary_lines.append("-" * 50)
    for language, count in language_counts.items():
        summary_lines.append(f"{language}: {count}")
    summary_lines.append("")

    plt.figure(figsize=(10, 6))
    language_counts.sort_values().plot(kind="barh")
    plt.title("Top 10 Programming Languages")
    plt.xlabel("Number of Repositories")
    plt.ylabel("Language")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "top_languages.png", dpi=300)
    plt.close()

if "stars_count" in repositories.columns:
    stars_count = pd.to_numeric(repositories["stars_count"], errors="coerce").dropna()
    summary_lines.append("Repository Star Count Distribution")
    summary_lines.append("-" * 50)
    summary_lines.append(f"Minimum stars_count: {stars_count.min()}")
    summary_lines.append(f"Maximum stars_count: {stars_count.max()}")
    summary_lines.append(f"Mean stars_count: {stars_count.mean():.2f}")
    summary_lines.append(f"Median stars_count: {stars_count.median():.2f}")
    summary_lines.append("")

    plt.figure(figsize=(10, 6))
    plt.hist(stars_count, bins=50)
    plt.title("Distribution of Repository GitHub Star Counts")
    plt.xlabel("GitHub Star Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "stars_count_distribution.png", dpi=300)
    plt.close()

if "forks_count" in repositories.columns:
    forks_count = pd.to_numeric(repositories["forks_count"], errors="coerce").dropna()
    summary_lines.append("Repository Fork Count Distribution")
    summary_lines.append("-" * 50)
    summary_lines.append(f"Minimum forks_count: {forks_count.min()}")
    summary_lines.append(f"Maximum forks_count: {forks_count.max()}")
    summary_lines.append(f"Mean forks_count: {forks_count.mean():.2f}")
    summary_lines.append(f"Median forks_count: {forks_count.median():.2f}")
    summary_lines.append("")

    plt.figure(figsize=(10, 6))
    plt.hist(forks_count, bins=50)
    plt.title("Distribution of Repository Fork Counts")
    plt.xlabel("Fork Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "forks_count_distribution.png", dpi=300)
    plt.close()

if "became_high_growth" in herd.columns:
    target_counts = herd["became_high_growth"].value_counts().sort_index()
    target_counts.to_csv(OUTPUT_DIR / "target_class_balance.csv", header=["repo_count"])

    summary_lines.append("Class Balance of became_high_growth")
    summary_lines.append("-" * 50)
    for label, count in target_counts.items():
        summary_lines.append(f"Class {label}: {count}")
    summary_lines.append("")

    plt.figure(figsize=(7, 5))
    target_counts.plot(kind="bar")
    plt.title("Class Balance of Herd-Behavior Target")
    plt.xlabel("became_high_growth")
    plt.ylabel("Number of Repositories")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "target_class_balance.png", dpi=300)
    plt.close()

if "repo_full_name" in stars.columns:
    top_collected = stars.groupby("repo_full_name").size().sort_values(ascending=False).head(10)
    top_collected.to_csv(OUTPUT_DIR / "top_repositories_by_collected_interactions.csv", header=["collected_star_interactions"])

    summary_lines.append("Top 10 Repositories by Collected Star Interactions")
    summary_lines.append("-" * 50)
    for repo, count in top_collected.items():
        summary_lines.append(f"{repo}: {count}")
    summary_lines.append("")

    plt.figure(figsize=(10, 6))
    top_collected.sort_values().plot(kind="barh")
    plt.title("Top 10 Repositories by Collected Star Interactions")
    plt.xlabel("Collected Star Interactions")
    plt.ylabel("Repository")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "top_repositories_by_collected_interactions.png", dpi=300)
    plt.close()

if "full_name" in repositories.columns and "stars_count" in repositories.columns:
    top_github = repositories[["full_name", "stars_count"]].copy()
    top_github["stars_count"] = pd.to_numeric(top_github["stars_count"], errors="coerce")
    top_github = top_github.dropna().sort_values("stars_count", ascending=False).head(10)
    top_github.to_csv(OUTPUT_DIR / "top_repositories_by_github_stars.csv", index=False)

    summary_lines.append("Top 10 Repositories by GitHub Star Count")
    summary_lines.append("-" * 50)
    for _, row in top_github.iterrows():
        summary_lines.append(f"{row['full_name']}: {int(row['stars_count'])}")
    summary_lines.append("")

    plt.figure(figsize=(10, 6))
    plt.barh(top_github["full_name"][::-1], top_github["stars_count"][::-1])
    plt.title("Top 10 Repositories by GitHub Star Count")
    plt.xlabel("GitHub Star Count")
    plt.ylabel("Repository")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "top_repositories_by_github_stars.png", dpi=300)
    plt.close()

if "week" in weekly.columns and "weekly_new_stars" in weekly.columns:
    weekly_plot = weekly.copy()
    weekly_plot["week_start"] = weekly_plot["week"].apply(parse_week_start)
    weekly_plot["weekly_new_stars"] = pd.to_numeric(weekly_plot["weekly_new_stars"], errors="coerce")
    weekly_plot = weekly_plot.dropna(subset=["week_start", "weekly_new_stars"])

    if "repo_full_name" in weekly_plot.columns:
        repo_label_col = "repo_full_name"
    else:
        repo_label_col = "repo_id"

    top_weekly_repos = weekly_plot.groupby(repo_label_col)["weekly_new_stars"].sum().sort_values(ascending=False).head(5).index

    plt.figure(figsize=(12, 7))
    for repo in top_weekly_repos:
        temp = weekly_plot[weekly_plot[repo_label_col] == repo].sort_values("week_start")
        if len(temp) > 0:
            plt.plot(temp["week_start"], temp["weekly_new_stars"], label=str(repo))

    plt.title("Weekly Star Growth for Top 5 Repositories")
    plt.xlabel("Week")
    plt.ylabel("Weekly New Stars")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "weekly_growth_top_repositories.png", dpi=300)
    plt.close()

if "early_4week_stars" in herd.columns and "later_stars" in herd.columns and "became_high_growth" in herd.columns:
    group_summary = herd.groupby("became_high_growth")[["early_4week_stars", "later_stars"]].mean().round(2)
    group_summary.to_csv(OUTPUT_DIR / "early_later_stars_by_class.csv")

    summary_lines.append("Average Early and Later Stars by Herd Class")
    summary_lines.append("-" * 50)
    summary_lines.append(group_summary.to_string())
    summary_lines.append("")

    plt.figure(figsize=(8, 6))
    group_summary.plot(kind="bar")
    plt.title("Average Early and Later Stars by Herd Class")
    plt.xlabel("became_high_growth")
    plt.ylabel("Average Star Count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "early_later_stars_by_class.png", dpi=300)
    plt.close()

numeric_summary = {}
for name, df in data.items():
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] > 0:
        numeric_summary[name] = numeric_df.describe().T
        numeric_summary[name].to_csv(OUTPUT_DIR / f"{name}_numeric_summary.csv")

with open(OUTPUT_DIR / "eda_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("EDA completed successfully.")
print(f"EDA summary saved to: {OUTPUT_DIR / 'eda_summary.txt'}")
print(f"EDA tables saved to: {OUTPUT_DIR}")
print(f"EDA figures saved to: {FIGURE_DIR}")