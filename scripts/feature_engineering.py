from pathlib import Path
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "section2"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def read_processed_file(filename):
    return pd.read_csv(PROCESSED_DIR / filename)

def parse_topics(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if text == "" or text == "[]":
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).lower() for x in parsed]
    except Exception:
        pass
    return [x.strip().lower() for x in text.split(",") if x.strip()]

def has_any_topic(value, keywords):
    topics = parse_topics(value)
    joined = " ".join(topics)
    return int(any(keyword in joined for keyword in keywords))

def parse_week_start(value):
    if pd.isna(value):
        return pd.NaT
    text = str(value)
    if "/" in text:
        text = text.split("/")[0]
    return pd.to_datetime(text, errors="coerce")

repositories = read_processed_file("repositories_preprocessed.csv")
interactions = read_processed_file("user_repo_interactions_preprocessed.csv")
weekly = read_processed_file("weekly_timeseries_preprocessed.csv")
herd = read_processed_file("herd_modeling_preprocessed.csv")

summary_lines = []
summary_lines.append("Section 2 Feature Engineering Summary")
summary_lines.append("=" * 50)
summary_lines.append("")

if "topics" in repositories.columns:
    repositories["has_ai_topic"] = repositories["topics"].apply(lambda x: has_any_topic(x, ["ai", "artificial-intelligence"]))
    repositories["has_ml_topic"] = repositories["topics"].apply(lambda x: has_any_topic(x, ["machine-learning", "ml"]))
    repositories["has_deep_learning_topic"] = repositories["topics"].apply(lambda x: has_any_topic(x, ["deep-learning", "neural-network"]))
    repositories["has_data_science_topic"] = repositories["topics"].apply(lambda x: has_any_topic(x, ["data-science", "data-analysis"]))
    repositories["has_web_topic"] = repositories["topics"].apply(lambda x: has_any_topic(x, ["web", "react", "javascript", "frontend", "backend"]))
    repositories["has_llm_topic"] = repositories["topics"].apply(lambda x: has_any_topic(x, ["llm", "large-language-model", "generative-ai"]))

if "topics" in herd.columns:
    herd["has_ai_topic"] = herd["topics"].apply(lambda x: has_any_topic(x, ["ai", "artificial-intelligence"]))
    herd["has_ml_topic"] = herd["topics"].apply(lambda x: has_any_topic(x, ["machine-learning", "ml"]))
    herd["has_deep_learning_topic"] = herd["topics"].apply(lambda x: has_any_topic(x, ["deep-learning", "neural-network"]))
    herd["has_data_science_topic"] = herd["topics"].apply(lambda x: has_any_topic(x, ["data-science", "data-analysis"]))
    herd["has_web_topic"] = herd["topics"].apply(lambda x: has_any_topic(x, ["web", "react", "javascript", "frontend", "backend"]))
    herd["has_llm_topic"] = herd["topics"].apply(lambda x: has_any_topic(x, ["llm", "large-language-model", "generative-ai"]))

if "early_4week_stars" in herd.columns and "stars_count" in herd.columns:
    herd["early_to_total_star_ratio"] = herd["early_4week_stars"] / (herd["stars_count"] + 1)

if "early_max_weekly_stars" in herd.columns and "early_avg_weekly_stars" in herd.columns:
    herd["early_peak_to_average_ratio"] = herd["early_max_weekly_stars"] / (herd["early_avg_weekly_stars"] + 1)

if "forks_count" in herd.columns and "stars_count" in herd.columns:
    herd["forks_per_star"] = herd["forks_count"] / (herd["stars_count"] + 1)

if "open_issues_count" in herd.columns and "stars_count" in herd.columns:
    herd["issues_per_star"] = herd["open_issues_count"] / (herd["stars_count"] + 1)

if "week_start" in weekly.columns:
    weekly["week_start"] = pd.to_datetime(weekly["week_start"], errors="coerce")
else:
    weekly["week_start"] = weekly["week"].apply(parse_week_start)
weekly["weekly_new_stars"] = pd.to_numeric(weekly["weekly_new_stars"], errors="coerce").fillna(0)
weekly["cumulative_stars"] = pd.to_numeric(weekly["cumulative_stars"], errors="coerce").fillna(0)
weekly["weekly_growth_rate"] = pd.to_numeric(weekly["weekly_growth_rate"], errors="coerce").fillna(0)

repo_col = "repo_id"
weekly = weekly.dropna(subset=["week_start", repo_col])
weekly = weekly.sort_values([repo_col, "week_start"])

weekly["lag_1_week_stars"] = weekly.groupby(repo_col)["weekly_new_stars"].shift(1).fillna(0)
weekly["lag_2_week_stars"] = weekly.groupby(repo_col)["weekly_new_stars"].shift(2).fillna(0)
weekly["rolling_3week_mean_stars"] = weekly.groupby(repo_col)["weekly_new_stars"].transform(lambda x: x.rolling(3, min_periods=1).mean())
weekly["rolling_4week_sum_stars"] = weekly.groupby(repo_col)["weekly_new_stars"].transform(lambda x: x.rolling(4, min_periods=1).sum())
weekly["weekly_growth_acceleration"] = weekly.groupby(repo_col)["weekly_new_stars"].diff().fillna(0)
weekly["cumulative_growth_lag_1"] = weekly.groupby(repo_col)["cumulative_stars"].shift(1).fillna(0)

user_activity = interactions.groupby("user_id").size().reset_index(name="user_activity_count")
repo_popularity = interactions.groupby("repo_id").size().reset_index(name="repo_popularity_count")

recommendation_features = interactions.merge(user_activity, on="user_id", how="left")
recommendation_features = recommendation_features.merge(repo_popularity, on="repo_id", how="left")

repo_side_features = repositories[["repo_id", "language", "topic_count", "stars_count", "forks_count"]].copy()
recommendation_features = recommendation_features.merge(repo_side_features, on="repo_id", how="left")

recommendation_features["user_activity_count"] = recommendation_features["user_activity_count"].fillna(0)
recommendation_features["repo_popularity_count"] = recommendation_features["repo_popularity_count"].fillna(0)
recommendation_features["language"] = recommendation_features["language"].fillna("Unknown")
recommendation_features["topic_count"] = recommendation_features["topic_count"].fillna(0)
recommendation_features["stars_count"] = recommendation_features["stars_count"].fillna(0)
recommendation_features["forks_count"] = recommendation_features["forks_count"].fillna(0)

herd_model_ready = herd.copy()

if "language" in herd_model_ready.columns:
    top_languages = herd_model_ready["language"].value_counts().head(8).index
    herd_model_ready["language_grouped"] = herd_model_ready["language"].where(herd_model_ready["language"].isin(top_languages), "Other")
else:
    herd_model_ready["language_grouped"] = "Unknown"

if "license" in herd_model_ready.columns:
    top_licenses = herd_model_ready["license"].value_counts().head(6).index
    herd_model_ready["license_grouped"] = herd_model_ready["license"].where(herd_model_ready["license"].isin(top_licenses), "Other")
else:
    herd_model_ready["license_grouped"] = "No License"

categorical_cols = ["language_grouped", "license_grouped"]
herd_model_ready = pd.get_dummies(herd_model_ready, columns=categorical_cols, drop_first=False)

leakage_cols = [
    "later_stars",
    "later_avg_weekly_stars",
    "later_max_weekly_stars",
    "stars_count",
    "forks_count",
    "watchers_count",
    "open_issues_count",
    "early_to_total_star_ratio",
    "forks_per_star",
    "issues_per_star"
]

for col in leakage_cols:
    if col in herd_model_ready.columns:
        herd_model_ready = herd_model_ready.drop(columns=[col])

drop_non_model_cols = [
    "full_name",
    "repo_full_name",
    "owner",
    "repo_name",
    "description",
    "topics",
    "html_url",
    "homepage",
    "created_at",
    "updated_at",
    "pushed_at",
    "collected_at",
    "language",
    "license"
]

for col in drop_non_model_cols:
    if col in herd_model_ready.columns:
        herd_model_ready = herd_model_ready.drop(columns=[col])

target_col = "became_high_growth"
id_cols = ["repo_id"]

numeric_cols = []
for col in herd_model_ready.columns:
    if col not in id_cols + [target_col]:
        if pd.api.types.is_numeric_dtype(herd_model_ready[col]):
            numeric_cols.append(col)

for col in numeric_cols:
    herd_model_ready[col] = pd.to_numeric(herd_model_ready[col], errors="coerce").fillna(0)

scaled_features = herd_model_ready.copy()
scaler = StandardScaler()

if len(numeric_cols) > 0:
    scaled_features[numeric_cols] = scaler.fit_transform(scaled_features[numeric_cols])

repositories.to_csv(PROCESSED_DIR / "repositories_feature_engineered.csv", index=False)
weekly.to_csv(PROCESSED_DIR / "weekly_timeseries_features.csv", index=False)
recommendation_features.to_csv(PROCESSED_DIR / "recommendation_features.csv", index=False)
herd.to_csv(PROCESSED_DIR / "herd_features_unscaled.csv", index=False)
scaled_features.to_csv(PROCESSED_DIR / "herd_model_ready.csv", index=False)

feature_files = {
    "repositories_feature_engineered.csv": repositories,
    "weekly_timeseries_features.csv": weekly,
    "recommendation_features.csv": recommendation_features,
    "herd_features_unscaled.csv": herd,
    "herd_model_ready.csv": scaled_features
}

summary_lines.append("Generated Feature-Engineered Files")
summary_lines.append("-" * 50)
for name, df in feature_files.items():
    summary_lines.append(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")
summary_lines.append("")

summary_lines.append("Repository-Level Features")
summary_lines.append("-" * 50)
summary_lines.append("Created topic-based binary features such as has_ai_topic, has_ml_topic, has_web_topic, and has_llm_topic.")
summary_lines.append("Kept repository-level engineered features such as repo_age_days, stars_per_day, forks_per_star, issues_per_star, topic_count, and description_length.")
summary_lines.append("")

summary_lines.append("Herd-Behavior Features")
summary_lines.append("-" * 50)
summary_lines.append("Created early-period growth features such as early_peak_to_average_ratio. Potential leakage-prone popularity-ratio features were excluded from the model-ready classification dataset.")
summary_lines.append("Removed future-information leakage columns from the model-ready classification dataset.")
summary_lines.append("Encoded language and license categories using one-hot encoding.")
summary_lines.append("Standardized numeric predictors using StandardScaler.")
summary_lines.append("")

summary_lines.append("Time-Series Features")
summary_lines.append("-" * 50)
summary_lines.append("Created lag_1_week_stars, lag_2_week_stars, rolling_3week_mean_stars, rolling_4week_sum_stars, weekly_growth_acceleration, and cumulative_growth_lag_1.")
summary_lines.append("")

summary_lines.append("Recommendation-System Features")
summary_lines.append("-" * 50)
summary_lines.append("Created user_activity_count and repo_popularity_count from implicit star interactions.")
summary_lines.append("Merged repository metadata features into the interaction table.")
summary_lines.append("")

summary_lines.append("Model-Ready Classification Dataset")
summary_lines.append("-" * 50)
summary_lines.append(f"Target column: {target_col}")
summary_lines.append(f"Number of predictors after preprocessing and encoding: {len([c for c in scaled_features.columns if c not in id_cols + [target_col]])}")
summary_lines.append(f"Class distribution: {scaled_features[target_col].value_counts().sort_index().to_dict()}")
summary_lines.append("")

with open(OUTPUT_DIR / "feature_engineering_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("Feature engineering completed successfully.")
print(f"Feature-engineered files saved to: {PROCESSED_DIR}")
print(f"Feature engineering summary saved to: {OUTPUT_DIR / 'feature_engineering_summary.txt'}")