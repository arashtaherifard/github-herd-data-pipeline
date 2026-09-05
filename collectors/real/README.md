# GitHub Herd Behavior Real Dataset Collector

This is the larger version of the collector for the final dataset.

It collects real GitHub data using the GitHub REST API and creates files for:

1. Repository metadata
2. User-repository star interactions
3. Daily and weekly popularity time series
4. Recommendation-system interaction table
5. Herd-behavior modeling table

## Files created

The script creates the folder:

```bash
github_herd_real_dataset_output
```

Inside it, you will get:

```text
repositories_raw.csv
repositories_enriched.csv
stars_raw.csv
user_repo_interactions.csv
repo_star_daily_timeseries.csv
repo_star_weekly_timeseries.csv
repo_herd_modeling_table.csv
dataset_summary.csv
dataset_summary.json
DATA_DICTIONARY.md
raw/
```

## Setup on Mac

From this folder:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## GitHub token

For the real dataset, use a GitHub personal access token.

In the current terminal session:

```bash
export GITHUB_TOKEN="YOUR_REAL_TOKEN_HERE"
```

Do not share your token with anyone.

## Recommended first real run

This is a good real dataset size without being too extreme:

```bash
python collect_github_herd_real_dataset.py \
  --repos-per-topic 100 \
  --max-repos-for-stars 100 \
  --max-stargazers-per-repo 1000
```

Expected approximate scale:

- up to about 1000 repositories before duplicates
- around 100 repositories with stargazer histories
- up to about 100,000 user-repository star interactions

## Larger run

Only do this after the recommended run works:

```bash
python collect_github_herd_real_dataset.py \
  --repos-per-topic 150 \
  --max-repos-for-stars 200 \
  --max-stargazers-per-repo 1500
```

## Resume behavior

The script saves files while running. If it stops because of internet or rate limits, run the same command again and it will resume.

To force a fresh run:

```bash
python collect_github_herd_real_dataset.py --no-resume
```

## How this supports the project

### Herd behavior

Use:

```text
repo_star_weekly_timeseries.csv
repo_herd_modeling_table.csv
```

These contain variables such as early stars, later stars, previous-week stars, growth rate, and herd momentum.

### Recommendation system

Use:

```text
user_repo_interactions.csv
repositories_enriched.csv
```

These support popularity-based, content-based, collaborative-filtering, and hybrid recommendation models.

### Time series

Use:

```text
repo_star_daily_timeseries.csv
repo_star_weekly_timeseries.csv
```

These support popularity growth analysis and future-star prediction.
