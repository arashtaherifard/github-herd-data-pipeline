# GitHub Herd Behavior and Repository Recommendation Dataset Collector

This folder contains a ready Python script for collecting a small Phase 1 sample dataset from the GitHub REST API.

## What it creates

After running the script, you will get:

- `repositories_sample.csv`
- `stars_sample.csv`
- `repo_star_timeseries_sample.csv`
- `user_repo_interactions_sample.csv`
- `sample_summary.csv`

These files are enough for the Phase 1 sample data link.

## Why this dataset fits the project

The dataset combines:

1. Time-series modeling
   Repository popularity can be tracked using daily/weekly star growth.

2. Recommendation systems
   Stargazers create user-repository interactions that can be used for recommendation.

3. Herd behavior
   We can test whether early popularity signals such as fast star growth or high cumulative stars predict later adoption.

## Setup

Install Python packages:

```bash
pip install requests pandas
```

Create a GitHub token and set it as an environment variable.

On macOS/Linux:

```bash
export GITHUB_TOKEN="YOUR_TOKEN_HERE"
```

On Windows PowerShell:

```powershell
setx GITHUB_TOKEN "YOUR_TOKEN_HERE"
```

Then restart the terminal.

## Run the sample collector

```bash
python collect_github_herd_sample.py
```

A slightly larger sample:

```bash
python collect_github_herd_sample.py --repos-per-topic 30 --max-repos-for-stars 20 --max-stargazers-per-repo 500
```

## Suggested sample upload

Upload the output folder `github_herd_sample_output` to Google Drive or GitHub and paste the folder link into the email as the sample data link.
