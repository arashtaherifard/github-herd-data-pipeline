# GitHub Herd Behavior and Repository Recommendation Dataset Pipeline

## Project Overview

This project is part of Phase 2 of the Introduction to Data Science final project. The dataset is the GitHub Herd Behavior and Repository Recommendation dataset. The goal is to prepare a real API-collected GitHub dataset for future modeling tasks such as herd-behavior classification, repository popularity analysis, time-series modeling, and recommendation-system modeling.

The project includes database implementation, SQL querying, exploratory data analysis, preprocessing, feature engineering, and an automated data science pipeline.

## Project Structure

```text
P2-S1/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── database/
│   └── github_herd.db
│
├── scripts/
│   ├── database_connection.py
│   ├── import_to_db.py
│   ├── run_sql_queries.py
│   ├── load_data.py
│   ├── eda.py
│   ├── preprocess.py
│   └── feature_engineering.py
│
├── outputs/
│   ├── figures/
│   ├── section2/
│   └── sql_query_outputs.txt
│
├── pipeline.py
├── requirements.txt
└── README.md
```

## Main Scripts

### `scripts/database_connection.py`

Creates a reusable connection to the local SQLite database.

### `scripts/import_to_db.py`

Imports the cleaned CSV files from `data/raw/` into the SQLite database.

### `scripts/run_sql_queries.py`

Runs SQL queries to validate the database and generate summary outputs.

### `scripts/load_data.py`

Loads the database tables into Pandas DataFrames and saves intermediate CSV files.

### `scripts/eda.py`

Performs exploratory data analysis and generates EDA summaries, tables, and figures.

### `scripts/preprocess.py`

Handles missing values, converts date columns, cleans invalid values, and creates preprocessed datasets.

### `scripts/feature_engineering.py`

Creates advanced repository-level, time-series, recommendation-system, and leakage-aware classification features.

### `pipeline.py`

Runs the full data science pipeline in sequence.

## How to Run the Project

First, create and activate a virtual environment.

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the complete pipeline:

```bash
python pipeline.py
```

The pipeline runs the following scripts in order:

```text
scripts/import_to_db.py
scripts/run_sql_queries.py
scripts/load_data.py
scripts/eda.py
scripts/preprocess.py
scripts/feature_engineering.py
```

## Generated Outputs

The pipeline generates:

```text
database/github_herd.db
outputs/sql_query_outputs.txt
outputs/section2/eda_summary.txt
outputs/section2/preprocessing_summary.txt
outputs/section2/feature_engineering_summary.txt
outputs/figures/
data/interim/
data/processed/
```

The final processed datasets include:

```text
repositories_preprocessed.csv
stars_preprocessed.csv
user_repo_interactions_preprocessed.csv
daily_timeseries_preprocessed.csv
weekly_timeseries_preprocessed.csv
herd_modeling_preprocessed.csv
repositories_feature_engineered.csv
weekly_timeseries_features.csv
recommendation_features.csv
herd_features_unscaled.csv
herd_model_ready.csv
```

## Final Model-Ready Dataset

The main classification-ready dataset is:

```text
data/processed/herd_model_ready.csv
```

This file is prepared for predicting the target variable:

```text
became_high_growth
```

Leakage-prone columns such as later popularity features and final GitHub popularity counts were removed from this file.
