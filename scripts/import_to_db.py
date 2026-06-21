from pathlib import Path
import pandas as pd
from database_connection import get_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

TABLES_MAPPING = {
    "repositories_enriched.csv": "repositories",
    "stars_raw.csv": "stars",
    "user_repo_interactions.csv": "user_repo_interactions",
    "repo_star_daily_timeseries.csv": "daily_timeseries",
    "repo_star_weekly_timeseries.csv": "weekly_timeseries",
    "repo_herd_modeling_table.csv": "herd_modeling"
}

def main():
    engine = get_engine()

    for file_name, table_name in TABLES_MAPPING.items():
        file_path = DATA_DIR / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        df = pd.read_csv(file_path)
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"Imported {file_name} into table '{table_name}' with {len(df)} rows.")

    print("Database import completed successfully.")

if __name__ == "__main__":
    main()
