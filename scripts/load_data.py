from pathlib import Path
import pandas as pd
from database_connection import get_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "section2"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

engine = get_engine()

tables = [
    "repositories",
    "stars",
    "user_repo_interactions",
    "daily_timeseries",
    "weekly_timeseries",
    "herd_modeling"
]

summary_lines = []

with engine.connect() as conn:
    for table in tables:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        df.to_csv(INTERIM_DIR / f"{table}.csv", index=False)
        summary_lines.append(f"{table}: {df.shape[0]} rows, {df.shape[1]} columns")

with open(OUTPUT_DIR / "load_data_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("Data loaded from database successfully.")
print(f"Loaded data saved to: {INTERIM_DIR}")
print(f"Summary saved to: {OUTPUT_DIR / 'load_data_summary.txt'}")