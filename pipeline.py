import subprocess
import sys

scripts = [
    "scripts/import_to_db.py",
    "scripts/run_sql_queries.py",
    "scripts/load_data.py",
    "scripts/eda.py",
    "scripts/preprocess.py",
    "scripts/feature_engineering.py"
]

for script in scripts:
    print("=" * 80)
    print(f"Running {script}")
    print("=" * 80)
    subprocess.run([sys.executable, script], check=True)

print("=" * 80)
print("Full data science pipeline completed successfully.")
print("=" * 80)