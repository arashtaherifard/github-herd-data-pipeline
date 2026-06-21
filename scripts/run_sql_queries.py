from pathlib import Path
import pandas as pd
from database_connection import get_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "sql_query_outputs.txt"

QUERIES = {
    "Total Repositories": """
        SELECT COUNT(*) AS num_repositories
        FROM repositories;
    """,

    "Total Star Interactions": """
        SELECT COUNT(*) AS num_star_interactions
        FROM stars;
    """,

    "Total Unique Users": """
        SELECT COUNT(DISTINCT user_id) AS num_unique_users
        FROM stars;
    """,

    "Top 10 Programming Languages": """
        SELECT COALESCE(language, 'Unknown') AS language, COUNT(*) AS repo_count
        FROM repositories
        GROUP BY COALESCE(language, 'Unknown')
        ORDER BY repo_count DESC
        LIMIT 10;
    """,

    "Top 10 Repositories by GitHub Star Count": """
        SELECT full_name, language, stars_count, forks_count
        FROM repositories
        ORDER BY stars_count DESC
        LIMIT 10;
    """,

    "Top 10 Repositories by Collected Stargazer Interactions": """
        SELECT repo_full_name, COUNT(*) AS collected_star_events
        FROM stars
        GROUP BY repo_id, repo_full_name
        ORDER BY collected_star_events DESC
        LIMIT 10;
    """,

    "Herd Modeling Class Balance": """
        SELECT became_high_growth, COUNT(*) AS repo_count
        FROM herd_modeling
        GROUP BY became_high_growth;
    """,

    "Average Early and Later Stars by Herd Class": """
        SELECT became_high_growth,
               AVG(early_4week_stars) AS avg_early_4week_stars,
               AVG(later_stars) AS avg_later_stars
        FROM herd_modeling
        GROUP BY became_high_growth;
    """,

    "Weekly Time-Series Example": """
        SELECT repo_full_name, week, weekly_new_stars, cumulative_stars, herd_momentum_score
        FROM weekly_timeseries
        ORDER BY cumulative_stars DESC
        LIMIT 20;
    """
}

def main():
    engine = get_engine()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for title, query in QUERIES.items():
            result_df = pd.read_sql_query(query, engine)

            f.write("=" * 80 + "\n")
            f.write(title + "\n")
            f.write("=" * 80 + "\n")
            f.write(result_df.to_string(index=False))
            f.write("\n\n")

            print("\n" + "=" * 80)
            print(title)
            print("=" * 80)
            print(result_df)

    print(f"\nSQL query outputs saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
