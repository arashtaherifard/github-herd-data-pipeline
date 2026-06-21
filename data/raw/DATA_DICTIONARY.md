# Data Dictionary

## repositories_enriched.csv
Repository-level metadata collected from GitHub.

Key columns:
- repo_id: GitHub repository ID
- full_name: owner/repository name
- language: primary programming language
- topics: GitHub topics separated by |
- stars_count: total stars at collection time
- forks_count: number of forks
- created_at / updated_at / pushed_at: repository timestamps
- license: repository license identifier when available

## stars_raw.csv
Raw user-repository star interactions.

Key columns:
- repo_id: GitHub repository ID
- repo_full_name: owner/repository name
- user_id: GitHub user ID
- username: GitHub username
- starred_at: timestamp when the user starred the repository
- collected_at: timestamp when the row was collected

## user_repo_interactions.csv
Recommendation-system interaction table.

Key columns:
- user_id
- repo_id
- interaction: always 1, meaning the user starred the repository

## repo_star_daily_timeseries.csv
Daily repository popularity time series.

Key columns:
- daily_new_stars: new stars on that date
- cumulative_stars: cumulative stars observed in the collected stargazer sample
- daily_growth_rate: relative change in daily stars

## repo_star_weekly_timeseries.csv
Weekly repository popularity time series.

Key columns:
- weekly_new_stars: new stars in that week
- previous_week_stars: stars in previous week
- weekly_growth_rate: relative change in weekly stars
- herd_momentum_score: previous_week_stars * cumulative_stars

## repo_herd_modeling_table.csv
Repository-level modeling table for herd behavior prediction.

Key columns:
- early_4week_stars: stars in the first four observed weeks
- later_stars: stars after the first four observed weeks
- became_high_growth: binary target for popularity prediction
