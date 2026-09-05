import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd


API_BASE = "https://api.github.com"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def make_session():
    token = os.getenv("GITHUB_TOKEN")
    session = requests.Session()
    session.headers.update({
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "github-herd-behavior-final-dataset"
    })
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})
    return session


def safe_name(text):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path, default):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def request_json(session, url, params=None, accept=None, max_retries=5):
    headers = {}
    if accept:
        headers["Accept"] = accept

    for attempt in range(max_retries):
        response = session.get(url, params=params, headers=headers, timeout=40)

        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")

        if response.status_code == 403 and remaining == "0":
            reset_time = int(reset or time.time() + 60)
            sleep_seconds = max(reset_time - int(time.time()) + 5, 5)
            print(f"Primary rate limit reached. Sleeping {sleep_seconds} seconds...")
            time.sleep(sleep_seconds)
            continue

        if response.status_code in [403, 429] and "secondary rate limit" in response.text.lower():
            sleep_seconds = 60 + attempt * 30
            print(f"Secondary rate limit. Sleeping {sleep_seconds} seconds...")
            time.sleep(sleep_seconds)
            continue

        if response.status_code in [500, 502, 503, 504]:
            sleep_seconds = 10 * (attempt + 1)
            print(f"Temporary GitHub error {response.status_code}. Sleeping {sleep_seconds} seconds...")
            time.sleep(sleep_seconds)
            continue

        if response.status_code == 422:
            print(f"Skipping invalid/unavailable query: {url} params={params}")
            return None

        response.raise_for_status()
        return response.json()

    response.raise_for_status()


def search_repositories(session, topics, repos_per_topic, min_stars, outdir, resume=True):
    rows = []
    seen_ids = set()
    cache_dir = outdir / "raw" / "repository_search"

    if resume and (outdir / "repositories_raw.csv").exists():
        old = pd.read_csv(outdir / "repositories_raw.csv")
        rows = old.to_dict("records")
        seen_ids = set(old["repo_id"].dropna().astype(int).tolist())
        print(f"Resume: loaded {len(rows)} existing repositories.")

    for topic in topics:
        topic_done_path = cache_dir / f"{safe_name(topic)}.done.json"
        if resume and topic_done_path.exists():
            print(f"Skipping completed topic: {topic}")
            continue

        print(f"Searching topic: {topic}")
        pages_needed = (repos_per_topic + 99) // 100
        collected_for_topic = 0

        for page in range(1, pages_needed + 1):
            per_page = min(100, repos_per_topic - collected_for_topic)
            if per_page <= 0:
                break

            url = f"{API_BASE}/search/repositories"
            params = {
                "q": f"topic:{topic} stars:>{min_stars}",
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page
            }

            data = request_json(session, url, params=params)
            if not data:
                break

            save_json(cache_dir / f"{safe_name(topic)}_page_{page}.json", data)

            items = data.get("items", [])
            if not items:
                break

            for repo in items:
                repo_id = repo.get("id")
                if repo_id in seen_ids:
                    continue
                seen_ids.add(repo_id)
                rows.append({
                    "repo_id": repo_id,
                    "full_name": repo.get("full_name"),
                    "owner": repo.get("owner", {}).get("login"),
                    "owner_id": repo.get("owner", {}).get("id"),
                    "repo_name": repo.get("name"),
                    "description": repo.get("description"),
                    "language": repo.get("language"),
                    "topics": "|".join(repo.get("topics", [])),
                    "stars_count": repo.get("stargazers_count"),
                    "forks_count": repo.get("forks_count"),
                    "watchers_count": repo.get("watchers_count"),
                    "open_issues_count": repo.get("open_issues_count"),
                    "size": repo.get("size"),
                    "created_at": repo.get("created_at"),
                    "updated_at": repo.get("updated_at"),
                    "pushed_at": repo.get("pushed_at"),
                    "default_branch": repo.get("default_branch"),
                    "archived": repo.get("archived"),
                    "disabled": repo.get("disabled"),
                    "is_fork": repo.get("fork"),
                    "license": (repo.get("license") or {}).get("spdx_id"),
                    "html_url": repo.get("html_url"),
                    "source_topic": topic,
                    "collected_at": utc_now()
                })

            collected_for_topic += len(items)
            pd.DataFrame(rows).to_csv(outdir / "repositories_raw.csv", index=False)
            time.sleep(1.2)

        save_json(topic_done_path, {"topic": topic, "completed_at": utc_now()})

    repos_df = pd.DataFrame(rows)
    repos_df = repos_df.drop_duplicates(subset=["repo_id"])
    repos_df.to_csv(outdir / "repositories_raw.csv", index=False)
    return repos_df


def enrich_repository_details(session, repos_df, outdir, resume=True):
    details_path = outdir / "repositories_enriched.csv"
    raw_dir = outdir / "raw" / "repository_details"

    done = set()
    rows = []

    if resume and details_path.exists():
        old = pd.read_csv(details_path)
        rows = old.to_dict("records")
        done = set(old["repo_id"].dropna().astype(int).tolist())
        print(f"Resume: loaded {len(done)} enriched repositories.")

    for _, repo in repos_df.iterrows():
        repo_id = int(repo["repo_id"])
        full_name = repo["full_name"]

        if repo_id in done:
            continue

        print(f"Enriching repository: {full_name}")
        owner, repo_name = full_name.split("/", 1)
        url = f"{API_BASE}/repos/{owner}/{repo_name}"
        data = request_json(session, url)

        if not data:
            continue

        save_json(raw_dir / f"{safe_name(full_name)}.json", data)

        rows.append({
            "repo_id": data.get("id"),
            "full_name": data.get("full_name"),
            "owner": data.get("owner", {}).get("login"),
            "owner_id": data.get("owner", {}).get("id"),
            "repo_name": data.get("name"),
            "description": data.get("description"),
            "language": data.get("language"),
            "topics": "|".join(data.get("topics", [])),
            "stars_count": data.get("stargazers_count"),
            "forks_count": data.get("forks_count"),
            "watchers_count": data.get("watchers_count"),
            "subscribers_count": data.get("subscribers_count"),
            "network_count": data.get("network_count"),
            "open_issues_count": data.get("open_issues_count"),
            "size": data.get("size"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "pushed_at": data.get("pushed_at"),
            "default_branch": data.get("default_branch"),
            "archived": data.get("archived"),
            "disabled": data.get("disabled"),
            "is_fork": data.get("fork"),
            "has_issues": data.get("has_issues"),
            "has_projects": data.get("has_projects"),
            "has_downloads": data.get("has_downloads"),
            "has_wiki": data.get("has_wiki"),
            "license": (data.get("license") or {}).get("spdx_id"),
            "html_url": data.get("html_url"),
            "homepage": data.get("homepage"),
            "collected_at": utc_now()
        })

        pd.DataFrame(rows).to_csv(details_path, index=False)
        time.sleep(0.8)

    details_df = pd.DataFrame(rows).drop_duplicates(subset=["repo_id"])
    details_df.to_csv(details_path, index=False)
    return details_df


def collect_stargazers(session, repos_df, outdir, max_repos_for_stars, max_stargazers_per_repo, resume=True):
    stars_path = outdir / "stars_raw.csv"
    raw_dir = outdir / "raw" / "stargazers"

    rows = []
    completed = load_json(outdir / "stargazers_completed.json", {})

    if resume and stars_path.exists():
        old = pd.read_csv(stars_path)
        rows = old.to_dict("records")
        print(f"Resume: loaded {len(rows)} existing star interactions.")

    selected = repos_df.sort_values("stars_count", ascending=False).head(max_repos_for_stars)

    for _, repo in selected.iterrows():
        full_name = repo["full_name"]
        repo_id = int(repo["repo_id"])

        if resume and completed.get(str(repo_id), 0) >= max_stargazers_per_repo:
            print(f"Skipping completed stargazers: {full_name}")
            continue

        existing_count = int(completed.get(str(repo_id), 0)) if resume else 0
        start_page = existing_count // 100 + 1
        collected = existing_count

        print(f"Collecting stargazers for {full_name}, already={existing_count}")

        owner, repo_name = full_name.split("/", 1)

        while collected < max_stargazers_per_repo:
            per_page = min(100, max_stargazers_per_repo - collected)
            page = collected // 100 + 1

            url = f"{API_BASE}/repos/{owner}/{repo_name}/stargazers"
            params = {"per_page": per_page, "page": page}

            data = request_json(
                session,
                url,
                params=params,
                accept="application/vnd.github.star+json"
            )

            if not data:
                break

            save_json(raw_dir / safe_name(full_name) / f"page_{page}.json", data)

            new_rows = []
            for item in data:
                user = item.get("user", item)
                new_rows.append({
                    "repo_id": repo_id,
                    "repo_full_name": full_name,
                    "user_id": user.get("id"),
                    "username": user.get("login"),
                    "user_type": user.get("type"),
                    "starred_at": item.get("starred_at"),
                    "collected_at": utc_now()
                })

            rows.extend(new_rows)
            collected += len(new_rows)
            completed[str(repo_id)] = collected

            pd.DataFrame(rows).drop_duplicates(subset=["repo_id", "user_id"]).to_csv(stars_path, index=False)
            save_json(outdir / "stargazers_completed.json", completed)

            if len(data) < per_page:
                break

            time.sleep(1.5)

    stars_df = pd.DataFrame(rows)
    if not stars_df.empty:
        stars_df = stars_df.drop_duplicates(subset=["repo_id", "user_id"])
    stars_df.to_csv(stars_path, index=False)
    return stars_df


def build_timeseries(stars_df, outdir):
    valid = stars_df.dropna(subset=["starred_at"]).copy()
    if valid.empty:
        daily = pd.DataFrame()
        weekly = pd.DataFrame()
        daily.to_csv(outdir / "repo_star_daily_timeseries.csv", index=False)
        weekly.to_csv(outdir / "repo_star_weekly_timeseries.csv", index=False)
        return daily, weekly

    valid["starred_at_dt"] = pd.to_datetime(valid["starred_at"], utc=True)
    valid["date"] = valid["starred_at_dt"].dt.date
    valid["week"] = valid["starred_at_dt"].dt.to_period("W").astype(str)

    daily = (
        valid.groupby(["repo_id", "repo_full_name", "date"])
        .size()
        .reset_index(name="daily_new_stars")
        .sort_values(["repo_id", "date"])
    )
    daily["cumulative_stars"] = daily.groupby("repo_id")["daily_new_stars"].cumsum()
    daily["daily_growth_rate"] = daily.groupby("repo_id")["daily_new_stars"].pct_change().replace([float("inf"), -float("inf")], pd.NA)

    weekly = (
        valid.groupby(["repo_id", "repo_full_name", "week"])
        .size()
        .reset_index(name="weekly_new_stars")
        .sort_values(["repo_id", "week"])
    )
    weekly["cumulative_stars"] = weekly.groupby("repo_id")["weekly_new_stars"].cumsum()
    weekly["previous_week_stars"] = weekly.groupby("repo_id")["weekly_new_stars"].shift(1)
    weekly["weekly_growth_rate"] = weekly.groupby("repo_id")["weekly_new_stars"].pct_change().replace([float("inf"), -float("inf")], pd.NA)
    weekly["herd_momentum_score"] = weekly["previous_week_stars"].fillna(0) * weekly["cumulative_stars"].fillna(0)

    daily.to_csv(outdir / "repo_star_daily_timeseries.csv", index=False)
    weekly.to_csv(outdir / "repo_star_weekly_timeseries.csv", index=False)

    return daily, weekly


def build_interactions(stars_df, outdir):
    interactions = stars_df[["user_id", "username", "repo_id", "repo_full_name", "starred_at"]].copy()
    interactions["interaction"] = 1
    interactions = interactions.drop_duplicates(subset=["user_id", "repo_id"])
    interactions.to_csv(outdir / "user_repo_interactions.csv", index=False)
    return interactions


def build_modeling_table(repos_df, weekly_df, outdir):
    if weekly_df.empty:
        modeling = pd.DataFrame()
        modeling.to_csv(outdir / "repo_herd_modeling_table.csv", index=False)
        return modeling

    temp = weekly_df.copy()
    temp["week_index"] = temp.groupby("repo_id").cumcount() + 1

    early = (
        temp[temp["week_index"] <= 4]
        .groupby(["repo_id", "repo_full_name"])
        .agg(
            early_4week_stars=("weekly_new_stars", "sum"),
            early_avg_weekly_stars=("weekly_new_stars", "mean"),
            early_max_weekly_stars=("weekly_new_stars", "max")
        )
        .reset_index()
    )

    later = (
        temp[temp["week_index"] > 4]
        .groupby(["repo_id", "repo_full_name"])
        .agg(
            later_stars=("weekly_new_stars", "sum"),
            later_avg_weekly_stars=("weekly_new_stars", "mean")
        )
        .reset_index()
    )

    modeling = early.merge(later, on=["repo_id", "repo_full_name"], how="left")
    modeling["later_stars"] = modeling["later_stars"].fillna(0)
    modeling["later_avg_weekly_stars"] = modeling["later_avg_weekly_stars"].fillna(0)
    modeling["became_high_growth"] = (modeling["later_stars"] >= modeling["later_stars"].median()).astype(int)

    repo_cols = [
        "repo_id", "language", "topics", "stars_count", "forks_count",
        "watchers_count", "open_issues_count", "created_at", "updated_at", "pushed_at",
        "archived", "is_fork", "license", "html_url"
    ]
    repo_cols = [c for c in repo_cols if c in repos_df.columns]
    modeling = modeling.merge(repos_df[repo_cols].drop_duplicates("repo_id"), on="repo_id", how="left")

    modeling.to_csv(outdir / "repo_herd_modeling_table.csv", index=False)
    return modeling


def write_data_dictionary(outdir):
    text = """# Data Dictionary

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
"""
    (outdir / "DATA_DICTIONARY.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topics",
        nargs="+",
        default=[
            "machine-learning", "data-science", "deep-learning", "artificial-intelligence",
            "python", "react", "javascript", "web-development", "llm", "generative-ai"
        ]
    )
    parser.add_argument("--repos-per-topic", type=int, default=100)
    parser.add_argument("--min-stars", type=int, default=50)
    parser.add_argument("--max-repos-for-stars", type=int, default=100)
    parser.add_argument("--max-stargazers-per-repo", type=int, default=1000)
    parser.add_argument("--outdir", type=str, default="github_herd_real_dataset_output")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-enrichment", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    session = make_session()
    resume = not args.no_resume

    repos_df = search_repositories(
        session=session,
        topics=args.topics,
        repos_per_topic=args.repos_per_topic,
        min_stars=args.min_stars,
        outdir=outdir,
        resume=resume
    )

    if args.skip_enrichment:
        final_repos = repos_df
        final_repos.to_csv(outdir / "repositories_enriched.csv", index=False)
    else:
        final_repos = enrich_repository_details(session, repos_df, outdir, resume=resume)

    stars_df = collect_stargazers(
        session=session,
        repos_df=final_repos,
        outdir=outdir,
        max_repos_for_stars=args.max_repos_for_stars,
        max_stargazers_per_repo=args.max_stargazers_per_repo,
        resume=resume
    )

    daily_df, weekly_df = build_timeseries(stars_df, outdir)
    interactions_df = build_interactions(stars_df, outdir)
    modeling_df = build_modeling_table(final_repos, weekly_df, outdir)
    write_data_dictionary(outdir)

    summary = {
        "created_at": utc_now(),
        "topics": ",".join(args.topics),
        "num_repositories": int(len(final_repos)),
        "num_repositories_with_stargazers": int(stars_df["repo_id"].nunique()) if not stars_df.empty else 0,
        "num_star_interactions": int(len(stars_df)),
        "num_unique_users": int(stars_df["user_id"].nunique()) if not stars_df.empty else 0,
        "num_daily_timeseries_rows": int(len(daily_df)),
        "num_weekly_timeseries_rows": int(len(weekly_df)),
        "num_recommendation_interactions": int(len(interactions_df)),
        "num_modeling_rows": int(len(modeling_df)),
        "used_token": bool(os.getenv("GITHUB_TOKEN"))
    }

    pd.DataFrame([summary]).to_csv(outdir / "dataset_summary.csv", index=False)
    save_json(outdir / "dataset_summary.json", summary)

    print("\nDone. Output folder:")
    print(outdir)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
