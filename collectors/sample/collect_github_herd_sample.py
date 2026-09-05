import os
import time
import argparse
from datetime import datetime
from collections import defaultdict

import requests
import pandas as pd


API_BASE = "https://api.github.com"


def make_session():
    token = os.getenv("GITHUB_TOKEN")
    session = requests.Session()
    session.headers.update({
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "github-herd-behavior-data-science-project"
    })
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})
    return session


def request_json(session, url, params=None, accept=None):
    headers = {}
    if accept:
        headers["Accept"] = accept

    while True:
        response = session.get(url, params=params, headers=headers, timeout=30)

        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            sleep_seconds = max(reset_time - int(time.time()) + 5, 5)
            print(f"Rate limit reached. Sleeping for {sleep_seconds} seconds...")
            time.sleep(sleep_seconds)
            continue

        if response.status_code in [429, 500, 502, 503, 504]:
            print(f"Temporary error {response.status_code}. Sleeping 10 seconds...")
            time.sleep(10)
            continue

        response.raise_for_status()
        return response.json()


def search_repositories(session, topics, repos_per_topic):
    rows = []
    seen = set()

    for topic in topics:
        print(f"Searching topic: {topic}")
        url = f"{API_BASE}/search/repositories"
        params = {
            "q": f"topic:{topic} stars:>50",
            "sort": "stars",
            "order": "desc",
            "per_page": min(repos_per_topic, 100),
            "page": 1
        }

        data = request_json(session, url, params=params)
        for repo in data.get("items", []):
            repo_id = repo.get("id")
            if repo_id in seen:
                continue
            seen.add(repo_id)

            rows.append({
                "repo_id": repo_id,
                "full_name": repo.get("full_name"),
                "owner": repo.get("owner", {}).get("login"),
                "repo_name": repo.get("name"),
                "description": repo.get("description"),
                "language": repo.get("language"),
                "topics": "|".join(repo.get("topics", [])),
                "stars_count": repo.get("stargazers_count"),
                "forks_count": repo.get("forks_count"),
                "watchers_count": repo.get("watchers_count"),
                "open_issues_count": repo.get("open_issues_count"),
                "created_at": repo.get("created_at"),
                "updated_at": repo.get("updated_at"),
                "pushed_at": repo.get("pushed_at"),
                "default_branch": repo.get("default_branch"),
                "archived": repo.get("archived"),
                "disabled": repo.get("disabled"),
                "license": (repo.get("license") or {}).get("spdx_id"),
                "html_url": repo.get("html_url"),
                "source_topic": topic
            })

    return pd.DataFrame(rows)


def collect_stargazers(session, repos_df, max_repos_for_stars, max_stargazers_per_repo):
    rows = []

    selected = repos_df.sort_values("stars_count", ascending=False).head(max_repos_for_stars)

    for _, repo in selected.iterrows():
        full_name = repo["full_name"]
        print(f"Collecting stargazers for: {full_name}")

        owner, repo_name = full_name.split("/", 1)
        collected = 0
        page = 1

        while collected < max_stargazers_per_repo:
            per_page = min(100, max_stargazers_per_repo - collected)
            url = f"{API_BASE}/repos/{owner}/{repo_name}/stargazers"
            params = {"per_page": per_page, "page": page}

            try:
                data = request_json(
                    session,
                    url,
                    params=params,
                    accept="application/vnd.github.star+json"
                )
            except requests.HTTPError as e:
                print(f"Skipping {full_name} because of error: {e}")
                break

            if not data:
                break

            for item in data:
                user = item.get("user", item)
                rows.append({
                    "repo_id": repo["repo_id"],
                    "repo_full_name": full_name,
                    "user_id": user.get("id"),
                    "username": user.get("login"),
                    "user_type": user.get("type"),
                    "starred_at": item.get("starred_at")
                })

            collected += len(data)
            page += 1
            time.sleep(0.2)

    return pd.DataFrame(rows)


def build_timeseries(stars_df):
    valid = stars_df.dropna(subset=["starred_at"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=[
            "repo_id", "repo_full_name", "date",
            "daily_new_stars", "cumulative_stars", "weekly_new_stars"
        ])

    valid["date"] = pd.to_datetime(valid["starred_at"], utc=True).dt.date
    daily = (
        valid.groupby(["repo_id", "repo_full_name", "date"])
        .size()
        .reset_index(name="daily_new_stars")
        .sort_values(["repo_id", "date"])
    )

    daily["cumulative_stars"] = daily.groupby("repo_id")["daily_new_stars"].cumsum()
    daily["date_dt"] = pd.to_datetime(daily["date"])
    daily["week"] = daily["date_dt"].dt.to_period("W").astype(str)

    weekly_map = (
        daily.groupby(["repo_id", "week"])["daily_new_stars"]
        .sum()
        .rename("weekly_new_stars")
        .reset_index()
    )

    daily = daily.merge(weekly_map, on=["repo_id", "week"], how="left")
    return daily.drop(columns=["date_dt", "week"])


def build_user_item_matrix(stars_df):
    matrix = stars_df[["user_id", "username", "repo_id", "repo_full_name", "starred_at"]].copy()
    matrix["interaction"] = 1
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topics",
        nargs="+",
        default=["machine-learning", "data-science", "deep-learning", "react", "python"],
        help="GitHub topics to search."
    )
    parser.add_argument("--repos-per-topic", type=int, default=20)
    parser.add_argument("--max-repos-for-stars", type=int, default=15)
    parser.add_argument("--max-stargazers-per-repo", type=int, default=300)
    parser.add_argument("--outdir", type=str, default="github_herd_sample_output")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    session = make_session()

    repos_df = search_repositories(session, args.topics, args.repos_per_topic)
    repos_path = outdir / "repositories_sample.csv"
    repos_df.to_csv(repos_path, index=False)
    print(f"Saved {len(repos_df)} repositories to {repos_path}")

    stars_df = collect_stargazers(
        session,
        repos_df,
        args.max_repos_for_stars,
        args.max_stargazers_per_repo
    )
    stars_path = outdir / "stars_sample.csv"
    stars_df.to_csv(stars_path, index=False)
    print(f"Saved {len(stars_df)} star interactions to {stars_path}")

    timeseries_df = build_timeseries(stars_df)
    timeseries_path = outdir / "repo_star_timeseries_sample.csv"
    timeseries_df.to_csv(timeseries_path, index=False)
    print(f"Saved {len(timeseries_df)} time-series rows to {timeseries_path}")

    user_item_df = build_user_item_matrix(stars_df)
    user_item_path = outdir / "user_repo_interactions_sample.csv"
    user_item_df.to_csv(user_item_path, index=False)
    print(f"Saved {len(user_item_df)} user-repo interactions to {user_item_path}")

    summary = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "topics": ",".join(args.topics),
        "num_repositories": len(repos_df),
        "num_star_interactions": len(stars_df),
        "num_unique_users": stars_df["user_id"].nunique() if not stars_df.empty else 0,
        "num_timeseries_rows": len(timeseries_df)
    }
    pd.DataFrame([summary]).to_csv(outdir / "sample_summary.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    from pathlib import Path
    main()
