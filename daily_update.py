import os, json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from huggingface_hub import hf_hub_download, upload_file

INDEX_REPO = "gulamsakaria/commentlens-news-index"
ARCHIVE_REPO = "gulamsakaria/commentlens-news-archive"

# The existing daily news scraper (a separate job) writes one CSV per day to
# the commentlens-news-archive dataset repo, at exports/YYYY-MM-DD.csv, with
# columns: id, source, headline, body, url, published_date, category, scraped_at.
# This workflow runs at 03:00 Asia/Dhaka (21:00 UTC), just after midnight BDT,
# so the most recently *completed* scrape is "yesterday" in Dhaka time.
DHAKA = ZoneInfo("Asia/Dhaka")
TARGET_DATE = (datetime.now(DHAKA) - timedelta(days=1)).strftime("%Y-%m-%d")

# Reading the archive repo needs its own token (HF_ARCHIVE_TOKEN) since
# HF_TOKEN is intentionally scoped to write-only on commentlens-news-index.
# Falls back to HF_TOKEN if no separate archive token is configured yet.
ARCHIVE_TOKEN = os.environ.get("HF_ARCHIVE_TOKEN") or os.environ.get("HF_TOKEN")


def main():
    try:
        today_csv = hf_hub_download(
            ARCHIVE_REPO,
            f"exports/{TARGET_DATE}.csv",
            repo_type="dataset",
            token=ARCHIVE_TOKEN,
        )
    except Exception as exc:
        print(f"No archive export found for {TARGET_DATE} yet -- skipping ({exc}).")
        return

    new_df = pd.read_csv(today_csv)
    if new_df.empty:
        print(f"{TARGET_DATE} export was empty -- skipping.")
        return

    # normalize the archive's column names to what the index expects
    new_df = new_df.rename(columns={"published_date": "date", "url": "link"})

    meta_path = hf_hub_download(INDEX_REPO, "meta.json", repo_type="dataset")
    meta = json.load(open(meta_path, encoding="utf-8"))

    # skip anything already indexed (by link), in case this ever re-runs for
    # a date it already processed
    seen_links = {row.get("link") for row in meta if row.get("link")}
    new_df = new_df[~new_df["link"].isin(seen_links)]
    if new_df.empty:
        print(f"All {TARGET_DATE} rows were already indexed -- skipping.")
        return

    # No embedding step needed anymore - /match_claim builds a TF-IDF index
    # over the headlines at query time in app.py, so this job just has to
    # keep meta.json (headline + date + link) up to date.
    meta.extend(new_df[["headline", "date", "link"]].to_dict(orient="records"))

    json.dump(meta, open("meta.json", "w", encoding="utf-8"), ensure_ascii=False)

    upload_file(path_or_fileobj="meta.json", path_in_repo="meta.json",
        repo_id=INDEX_REPO, repo_type="dataset", token=os.environ["HF_TOKEN"])
    print(f"Added {len(new_df)} new articles ({TARGET_DATE}) to the index.")


if __name__ == "__main__":
    main()
