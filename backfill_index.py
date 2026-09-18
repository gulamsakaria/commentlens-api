"""
One-time backfill: rebuild meta.json in gulamsakaria/commentlens-news-index
from every daily CSV export already sitting in
gulamsakaria/commentlens-news-archive (exports/*.csv), instead of waiting
for daily_update.py to add them one day at a time.

Run manually via the "backfill-news-index" GitHub Actions workflow
(workflow_dispatch) after HF_ARCHIVE_TOKEN is set as a repo secret.
"""

import json
import os

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files, upload_file

INDEX_REPO = "gulamsakaria/commentlens-news-index"
ARCHIVE_REPO = "gulamsakaria/commentlens-news-archive"

# Same fallback as daily_update.py: HF_TOKEN is intentionally scoped
# write-only on commentlens-news-index, so reading the archive needs its
# own token unless one hasn't been configured yet.
ARCHIVE_TOKEN = os.environ.get("HF_ARCHIVE_TOKEN") or os.environ.get("HF_TOKEN")


def main():
    files = list_repo_files(ARCHIVE_REPO, repo_type="dataset", token=ARCHIVE_TOKEN)
    export_files = sorted(f for f in files if f.startswith("exports/") and f.endswith(".csv"))

    if not export_files:
        print("No exports/*.csv found in the archive repo -- nothing to backfill.")
        return

    print(f"Found {len(export_files)} daily exports: {export_files[0]} .. {export_files[-1]}")

    frames = []
    for f in export_files:
        path = hf_hub_download(ARCHIVE_REPO, f, repo_type="dataset", token=ARCHIVE_TOKEN)
        df = pd.read_csv(path)
        if not df.empty:
            # f is "exports/YYYY-MM-DD.csv" - that filename is the record's
            # real export_date (app.py needs this for QUOTE-claim body
            # lookups; it can differ from the article's own published_date,
            # see app.py's NEWS_ARCHIVE_REPO comment).
            df["export_date"] = os.path.splitext(os.path.basename(f))[0]
            frames.append(df)

    if not frames:
        print("Every export file was empty -- nothing to backfill.")
        return

    full_df = pd.concat(frames, ignore_index=True)
    full_df = full_df.rename(columns={"published_date": "date", "url": "link"})

    # de-dupe by link, keep the first occurrence
    full_df = full_df.drop_duplicates(subset="link", keep="first")

    meta = full_df[["headline", "date", "link", "export_date"]].to_dict(orient="records")

    json.dump(meta, open("meta.json", "w", encoding="utf-8"), ensure_ascii=False)

    upload_file(
        path_or_fileobj="meta.json",
        path_in_repo="meta.json",
        repo_id=INDEX_REPO,
        repo_type="dataset",
        token=os.environ["HF_TOKEN"],
    )
    print(f"Backfilled {len(meta)} articles ({export_files[0]} .. {export_files[-1]}) into {INDEX_REPO}.")


if __name__ == "__main__":
    main()
