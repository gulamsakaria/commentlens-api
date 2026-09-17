import os, json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download, upload_file

REPO_ID = "gulamsakaria/commentlens-news-index"
MODEL_NAME = "intfloat/multilingual-e5-small"
TODAY_CSV = "todays_news.csv"  # TODO: point this at wherever the existing
                                # daily scraper writes today's new rows
                                # (columns: headline, body, date, link)

def main():
    if not os.path.exists(TODAY_CSV):
        print("No new news file found today — skipping.")
        return
    new_df = pd.read_csv(TODAY_CSV)
    if new_df.empty:
        print("No new rows today — skipping.")
        return

    model = SentenceTransformer(MODEL_NAME)
    texts = ("query: " + new_df["headline"].fillna("")).tolist()
    new_vectors = model.encode(texts, normalize_embeddings=True)

    idx_path = hf_hub_download(REPO_ID, "index.faiss", repo_type="dataset")
    meta_path = hf_hub_download(REPO_ID, "meta.json", repo_type="dataset")
    index = faiss.read_index(idx_path)
    meta = json.load(open(meta_path, encoding="utf-8"))

    index.add(np.array(new_vectors, dtype="float32"))
    meta.extend(new_df[["headline", "date", "link"]].to_dict(orient="records"))

    faiss.write_index(index, "index.faiss")
    json.dump(meta, open("meta.json", "w", encoding="utf-8"), ensure_ascii=False)

    upload_file(path_or_fileobj="index.faiss", path_in_repo="index.faiss",
                repo_id=REPO_ID, repo_type="dataset", token=os.environ["HF_TOKEN"])
    upload_file(path_or_fileobj="meta.json", path_in_repo="meta.json",
                repo_id=REPO_ID, repo_type="dataset", token=os.environ["HF_TOKEN"])
    print(f"Added {len(new_df)} new articles to the index.")

if __name__ == "__main__":
    main()
