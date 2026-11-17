"""Compare the current TF-IDF news matching with dense and hybrid retrieval.

For a set of articles we take a sentence of the article body as the "claim" and
check whether each method finds that article's headline among the indexed
headlines. Two kinds of query:
  lead  the first sentence of the body (usually shares many words with the headline)
  mid   a later sentence of the body (needs meaning, not just shared words)

Methods
  tfidf   the API's current score: (char 3-5-gram TF-IDF + word TF-IDF) / 2
  dense   multilingual-e5-small (int8 ONNX) cosine similarity, FAISS flat index
  hybrid  Reciprocal Rank Fusion of the two rankings

Reported: Recall@1, Recall@5, Recall@10 and MRR@10, plus encode speed and memory.

Run on a public Bangla news corpus (needs no token):
    python eval_hybrid.py --source public --parquet part0.parquet
Run on the team's own archive (needs HF_TOKEN with read access to commentlens-news-archive):
    python eval_hybrid.py --source archive --days 30

This only measures retrieval quality. It does not change the API.
"""
import argparse
import json
import os
import random
import re
import resource
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from hybrid_retrieval import DenseEncoder, DenseIndex, E5_REPO, hybrid_search, make_int8_model

ARCHIVE_REPO = "gulamsakaria/commentlens-news-archive"
BANGLA_WORD_TOKEN_PATTERN = "[" + chr(0x0980) + "-" + chr(0x09FF) + chr(0x200C) + chr(0x200D) + "]+|[A-Za-z0-9]+"   # same as app.py
KS = (1, 5, 10)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def sentences(body: str) -> List[str]:
    parts = re.split(r"(?<=[।?!\.])\s+", body.strip())
    return [p.strip() for p in parts if 5 <= len(p.split()) <= 40]


def load_public(parquet: str) -> List[Tuple[str, str]]:
    import pandas as pd

    df = pd.read_parquet(parquet, columns=["title", "content"]).dropna()
    return list(zip(df["title"].str.strip(), df["content"].str.strip()))


def load_archive(days: int) -> List[Tuple[str, str]]:
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (read access to the news archive) to use --source archive.")
    files = sorted(f for f in HfApi().list_repo_files(ARCHIVE_REPO, repo_type="dataset", token=token) if f.startswith("exports/"))
    rows: List[Tuple[str, str]] = []
    for name in files[-days:]:
        path = hf_hub_download(ARCHIVE_REPO, name, repo_type="dataset", token=token)
        df = pd.read_csv(path, usecols=["headline", "body"]).dropna()
        rows += list(zip(df["headline"].str.strip(), df["body"].str.strip()))
    return rows


def prepare(articles: List[Tuple[str, str]], corpus_size: int, n_queries: int, seed: int):
    rng = random.Random(seed)
    seen, unique = set(), []
    for title, body in articles:
        key = normalize(title)
        if key and key not in seen and body:
            seen.add(key)
            unique.append((key, body))
    rng.shuffle(unique)
    corpus = unique[:corpus_size]
    headlines = [h for h, _ in corpus]
    queries: Dict[str, List[Tuple[str, int]]] = {"lead": [], "mid": []}
    order = list(range(len(corpus)))
    rng.shuffle(order)
    for idx in order:
        sents = sentences(corpus[idx][1])
        if len(sents) >= 4 and len(queries["lead"]) < n_queries:
            queries["lead"].append((normalize(sents[0])[:300], idx))
            queries["mid"].append((normalize(rng.choice(sents[2:]))[:300], idx))
    return headlines, queries


def rank_metrics(top_lists: List[List[int]], targets: List[int]) -> Dict[str, float]:
    result = {}
    for k in KS:
        result[f"recall@{k}"] = round(sum(t in top[:k] for top, t in zip(top_lists, targets)) / len(targets), 4)
    rr = [1.0 / (top.index(t) + 1) if t in top else 0.0 for top, t in zip(top_lists, targets)]
    result["mrr@10"] = round(sum(rr) / len(rr), 4)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["public", "archive"], required=True)
    ap.add_argument("--parquet", help="parquet file with title and content columns (public source)")
    ap.add_argument("--days", type=int, default=30, help="most recent archive days to use (archive source)")
    ap.add_argument("--corpus", type=int, default=20000, help="number of headlines to index")
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-dir", default="e5")
    ap.add_argument("--out", default="hybrid_eval_results.json")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    int8 = model_dir / "model_int8.onnx"
    tok = model_dir / "onnx" / "sentencepiece.bpe.model"
    if not int8.exists() or not tok.exists():
        from huggingface_hub import hf_hub_download

        fp32 = hf_hub_download(E5_REPO, "onnx/model.onnx", local_dir=model_dir)
        hf_hub_download(E5_REPO, "onnx/sentencepiece.bpe.model", local_dir=model_dir)
        make_int8_model(fp32, str(int8))

    articles = load_public(args.parquet) if args.source == "public" else load_archive(args.days)
    headlines, queries = prepare(articles, args.corpus, args.queries, args.seed)
    print(f"corpus: {len(headlines)} headlines; queries: {len(queries['lead'])} per type", flush=True)

    # --- the API's current TF-IDF scoring, built exactly as in app.py ---
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000)
    char_matrix = char_vec.fit_transform(headlines)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=50000, max_df=0.15, token_pattern=BANGLA_WORD_TOKEN_PATTERN)
    word_matrix = word_vec.fit_transform(headlines)

    def tfidf_scores(text: str) -> np.ndarray:
        return (linear_kernel(char_vec.transform([text]), char_matrix)[0] + linear_kernel(word_vec.transform([text]), word_matrix)[0]) / 2

    # --- dense index ---
    encoder = DenseEncoder(str(int8), str(tok), threads=2)
    t0 = time.time()
    vectors = encoder.encode(headlines)
    encode_seconds = time.time() - t0
    index = DenseIndex(vectors)
    print(f"encoded {len(headlines)} headlines in {encode_seconds:.0f}s ({len(headlines) / encode_seconds:.0f}/s)", flush=True)

    results: Dict[str, Dict] = {}
    for kind, items in queries.items():
        texts = [t for t, _ in items]
        targets = [i for _, i in items]
        qvecs = encoder.encode(texts)
        _, dense_ids = index.search(qvecs, 50)
        tops = {"tfidf": [], "dense": [], "hybrid": []}
        for row, text in enumerate(texts):
            scores = tfidf_scores(text)
            tops["tfidf"].append(np.argsort(scores)[::-1][:10].tolist())
            tops["dense"].append([int(i) for i in dense_ids[row][:10]])
            fused = hybrid_search(scores, index, qvecs[row], pool=50, top_n=10)
            tops["hybrid"].append([f["index"] for f in fused])
        results[kind] = {m: rank_metrics(tops[m], targets) for m in tops}

    # --- speed and memory of the dense part, one query at a time (what the API would do) ---
    single = []
    for text in [t for t, _ in queries["lead"][:50]]:
        s = time.time()
        encoder.encode([text])
        single.append((time.time() - s) * 1000)
    summary = {
        "source": args.source,
        "corpus_headlines": len(headlines),
        "queries_per_type": len(queries["lead"]),
        "results": results,
        "dense_query_encode_ms_median": round(float(np.median(single)), 1),
        "headline_encode_per_second": round(len(headlines) / encode_seconds, 1),
        "vectors_megabytes": round(vectors.nbytes / 1e6, 1),
        "process_peak_rss_megabytes": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
