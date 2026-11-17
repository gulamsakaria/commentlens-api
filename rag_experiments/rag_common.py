"""Shared helpers for the three RAG experiments (chunking, contextual, graph).

Task used everywhere: "claim -> article". The corpus is a sample of Bangla news
articles; a query is either the article's headline or one sentence taken out of
the body (and removed from the indexed text, so there is no verbatim match).
The retriever has to return the right article. Article score = best chunk score.

Experimental only. None of this is wired into the production API.
"""
import json
import random
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.insert(0, "/home/claude/api-clone")
from hybrid_retrieval import DenseEncoder, DenseIndex  # noqa: E402

PARQUET = "/home/claude/hybrid-work/part0.parquet"
E5_INT8 = "/home/claude/hybrid-work/e5/model_int8.onnx"
E5_SPM = "/home/claude/hybrid-work/e5/onnx/sentencepiece.bpe.model"
BN = "[" + chr(0x0980) + "-" + chr(0x09FF) + chr(0x200C) + chr(0x200D) + "]+|[A-Za-z0-9]+"


def norm(t: str) -> str:
    return unicodedata.normalize("NFC", t)


def split_sentences(body: str) -> List[str]:
    parts = re.split(r"(?<=[।?!\.])\s*", norm(body).strip())
    return [p.strip() for p in parts if len(p.split()) >= 3]


def load_articles(n: int, seed: int = 7, min_words: int = 150, max_words: int = 900) -> List[Dict]:
    """n random articles, long enough that chunking matters; each gets one held-out sentence."""
    df = pd.read_parquet(PARQUET, columns=["title", "content"]).dropna()
    df["title"] = df["title"].map(lambda s: norm(s.strip()))
    df = df[df["content"].str.split().str.len().between(min_words, max_words)]
    df = df.drop_duplicates("title")
    df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    rng = random.Random(seed)
    arts = []
    for i, row in df.iterrows():
        sents = split_sentences(row["content"])
        usable = [j for j in range(2, len(sents) - 2) if 8 <= len(sents[j].split()) <= 35]
        if len(sents) < 8 or not usable:
            continue
        held = rng.choice(usable)
        arts.append({
            "id": len(arts),
            "title": row["title"],
            "sentences": [s for j, s in enumerate(sents) if j != held],   # held-out sentence removed
            "heldout": sents[held],
        })
    return arts


def encoder(threads: int = 2, max_length: int = 256) -> DenseEncoder:
    return DenseEncoder(E5_INT8, E5_SPM, max_length=max_length, threads=threads)


def rank_articles(scores: np.ndarray, chunk_doc: np.ndarray, n_docs: int) -> np.ndarray:
    """Best-chunk score per article -> article ranking (descending)."""
    best = np.full(n_docs, -1e9)
    np.maximum.at(best, chunk_doc, scores)
    return np.argsort(-best)


def recall_report(rankings: Sequence[np.ndarray], gold: Sequence[int]) -> Dict[str, float]:
    ranks = []
    for r, g in zip(rankings, gold):
        pos = np.where(r == g)[0]
        ranks.append(int(pos[0]) + 1 if len(pos) else 10 ** 6)
    ranks = np.array(ranks)
    out = {f"recall@{k}": round(float((ranks <= k).mean()), 3) for k in (1, 5, 10)}
    out["mrr@10"] = round(float(np.where(ranks <= 10, 1.0 / ranks, 0.0).mean()), 3)
    return out


class SparseChunkIndex:
    """char 3-5 + word 1-2 TF-IDF over chunk texts, same family as the production matcher."""

    def __init__(self, texts: Sequence[str]):
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
        self.word = TfidfVectorizer(token_pattern=BN, ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.mc = self.char.fit_transform(texts)
        self.mw = self.word.fit_transform(texts)

    def scores(self, query: str) -> np.ndarray:
        qc, qw = self.char.transform([query]), self.word.transform([query])
        return (linear_kernel(qc, self.mc)[0] + linear_kernel(qw, self.mw)[0]) / 2


def rrf(rank_lists: Sequence[np.ndarray], k: int = 60) -> np.ndarray:
    score: Dict[int, float] = {}
    for rl in rank_lists:
        for pos, d in enumerate(rl[:100], start=1):
            score[int(d)] = score.get(int(d), 0.0) + 1.0 / (k + pos)
    return np.array([d for d, _ in sorted(score.items(), key=lambda kv: -kv[1])])


def save(path: str, obj) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2))
