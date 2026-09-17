"""
CommentLens inference API.

Loads the fine-tuned BanglishBERT (ELECTRA) 5-class comment classifier
from gulamsakaria/commentlens-banglishbert (the ONNX-quantized export,
for a light, fast CPU footprint) and serves it over a small FastAPI app.

Classes: claim, general, opinion, spam-scam, toxic
"""

import time
import json
import logging
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("commentlens-api")

MODEL_REPO = "gulamsakaria/commentlens-banglishbert"
ONNX_FILENAME = "onnx/model_quantized.onnx"
MAX_LENGTH = 128

NEWS_INDEX_REPO = "gulamsakaria/commentlens-news-index"

ID2LABEL = {0: "claim", 1: "general", 2: "opinion", 3: "spam-scam", 4: "toxic"}

app = FastAPI(
    title="CommentLens API",
    description="Bangla/Banglish comment classifier (claim / general / opinion / spam-scam / toxic)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_tokenizer = None
_session = None
_load_seconds = None

# News-matching is done with a TF-IDF (character n-gram) index built at
# load time from meta.json's headlines - no second neural model, so this
# has to share Render's 512MB free tier only with the onnxruntime
# classifier session above, and comfortably does.
_tfidf_vectorizer = None
_tfidf_matrix = None
_news_meta: List[dict] = []
_news_index_ready = False
_news_index_loaded_at = None


def _onnx_session_options():
    # Keep the ONNX Runtime memory footprint as small as possible.
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return opts


def _load_model():
    global _tokenizer, _session, _load_seconds
    t0 = time.time()
    logger.info("Loading tokenizer from %s ...", MODEL_REPO)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    logger.info("Downloading ONNX weights (%s) ...", ONNX_FILENAME)
    onnx_path = hf_hub_download(MODEL_REPO, ONNX_FILENAME)
    logger.info("Starting ONNX Runtime session ...")
    _session = ort.InferenceSession(
        onnx_path,
        sess_options=_onnx_session_options(),
        providers=["CPUExecutionProvider"],
    )
    _load_seconds = round(time.time() - t0, 2)
    logger.info("Model ready in %.2fs", _load_seconds)


def _load_news_index():
    """Download the headline metadata for the news-matching feature from
    the commentlens-news-index dataset repo on Hugging Face and build a
    TF-IDF (character n-gram) index over it in memory. Raises if the
    metadata file isn't there yet."""
    global _tfidf_vectorizer, _tfidf_matrix, _news_meta, _news_index_ready, _news_index_loaded_at

    logger.info("Downloading news index metadata from %s ...", NEWS_INDEX_REPO)
    meta_path = hf_hub_download(NEWS_INDEX_REPO, "meta.json", repo_type="dataset")
    with open(meta_path, encoding="utf-8") as f:
        _news_meta = json.load(f)

    if _news_meta:
        headlines = [rec.get("headline", "") for rec in _news_meta]
        # char n-grams work for Bangla script and romanized Banglish alike,
        # and don't depend on a language-specific word tokenizer.
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000)
        _tfidf_matrix = vectorizer.fit_transform(headlines)
        _tfidf_vectorizer = vectorizer
    else:
        _tfidf_vectorizer = None
        _tfidf_matrix = None

    _news_index_ready = True
    _news_index_loaded_at = time.time()
    logger.info("News index loaded: %d headlines", len(_news_meta))


@app.on_event("startup")
def startup_event():
    _load_model()
    try:
        _load_news_index()
    except Exception as exc:
        logger.warning(
            "News index not available yet (%s) - /match_claim will 503 "
            "until /reload_index is called successfully.",
            exc,
        )


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Bangla or Banglish comment text")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    scores: Dict[str, float]
    inference_ms: float


class MatchClaimRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Claim or headline text to search for")
    top_k: int = Field(5, ge=1, le=20, description="Number of nearest matches to return")


class NewsMatch(BaseModel):
    headline: str
    date: Optional[str] = None
    link: Optional[str] = None
    score: float


class MatchClaimResponse(BaseModel):
    query: str
    matches: List[NewsMatch]


class ReloadIndexResponse(BaseModel):
    status: str
    vectors: int
    meta_records: int


@app.get("/")
def root():
    return {
        "status": "ok" if _session is not None else "loading",
        "model": MODEL_REPO,
        "labels": list(ID2LABEL.values()),
        "model_load_seconds": _load_seconds,
    }


@app.get("/health")
def health():
    if _session is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _session is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    t0 = time.time()
    enc = _tokenizer(req.text, return_tensors="np", truncation=True, max_length=MAX_LENGTH)
    inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
        "token_type_ids": enc.get(
            "token_type_ids", np.zeros_like(enc["input_ids"])
        ).astype(np.int64),
    }
    logits = _session.run(None, inputs)[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    label_idx = int(np.argmax(probs))
    inference_ms = round((time.time() - t0) * 1000, 2)

    return PredictResponse(
        label=ID2LABEL[label_idx],
        confidence=float(probs[label_idx]),
        scores={ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
        inference_ms=inference_ms,
    )


@app.post("/match_claim", response_model=MatchClaimResponse)
def match_claim(req: MatchClaimRequest):
    if not _news_index_ready:
        raise HTTPException(
            status_code=503,
            detail="news index not loaded yet - call /reload_index",
        )

    if _tfidf_matrix is None or _tfidf_vectorizer is None:
        # index is ready but has no articles indexed yet
        return MatchClaimResponse(query=req.text, matches=[])

    query_vector = _tfidf_vectorizer.transform([req.text])
    scores = linear_kernel(query_vector, _tfidf_matrix)[0]
    top_indices = scores.argsort()[::-1][: req.top_k]

    matches = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        record = _news_meta[idx]
        matches.append(
            NewsMatch(
                headline=record.get("headline", ""),
                date=record.get("date"),
                link=record.get("link"),
                score=score,
            )
        )

    return MatchClaimResponse(query=req.text, matches=matches)


@app.post("/reload_index", response_model=ReloadIndexResponse)
def reload_index():
    try:
        _load_news_index()
    except Exception as exc:
        logger.exception("Failed to reload news index")
        raise HTTPException(status_code=502, detail=f"failed to reload news index: {exc}")

    return ReloadIndexResponse(
        status="ok",
        vectors=len(_news_meta),
        meta_records=len(_news_meta),
    )
