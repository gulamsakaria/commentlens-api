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

# --- News-matching tuning -----------------------------------------------
# These are starting points, not exact science - tune them against a
# handful of known true-positive and known false-positive claims once the
# index has real data in it (see "Not done yet" note at the bottom of this
# file for how to do that).
#
# A candidate headline only counts as a "match" if BOTH signals clear their
# own floor. This is the fix for a specific failure mode: a claim naming a
# person (e.g. a politician) was matching unrelated headlines that only
# shared generic topic words ("doctor", "Malaysia") with the claim, and
# none of the matched headlines even mentioned the person named in the
# claim. Char n-grams alone can't tell a generic shared word from a shared
# name - word-level TF-IDF can, because a word's IDF weight drops the more
# headlines it appears in, so a common topic word ends up contributing
# little to the score while a name that appears in only a few headlines
# contributes a lot. Requiring both signals means a match can't be driven
# by common vocabulary alone.
# Generic geographic/connector/frequent words that, on their own, say
# nothing about WHAT a headline is actually about - they showed up as the
# cause of a second false-positive pattern: a claim mentioning "দক্ষিণ
# এশিয়ার" (South Asia) matched unrelated headlines that only shared those
# two words, because a small archive doesn't have enough documents for
# TF-IDF's IDF weighting to recognize them as generic on its own. Stripping
# them out of the word-level vectorizer means they contribute nothing to
# word_score - a match can only come from words that actually describe the
# subject. Extend this list as new false positives like this turn up.
BANGLA_STOPWORDS = [
    "দক্ষিণ", "উত্তর", "পূর্ব", "পশ্চিম",
    "এশিয়া", "এশিয়ার", "বিশ্ব", "বিশ্বের",
    "বাংলাদেশ", "বাংলাদেশের", "দেশ", "দেশের", "সরকার", "সরকারের",
    "নিয়ে", "বিষয়ে", "সম্পর্কে", "জানিয়েছে", "জানান", "বলেছেন", "বললেন",
    "এবং", "ও", "এর", "একটি", "এই", "সেই", "আজ", "গতকাল", "নতুন",
]

MIN_CHAR_SCORE = 0.12
MIN_WORD_SCORE = 0.08
# A match can pass the floor above yet still be a fairly weak, low-confidence
# echo. "backed" is reserved for a genuinely strong match so the popup badge
# doesn't call a weak coincidence "news-backed".
BACKED_MIN_SCORE = 0.25

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

# News-matching uses two TF-IDF indexes built at load time from meta.json's
# headlines - no second neural model, so this has to share Render's 512MB
# free tier only with the onnxruntime classifier session above, and
# comfortably does.
_tfidf_vectorizer = None   # char n-gram index (spelling/inflection variants)
_tfidf_matrix = None
_word_vectorizer = None    # word-level index (the entity/topic discriminator)
_word_matrix = None
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
    the commentlens-news-index dataset repo on Hugging Face and build two
    TF-IDF indexes over it in memory (char n-gram + word-level - see the
    tuning comment near MIN_CHAR_SCORE for why both). Raises if the
    metadata file isn't there yet."""
    global _tfidf_vectorizer, _tfidf_matrix, _word_vectorizer, _word_matrix
    global _news_meta, _news_index_ready, _news_index_loaded_at

    logger.info("Downloading news index metadata from %s ...", NEWS_INDEX_REPO)
    meta_path = hf_hub_download(NEWS_INDEX_REPO, "meta.json", repo_type="dataset")
    with open(meta_path, encoding="utf-8") as f:
        _news_meta = json.load(f)

    if _news_meta:
        headlines = [rec.get("headline", "") for rec in _news_meta]

        # char n-grams work for Bangla script and romanized Banglish alike,
        # and don't depend on a language-specific word tokenizer - good at
        # catching spelling/inflection variants of the same word.
        char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000)
        _tfidf_matrix = char_vectorizer.fit_transform(headlines)
        _tfidf_vectorizer = char_vectorizer

        # word-level TF-IDF: the discriminator between "shares a common
        # topic word" and "shares the actual distinctive subject" - see the
        # tuning comment near MIN_CHAR_SCORE above for why this matters.
        word_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_features=50000,
            stop_words=BANGLA_STOPWORDS,
        )
        _word_matrix = word_vectorizer.fit_transform(headlines)
        _word_vectorizer = word_vectorizer
    else:
        _tfidf_vectorizer = None
        _tfidf_matrix = None
        _word_vectorizer = None
        _word_matrix = None

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
    # Reserved for a genuinely strong match (see BACKED_MIN_SCORE) - the
    # popup's badge reads this directly, it no longer has to guess it from
    # matches.length on the frontend.
    backed: bool = False


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
    if _tfidf_matrix is None or _tfidf_vectorizer is None or _word_matrix is None or _word_vectorizer is None:
        # index is ready but has no articles indexed yet
        return MatchClaimResponse(query=req.text, matches=[], backed=False)

    char_scores = linear_kernel(_tfidf_vectorizer.transform([req.text]), _tfidf_matrix)[0]
    word_scores = linear_kernel(_word_vectorizer.transform([req.text]), _word_matrix)[0]
    # Average the two signals into one ranking/display score - char catches
    # spelling variants, word catches the actual distinctive subject; a
    # genuinely relevant headline should score reasonably on both.
    combined_scores = (char_scores + word_scores) / 2

    top_indices = combined_scores.argsort()[::-1][: req.top_k]

    matches = []
    for idx in top_indices:
        char_score = float(char_scores[idx])
        word_score = float(word_scores[idx])
        # Both signals must individually clear their floor - a headline
        # that only wins on shared common words (high char, near-zero word)
        # or only on a coincidental character overlap (high word from
        # noise, near-zero char) gets dropped either way.
        if char_score < MIN_CHAR_SCORE or word_score < MIN_WORD_SCORE:
            continue
        record = _news_meta[idx]
        matches.append(
            NewsMatch(
                headline=record.get("headline", ""),
                date=record.get("date"),
                link=record.get("link"),
                score=float(combined_scores[idx]),
            )
        )

    backed = any(m.score >= BACKED_MIN_SCORE for m in matches)
    return MatchClaimResponse(query=req.text, matches=matches, backed=backed)


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
