"""Hybrid retrieval for /match_claim: TF-IDF ranking + dense (embedding) ranking.

EXPERIMENTAL, not wired into app.py. It adds what the current news matching does
not have: a multilingual embedding of every headline, searched with FAISS, whose
ranking is fused with the existing TF-IDF ranking by Reciprocal Rank Fusion.

Pieces
  DenseEncoder   multilingual-e5-small as an int8 ONNX model (about 118 MB) run with
                 onnxruntime; text is tokenised with the model's own sentencepiece file
  DenseIndex     a FAISS flat inner-product index over normalised headline vectors
  rrf_fuse       Reciprocal Rank Fusion of any number of ranked lists
  hybrid_search  TF-IDF scores + dense index -> fused ranking

Build the int8 model once (make_int8_model) and the headline vectors once per index
update (not in the API process): the API only needs to encode the incoming claim.
"""
from typing import Dict, List, Sequence, Tuple

import faiss
import numpy as np
import onnxruntime as ort
import sentencepiece as spm

E5_REPO = "intfloat/multilingual-e5-small"
E5_PREFIX = "query: "   # e5 uses "query: " on both sides for symmetric short-text matching
RRF_K = 60              # the usual Reciprocal Rank Fusion constant


def make_int8_model(fp32_path: str, int8_path: str) -> None:
    """Dynamic int8 quantisation: 470 MB fp32 -> about 118 MB. Run once, offline."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"])


class DenseEncoder:
    """Turns texts into L2-normalised 384-dim vectors."""

    def __init__(self, model_path: str, spm_path: str, max_length: int = 64, threads: int = 1):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        # Memory arena and pattern cache off, to keep memory use low (loading the int8
        # session adds about 150 MB to the process).
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        # The Hugging Face tokenizers library needs about 300 MB of RAM for this 250k-piece
        # vocabulary; the sentencepiece library needs a few MB. XLM-R ids are the
        # sentencepiece ids plus one (unknown piece -> 3), with <s>=0, <pad>=1, </s>=2.
        self.sp = spm.SentencePieceProcessor(model_file=spm_path)
        self.max_length = max_length

    def _ids(self, text: str) -> List[int]:
        pieces = [3 if i == 0 else i + 1 for i in self.sp.encode(text)]
        return [0] + pieces[: self.max_length - 2] + [2]

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        out: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = [E5_PREFIX + t for t in texts[start:start + batch_size]]
            rows = [self._ids(t) for t in batch]
            longest = max(len(r) for r in rows)
            ids = np.array([r + [1] * (longest - len(r)) for r in rows], dtype=np.int64)
            mask = np.array([[1] * len(r) + [0] * (longest - len(r)) for r in rows], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feed)[0]                      # (batch, tokens, 384)
            m = mask[:, :, None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)   # mean pooling
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
            out.append(pooled.astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)


class DenseIndex:
    """Exact inner-product search (cosine, because the vectors are normalised)."""

    def __init__(self, vectors: np.ndarray):
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, query_vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        scores, ids = self.index.search(np.ascontiguousarray(query_vectors, dtype=np.float32), k)
        return scores, ids


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = RRF_K, top_n: int = 10) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank of d)."""
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[int(doc)] = scores.get(int(doc), 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


def hybrid_search(tfidf_scores: np.ndarray, dense_index: DenseIndex, query_vector: np.ndarray,
                  pool: int = 50, top_n: int = 10) -> List[Dict]:
    """Fuse the TF-IDF ranking with the dense ranking for one query.

    tfidf_scores: one score per indexed headline (the API's (char + word) / 2 score).
    Returns up to top_n dicts: index, rrf, tfidf, dense (cosine).
    """
    tfidf_top = np.argsort(tfidf_scores)[::-1][:pool]
    dense_scores, dense_ids = dense_index.search(query_vector.reshape(1, -1), pool)
    dense_top = [int(i) for i in dense_ids[0] if i >= 0]
    dense_lookup = {int(i): float(s) for i, s in zip(dense_ids[0], dense_scores[0]) if i >= 0}
    fused = rrf_fuse([tfidf_top.tolist(), dense_top], top_n=top_n)
    return [
        {
            "index": idx,
            "rrf": round(score, 6),
            "tfidf": float(tfidf_scores[idx]),
            "dense": dense_lookup.get(idx),
        }
        for idx, score in fused
    ]
