"""
Shared multilingual sentence-embedding helper for CommentLens.

Runs the ONNX-quantized export of intfloat/multilingual-e5-small
(converted at Xenova/multilingual-e5-small) through onnxruntime, so
neither app.py (Render, 512MB free tier) nor daily_update.py (GitHub
Actions) need PyTorch / sentence-transformers just to embed text.

Pooling mirrors what multilingual-e5-small itself uses: mean-pool the
last hidden state over non-padding tokens, then L2-normalize - the
same kind of vectors sentence-transformers would produce, just
without the heavy dependency.
"""

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

EMBED_MODEL_REPO = "Xenova/multilingual-e5-small"
EMBED_ONNX_FILENAME = "onnx/model_quantized.onnx"

_tokenizer = None
_session = None


def _load():
    global _tokenizer, _session
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_REPO)
    if _session is None:
        onnx_path = hf_hub_download(EMBED_MODEL_REPO, EMBED_ONNX_FILENAME)
        _session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def embed_texts(texts):
    """Return L2-normalized (len(texts), 384) float32 embeddings for a list of strings."""
    _load()
    enc = _tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )
    onnx_inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in enc:
        onnx_inputs["token_type_ids"] = enc["token_type_ids"].astype(np.int64)

    last_hidden_state = _session.run(None, onnx_inputs)[0]

    mask = enc["attention_mask"].astype(np.float32)[:, :, None]
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    mean_pooled = summed / counts

    norms = np.clip(np.linalg.norm(mean_pooled, axis=1, keepdims=True), 1e-9, None)
    return (mean_pooled / norms).astype("float32")
