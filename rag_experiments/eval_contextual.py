"""Contextual retrieval (Anthropic-style) with a LOCAL model, on both chunkers.

For every chunk a local LLM (Qwen2.5-1.5B-Instruct, Q4_K_M GGUF, llama.cpp, CPU) writes one
short line that situates the chunk inside its article (the article's first --doc-words
words are given as the document). The prompt asks for Bangla, but the model answered in
English. That line is put in front of the chunk before indexing (TF-IDF and e5), as in
contextual retrieval. Queries are the same as in eval_chunking.py. Four conditions:

    fixed | fixed+context | variable | variable+context

The last one is the "combo": contextual retrieval on top of variable (semantic) chunking.
Generated lines are cached in contexts.jsonl so a stopped run can resume.

Only --articles articles are used (generation is slow on 2 CPU cores), so the corpus is
much smaller than in eval_chunking.py and absolute numbers are higher. Compare the four
conditions with each other, not with the other experiment.
The reported run used: --articles 40 --doc-words 60 --max-tokens 28
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from llama_cpp import Llama

from eval_chunking import fixed_chunks, semantic_chunks
from rag_common import DenseIndex, SparseChunkIndex, encoder, load_articles, rank_articles, recall_report, rrf, save

def h(t):
    return hashlib.md5(t.encode()).hexdigest()[:10]


PROMPT = (
    "<document>\n{doc}\n</document>\n\n"
    "Here is a chunk from that document:\n<chunk>\n{chunk}\n</chunk>\n\n"
    "Write one short sentence in Bangla (at most 25 words) that says what this chunk is about "
    "and how it fits in the whole document, to help a search engine find the chunk. "
    "Answer with only that sentence."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/claude/localllm/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--articles", type=int, default=120)
    ap.add_argument("--doc-words", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--cache", default="contexts.jsonl")
    ap.add_argument("--out", default="contextual_results.json")
    ap.add_argument("--smoke", type=int, default=0, help="only generate N contexts and print them")
    args = ap.parse_args()

    arts = load_articles(2000)[: args.articles]
    enc = encoder(threads=2, max_length=64)
    sent_vecs = {a["id"]: enc.encode(a["sentences"], batch_size=64) for a in arts}

    chunks = {"fixed": [], "variable": []}
    for a in arts:
        for name, cs in (("fixed", fixed_chunks(a["sentences"], 100)),
                         ("variable", semantic_chunks(a["sentences"], sent_vecs[a["id"]], 30.0, 50, 180))):
            chunks[name] += [(a["id"], c) for c in cs]
    print({k: len(v) for k, v in chunks.items()}, flush=True)

    cache = {}
    cpath = Path(args.cache)
    if cpath.exists():
        for line in cpath.read_text().splitlines():
            r = json.loads(line)
            cache[r["key"]] = r["context"]

    llm = Llama(model_path=args.model, n_ctx=3072, n_threads=2, verbose=False)
    by_art = {a["id"]: a for a in arts}
    t0, done = time.time(), 0
    for name in ("fixed", "variable"):
        for doc_id, chunk in chunks[name]:
            key = f"{doc_id}|{h(chunk)}|{len(chunk)}"
            if key in cache:
                continue
            doc = by_art[doc_id]["title"] + "\n" + " ".join(" ".join(by_art[doc_id]["sentences"]).split()[: args.doc_words])
            out = llm.create_chat_completion(
                messages=[{"role": "user", "content": PROMPT.format(doc=doc, chunk=chunk[:500])}],
                max_tokens=args.max_tokens, temperature=0.0)
            ctx = out["choices"][0]["message"]["content"].strip().replace("\n", " ")
            cache[key] = ctx
            with cpath.open("a") as f:
                f.write(json.dumps({"key": key, "context": ctx}, ensure_ascii=False) + "\n")
            done += 1
            if args.smoke:
                print("-", ctx, flush=True)
                if done >= args.smoke:
                    return
            if done % 25 == 0:
                print(f"generated {done} in {time.time() - t0:.0f}s ({(time.time() - t0) / done:.1f}s each)", flush=True)

    gold = [a["id"] for a in arts]
    enc.max_length = 256
    queries = {"headline": [a["title"] for a in arts], "heldout_sentence": [a["heldout"] for a in arts]}
    qvecs = {qn: enc.encode(qs, batch_size=64) for qn, qs in queries.items()}
    results = {"articles": len(arts), "model": Path(args.model).name, "results": {}, "chunks": {}}

    for scheme in ("fixed", "variable"):
        docs = np.array([d for d, _ in chunks[scheme]])
        for use_ctx in (False, True):
            texts = []
            for doc_id, c in chunks[scheme]:
                key = f"{doc_id}|{h(c)}|{len(c)}"
                texts.append((cache[key] + " " + c) if use_ctx else c)
            cond = f"{scheme}{'+context' if use_ctx else ''}"
            results["chunks"][cond] = len(texts)
            sparse, dense = SparseChunkIndex(texts), None
            cvecs = enc.encode(texts, batch_size=32)
            for qn, qs in queries.items():
                rs, rd, rh = [], [], []
                for qi, q in enumerate(qs):
                    s_rank = rank_articles(sparse.scores(q), docs, len(arts))
                    d_rank = rank_articles(cvecs @ qvecs[qn][qi], docs, len(arts))
                    rs.append(s_rank); rd.append(d_rank); rh.append(rrf([s_rank, d_rank]))
                results["results"].setdefault(qn, {})[cond] = {
                    "sparse": recall_report(rs, gold), "dense": recall_report(rd, gold),
                    "hybrid_rrf": recall_report(rh, gold)}
                print(cond, qn, results["results"][qn][cond], flush=True)
    save(args.out, results)


if __name__ == "__main__":
    main()
