"""Fixed-size vs variable (semantic) chunking for claim -> article retrieval.

fixed     consecutive windows of about --fixed-words words (cut at sentence ends once the
          window is full), no regard for topic
variable  semantic chunking: embed every sentence with multilingual-e5-small, start a new
          chunk where the cosine similarity between neighbouring sentences falls into the
          lowest --pct percent of that article's similarities (topic shift), with a minimum
          and a maximum chunk size in words. Chunk sizes therefore vary from article to
          article and inside an article.

Both are scored with sparse TF-IDF, dense e5 and their RRF fusion.

    python eval_chunking.py --articles 2000 --out chunking_results.json
"""
import argparse
import time

import numpy as np

from rag_common import (DenseIndex, SparseChunkIndex, encoder, load_articles, rank_articles,
                        recall_report, rrf, save)


def fixed_chunks(sentences, target_words):
    chunks, cur, n = [], [], 0
    for s in sentences:
        cur.append(s)
        n += len(s.split())
        if n >= target_words:
            chunks.append(" ".join(cur))
            cur, n = [], 0
    if cur:
        if chunks and n < target_words // 3:
            chunks[-1] += " " + " ".join(cur)
        else:
            chunks.append(" ".join(cur))
    return chunks


def semantic_chunks(sentences, sent_vecs, pct, min_words, max_words):
    if len(sentences) < 2:
        return [" ".join(sentences)]
    sims = np.sum(sent_vecs[:-1] * sent_vecs[1:], axis=1)          # similarity of neighbours
    cut = np.percentile(sims, pct)
    chunks, cur, n = [], [], 0
    for i, s in enumerate(sentences):
        cur.append(s)
        n += len(s.split())
        boundary = i < len(sims) and sims[i] <= cut
        if (boundary and n >= min_words) or n >= max_words:
            chunks.append(" ".join(cur))
            cur, n = [], 0
    if cur:
        if chunks and n < min_words:
            chunks[-1] += " " + " ".join(cur)
        else:
            chunks.append(" ".join(cur))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", type=int, default=2000)
    ap.add_argument("--fixed-words", type=int, default=100)
    ap.add_argument("--pct", type=float, default=30.0)
    ap.add_argument("--min-words", type=int, default=50)
    ap.add_argument("--max-words", type=int, default=180)
    ap.add_argument("--out", default="chunking_results.json")
    args = ap.parse_args()

    arts = load_articles(args.articles)
    print("articles:", len(arts), flush=True)
    enc = encoder(threads=2, max_length=64)

    t0 = time.time()
    all_sents = [s for a in arts for s in a["sentences"]]
    vecs = enc.encode(all_sents, batch_size=128)
    print(f"encoded {len(all_sents)} sentences in {time.time() - t0:.0f}s", flush=True)

    schemes = {"fixed": [], "variable": []}
    doc_of = {"fixed": [], "variable": []}
    pos = 0
    for a in arts:
        k = len(a["sentences"])
        sv = vecs[pos:pos + k]
        pos += k
        for name, chunks in (
            ("fixed", fixed_chunks(a["sentences"], args.fixed_words)),
            ("variable", semantic_chunks(a["sentences"], sv, args.pct, args.min_words, args.max_words)),
        ):
            schemes[name] += chunks
            doc_of[name] += [a["id"]] * len(chunks)

    enc.max_length = 256
    queries = {"headline": [a["title"] for a in arts], "heldout_sentence": [a["heldout"] for a in arts]}
    gold = [a["id"] for a in arts]
    qvecs = {qn: enc.encode(qs, batch_size=64) for qn, qs in queries.items()}
    results = {"articles": len(arts), "settings": vars(args), "schemes": {}, "results": {}}

    for name in schemes:
        chunks, docs = schemes[name], np.array(doc_of[name])
        words = np.array([len(c.split()) for c in chunks])
        results["schemes"][name] = {
            "chunks": len(chunks), "avg_words": round(float(words.mean()), 1),
            "min_words": int(words.min()), "max_words": int(words.max()), "std_words": round(float(words.std()), 1),
        }
        print(name, results["schemes"][name], flush=True)
        t0 = time.time()
        cvecs = enc.encode(chunks, batch_size=32)
        print(f"  encoded {len(chunks)} chunks in {time.time() - t0:.0f}s", flush=True)
        sparse, dense = SparseChunkIndex(chunks), DenseIndex(cvecs)
        for qn, qs in queries.items():
            rs, rd, rh = [], [], []
            for qi, q in enumerate(qs):
                s_rank = rank_articles(sparse.scores(q), docs, len(arts))
                d_scores = cvecs @ qvecs[qn][qi]
                d_rank = rank_articles(d_scores, docs, len(arts))
                rs.append(s_rank)
                rd.append(d_rank)
                rh.append(rrf([s_rank, d_rank]))
            results["results"].setdefault(qn, {})[name] = {
                "sparse": recall_report(rs, gold),
                "dense": recall_report(rd, gold),
                "hybrid_rrf": recall_report(rh, gold),
            }
            print(f"  {qn}: {results['results'][qn][name]}", flush=True)
    save(args.out, results)


if __name__ == "__main__":
    main()
