# Hybrid retrieval (experimental branch)

Status: **experimental. Not deployed. The live /match_claim endpoint is unchanged and still uses TF-IDF only.**

## What this branch adds

- hybrid_retrieval.py: a multilingual embedding of every headline (multilingual-e5-small, int8 ONNX, run with onnxruntime), a FAISS flat index over those vectors, and Reciprocal Rank Fusion with the existing TF-IDF ranking.
- eval_hybrid.py: a retrieval test that compares three methods, the current TF-IDF score, dense only, and hybrid.
- requirements-hybrid.txt: what the two files need.

## How it was tested

The team archive is private, so the first test used a public Bangla news corpus (Prothom Alo articles, first parquet shard of the Hugging Face dataset zabir-nabil/bangla_newspaper_dataset). 20,000 unique headlines were indexed. For 1,000 articles a sentence of the body was used as the query and the test checks whether the article's own headline is found.

- lead: the first sentence of the article (usually shares many words with the headline)
- mid: a later sentence (needs meaning, not only shared words)

| Query | Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---:|---:|---:|---:|
| lead | TF-IDF (current) | 36.8% | 51.3% | 55.5% | 0.428 |
| lead | dense only | 28.7% | 40.0% | 44.0% | 0.336 |
| lead | hybrid (RRF) | 39.8% | 52.3% | 58.3% | 0.454 |
| mid | TF-IDF (current) | 6.3% | 11.6% | 15.1% | 0.086 |
| mid | dense only | 2.8% | 5.8% | 8.3% | 0.042 |
| mid | hybrid (RRF) | 6.1% | 12.2% | 16.1% | 0.089 |

How to read it: hybrid is a little better than the current TF-IDF when the query shares words with the headline (Recall@1 up about 3 points, MRR@10 up 0.026) and about the same when it does not. Dense alone is worse than TF-IDF in this test. With 1,000 queries a gap of a few points is small, so this is a hint and not proof. It is also a different corpus from the team archive and a different kind of query from real viral claims.

## Cost

- Query encoding: about 3 to 6 ms per claim on 2 CPU cores. Headline encoding: about 400 per second.
- Vectors: 384 numbers per headline, about 1.5 KB each (30.7 MB for 20,000 headlines, about 154 MB for 100,000).
- Memory: a process that holds the int8 model, onnxruntime, FAISS and sentencepiece measured about 260 MB. The first version used the Hugging Face tokenizers library, which alone needed about 300 MB for this vocabulary, so it was replaced with sentencepiece (the vectors from the two tokenisers have a cosine similarity of about 0.996).
- That is too much to add to the 512 MB Render free instance next to BanglishBERT and the TF-IDF index. Options: a larger instance, or cutting the 250,000-piece vocabulary down to the pieces actually used.

## What is not done

- The "backed" decision still needs the TF-IDF floors and margins. Thresholds for a fused score have not been calibrated.
- The vectors are not built by the nightly job (daily_update.py) and app.py does not load them.
- Not tested on the team archive yet.
- This is not Contextual RAG, semantic chunking or Graph RAG. Headlines are embedded whole.

## Before this could be merged

1. Run it on the team archive: `HF_TOKEN=... python eval_hybrid.py --source archive --days 30` (the token needs read access to commentlens-news-archive).
2. Merge only if hybrid beats TF-IDF there too.
3. Decide how to fit the memory (see Cost).
4. Add the vector build to daily_update.py, load the vectors in app.py, and calibrate the backed rule on real claims.

## Run

```bash
pip install -r requirements-hybrid.txt
python eval_hybrid.py --source public --parquet part0.parquet
```

The first run downloads the e5 model from Hugging Face and quantises it to int8 (about 118 MB). Results are saved to hybrid_eval_results.json.
