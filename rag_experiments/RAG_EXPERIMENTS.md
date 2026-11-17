# RAG experiments: variable chunking, contextual retrieval, graph retrieval

Status: **experimental, not deployed.** The production `/match_claim` still uses TF-IDF only
(see `HYBRID_RETRIEVAL.md` for the embedding experiment). Everything here was run offline in a
2-core CPU sandbox, on a public Bangla news corpus (not on the team's own news archive).

Task for all three: "claim -> article". The corpus is a sample of Bangla news articles. A query is
either the article **headline** or **one body sentence that was removed from the indexed text**
(so there is no verbatim match). The retriever must return the right article. Article score = best
chunk score. Metrics: Recall@1/5/10, MRR@10.

## 1. Fixed vs variable (semantic) chunking  (`eval_chunking.py`)
1,987 articles of 150-900 words, queries: 1,987 headlines + 1,987 held-out sentences.
- fixed: about 100-word windows cut at sentence ends. 6,200 chunks, mean 99.7 words, sd 21.7.
- variable: multilingual-e5-small embeds every sentence; a new chunk starts where the similarity
  of neighbouring sentences is in the lowest 30% of that article (topic shift), min 50 / max 180
  words. 7,079 chunks, mean 87.3 words, sd 33.6 (sizes really vary).

Recall@1 (sparse = char+word TF-IDF, dense = e5, hybrid = RRF of both):

| query | chunking | sparse | dense | hybrid |
|---|---|---|---|---|
| headline | fixed | 84.0% | 61.5% | 80.5% |
| headline | variable | 83.3% | 63.5% | 78.8% |
| held-out sentence | fixed | 55.9% | 19.4% | 44.4% |
| held-out sentence | variable | 53.6% | 22.3% | 43.0% |

Result: variable chunking is about equal to fixed. It helps dense search by 2-3 points and hurts
sparse search by 1-2 points. No clear win on this task.

## 2. Contextual retrieval with a local model  (`eval_contextual.py`)
For each chunk, Qwen2.5-1.5B-Instruct (Q4_K_M, llama.cpp, CPU, about 8 s per chunk) writes one
sentence that places the chunk in its article; that sentence is put in front of the chunk before
indexing. Only **40 articles** (300 chunks) because generation is slow, so the corpus is small and
numbers are high; differences of 2-3 queries (about 5 points) are noise. The model answered in
English although asked for Bangla.

Recall@1, 40 queries per type:

| query | condition | sparse | dense | hybrid |
|---|---|---|---|---|
| headline | fixed | 92.5% | 82.5% | 95.0% |
| headline | fixed + context | 92.5% | 92.5% | 97.5% |
| headline | variable | 87.5% | 90.0% | 95.0% |
| headline | variable + context (combo) | 90.0% | 90.0% | 95.0% |
| held-out sentence | fixed | 85.0% | 62.5% | 82.5% |
| held-out sentence | fixed + context | 85.0% | 62.5% | 82.5% |
| held-out sentence | variable | 82.5% | 60.0% | 82.5% |
| held-out sentence | variable + context (combo) | 82.5% | 57.5% | 75.0% |

Result: context lines helped dense search on headline queries (82.5% -> 92.5% on fixed chunks) and
did nothing, or slightly hurt, elsewhere. With 40 articles this is a hint, not proof. The combo
(contextual + variable chunking) was not better than either part alone.

## 3. Entity graph retrieval  (`eval_graph.py`)
NER with `sagorsarker/mbert-bengali-ner` (PER/ORG/LOC) on the headline and the first 10 sentences of
1,000 articles, entities lightly stemmed (common Bangla case suffixes removed). networkx graph:
9,498 entity nodes, 12,283 co-occurrence edges (same sentence), about 13.7 entities per article.
A query's entities are the seeds; activation spreads one hop to neighbours; an article scores by the
idf-weighted activation of the entities it mentions.

| query | TF-IDF (whole article) | graph only | TF-IDF + graph (RRF) |
|---|---|---|---|
| headline, Recall@1 | 99.2% | 80.0% | 96.5% |
| held-out sentence, Recall@1 | 67.8% | 14.4% | 53.6% |

The graph found at least one entity for 89.9% of headlines but only 48.6% of held-out sentences.
Result: graph retrieval alone works for headlines and is weak for sentences; fusing it with equal
weight made results **worse** than TF-IDF alone. We did not tune the fusion weight on these queries.
A first run reported zero entities because of a label-mapping bug; it was fixed and rerun, and only
the rerun is reported.

## Not done
- No run on the team's own news archive, no Bangla-language context generation, no weight tuning.
- Nothing here is wired into the API; none of it changes the numbers of the live system.

Run: `python eval_chunking.py`, `python eval_contextual.py`, `python eval_graph.py`
(needs `hybrid_retrieval.py` from this repo, the e5 int8 model, llama-cpp-python for step 2,
torch + transformers + networkx for step 3).
