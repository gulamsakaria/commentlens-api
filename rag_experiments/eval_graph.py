"""Graph RAG (entity knowledge graph) for claim -> article retrieval.

Pipeline
  1. NER: sagorsarker/mbert-bengali-ner tags PER / ORG / LOC in the headline and the first
     --ner-sentences sentences of every article.
  2. Graph (networkx): nodes are entities; an edge joins two entities that appear in the same
     sentence (weight = number of such sentences); every article is linked to the entities it
     mentions (an article node per article).
  3. Retrieval: find the query's entities with the same NER, take them as seeds, spread
     activation to their graph neighbours (one hop, weight-damped) and score each article by
     the activation of the entities it mentions, weighted by entity rarity (idf).
  4. Fusion: Reciprocal Rank Fusion of the graph ranking with a whole-article TF-IDF ranking.

Compared: TF-IDF alone, graph alone, TF-IDF + graph (RRF). Same articles and queries as
eval_chunking.py (headline, or one sentence held out of the indexed text).

    python eval_graph.py --articles 1000 --out graph_results.json
"""
import argparse
import collections
import math
import time

import networkx as nx
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from rag_common import SparseChunkIndex, load_articles, recall_report, rrf, save

NER_MODEL = "sagorsarker/mbert-bengali-ner"


class Ner:
    def __init__(self, threads=2):
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(NER_MODEL)
        self.model = AutoModelForTokenClassification.from_pretrained(NER_MODEL).eval()
        # the checkpoint config only has LABEL_0..6; this is the mapping from its model card (wikiann)
        self.id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC"}

    SUFFIXES = sorted(["দের", "েরা", "ের", "কে", "ায়", "তে", "রা", "য়", "র", "ে", "টি", "টা"], key=len, reverse=True)

    @classmethod
    def stem(cls, word):
        word = word.strip("।,;:!?\"'()[]-")
        for suf in cls.SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= 3:
                return word[: -len(suf)]
        return word

    @torch.no_grad()
    def entities(self, texts, batch_size=32):
        """Word-level tagging: the label of a word is the label of its first sub-token."""
        out = []
        for start in range(0, len(texts), batch_size):
            batch = [t.split() for t in texts[start:start + batch_size]]
            enc = self.tok(batch, is_split_into_words=True, return_tensors="pt", padding=True,
                           truncation=True, max_length=128)
            pred = self.model(**{k: v for k, v in enc.items()}).logits.argmax(-1)
            for i, words in enumerate(batch):
                labels = {}
                for j, w in enumerate(enc.word_ids(i)):
                    if w is not None and w not in labels:
                        labels[w] = self.id2label[int(pred[i, j])]
                ents, cur = [], []
                for w, word in enumerate(words):
                    lab = labels.get(w, "O")
                    if lab.startswith("B-") or (lab.startswith("I-") and not cur):
                        if cur:
                            ents.append(cur)
                        cur = [word]
                    elif lab.startswith("I-") and cur:
                        cur.append(word)
                    else:
                        if cur:
                            ents.append(cur)
                        cur = []
                if cur:
                    ents.append(cur)
                names = {" ".join(self.stem(x) for x in e) for e in ents}
                out.append(sorted(n for n in names if len(n) >= 3))
        return out


def build_graph(articles, sent_entities, title_entities):
    g = nx.Graph()
    mentions = collections.defaultdict(set)            # entity -> article ids
    for a in articles:
        ents_in_article = set(title_entities[a["id"]])
        for ents in sent_entities[a["id"]]:
            ents_in_article.update(ents)
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    if g.has_edge(ents[i], ents[j]):
                        g[ents[i]][ents[j]]["weight"] += 1
                    else:
                        g.add_edge(ents[i], ents[j], weight=1)
        for e in ents_in_article:
            g.add_node(e)
            mentions[e].add(a["id"])
    return g, mentions


def graph_scores(query_ents, g, mentions, n_docs, damp=0.5):
    n_art = max(n_docs, 1)
    activation = collections.defaultdict(float)
    for e in query_ents:
        if e not in g:
            continue
        activation[e] += 1.0
        deg_w = sum(d["weight"] for d in g[e].values()) or 1.0
        for nb, d in g[e].items():
            activation[nb] += damp * d["weight"] / deg_w
    scores = np.zeros(n_docs)
    for e, act in activation.items():
        idf = math.log(1 + n_art / (1 + len(mentions[e])))
        for doc in mentions[e]:
            scores[doc] += act * idf
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", type=int, default=1000)
    ap.add_argument("--ner-sentences", type=int, default=10)
    ap.add_argument("--out", default="graph_results.json")
    args = ap.parse_args()

    arts = load_articles(2000)[: args.articles]
    ner = Ner()
    t0 = time.time()
    title_entities = dict(zip([a["id"] for a in arts], ner.entities([a["title"] for a in arts])))
    flat, owner = [], []
    for a in arts:
        for s in a["sentences"][: args.ner_sentences]:
            flat.append(s)
            owner.append(a["id"])
    flat_ents = ner.entities(flat)
    sent_entities = collections.defaultdict(list)
    for doc, ents in zip(owner, flat_ents):
        sent_entities[doc].append(ents)
    print(f"NER on {len(flat) + len(arts)} texts in {time.time() - t0:.0f}s", flush=True)

    g, mentions = build_graph(arts, sent_entities, title_entities)
    stats = {"articles": len(arts), "graph_nodes": g.number_of_nodes(), "graph_edges": g.number_of_edges(),
             "avg_entities_per_article": round(float(np.mean([sum(1 for e, m in mentions.items() if a["id"] in m) for a in arts[:200]])), 1)}
    print(stats, flush=True)

    # whole-article TF-IDF baseline (one "chunk" per article)
    sparse = SparseChunkIndex([a["title"] + " " + " ".join(a["sentences"]) for a in arts])
    gold = list(range(len(arts)))
    queries = {"headline": [a["title"] for a in arts], "heldout_sentence": [a["heldout"] for a in arts]}
    results = {"graph": stats, "results": {}}
    for qn, qs in queries.items():
        q_ents = ner.entities(qs)
        rs, rg, rf = [], [], []
        covered = 0
        for q, qe in zip(qs, q_ents):
            s_rank = np.argsort(-sparse.scores(q))
            g_sc = graph_scores(qe, g, mentions, len(arts))
            covered += int(g_sc.max() > 0)
            g_rank = np.argsort(-g_sc, kind="stable")
            rs.append(s_rank)
            rg.append(g_rank)
            rf.append(rrf([s_rank, g_rank]) if g_sc.max() > 0 else s_rank)
        results["results"][qn] = {
            "tfidf": recall_report(rs, gold), "graph_only": recall_report(rg, gold),
            "tfidf+graph_rrf": recall_report(rf, gold),
            "queries_with_a_graph_hit": round(covered / len(qs), 3)}
        print(qn, results["results"][qn], flush=True)
    save(args.out, results)


if __name__ == "__main__":
    main()
