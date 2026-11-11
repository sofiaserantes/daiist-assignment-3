"""
Search over the index built by index.py. Used by server.py (search tool) and
evaluate.py (hit rate). Never re-embeds the corpus: only the query is embedded,
and query vectors are cached in index/query_vectors.jsonl so re-running the
evaluation costs no embeddings.

Two retrieval modes:
    "vector"  cosine similarity between Gemini embeddings (ChromaDB)
    "hybrid"  vector ranking + BM25 keyword ranking, merged with
              Reciprocal Rank Fusion (score = sum of 1 / (60 + rank))
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

import index as ix

QUERY_CACHE = ix.INDEX_DIR / "query_vectors.jsonl"
ACTIVE_MODE = "vector"      # set after evaluation
CANDIDATES = 20             # how many results each ranking contributes to the fusion
RRF_K = 60

_query_cache: dict | None = None


# --------------------------------------------------------------------------
# Query embedding (cached)
# --------------------------------------------------------------------------
def embed_query(query: str) -> list[float]:
    global _query_cache
    if _query_cache is None:
        _query_cache = ix.load_vectors(QUERY_CACHE)
    key = ix.vector_key("query:" + query)
    if key not in _query_cache:
        vector = ix.embed_texts([query], "RETRIEVAL_QUERY")[0]
        _query_cache[key] = vector
        ix.INDEX_DIR.mkdir(exist_ok=True)
        with QUERY_CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "q": query, "v": [round(x, 6) for x in vector]}) + "\n")
    return _query_cache[key]


# --------------------------------------------------------------------------
# BM25 keyword search (no embeddings; built from the stored chunk texts)
# --------------------------------------------------------------------------
STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it its of on or our
should that the their them they this to us we what when where which who will with would you your""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, ids, docs, metas, k1: float = 1.5, b: float = 0.75):
        self.ids, self.docs, self.metas = ids, docs, metas
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.k1, self.b = k1, b

    def top(self, query: str, n: int) -> list[int]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [i for i in order[:n] if scores[i] > 0]


_bm25: dict[str, tuple[int, BM25]] = {}


def get_bm25(design: str) -> BM25:
    """Built once per design; rebuilt if the collection grew (add_document)."""
    col = ix.get_collection(design)
    count = col.count()
    if design not in _bm25 or _bm25[design][0] != count:
        data = col.get(include=["documents", "metadatas"])
        _bm25[design] = (count, BM25(data["ids"], data["documents"], data["metadatas"]))
    return _bm25[design][1]


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------
def citation(md: dict) -> str:
    """Short citation, e.g. [gdpr, Article 22, p. 46]."""
    pages = md.get("pages", "")
    p = f", p. {pages}" if pages and "-" not in pages else (f", pp. {pages}" if pages else "")
    return f"[{md['source']}, {md['location']}{p}]"


def _hit(cid, text, md, distance):
    return {"id": cid, "text": text, "metadata": md, "citation": citation(md),
            "distance": round(distance, 4) if distance is not None else None}


def vector_search(query: str, n: int, design: str, source: str | None = None) -> list[dict]:
    res = ix.get_collection(design).query(
        query_embeddings=[embed_query(query)], n_results=n,
        where={"source": source} if source else None,
    )
    return [_hit(cid, t, md, d) for cid, t, md, d in
            zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]


def search(query: str, k: int = 5, design: str = ix.ACTIVE_DESIGN, mode: str = ACTIVE_MODE,
           source: str | None = None) -> list[dict]:
    """Top-k chunks for the query: text, metadata, citation and cosine distance
    (distance is None for a chunk found only by keywords)."""
    if mode == "vector":
        return vector_search(query, k, design, source)

    vec = vector_search(query, CANDIDATES, design, source)
    bm = get_bm25(design)
    kw = [i for i in bm.top(query, CANDIDATES * 3) if not source or bm.metas[i]["source"] == source][:CANDIDATES]

    scores, hits = Counter(), {}
    for rank, h in enumerate(vec):
        scores[h["id"]] += 1 / (RRF_K + rank + 1)
        hits[h["id"]] = h
    for rank, i in enumerate(kw):
        cid = bm.ids[i]
        scores[cid] += 1 / (RRF_K + rank + 1)
        hits.setdefault(cid, _hit(cid, bm.docs[i], bm.metas[i], None))
    return [hits[cid] for cid, _ in scores.most_common(k)]