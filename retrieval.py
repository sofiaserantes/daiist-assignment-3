"""
Search over the index built by index.py. Used by server.py (search tool) and
evaluate.py (hit rate). Never re-embeds the corpus: only the query is embedded,
and query vectors are cached in index/query_vectors.jsonl so re-running the
evaluation costs no embeddings.
"""

from __future__ import annotations

import json

import index as ix

QUERY_CACHE = ix.INDEX_DIR / "query_vectors.jsonl"
_query_cache: dict | None = None


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


def citation(md: dict) -> str:
    """Short citation, e.g. [gdpr, Article 22, p. 46]."""
    pages = md.get("pages", "")
    p = f", p. {pages}" if pages and "-" not in pages else (f", pp. {pages}" if pages else "")
    return f"[{md['source']}, {md['location']}{p}]"


def search(query: str, k: int = 5, design: str = ix.ACTIVE_DESIGN, source: str | None = None) -> list[dict]:
    """Top-k chunks for the query: text, metadata, citation and cosine distance."""
    col = ix.get_collection(design)
    res = col.query(
        query_embeddings=[embed_query(query)],
        n_results=k,
        where={"source": source} if source else None,
    )
    hits = []
    for text, md, dist, cid in zip(res["documents"][0], res["metadatas"][0], res["distances"][0], res["ids"][0]):
        hits.append({"id": cid, "text": text, "metadata": md, "citation": citation(md), "distance": round(dist, 4)})
    return hits
