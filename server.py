"""
Assignment 3: MCP server. Your editor and agent.py start it; by hand: `uv run server.py`.

- search(query): returns the most relevant chunks with source, location and pages.
- add_document(path): runs the file through index.py's pipeline (chunk_file,
  embed_and_store), embeds what has no saved vector and stores it.

Already-indexed documents (decision): a document is identified by its file name
(its "source"). If the same file is added again with identical content, nothing
changes. If its content changed, its old chunks are replaced by the new ones and
only new or changed chunks are embedded.

Loads index/ at startup; never re-embeds the corpus. Prints nothing to stdout
(it carries the protocol); logs go to stderr.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

import index as ix
import retrieval

ROOT = Path(__file__).resolve().parent
TOP_K = 5
SOURCES = ", ".join(sorted(ix.DOCS))

mcp = MCPServer("corpus", log_level="WARNING")


def log(*args) -> None:
    print(*args, file=sys.stderr, flush=True)


@mcp.tool()
def search(query: str, source: str = "") -> str:
    """Search EU law for HR and people-analytics questions: the AI Act (definitions,
    prohibited practices, high-risk AI systems incl. Annex III employment uses,
    provider/deployer obligations, transparency, penalties), the GDPR (all articles),
    and the Pay Transparency Directive (all articles). Recitals, national (e.g.
    Spanish) law and the AI Act chapters on general-purpose AI models, sandboxes,
    governance and final provisions are NOT included.

    Use it before answering any question about these laws. For a question with
    several parts (e.g. AI Act and GDPR), call it once per part.

    Args:
        query: the question or topic, in plain words.
        source: optional filter, one of: ai_act, gdpr, pay_transparency
            (or a document added later); leave empty to search everything.

    Returns the 5 most relevant passages, each with a citation
    [source, Article/Annex, page] and a distance (lower = closer match).
    """
    source = source.strip() or None
    try:
        hits = retrieval.search(query, k=TOP_K, source=source)
    except Exception as e:  # noqa: BLE001 - report the problem to the model
        return f"Search failed: {e}"
    if not hits:
        return f"No passages found{' in ' + source if source else ''}."
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"Result {i} {h['citation']} (distance {h['distance']})\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


@mcp.tool()
def add_document(path: str) -> str:
    """Add a document (PDF or .txt) to the searchable index.

    The file is extracted, cleaned and chunked by article (or by page if it has
    no articles), embedded and stored, so `search` can find it right away. The
    file name without extension becomes its source name in citations.
    If the same file was already added with the same content, nothing changes;
    if its content changed, the old version is replaced.

    Args:
        path: path to the file, absolute or relative to the project folder.

    Returns a short report: source name, number of chunks, how many were newly
    embedded, or why the document could not be added.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        return f"Not added: no file at {p}."
    if p.suffix.lower() not in {".pdf", ".txt"}:
        return f"Not added: {p.name} is not a PDF or .txt file."

    source = p.stem
    try:
        chunks = ix.chunk_file(p)
    except Exception as e:  # noqa: BLE001
        return f"Not added: could not read {p.name} ({e})."
    if not chunks:
        return f"Not added: no text could be extracted from {p.name}."

    col = ix.get_collection()
    existing = col.get(where={"source": source}, include=["metadatas"])
    old = {cid: md.get("hash") for cid, md in zip(existing["ids"], existing["metadatas"])}
    new = {c.id: c.metadata["hash"] for c in chunks}

    if old == new:
        return f"Already indexed: '{source}' ({len(chunks)} chunks) is unchanged. Nothing was done."

    stale = [cid for cid in old if cid not in new]
    if stale:
        col.delete(ids=stale)
    try:
        embedded = ix.embed_and_store(chunks)
    except ix.QuotaExhausted:
        return ("Not added: the daily embedding limit was reached. Chunks embedded so far are "
                "saved; try again after the limit resets.")
    action = "Replaced" if old else "Added"
    log(f"add_document: {action} {source}, {len(chunks)} chunks, {embedded} embedded")
    return (f"{action} '{source}': {len(chunks)} chunks stored, {embedded} newly embedded"
            f"{f', {len(stale)} old chunks removed' if stale else ''}. "
            f"It can now be searched with source='{source}'.")


if __name__ == "__main__":
    mcp.run()  # stdio