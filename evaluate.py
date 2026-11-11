"""
Assignment 3: evaluation (`main.py evaluate`).

Part 1 - retrieval: hit rate at k = 1, 3, 5 for every design in index.DESIGNS.
    A question is a hit at k when ALL its expected passages (source + article/annex)
    appear in the top-k chunks. For `multi` questions we also report a partial hit
    (at least one of the two passages found). Unanswerable questions have no
    expected passage; we record their top-1 distance to compare with answerable ones.

Part 2 - end-to-end (agent + code checks + LLM judge): added in the next step.

Outputs in eval/: retrieval_results.csv (one row per question and design)
and retrieval_summary.csv.
"""

from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path

import index as ix
import retrieval

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "eval"
QUESTIONS = EVAL_DIR / "questions.csv"
KS = (1, 3, 5)


def log(*args) -> None:
    print(*args, flush=True)


def load_questions() -> list[dict]:
    with QUESTIONS.open(newline="", encoding="utf-8") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def normalise_location(loc: str) -> str:
    """'Article 26(2)' -> 'Article 26'; 'annex iii' -> 'Annex III'."""
    m = re.match(r"\s*(article|annex)\s+([0-9]+[a-z]?|[ivxlc]+)", loc, re.I)
    if not m:
        return loc.strip()
    kind, num = m.group(1).capitalize(), m.group(2)
    return f"{kind} {num.upper() if kind == 'Annex' else num}"


def expected_passages(q: dict) -> list[tuple[str, str]]:
    sources = [s.strip() for s in q["expected_source"].split(";") if s.strip()]
    locations = [normalise_location(x) for x in q["expected_location"].split(";") if x.strip()]
    if len(sources) == 1 and len(locations) > 1:      # e.g. gdpr ; Article 13;Article 14
        sources = sources * len(locations)
    return list(zip(sources, locations))


def chunk_labels(md: dict) -> set[str]:
    """Articles/annexes a chunk covers: one for 'articles', possibly several for 'windows'."""
    if md.get("articles"):
        return {a.strip() for a in md["articles"].split(";")}
    kind = "Article" if md["kind"] == "article" else "Annex"
    return {f"{kind} {md['number']}"}


def covers(md: dict, passage: tuple[str, str]) -> bool:
    source, label = passage
    return md["source"] == source and label in chunk_labels(md)


def evaluate_retrieval(questions: list[dict]) -> list[dict]:
    rows = []
    for design in ix.DESIGNS:
        log(f"\n== retrieval: design '{design}'")
        for q in questions:
            hits = retrieval.search(q["question"], k=max(KS), design=design)
            row = {"design": design, "id": q["id"], "kind": q["kind"],
                   "top1_distance": hits[0]["distance"] if hits else "",
                   "retrieved": " | ".join(h["citation"] for h in hits)}
            passages = expected_passages(q)
            for k in KS:
                top = [h["metadata"] for h in hits[:k]]
                found = [any(covers(md, p) for md in top) for p in passages]
                row[f"hit@{k}"] = int(bool(passages) and all(found)) if passages else ""
                row[f"partial@{k}"] = int(any(found)) if passages else ""
                row[f"words@{k}"] = sum(h["metadata"]["words"] for h in hits[:k])
            rows.append(row)
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    summary = []
    for design in ix.DESIGNS:
        mine = [r for r in rows if r["design"] == design]
        answerable = [r for r in mine if r["kind"] != "unanswerable"]
        multi = [r for r in answerable if r["kind"] == "multi"]
        unans = [r for r in mine if r["kind"] == "unanswerable"]
        s = {"design": design, "questions": len(answerable)}
        for k in KS:
            s[f"hit@{k}"] = round(statistics.mean(r[f"hit@{k}"] for r in answerable), 3) if answerable else ""
            s[f"multi_full@{k}"] = round(statistics.mean(r[f"hit@{k}"] for r in multi), 3) if multi else ""
            s[f"multi_partial@{k}"] = round(statistics.mean(r[f"partial@{k}"] for r in multi), 3) if multi else ""
            s[f"avg_words@{k}"] = round(statistics.mean(r[f"words@{k}"] for r in mine))
        s["top1_dist_answerable"] = round(statistics.mean(r["top1_distance"] for r in answerable), 3) if answerable else ""
        s["top1_dist_unanswerable"] = round(statistics.mean(r["top1_distance"] for r in unans), 3) if unans else ""
        summary.append(s)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def print_summary(summary: list[dict]) -> None:
    log("\nRetrieval summary (answerable questions; multi = both passages found)")
    log(f"{'design':10} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6}   {'multi@5':>7} {'words@5':>8}"
        f"   top-1 distance: answerable / unanswerable")
    for s in summary:
        log(f"{s['design']:10} {s['hit@1']:>6} {s['hit@3']:>6} {s['hit@5']:>6}   "
            f"{s['multi_full@5']:>7} {s['avg_words@5']:>8}   "
            f"{s['top1_dist_answerable']} / {s['top1_dist_unanswerable']}")


def main() -> None:
    questions = load_questions()
    if not questions:
        sys.exit("eval/questions.csv has no questions yet.")
    rows = evaluate_retrieval(questions)
    summary = summarise(rows)
    write_csv(EVAL_DIR / "retrieval_results.csv", rows)
    write_csv(EVAL_DIR / "retrieval_summary.csv", summary)
    print_summary(summary)
    # Part 2 (end-to-end) is added in the next step.


if __name__ == "__main__":
    main()