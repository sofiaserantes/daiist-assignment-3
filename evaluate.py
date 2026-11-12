"""
Assignment 3: evaluation (`main.py evaluate`).

Part 1 - retrieval: hit rate at k = 1, 3, 5 for each configuration in CONFIGS
    (chunking design + retrieval mode: articles/vector, windows/vector, articles/hybrid).
    A question is a hit at k when ALL its expected passages (source + article/annex)
    appear in the top-k chunks. For `multi` questions we also report a partial hit
    (at least one of the two passages found). Unanswerable questions have no
    expected passage; we record their top-1 distance to compare with answerable ones.

Part 2 - end-to-end: the agent (agent.py) answers every question.
    Code checks: refusal (exact refusal sentence), citations present, citations
    valid (each one was returned by search), expected passage retrieved/cited.
    LLM judge (JUDGE_MODEL, a different model from the agent) grades the rest
    with the rubric below, using the retrieved text as evidence.
    Agreement with eval/human_labels.csv is reported when it has grades.

Agent answers and judge verdicts are cached (eval/answers.jsonl, eval/judgements.jsonl),
so re-running costs nothing; delete a line to redo that question.

    uv run python evaluate.py --skip-judge   # answers + grading sheet only (grade BEFORE seeing the judge)
    uv run python main.py evaluate           # everything

Outputs in eval/: retrieval_results.csv, retrieval_summary.csv, answers.jsonl,
grading_sheet.csv, judgements.jsonl, e2e_results.csv, e2e_summary.csv, agreement.csv.
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import index as ix
import retrieval

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "eval"
QUESTIONS = EVAL_DIR / "questions.csv"
KS = (1, 3, 5)
# (label, chunking design, retrieval mode)
CONFIGS = [
    ("articles", "articles", "vector"),
    ("windows", "windows", "vector"),
    ("articles+bm25", "articles", "hybrid"),
]


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
    for label, design, mode in CONFIGS:
        log(f"== retrieval: {label}")
        for q in questions:
            hits = retrieval.search(q["question"], k=max(KS), design=design, mode=mode)
            dists = [h["distance"] for h in hits if h["distance"] is not None]
            row = {"design": label, "id": q["id"], "kind": q["kind"],
                   "top1_distance": dists[0] if dists else "",
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
    for design, _, _ in CONFIGS:
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
        def mean_dist(rs):
            d = [r["top1_distance"] for r in rs if r["top1_distance"] != ""]
            return round(statistics.mean(d), 3) if d else ""
        s["top1_dist_answerable"] = mean_dist(answerable)
        s["top1_dist_unanswerable"] = mean_dist(unans)
        summary.append(s)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def print_summary(summary: list[dict]) -> None:
    log("\nRetrieval summary (answerable questions; multi = both passages found)")
    log(f"{'design':14} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6}   {'multi@5':>7} {'words@5':>8}"
        f"   top-1 distance: answerable / unanswerable")
    for s in summary:
        log(f"{s['design']:14} {s['hit@1']:>6} {s['hit@3']:>6} {s['hit@5']:>6}   "
            f"{s['multi_full@5']:>7} {s['avg_words@5']:>8}   "
            f"{s['top1_dist_answerable']} / {s['top1_dist_unanswerable']}")


# ==========================================================================
# Part 2: end-to-end
# ==========================================================================
ANSWERS = EVAL_DIR / "answers.jsonl"
JUDGEMENTS = EVAL_DIR / "judgements.jsonl"
HUMAN = EVAL_DIR / "human_labels.csv"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.1-flash-lite")   # not the agent's model
PAUSE_SECONDS = 4          # between questions, to stay under the free tier's per-minute limit

# Rubric (decisions: correct = main fact right and nothing wrong;
# supported = every claim in the retrieved text; an extra claim from memory
# makes it unsupported, not incorrect).
JUDGE_RUBRIC = """You grade answers from an assistant that answers HR questions about EU law
(AI Act, GDPR, Pay Transparency Directive) using only retrieved passages.

You get: the QUESTION, the REFERENCE answer written by an expert, the RETRIEVED passages
the assistant saw, and the assistant's ANSWER.

correct (1 or 0):
  1 if the answer gives the main fact of the REFERENCE (the core yes/no, date, number,
    obligation or party asked about) and states nothing that contradicts the REFERENCE
    or the law. Missing secondary details is fine.
  0 if the main fact is missing or wrong, if any statement is wrong, or if the assistant
    refuses although the REFERENCE contains an answer.

supported (1 or 0):
  1 if EVERY factual claim in the ANSWER can be found in the RETRIEVED passages
    (paraphrase is fine). Ignore citations' formatting and polite phrases.
  0 if at least one claim is not in the RETRIEVED passages, even if it is true
    (e.g. a date or rule the assistant knows from elsewhere).
  A correct claim from outside the passages makes supported 0 but does NOT change correct.
  Applying a retrieved rule to the user's own situation is not a new claim. A general
  statement about how things usually are (e.g. "HR typically processes special data")
  that is not in the passages IS a claim and makes supported 0.

Give a short reason (one or two sentences) naming the missing fact or unsupported claim.
Work in this order before deciding:
  1. Write the main fact of the REFERENCE in a few words.
  2. List every factual claim in the ANSWER, and for each one say whether it is in the
     RETRIEVED passages (quote a few words of the passage) or not.
  3. Say whether the ANSWER's first sentence / overall conclusion agrees with the REFERENCE.
Be strict: when in doubt about a claim, it is NOT supported.
Reply only with JSON:
{"main_fact": "...", "claims": [{"claim": "...", "in_retrieved": true/false, "evidence": "..."}],
 "conclusion_matches_reference": true/false, "correct": 0 or 1, "supported": 0 or 1, "reason": "..."}"""


def load_jsonl(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row
    return out


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def citation_passage(cite: str) -> tuple[str, str] | None:
    """'gdpr, Article 22(1), p. 46' or '[gdpr, Article 22]' -> ('gdpr', 'Article 22')."""
    m = re.match(r"\[?\s*([a-z][a-z0-9_]*)\s*,\s*((?:Article|Annex|Page)\s+[0-9IVXLC]+[a-z]?)", cite)
    return (m.group(1), normalise_location(m.group(2))) if m else None


def root_error(e: BaseException) -> str:
    """The real message behind asyncio/anyio 'unhandled errors in a TaskGroup' wrappers."""
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    if e.__cause__ is not None and not str(e):
        e = e.__cause__
    return f"{type(e).__name__}: {e}"


class DailyLimit(Exception):
    pass


def run_agent_on_all(questions: list[dict]) -> dict:
    import agent
    cache = load_jsonl(ANSWERS)
    todo = [q for q in questions if q["id"] not in cache or cache[q["id"]]["question"] != q["question"]]
    log(f"\n== end-to-end: agent ({agent.CHAT_MODEL}) on {len(questions)} questions, {len(todo)} to run")
    failures = 0
    for n, q in enumerate(todo, 1):
        log(f"  [{n}/{len(todo)}] {q['id']}: {q['question'][:70]} ...")
        for attempt in range(3):
            try:
                res = agent.answer(q["question"], approve=False)
                break
            except Exception as e:  # noqa: BLE001
                msg = root_error(e)
                if "PerDay" in msg or "per day" in msg.lower():
                    log(f"  {q['id']}: daily limit reached ({msg[:200]})")
                    raise DailyLimit(msg) from e
                if attempt < 2 and any(c in msg for c in ("429", "RESOURCE_EXHAUSTED", "504", "503", "DEADLINE", "TimeoutError")):
                    log(f"  {q['id']}: {msg[:120]} ... waiting 60s")
                    time.sleep(60)
                    continue
                log(f"  {q['id']}: FAILED ({msg[:300]})")
                res = None
                break
        if res is None:
            failures += 1
            if failures >= 3:
                log("  3 failures in a row: stopping. Answers so far are saved; run again later to continue.")
                break
            continue
        failures = 0
        res["id"] = q["id"]
        res["model"] = agent.CHAT_MODEL
        append_jsonl(ANSWERS, res)
        cache[q["id"]] = res
        log(f"  [{n}/{len(todo)}] {q['id']} {res['seconds']}s, {len(res['tool_calls'])} tool calls"
            f"{', REFUSED' if res['refused'] else ''}{', guardrail fired' if res['guardrail_events'] else ''}")
        time.sleep(PAUSE_SECONDS)
    return cache


def code_checks(q: dict, a: dict) -> dict:
    import agent
    expected = expected_passages(q)
    retrieved = {p for p in (citation_passage(c) for c in a["retrieved"]) if p}
    cited = {p for p in (citation_passage(c) for c in a["citations"]) if p}
    return {
        "refused": int(a["refused"]),
        "has_citation": int(bool(cited)),
        "citations_valid": int(not agent.check_citations(a["answer"], [a["retrieved_text"]], False)
                               or a["refused"]),
        "expected_retrieved": int(all(p in retrieved for p in expected)) if expected else "",
        "expected_cited": int(all(p in cited for p in expected)) if expected else "",
        "guardrail_fired": int(bool(a["guardrail_events"])),
        "search_errors": a["retrieved_text"].count("Search failed"),
        "search_calls": sum(1 for t in a["tool_calls"] if t["tool"] == "search"),
    }


def write_grading_sheet(questions: list[dict], answers: dict) -> None:
    """For your own grades: question, reference and answer, WITHOUT the judge's verdict."""
    rows = [{"id": q["id"], "kind": q["kind"], "question": q["question"],
             "reference_answer": q["reference_answer"], "agent_answer": answers[q["id"]]["answer"],
             "retrieved": " | ".join(answers[q["id"]]["retrieved"])}
            for q in questions if q["id"] in answers]
    write_csv(EVAL_DIR / "grading_sheet.csv", rows)
    if not HUMAN.exists() or not HUMAN.read_text(encoding="utf-8").strip():
        with HUMAN.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["id", "correct", "supported", "notes"])
    log(f"  wrote eval/grading_sheet.csv ({len(rows)} answers to grade)")


def judge(q: dict, a: dict, client) -> dict:
    prompt = (f"{JUDGE_RUBRIC}\n\nQUESTION:\n{q['question']}\n\nREFERENCE:\n{q['reference_answer']}\n\n"
              f"RETRIEVED:\n{a['retrieved_text'][:30000] or '(nothing retrieved)'}\n\nANSWER:\n{a['answer']}")
    from google.genai import types
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=JUDGE_MODEL, contents=prompt,
                config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
            )
            v = json.loads(resp.text)
            usage = resp.usage_metadata
            return {"correct": int(v["correct"]), "supported": int(v["supported"]),
                    "reason": str(v.get("reason", "")), "claims": v.get("claims", []),
                    "judge_model": JUDGE_MODEL,
                    "judge_tokens": (usage.prompt_token_count or 0) + (usage.candidates_token_count or 0)}
        except Exception as e:  # noqa: BLE001
            if attempt < 2:
                time.sleep(30 if "429" in str(e) else 5)
                continue
            return {"correct": "", "supported": "", "reason": f"judge failed: {str(e)[:120]}",
                    "judge_model": JUDGE_MODEL, "judge_tokens": 0}


def grade_all(questions: list[dict], answers: dict) -> dict:
    """Code decides refusals; the judge grades every answer that is not a refusal."""
    from google import genai
    from api_key import load_api_key
    client = genai.Client(api_key=load_api_key())
    cache = load_jsonl(JUDGEMENTS)
    for q in questions:
        a = answers.get(q["id"])
        if not a:
            continue
        key_ok = (q["id"] in cache and cache[q["id"]].get("answer") == a["answer"]
                  and cache[q["id"]].get("judge_model") in (JUDGE_MODEL, "code")
                  and cache[q["id"]].get("correct") != "")          # retry failed verdicts
        if key_ok:
            continue
        if a["refused"]:
            # Decided in code: a refusal is right for unanswerable, wrong otherwise.
            v = {"correct": int(q["kind"] == "unanswerable"), "supported": 1,
                 "reason": "refusal (decided in code)", "judge_model": "code", "judge_tokens": 0}
        else:
            v = judge(q, a, client)
            if q["kind"] == "unanswerable":
                v["correct"] = 0           # answering an unanswerable question is never correct
                v["reason"] = "answered an unanswerable question; " + v["reason"]
            time.sleep(PAUSE_SECONDS)
        v.update(id=q["id"], answer=a["answer"])
        append_jsonl(JUDGEMENTS, v)
        cache[q["id"]] = v
        log(f"  judged {q['id']}: correct={v['correct']} supported={v['supported']}")
    return cache


def suggested_cause(q: dict, checks: dict, v: dict) -> str:
    """A first guess at the failure cause, for you to confirm or change in the report."""
    if v["correct"] == 1 and v["supported"] == 1:
        return ""
    if q["kind"] == "unanswerable":
        return "unsupported claim (answered instead of refusing)" if not checks["refused"] else ""
    if checks["refused"]:
        return "wrong refusal" if checks["expected_retrieved"] else "retrieval miss (then refused)"
    if v["correct"] == 0:
        return "wrong answer from right passage" if checks["expected_retrieved"] else "retrieval miss"
    return "unsupported claim"


def build_results(questions, answers, verdicts) -> list[dict]:
    rows = []
    for q in questions:
        a, v = answers.get(q["id"]), verdicts.get(q["id"])
        if not a or not v:
            continue
        checks = code_checks(q, a)
        rows.append({
            "id": q["id"], "kind": q["kind"], **checks,
            "correct": v["correct"], "supported": v["supported"], "judge_reason": v["reason"],
            "suggested_cause": suggested_cause(q, checks, v),
            "input_tokens": a["input_tokens"], "output_tokens": a["output_tokens"],
            "judge_tokens": v.get("judge_tokens", 0), "seconds": a["seconds"],
        })
    return rows


def summarise_e2e(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["kind"] != "unanswerable"]
    una = [r for r in rows if r["kind"] == "unanswerable"]
    graded = [r for r in rows if r["correct"] != ""]
    mean = lambda xs: round(statistics.mean(xs), 3) if xs else ""  # noqa: E731
    return {
        "questions": len(rows),
        "correct": mean([r["correct"] for r in graded]),
        "correct_answerable": mean([r["correct"] for r in ans if r["correct"] != ""]),
        "supported_when_answered": mean([r["supported"] for r in graded if not r["refused"]]),
        "correct_refusals": f"{sum(r['refused'] for r in una)}/{len(una)}",
        "wrong_refusals": f"{sum(r['refused'] for r in ans)}/{len(ans)}",
        "citations_valid": mean([r["citations_valid"] for r in rows]),
        "expected_retrieved": mean([r["expected_retrieved"] for r in ans]),
        "guardrail_fired": sum(r["guardrail_fired"] for r in rows),
        "search_errors": sum(r["search_errors"] for r in rows),
        "avg_input_tokens": round(statistics.mean(r["input_tokens"] for r in rows)),
        "avg_output_tokens": round(statistics.mean(r["output_tokens"] for r in rows)),
        "avg_judge_tokens": round(statistics.mean(r["judge_tokens"] for r in rows)),
        "avg_seconds": round(statistics.mean(r["seconds"] for r in rows), 1),
    }


def agreement(rows: list[dict]) -> None:
    if not HUMAN.exists():
        return
    with HUMAN.open(newline="", encoding="utf-8-sig") as f:   # -sig: Excel adds a BOM
        human = {r["id"].strip(): r for r in csv.DictReader(f) if (r.get("id") or "").strip()}
    judged = {r["id"]: r for r in rows}
    pairs = [(human[i], judged[i]) for i in human if i in judged and human[i].get("correct", "").strip() != ""]
    if not pairs:
        log("\n(no human labels yet: grade eval/grading_sheet.csv into eval/human_labels.csv)")
        return
    out, agree_c, agree_s = [], 0, 0
    for h, j in pairs:
        hc, hs = int(h["correct"]), int(h["supported"])
        same_c, same_s = hc == j["correct"], hs == j["supported"]
        agree_c += same_c
        agree_s += same_s
        out.append({"id": j["id"], "human_correct": hc, "judge_correct": j["correct"],
                    "human_supported": hs, "judge_supported": j["supported"],
                    "agree_correct": int(same_c), "agree_supported": int(same_s),
                    "human_notes": h.get("notes", ""), "judge_reason": j["judge_reason"]})
    write_csv(EVAL_DIR / "agreement.csv", out)
    n = len(pairs)
    log(f"\nHuman vs judge on {n} answers: correct agree {agree_c}/{n} ({agree_c / n:.0%}), "
        f"supported agree {agree_s}/{n} ({agree_s / n:.0%})  -> eval/agreement.csv")


def main() -> None:
    questions = load_questions()
    if not questions:
        sys.exit("eval/questions.csv has no questions yet.")

    # Part 1: retrieval
    rows = evaluate_retrieval(questions)
    summary = summarise(rows)
    write_csv(EVAL_DIR / "retrieval_results.csv", rows)
    write_csv(EVAL_DIR / "retrieval_summary.csv", summary)
    print_summary(summary)

    # Part 2: end-to-end
    try:
        answers = run_agent_on_all(questions)
    except DailyLimit:
        log("\nThe model's daily free limit is reached. Answers so far are saved in eval/answers.jsonl;"
            "\nrun the same command again after the limit resets (09:00 in Spain) to continue.")
        answers = load_jsonl(ANSWERS)
    write_grading_sheet(questions, answers)
    if "--skip-judge" in sys.argv:
        log("\nSkipped the judge. Grade eval/grading_sheet.csv into eval/human_labels.csv, then run without --skip-judge.")
        return
    log(f"\n== judge ({JUDGE_MODEL})")
    verdicts = grade_all(questions, answers)
    results = build_results(questions, answers, verdicts)
    if not results:
        sys.exit("No graded answers.")
    write_csv(EVAL_DIR / "e2e_results.csv", results)
    s = summarise_e2e(results)
    write_csv(EVAL_DIR / "e2e_summary.csv", [s])
    log("\nEnd-to-end summary")
    for k, v in s.items():
        log(f"  {k:24} {v}")
    agreement(results)


if __name__ == "__main__":
    main()