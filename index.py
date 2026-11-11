"""
Assignment 3: build the index (`main.py index`).

Pipeline per document (each step is a function so server.py can import it):

    extract_pages(path)        PDF -> list of raw page texts
    clean_page(text)           drop Official Journal headers/footers and footnotes
    parse_units(source, pages) split into articles and annexes, with chapter/section/pages
    chunk_file(path)           all of the above + scope filter + chunking -> list of chunks

Every chunk carries: source, doc_title, url, kind (article/annex), number, title,
chapter, section, location (e.g. "Article 26(1)-(4)"), pages (e.g. "70-71").

Embedding and storage come in the next step.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# --------------------------------------------------------------------------
# Corpus scope (decision: Option 2, HR-focused selection)
# chapters/annexes = None means "keep all of them".
# --------------------------------------------------------------------------
DOCS = {
    "ai_act": {
        "title": "AI Act (Regulation (EU) 2024/1689)",
        "short": "AI Act",
        "url": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
        "chapters": {"I", "II", "III", "IV", "XII"},
        "annexes": {"III"},
    },
    "gdpr": {
        "title": "General Data Protection Regulation (Regulation (EU) 2016/679)",
        "short": "GDPR",
        "url": "https://eur-lex.europa.eu/eli/reg/2016/679/oj",
        "chapters": None,
        "annexes": None,
    },
    "pay_transparency": {
        "title": "Pay Transparency Directive (Directive (EU) 2023/970)",
        "short": "Pay Transparency Directive",
        "url": "https://eur-lex.europa.eu/eli/dir/2023/970/oj",
        "chapters": None,
        "annexes": None,
    },
}

MAX_WORDS = 300  # chunk size ceiling for article-based chunking


# --------------------------------------------------------------------------
# 1. Extraction
# --------------------------------------------------------------------------
def extract_pages(path: Path) -> list[str]:
    """Raw text of each page (index 0 = page 1)."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return [page.extract_text() or "" for page in PdfReader(path).pages]
    # Plain-text fallback: pages separated by form feeds, else one page.
    return path.read_text(encoding="utf-8").split("\f")


# --------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------
HEADER_FOOTER = [
    # GDPR / Pay Transparency running header, in any order, e.g.
    # "4.5.2016 L 119/32 Official Journal of the European Union EN"
    # "EN Official Journal of the European Union L 132/42 17.5.2023"
    re.compile(r"^(?=.*Official Journal of the European Union)(?=.*\bL\s*\d+/\d+)"
               r"[\w\s./]{0,25}Official Journal of the European Union[\w\s./]{0,25}$"),
    # AI Act: first-page header split over short lines ...
    re.compile(r"^(EN|L series|Official Journal|of the European Union)$"),
    re.compile(r"^\d{4}/\d+\s+\d{1,2}\.\d{1,2}\.\d{4}$"),
    # ... running header "OJ L, 12.7.2024 EN" / "EN OJ L, 12.7.2024" ...
    re.compile(r"^(EN\s+)?OJ L, \d{1,2}\.\d{1,2}\.\d{4}(\s+EN)?$"),
    # ... and footer "ELI: http://... 3/144" / "46/144 ELI: http://..."
    re.compile(r"^(\d+/\d+\s+)?ELI: http\S+(\s+\d+/\d+)?$"),
]
FOOTNOTE_START = re.compile(r"^\[\[FN\d+\]\]|^\(\d{1,3}\)\s+(?![‘'\"])")
OJ_REF = re.compile(r"\bOJ [LC]\b|\(OJ ")


def clean_page(text: str) -> str:
    # Typography: non-breaking hyphens/spaces would break keyword search.
    text = text.replace("‑", "-").replace(" ", " ").replace("\r", "")
    # Footnote markers split over lines, e.g. "(\n1\n)" -> "[[FN1]]".
    text = re.sub(r"\(\s*\n\s*(\d{1,3})\s*\n\s*\)", r"[[FN\1]]", text)

    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if not any(p.match(ln) for p in HEADER_FOOTER)]

    # Footnotes sit at the bottom of the page: cut from the first footnote line
    # that refers to the Official Journal ("(1) OJ C 517, ..." or "(3) Directive ... (OJ L ...)").
    for i, ln in enumerate(lines):
        if FOOTNOTE_START.match(ln) and OJ_REF.search(" ".join(lines[i:i + 3])):
            lines = lines[:i]
            break

    text = "\n".join(ln for ln in lines if ln)
    text = re.sub(r"\s*\[\[FN\d+\]\]", "", text)          # markers left in the body
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)           # words hyphenated across lines
    return text


# --------------------------------------------------------------------------
# 3. Structure: articles and annexes
# --------------------------------------------------------------------------
@dataclass
class Unit:
    source: str
    kind: str                 # "article" or "annex"
    number: str               # "26", "III"
    title: str
    chapter: str = ""         # "III"
    chapter_title: str = ""
    section: str = ""         # "Section 3 – Obligations of providers..."
    start: int = 0            # character offsets in the cleaned full text
    end: int = 0
    text: str = ""


RE_CHAPTER = re.compile(r"^CHAPTER ([IVXLC]+)$", re.M)
RE_SECTION = re.compile(r"^SECTION (\d+)$", re.M | re.I)
RE_ARTICLE = re.compile(r"^Article (\d+[a-z]?)$", re.M)
RE_ANNEX = re.compile(r"^ANNEX ([IVXLC]+)$", re.M)
RE_START = re.compile(r"HAVE ADOPTED THIS (REGULATION|DIRECTIVE)[:]?")
RE_SIGNATURE = re.compile(r"^(Done at|This (Regulation|Directive) shall be binding)", re.M)


def _next_line(text: str, pos: int) -> tuple[str, int]:
    """Line after position pos and the offset just after it."""
    nl = text.find("\n", pos)
    if nl == -1:
        return "", len(text)
    end = text.find("\n", nl + 1)
    end = len(text) if end == -1 else end
    return text[nl + 1:end].strip(), end


def parse_units(source: str, pages: list[str]) -> tuple[list[Unit], str, list[int]]:
    """Clean pages, join them, and split the operative part into units.
    Returns units, full text and page start offsets (for offset -> page)."""
    cleaned = [clean_page(p) for p in pages]
    offsets, parts, pos = [], [], 0
    for c in cleaned:
        offsets.append(pos)
        parts.append(c)
        pos += len(c) + 1
    full = "\n".join(parts)

    m = RE_START.search(full)
    body_start = m.end() if m else 0               # recitals are dropped
    sig = RE_SIGNATURE.search(full, body_start)
    annex_m = RE_ANNEX.search(full, sig.start() if sig else body_start)
    articles_end = sig.start() if sig else (annex_m.start() if annex_m else len(full))

    # Headings in the article part, in order of appearance.
    events = []
    for rx, kind in ((RE_CHAPTER, "chapter"), (RE_SECTION, "section"), (RE_ARTICLE, "article")):
        for h in rx.finditer(full, body_start, articles_end):
            events.append((h.start(), kind, h))
    for h in RE_ANNEX.finditer(full, articles_end):
        events.append((h.start(), "annex", h))
    events.sort(key=lambda e: e[0])

    units: list[Unit] = []
    chapter = chapter_title = section = ""
    for i, (start, kind, h) in enumerate(events):
        title, after_title = _next_line(full, h.start())
        if kind == "chapter":
            chapter, chapter_title, section = h.group(1), title, ""
            continue
        if kind == "section":
            section = f"Section {h.group(1)} – {title}"
            continue
        # article or annex: runs until the next heading of any kind
        stop = articles_end if kind == "article" else len(full)
        for nxt_start, _, _ in events[i + 1:]:
            if nxt_start > start:
                stop = min(stop, nxt_start)
                break
        u = Unit(source=source, kind=kind, number=h.group(1), title=title,
                 chapter=chapter if kind == "article" else "",
                 chapter_title=chapter_title if kind == "article" else "",
                 section=section if kind == "article" else "",
                 start=after_title, end=stop)
        u.text = full[after_title:stop].strip()
        units.append(u)
    return units, full, offsets


def in_scope(u: Unit) -> bool:
    cfg = DOCS.get(u.source, {})
    keep = cfg.get("chapters") if u.kind == "article" else cfg.get("annexes")
    if keep is None:
        return True
    return (u.chapter if u.kind == "article" else u.number) in keep


# --------------------------------------------------------------------------
# 4. Chunking
# --------------------------------------------------------------------------
@dataclass
class Chunk:
    id: str
    text: str                 # what gets embedded and shown (header + body)
    metadata: dict = field(default_factory=dict)


# Paragraph starts inside an article: "1. ", "(1) " (definitions), annex points "1. "
RE_PARA = re.compile(r"^(\d+)\.\s|^\((\d+)\)\s", re.M)


def _pages_for(start: int, end: int, offsets: list[int]) -> str:
    p1 = bisect.bisect_right(offsets, start)
    p2 = bisect.bisect_right(offsets, max(start, end - 1))
    return str(p1) if p1 == p2 else f"{p1}-{p2}"


def _split_long(block: str, max_words: int) -> list[str]:
    """Split one oversized paragraph at point boundaries ((a), (b)...) or sentences."""
    pieces = re.split(r"\n(?=\([a-z]{1,4}\)\s)|(?<=[.;])\s+(?=[A-Z])", block)
    out, cur = [], ""
    for p in pieces:
        if cur and len((cur + " " + p).split()) > max_words:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out


def chunk_unit(u: Unit, offsets: list[int], max_words: int = MAX_WORDS) -> list[Chunk]:
    cfg = DOCS.get(u.source, {"short": u.source, "title": u.source, "url": ""})
    label = f"Article {u.number}" if u.kind == "article" else f"Annex {u.number}"

    # Blocks = numbered paragraphs (with their character offset in the unit).
    starts = [m.start() for m in RE_PARA.finditer(u.text)]
    if not starts or starts[0] != 0:
        starts = [0] + starts
    blocks = []
    for a, b in zip(starts, starts[1:] + [len(u.text)]):
        para = RE_PARA.match(u.text, a)
        num = (para.group(1) or para.group(2)) if para else None
        for piece in _split_long(u.text[a:b].strip(), max_words):
            blocks.append((num, a, b, piece))

    # Greedy packing of consecutive blocks up to max_words.
    groups, cur = [], []
    for blk in blocks:
        if cur and sum(len(x[3].split()) for x in cur) + len(blk[3].split()) > max_words:
            groups.append(cur)
            cur = []
        cur.append(blk)
    if cur:
        groups.append(cur)
    # A short tail (e.g. a one-sentence last paragraph) is merged into the previous chunk.
    if len(groups) > 1 and sum(len(x[3].split()) for x in groups[-1]) < 50:
        groups[-2].extend(groups.pop())

    chunks = []
    for k, g in enumerate(groups, 1):
        nums = [x[0] for x in g if x[0]]
        location = label
        if len(groups) > 1 and nums:
            location += f"({nums[0]})" if nums[0] == nums[-1] else f"({nums[0]})-({nums[-1]})"
        body = "\n".join(x[3] for x in g)
        context = [cfg["short"]]
        if u.chapter:
            context.append(f"Chapter {u.chapter} – {u.chapter_title}")
        if u.section:
            context.append(u.section)
        header = " > ".join(context) + f"\n{label}: {u.title}"
        if len(groups) > 1:
            header += f" (part {k}/{len(groups)})"
        text = f"{header}\n{body}"
        pages = _pages_for(u.start + g[0][1], u.start + g[-1][2], offsets)
        chunk_id = f"{u.source}:{u.kind}-{u.number}:{k}"
        chunks.append(Chunk(
            id=chunk_id,
            text=text,
            metadata={
                "source": u.source,
                "doc_title": cfg["title"],
                "url": cfg["url"],
                "kind": u.kind,
                "number": u.number,
                "title": u.title,
                "chapter": u.chapter,
                "section": u.section,
                "location": location,
                "pages": pages,
                "words": len(body.split()),
                "hash": hashlib.sha1(text.encode()).hexdigest()[:12],
            },
        ))
    return chunks


def chunk_file(path: Path, max_words: int = MAX_WORDS) -> list[Chunk]:
    """Extract, clean, structure, filter to scope and chunk one document.
    The source name is the file stem (data/gdpr.pdf -> "gdpr")."""
    path = Path(path)
    source = path.stem
    units, _, offsets = parse_units(source, extract_pages(path))
    chunks = []
    for u in units:
        if in_scope(u) and u.text:
            chunks.extend(chunk_unit(u, offsets, max_words))
    return chunks


def corpus_files() -> list[Path]:
    return sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in {".pdf", ".txt"})


if __name__ == "__main__":
    import sys
    total = 0
    for f in corpus_files():
        chunks = chunk_file(f)
        words = sum(c.metadata["words"] for c in chunks)
        total += len(chunks)
        print(f"{f.stem:18} chunks={len(chunks):4} words={words:7,}", file=sys.stderr)
    print(f"total chunks: {total}", file=sys.stderr)