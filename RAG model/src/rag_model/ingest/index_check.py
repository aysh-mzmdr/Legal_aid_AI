"""Validate the parsed structure and the chunks against the PDF's own Contents pages.

The Contents list every Article (live or "—Omitted"), the Part it belongs to, and every
Schedule. They are used only as a checklist; they are never chunked.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from .chunking import Chunk
from .notes import render_notes
from .pdf_layout import Page
from .structure import ORDINALS, Unit

# Numbers are printed irregularly in the Contents: "243-I.", "371B .".
ENTRY = re.compile(r"^(\[?)\s*(\d+)\s*-?\s*([A-Z]{0,3})\s*\.\s*(.*)$")
PART = re.compile(r"^PART\s+([IVXL]+[A-C]?)\s*$")
SCHEDULE = re.compile(rf"^({'|'.join(ORDINALS)})\s+SCHEDULE")


@dataclass
class IndexEntry:
    number: str
    title: str
    part: str | None
    omitted: bool


@dataclass
class ContentsIndex:
    articles: list[IndexEntry] = field(default_factory=list)
    schedules: list[int] = field(default_factory=list)
    part_titles: dict[str, str] = field(default_factory=dict)


def parse_contents(pages: list[Page], start: int, end: int) -> ContentsIndex:
    idx = ContentsIndex()
    part = None
    current: IndexEntry | None = None
    in_part_title = False
    for page in pages[start:end]:
        for line in page.body:
            plain = line.plain
            m = PART.match(plain)
            if m and line.all_bold:
                part, current, in_part_title = m.group(1), None, True
                idx.part_titles[part] = ""
                continue
            if in_part_title and plain.upper() == plain and line.x0 > 60 and not ENTRY.match(plain):
                idx.part_titles[part] = (idx.part_titles[part] + " " + plain).strip()
                continue
            in_part_title = False
            m = SCHEDULE.match(plain)
            if m:
                n = ORDINALS.index(m.group(1)) + 1
                if n not in idx.schedules:
                    idx.schedules.append(n)
                current = None
                continue
            m = ENTRY.match(plain)
            if m and line.x0 < 60:
                current = IndexEntry(m.group(2) + m.group(3), m.group(4), part, bool(m.group(1)))
                idx.articles.append(current)
                continue
            # Continuation of an entry ("Omitted.]" is italic); centred italic lines are group
            # headings such as "Right to Equality".
            if (current and 60 < line.x0 < 100 and not line.all_bold and plain.upper() != plain
                    and (not current.title.rstrip().endswith(".") or re.match(r"^[⎯—]?\s*Omitted", plain))):
                current.title += " " + plain
    for e in idx.articles:
        # Omitted entries are bracketed: "[2A. Sikkim ...—Omitted.]", "[232. Articles 230, 231
        # and 232 substituted by articles 230 and 231.]".
        bracketed = e.omitted
        e.omitted = bool(re.search(r"Omitted|Repealed|\bRep\.|substituted by", e.title)) and (bracketed or "Omitted" in e.title)
        e.title = re.sub(r"[—⎯]\s*Omitted\.?\]?|\]$", "", e.title)
        e.title = re.sub(r"\s+", " ", e.title).strip(" .[]")
    return idx


def apply_contents_parts(contents: ContentsIndex, units: list[Unit]) -> list[str]:
    """Correct an Article's Part from the Contents when the body's Part heading is missing."""
    by_num = {e.number: e for e in contents.articles}
    changes = []
    for u in units:
        e = by_num.get(u.number) if u.kind == "article" else None
        if not e or not e.part or u.meta.get("part") == e.part:
            continue
        old = u.meta.get("part")
        u.meta["part"] = e.part
        if old != "?":
            u.meta.pop("chapter", None)
            u.meta.pop("chapter_title", None)
            u.meta.pop("sub_heading", None)
            u.meta["part_title"] = contents.part_titles.get(e.part, "")
        elif not u.meta.get("part_title"):
            u.meta["part_title"] = contents.part_titles.get(e.part, "")
        changes.append(f"Article {u.number}: Part {old} -> {e.part} (from Contents)")
    return changes


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def validate(contents: ContentsIndex, units: list[Unit], chunks: list[Chunk], warnings: list[str]) -> dict:
    report: dict = {"structure_warnings": warnings}
    parsed = [u for u in units if u.kind == "article"]
    parsed_by_num = {}
    duplicates = []
    for u in parsed:
        if u.number in parsed_by_num:
            duplicates.append(u.number)
        parsed_by_num[u.number] = u
    index_by_num = {e.number: e for e in contents.articles}

    live_index = {e.number for e in contents.articles if not e.omitted}
    live_parsed = {u.number for u in parsed if not u.deleted}
    report["articles"] = {
        "in_contents": len(contents.articles),
        "in_contents_live": len(live_index),
        "in_contents_omitted": len(contents.articles) - len(live_index),
        "parsed": len(parsed),
        "parsed_live": len(live_parsed),
        "parsed_deleted": len(parsed) - len(live_parsed),
        "missing_from_parse": sorted(live_index - live_parsed, key=_num_key),
        "not_in_contents": sorted(live_parsed - set(index_by_num), key=_num_key),
        "omitted_in_contents_but_parsed_live": sorted(live_parsed & {e.number for e in contents.articles if e.omitted}, key=_num_key),
        "live_in_contents_but_parsed_deleted": sorted(live_index & {u.number for u in parsed if u.deleted}, key=_num_key),
        "duplicate_numbers": duplicates,
    }

    title_mismatch, part_mismatch = [], []
    for num, e in index_by_num.items():
        u = parsed_by_num.get(num)
        if not u or e.omitted or u.deleted:
            continue
        ratio = difflib.SequenceMatcher(None, _norm(e.title), _norm(u.title or "")).ratio()
        if ratio < 0.85:
            title_mismatch.append({"article": num, "contents": e.title, "parsed": u.title, "similarity": round(ratio, 2)})
        if e.part and u.meta.get("part") != e.part:
            part_mismatch.append({"article": num, "contents": e.part, "parsed": u.meta.get("part")})
    report["articles"]["title_mismatches"] = title_mismatch
    report["articles"]["part_mismatches"] = part_mismatch

    order = [u.number for u in parsed if u.number in index_by_num]
    expected = [e.number for e in contents.articles if e.number in parsed_by_num]
    report["articles"]["order_matches_contents"] = order == expected

    found_schedules = sorted({u.meta["schedule"] for u in units if u.meta.get("schedule")})
    report["schedules"] = {"in_contents": contents.schedules, "parsed": found_schedules,
                           "missing": sorted(set(contents.schedules) - set(found_schedules))}
    report["appendices"] = sorted({u.meta["appendix"] for u in units if u.meta.get("appendix")})
    report["chunks"] = _chunk_checks(units, chunks)
    return report


def _chunk_checks(units: list[Unit], chunks: list[Chunk]) -> dict:
    problems: list[dict] = []
    ids = [c.id for c in chunks]
    dup_ids = sorted({i for i in ids if ids.count(i) > 1})
    for c in chunks:
        t = c.text
        if "⟦" in t or "§ROW§" in t or "\x02" in t or "\x03" in t:
            problems.append({"chunk": c.id, "issue": "internal marker left in text"})
        if re.search(r"\*\s*\*\s*\*", t):
            problems.append({"chunk": c.id, "issue": "deletion asterisks left in text"})
        body = re.sub(r"\[Note: [^\x00]*?\](?=\s|$|[,;:.)])", "", t) if not c.source.get("appendix") else ""
        if not c.source.get("appendix") and re.search(r"[\[\]]", body):
            problems.append({"chunk": c.id, "issue": "square bracket outside a note"})
        if not t.strip():
            problems.append({"chunk": c.id, "issue": "empty chunk"})

    # Every live unit's text must be fully covered by its chunks, in order.
    by_unit: dict[str, list[Chunk]] = {}
    for c in chunks:
        for k in c.units:
            by_unit.setdefault(k, []).append(c)
    uncovered = []
    for u in units:
        if u.deleted or not u.text.strip():
            continue
        cs = by_unit.get(u.key)
        if not cs:
            uncovered.append(u.key)
            continue
        if len(cs) > 1 or len(cs[0].units) == 1:
            joined = re.sub(r"\s+", "", "".join(c.text for c in cs))
            joined = joined.replace("][Note:(continued)", "")  # a note split across chunks
            if joined != re.sub(r"\s+", "", render_notes(u.text)):
                uncovered.append(u.key)
    sizes = sorted(len(c.text) for c in chunks)
    return {
        "count": len(chunks),
        "by_kind": _count(c.kind for c in chunks),
        "chars_min": sizes[0] if sizes else 0,
        "chars_median": sizes[len(sizes) // 2] if sizes else 0,
        "chars_max": sizes[-1] if sizes else 0,
        "largest": sorted(((len(c.text), c.id) for c in chunks), reverse=True)[:5],
        "duplicate_ids": dup_ids,
        "units_not_fully_covered": uncovered,
        "problems": problems,
    }


def _count(items) -> dict:
    out: dict = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


def _num_key(n: str) -> tuple:
    m = re.match(r"(\d+)(.*)", n)
    return (int(m.group(1)), m.group(2)) if m else (0, n)
