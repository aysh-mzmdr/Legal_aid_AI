"""Parse each page's footnote area into numbered/starred footnotes, and validate them.

Rules:
- Numbering restarts on every page, so a footnote is identified by (page, key).
- A footnote area that starts with an unnumbered line continues the previous page's last
  footnote.
- Duplicate keys, or a key lower than one already seen on the page (out of order), make the
  footnote ambiguous: it is excluded and logged. Gaps are logged but do not exclude.
- "ibid." is replaced by the enactment cited in the preceding footnote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .abbreviations import Abbreviation, expand
from .pdf_layout import Page
from .text import join_lines

_NUM_START = re.compile(r"^\s*(\d{1,2})\s*\.?\s*(?=\S)")
_STAR_START = re.compile(r"^\s*(\*{1,4})\s*(?=\S)")

_ENACTMENT = re.compile(
    r"\bthe\s+(?:(?:[A-Z(][^\s,;]*|and|of|for|in|the|on|to)\s+){0,16}?"
    r"(?:Act|Order|Ordinance|Regulation)s?,?\s+\d{4}(?:\s*\(\s*\d+\s+of\s+\d{4}\s*\))?"
    r"|\bC\.\s?O\.\s*\d+"
)
_IBID = re.compile(r"\bibid\b\.?", re.IGNORECASE)
_CONTINUED = re.compile(r"\(\s*Foot-?\s*note\s+Continue[ds]?\s*\)\s*,?", re.IGNORECASE)


@dataclass
class Footnote:
    page: int
    key: str
    lines: list[str] = field(default_factory=list)
    raw: str = ""
    text: str = ""  # ibid resolved, abbreviations expanded
    excluded: bool = False
    reason: str = ""
    continued_from_previous_page: bool = False


@dataclass
class FootnoteLog:
    excluded: list[dict] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)
    cross_page: list[dict] = field(default_factory=list)
    ibid_resolved: int = 0
    ibid_unresolved: list[dict] = field(default_factory=list)
    abbreviation_counts: dict[str, int] = field(default_factory=dict)


def parse_footnotes(pages: list[Page], first_page: int, vocab: set[str],
                    abbreviations: list[Abbreviation]) -> tuple[dict[tuple[int, str], Footnote], FootnoteLog]:
    log = FootnoteLog()
    notes: dict[tuple[int, str], Footnote] = {}
    ordered: list[Footnote] = []
    last: Footnote | None = None

    for page in pages[first_page:]:
        lines = page.footnote_lines
        if not lines:
            continue
        base_x = min(l.x0 for l in lines)
        page_notes: list[Footnote] = []
        current: Footnote | None = None
        highest = 0
        for line in lines:
            text = _CONTINUED.sub("", line.text)
            if not text.strip():
                continue
            at_margin = line.x0 <= base_x + 3
            m_num = _NUM_START.match(text) if at_margin else None
            m_star = _STAR_START.match(text) if at_margin else None
            # A wrapped line can start with a number ("60 of List I ..."); a real footnote
            # number is never far ahead of the ones already seen on the page.
            if m_num and int(m_num.group(1)) > highest + 3:
                m_num = None
            if m_num:
                highest = max(highest, int(m_num.group(1)))
            if m_num or m_star:
                m = m_num or m_star
                current = Footnote(page.number, m.group(1))
                current.lines.append(text[m.end():])
                page_notes.append(current)
            elif current is not None:
                current.lines.append(text)
            elif last is not None:
                last.lines.append(text)
                if not last.continued_from_previous_page or log.cross_page[-1]["page"] != page.number:
                    log.cross_page.append({"page": page.number, "continues": f"{last.page}:{last.key}"})
                last.continued_from_previous_page = True
            else:
                log.excluded.append({"page": page.number, "key": "?", "reason": "orphan footnote text"})

        _validate_page(page.number, page_notes, log)
        for fn in page_notes:
            key = (fn.page, fn.key)
            if key in notes:  # duplicate on the same page: both are ambiguous
                notes[key].excluded = True
                notes[key].reason = "duplicate footnote number on page"
                fn.excluded, fn.reason = True, "duplicate footnote number on page"
                continue
            notes[key] = fn
            ordered.append(fn)
        if page_notes:
            last = page_notes[-1]

    for fn in ordered:
        fn.raw = join_lines(fn.lines, vocab)

    _resolve_ibid(ordered, log)
    for fn in ordered:
        fn.text = expand(fn.text, abbreviations, log.abbreviation_counts).strip()
        if fn.excluded:
            log.excluded.append({"page": fn.page, "key": fn.key, "reason": fn.reason, "text": fn.raw[:160]})
    return notes, log


def _validate_page(page_no: int, notes: list[Footnote], log: FootnoteLog) -> None:
    seen: set[str] = set()
    highest = 0
    for fn in notes:
        if fn.key in seen:
            fn.excluded, fn.reason = True, "duplicate footnote number on page"
            continue
        seen.add(fn.key)
        if not fn.key.isdigit():
            continue
        n = int(fn.key)
        if n <= highest:
            fn.excluded, fn.reason = True, f"out of order (after {highest})"
            continue
        if n != highest + 1:
            log.gaps.append({"page": page_no, "expected": highest + 1, "found": n})
        highest = n


def _resolve_ibid(ordered: list[Footnote], log: FootnoteLog) -> None:
    previous_source: str | None = None
    for fn in ordered:
        text = fn.raw
        if _IBID.search(text):
            if previous_source:
                text = re.sub(r"\bby\s+ibid\b\.?", lambda m: "by " + previous_source, text, flags=re.I)
                text = re.sub(r",?\s*\bibid\b\.?", lambda m: " of " + previous_source, text, flags=re.I)
                log.ibid_resolved += 1
            else:
                log.ibid_unresolved.append({"page": fn.page, "key": fn.key, "text": text[:160]})
        fn.text = text
        sources = list(_ENACTMENT.finditer(text))
        if sources:
            previous_source = re.sub(r"\s+", " ", sources[-1].group(0)).strip()
