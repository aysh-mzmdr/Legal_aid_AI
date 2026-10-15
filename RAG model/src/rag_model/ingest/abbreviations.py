"""Expand the abbreviations defined on the PDF's "LIST OF ABBREVIATIONS USED" page.

The list is read from the PDF itself. Each entry is turned into a strict regex so ordinary
words are never touched (e.g. "P. Sambamurthy" is a name, "p. 449" is a page).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .pdf_layout import Page

LIST_TITLE = "LIST OF ABBREVIATIONS USED"

# Abbreviations that are only safe to expand when a number (or clause bracket) follows.
NEEDS_NUMBER = {"Art.", "arts.", "Cl.", "cls.", "P.", "pp.", "Pt.", "Ss.", "ss.", "s.", "C.O."}


@dataclass
class Abbreviation:
    short: str
    long: str
    pattern: re.Pattern


def _entry_pairs(text: str) -> list[tuple[str, str]]:
    """'Art., arts ....for Article, articles.' -> [('Art.', 'Article'), ('arts.', 'articles')]"""
    pairs = []
    joined = re.sub(r"\s*\n\s*(?=\.{3,})", " ", text)
    for raw in joined.splitlines():
        m = re.match(r"^\s*(.+?)\s*\.{4,}\s*(?:for|″)?\s*(.+?)\.?\s*$", raw)
        if not m:
            continue
        shorts = [s.strip() for s in m.group(1).split(",")]
        longs = [s.strip() for s in m.group(2).split(",")]
        if len(shorts) != len(longs):
            longs = longs[:1] * len(shorts)
        for s, l in zip(shorts, longs):
            if not s.endswith("."):
                s += "."
            pairs.append((s, l))
    return pairs


def _compile(short: str, long: str) -> Abbreviation:
    body = re.escape(short)
    follow = r"(?=\s*[\d(])" if short in NEEDS_NUMBER else r"(?=[\s,;:)\]]|$)"
    pattern = re.compile(rf"(?<![A-Za-z.]){body}{follow}")
    return Abbreviation(short, long, pattern)


def load_abbreviations(pages: list[Page]) -> list[Abbreviation]:
    page = next(p for p in pages if any(LIST_TITLE in l.plain for l in p.body))
    text = "\n".join(l.plain for l in page.body if LIST_TITLE not in l.plain)
    pairs = _entry_pairs(text)
    out: dict[str, Abbreviation] = {}
    for short, long in pairs:
        # The list writes "Ss., ss." for "Section, sections"; the text itself uses "s." for
        # one section, so map "s." as well.
        if short == "Ss.":
            out["s."] = _compile("s.", "section")
        out[short] = _compile(short, long)
        # Same abbreviation in the other case, e.g. "cl." / "Cl.", "ins." / "Ins.".
        other = short[0].swapcase() + short[1:]
        if short.count(".") == 1 and other not in out:
            out.setdefault(other, _compile(other, long[0].swapcase() + long[1:]))
    # Longest first so "w.r.e.f." wins over "w.e.f." style overlaps.
    return sorted(out.values(), key=lambda a: -len(a.short))


def expand(text: str, abbreviations: list[Abbreviation], counter: dict[str, int] | None = None) -> str:
    for a in abbreviations:
        def repl(m: re.Match, a: Abbreviation = a) -> str:
            if counter is not None:
                counter[a.short] = counter.get(a.short, 0) + 1
            nxt = m.string[m.end():m.end() + 1]
            return a.long + (" " if nxt and not nxt.isspace() and nxt not in ",;:)]" else "")
        text = a.pattern.sub(repl, text)
    return text
