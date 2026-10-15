"""Read the PDF into visual lines, with footnote markers turned into page-scoped tokens.

Layout facts this module relies on (checked against the 2026 Legislative Department edition):
- Body text is ~9pt Times; footnotes are ~7-8pt and sit below a short horizontal rule.
- Numeric footnote markers are separate spans smaller than 7pt.
- Symbol-font stars (U+F02A) are footnote markers too; Times-font "* * *" marks deleted text.
- Justified lines are sometimes split into several PDF "lines" on the same baseline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf as fitz

MARKER_MAX_SIZE = 7.0
SYMBOL_STAR = ""

# A token stands for "footnote <key> of page <page>", e.g. ⟦41:2⟧ or ⟦42:**⟧.
TOKEN_RE = re.compile(r"⟦(\d+):([0-9]+|\*{1,4})⟧")


def make_token(page: int, key: str) -> str:
    return f"⟦{page}:{key}⟧"


def strip_tokens(text: str) -> str:
    return TOKEN_RE.sub("", text)


@dataclass
class Span:
    text: str
    size: float
    font: str
    bold: bool
    italic: bool
    x0: float
    y0: float
    x1: float
    y1: float
    is_marker: bool = False


@dataclass
class VLine:
    """One visual line: every span that shares a baseline band, left to right."""

    page: int
    spans: list[Span]
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    text: str = ""  # with footnote tokens

    def finish(self) -> "VLine":
        self.spans.sort(key=lambda s: s.x0)
        self.x0 = min(s.x0 for s in self.spans)
        self.x1 = max(s.x1 for s in self.spans)
        content = [s for s in self.spans if not s.is_marker] or self.spans
        self.y0 = min(s.y0 for s in content)
        self.y1 = max(s.y1 for s in content)
        return self

    @property
    def plain(self) -> str:
        return strip_tokens(self.text).strip()

    @property
    def size(self) -> float:
        sizes = [s.size for s in self.spans if not s.is_marker and s.text.strip()]
        return max(sizes) if sizes else 0.0

    def _word_spans(self) -> list[Span]:
        """Spans carrying letters or digits (brackets and stars are often set in a plain font)."""
        return [s for s in self.spans if not s.is_marker and re.search(r"[A-Za-z0-9]", s.text)]

    @property
    def all_bold(self) -> bool:
        spans = self._word_spans()
        return bool(spans) and all(s.bold for s in spans)

    @property
    def all_italic(self) -> bool:
        spans = self._word_spans()
        return bool(spans) and all(s.italic for s in spans)

    def bold_prefix(self) -> str:
        """Leading bold text (markers, brackets and spaces may sit in between)."""
        out = []
        started = False
        for s in self.spans:
            t = s.text
            if s.is_marker:
                continue
            if not t.strip() or (not started and t.strip() in "["):
                out.append(t)
                continue
            if s.bold:
                started = True
                out.append(t)
            else:
                break
        return "".join(out)


@dataclass
class Page:
    number: int
    body: list[VLine] = field(default_factory=list)
    footnote_lines: list[VLine] = field(default_factory=list)
    width: float = 360.0
    words: list[tuple[float, float, float, float, str]] = field(default_factory=list)


def _is_page_number(line: VLine, page_height: float) -> bool:
    return bool(re.fullmatch(r"\d{1,3}", line.plain)) and line.x0 > 120 and line.y0 > page_height - 70


def _footnote_rule_y(page: fitz.Page, raw_lines: list[tuple[float, float, str]]) -> float | None:
    """y of the footnote separator rule, or None.

    Schedule tables also draw full-width rules, so take the lowest rule whose text below is
    footnote-sized (<= 8.3pt).
    """
    rules = sorted(
        (d["rect"].y0 for d in page.get_drawings()
         if d["rect"].height < 3 and 150 < d["rect"].width < 300 and d["rect"].x0 < 50),
        reverse=True,
    )
    for y in rules:
        below = [size for (ly, size, text) in raw_lines if ly >= y - 4 and not text.strip().isdigit()]
        if below and sum(1 for s in below if s <= 8.3) / len(below) >= 0.8:
            return y
    return None


def _merge_visual_lines(page_no: int, lines: list[list[Span]]) -> list[VLine]:
    """Merge PDF lines that overlap vertically into visual lines."""
    vlines: list[VLine] = []
    for spans in sorted(lines, key=lambda sp: (min(s.y0 for s in sp), min(s.x0 for s in sp))):
        content = [s for s in spans if not s.is_marker] or spans
        y0 = min(s.y0 for s in content)
        y1 = max(s.y1 for s in content)
        target = None
        for v in vlines[-4:]:
            overlap = min(v.y1, y1) - max(v.y0, y0)
            if overlap > 0.5 * min(v.y1 - v.y0, y1 - y0):
                target = v
                break
        if target:
            target.spans.extend(spans)
            target.finish()
        else:
            vlines.append(VLine(page_no, list(spans)).finish())
    return vlines


def _render(line: VLine) -> str:
    """Concatenate spans, inserting a space where a visible gap separates them."""
    out = ""
    prev: Span | None = None
    for s in line.spans:
        piece = make_token(line.page, s.text) if s.is_marker else s.text
        if prev is not None and out and not out.endswith(" ") and not s.text.startswith(" "):
            gap = s.x0 - prev.x1
            if gap > 1.0 and not s.is_marker and not prev.is_marker:
                out += " "
            elif gap > 1.0 and prev.is_marker and not s.text.startswith(("[", "*")):
                out += " "
        out += piece
        prev = s
    return re.sub(r"[ \t]+", " ", out)


def _span_is_marker(text: str, size: float, font: str, in_footnote_area: bool) -> bool:
    if in_footnote_area:
        return False
    t = text.strip()
    if not t:
        return False
    if "Symbol" in font and set(t) == {SYMBOL_STAR}:
        return True
    return size < MARKER_MAX_SIZE and t.isdigit()


def read_pdf(path: str) -> list[Page]:
    doc = fitz.open(path)
    pages: list[Page] = []
    for pno, pdf_page in enumerate(doc):
        raw = pdf_page.get_text("dict")
        raw_lines: list[list[dict]] = []
        for block in raw["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"]]
                if any(s["text"].strip() for s in spans):
                    raw_lines.append(spans)

        # Size from spans with letters/digits only: a footnote's leading star glyph is 10pt.
        size_index = [
            (min(s["bbox"][1] for s in sp),
             max((s["size"] for s in sp if re.search(r"[A-Za-z0-9]", s["text"])), default=0.0),
             "".join(s["text"] for s in sp))
            for sp in raw_lines
        ]
        rule_y = _footnote_rule_y(pdf_page, size_index)

        body_lines: list[list[Span]] = []
        foot_lines: list[list[Span]] = []
        for sp in raw_lines:
            top = min(s["bbox"][1] for s in sp)
            in_foot = rule_y is not None and top >= rule_y - 4
            spans = []
            for s in sp:
                text = s["text"].replace(SYMBOL_STAR, "*") if in_foot else s["text"]
                marker = _span_is_marker(s["text"], s["size"], s["font"], in_foot)
                if marker:
                    t = s["text"].strip()
                    text = "*" * len(t) if t[0] == SYMBOL_STAR else t
                spans.append(Span(
                    text=text, size=s["size"], font=s["font"],
                    bold=bool(s["flags"] & 16) or "Bold" in s["font"],
                    italic=bool(s["flags"] & 2) or "Italic" in s["font"],
                    x0=s["bbox"][0], y0=s["bbox"][1], x1=s["bbox"][2], y1=s["bbox"][3],
                    is_marker=marker,
                ))
            (foot_lines if in_foot else body_lines).append(spans)

        page = Page(number=pno, width=pdf_page.rect.width,
                    words=[tuple(w[:5]) for w in pdf_page.get_text("words")])
        for v in _merge_visual_lines(pno, body_lines):
            v.text = _render(v)
            if not _is_page_number(v, pdf_page.rect.height) and v.plain:
                page.body.append(v)
        for v in _merge_visual_lines(pno, foot_lines):
            v.text = _render(v)
            if not _is_page_number(v, pdf_page.rect.height) and v.plain:
                page.footnote_lines.append(v)
        pages.append(page)

    _convert_times_stars(pages)
    return pages


_FOOTNOTE_STAR_START = re.compile(r"^\s*(\*{1,4})\s*\S")


def _convert_times_stars(pages: list[Page]) -> None:
    """Times-font '*' glued to a word is a footnote marker when the page has a '*' footnote.

    Spaced runs such as '* * *' (deleted text) are left alone.
    """
    for page in pages:
        keys = set()
        for fl in page.footnote_lines:
            m = _FOOTNOTE_STAR_START.match(fl.text)
            if m and fl.x0 < 60:
                keys.add(m.group(1))
        if not keys:
            continue
        for line in page.body:
            text = line.text
            out, last = [], 0
            for m in re.finditer(r"(?<!\*)\*{1,4}(?!\*)", text):
                prev = text[m.start() - 1] if m.start() > 0 else ""
                nxt = text[m.end()] if m.end() < len(text) else ""
                spaced_run = re.search(r"\*\s+\*", text[max(0, m.start() - 3): m.end() + 3])
                # Glued to a word, or to an amendment bracket/marker as in "⟦1⟧*[239A." —
                # a deleted run is "* * *" or "***", never one or two stars against a bracket.
                glued = prev.isalnum() or nxt.isalnum() or (len(m.group(0)) <= 2 and (nxt == "[" or prev == "⟧"))
                if m.group(0) in keys and glued and not spaced_run:
                    out.append(text[last:m.start()])
                    out.append(make_token(page.number, m.group(0)))
                    last = m.end()
            out.append(text[last:])
            line.text = "".join(out)
