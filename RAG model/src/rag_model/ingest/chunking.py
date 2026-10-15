"""Units -> chunks.

- Preamble, Articles, Schedule paragraphs, forms, Appendix sections: one chunk per unit. A
  long unit is split at its top-level clauses "(1)", "(2)" ...; provisos, Explanations and
  sub-clauses stay with the clause they belong to. A clause that is still too long is split
  at its sub-clauses, then at sentences.
- Schedule list entries and table rows are short, so consecutive ones in the same list or
  section are packed into one chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .notes import render_notes, strip_notes
from .structure import Unit
from .text import nice_title

TARGET_CHARS = 1800
MAX_CHARS = 2600
TOP_CLAUSE = re.compile(r"^\((\d+[A-Z]?)\)")
SUB_CLAUSE = re.compile(r"^\(([a-z]{1,3}|[ivx]+)\)")


@dataclass
class Chunk:
    id: str
    kind: str
    header: str
    text: str
    text_without_notes: str
    source: dict
    provision: dict
    clauses: list[str] | None
    units: list[str]
    pages: list[int]
    chunk_index: int = 0
    chunk_count: int = 1
    refs: list[dict] = field(default_factory=list)
    cited_by: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "header": self.header, "text": self.text,
            "text_without_notes": self.text_without_notes, "source": self.source,
            "provision": self.provision, "clauses": self.clauses, "units": self.units,
            "pdf_pages": self.pages, "chunk_index": self.chunk_index, "chunk_count": self.chunk_count,
            "refs": self.refs, "cited_by": self.cited_by,
        }


def _source(meta: dict) -> dict:
    keep = ("part", "part_title", "chapter", "chapter_title", "sub_heading", "schedule", "schedule_name",
            "schedule_title", "schedule_basis", "section", "appendix", "appendix_title")
    src = {k: meta[k] for k in keep if meta.get(k)}
    for k in ("part_title", "chapter_title", "schedule_title", "appendix_title", "section"):
        if k in src:
            src[k] = nice_title(src[k])
    return src


def _header(u: Unit, label: str) -> str:
    m = u.meta
    trail = []
    if m.get("part"):
        trail.append(f"Part {m['part']}" + (f" — {nice_title(m.get('part_title'))}" if m.get("part_title") else ""))
    if m.get("chapter"):
        trail.append(f"Chapter {m['chapter']}" + (f" — {nice_title(m.get('chapter_title'))}" if m.get("chapter_title") else ""))
    if m.get("sub_heading"):
        trail.append(m["sub_heading"])
    if m.get("schedule"):
        title = nice_title(m.get("schedule_title"))
        trail.append(m["schedule_name"] + (f" — {title}" if title else ""))
    if m.get("appendix"):
        title = nice_title(m.get("appendix_title"))
        trail.append(f"Appendix {'I' * m['appendix']}" + (f" — {title}" if title else ""))
    if m.get("section"):
        trail.append(nice_title(m["section"]))
    if label.startswith(" (part") and trail:
        trail[-1] += label  # an unlabelled text block split in parts: "<section> (part 2 of 4)"
    else:
        trail.append(label)
    return "Constitution of India › " + " › ".join(t for t in trail if t)


def _label(u: Unit) -> str:
    title = nice_title(u.title)
    if u.kind == "preamble":
        return "Preamble"
    if u.kind == "article":
        return f"Article {u.number}" + (f": {title}" if title else "")
    if u.kind == "schedule_paragraph":
        return f"Paragraph {u.number}" + (f": {title}" if title else "")
    if u.kind == "appendix_section":
        word = "Paragraph" if u.meta.get("appendix") == 2 else "Section"
        return f"{word} {u.number}" + (f": {title}" if title else "")
    if u.kind == "schedule_form":
        return title or f"Form {u.number}"
    names = (u.meta.get("appendix_title"), u.meta.get("schedule_title"), u.meta.get("section"))
    if title and (title in {nice_title(n) for n in names if n} or (u.meta.get("section") or "").endswith(title)):
        return ""  # a text block named after its heading: the heading is already in the trail
    return title or ""


def _split_long(paragraph: str) -> list[str]:
    """Split one over-long paragraph at sentence ends.

    A note longer than a chunk (one Sixth Schedule footnote runs over two pages) is split
    too: the piece is closed and the next one reopens as "[Note (continued): ...".
    """
    parts, buf = [], ""
    for piece in re.split(r"(?<=[.;:—])\s+", paragraph):
        if buf and len(buf) + len(piece) + 1 > TARGET_CHARS:
            parts.append(buf)
            buf = piece
        else:
            buf = f"{buf} {piece}".strip()
    if buf:
        parts.append(buf)
    for i in range(len(parts) - 1):
        if parts[i].count("\x02") > parts[i].count("\x03"):
            parts[i] += "\x03"
            parts[i + 1] = "\x02(continued) " + parts[i + 1]
    return parts


def _segments(paragraphs: list[str], pattern: re.Pattern) -> list[tuple[str | None, list[str]]]:
    segs: list[tuple[str | None, list[str]]] = []
    for p in paragraphs:
        m = pattern.match(strip_notes(p))
        if m or not segs:
            segs.append((f"({m.group(1)})" if m else None, [p]))
        else:
            segs[-1][1].append(p)
    return segs


def _pack(segments: list[tuple[str | None, list[str]]]) -> list[tuple[list[str], list[str]]]:
    """Greedily pack (label, paragraphs) segments into groups of about TARGET_CHARS."""
    groups: list[tuple[list[str], list[str]]] = []
    labels: list[str] = []
    paras: list[str] = []
    size = 0
    for label, ps in segments:
        seg_size = sum(len(p) for p in ps)
        if paras and size + seg_size > TARGET_CHARS:
            groups.append((labels, paras))
            labels, paras, size = [], [], 0
        if label:
            labels.append(label)
        paras.extend(ps)
        size += seg_size
    if paras:
        groups.append((labels, paras))
    return groups


def split_unit(u: Unit) -> list[tuple[list[str], str]]:
    """Return [(clause labels, text)] for one unit."""
    paragraphs = [p for p in u.text.split("\n") if p.strip()]
    if len(u.text) <= MAX_CHARS:
        segs = _segments(paragraphs, TOP_CLAUSE)
        labels = [l for l, _ in segs if l]
        return [(labels, "\n".join(paragraphs))]
    out: list[tuple[list[str], str]] = []
    for label, ps in _segments(paragraphs, TOP_CLAUSE):
        if sum(len(p) for p in ps) <= MAX_CHARS:
            pieces = [(label, ps)]
        else:
            # Too long even alone: split at sub-clauses, then sentences, keeping the label.
            subs = _segments(ps, SUB_CLAUSE)
            pieces = []
            for _, sps in subs:
                for p in sps:
                    for part in (_split_long(p) if len(p) > MAX_CHARS else [p]):
                        pieces.append((label, [part]))
        out.extend((label, ps) for label, ps in pieces)
    grouped = _pack(out)
    return [(sorted(set(labels), key=labels.index), "\n".join(ps)) for labels, ps in grouped]


def build_chunks(units: list[Unit]) -> list[Chunk]:
    chunks: list[Chunk] = []
    live = [u for u in units if not u.deleted and u.text.strip()]
    i = 0
    while i < len(live):
        u = live[i]
        if u.packable:
            group = [u]
            size = len(u.text)
            j = i + 1
            while j < len(live) and live[j].packable and live[j].container == u.container \
                    and size + len(live[j].text) <= TARGET_CHARS:
                group.append(live[j])
                size += len(live[j].text)
                j += 1
            chunks.append(_packed_chunk(group))
            i = j
            continue
        pieces = split_unit(u)
        label = _label(u)
        for n, (clauses, text) in enumerate(pieces):
            suffix = ""
            if len(pieces) > 1:
                span = ""
                if clauses:
                    span = f", clause {clauses[0]}" if len(clauses) == 1 else f", clauses {clauses[0]}–{clauses[-1]}"
                suffix = f" (part {n + 1} of {len(pieces)}{span})"
            chunks.append(Chunk(
                id=u.key if len(pieces) == 1 else f"{u.key}-part-{n + 1}",
                kind=u.kind, header=_header(u, label + suffix),
                text=render_notes(text), text_without_notes=strip_notes(text),
                source=_source(u.meta),
                provision={"type": u.kind, "number": u.number, "title": nice_title(u.title)},
                clauses=clauses or None, units=[u.key], pages=sorted(p + 1 for p in u.pages),
                chunk_index=n, chunk_count=len(pieces),
            ))
        i += 1
    return chunks


def _packed_chunk(group: list[Unit]) -> Chunk:
    first, last = group[0], group[-1]
    numbers = [u.number for u in group if u.number]
    kind = "schedule_entries" if first.kind == "schedule_entry" else "appendix_rows"
    if numbers:
        span = numbers[0] if len(set(numbers)) == 1 else f"{numbers[0]}–{numbers[-1]}"
        label = f"{'Entry' if len(set(numbers)) == 1 else 'Entries'} {span}"
    else:
        label = "Rows"
    text = "\n".join(u.text for u in group)
    pages = sorted({p + 1 for u in group for p in u.pages})
    ident = f"{first.container}-{(numbers[0] + '-' + numbers[-1]) if numbers else first.key.rsplit('-', 1)[-1]}"
    return Chunk(
        id=ident, kind=kind, header=_header(first, label), text=render_notes(text),
        text_without_notes=strip_notes(text), source=_source(first.meta),
        provision={"type": kind, "numbers": numbers}, clauses=None,
        units=[u.key for u in group], pages=pages,
    )
