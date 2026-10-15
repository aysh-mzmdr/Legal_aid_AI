"""Cross-references between chunks, typed by how the text relies on the target.

Edge types (strongest first), used later by query-time expansion to weight what to follow:
  override     "Notwithstanding anything in article 19"
  exception    "save as otherwise provided in article 32", "except as provided by ..."
  subject_to   "Subject to the provisions of article 13"
  definition   uses a term defined elsewhere ("as defined in", or a defined term like
               "Panchayat" whose definition article covers this Part)
  basis        a Schedule's "[Articles 1 and 4]" line: the Articles it implements
  procedure    "in accordance with", "under", "referred to in", "specified in" ...
  mention      any other reference
  part         "this Part", "Part XIV": a whole Part or Chapter (very broad)
  same_provision  the other chunks of an Article or paragraph that was split by clause

References are read from the text without footnote notes, so amendment history ("Subs. by
... for art. 31") never becomes an edge.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .chunking import Chunk

ORDINAL_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
                 "ninth", "tenth", "eleventh", "twelfth"]
EDGE_RANK = ["override", "exception", "subject_to", "same_provision", "definition", "basis", "procedure", "mention", "part"]

ITEM = r"\d+[A-Z]{0,3}(?:\s*\(\s*[0-9A-Za-z]{1,4}\s*\))*"
ARTICLES = re.compile(
    rf"\b[Aa]rticles?\s+(?P<items>{ITEM}(?:\s*(?:,|and/or|and|or|to)\s*(?:articles?\s+)?{ITEM})*)"
)
CLAUSE_OF = re.compile(
    r"\b(?:sub-clause\s*\(\s*\w+\s*\)\s*of\s+)?clauses?\s*\(\s*(?P<cl>\d+[A-Z]?)\s*\)"
    r"(?:\s*(?:,|and|or)\s*\(\s*\d+[A-Z]?\s*\))*\s*(?:of|in)\s+(?=article\s+\d)"
)
REL_CLAUSE = re.compile(r"\bclauses?\s*\(\s*(?P<cl>\d+[A-Z]?)\s*\)(?!\s*(?:,|and|or)?\s*(?:\(\s*\d+[A-Z]?\s*\)\s*)*(?:of|in)\s+(?:article|the))")
SCHEDULE = re.compile(rf"\b(?P<ord>{'|'.join(o.title() for o in ORDINAL_WORDS)})\s+Schedule\b(?!\s+to\s+the\s+(?!Constitution))")
LIST_REF = re.compile(r"\b(?:entry\s+(?P<entry>\d+[A-Z]?)\s+(?:of|in)\s+)?List\s+(?P<list>I{1,3})\b")
SCHED_PARA = re.compile(rf"\bparagraphs?\s+(?P<para>\d+[A-Z]{{0,2}})(?:\s*(?:,|and|or)\s*\d+[A-Z]{{0,2}})*\s+of\s+the\s+(?P<ord>{'|'.join(o.title() for o in ORDINAL_WORDS)})\s+Schedule")
PART_REF = re.compile(r"\b(?:Chapter\s+(?P<chapter>[IVX]+)\s+of\s+)?Part\s+(?P<part>[IVXL]+[A-C]?)\b(?!\s+of\s+the\s+(?:table|First|said))")
THIS_PART = re.compile(r"\bthis\s+(?P<what>Part|Chapter)\b")

CONTEXT_RULES = [
    ("override", re.compile(r"notwithstanding\s+anything\s+(?:contained\s+)?in\s*(?:the\s+(?:foregoing\s+)?provisions?\s+of\s+)?(?:this\s+\w+\s+or\s+)?$", re.I)),
    # "nothing in article 19 shall restrict ..." limits or suspends the target.
    ("override", re.compile(r"\bnothing\s+in\s+(?:the\s+provisions\s+of\s+)?(?:sub-clause\s*\(\w+\)\s+of\s+)?(?:clauses?\s*\(\w+\)\s+(?:and\s+\(\w+\)\s+)?of\s+)?$", re.I)),
    ("exception", re.compile(r"(?:save\s+as\s+(?:otherwise\s+)?(?:expressly\s+)?provided\s+(?:in|by)|except\s+as\s+(?:otherwise\s+)?provided\s+(?:in|by)|other\s+than|except|excluding|without\s+prejudice\s+to)\s*(?:the\s+provisions\s+of\s+)?$", re.I)),
    ("subject_to", re.compile(r"subject\s+to\s+(?:the\s+(?:other\s+)?provisions\s+of\s+)?(?:this\s+\w+\s+and\s+(?:to\s+)?)?(?:the\s+provisions\s+of\s+)?$", re.I)),
    ("definition", re.compile(r"(?:as\s+defined\s+in|meaning\s+assigned\s+to\s+it\s+(?:in|by)|same\s+meaning\s+as\s+in|meanings?\s+as\s+in)\s*$", re.I)),
    ("procedure", re.compile(r"(?:in\s+accordance\s+with|in\s+the\s+manner\s+provided\s+(?:in|by)|under|referred\s+to\s+in|specified\s+in|mentioned\s+in|by\s+virtue\s+of|for\s+the\s+purposes\s+of|conferred\s+by|provided\s+(?:in|by|for\s+in)|enumerated\s+in|made\s+under|issued\s+under|applies?\s+to|apply\s+in\s+relation\s+to)\s*(?:the\s+provisions\s+of\s+)?$", re.I)),
]


def _edge_type(text: str, start: int) -> str:
    window = text[max(0, start - 120):start]
    window = re.split(r"[.;:]\s", window)[-1]  # stay in the same sentence
    for name, rule in CONTEXT_RULES:
        if rule.search(window):
            return name
    return "mention"


def _expand_items(items: str) -> list[tuple[str, str | None]]:
    """'14, 15 and 19(1)(a)' -> [('14', None), ('15', None), ('19', '1')]; '245 to 255' -> range."""
    out: list[tuple[str, str | None]] = []
    parts = re.split(r"\s*(,|and/or|and|or|to)\s*(?:articles?\s+)?", items)
    tokens = [p for p in parts if p and p.strip()]
    prev: str | None = None
    pending_range = False
    for tok in tokens:
        if tok in (",", "and", "or", "and/or"):
            continue
        if tok == "to":
            pending_range = True
            continue
        m = re.match(r"(\d+)([A-Z]{0,3})\s*(?:\(\s*([0-9A-Za-z]{1,4})\s*\))?", tok)
        if not m:
            continue
        num = m.group(1) + m.group(2)
        # Only numbered clauses are chunk boundaries; "(a)" sub-clauses are not.
        clause = m.group(3) if m.group(3) and re.fullmatch(r"\d+[A-Z]?", m.group(3)) else None
        if pending_range and prev and prev.isdigit() and m.group(1).isdigit() and not m.group(2):
            for n in range(int(prev) + 1, int(m.group(1)) + 1):
                out.append((str(n), None))
        else:
            out.append((num, clause))
        prev = m.group(1) if not m.group(2) else None
        pending_range = False
    return out


class Index:
    """Where each provision lives among the chunks."""

    def __init__(self, chunks: list[Chunk]):
        self.article: dict[str, list[Chunk]] = defaultdict(list)
        self.schedule: dict[int, list[Chunk]] = defaultdict(list)
        self.schedule_para: dict[tuple[int, str], list[Chunk]] = defaultdict(list)
        self.list_chunks: dict[str, list[Chunk]] = defaultdict(list)
        self.part: dict[str, list[Chunk]] = defaultdict(list)
        self.chapter: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
        for c in chunks:
            s = c.source
            if c.kind == "article":
                self.article[c.provision["number"]].append(c)
            if s.get("schedule"):
                self.schedule[s["schedule"]].append(c)
                if c.kind == "schedule_paragraph":
                    self.schedule_para[(s["schedule"], c.provision["number"])].append(c)
                m = re.match(r"List (I{1,3})\b", s.get("section", ""))
                if s["schedule"] == 7 and m:
                    self.list_chunks[m.group(1)].append(c)
            if s.get("part") and c.kind == "article":
                self.part[s["part"]].append(c)
                if s.get("chapter"):
                    self.chapter[(s["part"], s["chapter"])].append(c)

    def article_target(self, num: str, clause: str | None) -> list[Chunk]:
        chunks = self.article.get(num, [])
        if clause and len(chunks) > 1:
            hit = [c for c in chunks if c.clauses and f"({clause})" in c.clauses]
            if hit:
                return hit
        return chunks

    def list_target(self, lst: str, entry: str | None) -> list[Chunk]:
        chunks = self.list_chunks.get(lst, [])
        if entry:
            hit = [c for c in chunks if entry in (c.provision.get("numbers") or [])]
            if hit:
                return hit
        return chunks


def _add(refs: dict, target: Chunk, source: Chunk, etype: str, label: str, clause: str | None = None) -> None:
    if target.id == source.id:
        return
    r = refs.setdefault(target.id, {"target": target.id, "type": etype, "mentions": [], "clauses": []})
    if EDGE_RANK.index(etype) < EDGE_RANK.index(r["type"]):
        r["type"] = etype
    if label not in r["mentions"]:
        r["mentions"].append(label)
    if clause and f"({clause})" not in r["clauses"]:
        r["clauses"].append(f"({clause})")


def extract_references(chunks: list[Chunk]) -> dict:
    index = Index(chunks)
    stats = defaultdict(int)
    by_id = {c.id: c for c in chunks}

    for c in chunks:
        text = c.text_without_notes
        refs: dict[str, dict] = {}
        in_appendix_1 = c.source.get("appendix") == 1
        in_schedule = bool(c.source.get("schedule"))
        own_article = c.provision.get("number") if c.kind == "article" else None

        clause_hits = {m.end(): m.group("cl") for m in CLAUSE_OF.finditer(text)}
        for m in ARTICLES.finditer(text):
            after = text[m.end():m.end() + 40]
            if re.match(r"\s+of\s+the\s+(?!Constitution)", after) or (in_appendix_1 and not re.match(r"\s+of\s+the\s+Constitution", after)):
                continue  # an article of an agreement or another instrument
            etype = _edge_type(text, m.start())
            prefix_clause = clause_hits.get(m.start())
            items = _expand_items(m.group("items"))
            for num, clause in items:
                clause = clause or (prefix_clause if len(items) == 1 else None)
                for t in index.article_target(num, clause):
                    _add(refs, t, c, etype, f"article {num}" + (f"({clause})" if clause else ""), clause)
                    stats[f"article:{etype}"] += 1
                if num not in index.article:
                    stats["unresolved_article"] += 1

        if own_article and len(index.article.get(own_article, [])) > 1:
            for m in REL_CLAUSE.finditer(text):
                for t in index.article_target(own_article, m.group("cl")):
                    if t.id != c.id:
                        _add(refs, t, c, _edge_type(text, m.start()), f"clause ({m.group('cl')})", m.group("cl"))
                        stats["relative_clause"] += 1

        if not in_appendix_1 or re.search(r"Schedule to the Constitution", text):
            for m in SCHED_PARA.finditer(text):
                n = ORDINAL_WORDS.index(m.group("ord").lower()) + 1
                for t in index.schedule_para.get((n, m.group("para")), []):
                    _add(refs, t, c, _edge_type(text, m.start()), m.group(0))
                    stats["schedule_paragraph"] += 1
            for m in SCHEDULE.finditer(text):
                n = ORDINAL_WORDS.index(m.group("ord").lower()) + 1
                if c.source.get("schedule") == n:
                    continue
                targets = index.schedule.get(n, [])
                lm = LIST_REF.search(text, m.start() - 60 if m.start() > 60 else 0, m.end() + 5)
                if n == 7 and lm:
                    continue  # handled by the List rule below, which is more precise
                for t in targets:
                    _add(refs, t, c, _edge_type(text, m.start()), m.group(0))
                stats["schedule"] += 1
            for m in LIST_REF.finditer(text):
                for t in index.list_target(m.group("list"), m.group("entry")):
                    _add(refs, t, c, _edge_type(text, m.start()), m.group(0).strip())
                    stats["list"] += 1

        if not in_schedule and not c.source.get("appendix"):
            for m in PART_REF.finditer(text):
                part = m.group("part")
                targets = index.chapter.get((part, m.group("chapter")), []) if m.group("chapter") else index.part.get(part, [])
                for t in targets:
                    if t.source.get("part") != c.source.get("part") or m.group("chapter"):
                        _add(refs, t, c, "part", m.group(0))
                stats["part"] += 1 if targets else 0
            for m in THIS_PART.finditer(text):
                etype = _edge_type(text, m.start())
                if etype not in ("override", "exception", "subject_to"):
                    continue  # "this Part" is only worth an edge when it limits this provision
                key = c.source.get("part")
                targets = index.part.get(key, []) if m.group("what") == "Part" else index.chapter.get((key, c.source.get("chapter")), [])
                for t in targets:
                    _add(refs, t, c, "part", f"this {m.group('what')}")

        c.refs = sorted(refs.values(), key=lambda r: (EDGE_RANK.index(r["type"]), r["target"]))

    _schedule_basis(chunks, index, stats)
    definition_stats = _definitions(chunks, index)
    stats.update(definition_stats)
    _link_split_parts(chunks, stats)

    # Reverse edges.
    for c in chunks:
        for r in c.refs:
            by_id[r["target"]].cited_by.append({"source": c.id, "type": r["type"], "mentions": r["mentions"][:3]})
    for c in chunks:
        c.cited_by.sort(key=lambda r: (EDGE_RANK.index(r["type"]), r["source"]))
    return dict(stats)


def _link_split_parts(chunks: list[Chunk], stats: dict) -> None:
    """Link the parts of a provision that was split by clause, so one part can bring in the rest."""
    groups: dict[tuple, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if c.chunk_count > 1:
            groups[tuple(c.units)].append(c)
    for parts in groups.values():
        for c in parts:
            refs = {r["target"]: r for r in c.refs}
            for other in parts:
                if other is not c:
                    _add(refs, other, c, "same_provision", f"part {other.chunk_index + 1} of {other.chunk_count}")
                    stats["same_provision"] += 1
            c.refs = sorted(refs.values(), key=lambda r: (EDGE_RANK.index(r["type"]), r["target"]))


def _schedule_basis(chunks: list[Chunk], index: Index, stats: dict) -> None:
    for c in chunks:
        basis = c.source.get("schedule_basis")
        if not basis:
            continue
        refs = {r["target"]: r for r in c.refs}
        for b in basis:
            for t in index.article_target(b["article"], b.get("clause")):
                _add(refs, t, c, "basis", f"article {b['article']}" + (f"({b['clause']})" if b.get("clause") else ""), b.get("clause"))
                stats["basis"] += 1
        c.refs = sorted(refs.values(), key=lambda r: (EDGE_RANK.index(r["type"]), r["target"]))


DEFINITION = re.compile(r"[“\"]([^”\"]{2,60})[”\"]\s*(?:\([^)]*\)\s*)?(?:means|includes|shall\s+mean|shall\s+include|has\s+the\s+meaning)")
SCOPE = re.compile(r"\bIn\s+this\s+(Constitution|Part|Chapter|Schedule)\b")


def _definitions(chunks: list[Chunk], index: Index) -> dict:
    """Link chunks that use a defined term to the chunk that defines it."""
    stats = defaultdict(int)
    definitions: list[tuple[str, Chunk, list[Chunk]]] = []
    for c in chunks:
        text = c.text_without_notes
        if "unless the context otherwise requires" not in text and not (c.kind == "article" and c.provision.get("number") == "366"):
            # Continuation chunks of a definition article inherit its scope.
            first = next((x for x in chunks if x.units == c.units and x.chunk_index == 0), None)
            if not first or first is c or "unless the context otherwise requires" not in first.text_without_notes:
                continue
            scope_text = first.text_without_notes
        else:
            scope_text = text
        sm = SCOPE.search(scope_text)
        if not sm:
            continue
        what = sm.group(1)
        if what == "Constitution":
            scope = [x for x in chunks if not x.source.get("appendix")]
        elif what == "Part":
            scope = index.part.get(c.source.get("part"), [])
        elif what == "Chapter":
            scope = index.chapter.get((c.source.get("part"), c.source.get("chapter")), [])
        else:
            scope = index.schedule.get(c.source.get("schedule"), [])
        for term in {m.group(1).strip() for m in DEFINITION.finditer(text)}:
            definitions.append((term, c, scope))

    for term, target, scope in definitions:
        pattern = re.compile(rf"\b{re.escape(term)}\b" if term[:1].isupper() else rf"\b{re.escape(term)}\b", 0 if term[:1].isupper() else re.I)
        users = [x for x in scope if x.id != target.id and pattern.search(x.text_without_notes)]
        # A Constitution-wide term used almost everywhere ("State", "law") adds nothing.
        if len(scope) > 100 and len(users) > 0.15 * len(scope):
            stats["definition_terms_too_common"] += 1
            continue
        for u in users:
            refs = {r["target"]: r for r in u.refs}
            _add(refs, target, u, "definition", f"“{term}”")
            u.refs = sorted(refs.values(), key=lambda r: (EDGE_RANK.index(r["type"]), r["target"]))
            stats["definition"] += 1
        stats["definition_terms"] += 1
    return stats
