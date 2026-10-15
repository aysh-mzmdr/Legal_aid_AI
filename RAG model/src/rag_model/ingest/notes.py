"""Put each footnote into the text where its marker is, and remove deleted text.

Placement rules for a marker ⟦page:key⟧:
- followed by "[ ... ]" (amended text): the brackets are removed and the note goes right
  after the closing bracket. If the bracket closes in a later unit (e.g. a whole Part was
  inserted), the note goes where the marker is, at the start of the bracketed text.
- followed by "[ * * * ]" or "* * *" (deleted text): the marker, the asterisks and the note
  are all dropped.
- otherwise: the note goes where the marker is.

A note is dropped (and logged) when its footnote is missing, excluded as ambiguous, or the
marker is out of order on its page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .footnotes import Footnote
from .structure import Unit

TOKEN_RE = re.compile(r"⟦(\d+):([0-9]+|\*{1,4})⟧")
UNIT_SEP = "\x1f"
NOTE_OPEN, NOTE_CLOSE = "\x02", "\x03"
DELETION_RUN = re.compile(r"^\s*\*(?:\s*\*)*")


@dataclass
class NoteLog:
    placed: int = 0
    placed_cross_unit: int = 0
    deletions_removed: int = 0
    stray_brackets_removed: int = 0
    dropped: list[dict] = field(default_factory=list)
    unreferenced: list[dict] = field(default_factory=list)


def _is_deletion(content: str) -> bool:
    """Bracketed content that is asterisks plus at most numbering/punctuation is deleted text.

    "[(f)* * * *]" and "[18.* * *]" are deletions; "[11]" (a seat count) is not.
    """
    rest = TOKEN_RE.sub("", content)
    if "*" not in rest:
        return not rest.strip()
    rest = re.sub(r"\(\s*[A-Za-z0-9]{1,5}\s*\)", "", rest)
    rest = re.sub(r"\b\d+[A-Z]{0,3}\.?", "", rest)
    rest = re.sub(r"[\s.,;:\[\]—–-]", "", rest)
    return set(rest) == {"*"}


def _bracket_matches(text: str) -> dict[int, int]:
    """Match brackets within each unit first; only leftovers may pair across units.

    Matching everything in one pass would let a single unclosed bracket (a PDF typo) steal
    the closing bracket of every amendment after it.
    """
    match: dict[int, int] = {}
    leftovers: list[tuple[int, str]] = []
    start = 0
    for segment in text.split(UNIT_SEP):
        stack: list[int] = []
        for i, ch in enumerate(segment, start):
            if ch == "[":
                stack.append(i)
            elif ch == "]":
                if stack:
                    match[stack.pop()] = i
                else:
                    leftovers.append((i, "]"))
        leftovers.extend((i, "[") for i in stack)
        start += len(segment) + 1
    stack = []
    for i, ch in sorted(leftovers):
        if ch == "[":
            stack.append(i)
        elif stack:
            match[stack.pop()] = i
    return match


def _out_of_order(tokens: list[re.Match]) -> set[tuple[int, str]]:
    """Numeric markers whose first appearance on a page comes after a higher number."""
    bad: set[tuple[int, str]] = set()
    seen: dict[int, set[int]] = {}
    highest: dict[int, int] = {}
    for m in tokens:
        page, key = int(m.group(1)), m.group(2)
        if not key.isdigit():
            continue
        n = int(key)
        if n in seen.setdefault(page, set()):
            continue
        seen[page].add(n)
        if n < highest.get(page, 0):
            bad.add((page, key))
        highest[page] = max(highest.get(page, 0), n)
    return bad


def resolve_notes(units: list[Unit], footnotes: dict[tuple[int, str], Footnote]) -> NoteLog:
    log = NoteLog()
    live = [u for u in units if not u.deleted]
    text = UNIT_SEP.join(u.raw for u in live)
    tokens = list(TOKEN_RE.finditer(text))
    out_of_order = _out_of_order(tokens)
    match = _bracket_matches(text)

    edits: list[tuple[int, int, str]] = []  # (start, end, replacement), non-overlapping
    deleted_ranges: list[tuple[int, int]] = []
    referenced: set[tuple[int, str]] = set()

    def inside_deleted(pos: int) -> bool:
        return any(a <= pos < b for a, b in deleted_ranges)

    for m in tokens:
        if inside_deleted(m.start()):
            continue
        key = (int(m.group(1)), m.group(2))
        referenced.add(key)
        fn = footnotes.get(key)
        j = m.end()
        while j < len(text) and text[j] == " ":
            j += 1

        # Deleted text: "⟦p:k⟧[* * *]" or "⟦p:k⟧(4)* * *" or "⟦p:k⟧***".
        if j < len(text) and text[j] == "[" and j in match and _is_deletion(text[j + 1:match[j]]):
            end = match[j] + 1
            edits.append((m.start(), end, ""))
            deleted_ranges.append((m.start(), end))
            log.deletions_removed += 1
            continue
        tail = re.match(r"(?:\(\s*[A-Za-z0-9]{1,5}\s*\)\s*)?\*(?:\s*\*)*", text[j:])
        if tail and "*" in tail.group(0) and not key[1].startswith("*"):
            end = j + tail.end()
            edits.append((m.start(), end, ""))
            deleted_ranges.append((m.start(), end))
            log.deletions_removed += 1
            continue

        reason = None
        if fn is None:
            reason = "no footnote with this number on the page"
        elif fn.excluded:
            reason = f"footnote excluded: {fn.reason}"
        elif key in out_of_order:
            reason = "marker out of order on its page"
        elif key[1].startswith("*") and text[m.end():m.end() + 1] == "*":
            reason = "star marker run together with another star"
        note = f"{NOTE_OPEN}{fn.text}{NOTE_CLOSE}" if fn and not reason else ""

        opens = j < len(text) and text[j] == "["
        close = match.get(j) if opens else None
        if opens and close is None:
            reason = reason or "amendment bracket never closes"
            note = ""

        if reason:
            log.dropped.append({"page": key[0], "key": key[1], "reason": reason,
                                "context": _plain_context(text, m.start())})

        if opens and close is not None:
            same_unit = UNIT_SEP not in text[j:close]
            edits.append((m.start(), j + 1, "" if same_unit or not note else note + " "))
            edits.append((close, close + 1, (" " + note + " ") if same_unit and note else ""))
            if note:
                log.placed += 1
                log.placed_cross_unit += 0 if same_unit else 1
        elif opens:
            edits.append((m.start(), j + 1, ""))
        else:
            edits.append((m.start(), m.end(), (" " + note + " ") if note else ""))
            if note:
                log.placed += 1

    edits = [e for e in edits if not any(a <= e[0] < b and (e[0], e[1]) != (a, b) for a, b in deleted_ranges)]
    edits.sort()
    pieces, last = [], 0
    for start, end, repl in edits:
        if start < last:
            continue
        pieces.append(text[last:start])
        pieces.append(repl)
        last = end
    pieces.append(text[last:])
    resolved = "".join(pieces)

    for u, body in zip(live, resolved.split(UNIT_SEP)):
        if not u.kind.startswith("appendix"):
            # In the Constitution text square brackets only ever mark amendments. Any left
            # now belong to markers whose note was dropped, or to irregular PDF bracketing.
            stray = len(re.findall(r"[\[\]]", body))
            if stray:
                log.stray_brackets_removed += stray
                body = re.sub(r"[\[\]]", "", body)
        # Notes stay wrapped in NOTE_OPEN/NOTE_CLOSE so the chunker can split text and still
        # produce a note-free copy; render_notes() turns them into "[Note: ...]".
        u.text = _cleanup(body)
        u.text_plain = strip_notes(u.text)

    for key, fn in footnotes.items():
        if key not in referenced and not fn.excluded:
            log.unreferenced.append({"page": key[0], "key": key[1], "text": fn.text[:160]})
    return log


def render_notes(text: str) -> str:
    return text.replace(NOTE_OPEN, "[Note: ").replace(NOTE_CLOSE, "]")


def strip_notes(text: str) -> str:
    text = re.sub(f"\\s*{NOTE_OPEN}[^{NOTE_CLOSE}]*{NOTE_CLOSE}", "", text)
    return "\n".join(re.sub(r"\s+([,;:.])", r"\1", l).strip() for l in text.split("\n") if l.strip())


def _plain_context(text: str, pos: int) -> str:
    ctx = TOKEN_RE.sub("", text[max(0, pos - 60): pos + 60]).replace(UNIT_SEP, " | ")
    return re.sub(r"\s+", " ", ctx)


def _cleanup(text: str) -> str:
    # Deleted text that had no marker: "* * * *", "(f)* * *".
    text = re.sub(r"(?:\(\s*[A-Za-z0-9]{1,5}\s*\)\s*)?\*(?:\s*\*){2,}\s*\]?", "", text)
    text = re.sub(r"\[\s*\]", "", text)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        line = re.sub(r"\s+([,;:.])", r"\1", line)
        line = re.sub(rf"\s+{NOTE_CLOSE}", NOTE_CLOSE, line)
        line = re.sub(rf"{NOTE_OPEN}\s+", NOTE_OPEN, line)
        if line:
            lines.append(line)
    return "\n".join(lines)
