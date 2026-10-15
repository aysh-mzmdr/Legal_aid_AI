"""Turn the body's visual lines into structural units.

A unit is the smallest thing the document numbers: the Preamble, an Article, a Schedule
paragraph / list entry / form / table row, or an Appendix section. Units keep the raw text
(with footnote tokens); footnotes are resolved later across the whole document because an
amendment bracket can open in one unit and close in another.

Each Schedule is parsed with its own rule (see SCHEDULE_RULES), because their layouts differ:
tables, numbered lists, titled paragraphs, oath forms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .pdf_layout import Page, Span, VLine, _render, strip_tokens
from .text import join_lines, nice_title

TOKEN = r"⟦\d+:[0-9*]+⟧"
LEAD = rf"(?:\s|{TOKEN}|\[|(?<=⟧)-(?=⟦))*"
BASIS_LINE = re.compile(r"^[\[(]?\s*(?:See\s+)?(?:Articles?|article)\s+\d")
# "371-I." is how the PDF prints Article 371I; the hyphen is dropped from the number.
NUM_HEAD = re.compile(rf"^(?P<lead>{LEAD})(?P<num>\d+(?:-?[A-Z]{{1,3}})?)\s?\.(?!\d)\s*(?P<rest>.*)$", re.S)
PARA_START = re.compile(
    rf"^{LEAD}(?:\((?:\d+[A-Z]?|[a-z]{{1,4}}|[ivxlc]+)\)|Provided\b|Explanation\b|Illustration\b)"
)
ORDINALS = ["FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH", "SEVENTH", "EIGHTH",
            "NINTH", "TENTH", "ELEVENTH", "TWELFTH"]
SCHEDULE_HEAD = re.compile(rf"^\[?\s*({'|'.join(ORDINALS)})\s+SCHEDULE\s*$")
PART_HEAD = re.compile(r"^\[?\s*PART\s+([IVXL]+[A-Z]?|[A-E])\s*$")
CHAPTER_HEAD = re.compile(r"^\[?\s*CHAPTER\s+([IVX]+[A-Z]?)\s*\.?\s*[—–-]\s*(.*)$")
LIST_HEAD = re.compile(r"^\[?\s*List\s+(I{1,3})\s*[—–-]\s*(.+)$")
ROMAN_LINE = re.compile(r"^\[?\s*([IVX]{1,4})\s*$")
APPENDIX_HEAD = re.compile(r"^\[?\s*APPENDIX\s+([IVX]+)\s*$")
APPENDIX_II_TITLE = "THE CONSTITUTION (APPLICATION TO JAMMU AND KASHMIR)"
RUNNING_HEADER = "THE CONSTITUTION OF INDIA"
DOT_LEADER = re.compile(r"\s*\.{4,}\s*")

# How each Schedule is laid out.
#   two_col  : table of Name | Territories (First)
#   numbered : numbered paragraphs or list entries, no bold titles
#   titled   : numbered paragraphs with bold titles; untitled numbers stay inside a paragraph
#   forms    : oath forms headed by Roman numerals (Third)
#   rows     : one-line table rows with dot leaders (Fourth)
SCHEDULE_RULES = {
    1: {"mode": "two_col", "packable": True},
    2: {"mode": "numbered", "parts": True, "packable": False},
    3: {"mode": "forms", "packable": False},
    4: {"mode": "rows", "packable": True},
    5: {"mode": "titled", "parts": True, "packable": False},
    6: {"mode": "titled", "parts": False, "packable": False},
    7: {"mode": "numbered", "packable": True},
    8: {"mode": "numbered", "packable": True},
    9: {"mode": "numbered", "packable": True},
    10: {"mode": "titled", "packable": False},
    11: {"mode": "numbered", "packable": True},
    12: {"mode": "numbered", "packable": True},
}


@dataclass
class Unit:
    kind: str
    key: str
    number: str | None = None
    title: str | None = None
    meta: dict = field(default_factory=dict)
    paragraphs: list[list[str]] = field(default_factory=list)
    pages: set[int] = field(default_factory=set)
    deleted: bool = False
    deleted_reason: str = ""
    packable: bool = False
    container: str = ""  # units sharing a container may be packed into one chunk
    raw: str = ""
    text: str = ""
    text_plain: str = ""

    def add_line(self, text: str, page: int, new_paragraph: bool) -> None:
        if new_paragraph or not self.paragraphs:
            self.paragraphs.append([])
        self.paragraphs[-1].append(text)
        self.pages.add(page)


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", strip_tokens(text)).strip()


def _plain_head(text: str) -> str:
    return _plain(text).lstrip("[ ").strip()


def heading_number_is_bold(line: VLine) -> bool:
    """True when a numbered line is a titled heading: the number or the words after it are bold."""
    chars: list[str] = []
    bold: list[bool] = []
    for s in line.spans:
        if s.is_marker:
            continue
        chars.extend(s.text)
        bold.extend([s.bold] * len(s.text))
    text = "".join(chars)
    m = re.match(r"^[\s\[\-*]*(\d+(?:-?[A-Z]{1,3})?)\s*\.", text)
    if not m:
        return False
    if any(bold[i] for i in range(m.start(1), m.end(1))):
        return True
    for i in range(m.end(), len(text)):
        if text[i].isalpha():
            return bold[i]
    return False


def split_heading_tokens(raw: str) -> tuple[str, str]:
    """Remove footnote tokens from a heading and return (clean heading, text to carry).

    The carried text goes to the start of the next unit, so the footnote lands where the
    heading was. A bracket that opens in the heading and closes later is carried with it.
    """
    carry = []
    text = raw
    # Deleted words inside a heading: token followed by asterisks.
    text = re.sub(rf"{TOKEN}\s*\*(?:\s*\*)*", "", text)
    for m in list(re.finditer(rf"({TOKEN})\s*(\[?)", text)):
        token, bracket = m.group(1), m.group(2)
        if bracket:
            depth, close = 0, None
            for i in range(m.end() - 1, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        close = i
                        break
            carry.append(token if close is not None else token + "[")
        else:
            carry.append(token)
    clean = re.sub(TOKEN, "", text)
    clean = re.sub(r"[\[\]]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip(" .")
    return clean, "".join(carry)


class Parser:
    def __init__(self, pages: list[Page], body_start: int, vocab: set[str]):
        self.pages = pages
        self.body_start = body_start
        self.vocab = vocab
        self.units: list[Unit] = []
        self.unit: Unit | None = None
        self.carry = ""
        self.keys: dict[str, int] = {}
        # hierarchy state
        self.region = "constitution"
        self.part: dict | None = None
        self.chapter: dict | None = None
        self.sub_heading: str | None = None
        self.schedule: dict | None = None
        self.section: str | None = None
        self.appendix: dict | None = None
        self.form: dict | None = None
        self.pending_title: dict | None = None  # heading whose title continues on next lines
        self.prev_x0 = 0.0
        self.grid: dict | None = None
        self.two_col_row: dict | None = None
        self.warnings: list[str] = []
        self.last_line_was_sub_heading = False

    # ---- unit management -------------------------------------------------------------
    def _key(self, base: str) -> str:
        n = self.keys.get(base, 0)
        self.keys[base] = n + 1
        return base if n == 0 else f"{base}~{n + 1}"

    def new_unit(self, kind: str, key: str, number: str | None = None, title: str | None = None,
                 packable: bool = False, container: str = "") -> Unit:
        self.close_two_col_row()
        meta = self.context()
        unit = Unit(kind=kind, key=self._key(key), number=number, title=title, meta=meta,
                    packable=packable, container=container)
        self.units.append(unit)
        self.unit = unit
        if self.carry:
            unit.add_line(self.carry, -1, True)
            self.carry = ""
        return unit

    def context(self) -> dict:
        meta: dict = {}
        if self.region == "constitution":
            if self.part:
                meta["part"] = self.part["number"]
                meta["part_title"] = self.part["title"]
            if self.chapter:
                meta["chapter"] = self.chapter["number"]
                meta["chapter_title"] = self.chapter["title"]
            if self.sub_heading:
                meta["sub_heading"] = self.sub_heading
        elif self.region == "schedule" and self.schedule:
            meta["schedule"] = self.schedule["number"]
            meta["schedule_name"] = self.schedule["name"]
            if self.schedule.get("title"):
                meta["schedule_title"] = self.schedule["title"]
            if self.schedule.get("basis"):
                meta["schedule_basis"] = self.schedule["basis"]
            if self.section:
                meta["section"] = self.section
        elif self.region == "appendix" and self.appendix:
            meta["appendix"] = self.appendix["number"]
            meta["appendix_title"] = self.appendix.get("title", "")
            if self.section:
                meta["section"] = self.section
        return meta

    def add_text(self, line: VLine, text: str | None = None, force_new: bool = False) -> None:
        text = line.text if text is None else text
        if not _plain(text) and not re.search(TOKEN, text):
            return
        if self.unit is None:
            self.default_unit()
        new_para = force_new or (line.x0 > self.prev_x0 + 5) or bool(PARA_START.match(text) and line.x0 >= 45)
        self.unit.add_line(text, line.page, new_para)
        self.prev_x0 = line.x0

    def default_unit(self) -> None:
        if self.region == "appendix" and self.appendix:
            n = self.appendix["number"]
            slug = re.sub(r"[^a-z0-9]+", "-", (self.section or "text").lower()).strip("-")[:40]
            self.new_unit("appendix_text", f"appendix-{n}-{slug}", title=self.section or self.appendix.get("title"),
                          container=f"appendix-{n}-{slug}")
        elif self.region == "schedule" and self.schedule:
            n = self.schedule["number"]
            slug = re.sub(r"[^a-z0-9]+", "-", (self.section or "intro").lower()).strip("-")[:40]
            self.new_unit("schedule_text", f"schedule-{n}-{slug}", title=self.section or self.schedule.get("title"),
                          container=f"schedule-{n}-{slug}")
        else:
            self.new_unit("text", "text")

    # ---- main loop ---------------------------------------------------------------------
    def parse(self) -> list[Unit]:
        for page in self.pages[self.body_start:]:
            lines = page.body
            if self.region == "appendix" and self.appendix and self.appendix["number"] == 1:
                lines = self.grid_lines(page)
            for line in lines:
                self.handle(line)
        self.close_two_col_row()
        return self.units

    def handle(self, line: VLine) -> None:
        plain = _plain_head(line.text)
        if plain == RUNNING_HEADER:
            return

        # Region switches -------------------------------------------------------------
        m = SCHEDULE_HEAD.match(plain)
        if m and line.all_bold and self.region in ("constitution", "schedule"):
            return self.start_schedule(ORDINALS.index(m.group(1)) + 1, line)
        m = APPENDIX_HEAD.match(plain)
        if m and line.all_bold:
            return self.start_appendix(len(m.group(1)) if set(m.group(1)) == {"I"} else 0, line)
        if self.region == "appendix" and plain.startswith(APPENDIX_II_TITLE):
            return self.start_appendix(2, line, title_line=True)
        if (self.region == "schedule" and BASIS_LINE.match(plain) and "basis" in self.schedule
                and _plain(line.text).startswith(("[", "(")) and abs((line.x0 + line.x1) / 2 - 180) < 50
                and not (self.pending_title and self.pending_title["kind"] == "schedule")):
            # A schedule whose heading is missing from the PDF text (the Second Schedule):
            # its "[Articles ...]" line is the first thing we see.
            nxt = self.schedule["number"] + 1
            self.warnings.append(f"{ORDINALS[nxt - 1].title()} Schedule heading missing from PDF text on "
                                 f"page {line.page}; inferred from its line '{plain[:60]}'")
            self.start_schedule(nxt, None)
            self.continue_title(line, plain)
            return

        if self.pending_title:
            if self.continue_title(line, plain):
                return

        if self.region == "constitution":
            return self.handle_constitution(line, plain)
        if self.region == "schedule":
            return self.handle_schedule(line, plain)
        return self.handle_appendix(line, plain)

    # ---- titles that wrap over several lines --------------------------------------------
    def continue_title(self, line: VLine, plain: str) -> bool:
        pt = self.pending_title
        centered = line.x0 > 30 and abs((line.x0 + line.x1) / 2 - 180) < 70
        is_upper = plain and plain.upper() == plain and any(c.isalpha() for c in plain)
        is_heading = bool(CHAPTER_HEAD.match(plain) or PART_HEAD.match(plain) or SCHEDULE_HEAD.match(plain))
        if pt["kind"] in ("part", "chapter", "section") and centered and is_upper and not is_heading \
                and not NUM_HEAD.match(line.text):
            clean, carry = split_heading_tokens(line.text)
            pt["target"]["title"] = (pt["target"].get("title", "") + " " + clean).strip()
            self.carry += carry
            if pt["kind"] == "section":
                self.section = f"{pt['label']} — {nice_title(pt['target']['title'])}"
            return True
        if pt["kind"] == "schedule" and abs((line.x0 + line.x1) / 2 - 180) < 50:
            if BASIS_LINE.match(plain) and "basis" not in self.schedule:
                self.schedule["basis"] = _basis_articles(plain)
                self.schedule["basis_text"] = plain.strip("[]() ")
                self.carry += split_heading_tokens(line.text)[1]
                return True
            if line.all_bold and not PART_HEAD.match(plain) and not LIST_HEAD.match(plain):
                clean, carry = split_heading_tokens(line.text)
                self.schedule["title"] = (self.schedule.get("title", "") + " " + clean).strip()
                self.carry += carry
                return True
        roughly_centered = line.x0 > 40 and abs((line.x0 + line.x1) / 2 - 180) < 70
        if pt["kind"] == "appendix" and roughly_centered and (is_upper or plain.startswith(("[", "C.O."))):
            clean, carry = split_heading_tokens(line.text)
            if plain.startswith("["):
                self.appendix["date"] = clean
            else:
                self.appendix["title"] = (self.appendix.get("title", "") + " " + clean).strip()
            self.carry += carry
            return True
        self.pending_title = None
        return False

    # ---- Articles ----------------------------------------------------------------------
    def handle_constitution(self, line: VLine, plain: str) -> None:
        if plain == "PREAMBLE" and line.all_bold:
            self.new_unit("preamble", "preamble", title="Preamble")
            return
        m = PART_HEAD.match(plain)
        if m and line.all_bold:
            _, carry = split_heading_tokens(line.text)
            self.carry += carry
            self.part = {"number": m.group(1), "title": ""}
            self.chapter = None
            self.sub_heading = None
            self.pending_title = {"kind": "part", "target": self.part}
            self.unit = None
            return
        m = CHAPTER_HEAD.match(plain)
        if m:
            clean, carry = split_heading_tokens(line.text)
            self.carry += carry
            title = CHAPTER_HEAD.match(clean)
            self.chapter = {"number": m.group(1), "title": (title.group(2) if title else m.group(2)).strip(" ,")}
            self.sub_heading = None
            if plain.rstrip().endswith((",", "AND")):
                self.pending_title = {"kind": "chapter", "target": self.chapter}
            self.unit = None
            return
        centered = abs((line.x0 + line.x1) / 2 - 180) < 50
        if (line.all_italic and centered and line.x0 > 60 and len(plain) < 100
                and re.match(r"^[A-Za-z][a-z]+", plain) and "*" not in plain and not plain.endswith(("—", ":"))):
            clean, carry = split_heading_tokens(line.text)
            # A sub-heading that wraps onto a second italic line ("... and their Members").
            if self.unit is None and self.sub_heading and self.last_line_was_sub_heading:
                self.sub_heading = f"{self.sub_heading} {clean}"
            else:
                self.sub_heading = clean
            self.carry += carry
            self.unit = None
            self.last_line_was_sub_heading = True
            return
        self.last_line_was_sub_heading = False
        # A Part whose "PART ..." line is missing from the PDF text shows up as its bare,
        # centred, upper-case title (e.g. "THE UNION TERRITORIES" for Part VIII).
        if (centered and line.x0 > 90 and plain.upper() == plain and re.search(r"[A-Z]{3}", plain)
                and not line.all_bold and len(plain) < 60 and not (self.unit and self.unit.meta.get("_title_open"))):
            clean, carry = split_heading_tokens(line.text)
            self.warnings.append(f"Part heading missing from PDF text on page {line.page + 1}; "
                                 f"found only its title '{clean}' (Part number taken from Contents)")
            self.part = {"number": "?", "title": clean}
            self.chapter = None
            self.sub_heading = None
            self.carry += carry
            self.unit = None
            return
        m = NUM_HEAD.match(line.text)
        if m and line.x0 >= 45 and heading_number_is_bold(line):
            return self.start_numbered_heading(line, m, kind="article", key=f"article-{_num(m)}")
        if self.unit and self.unit.meta.get("_title_open"):
            return self.extend_title(line)
        self.add_text(line)

    def start_numbered_heading(self, line: VLine, m: re.Match, kind: str, key: str,
                               packable: bool = False, container: str = "") -> None:
        lead, num, rest = m.group("lead"), _num(m), m.group("rest")
        title_raw, sep, body = _split_dash(rest)
        unit = self.new_unit(kind, key, number=num, packable=packable, container=container)
        unit.meta["_lead"] = lead
        if sep:
            self.finish_title(unit, title_raw, body, line)
        else:
            unit.meta["_title_open"] = title_raw
            unit.pages.add(line.page)
        self.prev_x0 = line.x0

    def extend_title(self, line: VLine) -> None:
        unit = self.unit
        text = line.text
        so_far = unit.meta["_title_open"]
        title_raw, sep, body = _split_dash(text)
        bold_start = bool(line.spans) and next((s.bold for s in line.spans if not s.is_marker and s.text.strip()), False)
        if sep or bold_start:
            so_far = f"{so_far} {title_raw}"
            if sep:
                del unit.meta["_title_open"]
                self.finish_title(unit, so_far, body, line)
            else:
                unit.meta["_title_open"] = so_far
            return
        # The heading had no dash: everything so far was the title.
        del unit.meta["_title_open"]
        self.finish_title(unit, so_far, "", line)
        self.add_text(line)

    def finish_title(self, unit: Unit, title_raw: str, body: str, line: VLine) -> None:
        lead = unit.meta.pop("_lead", "")
        title, carry = split_heading_tokens(title_raw)
        unit.title = title
        prefix = re.sub(r"[^\[⟦⟧0-9:*]", "", lead)  # tokens and an opening bracket, if any
        body_text = (prefix + carry + " " + body).strip()
        unit.pages.add(line.page)
        if body_text and (_plain(body_text) or re.search(TOKEN, body_text)):
            unit.add_line(body_text, line.page, True)
        self.prev_x0 = line.x0

    # ---- Schedules ---------------------------------------------------------------------
    def start_schedule(self, number: int, line: VLine | None) -> None:
        self.close_two_col_row()
        self.region = "schedule"
        self.carry = split_heading_tokens(line.text)[1] if line else ""
        self.schedule = {"number": number, "name": f"{ORDINALS[number - 1].title()} Schedule",
                         "rule": SCHEDULE_RULES[number], "title": ""}
        self.section = None
        self.form = None
        self.unit = None
        self.two_col_ready = False
        self.pending_title = {"kind": "schedule", "target": self.schedule}

    def handle_schedule(self, line: VLine, plain: str) -> None:
        rule = self.schedule["rule"]
        n = self.schedule["number"]
        mode = rule["mode"]

        if PART_HEAD.match(plain) and line.all_bold and rule.get("parts"):
            label = f"Part {PART_HEAD.match(plain).group(1)}"
            _, carry = split_heading_tokens(line.text)
            self.carry += carry
            self.section = label
            self.pending_title = {"kind": "section", "label": label, "target": {"title": ""}}
            self.unit = None
            return
        m = LIST_HEAD.match(plain)
        if m and line.all_bold:
            _, carry = split_heading_tokens(line.text)
            self.carry += carry
            self.section = f"List {m.group(1)} — {m.group(2).strip()}"
            self.unit = None
            return

        if mode == "two_col":
            return self.handle_two_col(line, plain)
        if mode == "forms":
            return self.handle_forms(line, plain)

        text = line.text
        if mode == "rows":
            text = DOT_LEADER.sub(" — ", text)
        m = NUM_HEAD.match(text)
        if m and line.x0 >= 40:
            num = _num(m)
            if mode == "titled":
                if heading_number_is_bold(line):
                    return self.start_numbered_heading(line, m, "schedule_paragraph", f"schedule-{n}-para-{num}")
            elif mode == "numbered" and not rule["packable"]:
                return self.start_untitled(line, m, "schedule_paragraph", f"schedule-{n}-para-{num}")
            else:
                sec = _slug(self.section) if self.section else ""
                key = f"schedule-{n}-{sec + '-' if sec else ''}entry-{num}"
                container = f"schedule-{n}-{sec or 'entries'}"
                return self.start_untitled(line, m, "schedule_entry", key, packable=True, container=container, text=text)
        if self.unit and self.unit.meta.get("_title_open"):
            return self.extend_title(line)
        self.add_text(line, text)

    def start_untitled(self, line: VLine, m: re.Match, kind: str, key: str, packable: bool = False,
                       container: str = "", text: str | None = None) -> None:
        unit = self.new_unit(kind, key, number=_num(m), packable=packable, container=container)
        prefix = re.sub(r"[^\[⟦⟧0-9:*]", "", m.group("lead"))
        # Packed list entries keep their number in the text ("54. Taxes on ..."): several
        # share a chunk, and questions cite them by number ("entry 54 of List II").
        number = f"{_num(m)}. " if packable else ""
        unit.add_line((prefix + " " + number + m.group("rest")).strip(), line.page, True)
        self.prev_x0 = line.x0

    def handle_forms(self, line: VLine, plain: str) -> None:
        centered = abs((line.x0 + line.x1) / 2 - 180) < 40
        m = ROMAN_LINE.match(plain)
        if m and centered:
            _, carry = split_heading_tokens(line.text)
            self.carry += carry
            self.form = {"number": m.group(1), "sub": None}
            self.unit = None
            return
        if re.fullmatch(r"[A-D]", plain) and line.all_bold and self.form:
            self.form["sub"] = plain
            self.unit = None
            return
        if plain == "solemnly affirm":
            if self.unit and self.unit.paragraphs:
                last = self.unit.paragraphs[-1]
                last[-1] = last[-1].replace("in the name of God", "in the name of God / solemnly affirm")
            return
        if self.unit is None and self.form:
            label = self.form["number"] + (f"-{self.form['sub']}" if self.form["sub"] else "")
            self.new_unit("schedule_form", f"schedule-3-form-{label}", number=label, title=f"Form {label}")
        if self.unit and self.unit.kind == "schedule_form" and not self.unit.title.endswith(":—") \
                and len(self.unit.paragraphs) == 0 and re.search(r"Form of", plain):
            self.unit.title = f"Form {self.unit.number}: {plain.rstrip(':— ')}"
        self.add_text(line)

    # First Schedule: Name | Territories table -------------------------------------------
    def handle_two_col(self, line: VLine, plain: str) -> None:
        if re.match(r"^[IVX]+\.\s+THE\s+", plain):
            self.close_two_col_row()
            clean, carry = split_heading_tokens(line.text)
            self.carry += carry
            self.section = clean
            self.unit = None
            return
        if plain.replace(" ", "") in ("NameTerritories", "NameExtent"):
            return
        split_x = 100
        left = [s for s in line.spans if s.x0 < split_x]
        right = [s for s in line.spans if s.x0 >= split_x]
        ltext = _render(VLine(line.page, left).finish()) if left else ""
        rtext = _render(VLine(line.page, right).finish()) if right else ""
        m = NUM_HEAD.match(ltext) if ltext else None
        if m and re.match(r"^\d", _num(m)):
            self.close_two_col_row()
            self.two_col_row = {"num": _num(m), "lead": m.group("lead"),
                                "left": [m.group("rest")], "right": [rtext] if rtext else [], "page": line.page}
            return
        if self.two_col_row is None:
            self.add_text(line)
            return
        if ltext.strip():
            self.two_col_row["left"].append(ltext)
        if rtext.strip():
            self.two_col_row["right"].append(rtext)

    def close_two_col_row(self) -> None:
        row = self.two_col_row
        if not row:
            return
        self.two_col_row = None
        n = self.schedule["number"]
        sec = _slug(self.section) if self.section else "entries"
        unit = Unit(kind="schedule_entry", key=self._key(f"schedule-{n}-{sec}-entry-{row['num']}"),
                    number=row["num"], meta=self.context(), packable=True, container=f"schedule-{n}-{sec}")
        prefix = re.sub(r"[^\[⟦⟧0-9:*]", "", row["lead"])
        name = " ".join(p.strip() for p in row["left"] if p.strip())
        territory = join_lines(row["right"], self.vocab)
        unit.add_line(f"{prefix}{row['num']}. {name} — {territory}".strip(), row["page"], True)
        if self.carry:
            unit.paragraphs[0].insert(0, self.carry)
            self.carry = ""
        self.units.append(unit)
        self.unit = None

    # ---- Appendices --------------------------------------------------------------------
    def start_appendix(self, number: int, line: VLine, title_line: bool = False) -> None:
        self.close_two_col_row()
        self.region = "appendix"
        self.appendix = {"number": number, "title": ""}
        self.section = None
        self.unit = None
        clean, carry = split_heading_tokens(line.text)
        self.carry = carry
        if title_line:
            self.appendix["title"] = clean
        self.pending_title = {"kind": "appendix", "target": self.appendix}

    def handle_appendix(self, line: VLine, plain: str) -> None:
        n = self.appendix["number"]
        is_upper = plain.upper() == plain and sum(c.isalpha() for c in plain) > 3
        if n == 1 and is_upper and not line.text.startswith("§"):
            if re.match(r"^(THE\s+\w+\s+SCHEDULE|PART\s+[IVX]+|[IVX]+\.\s|[A-Z]\.\s)", plain):
                # The Act's own Schedules have Parts: keep "The First Schedule › Part II".
                if re.match(r"^THE\s+\w+\s+SCHEDULE", plain):
                    self.appendix["schedule"] = nice_title(plain)
                    self.section = plain
                elif re.match(r"^PART\s+[IVX]+", plain) and self.appendix.get("schedule"):
                    self.section = f"{self.appendix['schedule']} › {nice_title(plain)}"
                else:
                    self.section = plain
                self.appendix["_heading_open"] = True
                self.unit = None
                return
            if self.appendix.get("_heading_open") and self.unit is None:
                self.section = f"{self.section} {plain}"
                return
        self.appendix["_heading_open"] = False
        if line.text.startswith("§ROW§"):
            return self.add_grid_row(line)
        m = NUM_HEAD.match(line.text)
        if m and line.x0 >= 45 and (n == 2 or (n == 1 and heading_number_is_bold(line))):
            key = f"appendix-{n}-para-{_num(m)}" if n == 2 else f"appendix-{n}-section-{_num(m)}"
            if n == 1:
                return self.start_numbered_heading(line, m, "appendix_section", key)
            return self.start_untitled(line, m, "appendix_section", key)
        if self.unit and self.unit.meta.get("_title_open"):
            return self.extend_title(line)
        self.add_text(line)

    # Appendix I enclave tables: columns are given by a "1 2 3 4 5 6" header row -------------
    def grid_lines(self, page: Page) -> list[VLine]:
        """Rebuild Appendix I's enclave tables as one line per row: 'Column: value | ...'.

        Each table has a '1 2 3 4 5 6' header row (repeated on every page); its column titles
        sit just above the first one. A page can hold the end of one table and the start of
        the next, and a table ends at the first line of running prose.
        """
        headers = [l for l in page.body if re.fullmatch(r"1 2 3 4 5( 6)?", l.plain)]
        if not headers:
            return page.body
        out: list[VLine] = [l for l in page.body if l.y0 < headers[0].y0]
        for h_index, header in enumerate(headers):
            nxt_y = headers[h_index + 1].y0 if h_index + 1 < len(headers) else 1e9
            # Digit spans carry trailing spaces, so use the words' left edges as positions.
            digits = sorted(w[0] for w in page.words
                            if abs(w[1] - header.y0) < 3 and w[4].isdigit() and len(w[4]) == 1)
            names = self.grid["names"] if self.grid else None
            prev_y = headers[h_index - 1].y0 if h_index else -1
            title_lines = [l for l in page.body if prev_y < l.y0 < header.y0 and header.y0 - l.y0 < 45
                           and len({_col(s.x0, digits) for s in l.spans}) > 1]
            if title_lines:
                top = min(l.y0 for l in title_lines) - 1
                cols: list[list[str]] = [[] for _ in digits]
                for w in sorted(page.words, key=lambda w: (round(w[1]), w[0])):
                    if top <= w[1] < header.y0 - 2:
                        # Column titles are centred like the digits: use the nearest digit.
                        mid = (w[0] + w[2]) / 2
                        cols[min(range(len(digits)), key=lambda i: abs(digits[i] + 2.5 - mid))].append(w[4])
                names = [" ".join(c) for c in cols]
                # "Sl. No." is printed with "No." under the next column's title.
                if names and names[0] == "Sl." and len(names) > 1 and names[1].endswith(" No."):
                    names[0], names[1] = "Sl. No.", names[1][:-4]
                out = [l for l in out if l not in title_lines]
            self.grid = {"names": names, "last_num": (self.grid or {}).get("last_num") if not title_lines else None}
            seg = [l for l in page.body if header.y0 < l.y0 < nxt_y]
            if h_index + 1 < len(headers):  # the next table's column titles are not rows
                seg = [l for l in seg if headers[h_index + 1].y0 - l.y0 >= 45 or len({_col(s.x0, digits) for s in l.spans}) <= 1]
            out.extend(self._grid_rows(seg, digits, names))
        return out

    def _grid_rows(self, lines: list[VLine], digits: list[float], names: list[str] | None) -> list[VLine]:
        out: list[VLine] = []
        rows: list[dict] = []
        row: dict | None = None
        in_table = True
        for l in lines:
            if not in_table or (len(l.plain) > 45 and len({_col(s.x0, digits) for s in l.spans}) <= 2):
                in_table = False
                out.extend(self._emit_rows(rows, names))
                rows, row = [], None
                out.append(l)
                continue
            cells: list[list[Span]] = [[] for _ in digits]
            for s in l.spans:
                cells[_col(s.x0, digits)].append(s)
            text_of = lambda i: _render(VLine(l.page, cells[i]).finish()).strip() if cells[i] else ""
            c0, c1 = text_of(0), text_of(1) if len(cells) > 1 else ""
            if re.match(r"^([A-Z]\.\s+\S|Enclaves\b|Adverse\b)", l.plain) and not re.fullmatch(r"\d+\.", c0):
                out.extend(self._emit_rows(rows, names))
                rows, row = [], None
                out.append(l)
                continue
            starts = bool(re.fullmatch(r"\d+\.", c0)) or bool(re.match(r"^\([ivx]+\)", c1))
            if starts or row is None:
                num = c0.rstrip(".") if re.fullmatch(r"\d+\.", c0) else (row or {}).get("num", self.grid.get("last_num"))
                row = {"cells": [[] for _ in digits], "page": l.page, "num": num}
                self.grid["last_num"] = num
                rows.append(row)
            for i in range(len(cells)):
                if cells[i]:
                    row["cells"][i].append(text_of(i))
        out.extend(self._emit_rows(rows, names))
        return out

    @staticmethod
    def _emit_rows(rows: list[dict], names: list[str] | None) -> list[VLine]:
        out = []
        for r in rows:
            parts = []
            for i, cell in enumerate(r["cells"]):
                value = " ".join(x for x in cell if x)
                if i == 0 and not value and r.get("num"):
                    value = r["num"]
                if not value:
                    continue
                name = names[i] if names and i < len(names) and names[i] else f"Column {i + 1}"
                parts.append(f"{name}: {value.rstrip('.') if i == 0 else value}")
            fake = VLine(r["page"], [Span("", 9, "", False, False, 36, 0, 37, 1)]).finish()
            fake.text = "§ROW§" + " | ".join(parts)
            out.append(fake)
        return out

    def add_grid_row(self, line: VLine) -> None:
        text = line.text[len("§ROW§"):]
        sec = _slug(self.section or "table")
        unit = self.new_unit("appendix_entry", f"appendix-1-{sec}-row", packable=True,
                             container=f"appendix-1-{sec}")
        unit.add_line(text, line.page, True)
        self.unit = None


def _num(m: re.Match) -> str:
    return m.group("num").replace("-", "")


_HEADING_DASH = re.compile(r"—|––|--|⎯")


def _split_dash(text: str) -> tuple[str, str, str]:
    """Split a heading at the dash that ends its title ("21. Title.—Body"); some use "––"."""
    m = _HEADING_DASH.search(text)
    if not m:
        return text, "", ""
    return text[:m.start()], m.group(0), text[m.end():]


def _col(x: float, header_x: list[float]) -> int:
    """Column of a cell starting at x: the first header digit at or right of the cell's left edge.

    Header digits are centred over their columns, while cells are left-aligned, so a cell
    belongs to the first header position it does not overshoot.
    """
    for i, hx in enumerate(header_x):
        if hx >= x - 5:
            return i
    return max(len(header_x) - 1, 0)


def _slug(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:40]


def _basis_articles(text: str) -> list[dict]:
    """'[Articles 75(4), 99, 124(6)]' -> [{'article': '75', 'clause': '4'}, ...]"""
    out = []
    for m in re.finditer(r"(\d+[A-Z]{0,3})\s*(?:\((\w+)\))?", text):
        out.append({"article": m.group(1), "clause": m.group(2)})
    return out


def build_units(pages: list[Page], body_start: int, vocab: set[str]) -> tuple[list[Unit], list[str]]:
    """Parse the body into units; also return structural warnings (e.g. inferred headings)."""
    parser = Parser(pages, body_start, vocab)
    units = parser.parse()
    for u in units:
        u.meta.pop("_title_open", None)
        u.meta.pop("_lead", None)
        u.pages.discard(-1)
        u.raw = "\n".join(join_lines(p, vocab) for p in u.paragraphs if p)
        _mark_omitted(u)
    return units, parser.warnings


_OMITTED = re.compile(r"^\W*(Omitted|Rep\.|Repealed)\b")
_OMISSION_NOTE = re.compile(r"^\W*Articles?\b[^.]*?\b(?:subs\.|substituted|omitted|rep\.|repealed)\s+by\b", re.I)


def _mark_omitted(u: Unit) -> None:
    """Omitted provisions keep only a bracketed title and a note about the omission:
    "31. [Compulsory acquisition of property.].—Omitted by ...",
    "232. [Interpretation.] Articles 230, 231 and 232 subs. by articles 230 and 231 ...".
    Only the omission note counts: a bracketed title is not a signal, because Articles whose
    heading was substituted (e.g. 368) have one too.
    """
    body = _plain(u.raw)
    if _OMITTED.match(body) or _OMISSION_NOTE.match(body):
        u.deleted = True
        u.deleted_reason = body[:160]
