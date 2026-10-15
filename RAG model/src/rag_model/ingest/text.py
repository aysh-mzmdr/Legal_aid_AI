"""Small text helpers shared by the ingestion modules."""

from __future__ import annotations

import re

from .pdf_layout import Page, strip_tokens

_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with", "from", "into"}


def nice_title(text: str | None) -> str:
    """'PROVISIONS AS TO THE JUDGES OF THE SUPREME COURT' -> 'Provisions as to the Judges of the Supreme Court'.

    Only all-capitals text is changed; abbreviations ("C.O.") and Roman numerals keep their case.
    """
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip(" .")
    text = re.sub(r"\(\s+", "(", text)
    if text.upper() != text:
        return text
    out = []
    words = text.split(" ")
    for i, word in enumerate(words):
        w = word.lower()
        after_numeral = i > 0 and re.fullmatch(r"[IVXL]+\.|[A-Z]\.", words[i - 1])
        if "." in word.strip(".,") or re.fullmatch(r"[IVXL]+[A-C]?[.,]?", word) and (i > 0 or word.endswith(".")):
            out.append(word)  # abbreviations ("C.O.") and Roman numerals ("II.") as printed
        elif i > 0 and w in _SMALL_WORDS and not after_numeral:
            out.append(w)
        else:
            # Capitalise each hyphenated part that starts with a letter: "Auditor-General",
            # but "16th", not "16Th".
            out.append("-".join(re.sub(r"^(\(?)([a-z])", lambda m: m.group(1) + m.group(2).upper(), p) for p in w.split("-")))
    return " ".join(out)


def build_vocabulary(pages: list[Page]) -> set[str]:
    """Every word (lower-cased) the PDF uses anywhere other than at a line end."""
    vocab: set[str] = set()
    for page in pages:
        for line in page.body + page.footnote_lines:
            words = _WORD.findall(strip_tokens(line.text))
            vocab.update(w.lower() for w in words[:-1])
    return vocab


def join_lines(lines: list[str], vocab: set[str]) -> str:
    """Join wrapped lines, deciding whether a line-end hyphen is a real hyphen.

    'commence-' + 'ment' becomes 'commencement' only when the PDF uses that word elsewhere
    and never uses the hyphenated form. Hyphens in dates ('15-11-' + '2000') are kept.
    """
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
            continue
        if out.endswith("-") and not out.endswith("--"):
            left = re.search(r"([A-Za-z]+)-$", out)
            right = re.match(r"([A-Za-z]+)", line)
            if left and right:
                merged = (left.group(1) + right.group(1)).lower()
                hyphenated = f"{left.group(1)}-{right.group(1)}".lower()
                if merged in vocab and hyphenated not in vocab:
                    out = out[:-1] + line
                    continue
            out += line
            continue
        out += " " + line
    return re.sub(r"[ \t]+", " ", out).strip()
