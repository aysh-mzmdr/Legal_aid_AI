"""Build the chunk set: `uv run rag-ingest [pdf] [out_dir]`.

Writes:
  chunks.jsonl          one chunk per line, with header, text (footnotes inline as
                        "[Note: ...]"), text_without_notes, metadata, refs and cited_by
  report.json           every count, log and validation result
  report.md             the same, readable
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from .abbreviations import expand, load_abbreviations
from .chunking import build_chunks
from .footnotes import parse_footnotes
from .index_check import apply_contents_parts, parse_contents, validate
from .notes import resolve_notes, strip_notes
from .pdf_layout import read_pdf
from .references import extract_references
from .structure import build_units
from .text import build_vocabulary

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PDF = ROOT / "Raw Data" / "Constitution_of_India.pdf"
DEFAULT_OUT = ROOT / "data" / "processed"


def _find_pages(pages) -> tuple[int, int]:
    """(first Contents page, first body page = the Preamble)."""
    contents = next(p.number for p in pages if any(l.plain == "CONTENTS" for l in p.body))
    body = next(p.number for p in pages[contents + 1:]
                if any(l.plain == "PREAMBLE" and l.all_bold for l in p.body))
    return contents, body


def build(pdf: Path = DEFAULT_PDF, out_dir: Path = DEFAULT_OUT) -> dict:
    pages = read_pdf(str(pdf))
    contents_start, body_start = _find_pages(pages)
    vocab = build_vocabulary(pages)
    abbreviations = load_abbreviations(pages)

    footnotes, foot_log = parse_footnotes(pages, body_start, vocab, abbreviations)

    units, structure_warnings = build_units(pages, body_start, vocab)
    contents = parse_contents(pages, contents_start, body_start)
    structure_warnings += apply_contents_parts(contents, units)
    note_log = resolve_notes(units, footnotes)

    body_abbrev: dict[str, int] = {}
    for u in units:
        if u.text:
            u.text = expand(u.text, abbreviations, body_abbrev)
            u.text_plain = strip_notes(u.text)

    chunks = build_chunks(units)
    seen: Counter = Counter()
    for c in chunks:
        seen[c.id] += 1
        if seen[c.id] > 1:
            c.id = f"{c.id}~{seen[c.id]}"
    ref_stats = extract_references(chunks)

    report = validate(contents, units, chunks, structure_warnings)
    report["source"] = {"pdf": pdf.name, "pages": len(pages), "contents_pages": [contents_start + 1, body_start],
                        "body_starts_at_pdf_page": body_start + 1}
    report["footnotes"] = {
        "parsed": len(footnotes),
        "inlined": note_log.placed,
        "inlined_at_marker_because_bracket_spans_units": note_log.placed_cross_unit,
        "excluded_as_ambiguous": foot_log.excluded,
        "numbering_gaps": foot_log.gaps,
        "continued_across_pages": foot_log.cross_page,
        "ibid_resolved": foot_log.ibid_resolved,
        "ibid_unresolved": foot_log.ibid_unresolved,
        "notes_dropped": note_log.dropped,
        "never_referenced": note_log.unreferenced,
    }
    report["deleted_text"] = {
        "omitted_provisions_skipped": [{"unit": u.key, "reason": u.deleted_reason} for u in units if u.deleted],
        "deleted_passages_removed": note_log.deletions_removed,
        "stray_amendment_brackets_removed": note_log.stray_brackets_removed,
    }
    report["abbreviations"] = {
        "list": {a.short: a.long for a in abbreviations},
        "expanded_in_footnotes": foot_log.abbreviation_counts,
        "expanded_in_body": body_abbrev,
    }
    edge_types = Counter(r["type"] for c in chunks for r in c.refs)
    report["references"] = {
        "edges": sum(edge_types.values()),
        "by_type": dict(edge_types),
        "chunks_with_refs": sum(1 for c in chunks if c.refs),
        "extraction_stats": ref_stats,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    (out_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(r: dict) -> str:
    a, ch, fn = r["articles"], r["chunks"], r["footnotes"]
    lines = [
        "# Ingestion report",
        "",
        f"Source: `{r['source']['pdf']}`, {r['source']['pages']} pages; body starts at PDF page "
        f"{r['source']['body_starts_at_pdf_page']}.",
        "",
        "## Structure vs. Contents pages",
        "",
        f"- Articles in Contents: {a['in_contents']} ({a['in_contents_live']} live, {a['in_contents_omitted']} omitted)",
        f"- Articles parsed: {a['parsed']} ({a['parsed_live']} live, {a['parsed_deleted']} omitted and skipped)",
        f"- Missing from parse: {a['missing_from_parse'] or 'none'}",
        f"- Parsed but not in Contents: {a['not_in_contents'] or 'none'}",
        f"- Omitted in Contents but parsed live: {a['omitted_in_contents_but_parsed_live'] or 'none'}",
        f"- Live in Contents but parsed as omitted: {a['live_in_contents_but_parsed_deleted'] or 'none'}",
        f"- Part mismatches: {len(a['part_mismatches'])}; title mismatches: {len(a['title_mismatches'])}",
        f"- Article order matches Contents: {a['order_matches_contents']}",
        f"- Schedules in Contents: {r['schedules']['in_contents']}; parsed: {r['schedules']['parsed']}",
        f"- Appendices parsed: {r['appendices']}",
    ]
    for w in r["structure_warnings"]:
        lines.append(f"- Warning: {w}")
    for t in a["title_mismatches"]:
        lines.append(f"  - Article {t['article']}: Contents “{t['contents']}” vs parsed “{t['parsed']}” ({t['similarity']})")
    lines += [
        "",
        "## Chunks",
        "",
        f"- {ch['count']} chunks: {ch['by_kind']}",
        f"- Size (chars): min {ch['chars_min']}, median {ch['chars_median']}, max {ch['chars_max']}",
        f"- Duplicate ids: {ch['duplicate_ids'] or 'none'}",
        f"- Units not fully covered by their chunks: {ch['units_not_fully_covered'] or 'none'}",
        f"- Problems: {len(ch['problems'])}",
    ]
    for p in ch["problems"][:30]:
        lines.append(f"  - {p['chunk']}: {p['issue']}")
    lines += [
        "",
        "## Footnotes",
        "",
        f"- Parsed: {fn['parsed']}; inlined: {fn['inlined']} "
        f"({fn['inlined_at_marker_because_bracket_spans_units']} at the marker because the amended text spans provisions)",
        f"- “ibid.” resolved: {fn['ibid_resolved']}; unresolved: {len(fn['ibid_unresolved'])}",
        f"- Footnotes continued across pages: {len({x['continues'] for x in fn['continued_across_pages']})}",
        f"- Excluded as ambiguous: {len(fn['excluded_as_ambiguous'])}",
    ]
    for e in fn["excluded_as_ambiguous"]:
        lines.append(f"  - page {e['page'] + 1}, footnote {e['key']}: {e['reason']}")
    lines.append(f"- Notes dropped: {len(fn['notes_dropped'])}")
    reasons = Counter(d["reason"] for d in fn["notes_dropped"])
    for reason, n in reasons.most_common():
        lines.append(f"  - {reason}: {n}")
    lines.append(f"- Footnotes never referenced by a marker: {len(fn['never_referenced'])}")
    d = r["deleted_text"]
    lines += [
        "",
        "## Deleted text",
        "",
        f"- Omitted provisions skipped: {len(d['omitted_provisions_skipped'])}",
        f"- Deleted passages removed: {d['deleted_passages_removed']}",
        f"- Stray amendment brackets removed: {d['stray_amendment_brackets_removed']}",
        "",
        "## Abbreviations",
        "",
        f"- Expanded in footnotes: {r['abbreviations']['expanded_in_footnotes']}",
        f"- Expanded in body: {r['abbreviations']['expanded_in_body'] or 'none'}",
        "",
        "## Cross-references",
        "",
        f"- {r['references']['edges']} edges on {r['references']['chunks_with_refs']} chunks: {r['references']['by_type']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    report = build(pdf, out)
    print((out / "report.md").read_text(encoding="utf-8"))
    print(f"Wrote {report['chunks']['count']} chunks to {out / 'chunks.jsonl'}")


if __name__ == "__main__":
    main()
