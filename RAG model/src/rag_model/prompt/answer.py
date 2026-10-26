"""Answer a question from retrieved Constitution passages with the LLM.

ask() yields events, used by both the command line and the web server (server.py):

    {"type": "status", "stage": "searching" | "answering"}
    {"type": "sources", "query": "...", "sources": [{"n", "id", "label", "title", "kind"}, ...]}
    {"type": "token", "text": "..."}                    the answer, streamed
    {"type": "citations", "cited": [1, 3]}              passage numbers the answer cites
    {"type": "followups", "questions": ["...", ...]}    up to 3 questions to ask next
    {"type": "done", "model": "gemini-3.8-flash"}
    {"type": "error", "message": "...", "retryable": bool}

The answer cites passages by number ("... [1][3]"); "n" in sources is that number. The search
uses the question rewritten in legal terms ("query", from refine.py); the answer responds to
the question as asked. `history` (earlier messages of the chat, [{"role", "text"}], oldest
first) lets a follow-up like "what about my lawyer?" be understood: it goes to the rewrite and,
as earlier turns, to the answer.

    uv run rag-ask "Can the police arrest me without telling me why?"
"""

from __future__ import annotations

import re
import sys
import time
from typing import Iterator

from ..config import settings
from ..llm import LLMError, get_llm
from .refine import refine_question

SYSTEM_PROMPT = """You help people in India understand the law and their rights under the Constitution of India, and work out what it means for the situation they are in.

Source of truth:
- Answer only from the numbered passages in the message. They are your only source. Do not use your own knowledge of the law, case law, statutes or news, even if you are sure it is correct.
- Do not guess, fill gaps or invent provisions, numbers, rules or procedures. If a point is not in the passages, leave it out.
- Applying a passage to the person's situation is not outside knowledge, and it is your main job. The Constitution is written in general terms ("any person", "the State", "freedom of speech and expression", "admission into any educational institution"). When a passage's general rule covers their situation, say so and explain how it applies, even if the passage does not mention their exact circumstances. For example, freedom of speech and expression covers criticising the government online, subject to the limits the passages state.
- When they ask what they can do, explain the rights they can rely on and, if the passages include them, the remedies (such as going to the Supreme Court or a High Court).
- If the passages answer only part of the question, answer that part, then say in one sentence what they do not cover (for example, the procedure to follow or the ordinary laws on the subject).
- Only if none of the passages is relevant to the question, reply with: "I can't answer this from the data I have." followed by one sentence on what is missing. Never write that sentence and then answer anyway.

Earlier conversation:
- Earlier messages show what the person has already asked and been told. Use them only to understand what the new question refers to, e.g. "what about my lawyer?" after a question about arrest.
- They are not a source of law. Every statement about the law in your answer must come from, and cite, the passages sent with the new question. Do not repeat an earlier answer unless asked.
- If the new question is about the conversation itself ("what did you say?", "can you repeat that?", "summarise your last answer"), answer it from the earlier messages, without citations and without the "I can't answer" reply.

Citations:
- Put the passage number in square brackets after every statement drawn from it, e.g. "You must be told the grounds of your arrest [1]." Cite several as [1][3]. Use only numbers that appear in the passages.
- Name the provision directly when the passage makes it clear, e.g. "Article 22 does not allow ..." or "Under the Tenth Schedule, para 2, ...".
- Do not end with a list of references, sources or citations; the app shows the cited passages itself.

How to answer:
- Write in plain, calm language for someone without legal training. Explain any legal term you have to use.
- Start with a direct answer in one or two sentences. Then give short bullet points: what the right or rule is, how it applies to their situation, and any exceptions or limits the passages state.
- If the person describes their own case, focus on what the passages say they can rely on or do. Be specific, not general.
- Keep the answer under about 250 words.
- Do not add a disclaimer; the app already says that AI can make mistakes. Only if the person faces something urgent or serious (such as arrest, detention or a threat to their safety), add one short line suggesting they contact a lawyer or their local legal services authority."""

FOLLOWUP_PROMPT = """Suggest exactly 3 follow-up questions this person might ask next about their rights under the Constitution of India, based on their question and the answer they received.
- Each question must make sense on its own: name the topic, never "it" or "that" for something said earlier.
- Write them in the person's own voice, under 15 words each.
- Output only the 3 questions, one per line, with no numbering, bullets or other text."""

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
ROMAN = {1: "I", 2: "II", 3: "III"}


def clean_history(history) -> list[dict]:
    """Earlier messages as alternating user/assistant turns ending with an answer, without the
    old [n] marks (they pointed at another question's passages)."""
    turns: list[dict] = []
    for m in history or []:
        text = re.sub(r"\s*" + CITATION.pattern, "", m.get("text", "")).strip()
        if m.get("role") not in ("user", "assistant") or not text:
            continue
        if turns and turns[-1]["role"] == m["role"]:
            turns[-1] = {"role": m["role"], "text": text}  # keep the later of two same-role messages
        elif turns or m["role"] == "user":
            turns.append({"role": m["role"], "text": text})
    if turns and turns[-1]["role"] == "user":
        turns.pop()
    return turns


def build_messages(question: str, passages: list[dict], history: list[dict] = ()) -> list[dict]:
    context = "\n\n".join(f"[{i}] {p['header']}\n{p['text']}" for i, p in enumerate(passages, 1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *({"role": m["role"], "content": m["text"]} for m in history),
        {"role": "user", "content": f"Passages:\n\n{context}\n\nQuestion: {question}"},
    ]


def _span(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{items[0]}–{items[-1]}"


def short_label(chunk: dict) -> str:
    """A citation label: "Article 22(1)–(3)", "Tenth Schedule, para 2", "Seventh Schedule,
    List II, entries 1–10"."""
    kind, prov, src = chunk["kind"], chunk["provision"], chunk["source"]
    clauses = chunk.get("clauses") if chunk.get("chunk_count", 1) > 1 else None
    if kind == "preamble":
        return "Preamble"
    if kind == "article":
        if clauses:
            return f"Article {prov['number']}{clauses[0]}" + (f"–{clauses[-1]}" if len(clauses) > 1 else "")
        return f"Article {prov['number']}"
    if src.get("schedule"):
        name = src["schedule_name"]
        if kind == "schedule_paragraph":
            return f"{name}, para {prov['number']}"
        if kind == "schedule_form":
            return f"{name}, {(prov.get('title') or 'Form ' + prov['number']).split(':')[0]}"
        if kind == "schedule_entries":
            section = src.get("section", "")
            list_name = re.match(r"(List I{1,3})\b", section)
            numbers = prov.get("numbers") or []
            entries = (f"entry {numbers[0]}" if len(set(numbers)) == 1 else f"entries {_span(numbers)}") if numbers else ""
            return ", ".join(x for x in (name, list_name.group(1) if list_name else "", entries) if x)
        return name
    if src.get("appendix"):
        name = f"Appendix {ROMAN.get(src['appendix'], src['appendix'])}"
        if kind == "appendix_section":
            word = "para" if src["appendix"] == 2 else "section"
            return f"{name}, {word} {prov['number']}"
        return name
    return chunk["header"].split(" › ")[-1]


def cited_numbers(answer: str, count: int) -> list[int]:
    """Passage numbers the answer cites, in order of first use, ignoring numbers out of range."""
    seen: list[int] = []
    for m in CITATION.finditer(answer):
        for n in (int(x) for x in m.group(1).split(",")):
            if 1 <= n <= count and n not in seen:
                seen.append(n)
    return seen


def suggest_followups(question: str, answer: str, llm=None) -> list[str]:
    """Three follow-up questions; [] if the LLM fails or answers in the wrong format."""
    llm = llm or get_llm()
    plain_answer = CITATION.sub("", answer)
    try:
        text = llm.chat([
            {"role": "system", "content": FOLLOWUP_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nAnswer: {plain_answer}"},
        ])
    except LLMError:
        return []
    questions = []
    for line in text.splitlines():
        q = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"“”')
        if q.endswith("?") and 8 <= len(q) <= 160 and q not in questions:
            questions.append(q)
    return questions[:3]


def ask(question: str, searcher, llm=None, history=None) -> Iterator[dict]:
    """Retrieve, answer (streamed), then suggest follow-ups. See the module docstring."""
    llm = llm or get_llm()
    history = clean_history(history)
    yield {"type": "status", "stage": "searching"}
    query = refine_question(question, llm, history)
    passages = searcher.retrieve(query)
    yield {"type": "sources", "query": query, "sources": [
        {"n": i, "id": p["id"], "label": short_label(searcher.chunks[p["id"]]),
         "title": p["header"].split(" › ")[-1], "kind": searcher.chunks[p["id"]]["kind"]}
        for i, p in enumerate(passages, 1)
    ]}

    yield {"type": "status", "stage": "answering"}
    answer = ""
    try:
        for piece in llm.chat(build_messages(question, passages, history), stream=True):
            answer += piece
            yield {"type": "token", "text": piece}
    except LLMError as e:
        message = ("The AI service is busy right now. Please try again in a minute." if e.retryable
                   else "The AI service could not answer this question.")
        yield {"type": "error", "message": message, "detail": str(e), "retryable": e.retryable}
        return

    answered_by = llm.used_model  # the follow-up call may fall back to another model
    yield {"type": "citations", "cited": cited_numbers(answer, len(passages))}
    yield {"type": "followups", "questions": suggest_followups(question, answer, llm)}
    yield {"type": "done", "model": answered_by}


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: rag-ask "your question"')
        return
    from ..index.search import HybridSearcher

    question = " ".join(sys.argv[1:])
    cfg = settings()
    print(f"LLM: {cfg.llm.provider} / {cfg.llm.model}   embedding: {cfg.embedding.model}   ({cfg.source})\n")
    searcher = HybridSearcher()
    start = time.time()
    sources: list[dict] = []
    for event in ask(question, searcher):
        kind = event["type"]
        if kind == "sources":
            sources = event["sources"]
            print(f"Search: {event['query']}\n\nPassages:")
            for s in sources:
                print(f"  [{s['n']}] {s['label']}: {s['title']}")
            print()
        elif kind == "token":
            print(event["text"], end="", flush=True)
        elif kind == "citations":
            labels = [s["label"] for s in sources if s["n"] in event["cited"]]
            print("\n\nCited: " + (", ".join(labels) or "none"))
        elif kind == "followups":
            print("Follow-up questions:")
            for q in event["questions"]:
                print(f"  - {q}")
        elif kind == "done":
            print(f"\n({event['model']}, {time.time() - start:.0f}s)")
        elif kind == "error":
            print(f"\nerror: {event['message']}\n{event['detail']}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
