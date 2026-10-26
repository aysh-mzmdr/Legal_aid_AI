"""Rewrite a question in legal terms before searching, so retrieval doesn't depend on wording.

"My school refused my child admission" finds little; "denial of admission of a child to an
educational institution; right of children to free and compulsory education" finds Articles
21A and 29(2). Only the search uses the rewrite; the answer still responds to the original.
Given the chat's earlier messages, a follow-up ("what about my lawyer?") becomes a standalone query.

    uv run python -m rag_model.prompt.refine "Can the government take my land?"
"""

from __future__ import annotations

import re
import sys

from ..llm import LLMError, get_llm

REFINE_PROMPT = """Rewrite the person's question about the law in India in the legal terms the Constitution of India uses, so it can be used to search the Constitution.
- Keep the meaning exactly. Do not add facts, assumptions, conclusions or new questions, and do not drop any part of the question.
- Keep who is acting against whom. A landlord, employer, school or other private party is not "the State"; the government, police and public bodies are.
- Replace everyday words with the legal concepts they refer to.
- Name an Article, Schedule or Part only if the person named it, or the conversation so far names it as what the question refers to.
- If the question is already in legal terms, return it unchanged.
- The conversation so far, if given, is only there to resolve what the new question refers to ("it", "that", "what about my lawyer?"). Make the rewrite a standalone question that names that topic. Do not bring in earlier questions the new one doesn't refer to; if it starts a new topic, ignore the conversation.
- Output only the rewritten question on one line, with no quotes, labels or explanation.

Examples:
Can the government take my land? -> Can the State deprive a person of property or compulsorily acquire it?
The police kept my brother in jail for a week without a trial -> Can a person be detained in custody for a week without trial or production before a magistrate?
My school refused my child admission -> Denial of admission of a child to a school; right of children to education
My landlord is evicting me without notice -> Eviction of a tenant by a private landlord without notice
What does Article 14 say? -> What does Article 14 say?
(after a question about being arrested) what about my lawyer? -> Right of an arrested person to consult and be defended by a legal practitioner of his choice"""

CONTEXT_CHARS = 600  # of each earlier answer: enough to tell what a follow-up refers to


def _with_context(question: str, history: list[dict]) -> str:
    if not history:
        return question
    lines = [f"{'Person' if m['role'] == 'user' else 'Assistant'}: {m['text'][:CONTEXT_CHARS]}" for m in history]
    return "Conversation so far:\n" + "\n".join(lines) + f"\n\nNew question: {question}"


def refine_question(question: str, llm=None, history: list[dict] = ()) -> str:
    """The question in legal terms, standalone given `history` (earlier messages, as from
    answer.clean_history); the original question if the LLM fails or answers oddly."""
    llm = llm or get_llm()
    try:
        text = llm.chat([
            {"role": "system", "content": REFINE_PROMPT},
            {"role": "user", "content": _with_context(question, history)},
        ])
    except LLMError:
        return question
    lines = [line for line in text.strip().splitlines() if line.strip()]
    refined = re.sub(r"^(?:rewritten(?: question)?|(?:new )?question)\s*:\s*", "", lines[0], flags=re.I) if lines else ""
    refined = refined.strip().strip('"“”')
    return refined if 3 <= len(refined) <= 400 else question


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python -m rag_model.prompt.refine "your question"')
        return
    print(refine_question(" ".join(sys.argv[1:])))


if __name__ == "__main__":
    main()
