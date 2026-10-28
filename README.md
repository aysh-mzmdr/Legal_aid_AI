# Legal Aid AI

A chatbot that explains your rights under the **Constitution of India** in plain language, and cites the exact provision behind every statement.

Ask "the police arrested my brother and won't say why" and it answers from Article 22, with a citation you can click to read the provision's full text, including its amendment history. If the Constitution doesn't answer a question (tenancy, divorce, filing an FIR), it says so instead of guessing.

<!-- Demo GIF: docs/demo.gif -->

## Highlights

- **Answers come only from the source.** Every statement cites a numbered passage, and the model may not use outside knowledge. Questions the Constitution doesn't cover get "I can't answer this from the data I have."
- **Structure-aware parsing.** The 402-page PDF is split into 744 passages along its own structure (Preamble, Articles, Schedules, Appendices). Amendment footnotes are written into the text, and deleted text is removed. The parse is checked against the PDF's table of contents: all 504 Articles are found, in order, with no Part or title mismatches.
- **Hybrid search.** Qwen3 embeddings and BM25 keyword search, combined with reciprocal rank fusion. An Article named in the question ("What does Article 21A say?") is placed first.
- **Query rewriting.** An LLM turns everyday wording into the Constitution's terms before searching: "can the government take my land?" becomes "deprivation or compulsory acquisition of property", which finds Article 300A.
- **Cross-reference expansion.** Provisions the retrieved ones depend on are added when they're relevant to the question: Article 19 brings in Article 358 (its suspension during an Emergency).
- **Follow-up questions.** The last three exchanges are sent as context, so "what about my lawyer?" after a question about arrest is understood.
- **Measured.** A 100-question labelled evaluation set, with an ablation of each search step ([below](#evaluation)).

## How it works

```mermaid
flowchart LR
    Q[Question + recent chat] --> R[Rewrite in legal terms<br/>Gemini]
    R --> S[Hybrid search<br/>Qwen3 + BM25, RRF]
    S --> E[Cross-reference<br/>expansion]
    E --> A[Answer with citations<br/>Gemini, streamed]
    A --> F[Suggested<br/>follow-up questions]
```

**Ingestion (run once):** PDF layout → footnote parsing → structure (Preamble, Articles, Schedules, Appendices) → amendment notes written into the text → passages → cross-reference graph → validation report.

**Per question:**
1. **Rewrite:** the question in the Constitution's legal terms, made standalone using the chat history. Only the search uses it; the answer responds to the question as asked.
2. **Search (local):** dense search (Qwen3-Embedding-0.6B in ChromaDB) and BM25 keyword search, combined with reciprocal rank fusion. The top 5 results are the starting points.
3. **Expansion:** follows the cross-reference graph from those results, keeping only provisions relevant to the question (up to 12 passages).
4. **Answer:** streamed, with every statement cited as `[n]`. Only the passages may be used as a source of law, but their general rules must be applied to the person's situation.
5. **Follow-ups:** three suggested questions for the "Ask next" buttons.

## Evaluation

The search is evaluated on [`questions.jsonl`](RAG%20model/src/rag_model/eval/questions.jsonl): 100 questions, each labelled with the provisions that answer it. The set covers plain-English, Hinglish, legal-wording, hypothetical, broad and out-of-scope questions, with typos left in. The 11 out-of-scope questions have no correct provision, so recall is measured on the other 89.

**recall@5:** the share of questions where a correct provision is in the top 5 results, which are the results the app passes on to the LLM. **MRR:** the average of 1 / (rank of the first correct result), which measures how close to the top it is.

| Setup | Question as asked | After LLM rewrite |
|---|---|---|
| BM25 keyword search only | 60.7% (MRR 0.470) | 92.1% (MRR 0.815) |
| Embeddings only (Qwen3) | 88.8% (MRR 0.807) | 94.4% (MRR 0.851) |
| Hybrid (RRF) + named-Article boost | 88.8% (MRR 0.742) | **96.6% (MRR 0.906)** ← production |

(recall@5, with MRR in brackets. Hybrid without the named-Article boost scores 87.6%, MRR 0.713, on the question as asked.)

Hybrid search with query rewriting raises recall@5 from 88.8% (embeddings only) to 96.6%, cutting misses from 10 to 3. The passages actually sent to the LLM (9.4 on average, after expansion) contain a correct provision for the same 96.6%.

**Findings**
- **Query rewriting is the biggest single gain.** It lifts every setup, and BM25 most of all (60.7% → 92.1%), because the rewrite puts questions into the Constitution's own words.
- **BM25 only pays off after the rewrite.** On the question as asked, adding it to embeddings doesn't help (88.8% either way), because ordinary questions rarely use the Constitution's words. On the rewritten question, it raises recall@5 from 94.4% to 96.6% and MRR from 0.851 to 0.906: keyword matching on exact legal terms puts the right provision higher. The gain is 2 questions out of 89, so it's a small effect on a small set.
- **The remaining misses come from the rewrite step.** It added the wrong terms ("citizen or foreigner" pulled in the citizenship Articles), kept everyday wording unchanged, or used a word the Constitution doesn't use ("insolvent" rather than "public assistance … undeserved want").

Run it with `uv run rag-eval` (see [Commands](#commands)).

## Tech stack

| | |
|---|---|
| Frontend | React 19, Vite, CSS Modules |
| API | FastAPI, NDJSON streaming |
| Retrieval | Qwen3-Embedding-0.6B (sentence-transformers, CUDA fp16), ChromaDB, rank-bm25 |
| LLM | Google Gemini (`gemini-3.5-flash-lite`); Ollama and OpenAI-compatible clients also supported |
| PDF parsing | PyMuPDF |
| Tooling | uv, Python 3.12, ESLint |

## Getting started

**Requirements:** Node.js, Python 3.12 with [uv](https://docs.astral.sh/uv/), a [Gemini API key](https://aistudio.google.com/apikey), and preferably an NVIDIA GPU (4 GB is enough). It also runs on CPU, more slowly.

1. **Clone the repository.** The source PDF is included at `RAG model/Raw Data/Constitution_of_India.pdf` (published by the Legislative Department, Government of India).
2. **Add your API key.** Copy `.env.example` to `.env` in the repository root and fill in your key:
   ```bash
   cp .env.example .env                       # PowerShell: Copy-Item .env.example .env
   ```
   ```
   GEMINI_API_KEY=your-key
   ```
   `.env` is gitignored, so the key stays out of the repository.
3. **Build the index** (run in `RAG model/`):
   ```bash
   uv sync
   uv run rag-ingest                          # PDF -> passages, plus a validation report
   RAG_ALLOW_DOWNLOAD=1 uv run rag-index      # downloads the embedding model the first time; ~15 min on a GTX 1650
   ```
   On Windows PowerShell, set the variable first: `$env:RAG_ALLOW_DOWNLOAD=1; uv run rag-index`.
4. **Start the API:**
   ```bash
   uv run rag-start                           # http://127.0.0.1:8000, ready after ~20 s
   ```
5. **Start the web app** (in the repository root):
   ```bash
   npm install
   npm run dev                                # http://localhost:5173
   ```

### Commands

Run in `RAG model/`:

| Command | What it does |
|---|---|
| `uv run rag-ingest` | Parse the PDF into passages and write a validation report (`data/processed/report.md`) |
| `uv run rag-index` | Embed the passages into ChromaDB and build the BM25 index |
| `uv run rag-search "question"` | Search only, showing the scores and the expansion trace |
| `uv run rag-ask "question"` | Full answer on the command line |
| `uv run rag-eval` | Retrieval evaluation (`--no-rewrite` skips the LLM, `--refresh` redoes the cached rewrites) |
| `uv run rag-start` | Start the API |

All settings are in [`config.toml`](RAG%20model/config.toml). Any of them can be overridden for one run with `RAG_<SECTION>_<KEY>`, such as `RAG_LLM_TEMPERATURE=0`.

## Project structure

```
src/                         React app
  api/chat.js                streaming client for the API
  components/                ChatPage, Sidebar, MessageBubble, RichText, SourcePanel, ChatInput
RAG model/
  config.toml                all settings
  Raw Data/                  the Constitution of India PDF
  src/rag_model/
    ingest/                  PDF -> passages, cross-reference graph, validation report
    index/                   embeddings, ChromaDB, BM25, hybrid search
    retrieval/expand.py      cross-reference expansion
    prompt/                  query rewrite and answer prompts
    eval/                    evaluation set and script
    llm.py                   Gemini / Ollama / OpenAI-compatible clients
    server.py                FastAPI app
```

## Limitations

- **The Constitution only.** Most everyday legal problems (tenancy, consumer complaints, criminal procedure, family law) are governed by other laws, which aren't indexed. For those, the bot says it can't answer.
- **Answers vary between runs.** The same question may be answered fully once and only partly the next time.
- **The evaluation set is small and was labelled by one person.** Use it to compare setups against each other, not as a guarantee of accuracy.
- **Privacy:** search runs locally, but questions, chat history and the retrieved passages are sent to Google's Gemini API. Point `config.toml` at a local model through Ollama to keep everything on your machine.

## Disclaimer

This project is for information and education. It is not legal advice. For a real legal problem, contact a lawyer or your District Legal Services Authority, which offers free legal aid to eligible people.
