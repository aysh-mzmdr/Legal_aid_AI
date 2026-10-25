"""HTTP API for the web app: `uv run rag-serve`.

    GET  /api/health                 {"ready": bool, "error": str | None, "llm": {...}}
    POST /api/chat {"question": ...}  streamed answer, one JSON event per line (see answer.py)
    GET  /api/provisions/{id}        full text of a cited provision, for the source panel

The search models take ~25 s to load. They load in the background at startup, so /api/health
answers at once and reports "ready": false until then; /api/chat returns 503 meanwhile.
"""

from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .prompt.answer import ask, short_label
from .config import settings

cfg = settings()


class _State:
    searcher = None
    error: str | None = None


state = _State()


def _load_searcher() -> None:
    try:
        from .index.search import HybridSearcher

        state.searcher = HybridSearcher()
    except Exception as e:  # reported by /api/health instead of crashing the server
        state.error = f"{type(e).__name__}: {e}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load_searcher, daemon=True).start()
    yield


app = FastAPI(title="Legal Aid AI", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.server.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=cfg.server.max_question_chars)


def _searcher():
    if state.searcher is None:
        raise HTTPException(503, state.error or "The assistant is still starting up. Try again in a moment.")
    return state.searcher


@app.get("/api/health")
def health() -> dict:
    return {
        "ready": state.searcher is not None,
        "error": state.error,
        "llm": {"provider": cfg.llm.provider, "model": cfg.llm.model, "fallbacks": cfg.llm.fallback_models},
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    searcher = _searcher()
    question = req.question.strip()

    def events():
        try:
            for event in ask(question, searcher):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as e:  # anything unexpected still ends the stream with an error event
            yield json.dumps({"type": "error", "message": "Something went wrong while answering.",
                              "detail": f"{type(e).__name__}: {e}", "retryable": False}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/provisions/{chunk_id}")
def provision(chunk_id: str) -> dict:
    chunk = _searcher().chunks.get(chunk_id)
    if chunk is None:
        raise HTTPException(404, f"No provision {chunk_id!r}")
    return {
        "id": chunk["id"],
        "label": short_label(chunk),
        "title": chunk["header"].split(" › ")[-1],
        "trail": chunk["header"].split(" › ")[1:-1],
        "text": chunk["text"],
        "kind": chunk["kind"],
        "pdf_pages": chunk["pdf_pages"],
    }


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)


if __name__ == "__main__":
    main()
