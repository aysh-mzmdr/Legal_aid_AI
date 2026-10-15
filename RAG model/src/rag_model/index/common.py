"""Shared settings for building and querying the indexes."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
COLLECTION = "constitution"

# 0.6B is the size that fits a 4 GB GPU (1.2 GB in fp16); the 4B/8B models need 8-16 GB.
MODEL_NAME = os.environ.get("RAG_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
MAX_SEQ_LENGTH = 1024  # the longest chunk (header + text) is well under this
# Qwen3-Embedding works best with a task instruction on queries (documents get none).
QUERY_PROMPT = ("Instruct: Given a question about legal rights or the law in India, retrieve the "
                "provisions of the Constitution of India that answer it\nQuery:")


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunks_digest(path: Path = CHUNKS_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def document_text(chunk: dict) -> str:
    """What gets embedded and keyword-indexed: the heading trail carries Part, Article
    number and title, which the text alone often lacks."""
    return f"{chunk['header']}\n\n{chunk['text']}"


def load_model(dtype: str = "float16"):
    """Qwen3-Embedding on the GPU when there is one (fp16), else on the CPU (fp32)."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = os.environ.get("RAG_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        dtype = "float32"  # fp16 matmuls are slow or unsupported on CPU
    model = SentenceTransformer(MODEL_NAME, device=device,
                                model_kwargs={"torch_dtype": getattr(torch, dtype)})
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


# Legal function words plus the conversational filler of user questions ("I", "me", "won't",
# "what are my"): left in, they match the oath forms' "I, A.B., ... will not ... reveal".
_STOPWORDS = set("""a an the and or of in on to for by with as at from into under over any such this that these
those be is are was were been being shall may must will would can could should it its he his him she her they
their them which who whom whose what when where how than then there here so not no nor but if all each every
other same said i me my mine myself you your yours we us our ours do does did done doing have has had having
get got want wants why won t don doesn didn isn aren wasn can cannot couldn tell told please also just about
someone something anything""".split())


def tokenize(text: str) -> list[str]:
    """BM25 tokens: lower-case words and numbers ("21a" stays one token), stopwords dropped,
    with light suffix stripping so "arrested" matches "arrest"."""
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in _STOPWORDS:
            continue
        if t.isalpha() and len(t) > 5:
            for suffix in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s"):
                if t.endswith(suffix) and len(t) - len(suffix) >= 4:
                    t = t[: -len(suffix)]
                    break
        out.append(t)
    return out
