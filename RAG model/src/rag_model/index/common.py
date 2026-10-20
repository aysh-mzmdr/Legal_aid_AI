"""Shared settings for building and querying the indexes.

Everything runs offline (see rag_model/__init__.py, which sets the switches before anything
else is imported).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..config import settings

ROOT = Path(__file__).resolve().parents[3]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
BM25_PATH = INDEX_DIR / "bm25.json"
COLLECTION = "constitution"

# Set in config.toml [embedding]. 0.6B is the Qwen3-Embedding size that fits a 4 GB GPU.
_EMB = settings().embedding
MODEL_NAME = _EMB.model
MAX_SEQ_LENGTH = _EMB.max_seq_length
# Qwen3-Embedding works best with a task instruction on queries (documents get none).
QUERY_PROMPT = f"Instruct: {_EMB.query_instruction}\nQuery:"


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunks_digest(path: Path = CHUNKS_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def document_text(chunk: dict) -> str:
    """What gets embedded and keyword-indexed: the heading trail carries Part, Article
    number and title, which the text alone often lacks."""
    return f"{chunk['header']}\n\n{chunk['text']}"


def keyword_text(chunk: dict) -> str:
    """What BM25 indexes. Same as document_text, except table rows lose their repeated column
    labels: every Appendix I row says "Police station", which would match any question about
    the police."""
    text = chunk["text"]
    if chunk["kind"] == "appendix_rows":
        text = re.sub(r"(^|\| )[^:|\n]+: ", r"\1", text, flags=re.M)
    return f"{chunk['header']}\n\n{text}"


def chroma_client():
    """Local on-disk ChromaDB with telemetry off."""
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def load_model(dtype: str | None = None, device: str | None = None):
    """The embedding model on the device and precision set in config.toml [embedding]
    (`device` for building the index; search passes `query_device`)."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or _EMB.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not dtype:
        # fp16 is for GPUs; on the CPU use query_dtype (bfloat16 halves RAM, which matters
        # when Ollama is also holding the LLM in this machine's 8 GB).
        dtype = _EMB.dtype if device.startswith("cuda") else _EMB.query_dtype
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


# BM25Okapi's fitted statistics. The file stores these instead of a pickle (loading a pickle
# can run code); load_bm25 puts them back without re-reading the corpus. Attribute names
# are rank_bm25's own (version pinned in uv.lock).
_BM25_FIELDS = ("k1", "b", "epsilon", "corpus_size", "avgdl", "doc_len", "doc_freqs", "idf", "average_idf")


def build_bm25(chunks: list[dict]):
    """(chunk ids, BM25Okapi) over keyword_text, in the order of `chunks`."""
    from rank_bm25 import BM25Okapi

    return [c["id"] for c in chunks], BM25Okapi([tokenize(keyword_text(c)) for c in chunks])


def save_bm25(ids: list[str], bm25, digest: str, path: Path = BM25_PATH) -> None:
    data = {"chunks_sha256": digest, "ids": ids, **{f: getattr(bm25, f) for f in _BM25_FIELDS}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def load_bm25(digest: str, path: Path = BM25_PATH):
    """(chunk ids, BM25Okapi) from `path`, or None if it is missing or was built from a
    different chunks.jsonl."""
    from rank_bm25 import BM25Okapi

    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("chunks_sha256") != digest:
        return None
    bm25 = BM25Okapi.__new__(BM25Okapi)
    for f in _BM25_FIELDS:
        setattr(bm25, f, data[f])
    bm25.tokenizer = None
    return data["ids"], bm25
