"""Embed the chunks with Qwen3-Embedding and store them in ChromaDB, and fit the BM25
keyword index and store it in data/index/bm25.json: `uv run rag-index`.

`uv run rag-index --bm25-only` rebuilds just the keyword index (seconds instead of ~14 min),
for after a change to tokenize() or keyword_text() alone.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .common import (BM25_PATH, CHROMA_DIR, CHUNKS_PATH, COLLECTION, MANIFEST_PATH, MAX_SEQ_LENGTH,
                     MODEL_NAME, QUERY_PROMPT, build_bm25, chroma_client, chunks_digest, document_text,
                     load_chunks, load_model, save_bm25)

BATCH_SIZE = 8


def _metadata(chunk: dict) -> dict:
    """Chroma metadata must be flat str/int/float/bool values (no None, no lists)."""
    src, prov = chunk["source"], chunk["provision"]
    meta = {
        "kind": chunk["kind"],
        "header": chunk["header"],
        "chunk_index": chunk["chunk_index"],
        "chunk_count": chunk["chunk_count"],
        "pdf_pages": ",".join(str(p) for p in chunk["pdf_pages"]),
        "number": prov.get("number") or "",
        "part": src.get("part", ""),
        "chapter": src.get("chapter", ""),
        "schedule": src.get("schedule", 0),
        "appendix": src.get("appendix", 0),
        "section": src.get("section", ""),
    }
    return {k: v for k, v in meta.items() if v is not None}


def build_keyword_index(chunks: list[dict]) -> dict:
    t0 = time.time()
    ids, bm25 = build_bm25(chunks)
    save_bm25(ids, bm25, chunks_digest(CHUNKS_PATH))
    return {"path": BM25_PATH.name, "chunks": len(ids), "vocabulary": len(bm25.idf),
            "seconds": round(time.time() - t0, 2)}


def build() -> dict:
    import torch

    chunks = load_chunks()
    bm25 = build_keyword_index(chunks)
    texts = [document_text(c) for c in chunks]

    t0 = time.time()
    model = load_model()
    device = str(model.device)
    lengths = [len(model.tokenizer(t)["input_ids"]) for t in texts]
    too_long = [chunks[i]["id"] for i, n in enumerate(lengths) if n > MAX_SEQ_LENGTH]
    emb = model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True).astype(np.float32)

    # fp16 can overflow on long inputs (GTX 16-series cards are known for it): redo any
    # row that came out as NaN in fp32.
    bad = np.where(~np.isfinite(emb).all(axis=1))[0]
    if len(bad):
        del model
        torch.cuda.empty_cache()
        model = load_model("float32")
        emb[bad] = model.encode([texts[i] for i in bad], batch_size=BATCH_SIZE,
                                normalize_embeddings=True, convert_to_numpy=True)
    embed_seconds = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chroma_client()
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}},
                                          embedding_function=None)
    for start in range(0, len(chunks), 256):
        part = slice(start, start + 256)
        collection.add(
            ids=[c["id"] for c in chunks[part]],
            embeddings=emb[part].tolist(),
            documents=[c["text"] for c in chunks[part]],
            metadatas=[_metadata(c) for c in chunks[part]],
        )

    manifest = {
        "model": MODEL_NAME,
        "device": device,
        "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "max_seq_length": MAX_SEQ_LENGTH,
        "dimensions": int(emb.shape[1]),
        "chunks": len(chunks),
        "chunks_sha256": chunks_digest(CHUNKS_PATH),
        "query_prompt": QUERY_PROMPT,
        "max_tokens": max(lengths),
        "chunks_over_max_seq_length": too_long,
        "rows_redone_in_fp32": [chunks[i]["id"] for i in bad],
        "embed_seconds": round(embed_seconds, 1),
        "peak_vram_gb": round(peak_vram, 2),
        "bm25": bm25,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bm25-only", action="store_true", help="rebuild only the BM25 keyword index")
    args = parser.parse_args()
    if args.bm25_only:
        print(json.dumps(build_keyword_index(load_chunks()), indent=2))
    else:
        print(json.dumps(build(), indent=2))


if __name__ == "__main__":
    main()
