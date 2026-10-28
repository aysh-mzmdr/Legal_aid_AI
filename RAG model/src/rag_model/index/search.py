"""Hybrid search: Qwen3 embeddings (ChromaDB) + BM25 keywords, fused with reciprocal rank
fusion, then widened along the cross-reference graph by query relevance.

    uv run rag-search "Can the police arrest me without telling me why?"
"""

from __future__ import annotations

import json
import re
import sys
import threading
from dataclasses import dataclass

import numpy as np

from .common import (COLLECTION, MANIFEST_PATH, MODEL_NAME, QUERY_PROMPT, build_bm25, chroma_client,
                     chunks_digest, load_bm25, load_chunks, load_model, tokenize)
from ..config import settings
from ..retrieval.expand import expand, explain

RRF_K = 60  # standard constant for reciprocal rank fusion
ARTICLE_REF = re.compile(r"\barticles?\s+(\d+(?:-?[A-Za-z]{1,3})?)\b", re.I)


@dataclass
class Hit:
    id: str
    score: float  # fused score
    dense: float | None = None  # cosine similarity
    bm25: float | None = None
    direct: bool = False  # the query named this Article


class HybridSearcher:
    def __init__(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else None
        if manifest is None:
            raise RuntimeError("No index found. Run `uv run rag-ingest` and then `uv run rag-index`.")
        digest = chunks_digest()
        if manifest["chunks_sha256"] != digest:
            print("warning: chunks.jsonl changed since the index was built; run `uv run rag-index`.",
                  file=sys.stderr)
        for key, now in (("model", MODEL_NAME), ("query_prompt", QUERY_PROMPT)):
            if manifest.get(key) != now:
                print(f"warning: config.toml [embedding] {key} differs from the one the index was built "
                      f"with ({manifest.get(key)!r}); run `uv run rag-index`.", file=sys.stderr)
        self.chunks = {c["id"]: c for c in load_chunks()}
        self.model = load_model(device=settings().embedding.query_device)
        # embedding_function=None: we embed queries ourselves; Chroma's default would download
        # an ONNX model the first time it was used.
        self.collection = chroma_client().get_collection(COLLECTION, embedding_function=None)

        stored = self.collection.get(include=["embeddings"])
        self.ids: list[str] = stored["ids"]
        self.row = {cid: i for i, cid in enumerate(self.ids)}
        self.embeddings = np.asarray(stored["embeddings"], dtype=np.float32)

        stored_bm25 = load_bm25(digest)
        if stored_bm25 is None:
            print("warning: no BM25 index for the current chunks.jsonl; building one in memory. "
                  "Run `uv run rag-index --bm25-only` to store it.", file=sys.stderr)
            stored_bm25 = build_bm25(list(self.chunks.values()))
        self.bm25_ids, self.bm25 = stored_bm25

        # The embedding model and last_explanation are shared; the web server calls retrieve()
        # from several threads.
        self._lock = threading.Lock()
        self.by_article: dict[str, list[str]] = {}
        for c in self.chunks.values():
            if c["kind"] == "article":
                self.by_article.setdefault(c["provision"]["number"].upper(), []).append(c["id"])

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], prompt=QUERY_PROMPT, normalize_embeddings=True,
                                 convert_to_numpy=True)[0].astype(np.float32)

    def dense_ranked(self, q: np.ndarray, pool: int = 30) -> list[tuple[str, float]]:
        """(chunk id, cosine similarity), best first, for an embedded query."""
        dense = self.collection.query(query_embeddings=[q.tolist()], n_results=pool, include=["distances"])
        return [(cid, 1 - d) for cid, d in zip(dense["ids"][0], dense["distances"][0])]

    def sparse_ranked(self, query: str, pool: int = 30) -> list[tuple[str, float]]:
        """(chunk id, BM25 score), best first; chunks sharing no term with the query are left out."""
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:pool]
        return [(self.bm25_ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def search(self, query: str, k: int = 10, pool: int = 30,
               named: bool = True) -> tuple[list[Hit], np.ndarray]:
        """Fused hits; `named=False` turns off putting an Article named in the query first
        (only the retrieval evaluation does that)."""
        q = self.embed_query(query)
        dense_ranked = self.dense_ranked(q, pool)
        sparse_ranked = self.sparse_ranked(query, pool)

        hits: dict[str, Hit] = {}
        for rank, (cid, sim) in enumerate(dense_ranked):
            h = hits.setdefault(cid, Hit(cid, 0.0))
            h.score += 1 / (RRF_K + rank + 1)
            h.dense = sim
        for rank, (cid, s) in enumerate(sparse_ranked):
            h = hits.setdefault(cid, Hit(cid, 0.0))
            h.score += 1 / (RRF_K + rank + 1)
            h.bm25 = s
        # "Article 21A" in the question: that Article goes to the top, scored as if it had
        # ranked first in both lists (so the other results keep a usable relative score).
        for m in ARTICLE_REF.finditer(query) if named else ():
            for cid in self.by_article.get(m.group(1).replace("-", "").upper(), []):
                h = hits.setdefault(cid, Hit(cid, 0.0))
                h.score += 2 / (RRF_K + 1)
                h.direct = True
        ranked = sorted(hits.values(), key=lambda h: -h.score)[:k]
        for h in ranked:
            if h.dense is None:
                h.dense = float(self.embeddings[self.row[h.id]] @ q)
        return ranked, q

    def retrieve(self, query: str) -> list[dict]:
        """Top hybrid hits, widened along cross-references that are relevant to the query.

        Settings come from config.toml [retrieval]. Hits scoring under `min_seed_score` of the
        best hit are not used as seeds. Relevance of a referenced chunk is its cosine similarity
        to the query relative to the best seed's, so `threshold` means "about as on-topic as the
        seeds".
        """
        with self._lock:
            return self._retrieve(query)

    def _retrieve(self, query: str) -> list[dict]:
        cfg = settings().retrieval
        hits, q = self.search(query, k=cfg.seeds)
        if not hits:
            return []
        best = hits[0].score
        hits = [h for h in hits if h.score >= cfg.min_seed_score * best]
        top_sim = max(h.dense for h in hits) or 1.0

        def relevance(cid: str) -> float:
            return float(self.embeddings[self.row[cid]] @ q) / top_sim

        selected = expand([(h.id, h.score) for h in hits], self.chunks, relevance, threshold=cfg.threshold,
                          max_chunks=cfg.max_chunks, max_chars=cfg.max_chars)
        self.last_explanation = explain(selected, self.chunks)
        return [{"id": s.id, "score": round(s.score, 3), "depth": s.depth, "via": s.via, "edge": s.edge,
                 "header": self.chunks[s.id]["header"], "text": self.chunks[s.id]["text"]} for s in selected]


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: rag-search "your question"')
        return
    query = " ".join(sys.argv[1:])
    searcher = HybridSearcher()
    hits, _ = searcher.search(query)
    print(f"\nHybrid search: {query}\n")
    for h in hits:
        bm25 = f"{h.bm25:.1f}" if h.bm25 is not None else "-"
        tag = "  [named in query]" if h.direct else ""
        print(f"  {h.score:.4f}  cos {h.dense:.3f}  bm25 {bm25:>5}  {h.id}{tag}")
    searcher.retrieve(query)
    print("\nWith relevant cross-references:\n")
    print(searcher.last_explanation)


if __name__ == "__main__":
    main()
