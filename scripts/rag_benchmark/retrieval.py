"""Canonical BM25, dense, Hybrid, and HyDE retrieval primitives."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .data import CorpusChunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[RetrievedChunk]: ...


class TextEncoder(Protocol):
    model_name: str

    def encode(self, texts: Sequence[str]) -> Any: ...


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def corpus_fingerprint(chunks: Sequence[CorpusChunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return normalized or "model"


class SentenceTransformerEncoder:
    """Lazy local encoder; importing this module never imports ML libraries."""

    def __init__(self, model_name: str, revision: str, device: str = "cpu", batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers to build the dense retrieval index") from exc
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, revision=revision, device=device)

    def encode(self, texts: Sequence[str]) -> Any:
        return self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self.batch_size,
        )


class DenseRetriever:
    """Exact cosine retrieval over a local LanceDB table with normalized vectors."""

    def __init__(self, table: Any, encoder: TextEncoder) -> None:
        self._table = table
        self._encoder = encoder

    @classmethod
    def load_or_build(
        cls,
        chunks: Sequence[CorpusChunk],
        index_dir: Path,
        model_name: str,
        revision: str,
        device: str = "cpu",
        batch_size: int = 32,
        encoder: TextEncoder | None = None,
    ) -> "DenseRetriever":
        try:
            import lancedb
        except ImportError as exc:
            raise RuntimeError(
                "Install lancedb to use persistent dense retrieval (for example: pip install lancedb)"
            ) from exc

        encoder = encoder or SentenceTransformerEncoder(model_name, revision, device=device, batch_size=batch_size)
        fingerprint = corpus_fingerprint(chunks)
        table_name = "chunks_" + "_".join(
            (safe_name(model_name), safe_name(revision), fingerprint[:16])
        )
        database = lancedb.connect(str(index_dir))

        try:
            table = database.open_table(table_name)
        except (FileNotFoundError, KeyError, ValueError):
            vectors = _normalized_vectors(encoder.encode([chunk.text for chunk in chunks]))
            if len(vectors) != len(chunks):
                raise ValueError("Dense encoder returned an invalid corpus matrix")
            dimension = len(vectors[0]) if vectors else 0
            if any(len(vector) != dimension for vector in vectors):
                raise ValueError("Dense encoder returned vectors with inconsistent dimensions")
            rows = [
                {
                    "vector": vector,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True, allow_nan=False),
                    "corpus_sha256": fingerprint,
                    "embedding_model": model_name,
                    "embedding_revision": revision,
                    "embedding_dimension": dimension,
                    "normalized": True,
                }
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            table = database.create_table(table_name, data=rows)
        return cls(table, encoder)

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if k <= 0:
            return []
        vector = _normalized_vectors(self._encoder.encode([query]))[0]
        # No vector index is created for these benchmark tables, so LanceDB performs
        # exhaustive exact search rather than ANN retrieval.
        rows = self._table.search(vector).distance_type("cosine").limit(k).to_list()
        return [
            RetrievedChunk(
                chunk_id=str(row["chunk_id"]),
                text=str(row["text"]),
                score=-float(row["_distance"]),
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]


def _normalized_vectors(vectors: Any) -> list[list[float]]:
    """Convert encoder output to finite, unit-length vectors for cosine search."""

    normalized: list[list[float]] = []
    for raw_vector in vectors:
        vector = [float(value) for value in raw_vector]
        magnitude_squared = sum(value * value for value in vector)
        if not vector or magnitude_squared <= 0:
            raise ValueError("Dense encoder returned an empty or zero-length vector")
        magnitude = magnitude_squared ** 0.5
        normalized.append([value / magnitude for value in vector])
    return normalized


class BM25Retriever:
    """Okapi BM25 retrieval over the immutable corpus snapshot."""

    def __init__(self, chunks: Sequence[CorpusChunk]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("Install rank-bm25 to use Hybrid retrieval") from exc
        self._chunks = list(chunks)
        self._index = BM25Okapi([tokenize_for_bm25(chunk.text) for chunk in chunks])

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if k <= 0:
            return []
        scores = self._index.get_scores(tokenize_for_bm25(query))
        ranked = sorted(
            range(len(self._chunks)),
            key=lambda index: (-float(scores[index]), self._chunks[index].chunk_id),
        )[:k]
        return [
            RetrievedChunk(
                chunk_id=self._chunks[index].chunk_id,
                text=self._chunks[index].text,
                score=float(scores[index]),
                metadata=self._chunks[index].metadata,
            )
            for index in ranked
        ]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int,
    rank_constant: int = 60,
) -> list[tuple[str, float]]:
    """Fuse independent rankings with deterministic RRF and chunk-ID tie breaks."""

    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank_constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]


class HybridRetriever:
    """Canonical BM25 + dense retrieval combined only by rank fusion."""

    def __init__(self, chunks: Sequence[CorpusChunk], bm25: Retriever, dense: Retriever) -> None:
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._bm25 = bm25
        self._dense = dense

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        candidate_k = min(len(self._chunks), max(k * 4, 50))
        bm25_results = self._bm25.search(query, candidate_k)
        dense_results = self._dense.search(query, candidate_k)
        fused = reciprocal_rank_fusion(
            [
                [result.chunk_id for result in bm25_results],
                [result.chunk_id for result in dense_results],
            ],
            k=k,
        )
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=self._chunks[chunk_id].text,
                score=score,
                metadata=self._chunks[chunk_id].metadata,
            )
            for chunk_id, score in fused
        ]
