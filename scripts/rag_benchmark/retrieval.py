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
    """Cosine-similarity retrieval over a persistent normalized embedding matrix."""

    def __init__(self, chunks: Sequence[CorpusChunk], embeddings: Any, encoder: TextEncoder) -> None:
        self._chunks = list(chunks)
        self._embeddings = embeddings
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
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install numpy to use dense retrieval") from exc

        encoder = encoder or SentenceTransformerEncoder(model_name, revision, device=device, batch_size=batch_size)
        model_dir = index_dir / safe_name(model_name) / safe_name(revision)
        manifest_path = model_dir / "manifest.json"
        embeddings_path = model_dir / "embeddings.npy"
        fingerprint = corpus_fingerprint(chunks)
        expected_ids = [chunk.chunk_id for chunk in chunks]

        embeddings = None
        if manifest_path.exists() and embeddings_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("model_name") == model_name
                and manifest.get("revision") == revision
                and manifest.get("corpus_sha256") == fingerprint
                and manifest.get("chunk_ids") == expected_ids
            ):
                candidate = np.load(embeddings_path, allow_pickle=False)
                if candidate.shape[0] == len(chunks):
                    embeddings = candidate

        if embeddings is None:
            model_dir.mkdir(parents=True, exist_ok=True)
            embeddings = np.asarray(encoder.encode([chunk.text for chunk in chunks]), dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
                raise ValueError("Dense encoder returned an invalid corpus matrix")
            np.save(embeddings_path, embeddings, allow_pickle=False)
            manifest_path.write_text(
                json.dumps(
                    {
                        "model_name": model_name,
                        "revision": revision,
                        "corpus_sha256": fingerprint,
                        "chunk_ids": expected_ids,
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        return cls(chunks, embeddings, encoder)

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if k <= 0:
            return []
        vector = self._encoder.encode([query])[0]
        scores = self._embeddings @ vector
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
