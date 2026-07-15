"""Canonical benchmark dataset and retrieval-corpus loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


DatasetMode = Literal["sample10", "maternaqa_test"]
JSONL_REQUIRED_FIELDS = frozenset({"qa_id", "pregunta", "respuesta"})
MATERNAQA_REQUIRED_FIELDS = JSONL_REQUIRED_FIELDS | frozenset(
    {"contexto_fuente", "chunk_id", "source_pdf", "pages", "section"}
)


@dataclass(frozen=True)
class BenchmarkSample:
    """Normalized single-turn benchmark sample."""

    qa_id: str
    question: str
    reference: str
    source_context: str = ""
    reference_chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CorpusChunk:
    """Retrievable corpus unit."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_jsonl_dataset(
    path: Path,
    *,
    dataset_name: str = "dataset",
    expected_rows: int | None = None,
    required_fields: frozenset[str] = JSONL_REQUIRED_FIELDS,
) -> list[BenchmarkSample]:
    """Load the canonical benchmark JSONL schema into normalized samples."""

    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} dataset not found: {path}")

    samples: list[BenchmarkSample] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{dataset_name} row {line_number} must be a JSON object")
        missing = required_fields - row.keys()
        if missing:
            raise ValueError(f"{dataset_name} row {line_number} is missing: {sorted(missing)}")
        null_fields = sorted(field for field in required_fields if row[field] is None)
        if null_fields:
            raise ValueError(f"{dataset_name} row {line_number} has null required fields: {null_fields}")

        qa_id = str(row["qa_id"]).strip()
        question = str(row["pregunta"]).strip()
        reference = str(row["respuesta"]).strip()
        if not qa_id or not question or not reference:
            raise ValueError(f"{dataset_name} row {line_number} has an empty qa_id, pregunta, or respuesta")
        if qa_id in seen:
            raise ValueError(f"Duplicate {dataset_name} qa_id: {qa_id}")
        seen.add(qa_id)

        raw_chunk_id = row.get("chunk_id")
        reference_chunk_id = str(raw_chunk_id).strip() if raw_chunk_id is not None else None
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"qa_id", "pregunta", "respuesta", "contexto_fuente"}
        }
        samples.append(
            BenchmarkSample(
                qa_id=qa_id,
                question=question,
                reference=reference,
                source_context=str(row.get("contexto_fuente") or "").strip(),
                reference_chunk_id=reference_chunk_id or None,
                metadata=metadata,
            )
        )

    if expected_rows is not None and len(samples) != expected_rows:
        raise ValueError(f"{dataset_name} must contain exactly {expected_rows} rows, found {len(samples)}")
    return samples


def load_sample10(path: Path) -> list[BenchmarkSample]:
    """Load the canonical ten question/reference pairs from JSONL."""

    return load_jsonl_dataset(path, dataset_name="sample10", expected_rows=10)


def load_maternaqa_test(path: Path) -> list[BenchmarkSample]:
    """Load the canonical 328-row flat MaternaQA test split from JSONL."""

    return load_jsonl_dataset(
        path,
        dataset_name="maternaqa_test",
        expected_rows=328,
        required_fields=MATERNAQA_REQUIRED_FIELDS,
    )


def load_dataset(mode: DatasetMode, sample10_path: Path, maternaqa_path: Path) -> list[BenchmarkSample]:
    if mode == "sample10":
        return load_sample10(sample10_path)
    if mode == "maternaqa_test":
        return load_maternaqa_test(maternaqa_path)
    raise ValueError(f"Unsupported dataset mode: {mode}")


def load_corpus(path: Path) -> list[CorpusChunk]:
    """Load retrieval chunks without deriving content from evaluation rows."""

    if not path.exists():
        raise FileNotFoundError(f"Retrieval corpus not found: {path}")
    chunks: list[CorpusChunk] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        chunk_id = str(row.get("chunk_id") or "")
        text = str(row.get("text") or "").strip()
        if not chunk_id or not text:
            raise ValueError(f"Corpus row {line_number} requires chunk_id and text")
        if chunk_id in seen:
            raise ValueError(f"Duplicate corpus chunk_id: {chunk_id}")
        seen.add(chunk_id)
        metadata = {key: value for key, value in row.items() if key != "text"}
        chunks.append(CorpusChunk(chunk_id=chunk_id, text=text, metadata=metadata))
    return chunks


def validate_reference_chunks(samples: list[BenchmarkSample], corpus: list[CorpusChunk]) -> None:
    """Prove every canonical test reference chunk is available for retrieval."""

    references = {sample.reference_chunk_id for sample in samples if sample.reference_chunk_id}
    if references and len(references) != 108:
        raise ValueError(f"maternaqa_test must reference exactly 108 chunks, found {len(references)}")
    corpus_ids = {chunk.chunk_id for chunk in corpus}
    missing = sorted(references - corpus_ids)
    if missing:
        raise ValueError(f"Corpus is missing {len(missing)} reference chunk IDs: {missing[:10]}")
