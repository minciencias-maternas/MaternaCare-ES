"""Reusable hypothetical-document cache independent from answer models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .data import BenchmarkSample
from .generation import GenerationResult
from .telemetry import GenerationMeasurement


class HypotheticalGenerator(Protocol):
    def hypothetical_document(self, question: str) -> GenerationResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class HypotheticalRecord:
    sample_id: str
    question_sha256: str
    model_id: str
    generator_identity: dict[str, Any]
    text: str
    measurement: GenerationMeasurement
    cache_hit: bool


def question_fingerprint(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def load_hyde_cache(
    path: Path, model_id: str, generator_identity: dict[str, Any] | None = None
) -> dict[str, HypotheticalRecord]:
    if not path.exists():
        return {}
    records: dict[str, HypotheticalRecord] = {}
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            row_identity = row.get("generator_identity")
            if row.get("model_id") != model_id or (
                generator_identity is not None and row_identity != generator_identity
            ):
                continue
            measurement = GenerationMeasurement(
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                total_tokens=int(row["total_tokens"]),
                generation_latency_seconds=float(row["generation_latency_seconds"]),
                output_tokens_per_second=float(row["output_tokens_per_second"]),
            )
            records[str(row["sample_id"])] = HypotheticalRecord(
                sample_id=str(row["sample_id"]),
                question_sha256=str(row["question_sha256"]),
                model_id=model_id,
                generator_identity=dict(row_identity or {"model": model_id}),
                text=str(row["text"]),
                measurement=measurement,
                cache_hit=True,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, json.JSONDecodeError) and line_number == len(lines) and not text.endswith(("\n", "\r")):
                path.write_text(text[: -len(line)], encoding="utf-8")
                break
            raise ValueError(f"Invalid HyDE cache row {line_number} in {path}") from exc
    return records


def prepare_hypothetical_documents(
    samples: Sequence[BenchmarkSample],
    cache_path: Path,
    model_id: str,
    generator_identity: dict[str, Any],
    generator_factory: Callable[[], HypotheticalGenerator],
) -> dict[str, HypotheticalRecord]:
    """Generate only cache misses using a dedicated generator factory."""

    cached = load_hyde_cache(cache_path, model_id, generator_identity)
    records: dict[str, HypotheticalRecord] = {}
    missing: list[BenchmarkSample] = []
    for sample in samples:
        record = cached.get(sample.qa_id)
        if record is not None and record.question_sha256 == question_fingerprint(sample.question):
            records[sample.qa_id] = record
        else:
            missing.append(sample)

    if not missing:
        return records

    generator = generator_factory()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with cache_path.open("a", encoding="utf-8", newline="\n") as output:
            for sample in missing:
                generated = generator.hypothetical_document(sample.question)
                record = HypotheticalRecord(
                    sample_id=sample.qa_id,
                    question_sha256=question_fingerprint(sample.question),
                    model_id=model_id,
                    generator_identity=generator_identity,
                    text=generated.text,
                    measurement=generated.measurement,
                    cache_hit=False,
                )
                row = {
                    "sample_id": record.sample_id,
                    "question_sha256": record.question_sha256,
                    "model_id": record.model_id,
                    "generator_identity": record.generator_identity,
                    "text": record.text,
                    **record.measurement.to_dict(),
                }
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                records[sample.qa_id] = record
    finally:
        generator.close()
    return records
