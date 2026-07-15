"""Benchmark orchestration with resumable per-sample persistence."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from .data import BenchmarkSample, CorpusChunk, DatasetMode, load_corpus, load_dataset, validate_reference_chunks
from .generation import HYDE_INSTRUCTION, GenerationSettings, HuggingFaceGenerator, OpenAIHydeGenerator
from .hyde import HypotheticalRecord, prepare_hypothetical_documents
from .metrics import METRIC_NAMES, RagasEvaluator
from .model_registry import MODEL_REGISTRY, resolve_adapter_source
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever, RetrievedChunk, Retriever, safe_name
from .telemetry import combine_system_measurements


Strategy = Literal["no_rag", "hybrid", "hyde"]


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset_mode: DatasetMode
    strategy: Strategy
    model_key: str
    sample10_path: Path
    maternaqa_path: Path
    corpus_path: Path
    index_dir: Path
    output_dir: Path
    retrieval_k: int
    retrieval_embedding_model: str
    retrieval_embedding_revision: str
    retrieval_device: str
    retrieval_batch_size: int
    hyde_generator_model: str
    hyde_provider: Literal["openai", "huggingface"]
    evaluator_model: str
    embedding_model: str
    evaluator_max_completion_tokens: int
    evaluator_timeout_seconds: int
    generation_settings: GenerationSettings
    hyde_generation_settings: GenerationSettings
    adapter_path: Path | None = None
    load_in_4bit: bool = True
    hyde_load_in_4bit: bool = True
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    resume: bool = True
    limit: int | None = None


def load_and_validate_data(config: BenchmarkConfig) -> tuple[list[BenchmarkSample], list[CorpusChunk]]:
    samples = load_dataset(config.dataset_mode, config.sample10_path, config.maternaqa_path)
    if config.limit is not None:
        samples = samples[: config.limit]
    corpus = load_corpus(config.corpus_path)
    if config.dataset_mode == "maternaqa_test" and config.limit is None:
        validate_reference_chunks(samples, corpus)
    elif config.dataset_mode == "maternaqa_test":
        corpus_ids = {chunk.chunk_id for chunk in corpus}
        missing = sorted(
            sample.reference_chunk_id
            for sample in samples
            if sample.reference_chunk_id and sample.reference_chunk_id not in corpus_ids
        )
        if missing:
            raise ValueError(f"Corpus is missing reference chunk IDs: {missing[:10]}")
    return samples, corpus


def retrieve_for_strategy(
    strategy: Strategy,
    question: str,
    retrieval_k: int,
    hybrid_retriever: Retriever | None,
    dense_retriever: Retriever | None,
    hypothetical_text: str | None = None,
) -> tuple[list[RetrievedChunk], float]:
    if strategy == "no_rag":
        return [], 0.0
    started = time.perf_counter()
    if strategy == "hybrid":
        if hybrid_retriever is None:
            raise ValueError("Hybrid strategy requires a Hybrid retriever")
        results = hybrid_retriever.search(question, retrieval_k)
    elif strategy == "hyde":
        if dense_retriever is None or hypothetical_text is None:
            raise ValueError("HyDE strategy requires a hypothetical document and dense retriever")
        results = dense_retriever.search(hypothetical_text, retrieval_k)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    return results, time.perf_counter() - started


def _hyde_cache_path(config: BenchmarkConfig) -> Path:
    identity = _hyde_identity(config)
    canonical_identity = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    fingerprint = hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()[:12]
    return (
        config.index_dir
        / "hyde_cache"
        / config.dataset_mode
        / safe_name(f"{config.hyde_provider}-{config.hyde_generator_model}")
        / f"{fingerprint}.jsonl"
    )


def _experiment_configuration(config: BenchmarkConfig) -> dict[str, Any]:
    dataset_path = config.sample10_path if config.dataset_mode == "sample10" else config.maternaqa_path
    with dataset_path.open("rb") as source:
        dataset_sha = hashlib.file_digest(source, "sha256").hexdigest()
    with config.corpus_path.open("rb") as source:
        corpus_sha = hashlib.file_digest(source, "sha256").hexdigest()
    spec = MODEL_REGISTRY[config.model_key]
    adapter_source = resolve_adapter_source(spec, config.adapter_path)
    provenance = "explicit_local" if config.adapter_path else ("registry_remote" if adapter_source else "base_model")
    return {
        "dataset": {"mode": config.dataset_mode, "path": str(dataset_path.resolve()), "sha256": dataset_sha, "limit": config.limit},
        "corpus": {"path": str(config.corpus_path.resolve()), "sha256": corpus_sha}, "strategy": config.strategy,
        "retrieval": {"backend": "lancedb_exact", "k": config.retrieval_k, "model": config.retrieval_embedding_model, "revision": config.retrieval_embedding_revision, "device": config.retrieval_device, "batch_size": config.retrieval_batch_size},
        "hyde": _hyde_identity(config) if config.strategy == "hyde" else None,
        "answer": {"key": config.model_key, "model_id": spec.model_id, "base_model_id": spec.base_model_id, "adapter_source": adapter_source, "adapter_provenance": provenance, "generation": asdict(config.generation_settings), "load_in_4bit": config.load_in_4bit, "trust_remote_code": config.trust_remote_code, "attn_implementation": config.attn_implementation},
        "evaluator": {"model": config.evaluator_model, "embedding_model": config.embedding_model, "max_completion_tokens": config.evaluator_max_completion_tokens, "timeout_seconds": config.evaluator_timeout_seconds},
    }


def _hyde_identity(config: BenchmarkConfig) -> dict[str, Any]:
    """All inputs that can change a cached hypothetical document."""

    identity: dict[str, Any] = {
        "provider": config.hyde_provider,
        "model": config.hyde_generator_model,
        "prompt": HYDE_INSTRUCTION,
    }
    if config.hyde_provider == "huggingface":
        identity.update(
            {
                "generation": asdict(config.hyde_generation_settings),
                "load_in_4bit": config.hyde_load_in_4bit,
                "trust_remote_code": config.trust_remote_code,
                "attn_implementation": config.attn_implementation,
            }
        )
    else:
        settings = config.hyde_generation_settings
        if settings.repetition_penalty is not None or settings.no_repeat_ngram_size:
            raise ValueError("OpenAI HyDE does not support repetition_penalty or no_repeat_ngram_size")
        identity["generation"] = {
            "max_output_tokens": settings.max_new_tokens,
            "temperature": settings.temperature if settings.do_sample else None,
        }
    return identity


def _output_paths(config: BenchmarkConfig, identity: dict[str, Any]) -> tuple[Path, Path]:
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
    stem = f"{config.dataset_mode}__{config.strategy}__{config.model_key}__{fingerprint}"
    return config.output_dir / f"{stem}.jsonl", config.output_dir / f"{stem}_summary.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if line_number == len(lines) and not text.endswith(("\n", "\r")):
                path.write_text(text[: -len(line)], encoding="utf-8")
                break
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def _write_summary(path: Path, config: BenchmarkConfig, identity: dict[str, Any], output_jsonl: Path, rows: list[dict[str, Any]]) -> None:
    fingerprint = path.stem.rsplit("__", 1)[-1].removesuffix("_summary")
    summary = {
        "dataset_mode": config.dataset_mode,
        "strategy": config.strategy,
        "model": config.model_key,
        "model_id": MODEL_REGISTRY[config.model_key].model_id,
        "configuration_fingerprint": fingerprint,
        "rows": len(rows),
        "rows_with_error": sum(1 for row in rows if row.get("error") or row.get("metric_errors")),
        "metrics": {name: _mean(rows, name) for name in METRIC_NAMES},
        "operations": {
            key: _mean(rows, key)
            for key in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "retrieval_latency_seconds",
                "generation_latency_seconds",
                "end_to_end_latency_seconds",
                "output_tokens_per_second",
            )
        },
        "configuration": {
            "retrieval_k": config.retrieval_k,
            "retrieval_embedding_model": config.retrieval_embedding_model,
            "hyde_generator_model": config.hyde_generator_model if config.strategy == "hyde" else None,
            "evaluator_model": config.evaluator_model,
            "embedding_model": config.embedding_model,
            "output_jsonl": str(output_jsonl),
            **identity,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


async def run_benchmark(
    config: BenchmarkConfig,
    answer_generator_factory: Callable[[], Any] | None = None,
    hyde_generator_factory: Callable[[], Any] | None = None,
    evaluator_factory: Callable[[], RagasEvaluator] | None = None,
) -> tuple[Path, Path]:
    samples, corpus = load_and_validate_data(config)
    identity = _experiment_configuration(config)
    output_jsonl, summary_path = _output_paths(config, identity)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists() and not config.resume:
        output_jsonl.unlink()
    existing = _read_jsonl(output_jsonl) if config.resume else []
    successful = [row for row in existing if not row.get("error") and not row.get("metric_errors")]
    if successful != existing:
        output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in successful), encoding="utf-8")
    existing = successful
    completed = {str(row["qa_id"]) for row in existing if row.get("qa_id") is not None}

    dense: DenseRetriever | None = None
    hybrid: HybridRetriever | None = None
    if config.strategy in {"hybrid", "hyde"}:
        dense = DenseRetriever.load_or_build(
            chunks=corpus,
            index_dir=config.index_dir,
            model_name=config.retrieval_embedding_model,
            revision=config.retrieval_embedding_revision,
            device=config.retrieval_device,
            batch_size=config.retrieval_batch_size,
        )
    if config.strategy == "hybrid":
        hybrid = HybridRetriever(corpus, BM25Retriever(corpus), dense)

    hypothetical_records: dict[str, HypotheticalRecord] = {}
    if config.strategy == "hyde":
        if hyde_generator_factory is None and config.hyde_provider == "openai":
            hyde_generator_factory = lambda: OpenAIHydeGenerator.from_model(
                model_id=config.hyde_generator_model,
                settings=config.hyde_generation_settings,
            )
        elif hyde_generator_factory is None:
            hyde_generator_factory = lambda: HuggingFaceGenerator.from_base_model(
                model_id=config.hyde_generator_model,
                settings=config.hyde_generation_settings,
                load_in_4bit=config.hyde_load_in_4bit,
                trust_remote_code=config.trust_remote_code,
                attn_implementation=config.attn_implementation,
            )
        hypothetical_records = prepare_hypothetical_documents(
            samples=samples,
            cache_path=_hyde_cache_path(config),
            model_id=config.hyde_generator_model,
            generator_identity=_hyde_identity(config),
            generator_factory=hyde_generator_factory,
        )

    spec = MODEL_REGISTRY[config.model_key]
    answer_generator_factory = answer_generator_factory or (
        lambda: HuggingFaceGenerator.from_answer_spec(
            spec=spec,
            settings=config.generation_settings,
            adapter_path=config.adapter_path,
            load_in_4bit=config.load_in_4bit,
            trust_remote_code=config.trust_remote_code,
            attn_implementation=config.attn_implementation,
        )
    )
    evaluator_factory = evaluator_factory or (
        lambda: RagasEvaluator.from_models(
            evaluator_model=config.evaluator_model,
            embedding_model=config.embedding_model,
            max_completion_tokens=config.evaluator_max_completion_tokens,
            timeout_seconds=config.evaluator_timeout_seconds,
        )
    )
    answer_generator = answer_generator_factory()
    try:
        evaluator = evaluator_factory()
        with output_jsonl.open("a", encoding="utf-8", newline="\n") as output:
            for sample in samples:
                if sample.qa_id in completed:
                    continue
                row: dict[str, Any] = {
                    "qa_id": sample.qa_id,
                    "dataset_mode": config.dataset_mode,
                    "strategy": config.strategy,
                    "model": config.model_key,
                    "model_id": spec.model_id,
                    "configuration_fingerprint": summary_path.stem.rsplit("__", 1)[-1].removesuffix("_summary"),
                    "adapter_source": identity["answer"]["adapter_source"],
                    "adapter_provenance": identity["answer"]["adapter_provenance"],
                    "user_input": sample.question,
                    "reference": sample.reference,
                    "metadata": sample.metadata,
                }
                try:
                    hypothetical = hypothetical_records.get(sample.qa_id)
                    retrieved, retrieval_latency = retrieve_for_strategy(
                        strategy=config.strategy,
                        question=sample.question,
                        retrieval_k=config.retrieval_k,
                        hybrid_retriever=hybrid,
                        dense_retriever=dense,
                        hypothetical_text=hypothetical.text if hypothetical else None,
                    )
                    contexts = [item.text for item in retrieved]
                    answer = answer_generator.answer(sample.question, contexts)
                    row.update(
                        {
                            "response": answer.text,
                            "retrieved_contexts": contexts,
                            "retrieved_chunk_ids": [item.chunk_id for item in retrieved],
                            "retrieved_metadata": [item.metadata for item in retrieved],
                            **combine_system_measurements(
                                answer.measurement,
                                retrieval_latency,
                                hypothetical.measurement if hypothetical else None,
                                hypothetical_cache_hit=hypothetical.cache_hit if hypothetical else False,
                            ),
                        }
                    )
                    if hypothetical:
                        row.update(
                            {
                                "hyde_input_tokens": hypothetical.measurement.input_tokens,
                                "hyde_output_tokens": hypothetical.measurement.output_tokens,
                                "hyde_total_tokens": hypothetical.measurement.total_tokens,
                                "hyde_generation_latency_seconds": hypothetical.measurement.generation_latency_seconds,
                                "hyde_output_tokens_per_second": hypothetical.measurement.output_tokens_per_second,
                                "hyde_cache_hit": hypothetical.cache_hit,
                            }
                        )
                    row.update(
                        await evaluator.score(
                            user_input=sample.question,
                            response=answer.text,
                            retrieved_contexts=contexts,
                            reference=sample.reference,
                            include_context_metrics=config.strategy != "no_rag",
                        )
                    )
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    for metric_name in METRIC_NAMES:
                        row.setdefault(metric_name, None)
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                if not row.get("error") and not row.get("metric_errors"):
                    completed.add(sample.qa_id)
    finally:
        answer_generator.close()

    rows = _read_jsonl(output_jsonl)
    _write_summary(summary_path, config, identity, output_jsonl, rows)
    return output_jsonl, summary_path
