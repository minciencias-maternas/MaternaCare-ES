"""Benchmark orchestration with resumable per-sample persistence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from .data import BenchmarkSample, CorpusChunk, DatasetMode, load_corpus, load_dataset, validate_reference_chunks
from .generation import (
    ANSWER_WITH_CONTEXT_INSTRUCTION,
    ANSWER_WITHOUT_CONTEXT_INSTRUCTION,
    HYDE_INSTRUCTION,
    GenerationSettings,
    HuggingFaceGenerator,
    OpenAIHydeGenerator,
)
from .hyde import HypotheticalRecord, prepare_hypothetical_documents
from .metrics import METRIC_NAMES, RagasEvaluator
from .model_registry import MODEL_REGISTRY, resolve_adapter_source
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever, RetrievedChunk, Retriever, safe_name
from .telemetry import ResourceSampler, combine_system_measurements, resolve_cuda_device_identity, resolve_cuda_device_index


Strategy = Literal["no_rag", "hybrid", "hyde"]


def _log(message: str) -> None:
    print(f"[benchmark] {message}", flush=True)


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
    evaluator_concurrency: int
    generation_settings: GenerationSettings
    hyde_generation_settings: GenerationSettings
    adapter_path: Path | None = None
    load_in_4bit: bool = True
    hyde_load_in_4bit: bool = True
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    resume: bool = True
    limit: int | None = None
    telemetry_interval_seconds: float = 0.5


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
        "answer": {"key": config.model_key, "model_id": spec.model_id, "base_model_id": spec.base_model_id, "adapter_source": adapter_source, "adapter_provenance": provenance, "prompt": ANSWER_WITHOUT_CONTEXT_INSTRUCTION if config.strategy == "no_rag" else ANSWER_WITH_CONTEXT_INSTRUCTION, "generation": asdict(config.generation_settings), "load_in_4bit": config.load_in_4bit, "trust_remote_code": config.trust_remote_code, "attn_implementation": config.attn_implementation},
        "evaluator": {"model": config.evaluator_model, "embedding_model": config.embedding_model, "max_completion_tokens": config.evaluator_max_completion_tokens, "timeout_seconds": config.evaluator_timeout_seconds, "concurrency": config.evaluator_concurrency},
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


def _serialize_row(row: dict[str, Any]) -> str:
    def sanitize(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        return value

    finite = sanitize(row)
    try:
        return json.dumps(finite, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        row["error"] = row.get("error") or "serialization: fallback to str coercion"
        row["completion_status"] = "error"
        finite["error"] = row["error"]
        finite["completion_status"] = "error"
        return json.dumps(finite, ensure_ascii=False, allow_nan=False, default=str)


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def _mean_nested(rows: list[dict[str, Any]], parent: str, key: str) -> float | None:
    values = [float(value) for row in rows if isinstance(row.get(parent), dict) if (value := row[parent].get(key)) is not None]
    return statistics.mean(values) if values else None


def _count_context_tokens(generator: Any, contexts: list[str]) -> int | None:
    try:
        counter = getattr(generator, "count_context_tokens", None)
        if not callable(counter):
            return None
        return counter(contexts)
    except Exception:
        return None


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
        "operations_mean": {
            **{
                key: _mean(rows, key)
                for key in ("input_tokens", "output_tokens", "total_tokens", "retrieved_chunk_count", "retrieved_context_tokens", "retrieved_context_characters", "retrieved_context_utf8_bytes", "retrieval_latency_seconds", "generation_latency_seconds", "end_to_end_latency_seconds", "output_tokens_per_second", "retrieval_hit_at_k", "retrieval_recall_at_k", "retrieval_reciprocal_rank", "retrieval_ndcg_at_k", "prompt_preparation_latency_seconds", "model_generation_latency_seconds", "model_load_latency_seconds", "hyde_model_load_latency_seconds", "gpu_energy_joules_estimate")
            },
            "hyde_gpu_energy_joules_estimate": _mean_nested(rows, "hyde_resource_metrics", "gpu_energy_joules_estimate"),
        },
        "resource_metrics_mean_peak": {
            key: {stat: _mean(rows, f"{key}_{stat}") for stat in ("mean", "peak")}
            for key in ("process_cpu_percent", "system_cpu_percent", "process_rss_bytes", "process_thread_count", "system_available_ram_bytes", "cpu_physical_core_count", "cpu_logical_core_count", "cpu_temperature_celsius", "gpu_utilization_percent", "gpu_memory_used_bytes", "gpu_memory_total_bytes", "gpu_temperature_celsius", "gpu_power_watts", "gpu_sm_clock_mhz")
        },
        "runtime": _runtime_metadata(config),
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


def _runtime_metadata(config: BenchmarkConfig) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "ragas", "psutil", "nvidia-ml-py"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if cuda_available else []
    except Exception:
        cuda_available, devices = None, []
    return {"python": platform.python_version(), "platform": platform.platform(), "libraries": versions, "providers": {"psutil": versions["psutil"] is not None, "nvml": versions["nvidia-ml-py"] is not None}, "cuda_available": cuda_available, "gpu_devices": devices, "resource_sample_interval_seconds": config.telemetry_interval_seconds, "hyde_provider": config.hyde_provider if config.strategy == "hyde" else None}


def _retrieval_diagnostics(retrieved: list[RetrievedChunk], reference_chunk_id: str | None, k: int) -> dict[str, Any]:
    if not reference_chunk_id:
        return {"retrieval_hit_at_k": None, "retrieval_recall_at_k": None, "retrieval_reciprocal_rank": None, "retrieval_ndcg_at_k": None}
    rank = next((index for index, item in enumerate(retrieved[:k], start=1) if item.chunk_id == reference_chunk_id), None)
    reciprocal = 1.0 / rank if rank else 0.0
    ndcg = 1.0 / math.log2(rank + 1) if rank else 0.0
    return {"retrieval_hit_at_k": int(rank is not None), "retrieval_recall_at_k": int(rank is not None), "retrieval_reciprocal_rank": reciprocal, "retrieval_ndcg_at_k": ndcg}


async def run_benchmark(
    config: BenchmarkConfig,
    answer_generator_factory: Callable[[], Any] | None = None,
    hyde_generator_factory: Callable[[], Any] | None = None,
    evaluator_factory: Callable[[], RagasEvaluator] | None = None,
) -> tuple[Path, Path]:
    samples, corpus = load_and_validate_data(config)
    _log(
        f"loaded dataset={config.dataset_mode} samples={len(samples)} "
        f"corpus_chunks={len(corpus)} model={config.model_key} strategy={config.strategy}"
    )
    identity = _experiment_configuration(config)
    output_jsonl, summary_path = _output_paths(config, identity)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists() and not config.resume:
        output_jsonl.unlink()
    existing = _read_jsonl(output_jsonl) if config.resume else []
    successful = [row for row in existing if not row.get("error") and not row.get("metric_errors")]
    if successful != existing:
        dropped = len(existing) - len(successful)
        output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in successful), encoding="utf-8")
        print(f"Resume: dropped {dropped} failed row(s) from {output_jsonl} to re-run them", flush=True)
    existing = successful
    completed = {str(row["qa_id"]) for row in existing if row.get("qa_id") is not None}
    _log(
        f"output={output_jsonl} resume_completed={len(completed)} "
        f"remaining={len(samples) - len(completed)}"
    )

    dense: DenseRetriever | None = None
    hybrid: HybridRetriever | None = None
    if config.strategy in {"hybrid", "hyde"}:
        _log(
            f"retrieval index loading device={config.retrieval_device} "
            f"embedding_model={config.retrieval_embedding_model}"
        )
        dense = DenseRetriever.load_or_build(
            chunks=corpus,
            index_dir=config.index_dir,
            model_name=config.retrieval_embedding_model,
            revision=config.retrieval_embedding_revision,
            device=config.retrieval_device,
            batch_size=config.retrieval_batch_size,
        )
        _log("retrieval index ready")
    if config.strategy == "hybrid":
        hybrid = HybridRetriever(corpus, BM25Retriever(corpus), dense)
        _log("hybrid retriever ready (BM25 + dense RRF)")

    hypothetical_records: dict[str, HypotheticalRecord] = {}
    if config.strategy == "hyde":
        _log(
            f"HyDE preparation started provider={config.hyde_provider} "
            f"model={config.hyde_generator_model}"
        )
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
            sample_resource_metrics=config.hyde_provider == "huggingface",
            telemetry_interval_seconds=config.telemetry_interval_seconds,
        )
        cache_hits = sum(1 for record in hypothetical_records.values() if record.cache_hit)
        _log(
            f"HyDE preparation finished documents={len(hypothetical_records)} "
            f"cache_hits={cache_hits} generated={len(hypothetical_records) - cache_hits}"
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
            concurrency=config.evaluator_concurrency,
        )
    )
    _log(f"answer model loading model={spec.model_id}")
    load_started = time.perf_counter()
    answer_generator = answer_generator_factory()
    model_load_latency = time.perf_counter() - load_started
    _log("answer model ready")
    try:
        _log(f"evaluator loading model={config.evaluator_model}")
        evaluator = evaluator_factory()
        _log("evaluator ready")
        with output_jsonl.open("a", encoding="utf-8", newline="\n") as output:
            for sample_index, sample in enumerate(samples, start=1):
                if sample.qa_id in completed:
                    continue
                _log(f"sample {sample_index}/{len(samples)} qa_id={sample.qa_id} stage=retrieval")
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
                    if hypothetical and not hypothetical.cache_hit and config.hyde_provider == "huggingface":
                        row["hyde_model_load_latency_seconds"] = hypothetical.model_load_latency_seconds
                        row["hyde_resource_metrics"] = hypothetical.resource_metrics
                    retrieved, retrieval_latency = retrieve_for_strategy(
                        strategy=config.strategy,
                        question=sample.question,
                        retrieval_k=config.retrieval_k,
                        hybrid_retriever=hybrid,
                        dense_retriever=dense,
                        hypothetical_text=hypothetical.text if hypothetical else None,
                    )
                    contexts = [item.text for item in retrieved]
                    _log(
                        f"sample {sample_index}/{len(samples)} stage=retrieval_done "
                        f"chunks={len(contexts)} latency={retrieval_latency:.2f}s"
                    )
                    gpu_device_index = resolve_cuda_device_index(getattr(answer_generator, "model", None)) if config.telemetry_interval_seconds > 0 else None
                    gpu_device_identity = resolve_cuda_device_identity(
                        getattr(answer_generator, "torch", None), gpu_device_index
                    ) if gpu_device_index is not None else None
                    sampler = ResourceSampler(config.telemetry_interval_seconds, gpu_device_index, gpu_device_identity)
                    try:
                        with sampler:
                            answer = answer_generator.answer(
                                sample.question,
                                contexts,
                                require_retrieved_context=config.strategy != "no_rag",
                            )
                    finally:
                        resource_metrics = sampler.summary()
                        row.update(resource_metrics)
                    _log(
                        f"sample {sample_index}/{len(samples)} stage=generation_done "
                        f"latency={answer.measurement.generation_latency_seconds:.2f}s "
                        f"output_tokens={answer.measurement.output_tokens}"
                    )
                    row.update(
                        {
                            "response": answer.text,
                            "retrieved_contexts": contexts,
                            "retrieved_chunk_ids": [item.chunk_id for item in retrieved],
                            "retrieved_metadata": [item.metadata for item in retrieved],
                            "retrieved_chunk_count": len(retrieved),
                            "retrieved_context_characters": sum(len(text) for text in contexts),
                            "retrieved_context_utf8_bytes": sum(len(text.encode("utf-8")) for text in contexts),
                            "retrieved_context_tokens": _count_context_tokens(answer_generator, contexts),
                            "model_load_latency_seconds": model_load_latency,
                            "prompt_preparation_latency_seconds": answer.measurement.prompt_preparation_latency_seconds,
                            "model_generation_latency_seconds": answer.measurement.model_generation_latency_seconds,
                            "possible_truncation": answer.measurement.output_tokens >= config.generation_settings.max_new_tokens,
                            "completion_status": "possible_truncation" if answer.measurement.output_tokens >= config.generation_settings.max_new_tokens else "completed",
                            **_retrieval_diagnostics(retrieved, sample.reference_chunk_id, config.retrieval_k),
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
                    _log(f"sample {sample_index}/{len(samples)} stage=evaluation")
                    evaluation_started = time.perf_counter()
                    row.update(
                        await evaluator.score(
                            user_input=sample.question,
                            response=answer.text,
                            retrieved_contexts=contexts,
                            reference=sample.reference,
                            include_context_metrics=config.strategy != "no_rag",
                        )
                    )
                    _log(
                        f"sample {sample_index}/{len(samples)} stage=evaluation_done "
                        f"latency={time.perf_counter() - evaluation_started:.2f}s"
                    )
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    _log(f"sample {sample_index}/{len(samples)} stage=failed error={row['error']}")
                    row["completion_status"] = "error"
                    for metric_name in METRIC_NAMES:
                        row.setdefault(metric_name, None)
                output.write(_serialize_row(row) + "\n")
                output.flush()
                _log(
                    f"sample {sample_index}/{len(samples)} stage=written "
                    f"status={'error' if row.get('error') else 'ok'}"
                )
                if not row.get("error") and not row.get("metric_errors"):
                    completed.add(sample.qa_id)
    finally:
        try:
            answer_generator.close()
        except Exception as exc:
            print(f"WARNING: failed to close answer generator: {type(exc).__name__}: {exc}", flush=True)

    rows = _read_jsonl(output_jsonl)
    _write_summary(summary_path, config, identity, output_jsonl, rows)
    return output_jsonl, summary_path
