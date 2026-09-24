#!/usr/bin/env python3
"""Batch runner for the full RAG benchmark experiment matrix.

Modes (--mode):
  smoke     – 1 question × 1 config  (pipeline validation, ~1 min)
  sample10  – 10 questions × 12 configs (quality snapshot, ~15 min)
  full      – 328 questions × 12 configs (complete benchmark, ~hours)

Matrix: 4 models × 3 strategies = 12 experiments per dataset mode.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_benchmark.cli import build_parser, config_from_args
from rag_benchmark.model_registry import MODEL_REGISTRY
from rag_benchmark.retrieval import DenseRetriever
from rag_benchmark.runner import BenchmarkConfig, load_and_validate_data, run_benchmark


MODELS = tuple(MODEL_REGISTRY)
STRATEGIES = ("no_rag", "hybrid", "hyde")
MATRIX_SIZE = len(MODELS) * len(STRATEGIES)


@dataclass
class ExperimentResult:
    model: str
    strategy: str
    output_jsonl: Path | None
    summary_path: Path | None
    duration_seconds: float
    error: str | None = None
    rows: int = 0
    metrics: dict[str, float | None] | None = None


def _build_matrix_configs(base_config: BenchmarkConfig, limit: int | None) -> list[tuple[str, str, BenchmarkConfig]]:
    configs: list[tuple[str, str, BenchmarkConfig]] = []
    for model_key in MODELS:
        for strategy in STRATEGIES:
            cfg = BenchmarkConfig(
                dataset_mode=base_config.dataset_mode,
                strategy=strategy,
                model_key=model_key,
                sample10_path=base_config.sample10_path,
                maternaqa_path=base_config.maternaqa_path,
                corpus_path=base_config.corpus_path,
                index_dir=base_config.index_dir,
                output_dir=base_config.output_dir,
                retrieval_k=base_config.retrieval_k,
                retrieval_embedding_model=base_config.retrieval_embedding_model,
                retrieval_embedding_revision=base_config.retrieval_embedding_revision,
                retrieval_device=base_config.retrieval_device,
                retrieval_batch_size=base_config.retrieval_batch_size,
                hyde_generator_model=base_config.hyde_generator_model,
                hyde_provider=base_config.hyde_provider,
                evaluator_model=base_config.evaluator_model,
                embedding_model=base_config.embedding_model,
                evaluator_max_completion_tokens=base_config.evaluator_max_completion_tokens,
                evaluator_timeout_seconds=base_config.evaluator_timeout_seconds,
                evaluator_concurrency=base_config.evaluator_concurrency,
                generation_settings=base_config.generation_settings,
                hyde_generation_settings=base_config.hyde_generation_settings,
                adapter_path=base_config.adapter_path,
                load_in_4bit=base_config.load_in_4bit,
                hyde_load_in_4bit=base_config.hyde_load_in_4bit,
                trust_remote_code=base_config.trust_remote_code,
                attn_implementation=base_config.attn_implementation,
                resume=base_config.resume,
                limit=limit,
            )
            configs.append((model_key, strategy, cfg))
    return configs


def _read_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _preflight_retrieval(cfgs: list[tuple[str, str, BenchmarkConfig]]) -> None:
    """Validate shared dense retrieval before any expensive experiment starts."""
    retrieval_cfg = next(
        (item for item in cfgs if item[1] in {"hybrid", "hyde"}),
        None,
    )
    if retrieval_cfg is None:
        return

    _, strategy, config = retrieval_cfg
    print(
        "Preflight: loading retrieval index "
        f"model={config.retrieval_embedding_model} device={config.retrieval_device}",
        flush=True,
    )
    try:
        _, corpus = load_and_validate_data(config)
        DenseRetriever.load_or_build(
            chunks=corpus,
            index_dir=config.index_dir,
            model_name=config.retrieval_embedding_model,
            revision=config.retrieval_embedding_revision,
            device=config.retrieval_device,
            batch_size=config.retrieval_batch_size,
        )
    except Exception as exc:
        raise RuntimeError(
            "Retrieval preflight failed before any sample was processed "
            f"for strategy {strategy}: {type(exc).__name__}: {exc}. "
            "Repair the retrieval model cache or configuration and retry."
        ) from exc
    print("Preflight: retrieval index ready", flush=True)


async def run_matrix(
    mode: str,
    base_args: list[str],
    limit: int | None,
    specific_model: str | None,
    specific_strategy: str | None,
) -> list[ExperimentResult]:
    parser = build_parser()
    args = parser.parse_args(base_args + ["--model", "gemma4_base"])
    base_config = config_from_args(args)

    cfgs = _build_matrix_configs(base_config, limit)
    if specific_model:
        cfgs = [item for item in cfgs if item[0] == specific_model]
    if specific_strategy:
        cfgs = [item for item in cfgs if item[1] == specific_strategy]

    _preflight_retrieval(cfgs)

    results: list[ExperimentResult] = []
    total = len(cfgs)
    started_at = time.monotonic()

    for idx, (model_key, strategy, cfg) in enumerate(cfgs, start=1):
        header = f"[{idx}/{total}] {mode} | {model_key} × {strategy}"
        print(f"\n{'='*70}")
        print(f"  {header}")
        print(f"{'='*70}")
        experiment_start = time.monotonic()

        try:
            output_jsonl, summary_path = await run_benchmark(cfg)
            summary = _read_summary(summary_path)
            duration = time.monotonic() - experiment_start
            result = ExperimentResult(
                model=model_key,
                strategy=strategy,
                output_jsonl=output_jsonl,
                summary_path=summary_path,
                duration_seconds=duration,
                rows=summary.get("rows", 0),
                metrics={
                    name: summary.get("metrics", {}).get(name)
                    for name in summary.get("metrics", {})
                },
            )
            print(f"  OK  rows={result.rows}  duration={result.duration_seconds:.1f}s")
            if result.metrics:
                metric_line = "  ".join(
                    f"{name}={value:.3f}" if value is not None else f"{name}=N/A"
                    for name, value in result.metrics.items()
                )
                print(f"  Metrics: {metric_line}")
        except Exception as exc:
            duration = time.monotonic() - experiment_start
            result = ExperimentResult(
                model=model_key,
                strategy=strategy,
                output_jsonl=None,
                summary_path=None,
                duration_seconds=duration,
                error=f"{type(exc).__name__}: {exc}",
            )
            print(f"  FAIL  {result.error}")

        results.append(result)

    total_duration = time.monotonic() - started_at
    print(f"\n{'='*70}")
    print(f"  MATRIX COMPLETE — {total} experiments in {total_duration:.1f}s ({total_duration/60:.1f}m)")
    print(f"{'='*70}")

    _print_summary_table(results)

    return results


def _print_summary_table(results: list[ExperimentResult]) -> None:
    print(f"\n{'Model':<20} {'Strategy':<10} {'Rows':>6} {'Duration':>10} {'Error'}", flush=True)
    print("-" * 70)
    for r in results:
        error = r.error[:50] if r.error else ""
        print(f"{r.model:<20} {r.strategy:<10} {r.rows:>6} {r.duration_seconds:>9.1f}s {error}", flush=True)

    failed = [r for r in results if r.error]
    if failed:
        print(f"\n{len(failed)} FAILED experiments:")
        for r in failed:
            print(f"  {r.model} × {r.strategy}: {r.error}")
    else:
        print(f"\nAll {len(results)} experiments completed successfully.")


def _interactive_mode() -> str:
    print("\nMaternaCare-ES RAG Benchmark Matrix\n")
    print("  1. smoke    — 1 question × 1 combo    (~1 min, pipeline validation)")
    print("  2. sample10 — 10 questions × 12 combos (~15 min, quality snapshot)")
    print("  3. full     — 328 questions × 12 combos (complete benchmark)")
    print()
    while True:
        choice = input("Select mode [1/2/3]: ").strip()
        if choice == "1":
            return "smoke"
        elif choice == "2":
            return "sample10"
        elif choice == "3":
            return "full"
        print("Invalid choice. Enter 1, 2, or 3.")


def build_matrix_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full RAG benchmark experiment matrix (4 models × 3 strategies).",
        epilog="Examples:\n"
        "  python scripts/run_experiment_matrix.py --mode smoke\n"
        "  python scripts/run_experiment_matrix.py --mode sample10 --model gemma4_qlora\n"
        "  python scripts/run_experiment_matrix.py --mode full --strategy hyde --no-load-in-4bit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "sample10", "full"),
        default=None,
        help="Benchmark mode. If omitted, an interactive menu is shown.",
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        default=None,
        help="Run only this model (default: all 4).",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default=None,
        help="Run only this strategy (default: all 3).",
    )
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help="Only validate datasets and exit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions for the selected dataset.",
    )

    return parser


def main() -> None:
    matrix_parser = build_matrix_parser()
    matrix_args, passthrough = matrix_parser.parse_known_args()
    mode = matrix_args.mode or _interactive_mode()

    has_adapter_path = any(
        arg == "--adapter-path" or arg.startswith("--adapter-path=")
        for arg in sys.argv[1:]
    )
    if has_adapter_path and matrix_args.model is None and mode != "smoke":
        matrix_parser.error(
            "--adapter-path requires --model <model>; a single adapter override cannot be "
            "applied to a multi-model matrix. Re-run with --model <model> to execute one model."
        )

    passthrough = [arg for arg in passthrough if arg != "--"]

    if mode == "smoke":
        dataset_mode = "sample10"
        limit = 1
        specific_model = matrix_args.model or "gemma4_base"
        specific_strategy = matrix_args.strategy or "no_rag"
    elif mode == "sample10":
        dataset_mode = "sample10"
        limit = matrix_args.limit
        specific_model = matrix_args.model
        specific_strategy = matrix_args.strategy
    else:  # full
        dataset_mode = "maternaqa_test"
        limit = matrix_args.limit
        specific_model = matrix_args.model
        specific_strategy = matrix_args.strategy

    base_args = ["--dataset-mode", dataset_mode, "--strategy", "no_rag", *passthrough]
    if limit is not None:
        base_args.extend(["--limit", str(limit)])

    combos = 1
    if not specific_model:
        combos *= len(MODELS)
    if not specific_strategy:
        combos *= len(STRATEGIES)

    if matrix_args.validate_data_only:
        from rag_benchmark.cli import build_parser as bp, config_from_args as cfa
        from rag_benchmark.runner import load_and_validate_data

        parser = bp()
        args = parser.parse_args(base_args + ["--model", "gemma4_base"])
        config = cfa(args)
        samples, corpus = load_and_validate_data(config)
        ref = {s.reference_chunk_id for s in samples if s.reference_chunk_id}
        print(json.dumps({
            "mode": mode,
            "dataset_mode": dataset_mode,
            "samples": len(samples),
            "corpus_chunks": len(corpus),
            "reference_chunk_ids": len(ref),
            "limit": limit,
            "matrix_size": combos,
        }, ensure_ascii=False))
        return

    print(f"\nMode: {mode}  |  Dataset: {dataset_mode}  |  Limit: {limit or 'all'}")
    print(f"Experiments: {combos}  |  Resume: enabled")
    print(f"Passthrough args: {' '.join(passthrough) if passthrough else '(none)'}\n")
    sys.stdout.flush()

    asyncio.run(
        run_matrix(
            mode=mode,
            base_args=base_args,
            limit=limit,
            specific_model=specific_model,
            specific_strategy=specific_strategy,
        )
    )


if __name__ == "__main__":
    main()
