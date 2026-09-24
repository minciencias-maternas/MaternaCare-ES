"""Command-line interface for the RAG benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .generation import GenerationSettings
from .model_registry import MODEL_REGISTRY
from .runner import BenchmarkConfig, load_and_validate_data, run_benchmark


DEFAULT_SAMPLE10 = Path("datasets/sample10.jsonl")
DEFAULT_MATERNAQA = Path("datasets/obstetrics/qa/publication/qa_flat_jsonl/test.jsonl")
DEFAULT_CORPUS = Path("datasets/obstetrics/corpus/chunks.jsonl")
DEFAULT_INDEX_DIR = Path("artifacts/rag_benchmark/lancedb")
DEFAULT_OUTPUT_DIR = Path("outputs/rag_benchmark")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare four MaternaCare answer generators with no RAG, Hybrid, or canonical HyDE retrieval."
    )
    parser.add_argument("--dataset-mode", required=True, choices=("sample10", "maternaqa_test"))
    parser.add_argument("--strategy", required=True, choices=("no_rag", "hybrid", "hyde"))
    parser.add_argument("--model", dest="model_key", required=True, choices=tuple(MODEL_REGISTRY))
    parser.add_argument("--sample10-path", type=Path, default=DEFAULT_SAMPLE10)
    parser.add_argument("--maternaqa-test-path", type=Path, default=DEFAULT_MATERNAQA)
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument(
        "--retrieval-embedding-model",
        default="BAAI/bge-m3",
    )
    parser.add_argument("--retrieval-embedding-revision", default="0c6f0d0ea8f284b9070c3ffaa50677440943f984")
    parser.add_argument("--retrieval-device", default="cpu")
    parser.add_argument("--retrieval-batch-size", type=int, default=32)
    parser.add_argument("--hyde-provider", choices=("openai", "huggingface"), default="openai")
    parser.add_argument(
        "--hyde-generator-model",
        default=None,
        help="HyDE model; defaults to the selected provider's pinned model.",
    )
    parser.add_argument("--evaluator-model", default="gpt-5.4-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--evaluator-max-completion-tokens", type=int, default=2048)
    parser.add_argument(
        "--evaluator-timeout-seconds",
        type=int,
        default=600,
        help="Timeout máximo por métrica y muestra de RAGAS (por defecto: 600 s).",
    )
    parser.add_argument(
        "--evaluator-concurrency",
        type=int,
        default=3,
        help="Máximo de métricas RAGAS concurrentes por muestra (por defecto: 3).",
    )
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--hyde-max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--hyde-do-sample", action="store_true", default=False)
    parser.add_argument("--hyde-temperature", type=float, default=0.7)
    parser.add_argument("--hyde-top-p", type=float, default=0.9)
    parser.add_argument("--hyde-repetition-penalty", type=float, default=None)
    parser.add_argument("--hyde-no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hyde-load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help="Load and validate the selected dataset and corpus, then exit without loading models or clients.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    hyde_generator_model = args.hyde_generator_model or (
        DEFAULT_OPENAI_HYDE_MODEL
        if args.hyde_provider == "openai"
        else DEFAULT_HUGGINGFACE_HYDE_MODEL
    )
    generation = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    hyde_generation = GenerationSettings(
        max_new_tokens=args.hyde_max_new_tokens,
        do_sample=args.hyde_do_sample,
        temperature=args.hyde_temperature,
        top_p=args.hyde_top_p,
        repetition_penalty=args.hyde_repetition_penalty,
        no_repeat_ngram_size=args.hyde_no_repeat_ngram_size,
    )
    return BenchmarkConfig(
        dataset_mode=args.dataset_mode,
        strategy=args.strategy,
        model_key=args.model_key,
        sample10_path=args.sample10_path,
        maternaqa_path=args.maternaqa_test_path,
        corpus_path=args.corpus_path,
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        retrieval_k=args.retrieval_k,
        retrieval_embedding_model=args.retrieval_embedding_model,
        retrieval_embedding_revision=args.retrieval_embedding_revision,
        retrieval_device=args.retrieval_device,
        retrieval_batch_size=args.retrieval_batch_size,
        hyde_generator_model=hyde_generator_model,
        hyde_provider=args.hyde_provider,
        evaluator_model=args.evaluator_model,
        embedding_model=args.embedding_model,
        evaluator_max_completion_tokens=args.evaluator_max_completion_tokens,
        evaluator_timeout_seconds=args.evaluator_timeout_seconds,
        evaluator_concurrency=args.evaluator_concurrency,
        generation_settings=generation,
        hyde_generation_settings=hyde_generation,
        adapter_path=args.adapter_path,
        load_in_4bit=args.load_in_4bit,
        hyde_load_in_4bit=args.hyde_load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        resume=args.resume,
        limit=args.limit,
    )


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    if args.validate_data_only:
        samples, corpus = load_and_validate_data(config)
        reference_chunks = {sample.reference_chunk_id for sample in samples if sample.reference_chunk_id}
        print(
            json.dumps(
                {
                    "dataset_mode": config.dataset_mode,
                    "samples": len(samples),
                    "corpus_chunks": len(corpus),
                    "reference_chunk_ids": len(reference_chunks),
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return
    output_jsonl, summary = asyncio.run(run_benchmark(config))
    print(f"Per-sample output: {output_jsonl}")
    print(f"Summary: {summary}")
DEFAULT_OPENAI_HYDE_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_HUGGINGFACE_HYDE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
