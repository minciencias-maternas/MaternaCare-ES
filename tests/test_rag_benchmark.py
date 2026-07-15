from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts.rag_benchmark.data import (
    BenchmarkSample,
    load_corpus,
    load_maternaqa_test,
    load_sample10,
    validate_reference_chunks,
)
from scripts.rag_benchmark.cli import build_parser, config_from_args
from scripts.rag_benchmark.generation import GenerationResult
from scripts.rag_benchmark.hyde import prepare_hypothetical_documents
from scripts.rag_benchmark.hyde import load_hyde_cache
from scripts.rag_benchmark.metrics import METRIC_FIELDS, METRIC_NAMES, RagasEvaluator, build_metrics
from scripts.rag_benchmark.retrieval import reciprocal_rank_fusion
from scripts.rag_benchmark.runner import retrieve_for_strategy
from scripts.rag_benchmark.runner import _experiment_configuration, _output_paths, _read_jsonl, run_benchmark
from scripts.rag_benchmark.telemetry import GenerationMeasurement, combine_system_measurements


ROOT = Path(__file__).resolve().parents[1]
SAMPLE10 = ROOT / "datasets/Preguntas y Respuestas.xlsx"
MATERNAQA_TEST = ROOT / "datasets/obstetrics/qa/publication/qa_flat_jsonl/test.jsonl"
CORPUS = ROOT / "datasets/obstetrics/corpus/chunks.jsonl"


class CanonicalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = load_maternaqa_test(MATERNAQA_TEST)
        cls.corpus = load_corpus(CORPUS)

    def test_xlsx_loader_returns_ten_rows(self) -> None:
        rows = load_sample10(SAMPLE10)
        self.assertEqual(10, len(rows))
        self.assertTrue(all(row.question and row.reference for row in rows))

    def test_maternaqa_loader_returns_328_unique_ids(self) -> None:
        self.assertEqual(328, len(self.samples))
        self.assertEqual(328, len({sample.qa_id for sample in self.samples}))

    def test_corpus_contains_all_108_reference_chunk_ids(self) -> None:
        reference_ids = {sample.reference_chunk_id for sample in self.samples}
        self.assertEqual(108, len(reference_ids))
        validate_reference_chunks(self.samples, self.corpus)


class RetrievalTests(unittest.TestCase):
    def test_hybrid_rank_fusion_is_deterministic(self) -> None:
        rankings = [["chunk_b", "chunk_a", "chunk_c"], ["chunk_a", "chunk_b", "chunk_c"]]
        first = reciprocal_rank_fusion(rankings, k=3)
        second = reciprocal_rank_fusion(rankings, k=3)
        self.assertEqual(first, second)
        self.assertEqual(["chunk_a", "chunk_b", "chunk_c"], [item[0] for item in first])

    def test_hyde_uses_dedicated_generator_and_hypothetical_dense_query(self) -> None:
        measurement = GenerationMeasurement.from_counts(10, 4, 2.0)

        class FakeHydeGenerator:
            calls = 0

            def hypothetical_document(self, question: str) -> GenerationResult:
                self.calls += 1
                return GenerationResult("hypothetical clinical document", measurement)

            def close(self) -> None:
                pass

        class FakeAnswerGenerator:
            calls = 0

            def hypothetical_document(self, question: str) -> GenerationResult:
                self.calls += 1
                raise AssertionError("answer generator must not create HyDE documents")

        class FakeDenseRetriever:
            query: str | None = None

            def search(self, query: str, k: int) -> list[Any]:
                self.query = query
                return []

        sample = BenchmarkSample("q1", "clinical question", "reference")
        answer_generator = FakeAnswerGenerator()
        with tempfile.TemporaryDirectory() as temp_dir:
            hyde_generator = FakeHydeGenerator()
            records = prepare_hypothetical_documents(
                [sample],
                Path(temp_dir) / "hyde.jsonl",
                "independent/hyde-model",
                lambda: hyde_generator,
            )
        dense = FakeDenseRetriever()
        retrieve_for_strategy(
            strategy="hyde",
            question=sample.question,
            retrieval_k=5,
            hybrid_retriever=None,
            dense_retriever=dense,
            hypothetical_text=records[sample.qa_id].text,
        )
        self.assertEqual(1, hyde_generator.calls)
        self.assertEqual(0, answer_generator.calls)
        self.assertEqual("hypothetical clinical document", dense.query)


class TelemetryTests(unittest.TestCase):
    def test_token_and_timing_formulas(self) -> None:
        answer = GenerationMeasurement.from_counts(100, 20, 2.0)
        hypothetical = GenerationMeasurement.from_counts(40, 10, 1.0)
        combined = combine_system_measurements(answer, 0.5, hypothetical)
        self.assertEqual(140, combined["input_tokens"])
        self.assertEqual(30, combined["output_tokens"])
        self.assertEqual(170, combined["total_tokens"])
        self.assertEqual(3.0, combined["generation_latency_seconds"])
        self.assertEqual(3.5, combined["end_to_end_latency_seconds"])
        self.assertEqual(10.0, combined["output_tokens_per_second"])


class RagasTests(unittest.TestCase):
    def test_ragas_receives_single_turn_fields_and_exactly_six_metrics(self) -> None:
        calls: dict[str, dict[str, Any]] = {}
        construction: dict[str, Any] = {}

        class FakeClient:
            pass

        class FakeLLM:
            def __init__(self) -> None:
                self.model_args: dict[str, Any] = {}

        class FakeSample:
            def __init__(self, **kwargs: Any) -> None:
                self.fields = kwargs

            def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
                return self.fields

        class FakeResult:
            value = 0.5

        def metric_class(name: str) -> type[Any]:
            class FakeMetric:
                def __init__(self, **kwargs: Any) -> None:
                    construction[name] = kwargs

                async def ascore(self, **kwargs: Any) -> FakeResult:
                    calls[name] = kwargs
                    return FakeResult()

            return FakeMetric

        stack: dict[str, Any] = {
            "AsyncOpenAI": FakeClient,
            "SingleTurnSample": FakeSample,
            "llm_factory": lambda model, client: construction.setdefault("llm_model", model) and FakeLLM(),
            "embedding_factory": lambda provider, model, client: construction.setdefault(
                "embedding", (provider, model)
            ),
        }
        for class_name, metric_name in (
            ("ContextPrecision", "context_precision"),
            ("ContextRecall", "context_recall"),
            ("Faithfulness", "faithfulness"),
            ("AnswerRelevancy", "answer_relevancy"),
            ("AnswerCorrectness", "answer_correctness"),
            ("SemanticSimilarity", "semantic_similarity"),
        ):
            stack[class_name] = metric_class(metric_name)

        metrics, sample_class = build_metrics(
            evaluator_model="pinned-evaluator",
            embedding_model="pinned-embedding",
            max_completion_tokens=1024,
            stack=stack,
        )
        evaluator = RagasEvaluator(metrics, sample_class, timeout_seconds=5)
        result = asyncio.run(
            evaluator.score(
                user_input="question",
                response="answer",
                retrieved_contexts=["context"],
                reference="reference",
            )
        )
        self.assertEqual(METRIC_NAMES, tuple(metrics))
        self.assertEqual(set(METRIC_NAMES), set(result))
        self.assertEqual("pinned-evaluator", construction["llm_model"])
        self.assertEqual(("openai", "pinned-embedding"), construction["embedding"])
        expected = {
            "user_input": "question",
            "response": "answer",
            "retrieved_contexts": ["context"],
            "reference": "reference",
        }
        for metric_name in METRIC_NAMES:
            self.assertEqual(
                {field: expected[field] for field in METRIC_FIELDS[metric_name]},
                calls[metric_name],
            )

    def test_no_rag_skips_context_metrics_and_rejects_non_finite_values(self) -> None:
        class Sample:
            def __init__(self, **fields: Any) -> None:
                self.__dict__.update(fields)
        class Metric:
            def __init__(self, value: float) -> None:
                self.value = value
            async def ascore(self, **kwargs: Any) -> float:
                return self.value
        metrics = {name: Metric(float("nan") if name == "answer_relevancy" else 0.5) for name in METRIC_NAMES}
        result = asyncio.run(RagasEvaluator(metrics, Sample).score("q", "a", [], "r", include_context_metrics=False))
        self.assertEqual(([None] * 3, None, 1), ([result[name] for name in METRIC_NAMES[:3]], result["answer_relevancy"], len(result["metric_errors"])))
        self.assertIn("non-finite", result["metric_errors"][0])


class CorrectionTests(unittest.TestCase):
    @staticmethod
    def config(temp_dir: str, *extra: str):
        args = ["--dataset-mode", "sample10", "--strategy", "no_rag", "--model", "gemma4_qlora", "--output-dir", temp_dir, "--index-dir", str(Path(temp_dir) / "index"), "--limit", "1", *extra]
        return config_from_args(build_parser().parse_args(args))

    def test_configuration_identity_adapter_provenance_and_failed_row_retry(self) -> None:
        class Generator:
            calls = 0
            def answer(self, question: str, contexts: list[str]) -> GenerationResult:
                self.calls += 1
                return GenerationResult("answer", GenerationMeasurement.from_counts(1, 1, 1.0))
            def close(self) -> None: pass
        class Evaluator:
            async def score(self, **kwargs: Any) -> dict[str, float]: return {name: 0.5 for name in METRIC_NAMES}
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.config(temp_dir)
            identity = _experiment_configuration(config)
            self.assertEqual(("registry_remote", "4328cf26390c98c5e3c738b4460a05b95f4911f5"), (identity["answer"]["adapter_provenance"], identity["retrieval"]["revision"]))
            for options in (("--retrieval-k", "9"), ("--retrieval-embedding-revision", "other"), ("--max-new-tokens", "9"), ("--evaluator-model", "other"), ("--embedding-model", "other"), ("--strategy", "hyde")):
                changed = self.config(temp_dir, *options); self.assertNotEqual(_output_paths(config, identity), _output_paths(changed, _experiment_configuration(changed)))
            local = Path(temp_dir) / "adapter"; local.mkdir(); (local / "adapter_config.json").touch(); (local / "adapter_model.safetensors").touch()
            local_identity = _experiment_configuration(self.config(temp_dir, "--adapter-path", str(local)))
            self.assertEqual((str(local.resolve()), "explicit_local"), (local_identity["answer"]["adapter_source"], local_identity["answer"]["adapter_provenance"])); self.assertNotEqual(_output_paths(config, identity), _output_paths(config, local_identity))
            output, _ = _output_paths(config, identity); output.parent.mkdir(exist_ok=True); output.write_text('{"qa_id":"sample10-001","error":"transient"}\n')
            generator = Generator(); asyncio.run(run_benchmark(config, lambda: generator, evaluator_factory=Evaluator))
            self.assertEqual(1, generator.calls)
            rows = _read_jsonl(output); self.assertEqual(1, len(rows)); self.assertNotIn("error", rows[0])

    def test_result_and_hyde_truncated_tail_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = Path(temp_dir) / "result.jsonl"; result.write_text('{"qa_id":"ok"}\n{"qa_id":')
            self.assertEqual([{"qa_id": "ok"}], _read_jsonl(result)); self.assertEqual('{"qa_id":"ok"}\n', result.read_text())
            result.write_text('{}\n{bad}\n{}')
            with self.assertRaises(ValueError): _read_jsonl(result)
            cache = Path(temp_dir) / "hyde.jsonl"
            valid = '{"sample_id":"q","question_sha256":"x","model_id":"m","text":"h","input_tokens":1,"output_tokens":1,"total_tokens":2,"generation_latency_seconds":1,"output_tokens_per_second":1}\n'
            cache.write_text(valid + '{"sample_id":')
            self.assertIn("q", load_hyde_cache(cache, "m")); self.assertEqual(valid, cache.read_text())

    def test_dense_cache_identity_includes_encoder_revision(self) -> None:
        source = (ROOT / "scripts/rag_benchmark/retrieval.py").read_text()
        for expression in ("SentenceTransformer(model_name, revision=revision", "safe_name(model_name) / safe_name(revision)", 'manifest.get("revision") == revision', '"revision": revision'):
            self.assertIn(expression, source)

    def test_generation_timing_starts_before_prompt_preparation(self) -> None:
        source = (ROOT / "scripts/rag_benchmark/generation.py").read_text()
        self.assertLess(source.index("started = time.perf_counter()"), source.index("prompt = self.tokenizer.apply_chat_template"))


class CliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "scripts/run_rag_benchmark.py", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_help_has_no_runtime_model_side_effects(self) -> None:
        result = self.run_cli("--help")
        self.assertIn("--dataset-mode", result.stdout)
        self.assertIn("--hyde-generator-model", result.stdout)

    def test_both_dataset_modes_validate_without_models_or_api_clients(self) -> None:
        common = ("--strategy", "no_rag", "--model", "gemma4_base", "--validate-data-only")
        sample = self.run_cli("--dataset-mode", "sample10", *common)
        test = self.run_cli("--dataset-mode", "maternaqa_test", *common)
        self.assertIn('"samples": 10', sample.stdout)
        self.assertIn('"samples": 328', test.stdout)
        self.assertIn('"reference_chunk_ids": 108', test.stdout)


if __name__ == "__main__":
    unittest.main()
