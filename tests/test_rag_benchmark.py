from __future__ import annotations

import asyncio
import importlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts.rag_benchmark.data import (
    BenchmarkSample,
    CorpusChunk,
    load_corpus,
    load_jsonl_dataset,
    load_maternaqa_test,
    load_sample10,
    validate_reference_chunks,
)
from scripts.rag_benchmark.cli import build_parser, config_from_args
from scripts.rag_benchmark.generation import (
    ANSWER_WITH_CONTEXT_INSTRUCTION,
    ANSWER_WITHOUT_CONTEXT_INSTRUCTION,
    GenerationResult,
    GenerationSettings,
    OpenAIHydeGenerator,
    build_answer_messages,
)
from scripts.rag_benchmark.hyde import prepare_hypothetical_documents
from scripts.rag_benchmark.hyde import load_hyde_cache
from scripts.rag_benchmark.metrics import METRIC_FIELDS, METRIC_NAMES, RagasEvaluator, build_metrics
from scripts.rag_benchmark.retrieval import DenseRetriever, reciprocal_rank_fusion
from scripts.rag_benchmark.runner import retrieve_for_strategy
from scripts.rag_benchmark.runner import _experiment_configuration, _hyde_cache_path, _hyde_identity, _output_paths, _read_jsonl, run_benchmark
from scripts.rag_benchmark.telemetry import GenerationMeasurement, combine_system_measurements


ROOT = Path(__file__).resolve().parents[1]
SAMPLE10 = ROOT / "datasets/sample10.jsonl"
MATERNAQA_TEST = ROOT / "datasets/obstetrics/qa/publication/qa_flat_jsonl/test.jsonl"
CORPUS = ROOT / "datasets/obstetrics/corpus/chunks.jsonl"


class CanonicalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = load_maternaqa_test(MATERNAQA_TEST)
        cls.corpus = load_corpus(CORPUS)

    def test_sample10_jsonl_loader_returns_verified_ten_rows(self) -> None:
        rows = load_sample10(SAMPLE10)
        self.assertEqual(10, len(rows))
        self.assertTrue(all(row.question and row.reference for row in rows))
        self.assertEqual("sample10_001", rows[0].qa_id)
        self.assertEqual("DATA_GT", rows[0].metadata["section"])

    def test_runtime_dataset_loader_has_no_xlsx_parser(self) -> None:
        source = (ROOT / "scripts/rag_benchmark/data.py").read_text(encoding="utf-8")
        self.assertNotIn("zipfile", source)
        self.assertNotIn("ElementTree", source)

    def test_jsonl_loader_rejects_null_required_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.jsonl"
            path.write_text('{"qa_id": null, "pregunta": "q", "respuesta": "r"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "null required fields"):
                load_jsonl_dataset(path)

    def test_maternaqa_loader_returns_328_unique_ids(self) -> None:
        self.assertEqual(328, len(self.samples))
        self.assertEqual(328, len({sample.qa_id for sample in self.samples}))

    def test_corpus_contains_all_108_reference_chunk_ids(self) -> None:
        reference_ids = {sample.reference_chunk_id for sample in self.samples}
        self.assertEqual(108, len(reference_ids))
        validate_reference_chunks(self.samples, self.corpus)


class RetrievalTests(unittest.TestCase):
    def test_answer_prompt_contracts_are_strategy_specific(self) -> None:
        question = "¿Cuál es la recomendación?"
        no_rag = build_answer_messages(question, [], require_retrieved_context=False)[0]["content"]
        hybrid = build_answer_messages(question, ["evidencia hybrid"], require_retrieved_context=True)[0]["content"]
        hyde = build_answer_messages(question, ["evidencia HyDE"], require_retrieved_context=True)[0]["content"]

        self.assertIn(ANSWER_WITHOUT_CONTEXT_INSTRUCTION, no_rag)
        self.assertNotIn("Contexto recuperado", no_rag)
        self.assertIn(ANSWER_WITH_CONTEXT_INSTRUCTION, hybrid)
        self.assertIn("Contexto recuperado:\n[1] evidencia hybrid", hybrid)
        self.assertIn(ANSWER_WITH_CONTEXT_INSTRUCTION, hyde)
        self.assertIn("Contexto recuperado:\n[1] evidencia HyDE", hyde)

    def test_rag_prompt_retains_evidence_contract_when_retrieval_is_empty(self) -> None:
        content = build_answer_messages("¿Pregunta?", [], require_retrieved_context=True)[0]["content"]

        self.assertIn(ANSWER_WITH_CONTEXT_INSTRUCTION, content)
        self.assertIn("Contexto recuperado:", content)
        self.assertNotIn(ANSWER_WITHOUT_CONTEXT_INSTRUCTION, content)

    def test_dense_retriever_reports_missing_lancedb_dependency_lazily(self) -> None:
        with patch.dict(sys.modules, {"lancedb": None}):
            with self.assertRaisesRegex(RuntimeError, "Install lancedb"):
                DenseRetriever.load_or_build(
                    [CorpusChunk("chunk-1", "text")],
                    Path("unused"),
                    "test/model",
                    "revision",
                )

    def test_dense_retriever_uses_lancedb_exact_search_and_stable_identity(self) -> None:
        class FakeQuery:
            def __init__(self, rows: list[dict[str, Any]], vector: list[float]) -> None:
                self.rows = rows
                self.vector = vector
                self.metric: str | None = None
                self.k: int | None = None

            def distance_type(self, metric: str) -> "FakeQuery":
                self.metric = metric
                return self

            def limit(self, k: int) -> "FakeQuery":
                self.k = k
                return self

            def to_list(self) -> list[dict[str, Any]]:
                return [{**row, "_distance": index / 10} for index, row in enumerate(self.rows[: self.k])]

        class FakeTable:
            def __init__(self, rows: list[dict[str, Any]]) -> None:
                self.rows = rows
                self.query: FakeQuery | None = None

            def search(self, vector: list[float]) -> FakeQuery:
                self.query = FakeQuery(self.rows, vector)
                return self.query

        class FakeDatabase:
            def __init__(self) -> None:
                self.tables: dict[str, FakeTable] = {}

            def open_table(self, name: str) -> FakeTable:
                if name not in self.tables:
                    raise ValueError("table does not exist")
                return self.tables[name]

            def create_table(self, name: str, data: list[dict[str, Any]]) -> FakeTable:
                table = FakeTable(data)
                self.tables[name] = table
                return table

        class FakeLanceDB:
            def __init__(self) -> None:
                self.database = FakeDatabase()
                self.paths: list[str] = []

            def connect(self, path: str) -> FakeDatabase:
                self.paths.append(path)
                return self.database

        class FakeEncoder:
            model_name = "test/model"

            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def encode(self, texts: list[str]) -> list[list[float]]:
                self.calls.append(texts)
                return [[3.0, 4.0] for _ in texts]

        fake_lancedb = FakeLanceDB()
        previous_module = sys.modules.get("lancedb")
        sys.modules["lancedb"] = fake_lancedb  # type: ignore[assignment]
        chunks = [
            CorpusChunk("chunk-1", "first", {"section": "A"}),
            CorpusChunk("chunk-2", "second", {"section": "B"}),
        ]
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                encoder = FakeEncoder()
                first = DenseRetriever.load_or_build(chunks, Path(temp_dir) / "lancedb", "test/model", "revision", encoder=encoder)
                second = DenseRetriever.load_or_build(chunks, Path(temp_dir) / "lancedb", "test/model", "revision", encoder=encoder)
                self.assertEqual([["first", "second"]], encoder.calls)
                self.assertIs(first._table, second._table)
                row = first._table.rows[0]
                self.assertEqual(
                    {"vector", "chunk_id", "text", "metadata", "corpus_sha256", "embedding_model", "embedding_revision", "embedding_dimension", "normalized"},
                    set(row),
                )
                self.assertEqual({"section": "A"}, json.loads(row["metadata"]))
                self.assertEqual(("test/model", "revision", 2, True), (row["embedding_model"], row["embedding_revision"], row["embedding_dimension"], row["normalized"]))
                results = first.search("question", 1)
                self.assertEqual("chunk-1", results[0].chunk_id)
                self.assertEqual("cosine", first._table.query.metric)
                self.assertEqual(1, first._table.query.k)
                self.assertEqual([0.6, 0.8], first._table.query.vector)
        finally:
            if previous_module is None:
                del sys.modules["lancedb"]
            else:
                sys.modules["lancedb"] = previous_module

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
                {"provider": "fake", "model": "independent/hyde-model", "prompt": "test", "generation": {}},
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

    def test_openai_hyde_uses_responses_api_usage_and_observed_latency(self) -> None:
        calls: list[dict[str, Any]] = []

        class Usage:
            input_tokens = 17
            output_tokens = 9

        class Response:
            status = "completed"
            incomplete_details = None
            output_text = "documento clínico hipotético"
            usage = Usage()

        class Responses:
            def create(self, **kwargs: Any) -> Response:
                calls.append(kwargs)
                return Response()

        class Client:
            responses = Responses()
            closed = False

            def close(self) -> None:
                self.closed = True

        client = Client()
        result = OpenAIHydeGenerator(client, "gpt-5-mini-2025-08-07", GenerationSettings(max_new_tokens=77)).hypothetical_document("¿Pregunta?")
        self.assertEqual((17, 9, 26), (result.measurement.input_tokens, result.measurement.output_tokens, result.measurement.total_tokens))
        self.assertGreaterEqual(result.measurement.generation_latency_seconds, 0.0)
        self.assertEqual("gpt-5-mini-2025-08-07", calls[0]["model"])
        self.assertEqual(77, calls[0]["max_output_tokens"])
        self.assertFalse(calls[0]["store"])
        self.assertNotIn("temperature", calls[0])
        self.assertNotIn("top_p", calls[0])
        self.assertIn("documento clínico", calls[0]["input"])
        OpenAIHydeGenerator(client, "gpt-5-mini-2025-08-07", GenerationSettings()).close()
        self.assertTrue(client.closed)

    def test_openai_hyde_rejects_incomplete_responses_before_returning_text(self) -> None:
        class Usage:
            input_tokens = 17
            output_tokens = 9

        class Response:
            output_text = "partial document"
            usage = Usage()

            def __init__(self, status: str, incomplete_details: object | None = None) -> None:
                self.status = status
                self.incomplete_details = incomplete_details

        class Responses:
            def __init__(self, response: Response) -> None:
                self.response = response

            def create(self, **kwargs: Any) -> Response:
                return self.response

        class Client:
            def __init__(self, response: Response) -> None:
                self.responses = Responses(response)

        for response in (Response("incomplete"), Response("completed", {"reason": "max_output_tokens"})):
            with self.assertRaisesRegex(RuntimeError, "non-completed|incomplete HyDE"):
                OpenAIHydeGenerator(Client(response), "gpt-5-mini-2025-08-07", GenerationSettings()).hypothetical_document("q")


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

    def test_cache_hit_hyde_telemetry_is_historical_not_current_run_usage(self) -> None:
        answer = GenerationMeasurement.from_counts(100, 20, 2.0)
        cached_hyde = GenerationMeasurement.from_counts(40, 10, 1.0)
        combined = combine_system_measurements(answer, 0.5, cached_hyde, hypothetical_cache_hit=True)
        self.assertEqual((100, 20, 120), (combined["input_tokens"], combined["output_tokens"], combined["total_tokens"]))
        self.assertEqual(2.0, combined["generation_latency_seconds"])


class RagasTests(unittest.TestCase):
    def test_ragas_receives_single_turn_fields_and_exactly_seven_metrics(self) -> None:
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
            ("NoiseSensitivity", "noise_sensitivity"),
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
        self.assertEqual(([None] * 4, None, 1), ([result[name] for name in METRIC_NAMES[:4]], result["answer_relevancy"], len(result["metric_errors"])))
        self.assertIn("non-finite", result["metric_errors"][0])

    def test_ragas_limits_metric_concurrency(self) -> None:
        active = 0
        peak = 0

        class Sample:
            def __init__(self, **fields: Any) -> None:
                self.__dict__.update(fields)

            def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
                return self.__dict__

        class Metric:
            async def ascore(self, **kwargs: Any) -> float:
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1
                return 0.5

        metrics = {name: Metric() for name in METRIC_NAMES}
        result = asyncio.run(
            RagasEvaluator(metrics, Sample, concurrency=2).score("q", "a", ["c"], "r")
        )

        self.assertEqual(set(METRIC_NAMES), set(result))
        self.assertEqual(2, peak)


class CorrectionTests(unittest.TestCase):
    @staticmethod
    def config(temp_dir: str, *extra: str):
        args = ["--dataset-mode", "sample10", "--strategy", "no_rag", "--model", "gemma4_qlora", "--output-dir", temp_dir, "--index-dir", str(Path(temp_dir) / "index"), "--limit", "1", *extra]
        return config_from_args(build_parser().parse_args(args))

    def test_configuration_identity_adapter_provenance_and_failed_row_retry(self) -> None:
        class Generator:
            calls = 0
            require_retrieved_context: bool | None = None

            def answer(
                self,
                question: str,
                contexts: list[str],
                *,
                require_retrieved_context: bool,
            ) -> GenerationResult:
                self.calls += 1
                self.require_retrieved_context = require_retrieved_context
                return GenerationResult("answer", GenerationMeasurement.from_counts(1, 1, 1.0))

            def close(self) -> None: pass
        class Evaluator:
            async def score(self, **kwargs: Any) -> dict[str, float]: return {name: 0.5 for name in METRIC_NAMES}
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.config(temp_dir)
            identity = _experiment_configuration(config)
            self.assertEqual(("registry_remote", "0c6f0d0ea8f284b9070c3ffaa50677440943f984"), (identity["answer"]["adapter_provenance"], identity["retrieval"]["revision"]))
            self.assertEqual(ANSWER_WITHOUT_CONTEXT_INSTRUCTION, identity["answer"]["prompt"])
            for options in (("--retrieval-k", "9"), ("--retrieval-embedding-revision", "other"), ("--max-new-tokens", "9"), ("--evaluator-model", "other"), ("--embedding-model", "other"), ("--strategy", "hyde")):
                changed = self.config(temp_dir, *options); self.assertNotEqual(_output_paths(config, identity), _output_paths(changed, _experiment_configuration(changed)))
            local = Path(temp_dir) / "adapter"; local.mkdir(); (local / "adapter_config.json").touch(); (local / "adapter_model.safetensors").touch()
            local_identity = _experiment_configuration(self.config(temp_dir, "--adapter-path", str(local)))
            self.assertEqual((str(local.resolve()), "explicit_local"), (local_identity["answer"]["adapter_source"], local_identity["answer"]["adapter_provenance"])); self.assertNotEqual(_output_paths(config, identity), _output_paths(config, local_identity))
            output, _ = _output_paths(config, identity); output.parent.mkdir(exist_ok=True); output.write_text('{"qa_id":"sample10-001","error":"transient"}\n')
            generator = Generator(); asyncio.run(run_benchmark(config, lambda: generator, evaluator_factory=Evaluator))
            self.assertEqual(1, generator.calls)
            self.assertFalse(generator.require_retrieved_context)
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
        for expression in (
            "SentenceTransformer(model_name, revision=revision",
            "safe_name(model_name), safe_name(revision), fingerprint[:16]",
            '"embedding_revision": revision',
            '"corpus_sha256": fingerprint',
        ):
            self.assertIn(expression, source)

    def test_hyde_identity_prevents_cross_provider_cache_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.config(temp_dir, "--strategy", "hyde")
            identity = _hyde_identity(config)
            self.assertEqual("openai", identity["provider"])
            self.assertIn("documento clínico", identity["prompt"])
            hf_config = self.config(temp_dir, "--strategy", "hyde", "--hyde-provider", "huggingface")
            self.assertNotEqual(_hyde_cache_path(config), _hyde_cache_path(hf_config))
            path = Path(temp_dir) / "hyde.jsonl"
            path.write_text(json.dumps({
                "sample_id": "q", "question_sha256": "x", "model_id": config.hyde_generator_model,
                "generator_identity": identity, "text": "document", "input_tokens": 1,
                "output_tokens": 1, "total_tokens": 2, "generation_latency_seconds": 1,
                "output_tokens_per_second": 1,
            }) + "\n", encoding="utf-8")
            self.assertIn("q", load_hyde_cache(path, config.hyde_generator_model, identity))
            self.assertNotIn("q", load_hyde_cache(path, config.hyde_generator_model, {**identity, "provider": "huggingface"}))

    def test_openai_cache_identity_ignores_huggingface_only_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.config(temp_dir, "--strategy", "hyde")
            changed = self.config(
                temp_dir,
                "--strategy", "hyde",
                "--no-hyde-load-in-4bit",
                "--trust-remote-code",
                "--attn-implementation", "flash_attention_2",
            )
            self.assertEqual(_hyde_identity(config), _hyde_identity(changed))
            self.assertEqual(_hyde_cache_path(config), _hyde_cache_path(changed))

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

    def test_sample10_default_uses_jsonl_snapshot(self) -> None:
        args = build_parser().parse_args(
            ["--dataset-mode", "sample10", "--strategy", "no_rag", "--model", "gemma4_base"]
        )
        self.assertEqual(Path("datasets/sample10.jsonl"), args.sample10_path)
        self.assertEqual("BAAI/bge-m3", args.retrieval_embedding_model)
        self.assertEqual("0c6f0d0ea8f284b9070c3ffaa50677440943f984", args.retrieval_embedding_revision)
        self.assertEqual("text-embedding-3-large", args.embedding_model)
        config = config_from_args(args)
        self.assertEqual(("openai", "gpt-5-mini-2025-08-07"), (config.hyde_provider, config.hyde_generator_model))

    def test_hyde_default_model_follows_provider(self) -> None:
        args = build_parser().parse_args(
            ["--dataset-mode", "sample10", "--strategy", "hyde", "--model", "gemma4_base", "--hyde-provider", "huggingface"]
        )
        self.assertEqual("Qwen/Qwen2.5-1.5B-Instruct", config_from_args(args).hyde_generator_model)

    def test_both_dataset_modes_validate_without_models_or_api_clients(self) -> None:
        common = ("--strategy", "no_rag", "--model", "gemma4_base", "--validate-data-only")
        sample = self.run_cli("--dataset-mode", "sample10", *common)
        test = self.run_cli("--dataset-mode", "maternaqa_test", *common)
        self.assertIn('"samples": 10', sample.stdout)
        self.assertIn('"samples": 328', test.stdout)
        self.assertIn('"reference_chunk_ids": 108', test.stdout)


class MatrixCliTests(unittest.TestCase):
    def run_matrix_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "scripts/run_experiment_matrix.py", *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_adapter_path_rejects_multi_model_matrix(self) -> None:
        for adapter_arguments in (
            ("--adapter-path", "outputs/custom-adapter"),
            ("--adapter-path=outputs/custom-adapter",),
        ):
            with self.subTest(adapter_arguments=adapter_arguments):
                result = self.run_matrix_cli("--mode", "sample10", *adapter_arguments)

                self.assertEqual(2, result.returncode)
                self.assertIn("--adapter-path requires --model <model>", result.stderr)
                self.assertIn("Re-run with --model <model> to execute one model.", result.stderr)

    def test_interactive_adapter_path_guard_uses_selected_mode(self) -> None:
        with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
            matrix = importlib.import_module("run_experiment_matrix")

        for mode, should_reject in (
            ("smoke", False),
            ("sample10", True),
            ("full", True),
        ):
            with self.subTest(mode=mode), (
                patch.object(
                    sys,
                    "argv",
                    [
                        "run_experiment_matrix.py",
                        "--adapter-path",
                        "outputs/custom-adapter",
                        "--validate-data-only",
                    ],
                )
            ), patch.object(matrix, "_interactive_mode", return_value=mode), (
                redirect_stdout(io.StringIO())
            ), redirect_stderr(io.StringIO()) as stderr:
                if should_reject:
                    with self.assertRaisesRegex(SystemExit, "2"):
                        matrix.main()
                    self.assertIn("--adapter-path requires --model <model>", stderr.getvalue())
                else:
                    matrix.main()

        with (
            patch.object(
                sys,
                "argv",
                [
                    "run_experiment_matrix.py",
                    "--mode",
                    "sample10",
                    "--adapter-path",
                    "outputs/custom-adapter",
                ],
            ),
            patch.object(matrix, "_interactive_mode", side_effect=AssertionError("unexpected prompt")),
            redirect_stderr(io.StringIO()) as stderr,
        ):
            with self.assertRaisesRegex(SystemExit, "2"):
                matrix.main()
            self.assertIn("--adapter-path requires --model <model>", stderr.getvalue())

    def test_default_matrix_validation_remains_available(self) -> None:
        result = self.run_matrix_cli("--mode", "sample10", "--validate-data-only")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('"matrix_size": 12', result.stdout)

    def test_adapter_path_allows_single_model_regardless_of_option_order(self) -> None:
        for arguments in (
            (
                "--mode", "sample10", "--model", "gemma4_qlora", "--validate-data-only",
                "--adapter-path=outputs/custom-adapter",
            ),
            (
                "--mode", "sample10", "--adapter-path", "outputs/custom-adapter", "--model",
                "gemma4_qlora", "--validate-data-only",
            ),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_matrix_cli(*arguments)

                self.assertEqual(0, result.returncode, result.stderr)
                self.assertIn('"matrix_size": 3', result.stdout)

    def test_validation_matrix_size_honors_strategy_filter(self) -> None:
        result = self.run_matrix_cli(
            "--mode", "sample10", "--model", "gemma4_qlora", "--strategy", "hyde",
            "--validate-data-only",
        )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('"matrix_size": 1', result.stdout)

    def test_full_mode_limit_selects_maternaqa_subset(self) -> None:
        result = self.run_matrix_cli(
            "--mode", "full", "--model", "gemma4_base", "--limit", "10", "--validate-data-only",
        )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn('"dataset_mode": "maternaqa_test"', result.stdout)
        self.assertIn('"samples": 10', result.stdout)
        self.assertIn('"limit": 10', result.stdout)


if __name__ == "__main__":
    unittest.main()
