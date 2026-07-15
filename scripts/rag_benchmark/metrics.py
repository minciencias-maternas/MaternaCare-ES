"""RAGAS 0.4.3 collections evaluation with explicit evaluator dependencies."""

from __future__ import annotations

import asyncio
import math
from typing import Any


METRIC_NAMES = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "semantic_similarity",
)

METRIC_FIELDS: dict[str, tuple[str, ...]] = {
    "context_precision": ("user_input", "retrieved_contexts", "reference"),
    "context_recall": ("user_input", "retrieved_contexts", "reference"),
    "faithfulness": ("user_input", "response", "retrieved_contexts"),
    "answer_relevancy": ("user_input", "response"),
    "answer_correctness": ("user_input", "response", "reference"),
    "semantic_similarity": ("response", "reference"),
}


def import_ragas_stack() -> dict[str, Any]:
    try:
        from openai import AsyncOpenAI
        from ragas import SingleTurnSample
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
            SemanticSimilarity,
        )
    except ImportError as exc:
        raise RuntimeError("Install RAGAS 0.4.3 and evaluator dependencies from requirements.txt") from exc
    return {
        "AsyncOpenAI": AsyncOpenAI,
        "SingleTurnSample": SingleTurnSample,
        "embedding_factory": embedding_factory,
        "llm_factory": llm_factory,
        "ContextPrecision": ContextPrecision,
        "ContextRecall": ContextRecall,
        "Faithfulness": Faithfulness,
        "AnswerRelevancy": AnswerRelevancy,
        "AnswerCorrectness": AnswerCorrectness,
        "SemanticSimilarity": SemanticSimilarity,
    }


def configure_llm_args(llm: Any, model_name: str, max_completion_tokens: int) -> None:
    if not hasattr(llm, "model_args") or not isinstance(llm.model_args, dict):
        return
    if model_name.lower().startswith("gpt-5."):
        llm.model_args.pop("max_tokens", None)
        llm.model_args["max_completion_tokens"] = max_completion_tokens
        llm.model_args["temperature"] = 1.0
        llm.model_args.pop("top_p", None)
    else:
        llm.model_args["max_tokens"] = max_completion_tokens


def build_metrics(
    evaluator_model: str,
    embedding_model: str,
    max_completion_tokens: int,
    stack: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Build all six metrics with explicitly pinned LLM and embeddings."""

    stack = stack or import_ragas_stack()
    client = stack["AsyncOpenAI"]()
    llm = stack["llm_factory"](evaluator_model, client=client)
    configure_llm_args(llm, evaluator_model, max_completion_tokens)
    embeddings = stack["embedding_factory"](
        "openai", model=embedding_model, client=client
    )
    metrics = {
        "context_precision": stack["ContextPrecision"](llm=llm),
        "context_recall": stack["ContextRecall"](llm=llm),
        "faithfulness": stack["Faithfulness"](llm=llm),
        "answer_relevancy": stack["AnswerRelevancy"](llm=llm, embeddings=embeddings),
        "answer_correctness": stack["AnswerCorrectness"](llm=llm, embeddings=embeddings),
        "semantic_similarity": stack["SemanticSimilarity"](embeddings=embeddings),
    }
    return metrics, stack["SingleTurnSample"]


class RagasEvaluator:
    def __init__(self, metrics: dict[str, Any], sample_class: Any, timeout_seconds: int = 180) -> None:
        if tuple(metrics) != METRIC_NAMES:
            raise ValueError(f"RAGAS evaluator requires exactly these metrics: {METRIC_NAMES}")
        self.metrics = metrics
        self.sample_class = sample_class
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_models(
        cls,
        evaluator_model: str,
        embedding_model: str,
        max_completion_tokens: int,
        timeout_seconds: int,
    ) -> "RagasEvaluator":
        metrics, sample_class = build_metrics(
            evaluator_model=evaluator_model,
            embedding_model=embedding_model,
            max_completion_tokens=max_completion_tokens,
        )
        return cls(metrics, sample_class, timeout_seconds)

    async def score(
        self,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
        reference: str,
        include_context_metrics: bool = True,
    ) -> dict[str, Any]:
        """Score one SingleTurnSample while isolating every metric failure."""

        sample = self.sample_class(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
        )
        if hasattr(sample, "model_dump"):
            fields = sample.model_dump(exclude_none=True)
        else:
            fields = {
                "user_input": sample.user_input,
                "response": sample.response,
                "retrieved_contexts": sample.retrieved_contexts,
                "reference": sample.reference,
            }

        scored: dict[str, Any] = {}
        errors: list[str] = []
        for name, metric in self.metrics.items():
            if not include_context_metrics and name in ("context_precision", "context_recall", "faithfulness"):
                scored[name] = None
                continue
            kwargs = {field: fields[field] for field in METRIC_FIELDS[name]}
            try:
                result = await asyncio.wait_for(
                    metric.ascore(**kwargs), timeout=self.timeout_seconds
                )
                value = getattr(result, "value", result)
                scored[name] = None if value is None else float(value)
                if scored[name] is not None and not math.isfinite(scored[name]):
                    raise ValueError(f"non-finite metric value: {scored[name]}")
            except Exception as exc:
                scored[name] = None
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if errors:
            scored["metric_errors"] = errors
        return scored
