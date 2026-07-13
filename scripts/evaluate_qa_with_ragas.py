#!/usr/bin/env python3
"""Evaluate grounded QA datasets with standard Ragas metrics.

Typical usage:
    python scripts/evaluate_qa_with_ragas.py --input datasets/obstetrics/qa/final/train/raw.jsonl

The script computes Faithfulness and Answer Relevancy in two internal passes
and writes a single final report. This split avoids hangs observed in this
environment during combined execution.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

# Load environment variables from .env file (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
load_dotenv()


class CustomQuality(BaseModel):
    faithfulness: float
    answer_relevancy: float
    roundtrip_consistency: float
    question_groundness: float
    verdict: str
    reason: str

    @field_validator(
        "faithfulness", "answer_relevancy", "roundtrip_consistency",
        "question_groundness", mode="before",
    )
    @classmethod
    def clamp_score(cls, value: Any) -> float:
        score = float(value)
        return max(0.0, min(1.0, score))

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, value: Any) -> str:
        verdict = str(value).strip().lower()
        return verdict if verdict in {"accept", "reject"} else "reject"


def _detect_transport(custom_judge_model: str | None) -> str:
    """Return which transport will be used for the custom judge."""
    if not custom_judge_model:
        return "none"
    try:
        import litellm  # noqa: F811
        return "litellm"
    except ImportError:
        return "openai_native"


ROUNDTRIP_SYSTEM_PROMPT = """\
Eres un asistente clínico. Responde con base en el contexto provisto.
Puedes usar conocimiento médico general únicamente para mejorar redacción y claridad,
pero no introduzcas hechos nuevos que no estén respaldados por el contexto.
Si algo no está suficientemente respaldado por el contexto, responde:
"No hay evidencia suficiente en el contexto."
"""

ROUNDTRIP_USER_TEMPLATE = """\
<contexto>
{context}
</contexto>

Pregunta:
{question}
"""

CUSTOM_JUDGE_SYSTEM_PROMPT = """\
Eres un evaluador estricto de calidad para datasets QA médicos.
Evalúa:
1) faithfulness: qué tan respaldada está la respuesta por el contexto.
2) answer_relevancy: qué tan bien responde la pregunta.
3) roundtrip_consistency: qué tan consistente es con una segunda respuesta independiente.
4) question_groundness: si la pregunta es respondible usando únicamente información del contexto.
   Una pregunta bien fundamentada se puede responder con el contexto solo; no asume hechos externos,
   no es ambigua ni trivial, y no pide información que el contexto no contiene.
Retorna puntajes [0,1], verdict ("accept" o "reject") y reason breve.
"""

CUSTOM_JUDGE_USER_TEMPLATE = """\
Contexto fuente:
{context}

Pregunta generada:
{question}

Respuesta generada:
{answer}

Respuesta independiente (roundtrip):
{roundtrip_answer}
"""


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object, accepting common markdown fenced responses."""
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(candidate[start : end + 1])
        raise


META_REFERENCE_PATTERNS = [
    "según el texto",
    "según el fragmento",
    "según la tabla",
    "de acuerdo con el fragmento",
    "de acuerdo al fragmento",
    "con base en el texto",
    "basado en el texto",
    "el fragmento dice",
    "el contexto señala",
    "el contexto compartido",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_eval_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_chunk_lookup(lm_paths: List[Path]) -> Dict[str, str]:
    """Load LM chunk text indexed by chunk_id for honest RAGAS evaluation."""
    lookup: Dict[str, str] = {}
    for lm_path in lm_paths:
        if not lm_path.exists():
            print(f"[EVAL] LM chunks file not found (skipped): {lm_path}", flush=True)
            continue
        for line in lm_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            cid = record.get("metadata", {}).get("chunk_id")
            text = record.get("text")
            if cid and text:
                lookup[cid] = text
    print(f"[EVAL] Loaded {len(lookup)} LM chunks for source-context lookup", flush=True)
    return lookup


def resolve_chunk_text(row: Dict[str, Any], chunk_lookup: Dict[str, str]) -> str:
    """Return the full source chunk text for a QA row, falling back to contexto_fuente."""
    cid = row.get("chunk_id")
    if cid and cid in chunk_lookup:
        return chunk_lookup[cid]
    return str(row.get("contexto_fuente") or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula Faithfulness y Answer Relevancy con Ragas real."
    )
    parser.add_argument("--input", type=Path, required=True, help="Archivo raw_*.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Reporte JSON de salida. Si se omite, se crea eval_<nombre_input>.json en la misma carpeta.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evalúa solo los primeros N pares (útil para validar rápido).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Evalúa una muestra aleatoria estratificada por source_pdf. Recomendado para dataset final.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad cuando se usa --sample-size.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Timeout máximo por métrica/par.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="Modelo evaluador usado por Ragas.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Modelo de embeddings para Answer Relevancy.",
    )
    parser.add_argument(
        "--custom-judge-model",
        type=str,
        default=None,
        help=(
            "Opcional. Si se define, además de Ragas calcula la validación custom "
            "sobre la misma muestra usando este modelo."
        ),
    )
    parser.add_argument(
        "--custom-only",
        action="store_true",
        help=(
            "Ejecuta solo la validación custom y omite las pasadas de Ragas. "
            "Requiere --custom-judge-model."
        ),
    )
    parser.add_argument(
        "--custom-debug-raw",
        action="store_true",
        help=(
            "En errores del custom judge, incluye la respuesta cruda del judge "
            "y la respuesta roundtrip para diagnosticar parseos fallidos."
        ),
    )
    parser.add_argument(
        "--existing-report",
        type=Path,
        default=None,
        help=(
            "Reporte eval_*.json existente con `per_pair`. Útil para recalcular "
            "solo custom sobre exactamente los mismos pares, preservando RAGAS."
        ),
    )
    parser.add_argument(
        "--lm-chunks",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Ruta(s) al archivo LM chunks (ej. datasets/obstetrics/lm/train_lm.jsonl). "
            "Si se omite, se intenta derivar automáticamente del --input. "
            "Se usa el texto completo del chunk como contexto para RAGAS Faithfulness "
            "en lugar del contexto_fuente co-generado."
        ),
    )
    return parser.parse_args()


def stratified_sample(
    rows: List[Dict[str, Any]],
    sample_size: int | None,
    seed: int,
) -> List[Dict[str, Any]]:
    """Return a reproducible sample while avoiding PDF-level dominance."""
    if sample_size is None or sample_size >= len(rows):
        return rows
    if sample_size <= 0:
        return []

    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("source_pdf") or "unknown")
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        rng.shuffle(group_rows)

    total = len(rows)
    quotas = []
    for key, group_rows in groups.items():
        raw_quota = sample_size * len(group_rows) / total
        base = int(raw_quota)
        if base == 0 and len(groups) <= sample_size:
            base = 1
        quotas.append((key, base, raw_quota - int(raw_quota)))

    selected: List[Dict[str, Any]] = []
    for key, quota, _frac in quotas:
        selected.extend(groups[key][:quota])

    remaining = sample_size - len(selected)
    if remaining > 0:
        used_ids = {id(row) for row in selected}
        leftovers = [row for group_rows in groups.values() for row in group_rows if id(row) not in used_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining])
    elif remaining < 0:
        rng.shuffle(selected)
        selected = selected[:sample_size]

    rng.shuffle(selected)
    return selected


def has_meta_reference(text: str) -> bool:
    lower = str(text).lower()
    return any(pattern in lower for pattern in META_REFERENCE_PATTERNS)


def basic_cleanliness(row: Dict[str, Any]) -> Dict[str, Any]:
    question = str(row.get("pregunta", ""))
    answer = str(row.get("respuesta", ""))
    return {
        "question_has_meta_reference": has_meta_reference(question),
        "answer_has_meta_reference": has_meta_reference(answer),
        "question_mark_count": question.count("?") + question.count("¿"),
        "answer_word_count": len(answer.split()),
    }


def extract_metric_value(result: Any) -> Any:
    """Extract a numeric metric value from multiple possible result formats."""
    if isinstance(result, (int, float)):
        return float(result)
    if isinstance(result, dict):
        return result.get("value")
    return getattr(result, "value", None)


def extract_metric_reason(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("reason", "") or "")
    return str(getattr(result, "reason", "") or "")


async def evaluate_rows(
    rows: List[Dict[str, Any]],
    llm_model: str,
    embedding_model: str,
    timeout_seconds: int,
    metric: str,
    chunk_lookup: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings.base import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness
    except ImportError as exc:
        raise SystemExit(
            "Faltan dependencias para Ragas. Ejecuta: pip install -r requirements.txt"
        ) from exc

    client = AsyncOpenAI()
    llm = llm_factory(llm_model, client=client)
    # Compatibility patch for dotted GPT-5 model IDs (e.g., gpt-5.5):
    # Ragas may not remap max_tokens -> max_completion_tokens for these names.
    model_lower = llm_model.lower()
    if model_lower.startswith("gpt-5."):
        if hasattr(llm, "model_args") and isinstance(llm.model_args, dict):
            llm.model_args.pop("max_tokens", None)
            llm.model_args.setdefault("max_completion_tokens", 1024)
            llm.model_args["temperature"] = 1.0
            llm.model_args.pop("top_p", None)
    embeddings = embedding_factory("openai", model=embedding_model, client=client)
    faithfulness = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

    results: List[Dict[str, Any]] = []
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        try:
            source_context = resolve_chunk_text(row, chunk_lookup or {})
            contexts = [source_context]
            faith_value = None
            rel_value = None
            faith_reason = ""
            rel_reason = ""
            if metric == "faithfulness":
                print(f"[RAGAS] par {i}/{total} -> faithfulness...", flush=True)
                faith = await asyncio.wait_for(
                    faithfulness.ascore(
                        user_input=row["pregunta"],
                        response=row["respuesta"],
                        retrieved_contexts=contexts,
                    ),
                    timeout=timeout_seconds,
                )
                faith_value = extract_metric_value(faith)
                faith_reason = extract_metric_reason(faith)
                if faith_value is None:
                    raise RuntimeError(f"Ragas devolvió faithfulness vacío. reason={faith_reason}")
            elif metric == "answer_relevancy":
                print(f"[RAGAS] par {i}/{total} -> answer_relevancy...", flush=True)
                rel = await asyncio.wait_for(
                    relevancy.ascore(
                        user_input=row["pregunta"],
                        response=row["respuesta"],
                    ),
                    timeout=timeout_seconds,
                )
                rel_value = extract_metric_value(rel)
                rel_reason = extract_metric_reason(rel)
                if rel_value is None:
                    raise RuntimeError(f"Ragas devolvió answer_relevancy vacío. reason={rel_reason}")
            else:
                raise ValueError(f"Métrica no soportada: {metric}")
            results.append(
                {
                    "qa_id": row["qa_id"],
                    "chunk_id": row["chunk_id"],
                    "ragas_faithfulness": float(faith_value) if faith_value is not None else None,
                    "ragas_answer_relevancy": float(rel_value) if rel_value is not None else None,
                    "ragas_faithfulness_reason": faith_reason,
                    "ragas_answer_relevancy_reason": rel_reason,
                }
            )
        except Exception as exc:
            error_row = {
                "qa_id": row.get("qa_id"),
                "chunk_id": row.get("chunk_id"),
                "ragas_faithfulness": None,
                "ragas_answer_relevancy": None,
                "error_type": type(exc).__name__,
                "error": repr(exc),
            }
            results.append(error_row)
        print(f"[RAGAS] {i}/{total} pares evaluados", flush=True)
    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    faith = [r["ragas_faithfulness"] for r in results if r.get("ragas_faithfulness") is not None]
    rel = [r["ragas_answer_relevancy"] for r in results if r.get("ragas_answer_relevancy") is not None]
    custom_faith = [r["custom_faithfulness"] for r in results if r.get("custom_faithfulness") is not None]
    custom_rel = [r["custom_answer_relevancy"] for r in results if r.get("custom_answer_relevancy") is not None]
    custom_rt = [r["custom_roundtrip_consistency"] for r in results if r.get("custom_roundtrip_consistency") is not None]
    custom_qg = [r["custom_question_groundness"] for r in results if r.get("custom_question_groundness") is not None]
    errors = sum(1 for r in results if "error" in r)
    return {
        "pairs_evaluated": len(results),
        "pairs_scored_faithfulness": len(faith),
        "pairs_scored_answer_relevancy": len(rel),
        "pairs_with_error": errors,
        "avg_ragas_faithfulness": round(statistics.mean(faith), 4) if faith else None,
        "min_ragas_faithfulness": round(min(faith), 4) if faith else None,
        "max_ragas_faithfulness": round(max(faith), 4) if faith else None,
        "avg_ragas_answer_relevancy": round(statistics.mean(rel), 4) if rel else None,
        "min_ragas_answer_relevancy": round(min(rel), 4) if rel else None,
        "max_ragas_answer_relevancy": round(max(rel), 4) if rel else None,
        "custom_pairs_scored": len(custom_faith),
        "custom_acceptance_rate": (
            round(sum(1 for r in results if r.get("custom_verdict") == "accept") / len(custom_faith), 4)
            if custom_faith
            else None
        ),
        "avg_custom_faithfulness": round(statistics.mean(custom_faith), 4) if custom_faith else None,
        "avg_custom_answer_relevancy": round(statistics.mean(custom_rel), 4) if custom_rel else None,
        "avg_custom_roundtrip_consistency": round(statistics.mean(custom_rt), 4) if custom_rt else None,
        "avg_custom_question_groundness": round(statistics.mean(custom_qg), 4) if custom_qg else None,
        "questions_with_meta_reference": sum(1 for r in results if r.get("question_has_meta_reference")),
        "answers_with_meta_reference": sum(1 for r in results if r.get("answer_has_meta_reference")),
    }


async def evaluate_custom_quality(
    rows: List[Dict[str, Any]],
    model: str,
    timeout_seconds: int,
    chunk_lookup: Dict[str, str] | None = None,
    debug_raw: bool = False,
) -> List[Dict[str, Any]]:
    # ── provider routing: litellm for multi-provider, OpenAI fallback ──
    _use_litellm = False
    try:
        import litellm  # noqa: F811
        _use_litellm = True
    except ImportError:
        pass

    _oa_client = None
    if not _use_litellm:
        from openai import AsyncOpenAI
        _oa_client = AsyncOpenAI()

    results: List[Dict[str, Any]] = []
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        print(f"[CUSTOM] par {i}/{total} -> quality judge...", flush=True)
        roundtrip_answer = ""
        judge_text = ""
        try:
            context = resolve_chunk_text(row, chunk_lookup or {})

            # ── roundtrip generation ──
            if _use_litellm:
                rt_resp = await litellm.acompletion(
                    model=model,
                    messages=[
                        {"role": "system", "content": ROUNDTRIP_SYSTEM_PROMPT},
                        {"role": "user", "content": ROUNDTRIP_USER_TEMPLATE.format(
                            context=context,
                            question=row.get("pregunta", ""),
                        )},
                    ],
                    max_tokens=1024,
                    timeout=timeout_seconds,
                )
                roundtrip_answer = (rt_resp.choices[0].message.content or "").strip()
            else:
                rt_resp = await asyncio.wait_for(
                    _oa_client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": ROUNDTRIP_SYSTEM_PROMPT},
                            {"role": "user", "content": ROUNDTRIP_USER_TEMPLATE.format(
                                context=context,
                                question=row.get("pregunta", ""),
                            )},
                        ],
                    ),
                    timeout=timeout_seconds,
                )
                roundtrip_answer = (rt_resp.choices[0].message.content or "").strip()

            # ── quality judge ──
            if _use_litellm:
                judge_system = (
                    CUSTOM_JUDGE_SYSTEM_PROMPT
                    + "\n\nResponde ÚNICAMENTE con un objeto JSON. Usa los campos exactos: "
                    "faithfulness (float), answer_relevancy (float), "
                    "roundtrip_consistency (float), question_groundness (float), "
                    "verdict (string: \"accept\" o \"reject\"), reason (string breve)."
                )
                judge_raw = await litellm.acompletion(
                    model=model,
                    messages=[
                        {"role": "system", "content": judge_system},
                        {"role": "user", "content": CUSTOM_JUDGE_USER_TEMPLATE.format(
                            context=context,
                            question=row.get("pregunta", ""),
                            answer=row.get("respuesta", ""),
                            roundtrip_answer=roundtrip_answer,
                        )},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=1024,
                    timeout=timeout_seconds,
                )
                judge_text = (judge_raw.choices[0].message.content or "{}").strip()
                data = parse_json_object(judge_text)
                parsed = CustomQuality.model_validate(
                    {
                        "faithfulness": data.get("faithfulness", 0.0),
                        "answer_relevancy": data.get("answer_relevancy", 0.0),
                        "roundtrip_consistency": data.get("roundtrip_consistency", 0.0),
                        "question_groundness": data.get("question_groundness", 0.0),
                        "verdict": data.get("verdict", "reject"),
                        "reason": data.get("reason", ""),
                    }
                )
                parsed_data = {
                    "faithfulness": parsed.faithfulness,
                    "answer_relevancy": parsed.answer_relevancy,
                    "roundtrip_consistency": parsed.roundtrip_consistency,
                    "question_groundness": parsed.question_groundness,
                    "verdict": parsed.verdict,
                    "reason": parsed.reason,
                }
            else:
                judge_resp = await asyncio.wait_for(
                    _oa_client.chat.completions.parse(
                        model=model,
                        messages=[
                            {"role": "system", "content": CUSTOM_JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": CUSTOM_JUDGE_USER_TEMPLATE.format(
                                context=context,
                                question=row.get("pregunta", ""),
                                answer=row.get("respuesta", ""),
                                roundtrip_answer=roundtrip_answer,
                            )},
                        ],
                        response_format=CustomQuality,
                    ),
                    timeout=timeout_seconds,
                )
                parsed = judge_resp.choices[0].message.parsed
                if parsed is None:
                    raise RuntimeError("custom_judge_parse_none")
                parsed_data = {
                    "faithfulness": parsed.faithfulness,
                    "answer_relevancy": parsed.answer_relevancy,
                    "roundtrip_consistency": parsed.roundtrip_consistency,
                    "question_groundness": parsed.question_groundness,
                    "verdict": parsed.verdict,
                    "reason": parsed.reason,
                }

            results.append(
                {
                    "qa_id": row.get("qa_id"),
                    "custom_faithfulness": parsed_data["faithfulness"],
                    "custom_answer_relevancy": parsed_data["answer_relevancy"],
                    "custom_roundtrip_consistency": parsed_data["roundtrip_consistency"],
                    "custom_question_groundness": parsed_data["question_groundness"],
                    "custom_verdict": parsed_data["verdict"],
                    "custom_reason": parsed_data["reason"],
                }
            )
        except Exception as exc:
            error_row = {
                "qa_id": row.get("qa_id"),
                "custom_faithfulness": None,
                "custom_answer_relevancy": None,
                "custom_roundtrip_consistency": None,
                "custom_question_groundness": None,
                "custom_verdict": None,
                "custom_reason": "",
                "custom_error_type": type(exc).__name__,
                "custom_error": repr(exc),
            }
            if debug_raw:
                error_row["custom_roundtrip_answer"] = roundtrip_answer
                error_row["custom_judge_raw"] = judge_text
            results.append(error_row)
    return results


async def main_async(args: argparse.Namespace) -> None:
    if args.custom_only and not args.custom_judge_model:
        raise ValueError("--custom-only requiere --custom-judge-model")

    existing_report: Dict[str, Any] | None = None
    if args.existing_report is not None:
        existing_report = read_eval_report(args.existing_report)
        all_rows = list(existing_report.get("per_pair") or [])
        rows = all_rows
        if args.limit is not None:
            rows = rows[: args.limit]
    else:
        all_rows = read_jsonl(args.input)
        rows = stratified_sample(all_rows, args.sample_size, args.seed)
        if args.limit is not None:
            rows = rows[: args.limit]
    print(
        f"[EVAL] Iniciando evaluación de {len(rows)} pares "
        f"(dataset completo: {len(all_rows)})",
        flush=True,
    )

    # ── methodology guard: prevent self-evaluation bias ──
    if args.custom_judge_model and args.custom_judge_model == args.llm_model:
        print(
            "[EVAL] ⚠  WARNING: --custom-judge-model is the same as --llm-model "
            f"({args.llm_model}). This means the same model evaluates its own generated "
            "QA — introducing self-evaluation bias. For honest methodology, use a different model "
            "(e.g., --custom-judge-model gpt-4o while --llm-model gpt-4o-mini).",
            flush=True,
        )

    # Resolve LM chunk paths for honest source-context evaluation.
    lm_paths: List[Path] = list(args.lm_chunks or [])
    if not lm_paths:
        # Auto-derive from input path convention:
        #   datasets/obstetrics/qa/final/{split}/raw.jsonl
        #   -> datasets/obstetrics/lm/{split}_lm.jsonl
        input_str = str(args.input)
        for split_name in ("train", "validation", "test"):
            if f"/qa/final/{split_name}/" in input_str:
                candidate = Path(f"datasets/obstetrics/lm/{split_name}_lm.jsonl")
                if candidate.exists():
                    lm_paths.append(candidate)
                    print(f"[EVAL] Auto-derived LM chunks: {candidate}", flush=True)
                break
        if not lm_paths:
            print(
                "[EVAL] No --lm-chunks provided and could not auto-derive path. "
                "Falling back to contexto_fuente (co-generated excerpt). "
                "Pass --lm-chunks for honest source-context evaluation.",
                flush=True,
            )

    chunk_lookup = build_chunk_lookup(lm_paths) if lm_paths else {}

    # User-facing behavior: one command.
    # Internal implementation: two metric passes with a final merge.
    if args.custom_only:
        print("[CUSTOM] Modo custom-only: omitiendo métricas Ragas.", flush=True)
        faith_rows = []
        rel_rows = []
    else:
        print("[RAGAS] Calculando faithfulness...", flush=True)
        faith_rows = await evaluate_rows(
            rows,
            args.llm_model,
            args.embedding_model,
            args.timeout_seconds,
            metric="faithfulness",
            chunk_lookup=chunk_lookup or None,
        )
        print("[RAGAS] Calculando answer_relevancy...", flush=True)
        rel_rows = await evaluate_rows(
            rows,
            args.llm_model,
            args.embedding_model,
            args.timeout_seconds,
            metric="answer_relevancy",
            chunk_lookup=chunk_lookup or None,
        )

    faith_by_id = {r.get("qa_id"): r for r in faith_rows}
    rel_by_id = {r.get("qa_id"): r for r in rel_rows}

    custom_by_id: Dict[str, Dict[str, Any]] = {}
    if args.custom_judge_model:
        print(f"[CUSTOM] Calculando validación custom con {args.custom_judge_model}...", flush=True)
        custom_rows = await evaluate_custom_quality(
            rows, args.custom_judge_model, args.timeout_seconds,
            chunk_lookup=chunk_lookup or None,
            debug_raw=args.custom_debug_raw,
        )
        custom_by_id = {r.get("qa_id"): r for r in custom_rows}

    results = []
    for row in rows:
        qa_id = row.get("qa_id")
        f = faith_by_id.get(qa_id, {})
        r = rel_by_id.get(qa_id, {})
        c = custom_by_id.get(qa_id, {})
        merged = {
            "qa_id": qa_id,
            "chunk_id": row.get("chunk_id"),
            "source_pdf": row.get("source_pdf"),
            "tipo": row.get("tipo"),
            "dificultad": row.get("dificultad"),
            "pregunta": row.get("pregunta"),
            "respuesta": row.get("respuesta"),
            "ragas_faithfulness": f.get("ragas_faithfulness", row.get("ragas_faithfulness")),
            "ragas_answer_relevancy": r.get("ragas_answer_relevancy", row.get("ragas_answer_relevancy")),
            "ragas_faithfulness_reason": f.get("ragas_faithfulness_reason", row.get("ragas_faithfulness_reason", "")),
            "ragas_answer_relevancy_reason": r.get("ragas_answer_relevancy_reason", row.get("ragas_answer_relevancy_reason", "")),
            "custom_faithfulness": c.get("custom_faithfulness"),
            "custom_answer_relevancy": c.get("custom_answer_relevancy"),
            "custom_roundtrip_consistency": c.get("custom_roundtrip_consistency"),
            "custom_question_groundness": c.get("custom_question_groundness"),
            "custom_verdict": c.get("custom_verdict"),
            "custom_reason": c.get("custom_reason", ""),
            **basic_cleanliness(row),
        }
        if args.custom_debug_raw:
            if "custom_roundtrip_answer" in c:
                merged["custom_roundtrip_answer"] = c.get("custom_roundtrip_answer")
            if "custom_judge_raw" in c:
                merged["custom_judge_raw"] = c.get("custom_judge_raw")
        errs = []
        if f.get("error"):
            errs.append(f"faithfulness: {f.get('error_type','Error')} {f.get('error')}")
        if r.get("error"):
            errs.append(f"answer_relevancy: {r.get('error_type','Error')} {r.get('error')}")
        if c.get("custom_error"):
            errs.append(f"custom_judge: {c.get('custom_error_type','Error')} {c.get('custom_error')}")
        if errs:
            merged["error_type"] = "MetricPassError"
            merged["error"] = " | ".join(errs)
        results.append(merged)

    output = args.output
    if output is None:
        if args.existing_report is not None:
            output = args.existing_report.with_name(f"{args.existing_report.stem}_custom.json")
        else:
            prefix = f"eval_sample{len(rows)}_" if args.sample_size else "eval_"
            output = args.input.with_name(f"{prefix}{args.input.stem}.json")

    report = {
        "script_version": "qa_sample_eval_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(args.input),
        "total_pairs_in_input": len(all_rows),
        "sampled_pairs": len(rows),
        "sample_size_requested": args.sample_size,
        "sample_seed": args.seed if args.sample_size else None,
        "execution": "single_command_two_internal_passes",
        "source_context": {
            "method": "full_source_chunk" if chunk_lookup else "contexto_fuente_excerpt",
            "lm_chunk_files": [str(p) for p in lm_paths],
            "chunks_loaded": len(chunk_lookup),
        },
        "ragas_models": {
            "llm_model": args.llm_model,
            "embedding_model": args.embedding_model,
        },
        "custom_judge_model": args.custom_judge_model,
        "custom_judge_transport": _detect_transport(args.custom_judge_model),
        "summary": summarize(results),
        "per_pair": results,
    }
    if existing_report is not None:
        report["input_file"] = existing_report.get("input_file", str(args.input))
        report["sample_size_requested"] = existing_report.get("sample_size_requested")
        report["sample_seed"] = existing_report.get("sample_seed")
        report["source_context"] = existing_report.get("source_context", report["source_context"])
        report["ragas_models"] = existing_report.get("ragas_models", report["ragas_models"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[RAGAS] Reporte guardado en: {output}", flush=True)


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
