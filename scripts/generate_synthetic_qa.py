"""
generate_synthetic_qa.py
========================
Generate synthetic Spanish Question/Answer pairs from chunks already
processed in `datasets/obstetrics/lm/train_lm.jsonl`.

Uses OpenAI Structured Outputs (gpt-4o-mini / gpt-4o) to guarantee valid
JSON aligned with the Pydantic schema. All records are written in
`messages` format (SFT-ready) for Unsloth or HuggingFace TRL.

Flow:
  1. Read chunks from `train_lm.jsonl` (produced by `build_obstetrics_lm_dataset.py`).
  2. Determine how many pairs to generate per chunk from `token_estimate` and `clinical_score`.
  3. Call the API asynchronously (configurable semaphore).
  4. Save incremental checkpoints to support safe resume.
  5. Write two files:
       - `synthetic_qa_raw.jsonl` -> raw pairs with audit metadata
       - `synthetic_qa_sft.jsonl` -> {"messages": [...], "metadata": {...}}
                                     ready for SFT/QLoRA training.

Basic usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/generate_synthetic_qa.py

Dry-run (without API calls):
    python scripts/generate_synthetic_qa.py --dry-run

Cost estimate (856 chunks, May 2026):
    gpt-5.4-mini  ≈ $1.10 USD (budget recommendation)
    gpt-5.4       ≈ $4.50 USD (quality/cost balance)
    gpt-5.5       ≈ $8.00 USD (maximum clinical quality)

Note: gpt-4o and gpt-4o-mini were deprecated in February 2026.

Minimum required versions:
    openai>=1.68.0   (stable structured outputs, without .beta.)
    pydantic>=2.7.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, field_validator
from tqdm.asyncio import tqdm as atqdm

# ---------------------------------------------------------------------------
# Structured-output schemas returned by the OpenAI generation and judge calls.
# ---------------------------------------------------------------------------

TipoPregunta = Literal[
    "factual",
    "razonamiento",
    "definicion",
    "comparacion",
    "aplicacion",
    "hipotetico",
]
NivelDificultad = Literal["basico", "intermedio", "avanzado"]


class QAPar(BaseModel):
    pregunta: str
    respuesta: str
    tipo: TipoPregunta
    dificultad: NivelDificultad
    contexto_fuente: str

    @field_validator("pregunta", "respuesta", "contexto_fuente", mode="before")
    @classmethod
    def no_empty(cls, v: Any) -> str:
        text = str(v).strip()
        if not text:
            raise ValueError("El campo no puede estar vacío.")
        return text


class RespuestaGeneracion(BaseModel):
    pares: List[QAPar]


class PairQuality(BaseModel):
    faithfulness: float
    answer_relevancy: float
    roundtrip_consistency: float
    verdict: Literal["accept", "reject"]
    reason: str

    @field_validator("faithfulness", "answer_relevancy", "roundtrip_consistency", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> float:
        x = float(v)
        return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# This is the system prompt stored inside each SFT example, so it defines the
# behavior the fine-tuned model will learn at inference time. Keep it narrower
# than generation-time policy: it should describe the target assistant, not the
# data-construction process.
SFT_SYSTEM_PROMPT = (
    "Eres un asistente especializado en obstetricia y ginecología. "
    "Responde en español con precisión clínica, claridad y vocabulario médico apropiado. "
    "Prioriza respuestas fieles a la evidencia disponible, bien estructuradas y útiles "
    "para resolver la pregunta planteada. Si la evidencia disponible no permite una "
    "afirmación concluyente, indícalo con claridad en lugar de inventar detalles."
)

# Generation-time prompt used to create candidate QA pairs from each source chunk.
GENERATION_SYSTEM_PROMPT = """\
Eres un experto en generación de datasets de entrenamiento para modelos de lenguaje.
Tu tarea es leer fragmentos de una base de conocimiento médica en obstetricia y ginecología,
y generar pares de pregunta-respuesta de alta calidad en español para fine-tuning
supervisado (SFT).

Reglas estrictas:
0. Usa un enfoque "evidence-first": primero identifica evidencia en el contexto y después construye la pregunta y respuesta.
1. Las preguntas deben ser DIVERSAS en tipo: factuales, de razonamiento, de definición,
   de comparación, de aplicación práctica y de "qué pasa si" (hipotético).
2. Las respuestas deben ser completas y precisas, priorizando SIEMPRE el contexto dado como fuente principal.
   Puedes usar conocimiento médico general solo para redactar con claridad y coherencia clínica.
   Si el contexto es insuficiente o ambiguo para una afirmación, indícalo explícitamente y no fabriques detalles.
3. Usa español natural y vocabulario médico correcto; no traduzcas literalmente del inglés.
4. Varía la longitud de las respuestas: algunas cortas (1-2 oraciones) y otras desarrolladas
   (varios párrafos si el tema lo amerita).
5. El campo "contexto_fuente" debe ser un fragmento breve (máximo 2 oraciones del texto)
   que respalda directamente la respuesta.
6. Distribuye los tipos a lo largo de los pares: no repitas el mismo tipo consecutivamente
   si tienes más de 2 pares.
7. Para preguntas de tipo "aplicacion" usa viñetas clínicas cortas (paciente con X condición).
8. Las preguntas deben quedar limpias y autocontenidas: NO escribas frases como
   "según el texto", "según el fragmento", "de acuerdo con el fragmento",
   "con base en el texto", "según la tabla" ni expresiones equivalentes.
9. Cada pregunta debe contener UNA sola demanda central. Evita preguntas dobles
   unidas por "y", "además", "cuál es... y por qué...", etc. Si dos ideas merecen
   preguntarse, conviértelas en dos pares separados solo cuando cada una sea útil por sí misma.
10. No sacrifiques calidad por cantidad. Genera solo pares que el fragmento soporte bien;
    evita forzar tipos o preguntas adicionales si el contenido no lo permite.
11. Las respuestas también deben quedar limpias y autocontenidas: NO escribas
    "el fragmento dice", "el contexto señala", "según el texto" ni expresiones equivalentes.
    Responde directamente como si la pregunta fuera autónoma.
"""

# Per-chunk user message template sent to the generation model.
GENERATION_USER_TEMPLATE = """\
Documento fuente: {source_pdf}
Sección: {section}

<contexto>
{text}
</contexto>

Genera hasta {n_pairs} pares de pregunta-respuesta en español basados en el \
fragmento anterior. No intentes acercarte a ese número si el contenido no lo justifica:
si el fragmento solo permite 1 o 2 pares de alta calidad, genera solo esos.
"""

ROUNDTRIP_SYSTEM_PROMPT = """\
Eres un asistente clínico. Responde SOLO con base en el contexto provisto.
Puedes usar conocimiento médico general únicamente para mejorar redacción y claridad.
Si algo no está suficientemente respaldado por el contexto, responde: "No hay evidencia suficiente en el contexto."
Prioriza siempre el contexto sobre memoria general.
"""

ROUNDTRIP_USER_TEMPLATE = """\
<contexto>
{text}
</contexto>

Pregunta:
{question}
"""

QUALITY_JUDGE_SYSTEM_PROMPT = """\
Eres un evaluador estricto de calidad para datasets QA médicos.
Evalúa:
1) faithfulness: qué tan respaldada está la respuesta por el contexto.
2) answer_relevancy: qué tan bien responde la pregunta.
3) roundtrip_consistency: qué tan consistente es con una segunda respuesta independiente.
Retorna puntajes [0,1], verdict (accept/reject) y reason breve.
"""

QUALITY_JUDGE_USER_TEMPLATE = """\
Contexto:
{context}

Pregunta:
{question}

Respuesta original:
{answer}

Respuesta roundtrip:
{roundtrip_answer}
"""

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MAX_RETRIES = 5
BASE_BACKOFF_S = 2.0

# Active OpenAI API models as of May 2026.
# gpt-4o and gpt-4o-mini were deprecated in February 2026.
SUPPORTED_MODELS = (
    "gpt-5.2",
    "gpt-5.4-mini",  # Lower-cost, faster replacement for gpt-4o-mini.
    "gpt-5.4",  # Balanced quality/cost option.
    "gpt-5.5",  # Highest-quality option, with higher cost.
)

# Prices per million tokens (May 2026, source: platform.openai.com/api/docs/models).
PRICES_PER_M = {
    "gpt-5.2": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.5": {"input": 5.00, "output": 30.00},
}

# Conservative output-token estimate per generated QA pair.
TOKENS_PER_PAIR_OUTPUT_EST = 160


# ---------------------------------------------------------------------------
# File I/O helpers.
# ---------------------------------------------------------------------------


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_datasets_dir() -> Path:
    return project_root() / "datasets" / "obstetrics"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Append records so an interrupted run does not overwrite completed work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Checkpoint / progress
# ---------------------------------------------------------------------------


def load_progress(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("processed_chunk_ids", []))


def load_processed_ids_from_raw_output(path: Path) -> Set[str]:
    """Recover chunk IDs already written to raw output to avoid duplicates on resume."""
    if not path.exists():
        return set()
    processed: Set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = str(row.get("chunk_id", "")).strip()
            if chunk_id:
                processed.add(chunk_id)
    return processed


def save_progress(path: Path, processed_ids: Set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"processed_chunk_ids": sorted(processed_ids)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def status_path_for_progress(progress_file: Path) -> Path:
    return progress_file.with_name(f"{progress_file.stem}_status.json")


def save_run_status(path: Path, payload: Dict[str, Any]) -> None:
    """Persist human-readable run state for long jobs and safe resume checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Pair-count heuristic.
# ---------------------------------------------------------------------------


def n_pairs_for_chunk(
    token_estimate: int,
    clinical_score: int,
    min_pairs: int,
    max_pairs: int,
) -> int:
    """Choose how many QA pairs to request from a chunk.

    The heuristic biases generation toward larger and clinically richer chunks
    while respecting caller-provided limits:
    - small chunks (<400 tokens) request the minimum;
    - medium chunks (400-700 tokens) request one extra pair;
    - large chunks (>700 tokens) request near the maximum;
    - high clinical_score chunks get one additional pair, capped at max_pairs.
    """
    if token_estimate < 400:
        n = min_pairs
    elif token_estimate < 700:
        n = min_pairs + 1
    else:
        n = max(min_pairs + 1, max_pairs - 1)

    if clinical_score >= 20:
        n = min(n + 1, max_pairs)

    return min(max(n, min_pairs), max_pairs)


def estimate_cost(
    chunks: List[Dict[str, Any]],
    model: str,
    min_pairs: int,
    max_pairs: int,
) -> Tuple[int, float]:
    """Return the expected pair count and estimated generation cost in USD."""
    expected_pairs = sum(
        n_pairs_for_chunk(
            c.get("metadata", {}).get("token_estimate", 500),
            c.get("metadata", {}).get("clinical_score", 0),
            min_pairs,
            max_pairs,
        )
        for c in chunks
    )
    total_input_tokens = sum(
        c.get("metadata", {}).get("token_estimate", 500) for c in chunks
    )
    total_output_tokens = expected_pairs * TOKENS_PER_PAIR_OUTPUT_EST

    prices = PRICES_PER_M.get(model, PRICES_PER_M["gpt-5.4-mini"])
    cost = (total_input_tokens / 1_000_000) * prices["input"] + (
        total_output_tokens / 1_000_000
    ) * prices["output"]
    return expected_pairs, cost


# ---------------------------------------------------------------------------
# SFT record conversion.
# ---------------------------------------------------------------------------


def chunk_to_sft_records(
    chunk: Dict[str, Any],
    pairs: List[QAPar],
    qualities: Optional[List[Optional[PairQuality]]] = None,
) -> List[Dict[str, Any]]:
    """Convert generated QA pairs into chat-message records ready for fine-tuning."""
    meta = chunk.get("metadata", {})
    records = []
    for idx, pair in enumerate(pairs, start=1):
        q = (qualities[idx - 1] if qualities and idx - 1 < len(qualities) else None)
        chunk_id = str(meta.get("chunk_id", ""))
        qa_id = f"{chunk_id}_qa_{idx:03d}" if chunk_id else f"qa_{idx:03d}"
        records.append(
            {
                "messages": [
                    {"role": "system", "content": SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": pair.pregunta},
                    {"role": "assistant", "content": pair.respuesta},
                ],
                "metadata": {
                    "source": "obstetrics_spanish_synthetic",
                    "source_pdf": meta.get("source_pdf", ""),
                    "chunk_id": meta.get("chunk_id", ""),
                    "qa_id": qa_id,
                    "pages": meta.get("pages", []),
                    "section": meta.get("section", ""),
                    "section_type": meta.get("section_type", ""),
                    "content_role": meta.get("content_role", ""),
                    "topics": meta.get("topics", []) or meta.get("topic_tags", []),
                    "split": meta.get("split", ""),
                    "clinical_score": meta.get("clinical_score", 0),
                    "token_estimate": meta.get("token_estimate", 0),
                    "tipo": pair.tipo,
                    "dificultad": pair.dificultad,
                    "contexto_fuente": pair.contexto_fuente,
                    "faithfulness": q.faithfulness if q else None,
                    "answer_relevancy": q.answer_relevancy if q else None,
                    "roundtrip_consistency": q.roundtrip_consistency if q else None,
                    "quality_verdict": q.verdict if q else None,
                },
            }
        )
    return records


def chunk_to_raw_records(
    chunk: Dict[str, Any],
    pairs: List[QAPar],
    qualities: Optional[List[Optional[PairQuality]]] = None,
) -> List[Dict[str, Any]]:
    """Convert generated QA pairs into flat audit records for human review."""
    meta = chunk.get("metadata", {})
    rows: List[Dict[str, Any]] = []
    chunk_id = str(meta.get("chunk_id", ""))
    for idx, p in enumerate(pairs, start=1):
        q = (qualities[idx - 1] if qualities and idx - 1 < len(qualities) else None)
        qa_id = f"{chunk_id}_qa_{idx:03d}" if chunk_id else f"qa_{idx:03d}"
        rows.append(
            {
                "qa_id": qa_id,
                "chunk_id": meta.get("chunk_id", ""),
                "source_pdf": meta.get("source_pdf", ""),
                "section": meta.get("section", ""),
                "section_type": meta.get("section_type", ""),
                "content_role": meta.get("content_role", ""),
                "topics": meta.get("topics", []) or meta.get("topic_tags", []),
                "split": meta.get("split", ""),
                "pages": meta.get("pages", []),
                "clinical_score": meta.get("clinical_score", 0),
                "token_estimate": meta.get("token_estimate", 0),
                "pregunta": p.pregunta,
                "respuesta": p.respuesta,
                "tipo": p.tipo,
                "dificultad": p.dificultad,
                "contexto_fuente": p.contexto_fuente,
                "faithfulness": q.faithfulness if q else None,
                "answer_relevancy": q.answer_relevancy if q else None,
                "roundtrip_consistency": q.roundtrip_consistency if q else None,
                "quality_verdict": q.verdict if q else None,
                "quality_reason": q.reason if q else None,
            }
        )
    return rows


def _norm_tokens(text: str) -> Set[str]:
    return set(re.findall(r"\b\w+\b", str(text).lower(), flags=re.UNICODE))


def grounding_metrics_for_pairs(pairs: List[QAPar]) -> Dict[str, Any]:
    overlap_ratios: List[float] = []
    low_grounding = 0
    for pair in pairs:
        ctx = _norm_tokens(pair.contexto_fuente)
        ans = _norm_tokens(pair.respuesta)
        if not ctx:
            overlap_ratios.append(0.0)
            low_grounding += 1
            continue
        overlap = len(ctx & ans) / max(1, len(ctx))
        overlap_ratios.append(overlap)
        if overlap < 0.15:
            low_grounding += 1
    avg_overlap = sum(overlap_ratios) / max(1, len(overlap_ratios))
    return {
        "avg_context_answer_overlap": round(avg_overlap, 4),
        "low_grounding_pairs": low_grounding,
        "total_pairs": len(pairs),
    }


# ---------------------------------------------------------------------------
# Asynchronous generation.
# ---------------------------------------------------------------------------


async def generate_for_chunk(
    client: Any,  # AsyncOpenAI
    chunk: Dict[str, Any],
    model: str,
    min_pairs: int,
    max_pairs: int,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], List[QAPar], str]:
    """Call the API for a single chunk with exponential-backoff retries."""
    meta = chunk.get("metadata", {})
    chunk_id = meta.get("chunk_id", "unknown")
    n = n_pairs_for_chunk(
        meta.get("token_estimate", 500),
        meta.get("clinical_score", 0),
        min_pairs,
        max_pairs,
    )

    user_content = GENERATION_USER_TEMPLATE.format(
        source_pdf=meta.get("source_pdf", "documento"),
        section=meta.get("section", "sin sección") or "sin sección",
        text=chunk.get("text", "").strip(),
        n_pairs=n,
    )

    backoff = BASE_BACKOFF_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    response_format=RespuestaGeneracion,
                    temperature=0.75,
                )

            # Treat model refusals as terminal for this chunk so resume does not retry them forever.
            choice = response.choices[0]
            if getattr(choice.message, "refusal", None):
                logger.warning(
                    "Chunk %s fue rechazado por el modelo. Saltando.", chunk_id
                )
                return chunk, [], "refused"

            parsed = choice.message.parsed
            if parsed is None:
                raise ValueError("Structured output devolvió None.")

            return chunk, parsed.pares, "ok"

        except Exception as exc:
            # Import lazily so dry-runs and static inspection do not require the OpenAI package.
            try:
                from openai import APIStatusError, RateLimitError
            except ImportError:
                RateLimitError = None  # type: ignore
                APIStatusError = None  # type: ignore

            is_rate_limit = RateLimitError and isinstance(exc, RateLimitError)
            is_server_error = (
                APIStatusError
                and isinstance(exc, APIStatusError)
                and exc.status_code is not None
                and exc.status_code >= 500
            )

            if attempt == MAX_RETRIES:
                logger.error(
                    "Chunk %s: máximo de reintentos alcanzado (%d). Error: %s",
                    chunk_id,
                    MAX_RETRIES,
                    exc,
                )
                return chunk, [], "failed"

            if is_rate_limit or is_server_error:
                wait = backoff * (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.warning(
                    "Chunk %s: %s. Reintento %d/%d en %.1fs.",
                    chunk_id,
                    "Rate limit" if is_rate_limit else f"Error servidor ({exc})",
                    attempt,
                    MAX_RETRIES,
                    wait,
                )
                await asyncio.sleep(wait)
            else:
                logger.error("Chunk %s: error no recuperable: %s", chunk_id, exc)
                return chunk, [], "failed"

    return chunk, [], "failed"


async def roundtrip_answer_for_pair(
    client: Any,
    model: str,
    chunk_text: str,
    question: str,
) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ROUNDTRIP_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ROUNDTRIP_USER_TEMPLATE.format(text=chunk_text, question=question),
            },
        ],
    )
    content = response.choices[0].message.content or ""
    return content.strip()


async def quality_for_pair(
    client: Any,
    pair: QAPar,
    chunk_text: str,
    verifier_model: str,
) -> PairQuality:
    rt_answer = await roundtrip_answer_for_pair(
        client=client,
        model=verifier_model,
        chunk_text=chunk_text,
        question=pair.pregunta,
    )
    response = await client.chat.completions.parse(
        model=verifier_model,
        messages=[
            {"role": "system", "content": QUALITY_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": QUALITY_JUDGE_USER_TEMPLATE.format(
                    context=chunk_text,
                    question=pair.pregunta,
                    answer=pair.respuesta,
                    roundtrip_answer=rt_answer,
                ),
            },
        ],
        response_format=PairQuality,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        return PairQuality(
            faithfulness=0.0,
            answer_relevancy=0.0,
            roundtrip_consistency=0.0,
            verdict="reject",
            reason="quality_parse_none",
        )
    return parsed


async def evaluate_pairs_quality(
    client: Any,
    chunk: Dict[str, Any],
    pairs: List[QAPar],
    verifier_model: str,
    min_faithfulness: float,
    min_relevancy: float,
    min_roundtrip: float,
    logger: logging.Logger,
) -> List[PairQuality]:
    chunk_text = str(chunk.get("text", "")).strip()
    qualities: List[PairQuality] = []
    for pair in pairs:
        try:
            q = await quality_for_pair(
                client=client,
                pair=pair,
                chunk_text=chunk_text,
                verifier_model=verifier_model,
            )
            if (
                q.faithfulness < min_faithfulness
                or q.answer_relevancy < min_relevancy
                or q.roundtrip_consistency < min_roundtrip
            ):
                q.verdict = "reject"
            qualities.append(q)
        except Exception as exc:
            logger.warning("Fallo evaluación de calidad para una pareja QA: %s", exc)
            qualities.append(
                PairQuality(
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    roundtrip_consistency=0.0,
                    verdict="reject",
                    reason=f"quality_eval_error: {exc}",
                )
            )
    return qualities


async def run_generation(
    chunks: List[Dict[str, Any]],
    client: Any,
    model: str,
    min_pairs: int,
    max_pairs: int,
    concurrency: int,
    sft_output: Path,
    raw_output: Path,
    progress_file: Path,
    report_output: Path,
    quality_eval_enabled: bool,
    verifier_model: str,
    min_faithfulness: float,
    min_relevancy: float,
    min_roundtrip: float,
    quality_filter_enabled: bool,
    logger: logging.Logger,
) -> Dict[str, Any]:
    processed_ids = load_progress(progress_file)
    recovered_from_raw = load_processed_ids_from_raw_output(raw_output)
    if recovered_from_raw - processed_ids:
        logger.info(
            "Recuperados %d chunk_ids desde raw_output para reanudar sin duplicar.",
            len(recovered_from_raw - processed_ids),
        )
        processed_ids |= recovered_from_raw
        save_progress(progress_file, processed_ids)

    status_file = status_path_for_progress(progress_file)
    started_at = time.time()

    # Only schedule chunks that are not already complete; this keeps resume idempotent.
    pending = [
        c
        for c in chunks
        if c.get("metadata", {}).get("chunk_id", "") not in processed_ids
    ]
    already_processed_at_start = len(processed_ids)

    logger.info(
        "Chunks totales: %d | Ya procesados: %d | Pendientes: %d",
        len(chunks),
        len(processed_ids),
        len(pending),
    )

    save_run_status(
        status_file,
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": "running" if pending else "complete",
            "total_chunks": len(chunks),
            "already_processed_at_start": already_processed_at_start,
            "pending_at_start": len(pending),
            "processed_now": 0,
            "failed_now": 0,
            "qa_pairs_now": 0,
            "raw_output": str(raw_output),
            "sft_output": str(sft_output),
            "progress_file": str(progress_file),
        },
    )

    if not pending:
        logger.info("Todos los chunks ya fueron procesados.")
        return {
            "total": len(chunks),
            "skipped": len(chunks),
            "processed": 0,
            "failed": 0,
            "qa_pairs": 0,
        }

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        generate_for_chunk(
            client, chunk, model, min_pairs, max_pairs, semaphore, logger
        )
        for chunk in pending
    ]

    processed = 0
    failed = 0
    total_pairs = 0
    grounding_overlap_sum = 0.0
    grounding_pairs = 0
    low_grounding_pairs = 0
    quality_seen = 0
    quality_accepted = 0
    quality_faithfulness_sum = 0.0
    quality_relevancy_sum = 0.0
    quality_roundtrip_sum = 0.0

    # Stream completed tasks so outputs and checkpoints are flushed as soon as each chunk finishes.
    for coro in atqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Generando QA"
    ):
        chunk, pairs, status = await coro
        chunk_id = chunk.get("metadata", {}).get("chunk_id", "unknown")
        last_status = status

        if pairs:
            qualities: Optional[List[PairQuality]] = None
            if quality_eval_enabled:
                qualities = await evaluate_pairs_quality(
                    client=client,
                    chunk=chunk,
                    pairs=pairs,
                    verifier_model=verifier_model,
                    min_faithfulness=min_faithfulness,
                    min_relevancy=min_relevancy,
                    min_roundtrip=min_roundtrip,
                    logger=logger,
                )
                quality_seen += len(qualities)
                quality_accepted += sum(1 for q in qualities if q.verdict == "accept")
                quality_faithfulness_sum += sum(q.faithfulness for q in qualities)
                quality_relevancy_sum += sum(q.answer_relevancy for q in qualities)
                quality_roundtrip_sum += sum(q.roundtrip_consistency for q in qualities)
                if quality_filter_enabled:
                    kept_pairs: List[QAPar] = []
                    kept_qualities: List[PairQuality] = []
                    for pair, q in zip(pairs, qualities):
                        if q.verdict == "accept":
                            kept_pairs.append(pair)
                            kept_qualities.append(q)
                    pairs = kept_pairs
                    qualities = kept_qualities

            if not pairs:
                failed += 1
                processed_ids.add(chunk_id)
                save_progress(progress_file, processed_ids)
                continue

            append_jsonl(sft_output, chunk_to_sft_records(chunk, pairs, qualities))
            append_jsonl(raw_output, chunk_to_raw_records(chunk, pairs, qualities))
            gm = grounding_metrics_for_pairs(pairs)
            grounding_overlap_sum += float(gm["avg_context_answer_overlap"]) * int(gm["total_pairs"])
            grounding_pairs += int(gm["total_pairs"])
            low_grounding_pairs += int(gm["low_grounding_pairs"])
            processed += 1
            total_pairs += len(pairs)
            processed_ids.add(chunk_id)
            save_progress(progress_file, processed_ids)
            logger.info(
                "Chunk completado %s | pares=%d | procesados ahora=%d | pendientes=%d",
                chunk_id,
                len(pairs),
                processed,
                max(0, len(pending) - processed - failed),
            )
        elif status == "refused":
            failed += 1
            processed_ids.add(chunk_id)
            save_progress(progress_file, processed_ids)
            logger.warning("Chunk omitido por rechazo del modelo: %s", chunk_id)
        elif status == "ok":
            # An empty list is a valid model response. Mark it processed to avoid
            # retrying an unsupported/low-signal chunk indefinitely.
            failed += 1
            processed_ids.add(chunk_id)
            save_progress(progress_file, processed_ids)
            last_status = "empty"
            logger.warning("Chunk sin pares generados: %s", chunk_id)
        else:
            failed += 1
            logger.warning("Chunk fallido y reintentable al reanudar: %s", chunk_id)

        elapsed = max(0.001, time.time() - started_at)
        done_now = processed + failed
        rate = done_now / elapsed
        remaining = max(0, len(pending) - done_now)
        eta_seconds = int(remaining / rate) if rate > 0 else None
        save_run_status(
            status_file,
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "state": "running" if done_now < len(pending) else "complete",
                "total_chunks": len(chunks),
                "already_processed_at_start": already_processed_at_start,
                "pending_at_start": len(pending),
                "processed_now": processed,
                "failed_now": failed,
                "completed_now": done_now,
                "remaining_now": remaining,
                "qa_pairs_now": total_pairs,
                "last_chunk_id": chunk_id,
                "last_status": last_status,
                "elapsed_seconds": int(elapsed),
                "eta_seconds": eta_seconds,
                "raw_output": str(raw_output),
                "sft_output": str(sft_output),
                "progress_file": str(progress_file),
            },
        )

    stats = {
        "models": {
            "generator_model": model,
            "verifier_model": verifier_model if quality_eval_enabled else None,
        },
        "total": len(chunks),
        "skipped": len(chunks) - len(pending),
        "processed": processed,
        "failed": failed,
        "qa_pairs": total_pairs,
        "grounding": {
            "avg_context_answer_overlap": round(grounding_overlap_sum / max(1, grounding_pairs), 4),
            "low_grounding_pairs": low_grounding_pairs,
            "total_pairs": grounding_pairs,
            "low_grounding_rate": round(low_grounding_pairs / max(1, grounding_pairs), 4),
        },
        "quality": {
            "enabled": quality_eval_enabled,
            "verifier_model": verifier_model if quality_eval_enabled else None,
            "pairs_evaluated": quality_seen,
            "pairs_accepted": quality_accepted if quality_eval_enabled else None,
            "acceptance_rate": round(quality_accepted / max(1, quality_seen), 4) if quality_eval_enabled else None,
            "avg_faithfulness": round(quality_faithfulness_sum / max(1, quality_seen), 4) if quality_eval_enabled else None,
            "avg_answer_relevancy": round(quality_relevancy_sum / max(1, quality_seen), 4) if quality_eval_enabled else None,
            "avg_roundtrip_consistency": round(quality_roundtrip_sum / max(1, quality_seen), 4) if quality_eval_enabled else None,
            "quality_filter_enabled": quality_filter_enabled if quality_eval_enabled else None,
            "thresholds": {
                "min_faithfulness": min_faithfulness if quality_eval_enabled else None,
                "min_relevancy": min_relevancy if quality_eval_enabled else None,
                "min_roundtrip": min_roundtrip if quality_eval_enabled else None,
            },
        },
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    datasets_dir = default_datasets_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Genera pares sintéticos QA en español desde los chunks de obstetricia "
            "usando OpenAI Structured Outputs. Salida en formato SFT (messages)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=datasets_dir / "lm" / "train_lm.jsonl",
        help="JSONL de chunks limpios (train_lm.jsonl o validation_lm.jsonl).",
    )
    parser.add_argument(
        "--sft-output",
        type=Path,
        default=datasets_dir / "qa" / "synthetic_qa_sft.jsonl",
        help="Salida en formato messages, lista para SFT/QLoRA.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=datasets_dir / "qa" / "synthetic_qa_raw.jsonl",
        help="Salida de auditoría con los pares crudos y metadatos.",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=datasets_dir / "qa" / ".qa_generation_progress.json",
        help="Archivo de checkpoint para reanudar si el proceso se interrumpe.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=datasets_dir / "qa" / "qa_generation_report.json",
        help="Reporte de métricas de generación y grounding.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4-mini",
        choices=SUPPORTED_MODELS,
        help=(
            "Modelo de OpenAI a utilizar. "
            "gpt-5.4-mini es el más económico; "
            "gpt-5.4 para mejor calidad; "
            "gpt-5.5 para máxima calidad (más caro). "
            "Nota: gpt-4o y gpt-4o-mini fueron deprecados en feb 2026."
        ),
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=2,
        help="Número mínimo de pares QA por chunk.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=5,
        help="Número máximo de pares QA por chunk.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=15,
        help="Peticiones simultáneas a la API de OpenAI.",
    )
    parser.add_argument(
        "--min-clinical-score",
        type=int,
        default=0,
        help="Ignorar chunks con clinical_score menor a este valor.",
    )
    parser.add_argument(
        "--allowed-languages",
        type=str,
        default="es",
        help="Idiomas permitidos para generar QA (coma-separados).",
    )
    # Phase 7: content-role filtering for clinically useful QA generation
    CLINICAL_ROLES = ("evidence", "recommendation", "procedure", "diagnostic", "treatment")
    parser.add_argument(
        "--allowed-content-roles",
        type=str,
        default=",".join(CLINICAL_ROLES),
        help="Roles de contenido permitidos (coma-separados). Fase 7: solo roles clínicamente útiles.",
    )
    parser.add_argument(
        "--no-content-role-filter",
        action="store_true",
        help="Deshabilitar el filtro por content_role (backward-compatible).",
    )
    parser.add_argument(
        "--min-topic-coverage",
        type=float,
        default=0.30,
        help="Fase 7: reporta cobertura mínima por topic sin reinyectar roles no accionables (0.0-1.0).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key de OpenAI. Por defecto usa la variable OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra estadísticas y estimación de costo sin llamar a la API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Procesa solo los primeros N chunks tras los filtros. Útil para pruebas pequeñas.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad.",
    )
    parser.add_argument(
        "--enable-quality-eval",
        action="store_true",
        help="Activa evaluación de calidad por par (faithfulness, relevancy, roundtrip).",
    )
    parser.add_argument(
        "--quality-verifier-model",
        type=str,
        default="gpt-5.4",
        choices=SUPPORTED_MODELS,
        help="Modelo verificador para evaluación de calidad.",
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Si se activa calidad, filtra y conserva solo pares QA aceptados.",
    )
    parser.add_argument(
        "--min-faithfulness",
        type=float,
        default=0.80,
        help="Umbral mínimo de faithfulness para aceptar un par QA.",
    )
    parser.add_argument(
        "--min-relevancy",
        type=float,
        default=0.75,
        help="Umbral mínimo de relevancy para aceptar un par QA.",
    )
    parser.add_argument(
        "--min-roundtrip",
        type=float,
        default=0.75,
        help="Umbral mínimo de consistencia roundtrip para aceptar un par QA.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase 7: content-role filtering with topic-coverage preservation
# ---------------------------------------------------------------------------


CLINICAL_CONTENT_ROLES = {
    "evidence",
    "recommendation",
    "procedure",
    "diagnostic",
    "treatment",
}


def compute_topic_distribution(chunks: List[Dict[str, Any]]) -> Counter[str]:
    """Compute topic frequency across chunks."""
    dist: Counter[str] = Counter()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        topics = meta.get("topics", []) or meta.get("topic_tags", [])
        for topic in topics:
            if topic:
                dist[topic] += 1
    return dist


def chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    metadata = chunk.get("metadata")
    return metadata if isinstance(metadata, dict) else chunk


def chunk_content_role(chunk: Dict[str, Any]) -> str:
    return str(chunk_metadata(chunk).get("content_role", "")).strip()


def looks_spanish(text: str) -> bool:
    """Lightweight Spanish detector used when metadata.language is missing."""
    lowered = f" {str(text).lower()} "
    score = 0
    spanish_signals = [
        " el ", " la ", " los ", " las ", " de ", " del ", " que ", " por ",
        " para ", " con ", " una ", " uno ", " al ", " se ", " en ",
    ]
    for token in spanish_signals:
        if token in lowered:
            score += 1
    if any(ch in lowered for ch in "áéíóúñü"):
        score += 2
    return score >= 4


def language_matches(chunk: Dict[str, Any], allowed_languages: Set[str]) -> bool:
    """Return whether a chunk passes the language filter, with metadata fallback."""
    meta = chunk_metadata(chunk)
    lang = str(meta.get("language", "")).strip().lower()
    if lang:
        return lang in allowed_languages

    # When language metadata is absent, accept Spanish-looking chunks if Spanish is allowed.
    if "es" in allowed_languages:
        return looks_spanish(str(chunk.get("text", "")))

    # Avoid aggressive filtering when the language is unknown and Spanish is not the target.
    return True


def filter_by_content_role(
    chunks: List[Dict[str, Any]],
    allowed_roles: Set[str],
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter chunks to only actionable content roles and report skipped stats."""
    before = len(chunks)
    filtered: List[Dict[str, Any]] = []
    kept_by_role: Counter[str] = Counter()
    skipped_by_role: Counter[str] = Counter()
    for chunk in chunks:
        role = chunk_content_role(chunk) or "unknown"
        if role in allowed_roles:
            filtered.append(chunk)
            kept_by_role[role] += 1
        else:
            skipped_by_role[role] += 1
    logger.info(
        "Filtro content_role %s: %d → %d chunks",
        allowed_roles,
        before,
        len(filtered),
    )
    if skipped_by_role:
        logger.info(
            "Chunks excluidos por content_role: %s",
            dict(sorted(skipped_by_role.items())),
        )
    return filtered, {
        "input_chunks": before,
        "eligible_chunks": len(filtered),
        "skipped_chunks": sum(skipped_by_role.values()),
        "eligible_by_content_role": dict(sorted(kept_by_role.items())),
        "skipped_by_content_role": dict(sorted(skipped_by_role.items())),
        "allowed_content_roles": sorted(allowed_roles),
    }


def preserve_topic_coverage(
    original: List[Dict[str, Any]],
    filtered: List[Dict[str, Any]],
    min_coverage: float,
    allowed_roles: Set[str],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Report topic coverage without relaxing the actionable-role filter."""
    if min_coverage <= 0.0:
        return filtered

    original_dist = compute_topic_distribution(original)
    if not original_dist:
        return filtered

    filtered_dist = compute_topic_distribution(filtered)
    below_threshold: Dict[str, Dict[str, float]] = {}

    for topic, orig_count in original_dist.items():
        if orig_count == 0:
            continue
        filt_count = filtered_dist.get(topic, 0)
        coverage = filt_count / orig_count
        if coverage >= min_coverage:
            continue

        below_threshold[topic] = {
            "original": float(orig_count),
            "filtered": float(filt_count),
            "coverage": round(coverage, 4),
            "allowed_roles": sorted(allowed_roles),
        }

    if below_threshold:
        logger.warning(
            "Cobertura por topic por debajo de %.0f%% tras el filtro accional: %s",
            min_coverage * 100,
            below_threshold,
        )

    return filtered


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # ── Validar API key ──────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        logger.error(
            "API key no encontrada. Configura la variable OPENAI_API_KEY "
            "o pasa --api-key <key>."
        )
        sys.exit(1)

    # ── Cargar chunks ────────────────────────────────────────────────────────
    if not args.input.exists():
        logger.error("Archivo de entrada no encontrado: %s", args.input)
        sys.exit(1)

    chunks = read_jsonl(args.input)
    logger.info("Chunks cargados: %d desde %s", len(chunks), args.input)

    allowed_languages = {x.strip().lower() for x in args.allowed_languages.split(",") if x.strip()}
    if allowed_languages:
        before = len(chunks)
        chunks = [
            c for c in chunks
            if language_matches(c, allowed_languages)
        ]
        logger.info(
            "Filtro language in %s: %d → %d chunks",
            sorted(allowed_languages),
            before,
            len(chunks),
        )

    # ── Filtrar por clinical_score si se pide ────────────────────────────────
    if args.min_clinical_score > 0:
        before = len(chunks)
        chunks = [
            c
            for c in chunks
            if c.get("metadata", {}).get("clinical_score", 0) >= args.min_clinical_score
        ]
        logger.info(
            "Filtro clinical_score >= %d: %d → %d chunks",
            args.min_clinical_score,
            before,
            len(chunks),
        )

    if not chunks:
        logger.warning("Sin chunks para procesar. Verifica --min-clinical-score.")
        sys.exit(0)

    # ── Phase 7: Filtrar por content_role ─────────────────────────────────────
    allowed_roles = set(r.strip() for r in args.allowed_content_roles.split(",") if r.strip())
    filter_stats = {
        "input_chunks": len(chunks),
        "eligible_chunks": len(chunks),
        "skipped_chunks": 0,
        "eligible_by_content_role": {},
        "skipped_by_content_role": {},
        "allowed_content_roles": sorted(allowed_roles),
    }
    if not args.no_content_role_filter:
        original_chunks = chunks[:]
        chunks, filter_stats = filter_by_content_role(chunks, allowed_roles, logger)
        chunks = preserve_topic_coverage(
            original_chunks, chunks, args.min_topic_coverage, allowed_roles, logger
        )
    else:
        logger.info("Filtro por content_role deshabilitado (--no-content-role-filter).")

    # Log topic coverage after filtering so quality regressions are easy to spot.
    topic_dist = compute_topic_distribution(chunks)
    if topic_dist:
        logger.info("Topics tras filtro: %s", dict(sorted(topic_dist.items())))

    if not chunks:
        logger.warning("Sin chunks para procesar tras filtros de content_role.")
        sys.exit(0)

    if args.limit is not None:
        if args.limit <= 0:
            logger.error("--limit debe ser mayor que 0.")
            sys.exit(1)
        before = len(chunks)
        chunks = chunks[: args.limit]
        logger.info("Limitando chunks: %d → %d", before, len(chunks))

    # ── Cost estimate and run summary ────────────────────────────────────────
    expected_pairs, cost_est = estimate_cost(
        chunks, args.model, args.min_pairs, args.max_pairs
    )

    print()
    print("=" * 60)
    print("  RESUMEN DE GENERACIÓN SINTÉTICA")
    print("=" * 60)
    print(f"  Modelo             : {args.model}")
    print(f"  Chunks de entrada  : {filter_stats['input_chunks']}")
    print(f"  Chunks a procesar  : {len(chunks)}")
    if not args.no_content_role_filter:
        print(f"  Content roles      : {', '.join(sorted(allowed_roles))}")
        print(f"  Topic coverage min : {args.min_topic_coverage:.0%}")
        print(f"  Chunks excluidos   : {filter_stats['skipped_chunks']}")
        if filter_stats["skipped_by_content_role"]:
            print("  Excluidos por rol  :")
            for role, count in filter_stats["skipped_by_content_role"].items():
                print(f"    - {role}: {count}")
    else:
        print("  Content roles      : (sin filtro)")
    print(f"  Pares esperados    : ~{expected_pairs}")
    print(f"  Costo estimado     : ~${cost_est:.2f} USD")
    print(f"  Concurrencia       : {args.concurrency} peticiones simultáneas")
    print(f"  Quality eval       : {'ON' if args.enable_quality_eval else 'OFF'}")
    if args.enable_quality_eval:
        print(f"  Verifier model     : {args.quality_verifier_model}")
        print(
            "  Thresholds QA      : "
            f"faithfulness>={args.min_faithfulness:.2f}, "
            f"relevancy>={args.min_relevancy:.2f}, "
            f"roundtrip>={args.min_roundtrip:.2f}"
        )
        print(f"  Quality filter     : {'ON' if args.quality_filter else 'OFF'}")
    print(f"  Salida SFT         : {args.sft_output}")
    print(f"  Salida raw         : {args.raw_output}")
    print(f"  Reporte QA         : {args.report_output}")
    print(f"  Checkpoint         : {args.progress_file}")
    print("=" * 60)
    print()

    if args.dry_run:
        logger.info("Modo dry-run: no se llamó a la API.")
        return

    # ── Execute generation ───────────────────────────────────────────────────
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("openai no está instalado. Ejecuta: pip install 'openai>=1.50.0'")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)

    start = time.monotonic()
    stats = asyncio.run(
        run_generation(
            chunks=chunks,
            client=client,
            model=args.model,
            min_pairs=args.min_pairs,
            max_pairs=args.max_pairs,
            concurrency=args.concurrency,
            sft_output=args.sft_output,
            raw_output=args.raw_output,
            progress_file=args.progress_file,
            report_output=args.report_output,
            quality_eval_enabled=args.enable_quality_eval,
            verifier_model=args.quality_verifier_model,
            min_faithfulness=args.min_faithfulness,
            min_relevancy=args.min_relevancy,
            min_roundtrip=args.min_roundtrip,
            quality_filter_enabled=args.quality_filter,
            logger=logger,
        )
    )
    elapsed = time.monotonic() - start

    print()
    print("=" * 60)
    print("  RESULTADOS")
    print("=" * 60)
    print(f"  Chunks totales     : {stats['total']}")
    print(f"  Ya procesados      : {stats['skipped']}")
    print(f"  Procesados ahora   : {stats['processed']}")
    print(f"  Fallidos           : {stats['failed']}")
    print(f"  Pares QA generados : {stats['qa_pairs']}")
    print(
        "  Grounding (overlap): "
        f"{stats['grounding']['avg_context_answer_overlap']:.3f} "
        f"(bajo={stats['grounding']['low_grounding_rate']:.1%})"
    )
    if stats.get("quality", {}).get("enabled"):
        print(
            "  Quality (avg)      : "
            f"faith={stats['quality']['avg_faithfulness']:.3f} | "
            f"rel={stats['quality']['avg_answer_relevancy']:.3f} | "
            f"rt={stats['quality']['avg_roundtrip_consistency']:.3f}"
        )
        print(
            "  Quality acceptance : "
            f"{stats['quality']['pairs_accepted']}/{stats['quality']['pairs_evaluated']} "
            f"({stats['quality']['acceptance_rate']:.1%})"
        )
    print(f"  Tiempo             : {elapsed:.0f}s")
    print(f"  SFT output         : {args.sft_output}")
    print(f"  Raw output         : {args.raw_output}")
    print(f"  QA report          : {args.report_output}")
    print("=" * 60)

    if stats["failed"] > 0:
        logger.warning(
            "%d chunks fallaron. Puedes relanzar el script para reintentarlos "
            "(el checkpoint excluirá los ya completados).",
            stats["failed"],
        )


if __name__ == "__main__":
    main()
