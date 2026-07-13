#!/usr/bin/env python3
"""Prepare clean QA dataset variants for SFT training and Hub publication."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT_DIR = Path("datasets/obstetrics/qa/final")
DEFAULT_OUTPUT_DIR = Path("datasets/obstetrics/qa/publication")
SPLITS = ("train", "validation", "test")
RAW_VARIANT_DIR = "qa_flat_jsonl"
REMOVED_QUALITY_FIELDS = {
    "faithfulness",
    "answer_relevancy",
    "roundtrip_consistency",
    "quality_verdict",
    "quality_reason",
}


GROUNDED_USER_TEMPLATE = """Contexto fuente:
{context}

Pregunta:
{question}"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def remove_quality_fields(value: Any) -> Any:
    """Return a deep-cleaned object without empty quality metric placeholders."""
    if isinstance(value, dict):
        return {
            key: remove_quality_fields(item)
            for key, item in value.items()
            if key not in REMOVED_QUALITY_FIELDS
        }
    if isinstance(value, list):
        return [remove_quality_fields(item) for item in value]
    return value


def as_grounded_sft(row: dict[str, Any]) -> dict[str, Any]:
    grounded = remove_quality_fields(deepcopy(row))
    messages = grounded["messages"]
    metadata = grounded.get("metadata", {})
    question = messages[1]["content"]
    context = metadata.get("contexto_fuente", "").strip()
    messages[1]["content"] = GROUNDED_USER_TEMPLATE.format(
        context=context,
        question=question,
    )
    return grounded


def summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pairs": len(rows),
        "unique_qa_ids": len({row.get("qa_id") for row in rows if row.get("qa_id")}),
        "unique_chunks": len(
            {row.get("chunk_id") for row in rows if row.get("chunk_id")}
        ),
        "unique_source_pdfs": len(
            {row.get("source_pdf") for row in rows if row.get("source_pdf")}
        ),
    }


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Variantes de publicación del QA de obstetricia",
        "",
        "Archivos JSONL limpios generados a partir de `datasets/obstetrics/qa/final/` para fine-tuning supervisado y publicación en Hugging Face Hub.",
        "",
        "## Variantes",
        "",
        "- `sft_closed_book/{train,validation,test}.jsonl`: formato conversacional SFT. El mensaje del usuario contiene únicamente la pregunta.",
        "- `sft_grounded/{train,validation,test}.jsonl`: formato conversacional SFT. El mensaje del usuario contiene `Contexto fuente` más la pregunta.",
        "- `qa_flat_jsonl/{train,validation,test}.jsonl`: registros planos de auditoría/evaluación con pregunta, respuesta, contexto fuente, procedencia y metadata.",
        "- `qa_flat_jsonl/all.jsonl`: consolidado completo de los tres splits para exploración o publicación simple.",
        "",
        "En todas las variantes se eliminan los campos vacíos de métricas de calidad: `faithfulness`, `answer_relevancy`, `roundtrip_consistency`, `quality_verdict` y `quality_reason`.",
        "",
        "## Uso de los splits",
        "",
        "- `train`: entrenamiento del modelo.",
        "- `validation`: evaluación durante entrenamiento, selección de hiperparámetros y early stopping.",
        "- `test`: evaluación final retenida; no debe usarse para decidir la configuración del entrenamiento.",
        "",
        "## Conteos",
        "",
        "| Split | Pares | Chunks únicos | PDFs fuente únicos |",
        "|---|---:|---:|---:|",
    ]
    for split in SPLITS:
        split_summary = summary["splits"][split]
        lines.append(
            f"| {split} | {split_summary['pairs']} | {split_summary['unique_chunks']} | {split_summary['unique_source_pdfs']} |"
        )
    lines.extend(
        [
            "",
            "## Recomendación de entrenamiento",
            "",
            "Usa `sft_closed_book` para medir adaptación al dominio sin contexto explícito.",
            "Usa `sft_grounded` para entrenar y evaluar comportamiento guiado por evidencia, donde el modelo recibe el contexto fuente en tiempo de inferencia.",
            "",
            "## Cómo se usan las variantes durante el fine-tuning",
            "",
            "`sft_closed_book` y `sft_grounded` son variantes del dataset, no carpetas ligadas a una librería específica.",
            "Cada variante puede entrenarse tanto con la ruta canónica de Hugging Face como con la ruta optimizada de Unsloth.",
            "",
            "Matriz recomendada de experimentos:",
            "",
            "| Familia de modelo | Variante de dataset | Ruta de entrenamiento | Propósito |",
            "|---|---|---|---|",
            "| Gemma 4 instruct | `sft_closed_book` | TRL + PEFT + bitsandbytes (QLoRA) | Medir adaptación al dominio sin contexto. |",
            "| Gemma 4 instruct | `sft_grounded` | TRL + PEFT + bitsandbytes (QLoRA) | Medir QA guiada por evidencia. |",
            "| MedGemma 1.5 4B IT | `sft_closed_book` | TRL + PEFT + bitsandbytes (QLoRA) | Medir adaptación sobre un modelo médico instruct sin contexto. |",
            "| MedGemma 1.5 4B IT | `sft_grounded` | TRL + PEFT + bitsandbytes (QLoRA) | Medir QA médica guiada por evidencia. |",
            "| Gemma 4 instruct | `sft_closed_book` / `sft_grounded` | Unsloth + TRL (QLoRA) | Repetir los mismos experimentos en versión optimizada si Unsloth es estable en la workstation. |",
            "| MedGemma 1.5 4B IT | `sft_closed_book` / `sft_grounded` | Unsloth + TRL (QLoRA) | Ejecutar solo después de validar con smoke test que el checkpoint de Unsloth carga, entrena y guarda bien. |",
            "",
            "Qué define la variante del dataset:",
            "",
            "- `sft_closed_book`: el modelo recibe solo la pregunta clínica y debe responder con sus parámetros adaptados.",
            "- `sft_grounded`: el modelo recibe `Contexto fuente` más la pregunta y debe responder usando la evidencia provista.",
            "",
            "Qué define la ruta de entrenamiento:",
            "",
            "- `TRL + PEFT + bitsandbytes`: ruta canónica y reproducible de Hugging Face para QLoRA.",
            "- `Unsloth + TRL`: ruta optimizada donde Unsloth carga/parchea el modelo y prepara LoRA de forma eficiente, mientras TRL sigue ejecutando el ciclo de SFT.",
            "",
            "En ambas rutas, `train.jsonl` se pasa como `train_dataset`, `validation.jsonl` como `eval_dataset` y `test.jsonl` se mantiene reservado para evaluación final.",
            "",
            "## Notas de compatibilidad de modelos",
            "",
            "- Gemma 4: Unsloth documenta soporte explícito para fine-tuning de Gemma 4 E2B, E4B, 26B A4B y 31B. En una workstation con 16GB VRAM, el candidato preferido es Gemma 4 E4B QLoRA si entra; si no, E2B.",
            "- MedGemma 1.5 4B IT: la ruta segura y canónica es QLoRA con TRL/PEFT/bitsandbytes. Unsloth tiene artefactos de MedGemma 1.5 en Hugging Face, pero el proyecto debe validar con smoke test local antes de depender de Unsloth para ese modelo.",
            "- Full fine-tuning no es la ruta objetivo para esta workstation; QLoRA es el método por defecto.",
            "",
            "## Script de entrenamiento TRL listo",
            "",
            "El repositorio incluye `scripts/train_qlora_trl.py` para ejecutar la ruta canónica `TRL + PEFT + bitsandbytes (QLoRA)`.",
            "El script no instala dependencias ni hace login en Hugging Face; antes de entrenar, instala `requirements.txt`, ejecuta `huggingface-cli login` y acepta los términos de acceso de los modelos de Google/MedGemma.",
            "",
            "Ejemplo de smoke test con pocos pasos:",
            "",
            "```bash",
            "python scripts/train_qlora_trl.py \\",
            "  --model-name google/gemma-4-E2B-it \\",
            "  --dataset-variant sft_grounded \\",
            "  --output-dir outputs/smoke-gemma4-e2b-grounded \\",
            "  --max-steps 10 \\",
            "  --train-limit 64 \\",
            "  --eval-limit 32",
            "```",
            "",
            "Ejemplo de entrenamiento closed-book cambiando solo la variante:",
            "",
            "```bash",
            "python scripts/train_qlora_trl.py \\",
            "  --model-name google/gemma-4-E2B-it \\",
            "  --dataset-variant sft_closed_book \\",
            "  --output-dir outputs/gemma4-e2b-closed-book-qlora",
            "```",
            "",
            "## Carga mínima",
            "",
            "```python",
            "from datasets import load_dataset",
            "",
            "dataset = load_dataset(",
            "    \"json\",",
            "    data_files={",
            "        \"train\": \"datasets/obstetrics/qa/publication/sft_grounded/train.jsonl\",",
            "        \"validation\": \"datasets/obstetrics/qa/publication/sft_grounded/validation.jsonl\",",
            "        \"test\": \"datasets/obstetrics/qa/publication/sft_grounded/test.jsonl\",",
            "    },",
            ")",
            "",
            "train_dataset = dataset[\"train\"]",
            "eval_dataset = dataset[\"validation\"]",
            "# dataset[\"test\"] queda reservado para evaluación final.",
            "```",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

def build_variants(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "source_dir": str(input_dir),
        "output_dir": str(output_dir),
        "removed_fields": sorted(REMOVED_QUALITY_FIELDS),
        "splits": {},
        "files": {},
    }

    for split in SPLITS:
        raw_rows = load_jsonl(input_dir / split / "raw.jsonl")
        sft_rows = load_jsonl(input_dir / split / "sft.jsonl")

        clean_raw = [remove_quality_fields(row) for row in raw_rows]
        closed_book = [remove_quality_fields(row) for row in sft_rows]
        grounded = [as_grounded_sft(row) for row in sft_rows]

        files = {
            "qa_flat_jsonl": output_dir / RAW_VARIANT_DIR / f"{split}.jsonl",
            "sft_closed_book": output_dir / "sft_closed_book" / f"{split}.jsonl",
            "sft_grounded": output_dir / "sft_grounded" / f"{split}.jsonl",
        }
        counts = {
            "qa_flat_jsonl": write_jsonl(files["qa_flat_jsonl"], clean_raw),
            "sft_closed_book": write_jsonl(files["sft_closed_book"], closed_book),
            "sft_grounded": write_jsonl(files["sft_grounded"], grounded),
        }

        if len(set(counts.values())) != 1:
            raise RuntimeError(f"Variant row count mismatch for split {split}: {counts}")

        summary["splits"][split] = summarize_split(raw_rows)
        summary["files"][split] = {name: str(path) for name, path in files.items()}

    consolidated_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        consolidated_rows.extend(load_jsonl(output_dir / RAW_VARIANT_DIR / f"{split}.jsonl"))

    output_dir.mkdir(parents=True, exist_ok=True)
    consolidated_path = output_dir / RAW_VARIANT_DIR / "all.jsonl"
    consolidated_count = write_jsonl(consolidated_path, consolidated_rows)
    summary["files"]["all"] = {"qa_flat_jsonl": str(consolidated_path)}
    summary["counts"] = {"qa_flat_jsonl_all": consolidated_count}

    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(output_dir, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare clean closed-book, grounded, and flat-QA publication JSONL variants."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_variants(args.input_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
