#!/usr/bin/env python3
"""Run inference with an unfine-tuned base model on the held-out QA test split.

This script mirrors scripts/inference_qlora.py, but intentionally does not load
a PEFT adapter. Use it to create baseline predictions that can be compared
against QLoRA fine-tuned predictions with scripts/evaluate_model_predictions.py.

Usage:
    python scripts/inference_base.py \
        --model-name google/gemma-4-E2B-it \
        --output-prefix outputs/gemma4-base/test

    python scripts/inference_base.py \
        --model-name google/medgemma-1.5-4b-it \
        --output-prefix outputs/medgemma-base/test \
        --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal


DATASET_ROOT = Path("datasets/obstetrics/qa/publication")


def remove_project_root_from_imports() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path = [
        entry
        for entry in sys.path
        if entry and Path(entry).resolve() != project_root
    ]


def import_stack() -> dict[str, Any]:
    remove_project_root_from_imports()
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise SystemExit(
            "Faltan dependencias. Instala requirements.txt primero. "
            f"Error: {exc}"
        ) from exc
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForImageTextToText": AutoModelForImageTextToText,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia baseline con modelo base sin adapter QLoRA."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="ID de Hugging Face o ruta local del modelo base.",
    )
    parser.add_argument(
        "--model-class",
        default="auto",
        choices=("auto", "causal-lm", "image-text-to-text"),
        help=(
            "Clase de carga del modelo. `auto` usa ImageTextToText para "
            "Gemma 4/MedGemma y CausalLM para otros modelos."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Carpeta raíz con las variantes de publicación.",
    )
    parser.add_argument(
        "--dataset-variant",
        default="sft_grounded",
        choices=("sft_closed_book", "sft_grounded"),
        help="Variante SFT a evaluar.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Prefijo para archivos de salida (se genera <prefix>_predictions.jsonl).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=False,
        help="Usa muestreo estocástico. Por defecto se usa greedy decoding reproducible.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cantidad máxima de ejemplos a evaluar (para smoke tests).",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Carga el modelo base en 4 bits para mantener inferencia comparable y reducir VRAM.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reanuda desde <output-prefix>_predictions.jsonl si existe, "
            "saltando qa_id ya procesados. Usa --no-resume para sobrescribir."
        ),
    )
    return parser.parse_args()


def resolve_model_class(
    model_name: str, requested: str
) -> Literal["causal-lm", "image-text-to-text"]:
    if requested != "auto":
        return requested  # type: ignore[return-value]

    normalized = model_name.lower()
    if "gemma-4" in normalized or "medgemma" in normalized:
        return "image-text-to-text"
    return "causal-lm"


def resolve_dtype(torch: Any) -> Any:
    return (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )


def load_model_and_tokenizer(args: argparse.Namespace, stack: dict[str, Any]) -> tuple[Any, Any]:
    torch = stack["torch"]
    if args.load_in_4bit and not torch.cuda.is_available():
        raise SystemExit("Se requiere GPU NVIDIA con CUDA para inferencia base en 4-bit.")

    compute_dtype = resolve_dtype(torch)

    tokenizer = stack["AutoTokenizer"].from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": compute_dtype,
        "token": os.environ.get("HF_TOKEN"),
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = stack["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model_class = resolve_model_class(args.model_name, args.model_class)
    if model_class == "image-text-to-text":
        model = stack["AutoModelForImageTextToText"].from_pretrained(
            args.model_name, **model_kwargs
        )
    else:
        model = stack["AutoModelForCausalLM"].from_pretrained(
            args.model_name, **model_kwargs
        )

    model.eval()
    print(f"Modelo base cargado: {args.model_name}")
    print(f"Clase de modelo: {model_class}")
    print(f"Cuantización 4-bit: {args.load_in_4bit}")
    return model, tokenizer


def build_prompt(messages: list[dict[str, str]], tokenizer: Any) -> str:
    assistant_indices = [
        i
        for i, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("No se encontro mensaje assistant en el ejemplo.")

    completion_start = assistant_indices[-1]
    prompt_messages = messages[:completion_start]

    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_reference_answer(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "").strip()
    return ""


def extract_question(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()
        marker = "Pregunta:"
        if marker in content:
            return content.split(marker, 1)[1].strip()
        return content
    return ""


def load_test_examples(
    dataset_root: Path, dataset_variant: str, limit: int | None = None
) -> list[dict[str, Any]]:
    test_file = dataset_root / dataset_variant / "test.jsonl"
    if not test_file.exists():
        raise FileNotFoundError(f"No se encontro test.jsonl en {test_file}")

    examples = []
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if limit:
        examples = examples[:limit]
    return examples


def example_id(example: dict[str, Any], idx: int) -> Any:
    return example.get("metadata", {}).get("qa_id", idx)


def load_processed_ids(output_file: Path) -> set[str]:
    if not output_file.exists():
        return set()

    processed: set[str] = set()
    with open(output_file, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{output_file} contiene JSON inválido en línea {line_number}. "
                    "Revisa o elimina la línea incompleta antes de reanudar."
                ) from exc
            row_id = row.get("id") or row.get("qa_id") or row.get("metadata", {}).get("qa_id")
            if row_id is not None:
                processed.add(str(row_id))
    return processed


def run_inference(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
    torch: Any,
    output_file: Path,
    processed_ids: set[str],
) -> list[dict[str, Any]]:
    results = []
    total = len(examples)
    start_time = time.time()
    generated_count = 0
    skipped_count = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8", newline="\n") as f:
        for idx, example in enumerate(examples):
            qa_id = example_id(example, idx)
            if str(qa_id) in processed_ids:
                skipped_count += 1
                continue

            result = generate_prediction(model, tokenizer, example, args, torch, idx)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            processed_ids.add(str(qa_id))
            results.append(result)
            generated_count += 1

            completed = skipped_count + generated_count
            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = generated_count / elapsed if elapsed > 0 else 0
                print(
                    f"  [{completed}/{total}] nuevos={generated_count} "
                    f"saltados={skipped_count} {rate:.1f} ejemplos/seg"
                )

    print(
        f"Completado en {time.time() - start_time:.1f}s "
        f"(nuevos={generated_count}, saltados={skipped_count})"
    )
    return results


def generate_prediction(
    model: Any,
    tokenizer: Any,
    example: dict[str, Any],
    args: argparse.Namespace,
    torch: Any,
    idx: int,
) -> dict[str, Any]:
    messages = example.get("messages", [])
    prompt_text = build_prompt(messages, tokenizer)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    generated = tokenizer.decode(
        generated_ids[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    metadata = example.get("metadata", {})
    return {
        "id": metadata.get("qa_id", idx),
        "model_role": "base",
        "model_name": args.model_name,
        "adapter_dir": None,
        "dataset_variant": args.dataset_variant,
        "question": extract_question(messages),
        "generated_answer": generated,
        "reference_answer": extract_reference_answer(messages),
        "source_context": metadata.get("contexto_fuente", ""),
        "prompt_text": prompt_text,
        "generated": generated,
        "reference_messages": messages,
        "metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    stack = import_stack()

    model, tokenizer = load_model_and_tokenizer(args, stack)
    examples = load_test_examples(args.dataset_root, args.dataset_variant, args.limit)
    print(f"Ejemplos cargados: {len(examples)}")

    output_file = Path(str(args.output_prefix) + "_predictions.jsonl")
    if output_file.exists() and not args.resume:
        output_file.unlink()
        print(f"Salida previa eliminada por --no-resume: {output_file}")
    processed_ids = load_processed_ids(output_file) if args.resume else set()
    if processed_ids:
        print(f"Reanudando: {len(processed_ids)} predicciones ya existentes en {output_file}")

    run_inference(
        model,
        tokenizer,
        examples,
        args,
        stack["torch"],
        output_file,
        processed_ids,
    )
    print(f"Resultados guardados en: {output_file}")


if __name__ == "__main__":
    main()
