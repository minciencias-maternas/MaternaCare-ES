#!/usr/bin/env python3
"""Fine-tune the obstetrics QA dataset with Hugging Face TRL and QLoRA.

Typical smoke test:
    python scripts/train_qlora_trl.py \
        --model-name google/gemma-4-E2B-it \
        --dataset-variant sft_grounded \
        --output-dir outputs/smoke-gemma4-e2b-grounded \
        --max-steps 10 \
        --train-limit 64 \
        --eval-limit 32

The trainer converts the published conversational JSONL files to TRL's
prompt/completion format in memory. This keeps loss masking explicit and avoids
depending on model-specific chat templates that may not expose assistant masks.

The script intentionally avoids installing dependencies or logging in to
Hugging Face. Install `requirements.txt`, run `huggingface-cli login`, and
accept gated model terms before launching training on the workstation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Literal


DATASET_ROOT = Path("datasets/obstetrics/qa/publication")
DEFAULT_DATASET_HF_ID = "iue-edu/MaternaQA-es"
DATASET_VARIANTS = ("sft_closed_book", "sft_grounded")
SPLITS = ("train", "validation", "test")


def remove_project_root_from_imports() -> None:
    """Avoid shadowing installed packages with top-level artifact folders."""
    # The repository has a top-level `datasets/` directory for data artifacts.
    # Remove the project root from import lookup so it cannot shadow the
    # Hugging Face `datasets` package when this script is run from repo root.
    project_root = Path(__file__).resolve().parents[1]
    sys.path = [
        entry
        for entry in sys.path
        if entry
        and Path(entry).resolve() != project_root
    ]


def import_dataset_loader() -> Any:
    """Import the minimal dependency needed for dataset validation."""
    remove_project_root_from_imports()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Falta la dependencia `datasets`. Instala primero `requirements.txt`. "
            f"Error original: {exc}"
        ) from exc
    return load_dataset


def import_training_stack() -> dict[str, Any]:
    """Import optional fine-tuning dependencies with an actionable error."""
    remove_project_root_from_imports()
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Faltan dependencias de fine-tuning. Instala primero `requirements.txt` "
            "en la workstation de entrenamiento. Error original: "
            f"{exc}"
        ) from exc

    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForImageTextToText": AutoModelForImageTextToText,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un adapter QLoRA con TRL sobre las variantes QA publicadas."
    )
    parser.add_argument(
        "--model-name",
        default="google/gemma-4-E2B-it",
        help="ID de Hugging Face o ruta local del modelo base.",
    )
    parser.add_argument(
        "--model-class",
        choices=("auto", "causal-lm", "image-text-to-text"),
        default="auto",
        help=(
            "Clase de carga del modelo. `auto` usa ImageTextToText para Gemma 4/MedGemma "
            "y CausalLM para otros modelos."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Carpeta raíz con las variantes de publicación (modo local).",
    )
    parser.add_argument(
        "--dataset-hf-id",
        default=DEFAULT_DATASET_HF_ID,
        help=(
            "ID de Hugging Face del dataset (ej: iue-edu/MaternaQA-es). "
            "Si se proporciona, carga desde HF usando la variante como subset. "
            "Usa un valor vacío para cargar desde --dataset-root en modo local."
        ),
    )
    parser.add_argument(
        "--dataset-variant",
        choices=DATASET_VARIANTS,
        default="sft_grounded",
        help="Variante SFT a usar: closed-book o grounded.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directorio donde se guardará el adapter LoRA y tokenizer.",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-strategy", choices=("no", "steps", "epoch"), default="epoch")
    parser.add_argument("--eval-strategy", choices=("no", "steps", "epoch"), default="epoch")
    parser.add_argument("--train-limit", type=int, default=None, help="Limita train para smoke tests.")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limita validation para smoke tests.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--assistant-only-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Modo avanzado: usa máscaras del chat template para calcular pérdida solo "
            "sobre assistant. Requiere templates con {%% generation %%}. Por defecto se "
            "usa prompt/completion y completion_only_loss=True."
        ),
    )
    parser.add_argument(
        "--packing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Empaqueta secuencias cortas para mejorar uso de GPU. Por defecto está "
            "desactivado para evitar contaminación entre muestras si la atención del "
            "modelo no soporta packing seguro."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reduce VRAM a cambio de algo más de cómputo.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Actívalo solo si el modelo lo requiere.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Opcional: eager, sdpa, flash_attention_2, etc.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Carga y valida el dataset, pero no carga modelo ni entrena.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permite dry-runs sin CUDA. QLoRA real requiere GPU NVIDIA/CUDA.",
    )
    return parser.parse_args()


def split_prompt_completion(example: dict[str, Any]) -> dict[str, Any]:
    """Convert a messages row to TRL conversational prompt/completion format."""
    messages = example.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Cada ejemplo debe contener una lista `messages`.")

    assistant_indices = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if not assistant_indices:
        raise ValueError("Cada ejemplo debe contener al menos un mensaje assistant.")

    completion_start = assistant_indices[-1]
    prompt = messages[:completion_start]
    completion = messages[completion_start:]
    if not prompt:
        raise ValueError("Cada ejemplo debe contener mensajes previos al assistant.")

    return {"prompt": prompt, "completion": completion}


def resolve_model_class(model_name: str, requested: str) -> Literal["causal-lm", "image-text-to-text"]:
    """Choose a model class that works for text-only and multimodal Gemma checkpoints."""
    if requested != "auto":
        return requested  # type: ignore[return-value]

    normalized = model_name.lower()
    if "gemma-4" in normalized or "medgemma" in normalized:
        return "image-text-to-text"
    return "causal-lm"


def resolve_dtype(torch: Any) -> Any:
    """Use bf16 on Ampere+ GPUs and fp16 on older CUDA devices."""
    if not torch.cuda.is_available():
        return torch.float16
    major, _minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def dataset_files(dataset_root: Path, variant: str) -> dict[str, str]:
    """Return local JSONL files in Hugging Face Datasets format."""
    variant_dir = dataset_root / variant
    files = {split: str(variant_dir / f"{split}.jsonl") for split in SPLITS}
    missing = [path for path in files.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "No se encontraron archivos de dataset: " + ", ".join(missing)
        )
    return files


def keep_prompt_completion_only(dataset: Any) -> Any:
    """Drop metadata before TRL preprocessing to keep the training schema minimal."""
    columns_to_remove = [
        column for column in dataset.column_names if column not in {"prompt", "completion"}
    ]
    if not columns_to_remove:
        return dataset
    return dataset.remove_columns(columns_to_remove)


def to_prompt_completion(dataset: Any) -> Any:
    """Use explicit completion boundaries instead of tokenizer assistant masks."""
    return keep_prompt_completion_only(
        dataset.map(split_prompt_completion, desc="Construyendo prompt/completion")
    )


def maybe_limit(dataset: Any, limit: int | None) -> Any:
    """Select the first N rows for reproducible smoke tests."""
    if limit is None or limit >= len(dataset):
        return dataset
    if limit <= 0:
        raise ValueError("Los límites de dataset deben ser positivos.")
    return dataset.select(range(limit))


def load_sft_datasets(args: argparse.Namespace, load_dataset: Any) -> tuple[Any, Any, Any]:
    """Load train, validation, and held-out test splits from HF or local JSONL files."""
    if args.dataset_hf_id:
        dataset = load_dataset(args.dataset_hf_id, args.dataset_variant)
    else:
        dataset = load_dataset(
            "json",
            data_files=dataset_files(args.dataset_root, args.dataset_variant),
        )
    train_dataset = to_prompt_completion(maybe_limit(dataset["train"], args.train_limit))
    eval_dataset = to_prompt_completion(maybe_limit(dataset["validation"], args.eval_limit))
    test_dataset = to_prompt_completion(dataset["test"])
    return train_dataset, eval_dataset, test_dataset


def build_quantization_config(args: argparse.Namespace, stack: dict[str, Any]) -> Any:
    """Create the bitsandbytes configuration used by QLoRA."""
    torch = stack["torch"]
    BitsAndBytesConfig = stack["BitsAndBytesConfig"]
    compute_dtype = resolve_dtype(torch)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model_and_tokenizer(args: argparse.Namespace, stack: dict[str, Any]) -> tuple[Any, Any]:
    """Load a 4-bit base model and tokenizer for QLoRA training."""
    torch = stack["torch"]
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "No se detectó CUDA. QLoRA con bitsandbytes requiere una GPU NVIDIA. "
            "Usa --dry-run --allow-cpu solo para validar dataset sin entrenar."
        )

    tokenizer = stack["AutoTokenizer"].from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "quantization_config": build_quantization_config(args, stack),
        "device_map": "auto",
        "trust_remote_code": args.trust_remote_code,
        "token": os.environ.get("HF_TOKEN"),
    }
    dtype = resolve_dtype(torch)
    model_kwargs["dtype"] = dtype
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model_class = resolve_model_class(args.model_name, args.model_class)
    if model_class == "image-text-to-text":
        model_loader = stack["AutoModelForImageTextToText"]
    else:
        model_loader = stack["AutoModelForCausalLM"]

    model = model_loader.from_pretrained(args.model_name, **model_kwargs)
    return model, tokenizer


def build_peft_config(args: argparse.Namespace, stack: dict[str, Any]) -> Any:
    """Create LoRA adapters on all linear layers, the standard QLoRA target."""
    LoraConfig = stack["LoraConfig"]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )


def build_training_args(args: argparse.Namespace, stack: dict[str, Any]) -> Any:
    """Create SFTConfig with conservative defaults for a 16GB VRAM workstation."""
    if args.assistant_only_loss:
        raise SystemExit(
            "`--assistant-only-loss` requiere un chat template con {% generation %}. "
            "Este script usa por defecto prompt/completion con `completion_only_loss=True`, "
            "que es más estable para Gemma/MedGemma. Ejecuta sin `--assistant-only-loss`."
        )

    torch = stack["torch"]
    SFTConfig = stack["SFTConfig"]
    dtype = resolve_dtype(torch)

    config_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "max_length": args.max_length,
        "packing": args.packing,
        "completion_only_loss": True,
        "assistant_only_loss": False,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "eval_strategy": args.eval_strategy,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "adamw_8bit",
        "lr_scheduler_type": "cosine",
        "seed": args.seed,
        "bf16": dtype is torch.bfloat16,
        "fp16": dtype is torch.float16,
        "report_to": ["tensorboard"],
        "dataset_num_proc": 1,
        "push_to_hub": False,
    }
    if args.max_steps is not None:
        config_kwargs["max_steps"] = args.max_steps
    if args.save_steps is not None:
        config_kwargs["save_steps"] = args.save_steps
    if args.eval_steps is not None:
        config_kwargs["eval_steps"] = args.eval_steps

    return SFTConfig(**config_kwargs)


def main() -> None:
    args = parse_args()
    stack: dict[str, Any] | None = None
    if args.dry_run:
        load_dataset = import_dataset_loader()
    else:
        stack = import_training_stack()
        load_dataset = stack["load_dataset"]
    train_dataset, eval_dataset, test_dataset = load_sft_datasets(
        args,
        load_dataset,
    )

    print(
        "Dataset cargado:",
        {
            "variant": args.dataset_variant,
            "train": len(train_dataset),
            "validation": len(eval_dataset),
            "test_holdout": len(test_dataset),
        },
    )
    print("Ejemplo prompt:", train_dataset[0]["prompt"])
    print("Ejemplo completion:", train_dataset[0]["completion"])

    if args.dry_run:
        print("Dry-run completado: no se cargó modelo ni se entrenó.")
        return

    if stack is None:
        stack = import_training_stack()
    model, tokenizer = load_model_and_tokenizer(args, stack)
    training_args = build_training_args(args, stack)
    peft_config = build_peft_config(args, stack)

    trainer = stack["SFTTrainer"](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Adapter QLoRA guardado en: {args.output_dir}")


if __name__ == "__main__":
    main()
