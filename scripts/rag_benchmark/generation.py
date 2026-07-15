"""Lazy Hugging Face answer and hypothetical-document generation."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .model_registry import ModelSpec, resolve_adapter_source
from .telemetry import GenerationMeasurement


ANSWER_INSTRUCTION = (
    "Responde la pregunta clínica en español de forma precisa y directa. "
    "No inventes información que no esté respaldada por el contexto disponible."
)
HYDE_INSTRUCTION = (
    "Redacta un documento clínico breve en español que probablemente contendría la información "
    "necesaria para resolver la pregunta. Escribe solo el documento hipotético, sin explicar el proceso."
)


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float | None = None
    no_repeat_ngram_size: int = 0


@dataclass(frozen=True)
class GenerationResult:
    text: str
    measurement: GenerationMeasurement


def build_answer_messages(question: str, contexts: Sequence[str]) -> list[dict[str, str]]:
    if contexts:
        context = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(contexts, start=1))
        content = f"{ANSWER_INSTRUCTION}\n\nContexto recuperado:\n{context}\n\nPregunta: {question}"
    else:
        content = f"{ANSWER_INSTRUCTION}\n\nPregunta: {question}"
    return [{"role": "user", "content": content}]


def build_hyde_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"{HYDE_INSTRUCTION}\n\nPregunta: {question}"}]


def resolve_model_class(model_id: str) -> str:
    normalized = model_id.lower()
    return "image-text-to-text" if "gemma-4" in normalized or "medgemma" in normalized else "causal-lm"


def import_generation_stack() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install the inference dependencies from requirements.txt") from exc
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForImageTextToText": AutoModelForImageTextToText,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


class HuggingFaceGenerator:
    """One loaded generator with identical prompt and decoding semantics across model roles."""

    def __init__(self, model: Any, tokenizer: Any, torch: Any, settings: GenerationSettings) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        self.settings = settings

    @classmethod
    def from_answer_spec(
        cls,
        spec: ModelSpec,
        settings: GenerationSettings,
        adapter_path: Path | None = None,
        load_in_4bit: bool = True,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
    ) -> "HuggingFaceGenerator":
        adapter_source = resolve_adapter_source(spec, adapter_path)
        return cls._load(
            model_id=spec.base_model_id,
            tokenizer_source=adapter_source or spec.base_model_id,
            adapter_source=adapter_source,
            settings=settings,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )

    @classmethod
    def from_base_model(
        cls,
        model_id: str,
        settings: GenerationSettings,
        load_in_4bit: bool = True,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
    ) -> "HuggingFaceGenerator":
        return cls._load(
            model_id=model_id,
            tokenizer_source=model_id,
            adapter_source=None,
            settings=settings,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )

    @classmethod
    def _load(
        cls,
        model_id: str,
        tokenizer_source: str,
        adapter_source: str | None,
        settings: GenerationSettings,
        load_in_4bit: bool,
        trust_remote_code: bool,
        attn_implementation: str | None,
    ) -> "HuggingFaceGenerator":
        stack = import_generation_stack()
        torch = stack["torch"]
        if load_in_4bit and not torch.cuda.is_available():
            raise RuntimeError("4-bit inference requires an NVIDIA CUDA device; use --no-load-in-4bit otherwise")
        if not torch.cuda.is_available():
            dtype = torch.float32
        elif torch.cuda.get_device_capability()[0] >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        try:
            tokenizer = stack["AutoTokenizer"].from_pretrained(
                tokenizer_source,
                trust_remote_code=trust_remote_code,
            )
        except (OSError, ValueError):
            if adapter_source is None or tokenizer_source == model_id:
                raise
            tokenizer = stack["AutoTokenizer"].from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
            )
        if adapter_source and Path(adapter_source).is_dir():
            chat_template = Path(adapter_source) / "chat_template.jinja"
            if chat_template.exists():
                tokenizer.chat_template = chat_template.read_text(encoding="utf-8")
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": trust_remote_code,
            "torch_dtype": dtype,
        }
        if load_in_4bit:
            kwargs["quantization_config"] = stack["BitsAndBytesConfig"](
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        model_class = (
            stack["AutoModelForImageTextToText"]
            if resolve_model_class(model_id) == "image-text-to-text"
            else stack["AutoModelForCausalLM"]
        )
        model = model_class.from_pretrained(model_id, **kwargs)
        if adapter_source:
            model = stack["PeftModel"].from_pretrained(model, adapter_source)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, torch=torch, settings=settings)

    def generate_messages(self, messages: Sequence[dict[str, str]]) -> GenerationResult:
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        prompt = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[1])
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.settings.max_new_tokens,
            "do_sample": self.settings.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.settings.do_sample:
            kwargs["temperature"] = self.settings.temperature
            kwargs["top_p"] = self.settings.top_p
        if self.settings.repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.settings.repetition_penalty
        if self.settings.no_repeat_ngram_size > 0:
            kwargs["no_repeat_ngram_size"] = self.settings.no_repeat_ngram_size

        with self.torch.no_grad():
            generated_ids = self.model.generate(**inputs, **kwargs)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        latency = time.perf_counter() - started
        completion_ids = generated_ids[0][input_tokens:]
        output_tokens = int(completion_ids.shape[0])
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            measurement=GenerationMeasurement.from_counts(input_tokens, output_tokens, latency),
        )

    def answer(self, question: str, contexts: Sequence[str]) -> GenerationResult:
        return self.generate_messages(build_answer_messages(question, contexts))

    def hypothetical_document(self, question: str) -> GenerationResult:
        return self.generate_messages(build_hyde_messages(question))

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
