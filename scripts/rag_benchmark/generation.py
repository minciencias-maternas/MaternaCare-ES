"""Lazy answer generation plus provider-specific HyDE generation."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .model_registry import ModelSpec, resolve_adapter_source
from .telemetry import GenerationMeasurement


ANSWER_WITH_CONTEXT_INSTRUCTION = (
    "Responde la pregunta clínica en español de forma precisa y directa. "
    "Usa exclusivamente la información respaldada por el contexto recuperado. "
    "Si el contexto no permite responder, indícalo claramente."
)
ANSWER_WITHOUT_CONTEXT_INSTRUCTION = (
    "Responde la pregunta clínica en español de forma precisa y directa usando tus conocimientos generales. "
    "Si no puedes establecer la respuesta con seguridad, indícalo claramente."
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


def build_answer_messages(
    question: str,
    contexts: Sequence[str],
    *,
    require_retrieved_context: bool,
) -> list[dict[str, str]]:
    if require_retrieved_context:
        context = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(contexts, start=1))
        content = f"{ANSWER_WITH_CONTEXT_INSTRUCTION}\n\nContexto recuperado:\n{context}\n\nPregunta: {question}"
    else:
        content = f"{ANSWER_WITHOUT_CONTEXT_INSTRUCTION}\n\nPregunta: {question}"
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

    def answer(
        self,
        question: str,
        contexts: Sequence[str],
        *,
        require_retrieved_context: bool,
    ) -> GenerationResult:
        return self.generate_messages(
            build_answer_messages(
                question,
                contexts,
                require_retrieved_context=require_retrieved_context,
            )
        )

    def hypothetical_document(self, question: str) -> GenerationResult:
        return self.generate_messages(build_hyde_messages(question))

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class OpenAIHydeGenerator:
    """Generate HyDE documents with the Responses API without loading local weights."""

    DEFAULT_TIMEOUT_SECONDS = 60.0
    DEFAULT_MAX_RETRIES = 2

    def __init__(self, client: Any, model_id: str, settings: GenerationSettings) -> None:
        self.client = client
        self.model_id = model_id
        self.settings = settings

    @classmethod
    def from_model(
        cls,
        model_id: str,
        settings: GenerationSettings,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> "OpenAIHydeGenerator":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the OpenAI client from requirements.txt for OpenAI HyDE generation") from exc
        return cls(
            client=OpenAI(timeout=timeout_seconds, max_retries=max_retries),
            model_id=model_id,
            settings=settings,
        )

    def hypothetical_document(self, question: str) -> GenerationResult:
        # This includes prompt construction, request transmission, and response receipt.
        started = time.perf_counter()
        prompt = build_hyde_messages(question)[0]["content"]
        request: dict[str, Any] = {
            "model": self.model_id,
            "input": prompt,
            "max_output_tokens": self.settings.max_new_tokens,
            "store": False,
        }
        # The default is deterministic at the benchmark level: do not send sampling
        # parameters unless the caller explicitly opted into sampling.
        if self.settings.do_sample:
            request["temperature"] = self.settings.temperature
        response = self.client.responses.create(**request)
        latency = time.perf_counter() - started
        if getattr(response, "status", None) != "completed":
            raise RuntimeError(f"OpenAI Responses API returned non-completed status: {getattr(response, 'status', None)!r}")
        if getattr(response, "incomplete_details", None) is not None:
            raise RuntimeError("OpenAI Responses API returned incomplete HyDE output")
        usage = getattr(response, "usage", None)
        if usage is None:
            raise RuntimeError("OpenAI Responses API returned no usage; cannot record benchmark token telemetry")
        try:
            input_tokens = int(usage.input_tokens)
            output_tokens = int(usage.output_tokens)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("OpenAI Responses API returned incomplete usage telemetry") from exc
        text = str(getattr(response, "output_text", "")).strip()
        if not text:
            raise RuntimeError("OpenAI Responses API returned no HyDE text")
        return GenerationResult(
            text=text,
            measurement=GenerationMeasurement.from_counts(input_tokens, output_tokens, latency),
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
