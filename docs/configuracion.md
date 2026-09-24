# Configuración de MaternaCare-ES

Este documento reúne todo lo configurable del proyecto: entorno, rutas por defecto, registro de modelos y argumentos de los scripts de entrenamiento y benchmark.

## 1. Entorno

### 1.1 Variables de entorno (`.env`)

Copia `.env.example` a `.env`:

| Variable | Para qué se usa |
|----------|-----------------|
| `OPENAI_API_KEY` | Evaluación RAGAS (LLM-as-judge en `metrics.py`), generación HyDE con OpenAI (`generation.py`), generación de QA sintético. |
| `HF_TOKEN` | Descarga de modelos gated del Hub (Gemma, MedGemma, Qwen, bge-m3) y publicación de adapters. |

### 1.2 Dependencias

| Archivo | Contenido |
|---------|-----------|
| `requirements.txt` | Dependencias comunes **sin torch** (ragas, lancedb, rank_bm25, transformers, peft, trl, etc.). |
| `requirements-cuda.txt` | Añade torch CUDA + bitsandbytes + accelerate. **Úsalo si tienes GPU.** |
| `requirements-cpu.txt` | Añade torch CPU (para recuperación/validación sin entrenar). |

## 2. Rutas por defecto

| Concepto | Ruta |
|----------|------|
| Corpus de chunks | `datasets/obstetrics/corpus/chunks.jsonl` |
| QA de referencia (10) | `datasets/sample10.jsonl` |
| Test de MaternaQA-es (328) | `datasets/obstetrics/qa/publication/qa_flat_jsonl/test.jsonl` |
| Dataset SFT (variantes) | `datasets/obstetrics/qa/publication/{sft_grounded,sft_closed_book}/` |
| Índice LanceDB + caché HyDE | `artifacts/rag_benchmark/lancedb/` |
| Salidas del benchmark | `outputs/rag_benchmark/` |
| Modelos entrenados (adapters) | `outputs/{gemma4,medgemma}-{grounded,qlora}/` |

## 3. Registro de modelos (`model_registry.py`)

Los modelos se definen en `MODEL_REGISTRY` de `scripts/rag_benchmark/model_registry.py`. Cada entrada es una `ModelSpec` con: `key`, `model_id` (ID de generación), `base_model_id`, `adapter_id` (HF), y `local_adapter_candidates` (rutas locales que se prueban antes de bajar del Hub).

```python
"gemma4_qlora": ModelSpec(
    key="gemma4_qlora",
    model_id="iue-edu/MaternaCare-ES-gemma4-qlora",
    base_model_id="google/gemma-4-E2B-it",
    adapter_id="iue-edu/MaternaCare-ES-gemma4-qlora",
    local_adapter_candidates=("outputs/gemma4-grounded", "outputs/gemma4-qlora"),
)
```

Resolución del adapter (`resolve_adapter_source`):
1. Si pasas `--adapter-path`, debe ser un directorio con `adapter_config.json` + `adapter_model.safetensors`/`.bin`.
2. Si no, se usa `adapter_id` del Hub.
3. En modelos base no hay adapter.

## 4. Configuración del entrenamiento QLoRA

Script: `scripts/train_qlora_trl.py`.

| Argumento | Default | Nota |
|-----------|---------|------|
| `--model-name` | — | obligatorio (ej. `google/gemma-4-E2B-it`) |
| `--dataset-hf-id` | `iue-edu/MaternaQA-es` | dataset del Hub |
| `--dataset-root` | `datasets/obstetrics/qa/publication` | carga local |
| `--dataset-files` | — | lista de JSONL locales |
| `--dataset-variant` | `sft_grounded` | `sft_grounded` \| `sft_closed_book` |
| `--output-dir` | — | obligatorio |
| `--resume-from-checkpoint` | `None` | reanuda desde un checkpoint |
| `--num-train-epochs` | 2.0 | |
| `--per-device-train-batch-size` | 1 | |
| `--gradient-accumulation-steps` | 8 | batch efectivo 8 |
| `--learning-rate` | 2e-4 | |
| `--warmup-ratio` | 0.03 | |
| `--max-grad-norm` | 0.3 | |
| `--lora-r` | 16 | final: 32 |
| `--lora-alpha` | 16 | final: 64 |
| `--lora-dropout` | 0.05 | |
| `--max-length` | 1024 | |
| `--logging-steps` | 10 | |
| `--max-steps` | `None` | útil para smoke tests |
| `--train-limit` / `--eval-limit` | `None` | recorta el dataset |
| `--model-class` | auto | auto \| causal-lm \| image-text-to-text |
| `--dry-run` | `False` | solo carga el dataset y sale |
| `--allow-cpu` | `False` | entrenar sin GPU (no recomendado) |
| `--gradient-checkpointing` | `True` | |
| `--packing` | `False` | |
| `--seed` | 3407 | |

## 5. Configuración del benchmark RAG

Scripts: `scripts/run_rag_benchmark.py` y `scripts/run_experiment_matrix.py`.

### 5.1 Argumentos generales

| Argumento | Default | Nota |
|-----------|---------|------|
| `--dataset-mode` | — | obligatorio: `sample10` \| `maternaqa_test` |
| `--strategy` | — | obligatorio: `no_rag` \| `hybrid` \| `hyde` |
| `--model` | — | clave de `MODEL_REGISTRY` (obligatorio) |
| `--sample10-path` | `datasets/sample10.jsonl` | |
| `--maternaqa-test-path` | `datasets/obstetrics/qa/publication/qa_flat_jsonl/test.jsonl` | |
| `--corpus-path` | `datasets/obstetrics/corpus/chunks.jsonl` | |
| `--index-dir` | `artifacts/rag_benchmark/lancedb` | |
| `--output-dir` | `outputs/rag_benchmark` | |
| `--limit` | `None` | limita el nº de preguntas |
| `--resume` | `True` | salta preguntas ya escritas |
| `--validate-data-only` | `False` | valida datos e imprime JSON, sale |
| `--telemetry-interval-seconds` | 0.5 | 0 desactiva el muestreo |

### 5.2 Recuperación

| Argumento | Default |
|-----------|---------|
| `--retrieval-k` | 5 |
| `--retrieval-embedding-model` | `BAAI/bge-m3` |
| `--retrieval-embedding-revision` | `0c6f0d0ea8f284b9070c3ffaa50677440943f984` |
| `--retrieval-device` | `cpu` |
| `--retrieval-batch-size` | 32 |

### 5.3 HyDE

| Argumento | Default |
|-----------|---------|
| `--hyde-provider` | `openai` (`openai` \| `huggingface`) |
| `--hyde-generator-model` | `gpt-5-mini-2025-08-07` (openai) / `Qwen/Qwen2.5-1.5B-Instruct` (hf) |
| `--hyde-max-new-tokens` | 256 |
| `--hyde-load-in-4bit` | `True` |
| `--hyde-temperature` / `--hyde-top-p` / `--hyde-repetition-penalty` / `--hyde-no-repeat-ngram-size` | 0.7 / 0.9 / — / 0 |

### 5.4 Evaluación RAGAS

| Argumento | Default |
|-----------|---------|
| `--evaluator-model` | `gpt-5.4-mini` |
| `--embedding-model` | `text-embedding-3-large` |
| `--evaluator-max-completion-tokens` | 2048 |
| `--evaluator-timeout-seconds` | 600 |
| `--evaluator-concurrency` | 3 |

### 5.5 Generación

| Argumento | Default |
|-----------|---------|
| `--adapter-path` | `None` (ruta local del adapter) |
| `--max-new-tokens` | 512 |
| `--do-sample` | `False` |
| `--temperature` | 0.7 |
| `--top-p` | 0.9 |
| `--repetition-penalty` | `None` |
| `--no-repeat-ngram-size` | 0 |
| `--load-in-4bit` | `True` |
| `--trust-remote-code` | `False` |
| `--attn-implementation` | `None` |

## 6. Matriz de experimentos (`run_experiment_matrix.py`)

| Argumento | Default |
|-----------|---------|
| `--mode` | menú interactivo | `smoke` \| `sample10` \| `full` |
| `--model` | todos | filtra la matriz |
| `--strategy` | todas | filtra la matriz |
| `--adapter-path` | `None` | requiere `--model` único |

Modos: `smoke` (1 pregunta × 1 combo, ~1 min), `sample10` (10 preguntas × 12 configs, ~15 min), `full` (328 preguntas × 12 configs, horas).

## 7. Identidad y fingerprints de configuración

La salida del benchmark se nombra con un fingerprint (sha256 de la configuración canónica). **Cualquier cambio que altere el resultado genera un archivo nuevo en vez de sobrescribir**:

- Modelo (id, base, adapter source y provenance), prompt, ajustes de generación, flags de carga.
- Estrategia, `retrieval_k`, embedding model/revisión, proveedor HyDE.
- Evaluador y embedding de evaluación.
- sha256 del dataset y del corpus.

Esto garantiza que `outputs/rag_benchmark/maternaqa_test__hybrid__gemma4_qlora__1d9c6e7c.jsonl` corresponda exactamente a una configuración reproducible.