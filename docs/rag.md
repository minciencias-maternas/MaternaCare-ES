# Pipeline RAG de MaternaCare-ES

Este documento explica cómo funciona el sistema de retrieval-augmented generation (RAG) del proyecto: qué componentes lo forman, cómo se recupera el contexto, cómo se generan las respuestas y cómo se evalúan.

## 1. Visión general

El benchmark RAG responde las **328 preguntas del test de MaternaQA-es** (o las 10 de `sample10`) combinando un **corpus clínico recuperado** con un **modelo de generación** (base o QLoRA). El objetivo es medir cuánto mejora la respuesta al añadir contexto recuperado, con y sin HyDE.

La combinación posible es: **4 modelos × 3 estrategias = 12 configuraciones**.

| Modelo | Base | Adapter |
|--------|------|---------|
| `gemma4_base` | `google/gemma-4-E2B-it` | — |
| `gemma4_qlora` | `google/gemma-4-E2B-it` | `iue-edu/MaternaCare-ES-gemma4-qlora` |
| `medgemma_base` | `google/medgemma-1.5-4b-it` | — |
| `medgemma_qlora` | `google/medgemma-1.5-4b-it` | `iue-edu/MaternaCare-ES-medgemma-qlora` |

| Estrategia | Qué hace |
|------------|----------|
| `no_rag` | Sin recuperación. El modelo responde con sus conocimientos (`ANSWER_WITHOUT_CONTEXT_INSTRUCTION`). |
| `hybrid` | Recupera los top-k chunks con recuperación híbrida (BM25 + densa fusionadas con RRF) y los inyecta en el prompt. |
| `hyde` | Primero genera un documento hipotético con un LLM (HyDE), lo usa como query para recuperación densa, y los chunks recuperados se inyectan en el prompt. |

## 2. Código

El pipeline vive en `scripts/rag_benchmark/`, un paquete con una responsabilidad por módulo:

| Módulo | Responsabilidad |
|--------|-----------------|
| `cli.py` | Parser de argumentos y punto de entrada (`run_rag_benchmark.py` solo llama a `cli.main()`). |
| `data.py` | Carga y validación de preguntas y corpus. |
| `model_registry.py` | Catálogo de los 4 modelos (base/adapter, IDs HF, rutas locales). |
| `retrieval.py` | Recuperadores: BM25, denso (LanceDB), híbrido (fusión RRF). |
| `generation.py` | Generación de respuestas y de documentos HyDE (HF y OpenAI). |
| `metrics.py` | Evaluación con RAGAS (7 métricas) vía LLM-as-judge. |
| `runner.py` | Orquestación: recuperación → generación → evaluación → persistencia. |
| `hyde.py` | Caché de documentos hipotéticos. |
| `telemetry.py` | Muestreo de recursos (CPU/GPU/VRAM) durante el benchmark. |

## 3. Corpus y datos de entrada

- **Corpus** (`datasets/obstetrics/corpus/chunks.jsonl`, 2,268 chunks): cada chunk tiene `text`, `chunk_id` y metadatos (`source_pdf`, `section`, `pages`, `token_estimate`, `clinical_score`, ...). El `chunk_id` sigue el formato `{pdf_slug}_{secuencia_5_digitos}`.
- **Preguntas** (`datasets/obstetrics/qa/publication/qa_flat_jsonl/test.jsonl`, 328 filas): cada fila trae `qa_id`, `pregunta`, `respuesta` (referencia), `chunk_id` (el chunk del que deriva la pregunta) y más metadatos.
- **Validación**: el test debe referenciar exactamente **108 chunks distintos**, todos presentes en el corpus; si no, el benchmark aborta.

## 4. Recuperación (`retrieval.py`)

### 4.1 Recuperación densa (DenseRetriever)

- Embeddings con **`BAAI/bge-m3`** (revisión `0c6f0d...`, device CPU por defecto, batch 32, normalizados a longitud unitaria).
- Índice **LanceDB** en `artifacts/rag_benchmark/lancedb/`, tabla con nombre derivado del modelo + revisión + huella del corpus (`chunks_{model}_{revision}_{fingerprint[:16]}`). Si no existe, se construye.
- **No hay índice vectorial**: la búsqueda es **coseno exacto exhaustivo** (`lancedb_exact`) sobre todos los vectores. Aceptable por el tamaño del corpus.
- `score = -distancia_coseno` (a mayor score, más similar).

### 4.2 Recuperación BM25

- Tokenización simple (`\w+` en minúsculas) y ranking con `rank_bm25` (BM25Okapi) sobre el corpus completo.

### 4.3 Recuperación híbrida (HybridRetriever)

- Combina BM25 + densa: cada recuperador devuelve `candidate_k = min(len_corpus, max(k*4, 50))` candidatos.
- Fusiona los rankings con **Reciprocal Rank Fusion (RRF)**: cada chunk suma `1/(60 + rango)`, desempate por `chunk_id`. Devuelve los top-k.

### 4.4 HyDE

- Con `--strategy hyde`, la pregunta no se usa directamente como query: primero un LLM redacta un "documento hipotético" (`HYDE_INSTRUCTION`), y **ese documento es la query** para la recuperación densa.
- Proveedores: `openai` (por defecto, `gpt-5-mini-2025-08-07`) o `huggingface` (`Qwen/Qwen2.5-1.5B-Instruct`).
- Los documentos hipotéticos se cachean en `artifacts/rag_benchmark/lancedb/hyde_cache/{dataset_mode}/{modelo}/{fingerprint}.jsonl`; la huella deriva de proveedor, modelo, prompt, ajustes de generación y flags de carga. En cache hit se evita la llamada a la API.

## 5. Generación (`generation.py`)

### 5.1 Prompt de respuesta

Los prompts están en español y marcan explícitamente el uso del contexto:

- **Con contexto** (`ANSWER_WITH_CONTEXT_INSTRUCTION`): "Responde la pregunta clínica en español de forma precisa y directa. Usa exclusivamente la información respaldada por el contexto recuperado. Si el contexto no permite responder, indícalo claramente."
- **Sin contexto** (`ANSWER_WITHOUT_CONTEXT_INSTRUCTION`): "…usando tus conocimientos generales. Si no puedes establecer la respuesta con seguridad, indícalo claramente."

El contexto se serializa como bloques `[i]\n{text}` separados por `\n\n`.

### 5.2 Ajustes de generación (GenerationSettings)

| Parámetro | Valor por defecto |
|-----------|-------------------|
| `max_new_tokens` | 512 |
| `do_sample` | `False` |
| `temperature` | 0.7 |
| `top_p` | 0.9 |
| `repetition_penalty` | `None` |
| `no_repeat_ngram_size` | 0 |

### 5.3 Modelo de generación

- Si el modelo es QLoRA, se carga el **base 4-bit** y el adapter se fusiona en el momento del `from_pretrained` (via `adapter_source`), usando el `tokenizer` del modelo base.
- `--load-in-4bit` activado por defecto; requiere GPU con suficiente VRAM (las respuestas se generan con el modelo local, no con la API).
- Gemma 4 y MedGemma se detectan como `image-text-to-text` (multimodal), aunque aquí solo se usa texto.

## 6. Evaluación (`metrics.py`)

Usa **RAGAS 0.4.3** con 7 métricas:

| Métrica | Qué mide |
|---------|----------|
| `context_precision` | ¿El contexto recuperado es relevante para la respuesta? |
| `context_recall` | ¿El contexto cubre la referencia? |
| `faithfulness` | ¿La respuesta se apoya en el contexto? |
| `noise_sensitivity` | ¿El ruido del contexto afecta la respuesta? (modo `irrelevant`) |
| `answer_relevancy` | ¿La respuesta es relevante a la pregunta? |
| `answer_correctness` | ¿La respuesta coincide con la referencia? |
| `semantic_similarity` | Similitud semántica respuesta vs referencia |

- **LLM-as-judge**: el evaluador por defecto es `gpt-5.4-mini` con embeddings `text-embedding-3-large`. Para modelos `gpt-5.x` se usan `max_completion_tokens` y `temperature=1.0`; para el resto, `max_tokens`.
- Las métricas de contexto (`context_precision`, `context_recall`, `faithfulness`, `noise_sensitivity`) se **omiten en `no_rag`** (no hay contexto recuperado).
- Cada métrica se ejecuta de forma concurrente (límite por defecto 3) y se aísla: el fallo de una métrica no tira el resto; valores no finitos se registran como error.

## 7. Orquestación (`runner.py`)

Por cada pregunta, el runner sigue la secuencia:

```
retrieval → generation → evaluation → persistencia
```

- **Resume**: con `--resume` (activo por defecto), las preguntas ya escritas en el `.jsonl` de salida se saltan; útil para cortes de proceso o CUDA OOM.
- **Persistencia**: filas en el `.jsonl` de salida + un `_summary.json` con medias de métricas y operaciones.
- **Nombrado**: `{dataset_mode}__{strategy}__{model_key}__{fingerprint[:16]}` donde el fingerprint es sha256 de la configuración canónica (modelo, prompt, generación, carga, evaluador, datos, corpus). Cambiar cualquier cosa que afecte el resultado cambia el fingerprint → archivo nuevo, no colisión.
- **Telemetría**: muestreo de CPU/GPU/VRAM cada `--telemetry-interval-seconds` (0.5 s por defecto) durante la inferencia.

## 8. Cómo ejecutarlo

### Un solo experimento

```bash
python scripts/run_rag_benchmark.py \
  --dataset-mode maternaqa_test \
  --strategy hybrid \
  --model gemma4_qlora
```

Flags útiles: `--retrieval-k`, `--retrieval-embedding-model`, `--retrieval-device cpu|cuda`, `--hyde-provider`, `--evaluator-model`, `--max-new-tokens`, `--limit`, `--validate-data-only` (valida datos y sale), `--output-dir`.

### Matriz completa (12 configuraciones)

```bash
python scripts/run_experiment_matrix.py --mode sample10   # rápido
python scripts/run_experiment_matrix.py --mode full       # 328 preguntas × 12 configs
python scripts/run_experiment_matrix.py --mode smoke      # 1 pregunta × 1 combo
```

Con `--model` o `--strategy` se filtra la matriz. `--adapter-path` solo se permite con un único `--model` (un adapter no aplica a una matriz multi-modelo). Para correr cada modelo en paralelo entre máquinas, ver [RUN_BENCHMARK_MATRIX.md](./RUN_BENCHMARK_MATRIX.md).

### Salida de ejemplo

```json
{
  "dataset_mode": "maternaqa_test",
  "strategy": "hybrid",
  "model": "gemma4_qlora",
  "metrics": {
    "context_precision": 0.62, "context_recall": 0.81,
    "faithfulness": 0.88, "answer_relevancy": 0.91,
    "answer_correctness": 0.55, "semantic_similarity": 0.83
  },
  "operations": {
    "input_tokens": 483201, "output_tokens": 162048,
    "retrieval_latency_seconds": 18.2, "generation_latency_seconds": 2110.4,
    "end_to_end_latency_seconds": 2140.9, "output_tokens_per_second": 76.8
  }
}
```

## 9. Relación con el entrenamiento

El corpus, las preguntas y los modelos QLoRA usados aquí provienen de:

- **Corpus**: chunks generados por `extract_pdfs.py` → `utils.py` (ver [training-qlora.md](./training-qlora.md)).
- **Modelos**: adapters QLoRA entrenados con `train_qlora_trl.py` (ver [training-qlora.md](./training-qlora.md)).
- **Preguntas**: el test de `qa_flat_jsonl` es la partición de test de MaternaQA-es, misma distribución que las variantes SFT `sft_grounded`/`sft_closed_book`.