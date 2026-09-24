# Estructura y datos de MaternaCare-ES

Este documento muestra cómo se ve el proyecto por dentro: el árbol del repositorio, los esquemas de datos (JSONL) en cada etapa de la pipeline y las salidas de entrenamiento y benchmark.

## 1. Árbol del repositorio

```
MaternaCare-ES/
├── README.md
├── requirements.txt            # dependencias comunes (sin torch)
├── requirements-cuda.txt       # + torch CUDA, bitsandbytes
├── requirements-cpu.txt        # + torch CPU
├── .env.example                # OPENAI_API_KEY, HF_TOKEN
├── datasets/
│   ├── sample10.jsonl          # 10 pares QA de referencia (smoke/sample)
│   ├── Preguntas y Respuestas.xlsx  # fuente de sample10
│   └── obstetrics/
│       ├── corpus/
│       │   └── chunks.jsonl    # 2,268 chunks de recuperación
│       └── qa/
│           ├── publication/
│           │   ├── qa_flat_jsonl/{train,validation,test}.jsonl
│           │   ├── sft_grounded/{train,validation,test}.jsonl
│           │   └── sft_closed_book/{train,validation,test}.jsonl
│           └── final/{train,validation,test}/
│               ├── raw.jsonl           # QA sintéticos crudos
│               ├── sft.jsonl           # QA en formato SFT
│               ├── progress.json       # checkpoint de generación
│               └── generation_report.json
├── artifacts/
│   └── rag_benchmark/lancedb/   # índice LanceDB + caché HyDE
├── outputs/
│   ├── gemma4-base/, gemma4-grounded/, gemma4-qlora/
│   ├── medgemma-base/, medgemma-grounded/, medgemma-qlora/
│   ├── smoke-*/                # outputs de smoke tests
│   ├── rag_benchmark/          # resultados del benchmark
│   ├── logs/                   # logs por modelo (tee)
│   └── csv_outputs/
├── scripts/
│   ├── extract_pdfs.py          # PDF → raw_pages + manifest
│   ├── clean_text.py, document_manifest.py
│   ├── utils.py                 # chunking, dedupe, filtros, splits
│   ├── generate_synthetic_qa.py # QA sintético (OpenAI)
│   ├── build_lm_dataset.py, prepare_qa_publication_variants.py
│   ├── train_qlora_trl.py       # entrenamiento QLoRA
│   ├── run_rag_benchmark.py     # entry → rag_benchmark.cli
│   ├── run_experiment_matrix.py # matriz 4 modelos × 3 estrategias
│   ├── inference_qlora.py, inference_base.py
│   ├── evaluate_qa_with_ragas.py, evaluate_model_predictions.py
│   ├── convert_eval_to_csv.py, backfill_prediction_metadata.py
│   ├── audit_dataset.py, vram_smoke_test*.py
│   └── rag_benchmark/
│       ├── cli.py, data.py, retrieval.py, generation.py,
│       ├── hyde.py, metrics.py, model_registry.py, runner.py,
│       └── telemetry.py
├── tests/test_rag_benchmark.py
└── docs/                        # esta documentación
    ├── README.md, rag.md, training-qlora.md,
    ├── configuracion.md, modificacion.md, estructura-y-datos.md,
    ├── RUN_BENCHMARK_MATRIX.md, operational-metrics.md
```

## 2. Esquemas de datos por etapa

### 2.1 Corpus de chunks (`corpus/chunks.jsonl`)

```json
{
  "text": "…texto completo del chunk…",
  "chunk_id": "gpc_atencion_prenatal_de_bajo_riesgo_2023_00042",
  "source_pdf": "gpc_atencion_prenatal_de_bajo_riesgo_2023.pdf",
  "section": "Reevaluación",
  "pages": [12, 13],
  "token_estimate": 812,
  "clinical_score": 7,
  "quality_flags": [],
  "doc_type": "GPC"
}
```

- `chunk_id`: `{pdf_slug}_{secuencia_5_digitos}`.
- Generados por `extract_pdfs.py` + `utils.py` (chunks de 500–1200 tokens, overlap 100).

### 2.2 QA de referencia (`sample10.jsonl`, `qa_flat_jsonl/test.jsonl`)

`sample10` (10 filas, smoke):

```json
{
  "qa_id": "sample10_001",
  "pregunta": "¿En qué momento y quién debe reevaluar…?",
  "respuesta": "El Ginecobstetra en la semana 28 - 30 y semana 34 – 36.",
  "contexto_fuente": "",
  "chunk_id": null,
  "source_pdf": null,
  "pages": [],
  "section": "DATA_GT",
  "dataset_mode": "sample10",
  "provenance": {
    "source_file": "datasets/Preguntas y Respuestas.xlsx",
    "source_sheet": "DATA_GT",
    "source_row": 2
  }
}
```

`qa_flat_jsonl/test.jsonl` (328 filas, el test real):

```json
{
  "qa_id": "gpc_atencion_prenatal_de_bajo_riesgo_2023_00042_q1",
  "chunk_id": "gpc_atencion_prenatal_de_bajo_riesgo_2023_00042",
  "source_pdf": "gpc_….pdf",
  "section": "…",
  "section_type": "…",
  "content_role": "evidence",
  "topics": ["…"],
  "split": "test",
  "pages": [12],
  "clinical_score": 7,
  "token_estimate": 90,
  "pregunta": "…",
  "respuesta": "…",
  "tipo": "factual",
  "dificultad": "media",
  "contexto_fuente": "…"
}
```

El benchmark solo exige `qa_id`, `pregunta`, `respuesta` (más validación de que los `chunk_id` existen en el corpus).

### 2.3 Variantes SFT (`sft_grounded/`, `sft_closed_book/`)

Formato chat `messages` + `metadata` (se conserva toda la traza del chunk):

```json
{
  "messages": [
    {"role": "system", "content": "Eres un asistente especializado en obstetricia y ginecología…"},
    {"role": "user", "content": "Contexto fuente:\n…\n\nPregunta:\n…"},
    {"role": "assistant", "content": "…respuesta…"}
  ],
  "metadata": {
    "source": "obstetrics_spanish_synthetic",
    "source_pdf": "gpc_….pdf",
    "chunk_id": "gpc_….00042",
    "qa_id": "…",
    "pages": [12],
    "section": "…",
    "section_type": "…",
    "content_role": "evidence",
    "topics": ["…"],
    "split": "test",
    "clinical_score": 7,
    "token_estimate": 90,
    "tipo": "factual",
    "dificultad": "media",
    "contexto_fuente": "…"
  }
}
```

Diferencias: en `sft_closed_book` el mensaje `user` no incluye el `Contexto fuente:`.

### 2.4 QA final (`qa/final/{split}/`)

- `raw.jsonl` — pares QA sintéticos generados por OpenAI (con `contexto_fuente`).
- `sft.jsonl` — versiones SFT para entrenamiento.
- `progress.json` — checkpoint para reanudar la generación.
- `generation_report.json` — resumen del proceso.

## 3. Salidas del entrenamiento (`outputs/<run>/`)

```
outputs/gemma4-grounded/
├── adapter_config.json        # config LoRA (r, alpha, target_modules)
├── adapter_model.safetensors  # pesos del adapter (~200 MB)
├── tokenizer/                 # tokenizer guardado
├── checkpoint-1274/           # checkpoints por época (para resume)
└── runs/                      # tensorboard
```

## 4. Salidas del benchmark (`outputs/rag_benchmark/`)

Cada configuración produce dos archivos con el patrón:

```
{dataset_mode}__{strategy}__{model_key}__{fingerprint[:16]}.jsonl        # una fila por pregunta
{dataset_mode}__{strategy}__{model_key}__{fingerprint[:16]}_summary.json # resumen agregado
```

Ejemplo real: `maternaqa_test__hybrid__gemma4_qlora__1d9c6e7c….jsonl`.

### 4.1 Filas del `.jsonl` (una por pregunta)

`qa_id`, `question`, `reference`, estrategia, `retrieved_chunks` (con `chunk_id` y score), `hypothetical_document` (en hyde), `response`, y las **7 métricas RAGAS** (con `metric_errors` si alguna falló).

### 4.2 `_summary.json`

```json
{
  "dataset_mode": "maternaqa_test",
  "strategy": "hybrid",
  "model": "gemma4_qlora",
  "model_id": "iue-edu/MaternaCare-ES-gemma4-qlora",
  "configuration_fingerprint": "1d9c6e7c…",
  "rows": 328,
  "rows_with_error": 0,
  "metrics": {
    "context_precision": 0.62,
    "context_recall": 0.81,
    "faithfulness": 0.88,
    "answer_relevancy": 0.91,
    "answer_correctness": 0.55,
    "semantic_similarity": 0.83
  },
  "operations": {
    "input_tokens": 483201,
    "output_tokens": 162048,
    "total_tokens": 645249,
    "retrieval_latency_seconds": 18.2,
    "generation_latency_seconds": 2110.4,
    "end_to_end_latency_seconds": 2140.9,
    "output_tokens_per_second": 76.8
  },
  "configuration": {
    "retrieval_k": 5,
    "retrieval_embedding_model": "BAAI/bge-m3",
    "hyde_generator_model": "gpt-5-mini-2025-08-07",
    "evaluator_model": "gpt-5.4-mini",
    "embedding_model": "text-embedding-3-large",
    "output_jsonl": "…/maternaqa_test__hybrid__gemma4_qlora__1d9c6e7c….jsonl",
    "dataset": {"mode": "maternaqa_test", "path": "…", "sha256": "…", "limit": null},
    "corpus": {"path": "…", "sha256": "…"},
    "strategy": "hybrid",
    "retrieval": {"backend": "lancedb_exact", "k": 5, "model": "BAAI/bge-m3",
                  "revision": "0c6f0d0e…", "device": "cpu", "batch_size": 32},
    "hyde": {…},
    "answer": {"key": "gemma4_qlora", "model_id": "…", "base_model_id": "google/gemma-4-E2B-it",
               "adapter_source": "…", "adapter_provenance": "registry_remote",
               "prompt": "…", "generation": {…}, "load_in_4bit": true,
               "trust_remote_code": false, "attn_implementation": null},
    "evaluator": {"model": "gpt-5.4-mini", "embedding_model": "text-embedding-3-large",
                  "max_completion_tokens": 2048, "timeout_seconds": 600}
  }
}
```

El bloque `configuration` es **autocontenido y reproducible**: con él puedes reconstruir exactamente el experimento. El `sha256` de dataset y corpus garantiza que se evaluó exactamente con esos datos.

## 5. Cómo se ve el proceso corriendo

El runner loguea cada pregunta con su etapa (`[benchmark] …`):

```
[benchmark] maternaqa_test hybrid gemma4_qlora: sample 1/328
[benchmark]   stage retrieval  chunks=5 latency=0.42s
[benchmark]   stage generation latency=6.1s output_tokens=512
[benchmark]   stage evaluation ok
[benchmark]   stage written
```

La matriz `run_experiment_matrix.py` agrega un resumen por experimento (OK/FAIL, filas, duración, métricas) y una tabla final. Con `--validate-data-only` imprime un JSON de validación: `{dataset_mode, samples, corpus_chunks, reference_chunk_ids}`.

## 6. Datos derivados de los resultados

- `convert_eval_to_csv.py` — convierte evaluaciones a CSV (`outputs/csv_outputs/`) para análisis en tablas.
- `evaluate_qa_with_ragas.py` / `evaluate_model_predictions.py` — evaluaciones RAGAS independientes del benchmark (sobre `raw_*.jsonl` o predicciones de modelos).
- `operational-metrics.md` — notas operativas de métricas (ver [operational-metrics.md](./operational-metrics.md)).