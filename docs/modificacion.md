# Cómo modificar MaternaCare-ES

Guía para extender el proyecto: agregar modelos, estrategias, métricas, datasets o pasos nuevos sin romper el resto.

## 0. Regla de oro: los fingerprints se encargan de ti

Toda salida del benchmark lleva un fingerprint derivado de su configuración. **Si cambias algo que afecta el resultado, el fingerprint cambia y se crea un archivo nuevo**; no sobrescribes resultados viejos. Por eso no hay que preocuparse por "ensuciar" salidas previas al experimentar.

## 1. Agregar un modelo al registro

Edita `MODEL_REGISTRY` en `scripts/rag_benchmark/model_registry.py`:

```python
"mi_modelo_qlora": ModelSpec(
    key="mi_modelo_qlora",
    model_id="org/MiAdapter",            # cómo se carga para generar
    base_model_id="org/ModeloBase",
    adapter_id="org/MiAdapter",          # adapter en el Hub
    local_adapter_candidates=("outputs/mi-modelo",),  # rutas locales
),
```

Si es un modelo **base** (sin adapter), deja `adapter_id=None`. El adapter se resuelve en este orden: `--adapter-path` explícito → `adapter_id` del Hub → rutas locales.

Para **entrenar** el adapter nuevo con QLoRA, ver [training-qlora.md](./training-qlora.md).

## 2. Agregar una estrategia RAG

Las estrategias se definen en `scripts/rag_benchmark/runner.py`:

1. Añade la opción al enum `Strategy` y a `cli.py` (`choices` del parser + rutas de argumentos).
2. Implementa la recuperación en `retrieve_for_strategy()` en `runner.py` — devuelve una lista de `RetrievedChunk` o `[]`.
3. Añade los argumentos específicos al CLI (si aplica).
4. El fingerprint incorpora la estrategia automáticamente (viene del nombre en `_experiment_configuration`).

Los recuperadores nuevos se implementan en `retrieval.py` (deben cumplir el protocolo `Retriever` con `search(query, k)`).

## 3. Cambiar el modelo de embeddings o el backend de recuperación

- **Embedding model/revisión**: flags `--retrieval-embedding-model` y `--retrieval-embedding-revision` (ver [configuracion.md](./configuracion.md)). Al cambiarlos, LanceDB crea una **tabla nueva** (el nombre incluye modelo + revisión + huella del corpus), así que no hay que borrar el índice viejo.
- **Backend**: hoy la búsqueda densa es coseno exacto (`lancedb_exact`) sin índice vectorial — correcto para 2,268 chunks. Si el corpus crece mucho, puedes crear un índice ANN de LanceDB en `DenseRetriever` (`retrieval.py`); conserva la normalización de vectores y la convención `score = -distancia`.
- **Constantes de RRF**: `reciprocal_rank_fusion()` en `retrieval.py` — `rank_constant=60`, `candidate_k = max(k*4, 50)`.

## 4. Cambiar métricas de evaluación

En `scripts/rag_benchmark/metrics.py`:

- Añadir/quitar métricas en `build_metrics()` (todas viven en `METRIC_NAMES`).
- Las métricas de contexto se omiten en `no_rag` vía `include_context_metrics=False` (se guardan como `None`); respeta esa separación.
- Cambiar el **LLM-as-judge**: flags `--evaluator-model` y `--embedding-model` sin tocar código. Si usas un modelo que no sea `gpt-5.x`, revisa `configure_llm_args` (los gpt-5.x requieren `max_completion_tokens` + `temperature=1.0`; el resto `max_tokens`).

## 5. Cambiar los prompts

En `scripts/rag_benchmark/generation.py`:

- `ANSWER_WITH_CONTEXT_INSTRUCTION` — prompt con contexto recuperado.
- `ANSWER_WITHOUT_CONTEXT_INSTRUCTION` — prompt sin contexto (`no_rag`).
- `HYDE_INSTRUCTION` — prompt del documento hipotético.

Cuidado: **el texto del prompt forma parte del fingerprint**. Al cambiarlo, las corridas anteriores siguen siendo válidas y las nuevas se guardan aparte. En un paper, documenta qué versión de prompt usaste.

## 6. Agregar una variante de dataset

En `scripts/train_qlora_trl.py`:

1. Añade la variante a `DATASET_VARIANTS` y genera sus archivos en `datasets/obstetrics/qa/publication/` (splits `train`/`validation`/`test` en formato chat `messages`, con `metadata`).
2. Pásala con `--dataset-variant`.
3. Si es un formato de datos nuevo (no `messages`), ajusta `split_prompt_completion` en `train_qlora_trl.py` para derivar prompt/completion.

Para el benchmark, las preguntas del test se leen de `qa_flat_jsonl/test.jsonl` (ver esquema en [estructura-y-datos.md](./estructura-y-datos.md)). Si cambias el esquema de campos requeridos, actualiza `JSONL_REQUIRED_FIELDS` / `MATERNAQA_REQUIRED_FIELDS` en `data.py` — y recuerda que el **corpus de 108 chunks de referencia** debe seguir existiendo.

## 7. Agregar un paso a la pipeline de datos

La pipeline (PDF → chunks → QA sintético → SFT) es un conjunto de scripts independientes:

| Paso | Script | Modifica si… |
|------|--------|--------------|
| Extracción de PDFs | `extract_pdfs.py` | cambias fuentes o reglas de exclusión del manifest |
| Chunking / limpieza | `utils.py` | tamaños de chunk, overlap, filtros de calidad |
| QA sintético | `generate_synthetic_qa.py` | el generador OpenAI, filtros, calidad |
| Variantes SFT | `build_lm_dataset.py`, `prepare_qa_publication_variants.py` | formatos de salida |

Cada paso es reanudable (`progress.json` / `--resume`), así que al modificar un filtro basta re-ejecutar ese paso (los anteriores ya están persistidos).

## 8. Validar cambios

- **Datos**: `--validate-data-only` valida muestras, corpus y chunks de referencia sin correr el benchmark.
- **Smoke**: `python scripts/run_experiment_matrix.py --mode smoke` (1 pregunta × 1 config, ~1 min).
- **Rápido**: `--mode sample10` (10 preguntas × 12 configs, ~15 min).
- **Tests**: la suite `tests/test_rag_benchmark.py` cubre CLI, datos, recuperación y runner. Corre `pytest tests/test_rag_benchmark.py`.

## 9. Gotchas comunes

- **`datasets/` local vs paquete HF**: la carpeta del repo hace sombra al paquete `datasets`. El training script ya lo resuelve (`remove_project_root_from_imports`); si creas scripts nuevos que importen `datasets` de HF, aplica la misma corrección.
- **CUDA OOM en benchmark**: mueve solo los embeddings a CPU con `--retrieval-device cpu`; la generación sigue en GPU.
- **`--assistant-only-loss` no existe**: el script lo rechaza; usa `completion_only_loss=True` (default).
- **Caché HyDE**: no se invalidará si solo cambias datos; borra `artifacts/rag_benchmark/lancedb/hyde_cache/` para regenerarla.