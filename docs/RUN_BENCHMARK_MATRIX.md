# Ejecutar el benchmark completo

Cada persona debe usar el mismo commit del proyecto y ejecutar **un modelo diferente**.
No es necesario crear ramas ni modificar `--adapter-path`: los adaptadores registrados se cargan automáticamente.

## 1. Validar los datos

Desde la raíz del proyecto:

```bash
python scripts/run_experiment_matrix.py \
  --mode full \
  --model gemma4_base \
  --validate-data-only
```

Debe aparecer un resultado con `samples: 328`.

## 2. Ejecutar el modelo asignado

Primero cree la carpeta de logs:

```bash
mkdir -p logs
```

Después copie y pegue **solo el comando correspondiente al modelo asignado**.

### Gemma 4 base

```bash
python scripts/run_experiment_matrix.py \
  --mode full \
  --model gemma4_base \
  --retrieval-device cuda \
  2>&1 | tee logs/gemma4_base.log
```

### Gemma 4 QLoRA

```bash
python scripts/run_experiment_matrix.py \
  --mode full \
  --model gemma4_qlora \
  --retrieval-device cuda \
  2>&1 | tee logs/gemma4_qlora.log
```

### MedGemma base

```bash
python scripts/run_experiment_matrix.py \
  --mode full \
  --model medgemma_base \
  --retrieval-device cuda \
  2>&1 | tee logs/medgemma_base.log
```

### MedGemma QLoRA

```bash
python scripts/run_experiment_matrix.py \
  --mode full \
  --model medgemma_qlora \
  --retrieval-device cuda \
  2>&1 | tee logs/medgemma_qlora.log
```

## Qué ejecuta cada comando

Cada modelo ejecuta las tres estrategias sobre las 328 preguntas:

- `no_rag`: sin recuperación; usa el conocimiento paramétrico del modelo.
- `hybrid`: recuperación BM25 + búsqueda semántica.
- `hyde`: genera un documento hipotético y recupera evidencia real mediante búsqueda semántica.

## Qué entregar al finalizar

Entregue estos archivos:

```text
outputs/rag_benchmark/
logs/<modelo>.log
```

Los resultados incluyen el modelo, la estrategia y un fingerprint en el nombre, por lo que no se sobrescriben entre sí.

## Si aparece `CUDA out of memory`

Cambie únicamente `cuda` por `cpu`:

```bash
--retrieval-device cpu
```

El modelo generador seguirá usando la GPU; esta opción solo mueve a CPU el modelo de embeddings utilizado para la recuperación.
