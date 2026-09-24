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

## Ejecución rápida: 10 + 10 preguntas

Si no hay tiempo para las 328 preguntas, cada compañero puede ejecutar **dos comandos para su modelo**: uno con las 10 preguntas de `sample10` y otro con las primeras 10 preguntas de `maternaqa_test`. Cada comando ejecuta las tres estrategias.

Por ejemplo, para `gemma4_base`:

```bash
python scripts/run_experiment_matrix.py \
  --mode sample10 \
  --model gemma4_base \
  --retrieval-device cuda \
  2>&1 | tee logs/gemma4_base__sample10.log

python scripts/run_experiment_matrix.py \
  --mode full \
  --model gemma4_base \
  --limit 10 \
  --retrieval-device cuda \
  2>&1 | tee logs/gemma4_base__maternaqa10.log
```

Cambie `gemma4_base` por el modelo asignado. El parámetro `--limit 10` solo limita la ejecución de `maternaqa_test`; `sample10` ya contiene exactamente 10 preguntas. Los resultados quedan separados por dataset y fingerprint.

## Monitorear la ejecución

Los comandos anteriores guardan toda la salida en `logs/<modelo>.log` y también la muestran en pantalla. Para observar el progreso desde otra terminal, ejecute:

```bash
tail -f logs/gemma4_base.log
```

Cambie el nombre por el modelo correspondiente. Los mensajes tienen el prefijo `[benchmark]` y muestran:

- Carga del dataset, corpus, índice de recuperación, modelo y evaluador.
- Preparación de HyDE, incluyendo documentos generados y aciertos de caché.
- Número de muestra y `qa_id` actual.
- Etapa actual: `retrieval`, `generation`, `evaluation` o `written`.
- Chunks recuperados, tokens generados y latencia de cada etapa.
- Errores por muestra y estado final de escritura.

Ejemplo:

```text
[benchmark] sample 31/328 qa_id=... stage=retrieval
[benchmark] sample 31/328 stage=retrieval_done chunks=8 latency=1.42s
[benchmark] sample 31/328 stage=generation_done latency=10.31s output_tokens=84
[benchmark] sample 31/328 stage=evaluation_done latency=7.85s
[benchmark] sample 31/328 stage=written status=ok
```

Esto permite identificar si el proceso está trabajando en recuperación, generación local, evaluación RAGAS o escritura del resultado. Las preguntas, prompts y respuestas completas no se imprimen en los logs.

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
