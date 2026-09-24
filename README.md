<div align="center">

<img src="./public/app-icon.png" alt="MaternaCare-ES logo" width="140" height="140" />

# MaternaCare-ES

**MaternaCare-ES es un proyecto de fine-tuning de modelos de lenguaje especializados en preguntas y respuestas clínicas de ginecología y obstetricia en español. Entrena, evalúa y publica adapters QLoRA para Gemma 4 y MedGemma 1.5 usando el dataset MaternaQA-es.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=1e293b)](https://python.org)
[![Models](https://img.shields.io/badge/Models-2%20adapters%20QLoRA-8B5CF6?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=1e293b)](https://huggingface.co/iue-edu)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models%20%26%20Datasets-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=1e293b)](https://huggingface.co/iue-edu)
[![Base Models](https://img.shields.io/badge/Base%20Models-Gemma%204%20%7C%20MedGemma-10B3C6?style=for-the-badge&logo=google&logoColor=white&labelColor=1e293b)](https://huggingface.co/google)
[![Licencia](https://img.shields.io/badge/Licencia-MIT-FACC15?style=for-the-badge&logo=opensourceinitiative&logoColor=black&labelColor=1e293b)](./LICENSE)

[Modelos Publicados](#modelos-publicados) · [Entrenamiento](#entrenamiento-qlora) · [Inferencia](#inferencia) · [Evaluación](#métricas-de-evaluación) · [Reproducibilidad](#reproducibilidad)

</div>

---

> [!IMPORTANT]
> Este repositorio documenta el proceso completo de fine-tuning de modelos de lenguaje para QA clínico en español. Incluye scripts de entrenamiento QLoRA, inferencia, evaluación con métricas basadas en LLM-as-judge (RAGAS) y publicación de adapters en Hugging Face.

## ¿Qué es MaternaCare-ES?

**MaternaCare-ES** es un proyecto de investigación que aplica técnicas de fine-tuning a modelos de lenguaje de última generación para crear asistentes especializados en preguntas y respuestas clínicas sobre salud materna y perinatal en español.

El proyecto incluye:

- **Scripts reproducibles** de entrenamiento QLoRA con TRL y PEFT.
- **Adapters entrenados** para Gemma 4 E2B y MedGemma 1.5 4B.
- **Evaluaciones completas** con métricas basadas en LLM-as-judge (RAGAS: context precision, context recall, faithfulness, noise sensitivity, answer relevancy, answer correctness y semantic similarity).
- **Checkpoints intermedios** para reanudar entrenamiento o auditar el proceso.
- **Documentación técnica** de decisiones de arquitectura y hiperparámetros.

> [!NOTE]
> El dataset de entrenamiento utilizado es **MaternaQA-es** (5.727 pares Q+A en español), disponible en Hugging Face: [`iue-edu/MaternaQA-es`](https://huggingface.co/datasets/iue-edu/MaternaQA-es).

## Modelos Publicados

Los adapters QLoRA entrenados están disponibles en Hugging Face. Cada repositorio contiene el adapter final, checkpoints intermedios, configuraciones de entrenamiento, evaluaciones y logs de TensorBoard.

| Modelo Base | Adapter QLoRA | Parámetros | Uso |
|-------------|---------------|------------|-----|
| **Gemma 4 E2B IT** | [`iue-edu/MaternaCare-ES-gemma4-qlora`](https://huggingface.co/iue-edu/MaternaCare-ES-gemma4-qlora) | 2B | QA clínico general |
| **MedGemma 1.5 4B IT** | [`iue-edu/MaternaCare-ES-medgemma-qlora`](https://huggingface.co/iue-edu/MaternaCare-ES-medgemma-qlora) | 4B | QA clínico con enfoque médico |

### Contenido de cada repositorio

- **Adapter final** (`adapter_model.safetensors`) para inferencia directa.
- **Checkpoints intermedios** (`checkpoint-637/`, `checkpoint-1274/`) para reanudar entrenamiento.
- **Configuraciones** (`adapter_config.json`, `training_args.bin`) con hiperparámetros exactos.
- **Evaluaciones** (`test_predictions.jsonl`, `test_eval.jsonl`) con predicciones y métricas.
- **Logs de TensorBoard** (`runs/`) para visualizar el entrenamiento.

### Descargar adapters

```bash
# Instala huggingface-cli
pip install huggingface-hub

# Descarga el adapter completo (incluye checkpoints)
huggingface-cli download iue-edu/MaternaCare-ES-gemma4-qlora --local-dir outputs/gemma4-grounded

# Descarga solo el adapter final (más ligero, ~200MB)
huggingface-cli download iue-edu/MaternaCare-ES-gemma4-qlora adapter_config.json adapter_model.safetensors --local-dir outputs/gemma4-grounded
```

## Arquitectura Híbrida

Este proyecto usa una separación de responsabilidades entre **GitHub** (código) y **Hugging Face** (artefactos pesados):

| Plataforma | Contenido | Razón |
|------------|-----------|-------|
| **GitHub** (este repo) | Scripts, configs, documentación | Control de versiones para código |
| **Hugging Face** | Adapters, checkpoints, dataset externo y logs | Soporta archivos >100MB vía LFS |

> [!TIP]
> Los checkpoints completos (optimizer state, pesos) se preservan en Hugging Face. Esto permite que cualquier investigador descargue el estado exacto del entrenamiento para reanudarlo o auditarlo.

## Estructura del Repositorio

```text
MaternaCare-ES/
├── scripts/                           # Entrenamiento, inferencia y evaluación
│   ├── train_qlora_trl.py             # Entrenamiento QLoRA con TRL
│   ├── inference_qlora.py             # Inferencia con adapter QLoRA
│   ├── inference_base.py              # Inferencia con modelo base (baseline)
│   ├── evaluate_model_predictions.py  # Evaluación de predicciones
│   ├── evaluate_qa_with_ragas.py      # Evaluación con RAGAS
│   ├── convert_eval_to_csv.py         # Conversión de evaluaciones a CSV
│   ├── backfill_prediction_metadata.py
│   ├── vram_smoke_test.py             # Smoke test directo en GPU
│   └── vram_smoke_test_offload.py     # Smoke test con CPU offloading
│   └── backfill_prediction_metadata.py
├── outputs/                           # Pesos, checkpoints y evaluaciones
│   ├── gemma4-grounded/               # Adapter Gemma 4 grounded
│   ├── medgemma-grounded/             # Adapter MedGemma grounded
│   ├── smoke-gemma4/                  # Smoke test Gemma 4
│   ├── smoke-medgemma/                # Smoke test MedGemma
│   ├── gemma4-base/                   # Baseline Gemma 4
│   ├── medgemma-base/                 # Baseline MedGemma
│   └── logs/                          # Logs de ejecución
├── datasets/                          # Mirror local opcional de MaternaQA-es
│   └── obstetrics/qa/publication/     # Variantes SFT publicadas
├── csv_outputs/                       # Resultados agregados en CSV
├── docs/research_notes/               # Notas técnicas del proyecto
├── papers/                            # Documentos del paper
└── requirements.txt
```

## Uso Rápido

### 1. Instalar dependencias

```bash
git clone https://github.com/JhonHander/MaternaCare-ES.git
cd MaternaCare-ES
python -m venv .venv && source .venv/bin/activate
```

Elige el archivo de requisitos según tu backend de PyTorch:

- **Con GPU NVIDIA (RTX recomendado):**
  ```bash
  pip install -r requirements-cuda.txt
  ```
- **Sin GPU / solo CPU:**
  ```bash
  pip install -r requirements-cpu.txt
  ```

Verifica la instalación:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

> [!WARNING]
> No ejecutes `pip install -r requirements.txt` directamente: ese archivo contiene las dependencias comunes pero **no incluye PyTorch**. Si ya instalaste una versión CPU-only de torch por error, recrea el entorno virtual (`rm -rf .venv && python -m venv .venv`) antes de instalar con `requirements-cuda.txt`.

> [!NOTE]
> Para descargar modelos de Hugging Face se requiere `HF_TOKEN`. Los modelos Gemma 4 y MedGemma son **gated** — necesitas aceptar los términos en huggingface.co antes de usarlos.

### 2. Verificar setup con smoke test (~3 min)

```bash
python scripts/train_qlora_trl.py \
  --model-name google/gemma-4-E2B-it \
  --dataset-variant sft_grounded \
  --output-dir outputs/smoke-gemma4 \
  --max-steps 10 \
  --train-limit 64 \
  --eval-limit 32
```

## Entrenamiento QLoRA

El entrenamiento usa QLoRA (Quantized Low-Rank Adaptation) para fine-tuning eficiente en GPU consumer.

### Requisitos

- [ ] `pip install -r requirements-cuda.txt` (usar `requirements-cpu.txt` únicamente si no se dispone de GPU NVIDIA)
- [ ] `huggingface-cli login` o `export HF_TOKEN=<tu_token>`
- [ ] Aceptar términos de Gemma 4 / MedGemma en Hugging Face
- [ ] GPU NVIDIA con ≥ 16 GB VRAM

### Entrenamiento completo (~2–4 h según GPU)

**Gemma 4 E2B:**

```bash
# Grounded (recomendado) — contexto + pregunta → respuesta
python scripts/train_qlora_trl.py \
  --model-name google/gemma-4-E2B-it \
  --dataset-variant sft_grounded \
  --output-dir outputs/gemma4-grounded \
  --epochs 2 \
  --lora-r 32 --lora-alpha 64

# Closed-book — solo pregunta → respuesta
python scripts/train_qlora_trl.py \
  --model-name google/gemma-4-E2B-it \
  --dataset-variant sft_closed_book \
  --output-dir outputs/gemma4-closed-book
```

**MedGemma 1.5 4B:**

```bash
# Grounded (recomendado)
python scripts/train_qlora_trl.py \
  --model-name google/medgemma-1.5-4b-it \
  --dataset-variant sft_grounded \
  --output-dir outputs/medgemma-grounded

# Closed-book
python scripts/train_qlora_trl.py \
  --model-name google/medgemma-1.5-4b-it \
  --dataset-variant sft_closed_book \
  --output-dir outputs/medgemma-closed-book
```

> **¿Qué hace el script?** Carga el modelo en 4 bits, entrena adapters LoRA (r=32, alpha=64) sobre todas las capas lineales, calcula pérdida solo sobre la respuesta esperada, y guarda solo los adapters (~200MB). El modelo base no se modifica.

### Variantes del dataset

| Variante | Entrada del modelo | Cuándo usar |
|----------|-------------------|-------------|
| `sft_grounded` | Contexto clínico + Pregunta → Respuesta | **Recomendado.** El modelo razona sobre evidencia documental. |
| `sft_closed_book` | Solo Pregunta → Respuesta | Evalúa internalización del dominio sin contexto explícito. |

## Inferencia

Genera predicciones sobre el split de test con un adapter entrenado:

```bash
python scripts/inference_qlora.py \
  --model-path outputs/gemma4-grounded \
  --dataset-variant sft_grounded \
  --split test \
  --output-path outputs/gemma4-grounded/test_predictions.jsonl
```

Para comparar contra el modelo base (sin fine-tuning):

```bash
python scripts/inference_base.py \
  --model-name google/gemma-4-E2B-it \
  --dataset-variant sft_grounded \
  --split test \
  --output-path outputs/gemma4-base/test_predictions.jsonl
```

## Métricas de Evaluación

El proyecto evalúa las predicciones utilizando RAGAS con un LLM-as-judge (GPT-4o-mini) y embeddings (text-embedding-3-small) para capturar diferentes aspectos de la calidad:

| Métrica | Qué mide | Rango |
|---------|----------|-------|
| **Context Precision** | Qué tan relevantes son los fragmentos recuperados y su orden | 0–1 |
| **Context Recall** | Qué tanto de la información necesaria aparece en el contexto recuperado | 0–1 |
| **Faithfulness** | Qué tan fiel es la respuesta al contexto proporcionado | 0–1 |
| **Noise Sensitivity** | Cuánto afecta el contexto irrelevante a la respuesta (`mode=irrelevant`) | 0–1 |
| **Answer Relevancy** | Qué tan relevante es la respuesta respecto a la pregunta | 0–1 |
| **Answer Correctness** | Qué tan correcta es la respuesta comparada con la referencia | 0–1 |
| **Semantic Similarity** | Similitud semántica entre la respuesta y la referencia (embeddings) | 0–1 |

El timeout predeterminado de RAGAS es de **600 segundos por métrica y por predicción**. Los errores de una métrica se conservan en el campo `error` o `metric_errors`; un `null` acompañado de ese campo no debe interpretarse como una puntuación válida. En `no_rag`, las métricas que requieren contexto recuperado se omiten intencionalmente.

### Evaluación de predicciones

```bash
# Evaluar predicciones con RAGAS (faithfulness, noise sensitivity, answer relevancy, answer correctness, semantic similarity)
python scripts/evaluate_model_predictions.py \
  --input outputs/gemma4-grounded/test_predictions.jsonl \
  --output outputs/gemma4-grounded/test_eval.jsonl
```

### Evaluación de calidad del dataset

```bash
# Evaluar calidad de los pares Q+A del dataset (no de las predicciones del modelo)
python scripts/evaluate_qa_with_ragas.py \
  --input datasets/obstetrics/qa/final/train/raw.jsonl \
  --output datasets/obstetrics/qa/final/train/eval.json
```

## Reproducibilidad

### Replicar el entrenamiento desde cero

```bash
# 1. Entrenamiento (el dataset se descarga automáticamente desde HF)
python scripts/train_qlora_trl.py \
  --model-name google/gemma-4-E2B-it \
  --dataset-hf-id iue-edu/MaternaQA-es \
  --dataset-variant sft_grounded \
  --output-dir outputs/gemma4-grounded \
  --epochs 2 \
  --lora-r 32 --lora-alpha 64 \
  --lr 2e-4 \
  --warmup-steps 100

# 2. Inferencia sobre el split de test
python scripts/inference_qlora.py \
  --model-path outputs/gemma4-grounded \
  --dataset-hf-id iue-edu/MaternaQA-es \
  --dataset-variant sft_grounded \
  --split test

# 3. Evaluación
python scripts/evaluate_model_predictions.py \
  --input outputs/gemma4-grounded/test_predictions.jsonl
```

### Reanudar entrenamiento desde checkpoint

```bash
# Descarga el checkpoint desde Hugging Face
huggingface-cli download iue-edu/MaternaCare-ES-gemma4-qlora \
  checkpoint-1274/ --local-dir outputs/gemma4-grounded

# Reanuda el entrenamiento (el script detecta el checkpoint automáticamente)
python scripts/train_qlora_trl.py \
  --model-name google/gemma-4-E2B-it \
  --resume-from-checkpoint outputs/gemma4-grounded/checkpoint-1274 \
  --dataset-variant sft_grounded \
  --output-dir outputs/gemma4-grounded
```

## Dataset

El dataset de entrenamiento es **MaternaQA-es** (5.727 pares Q+A en español sobre salud materna):

```python
from datasets import load_dataset

dataset = load_dataset("iue-edu/MaternaQA-es")
# Split: train (5093), validation (306), test (328)
```

| Split | Pares Q+A | Chunks fuente | PDFs fuente |
|:------|----------:|--------------:|------------:|
| Entrenamiento | 5.093 | 1.744 | 52 |
| Validación | 306 | 101 | 2 |
| Test | 328 | 108 | 3 |
| **Total** | **5.727** | **1.953** | **57** |

> [!NOTE]
> El dataset completo, su metodología de construcción y los PDFs fuente están documentados en el repositorio [MaternaQA-es](https://github.com/NicolasHoyosDevss/MaternaQA-es).

## Documentación

| Documento | Descripción |
|-----------|-------------|
| [Notas de investigación](./docs/research_notes/) | Decisiones técnicas, gotchas y seguimiento del proyecto. |
| [Planeación del paper](./papers/README.md) | Posicionamiento, contribuciones y estrategia de escritura. |
| [Dataset MaternaQA-es](https://github.com/NicolasHoyosDevss/MaternaQA-es) | Metodología de construcción del dataset de entrenamiento. |

## Consideraciones Éticas y de Uso

- Los modelos entrenados son recursos de investigación; **no reemplazan criterio clínico** ni guías médicas oficiales.
- Las predicciones generadas deben interpretarse como asistencia para investigación, no como recomendaciones médicas.
- Este repositorio se publica bajo licencia MIT. Para el dataset y los modelos/adapters, revisa la licencia declarada en cada repositorio de Hugging Face.

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Ver [LICENSE](./LICENSE) para más detalles.

---

<div align="center">

**Institución Universitaria de Envigado (IUE)** · **MinCiencias** · Colombia

Fine-tuning de LLMs para QA clínico en español sobre embarazo y maternidad.

[Hugging Face](https://huggingface.co/iue-edu) · [Dataset MaternaQA-es](https://github.com/NicolasHoyosDevss/MaternaQA-es)

</div>
