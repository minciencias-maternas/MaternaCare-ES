# MaternaCare-ES: Fine-tuning de LLMs para QA en Ginecología y Obstetricia

Repositorio de fine-tuning, evaluación e inferencia de modelos de lenguaje especializados en preguntas y respuestas clínicas de ginecología y obstetricia en español.

Este repositorio es la contraparte de fine-tuning del proyecto **MaternaQA-es** (creación de dataset). El dataset de entrenamiento está publicado en Hugging Face: [`iue-edu/MaternaCare-ES`](https://huggingface.co/datasets/iue-edu/MaternaCare-ES).

## Modelos Publicados

Los adapters QLoRA entrenados están disponibles en HuggingFace:

- **Gemma 4 QLoRA**: [`iue-edu/MaternaCare-ES-gemma4-qlora`](https://huggingface.co/iue-edu/MaternaCare-ES-gemma4-qlora)
- **MedGemma 1.5 4B QLoRA**: [`iue-edu/MaternaCare-ES-medgemma-qlora`](https://huggingface.co/iue-edu/MaternaCare-ES-medgemma-qlora)

## Modelos Base

- **Gemma 4 IT**: `google/gemma-4-E2B-it`
- **MedGemma 1.5 4B IT**: `google/medgemma-1.5-4b-it`

## Estructura del Repositorio

```
scripts/                # Scripts de entrenamiento, inferencia y evaluación
  train_qlora_trl.py           # Entrenamiento QLoRA con TRL
  inference_qlora.py           # Inferencia con modelo QLoRA fine-tuneado
  inference_base.py          # Inferencia con modelo base (baseline)
  evaluate_model_predictions.py   # Evaluación de predicciones
  evaluate_qa_with_ragas.py       # Evaluación con RAGAS
  convert_eval_to_csv.py          # Conversión de evaluaciones a CSV
  backfill_prediction_metadata.py # Metadatos de predicciones
outputs/               # Pesos, checkpoints y evaluaciones generadas
  gemma4-grounded/     # Adapter y evaluaciones de Gemma 4 grounded
  medgemma-grounded/   # Adapter y evaluaciones de MedGemma grounded
  smoke-gemma4/        # Smoke test de Gemma 4
  smoke-medgemma/      # Smoke test de MedGemma
  gemma4-base/         # Evaluaciones del modelo base
  medgemma-base/       # Evaluaciones del modelo base
  logs/                # Logs de ejecución
csv_outputs/           # Resultados agregados en CSV (wide_comparison, master_eval, etc.)
datasets/              # Dataset de entrenamiento (publication y final)
  obstetrics/qa/publication/    # Variant lists publicadas (sft_grounded, sft_closed_book, qa_flat_jsonl)
  obstetrics/qa/final/         # Archivos raw y SFT previos a la publicación
docs/research_notes/   # Notas técnicas del proyecto
papers/                # Documentos y borradores del paper
```

## Dependencias

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

### Entrenamiento (QLoRA)

```bash
python scripts/train_qlora_trl.py \
  --model-name google/gemma-4-E2B-it \
  --dataset-variant sft_grounded \
  --output-dir outputs/gemma4-grounded \
  --epochs 1 \
  --lora-r 32 --lora-alpha 64
```

### Inferencia

```bash
python scripts/inference_qlora.py \
  --model-path outputs/gemma4-grounded \
  --dataset-variant sft_grounded \
  --split test \
  --output-path outputs/gemma4-grounded/test_predictions.jsonl
```

### Evaluación con RAGAS

```bash
python scripts/evaluate_qa_with_ragas.py \
  --input outputs/gemma4-grounded/test_predictions.jsonl \
  --output outputs/gemma4-grounded/test_eval.jsonl
```

## Dataset

El dataset se carga por defecto desde Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("iue-edu/MaternaCare-ES")
```

Para uso offline, las variantes también están disponibles localmente en `datasets/obstetrics/qa/publication/`.

## Métricas de Evaluación

- **ROUGE-1 / ROUGE-L**: Métricas de solapamiento léxico
- **BERTScore F1**: Similitud semántica basada en embeddings
- **Exact Match**: Coincidencia exacta de la respuesta
- **RAGAS**: Evaluación de calidad de generación en QA

## Licencia

Este proyecto es parte de la investigación de la **Institución Universitaria de Envigado (IUE)**.

## Contacto

- Organización Hugging Face: [https://huggingface.co/iue-edu](https://huggingface.co/iue-edu)
- Repositorio original (dataset): [NicolasHoyosDevss/MaternaQA-es](https://github.com/NicolasHoyosDevss/MaternaQA-es)
