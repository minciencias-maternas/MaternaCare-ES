# Variantes de publicación del QA de obstetricia

Archivos JSONL limpios generados a partir de `datasets/obstetrics/qa/final/` para fine-tuning supervisado y publicación en Hugging Face Hub.

## Variantes

- `sft_closed_book/{train,validation,test}.jsonl`: formato conversacional SFT. El mensaje del usuario contiene únicamente la pregunta.
- `sft_grounded/{train,validation,test}.jsonl`: formato conversacional SFT. El mensaje del usuario contiene `Contexto fuente` más la pregunta.
- `qa_flat_jsonl/{train,validation,test}.jsonl`: registros planos de auditoría/evaluación con pregunta, respuesta, contexto fuente, procedencia y metadata.
- `qa_flat_jsonl/all.jsonl`: consolidado completo de los tres splits para exploración o publicación simple.

En todas las variantes se eliminan los campos vacíos de métricas de calidad: `faithfulness`, `answer_relevancy`, `roundtrip_consistency`, `quality_verdict` y `quality_reason`.

## Uso de los splits

- `train`: entrenamiento del modelo.
- `validation`: evaluación durante entrenamiento, selección de hiperparámetros y early stopping.
- `test`: evaluación final retenida; no debe usarse para decidir la configuración del entrenamiento.

## Conteos

| Split | Pares | Chunks únicos | PDFs fuente únicos |
|---|---:|---:|---:|
| train | 5093 | 1744 | 52 |
| validation | 306 | 101 | 2 |
| test | 328 | 108 | 3 |

## Recomendación de entrenamiento

Usa `sft_closed_book` para medir adaptación al dominio sin contexto explícito.
Usa `sft_grounded` para entrenar y evaluar comportamiento guiado por evidencia, donde el modelo recibe el contexto fuente en tiempo de inferencia.

## Cómo se usan las variantes durante el fine-tuning

`sft_closed_book` y `sft_grounded` son variantes del dataset, no carpetas ligadas a una librería específica.
Cada variante puede entrenarse tanto con la ruta canónica de Hugging Face como con la ruta optimizada de Unsloth.

Matriz recomendada de experimentos:

| Familia de modelo | Variante de dataset | Ruta de entrenamiento | Propósito |
|---|---|---|---|
| Gemma 4 instruct | `sft_closed_book` | TRL + PEFT + bitsandbytes (QLoRA) | Medir adaptación al dominio sin contexto. |
| Gemma 4 instruct | `sft_grounded` | TRL + PEFT + bitsandbytes (QLoRA) | Medir QA guiada por evidencia. |
| MedGemma 1.5 4B IT | `sft_closed_book` | TRL + PEFT + bitsandbytes (QLoRA) | Medir adaptación sobre un modelo médico instruct sin contexto. |
| MedGemma 1.5 4B IT | `sft_grounded` | TRL + PEFT + bitsandbytes (QLoRA) | Medir QA médica guiada por evidencia. |
| Gemma 4 instruct | `sft_closed_book` / `sft_grounded` | Unsloth + TRL (QLoRA) | Repetir los mismos experimentos en versión optimizada si Unsloth es estable en la workstation. |
| MedGemma 1.5 4B IT | `sft_closed_book` / `sft_grounded` | Unsloth + TRL (QLoRA) | Ejecutar solo después de validar con smoke test que el checkpoint de Unsloth carga, entrena y guarda bien. |

Qué define la variante del dataset:

- `sft_closed_book`: el modelo recibe solo la pregunta clínica y debe responder con sus parámetros adaptados.
- `sft_grounded`: el modelo recibe `Contexto fuente` más la pregunta y debe responder usando la evidencia provista.

Qué define la ruta de entrenamiento:

- `TRL + PEFT + bitsandbytes`: ruta canónica y reproducible de Hugging Face para QLoRA.
- `Unsloth + TRL`: ruta optimizada donde Unsloth carga/parchea el modelo y prepara LoRA de forma eficiente, mientras TRL sigue ejecutando el ciclo de SFT.

En ambas rutas, `train.jsonl` se pasa como `train_dataset`, `validation.jsonl` como `eval_dataset` y `test.jsonl` se mantiene reservado para evaluación final.

## Notas de compatibilidad de modelos

- Gemma 4: Unsloth documenta soporte explícito para fine-tuning de Gemma 4 E2B, E4B, 26B A4B y 31B. En una workstation con 16GB VRAM, el candidato preferido es Gemma 4 E4B QLoRA si entra; si no, E2B.
- MedGemma 1.5 4B IT: la ruta segura y canónica es QLoRA con TRL/PEFT/bitsandbytes. Unsloth tiene artefactos de MedGemma 1.5 en Hugging Face, pero el proyecto debe validar con smoke test local antes de depender de Unsloth para ese modelo.
- Full fine-tuning no es la ruta objetivo para esta workstation; QLoRA es el método por defecto.

## Script de entrenamiento TRL listo

El repositorio incluye `scripts/train_qlora_trl.py` para ejecutar la ruta canónica `TRL + PEFT + bitsandbytes (QLoRA)`.
El script no instala dependencias ni hace login en Hugging Face; antes de entrenar, instala `requirements.txt`, ejecuta `huggingface-cli login` y acepta los términos de acceso de los modelos de Google/MedGemma.

Los JSONL publicados mantienen el formato conversacional `messages` para ser fáciles de auditar y reutilizar. Durante el entrenamiento, el script los transforma en memoria a formato `prompt`/`completion`; TRL calcula la pérdida solo sobre `completion`, sin depender de máscaras `assistant` del chat template del tokenizer.

> **Flujo recomendado**: primero un smoke test para validar que carga, entrena y guarda. Después el fine-tuning real sobre el dataset completo.

---

### Smoke test (validación rápida)

Entrena 10 pasos con pocos ejemplos. Sirve para verificar que la configuración funciona.

```bash
# Gemma 4 E2B
python scripts/train_qlora_trl.py \
  --model-name google/gemma-4-E2B-it \
  --dataset-variant sft_grounded \
  --output-dir outputs/smoke-gemma4-e2b-grounded \
  --max-steps 10 \
  --train-limit 64 \
  --eval-limit 32

# MedGemma 1.5 4B IT
python scripts/train_qlora_trl.py \
  --model-name google/medgemma-1.5-4b-it \
  --dataset-variant sft_grounded \
  --output-dir outputs/smoke-medgemma-grounded \
  --max-steps 10 \
  --train-limit 64 \
  --eval-limit 32
```

Si MedGemma falla con error de arquitectura, agregá `--model-class causal-lm`.

---

### Fine-tuning real (dataset completo)

Sin flags limitantes, entrena con los defaults del script (2 épocas, batch 8, QLoRA). El `packing` está desactivado por defecto para evitar contaminación entre muestras en modelos cuya atención no soporte empaquetado seguro.

```bash
# Gemma 4 — grounded
python scripts/train_qlora_trl.py \
  --model-name google/gemma-4-E2B-it \
  --dataset-variant sft_grounded \
  --output-dir outputs/gemma4-e2b-grounded

# Gemma 4 — closed-book
python scripts/train_qlora_trl.py \
  --model-name google/gemma-4-E2B-it \
  --dataset-variant sft_closed_book \
  --output-dir outputs/gemma4-e2b-closed-book

# MedGemma 1.5 — grounded
python scripts/train_qlora_trl.py \
  --model-name google/medgemma-1.5-4b-it \
  --dataset-variant sft_grounded \
  --output-dir outputs/medgemma-grounded

# MedGemma 1.5 — closed-book
python scripts/train_qlora_trl.py \
  --model-name google/medgemma-1.5-4b-it \
  --dataset-variant sft_closed_book \
  --output-dir outputs/medgemma-closed-book
```

## Carga mínima

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train": "datasets/obstetrics/qa/publication/sft_grounded/train.jsonl",
        "validation": "datasets/obstetrics/qa/publication/sft_grounded/validation.jsonl",
        "test": "datasets/obstetrics/qa/publication/sft_grounded/test.jsonl",
    },
)

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
# dataset["test"] queda reservado para evaluación final.
```
