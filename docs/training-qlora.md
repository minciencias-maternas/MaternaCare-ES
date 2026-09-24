# Entrenamiento QLoRA de MaternaCare-ES

Este documento explica cómo se entrenaron los adapters QLoRA: la receta de cuantización y LoRA, las variantes de datos, los hiperparámetros, los comandos de entrenamiento y qué produce.

## 1. Qué se entrenó

Se entrenaron dos adapters LoRA con **QLoRA** (cuantización 4-bit + LoRA) sobre modelos Gemma en español clínico:

| Adapter | Modelo base | Dataset |
|---------|-------------|---------|
| `iue-edu/MaternaCare-ES-gemma4-qlora` | `google/gemma-4-E2B-it` (2B) | MaternaQA-es |
| `iue-edu/MaternaCare-ES-medgemma-qlora` | `google/medgemma-1.5-4b-it` (4B) | MaternaQA-es |

La idea de QLoRA: **el modelo base se congela y se cuantiza a 4 bits** (para que quepa en VRAM), y solo se entrenan los **adapters LoRA** de bajo rango (~200 MB). Así se fine-tunean modelos grandes en una GPU de 16 GB.

## 2. Código y dependencias

- Script: `scripts/train_qlora_trl.py`.
- Stack: `transformers` + `peft` + `trl` (`SFTTrainer`/`SFTConfig`) + `bitsandbytes` + `datasets`.
- Requisitos: GPU con **≥16 GB de VRAM**, `HF_TOKEN` con acceso a los modelos gated de Gemma, y aceptar los términos de Gemma en Hugging Face.

## 3. Dataset de entrenamiento

### 3.1 Fuente

**MaternaQA-es** (`iue-edu/MaternaQA-es` en HF, o la copia local en `datasets/obstetrics/qa/publication/`). Total: **5,727 pares QA** (train 5,093 / validation 306 / test 328) derivados de 1,953 chunks de 57 PDFs de GPC/protocolos.

### 3.2 Variantes SFT

El script entrena sobre una de dos variantes (subsets del dataset):

| Variante | Formato del mensaje user |
|----------|--------------------------|
| `sft_closed_book` | Pregunta sola: `Eres un asistente especializado…` + `Pregunta: …` |
| `sft_grounded` | Pregunta con contexto: `Contexto fuente:\n…\n\nPregunta:\n…` |

En ambos casos el formato es un chat `messages` (system/user/assistant). El script convierte cada fila a `prompt`/`completion`:
- `prompt` = todos los mensajes antes del último `assistant`.
- `completion` = el contenido del último mensaje `assistant`.

### 3.3 Carga

- El dataset se puede cargar desde HF (`--dataset-hf-id`, subset = variante) o desde archivos locales (`--dataset-root` + `--dataset-files`). Por defecto busca en `datasets/obstetrics/qa/publication/` las variantes y splits `train`/`validation`/`test`.
- Detalle técnico: el script **elimina la raíz del repo de `sys.path`** antes de importar `datasets`, para que la carpeta local `datasets/` no le haga sombra al paquete de Hugging Face.

## 4. Receta de cuantización y LoRA

### 4.1 Cuantización 4-bit (`build_quantization_config`)

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,   # doble cuantización (ahorra memoria)
    bnb_4bit_quant_type="nf4",        # NormalFloat4
    bnb_4bit_compute_dtype=resolve_dtype(),
)
```

- `resolve_dtype`: **bf16** en GPUs CUDA con compute capability ≥ 8.0, **fp16** en el resto, **float16** sin CUDA.
- El modelo se carga con `device_map="auto"` y `token=HF_TOKEN`.

### 4.2 LoRA (`build_peft_config`)

```python
LoraConfig(
    r=args.lora_r,           # 16 (default) / 32 (entrenamiento final)
    lora_alpha=args.lora_alpha,  # 16 (default) / 64 (entrenamiento final)
    lora_dropout=args.lora_dropout,  # 0.05
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
```

- `target_modules="all-linear"`: se atacan **todas** las capas lineales (flexible entre arquitecturas Gemma/MedGemma, sin listas manuales de nombres).

### 4.3 Clase de modelo

- Si el nombre del modelo contiene `gemma-4` o `medgemma` → `AutoModelForImageTextToText` (modelos multimodales, aunque aquí se usa texto).
- En caso contrario → `AutoModelForCausalLM`.
- Se puede forzar con `--model-class {auto,causal-lm,image-text-to-text}`.

## 5. Configuración de entrenamiento (`SFTConfig`)

Valores clave:

| Parámetro | Valor | Nota |
|-----------|-------|------|
| `num_train_epochs` | 2.0 | |
| `per_device_train_batch_size` | 1 | con acumulación |
| `per_device_eval_batch_size` | 1 | |
| `gradient_accumulation_steps` | 8 | batch efectivo 8 |
| `learning_rate` | 2e-4 | |
| `lr_scheduler_type` | cosine | |
| `warmup_ratio` | 0.03 | |
| `weight_decay` | 0.0 | |
| `max_grad_norm` | 0.3 | |
| `optim` | `adamw_8bit` | |
| `max_length` | 1024 | |
| `packing` | `False` | |
| `gradient_checkpointing` | `True` | `use_reentrant=False` |
| `completion_only_loss` | `True` | solo se computa loss sobre el completion |
| `report_to` | `tensorboard` | |
| `seed` | 3407 | |

- `completion_only_loss=True`: la pérdida se calcula **solo sobre la respuesta** (completion), no sobre el prompt. Esto es lo que hace que el fine-tuning responda bien en vez de solo completar prompts.
- `--assistant-only-loss` está **rechazado** (el script sale con error): requiere un chat template con `{% generation %}`, y los templates actuales no lo soportan.
- Guardado y evaluación por `epoch` (`save_strategy`/`eval_strategy="epoch"`), `logging_steps=10`.

## 6. Comandos

### 6.1 Smoke test (validar que todo corre)

```bash
python scripts/train_qlora_trl.py \
  --model-name google/gemma-4-E2B-it \
  --dataset-variant sft_grounded \
  --output-dir outputs/smoke-gemma4 \
  --max-steps 10 \
  --train-limit 64 \
  --eval-limit 32
```

### 6.2 Entrenamiento final (README)

Los 4 entrenamientos publicados (2 modelos × 2 variantes), con r=32, alpha=64:

```bash
# Gemma 4 grounded
python scripts/train_qlora_trl.py \
  --model-name google/gemma-4-E2B-it \
  --dataset-variant sft_grounded \
  --output-dir outputs/gemma4-grounded \
  --lora-r 32 --lora-alpha 64

# Gemma 4 closed-book
python scripts/train_qlora_trl.py \
  --model-name google/gemma-4-E2B-it \
  --dataset-variant sft_closed_book \
  --output-dir outputs/gemma4-qlora \
  --lora-r 32 --lora-alpha 64

# MedGemma grounded
python scripts/train_qlora_trl.py \
  --model-name google/medgemma-1.5-4b-it \
  --dataset-variant sft_grounded \
  --output-dir outputs/medgemma-grounded \
  --lora-r 32 --lora-alpha 64

# MedGemma closed-book
python scripts/train_qlora_trl.py \
  --model-name google/medgemma-1.5-4b-it \
  --dataset-variant sft_closed_book \
  --output-dir outputs/medgemma-qlora \
  --lora-r 32 --lora-alpha 64
```

### 6.3 Reanudar un entrenamiento interrumpido

```bash
python scripts/train_qlora_trl.py \
  --model-name google/gemma-4-E2B-it \
  --dataset-variant sft_grounded \
  --output-dir outputs/gemma4-grounded \
  --resume-from-checkpoint outputs/gemma4-grounded/checkpoint-1274
```

## 7. Qué produce

Al terminar, `trainer.save_model(output_dir)` + `tokenizer.save_pretrained(output_dir)` guardan:

- `adapter_config.json` + `adapter_model.safetensors` (los pesos LoRA, ~200 MB) — lo mínimo necesario para usar el modelo con PEFT.
- Checkpoints por época (`checkpoint-*`) para reanudar.
- Logs de `tensorboard` en `outputs/<run>/runs/`.

El adapter se publica en Hugging Face (`iue-edu/MaternaCare-ES-*`), y el pipeline RAG lo carga automáticamente (ver [rag.md](./rag.md)).

## 8. Cómo se genera el dataset SFT (contexto)

El dataset QA no se escribió a mano; se construye con la pipeline:

1. `extract_pdfs.py` — extrae el texto de los 57 PDFs, crea el manifest y los `raw_pages.jsonl`.
2. `utils.py` — chunking (500–1200 tokens, overlap 100), dedupe, filtros de calidad, `assign_chunk_ids`, split train/validation.
3. `generate_synthetic_qa.py` — genera pares QA sintéticos con OpenAI por chunk, con filtros de idioma/rol/temas y evaluación de calidad (faithfulness, relevancy, roundtrip). Genera `raw_*.jsonl`, `progress.json` (para reanudar) y un reporte.
4. `build_lm_dataset.py` / `prepare_qa_publication_variants.py` — convierten los QA a las variantes `sft_grounded`/`sft_closed_book` (formato chat `messages`) en `qa/publication/`.

Detalles de formato de datos en [estructura-y-datos.md](./estructura-y-datos.md).