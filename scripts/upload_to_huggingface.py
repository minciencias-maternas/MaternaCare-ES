#!/usr/bin/env python3
"""
Upload MaternaQA-es dataset and MaternaCare-ES model artifacts to Hugging Face.

Usage:
    export HF_TOKEN=$(huggingface-cli whoami --token)
    /path/to/venv/bin/python scripts/upload_to_huggingface.py

What it uploads:
1. Dataset   -> iue-edu/MaternaQA-es
2. Gemma4 QLoRA adapter -> iue-edu/MaternaCare-ES-gemma4-qlora
3. MedGemma QLoRA adapter -> iue-edu/MaternaCare-ES-medgemma-qlora
"""
import json
import os
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_ORG = "iue-edu"
DATASET_REPO_ID = f"{HF_ORG}/MaternaQA-es"

def upload_dataset():
    """Upload the obstetrics QA dataset as a folder-based dataset."""
    dataset_dir = REPO_ROOT / "datasets" / "obstetrics" / "qa"
    readme_path = dataset_dir / "README.md"

    # Ensure dataset card exists
    if not readme_path.exists():
        readme_path.write_text(DATASET_CARD_MD, encoding="utf-8")

    repo_id = DATASET_REPO_ID
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"[1/3] Uploading dataset to {repo_id} …")
    url = upload_folder(
        folder_path=str(dataset_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="feat(dataset): publish MaternaQA-es obstetrics QA corpus",
    )
    print(f"    -> {url}")

def upload_adapter(adapter_dir: Path, repo_id_suffix: str, model_card_md: str):
    """Upload LoRA adapter weights + tokenizer as a model."""
    repo_id = f"{HF_ORG}/{repo_id_suffix}"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Minimal required files for a usable adapter on HF
    required = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
    ]
    missing = [f for f in required if not (adapter_dir / f).exists()]
    if missing:
        print(f"    WARNING: missing {missing} in {adapter_dir}")

    # Write model card
    card_path = adapter_dir / "README.md"
    if not card_path.exists():
        card_path.write_text(model_card_md, encoding="utf-8")

    print(f"[ ] Uploading adapter to {repo_id} …")
    url = upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"feat(model): upload {repo_id_suffix} LoRA adapter",
    )
    print(f"    -> {url}")

DATASET_CARD_MD = """---
language: es
dataset_info:
  features:
    - name: question
      dtype: string
    - name: answer
      dtype: string
    - name: context
      dtype: string
    - name: source
      dtype: string
    - name: chapter
      dtype: string
  splits:
    - name: train
      num_examples: 5093
    - name: validation
      num_examples: 306
    - name: test
      num_examples: 328
  config_names:
    - sft_closed_book
    - sft_grounded
    - qa_flat_jsonl
    - final
tags:
  - medical-qa
  - obstetrics
  - spanish
  - maternology
  - maternal-health
  - question-answering
  - synthetic-data
  - maternaqa
license: apache-2.0
---

# MaternaQA-es Dataset

**MaternaQA-es** is a Spanish synthetic clinical question-answering dataset
focused on pregnancy, maternal health, labor, postpartum care, fetal monitoring,
and related perinatal topics.

The canonical source repository is
[`NicolasHoyosDevss/MaternaQA-es`](https://github.com/NicolasHoyosDevss/MaternaQA-es).
The fine-tuning repository that consumes this dataset is
[`JhonHander/MaternaCare-ES`](https://github.com/JhonHander/MaternaCare-ES).

## Dataset Structure

- `publication/sft_closed_book/` — question to answer supervised fine-tuning format.
- `publication/sft_grounded/` — context plus question to answer supervised fine-tuning format.
- `publication/qa_flat_jsonl/` — flat records with metadata for audit and analysis.
- `final/` — final train/validation/test generation outputs and summary metadata.

## Splits

| Split | QA pairs | Source chunks | Source PDFs |
|:------|---------:|--------------:|------------:|
| train | 5,093 | 1,744 | 52 |
| validation | 306 | 101 | 2 |
| test | 328 | 108 | 3 |
| total | 5,727 | 1,953 | 57 |

## Usage

```python
from datasets import load_dataset
dataset = load_dataset("iue-edu/MaternaQA-es", "sft_grounded")
```

## Related Fine-Tuned Models

- `iue-edu/MaternaCare-ES-gemma4-qlora`
- `iue-edu/MaternaCare-ES-medgemma-qlora`
"""

GEMMA4_CARD_MD = """---
language: es
base_model: google/gemma-4-E2B-it
library_name: peft
tags:
  - medical-qa
  - obstetrics
  - spanish
  - qlora
  - gemma4
license: apache-2.0
---

# MaternaCare-ES Gemma 4 QLoRA

LoRA adapter fine-tuned on **MaternaQA-es** using **RAFT** methodology
with **google/gemma-4-E2B-it** base model.

## Training details

- Method: QLoRA (4-bit NF4, double quant, page optimiser)
- Base model: `google/gemma-4-E2B-it`
- Dataset: `iue-edu/MaternaQA-es`
- Epochs: 2
- Batch size: 1 (gradient accumulation 4)
- Max length: 1024
- LR: 2e-4 / warmup 100 steps

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it")
model = PeftModel.from_pretrained(base, "iue-edu/MaternaCare-ES-gemma4-qlora")
```
"""

MEDGEMMA_CARD_MD = """---
language: es
base_model: google/medgemma-1.5-4b-it
library_name: peft
tags:
  - medical-qa
  - obstetrics
  - spanish
  - qlora
  - medgemma
license: apache-2.0
---

# MaternaCare-ES MedGemma 1.5 4B QLoRA

LoRA adapter fine-tuned on **MaternaQA-es** using **RAFT** methodology
with **google/medgemma-1.5-4b-it** base model.

## Training details

- Method: QLoRA (4-bit NF4, double quant, page optimiser)
- Base model: `google/medgemma-1.5-4b-it`
- Dataset: `iue-edu/MaternaQA-es`
- Epochs: 2
- Batch size: 1 (gradient accumulation 4)
- Max length: 1024
- LR: 2e-4 / warmup 100 steps

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("google/medgemma-1.5-4b-it")
model = PeftModel.from_pretrained(base, "iue-edu/MaternaCare-ES-medgemma-qlora")
```
"""

if __name__ == "__main__":
    api = HfApi()
    print("Authenticated as:", api.whoami()["name"])

    upload_dataset()
    upload_adapter(
        REPO_ROOT / "outputs" / "gemma4-grounded",
        "MaternaCare-ES-gemma4-qlora",
        GEMMA4_CARD_MD,
    )
    upload_adapter(
        REPO_ROOT / "outputs" / "medgemma-grounded",
        "MaternaCare-ES-medgemma-qlora",
        MEDGEMMA_CARD_MD,
    )

    print("\n✅ All uploads complete.")
    print("   Dataset : https://huggingface.co/datasets/iue-edu/MaternaQA-es")
    print("   Gemma4  : https://huggingface.co/iue-edu/MaternaCare-ES-gemma4-qlora")
    print("   MedGemma: https://huggingface.co/iue-edu/MaternaCare-ES-medgemma-qlora")
