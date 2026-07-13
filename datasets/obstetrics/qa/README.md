---
language: es
license: apache-2.0
tags:
  - medical-qa
  - obstetrics
  - spanish
  - maternology
  - maternal-health
  - question-answering
  - synthetic-data
  - maternaqa
dataset_info:
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
---

# MaternaQA-es Dataset

**MaternaQA-es** is a Spanish synthetic clinical question-answering dataset
focused on pregnancy, maternal health, labor, postpartum care, fetal monitoring,
and related perinatal topics.

This local copy is kept in the **MaternaCare-ES** fine-tuning repository only for
offline smoke tests and reproducibility. The dataset's canonical repository is
[`NicolasHoyosDevss/MaternaQA-es`](https://github.com/NicolasHoyosDevss/MaternaQA-es)
and the canonical Hugging Face dataset is
[`iue-edu/MaternaQA-es`](https://huggingface.co/datasets/iue-edu/MaternaQA-es).

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

- [`iue-edu/MaternaCare-ES-gemma4-qlora`](https://huggingface.co/iue-edu/MaternaCare-ES-gemma4-qlora)
- [`iue-edu/MaternaCare-ES-medgemma-qlora`](https://huggingface.co/iue-edu/MaternaCare-ES-medgemma-qlora)

## Responsible Use

MaternaQA-es is a research dataset. It is not a medical device, and models
trained on it must not be used for diagnosis, treatment, or clinical decisions
without independent expert validation.
