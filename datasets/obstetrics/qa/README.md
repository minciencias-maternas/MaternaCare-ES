---
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
      num_examples: ~2800
    - name: validation
      num_examples: ~350
    - name: test
      num_examples: ~350
  config_names:
    - publication
    - final
tags:
  - medical-qa
  - obstetrics
  - spanish
  - maternology
  - qlora
  - maternaqa
license: apache-2.0
---

# MaternaCare-ES Dataset

**MaternaCare-ES** es un corpus de preguntas y respuestas clínicas en español
extraído de textos de obstetricia de libre acceso.

## Estructura

- `publication/` — splits en JSONL para publicación (qa_flat_jsonl, sft_closed_book, sft_grounded)
- `final/` — splits train/validation/test con evaluación RAGAS

## Uso con `datasets`

```python
from datasets import load_dataset
dataset = load_dataset("iue-edu/MaternaCare-ES", "final")
```

## Modelos entrenados

- `iue-edu/MaternaCare-ES-gemma4-qlora`
- `iue-edu/MaternaCare-ES-medgemma-qlora`
