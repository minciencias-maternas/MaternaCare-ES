# RAFT Methodology — Design Rationale & Experimental Matrix

> **Decision date**: 2026-07-06
> **Based on**: RAFT (Zhang et al., 2024, UC Berkeley), RA-DIT (Meta, 2023), Self-RAG (2023)
> **Status**: Data prepared, training pending

---

## 1. What is RAFT and why apply it here?

**RAFT (Retrieval Augmented Fine Tuning)** is a training technique that teaches an LLM
to **discriminate relevant evidence from distractors**. During training, the model
receives a question alongside a document set containing:

- **1 golden document** — the source of the correct answer
- **N distractor documents** — unrelated text from other sources

The model learns to identify and cite the golden document while ignoring distractors,
mimicking the conditions of a real-world RAG pipeline where retrieval may return
imperfect results.

### Why it's defendable in a "Fine-Tuning vs RAG" comparison

RAFT is a **training** technique, not a retrieval technique. It does not depend on
any retrieval system at inference time. This makes it orthogonal to the RAG comparison:

- **RAFT answers**: "Can we train a model to be a better evidence consumer?"
- **RAG answers**: "Can we give the model better evidence at inference time?"

These are **independent axes** that can be studied separately and in combination.
A model trained with RAFT can be evaluated:
1. **Without context** (closed-book, testing memorization vs evidence use)
2. **With fixed golden context** (same as grounded SFT)
3. **With RAG retrieval** (testing if RAFT-trained discrimination helps handle retrieval noise)

### Key reference

Zhang et al. (2024) showed that RAFT-style training consistently outperforms both
RAG-only and SFT-only baselines on PubMed QA, HotpotQA, and domain-specific QA tasks.
The hybrid approach (RAFT + RAG at inference) gave the best results.

---

## 2. How RAFT differs from `sft_grounded`

| Aspect | `sft_grounded` | `sft_raft` |
|---|---|---|
| Contexts per example | 1 (always the golden excerpt) | 3 (golden + 2 distractors from other PDFs) |
| Discrimination training | No — model never sees irrelevant context | Yes — must identify the golden context among noise |
| Golden-doc dropout | No | Optional (`sft_raft_dropout`): 20% of training examples have NO golden context → model learns to say "not supported" |
| System prompt | Generic clinical assistant | Instructs the model that only the first fragment (Contexto fuente) is relevant; additional fragments are from other documents and should be ignored |
| Realism for RAG | Low — assumes perfect retrieval | High — mirrors retrieval returning multiple chunks with varying relevance |

### Prompt format comparison

**Before** (`sft_grounded`):
```
System: Eres un asistente especializado en obstetricia...
User: Contexto fuente: <golden excerpt, 250 chars>
      Pregunta: <clinical question>
Assistant: <answer>
```

**After** (`sft_raft`):
```
System: Eres un asistente especializado... Se te proporcionarán varios fragmentos
        de contexto. SOLO el primer fragmento (Contexto fuente) contiene información
        relevante para la pregunta. Los fragmentos adicionales son de otros
        documentos y NO son pertinentes.
User: Contexto fuente: <golden excerpt, ~250 chars>
      Contexto adicional 1: <distractor excerpt from different PDF, ≤350 chars>
      Contexto adicional 2: <distractor excerpt from different PDF, ≤350 chars>
      Pregunta: <clinical question>
Assistant: <answer>
```

### `sft_raft_dropout` variant

20% of training examples have the golden context replaced with:
```
[Contexto no disponible — responde con tu conocimiento interno]
```

This forces the model to sometimes rely on internal knowledge while still learning
to use evidence when available — exactly RAFT's recommended P=0.2 dropout strategy.

---

## 3. Complete Experimental Matrix

### Models (Training × Base Architecture)

| ID | Base Model | Training Variant | Status |
|---|---|---|---|
| **gemma4-base** | google/gemma-4-E2B-it | None (zero-shot) | ✅ Trained, eval partial |
| **gemma4-closed_book** | google/gemma-4-E2B-it | sft_closed_book (question only) | ❌ Not trained |
| **gemma4-grounded** | google/gemma-4-E2B-it | sft_grounded (golden context in prompt) | ✅ Trained, eval partial |
| **gemma4-raft** | google/gemma-4-E2B-it | sft_raft (golden + 2 distractors) | 🔲 Data ready |
| **gemma4-raft-dropout** | google/gemma-4-E2B-it | sft_raft_dropout (20% golden dropout) | 🔲 Data ready |
| **medgemma-base** | google/medgemma-1.5-4b-it | None (zero-shot) | ✅ Trained, eval partial |
| **medgemma-closed_book** | google/medgemma-1.5-4b-it | sft_closed_book (question only) | ❌ Not trained |
| **medgemma-grounded** | google/medgemma-1.5-4b-it | sft_grounded (golden context in prompt) | ✅ Trained, eval partial |
| **medgemma-raft** | google/medgemma-1.5-4b-it | sft_raft (golden + 2 distractors) | 🔲 Data ready |
| **medgemma-raft-dropout** | google/medgemma-1.5-4b-it | sft_raft_dropout (20% golden dropout) | 🔲 Data ready |

### Inference Modes (per model)

Each trained model can be evaluated in 3 inference modes:

| Mode | Context Source | Tests |
|---|---|---|
| **Without context** | None (closed-book) | Memorization detection, catastrophic forgetting |
| **Golden context** | Fixed `contexto_fuente` excerpt | Faithfulness, evidence use (same as training condition for grounded/raft) |
| **RAG retrieval** | Top-k chunks from vector DB | Real-world retrieval quality, distractor handling (especially for RAFT) |

### Research Questions

```
RQ1 (Base vs FT):    Does grounded SFT improve over the base model for
                     clinical QA in obstetric Spanish?

RQ2 (Grounded vs RAFT): Does adding distractor training (RAFT) improve
                        faithfulness over simple grounded SFT?

RQ3 (Domain Model):  Does MedGemma (pre-trained on medical data) outperform
                     Gemma 4 base for obstetric Spanish QA?

RQ4 (Closed vs Open):  Does providing context at inference time improve
                       answer quality vs closed-book generation?

RQ5 (RAFT + Dropout): Does golden-doc dropout reduce hallucination when
                      no relevant context is available?

RQ6 (FT + RAG):      Does a fine-tuned model + RAG outperform a base
                     model + RAG for domain-specific retrieval? Does
                     RAFT's discrimination training improve RAG quality?
```

---

## 4. Evaluation Design

### Layer 1: Intrinsic QA quality (RAGAS LLM-as-judge)

For every (model, inference_mode) combination:

| Metric | What it measures | Judge |
|---|---|---|
| `faithfulness` | Are claims in the answer supported by source context? | GPT-5.4-mini |
| `answer_relevancy` | Does the answer address the question? | GPT-5.4-mini |
| `answer_correctness` | Is the answer factually correct vs. reference? | GPT-5.4-mini |
| `semantic_similarity` | Semantic overlap with reference (embedding-based) | text-embedding-3-small |

### Layer 2: Behavioral analysis

- **Counterfactual test** (inference without context): If RAFT-trained model still produces the same answer without context → memorization detected. If it refuses or is uncertain → it learned evidence grounding.
- **Checkpoint comparison** (637 vs 1274): Quantify overfitting damage by comparing checkpoint performance.
- **Hallucination pattern review**: Qualitative analysis of worst-performing answers.

---

## 5. Implementation Details

### Data preparation

- **Source**: `datasets/obstetrics/qa/publication/qa_flat_jsonl/*.jsonl`
- **Corpus**: `artifacts/obstetrics/corpus/chunks.jsonl` (2,268 chunks, 60 PDFs)
- **Script**: `scripts/prepare_raft_dataset.py`
- **Distractor selection**: 2 chunks from different PDFs, trimmed to 350 chars
- **Golden context**: Reuses the existing `contexto_fuente` excerpt (~250 chars)
- **User message size**: 882–1,424 chars (mean 1,127). Fits safely in `max_length=1024`.
- **Dropout**: Applied only to training split. Validation and test always have full golden context.

### Training configuration

| Parameter | Value |
|---|---|
| Method | QLoRA (4-bit, NF4, double quant) |
| LoRA rank | r=16, alpha=16 |
| Target modules | All linear layers |
| Epochs | 2 |
| Batch size | 1 (GA=8, effective=8) |
| LR | 2e-4, cosine schedule |
| Max length | 1024 tokens |
| Early stopping | patience=3, metric=eval_loss (NEW) |
| Best model | Loaded at end (NEW) |

### Output directories

```
outputs/
  gemma4-raft/             ← sft_raft data, gemma4 base
  gemma4-raft-dropout/     ← sft_raft_dropout data, gemma4 base
  medgemma-raft/           ← sft_raft data, medgemma base
  medgemma-raft-dropout/   ← sft_raft_dropout data, medgemma base
```

No interference with existing models in `outputs/gemma4-grounded/` or `outputs/medgemma-grounded/`.

---

## 6. Paper Argumentation Strategy

### Why the factorial design is a strength

Most papers in the clinical NLP space compare exactly 2 things (e.g., FT vs RAG, or FT vs base model). This design isolates **three independent factors**:

1. **Training method** (closed-book / grounded / RAFT / RAFT+dropout)
2. **Base model** (general Gemma 4 vs medical MedGemma)
3. **Inference context** (none / golden / RAG)

This yields 30 potential (model, inference_mode) pairs, providing rich comparative evidence.

### Anticipated criticisms and responses

| Criticism | Response |
|---|---|
| "RAFT uses context — isn't that cheating vs closed-book?" | RAFT is evaluated in ALL modes, including closed-book. The comparison is fair because each model faces the same evaluation conditions. |
| "Aren't you training on the evaluation domain?" | Train/val/test are split by document (0% document overlap). The test set was held out from both QA generation and training. |
| "Isn't 5K examples too little for fine-tuning?" | QLoRA on a 4B model with 2 epochs and 5K examples is standard for domain adaptation. The early stopping addition prevents overfitting that occurred in the initial runs. |
| "Why RAFT instead of full RAG training (REALM, RAG-end2end)?" | RAFT is lighter-weight and doesn't require joint retriever-generator training. It's a pragmatic increment: improves evidence use without changing the architecture. |

---

## 7. References

1. Zhang, T., Patil, S. G., Jain, N., et al. (2024). *RAFT: Adapting Language Model to Domain Specific RAG*. arXiv:2403.10131.
2. Lin, X. V., Chen, X., Chen, M., et al. (2023). *RA-DIT: Retrieval-Augmented Dual Instruction Tuning*. arXiv:2310.01352.
3. Asai, A., Wu, Z., Wang, Y., et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. arXiv:2310.11511.
4. Reynolds, L. (2025). *Mitigating Catastrophic Forgetting in Mathematical Reasoning Finetuning through Mixed Training*. arXiv:2501.12345.
