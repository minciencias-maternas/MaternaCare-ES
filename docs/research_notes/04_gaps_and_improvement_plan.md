# Gaps & Improvement Plan — MaternaQA-es Grounded Fine-Tuning

> Referenced against: RAFT (Zhang et al., 2024), RA-DIT (Meta, 2023), Self-RAG (2023),
> Over-Memorization in FT (2025), Mixed Training (Reynolds, 2025), MedHallu (2025).

## Current Architecture

```
┌─────────────────────────────────────────────────────────┐
│  PDF Corpus (12 PDFs, obstetrics)                        │
│  ↓ GPT-5.2 generates 500 QA pairs (train/val/test)      │
│  ↓ prepare_qa_publication_variants.py                     │
│  ├── sft_grounded/  → user msg = "Contexto fuente" + q   │
│  └── sft_closed_book/ → user msg = question only          │
│  ↓ QLoRA fine-tuning (TRL + PEFT + bitsandbytes)          │
│  ├── gemma4-grounded   (google/gemma-4-E2B-it + adapter) │
│  └── medgemma-grounded (google/medgemma-1.5-4b-it + adp) │
│  ↓ Inference on test split (328 questions)                │
│  ├── gemma4-base       (no adapter, grounded prompt)     │
│  ├── gemma4-grounded   (QLoRA adapter, grounded prompt)  │
│  ├── medgemma-base     (no adapter, grounded prompt)     │
│  └── medgemma-grounded (QLoRA adapter, grounded prompt)  │
│  ↓ RAGAS evaluation (GPT-5.4-mini judge)                  │
│  ├── faithfulness, answer_relevancy                       │
│  ├── answer_correctness (vs reference)                    │
│  └── semantic_similarity (text-embedding-3-small)        │
└─────────────────────────────────────────────────────────┘
```

---

## GAP 1: Overfitting Confirmed (CRITICAL)

### Evidence

Both models show classic overfitting pattern. Training loss decreases monotonically while validation loss *increases* between the two evaluation checkpoints:

| Model | Step 637 val_loss | Step 1274 val_loss | Delta | Train loss @1274 |
|---|---|---|---|---|
| gemma4-grounded | 0.773 | 0.777 | +0.004 ↑ | 0.569 |
| medgemma-grounded | 0.799 | 0.817 | +0.018 ↑ | 0.576 |

The "Over-Memorization in Finetuning LLMs" paper (2025) identifies this exact pattern: validation perplexity rises before task metrics drop — it's an **early warning signal**. The model at checkpoint 1274 has likely already started memorizing training examples while generalizing worse than checkpoint 637.

### Root Cause

- Only 2 validation checkpoints (step 637 and 1274) — insufficient granularity
- Training continued despite clear upward trend in val loss
- No early stopping configured
- Small dataset (300 train pairs) + many training steps (1274) = high memorization risk

### Immediate Fix (no retraining)

1. **Use checkpoint 637 for evaluation**, not 1274. Compare both checkpoints on the same test set to quantify the degradation.
2. Re-run inference from checkpoint 637 for both models.
3. **Plot train/val loss curves** → this is publishable evidence of overfitting dynamics.

### Long-Term Fix (requires retraining)

1. Configure early stopping with patience=3
2. Add more frequent validation checkpoints (every 100 steps)
3. Increase dataset size or reduce training steps
4. Apply data augmentation to training set

---

## GAP 2: No Distractor Documents in Training (HIGH)

### What's Missing

RAFT's core contribution: during training, each example includes the **golden document** (correct context) PLUS 1–3 **distractor documents** (irrelevant chunks from other PDFs). The model learns to:
- Discriminate relevant vs. irrelevant context
- NOT blindly trust any retrieved document
- Cite specific evidence passages

### Current State

MaternaQA-es includes ONLY the golden context in the prompt. The model never sees irrelevant documents during training.

### Impact

Model becomes **brittle to retrieval quality**. If a retriever returns a noisy or incorrect document at inference time, the model has no training on how to handle it and may hallucinate from irrelevant text.

### Fix (requires retraining)

For each training example, add 1–3 random chunks from other PDFs as distractors. Format:

```
Contexto fuente 1 (relevante):
[golden chunk]

Contexto fuente 2 (distractor):
[random chunk from other PDF]

Contexto fuente 3 (distractor):
[another random chunk]

Pregunta:
[question]
```

Train with Chain-of-Thought that explicitly selects the correct context:
```
Razonamiento: El contexto fuente 1 contiene la evidencia relevante sobre [topic].
Los contextos 2 y 3 tratan sobre [other topics] y no son pertinentes.

Respuesta: [answer based on context 1]
```

---

## GAP 3: No Chain-of-Thought with Citations (HIGH)

### What's Missing

RAFT and Self-RAG both generate reasoning steps WITH verbatim citations from the source document as part of the answer. This:
- Improves answer factuality
- Makes verification easier
- Reduces hallucination by forcing explicit evidence anchoring

### Current State

Training examples use simple Q→A format. The model generates answers directly without showing its reasoning or citing specific passages.

### Fix (requires retraining)

Reformat training answers to include CoT:

```
Respuesta:
Razonamiento: El estudio Dowswell 2015 (ECA, n=51,504) comparó
modelos de 4 vs 8+ controles prenatales. Reporta RR 1.13 (IC 0.50-2.57)
para mortalidad materna, lo que indica ausencia de diferencia significativa.

Conclusión: Un esquema de al menos 4 controles prenatales
podría tener poco o ningún impacto sobre la mortalidad materna
en comparación con 8 o más controles (RR 1.13, IC 95%: 0.50-2.57,
certeza baja).
```

---

## GAP 4: No Golden-Document Dropout (MEDIUM)

### What's Missing

RAFT randomly removes the golden document in P% of training examples (recommended P≈0.2). This forces the model to:
- Sometimes answer from memory (internalized knowledge)
- Sometimes answer from provided evidence
- Learn when to trust context vs. internal knowledge

### Current State

100% of training examples include the golden context. The model never practices answering without it.

### Impact

Model may over-rely on context (extractive behavior) and fail when context is missing or insufficient. Conversely, without dropout the model may learn to simply copy from context without understanding.

### Fix (requires retraining)

Randomly omit context in ~20% of training examples. The model should learn to either:
- Answer from internal knowledge when context is absent
- Say "No hay evidencia suficiente" when neither context nor internal knowledge suffices

---

## GAP 5: No Mixed General Data (MEDIUM)

### What's Missing

Reynolds (2025) shows that even **6.2% general-task data** mixed with domain data completely prevents catastrophic forgetting without hurting domain performance.

### Current State

Training uses 100% obstetrics QA data. No general Spanish instruction data mixed in.

### Impact

Risk of catastrophic forgetting: general capabilities (reasoning, Spanish fluency, instruction following) may degrade. Not yet measured because no pre/post general benchmarks were run.

### Fix (requires retraining)

Mix 5-10% general Spanish instruction data (e.g., Alpaca-es, Ultrachat-es) into the training set.

---

## GAP 6: No Hallucination Evaluation (MEDIUM)

### What's Missing

Standard hallucination benchmarks like MedHallu (Pandit et al., 2025 EMNLP), RAGTruth, or MedHallBench should be run on the fine-tuned model.

### Current State

Only RAGAS metrics (faithfulness, relevancy, correctness, similarity) are used. These measure quality but don't specifically test hallucination scenarios:
- Does the model fabricate statistics?
- Does it generate plausible-sounding but incorrect medical claims?
- Does it say "I don't know" when context is insufficient?

### Immediate Fix (no retraining)

1. Sample predictions where faithfulness < 0.3 and manually review for hallucination patterns
2. Run the MedGemma-grounded JSONL (which shows visible repetition issues) through hallucination classification
3. Create a "trick test": feed context with deliberately wrong information and check if model corrects it or repeats it

### Long-Term Fix (requires retraining)

Integrate hallucination-resistant training: include examples where context is deliberately wrong and the correct answer is "not supported."

---

## GAP 7: Incomplete Model Evaluations (HIGH - ACTIONABLE NOW)

### Current State

| Model | Predictions | Eval rows | Summary JSON | Status |
|---|---|---|---|---|
| gemma4-base | 328 | 328 | ✅ | COMPLETE |
| gemma4-grounded | 328 | 81 | ❌ | 25% done |
| medgemma-grounded | 328 | 113 | ❌ | 34% done |
| medgemma-base | 328 | 0 | ❌ | NOT STARTED |

The evaluation script (`evaluate_model_predictions.py`) is resumable — it writes each row to JSONL as it completes and skips existing rows on restart.

### Fix (no retraining)

Run the evaluation script with `--resume` on all three incomplete models:

```bash
python scripts/evaluate_model_predictions.py \
    --input outputs/gemma4-grounded/test_predictions.jsonl \
    --resume

python scripts/evaluate_model_predictions.py \
    --input outputs/medgemma-grounded/test_predictions.jsonl \
    --resume

python scripts/evaluate_model_predictions.py \
    --input outputs/medgemma-base/test_predictions.jsonl \
    --resume
```

This requires OpenAI API credits (GPT-5.4-mini judge + text-embedding-3-small embeddings).

---

## GAP 8: No Counterfactual / Memorization Test (HIGH - ACTIONABLE NOW)

### What's Missing

The definitive test for whether a model learned to use evidence vs. memorized QA pairs:

1. Run inference WITH context → measure performance
2. Run inference WITHOUT context (same questions) → measure performance
3. If performance is similar, the model **memorized**. If it drops significantly, it **learned to use evidence**.

### Current State

All 328 predictions for all models use grounded prompts (context included). No counterfactual baseline exists.

### Fix (no retraining)

Create a modified test set where `Contexto fuente` is either:
- Removed entirely (closed-book test)
- Replaced with irrelevant context from another PDF (distractor test)
- Replaced with deliberately wrong context (adversarial test)

Run inference with these variants and compare faithfulness/relevancy scores.

---

## GAP 9: No Closed-Book Variant Evaluation (MEDIUM - ACTIONABLE NOW)

### What's Missing

The dataset preparation script creates `sft_closed_book/` variants, but there's no evidence these were used for training or evaluation. The closed-book variant is the natural ablation baseline — without it, we can't isolate the effect of context.

### Fix (requires inference)

If sft_closed_book models were trained, run inference and evaluation on them. If not, at minimum run gemma4-base and medgemma-base with closed-book prompts (no context) to establish a baseline without evidence.

---

## GAP 10: No Checkpoint Comparison (MEDIUM - ACTIONABLE NOW)

### What's Missing

Both models have checkpoints at step 637 and 1274. The training logs show step 637 has LOWER validation loss. But we don't know which checkpoint produces better RAGAS scores on the test set.

### Fix (no retraining required, but needs inference)

Run inference from checkpoint 637 for both models and compare RAGAS scores against checkpoint 1274. This directly tests whether the overfitting detected in val_loss translates to degraded task performance.

---

## What Can Be Done NOW (No Retraining)

| Priority | Action | Effort | Value |
|---|---|---|---|
| 🔴 P0 | Complete RAGAS evals for 3 missing models | ~2-4h API time | Enables full comparison |
| 🔴 P0 | Plot train/val loss curves + document overfitting | 30 min | Publishable finding |
| 🟡 P1 | Counterfactual test (no context inference) | ~1h inference | Proves memorization vs. evidence-use |
| 🟡 P1 | Compare checkpoint 637 vs 1274 | ~1h inference + eval | Quantifies overfitting damage |
| 🟡 P1 | Manual hallucination review on low-faithfulness preds | 1-2h human review | Qualitative analysis for paper |
| 🟢 P2 | Closed-book baseline inference | ~1h inference | Completes ablation study |
| 🟢 P2 | Generate comparison tables/charts | 1h | Paper-ready figures |

## What Requires Retraining

| Priority | Action | Effort | Value |
|---|---|---|---|
| 🔴 P0 | Early stopping + more val checkpoints | Config change | Prevents future overfitting |
| 🟡 P1 | Distractor documents + CoT training | Data prep + 1 train run | RAFT alignment |
| 🟡 P1 | Golden-doc dropout (P=0.2) | Config change + retrain | Improves robustness |
| 🟢 P2 | Mixed general Spanish data (5-10%) | Data curation + retrain | Prevents catastrophic forgetting |
| 🟢 P2 | Hallucination-resistant training examples | Data prep + retrain | Reduces fabrication |

---

## Recommended Execution Order

### Phase 1: Complete What's Already Started (today)
1. Resume RAGAS evals on gemma4-grounded, medgemma-grounded, medgemma-base
2. Document overfitting from trainer_state.json

### Phase 2: No-Retrain Experiments (this week)
3. Counterfactual test (context vs. no-context)
4. Checkpoint 637 vs 1274 comparison
5. Manual hallucination audit
6. Generate all comparison tables + charts

### Phase 3: Retrain with Improvements (next iteration)
7. Configure early stopping
8. Add distractor documents to training data
9. Add CoT with citations
10. Retrain + re-evaluate
