# Quality, Validation, Ethics, and Limitations

This file documents how to argue quality responsibly in the first MaternaQA-es paper.

## 1. Quality Claims We Can Make

The paper can reasonably claim:

- the dataset is source-traceable;
- the dataset has document-level split controls;
- QA pairs were generated from curated maternal-health documents;
- QA variants support multiple research uses;
- automated quality estimates were computed on stratified samples;
- grounding overlap and RAGAS metrics provide useful but incomplete quality signals;
- the dataset is suitable for research and experimentation.

## 2. Quality Claims We Should Avoid

Avoid claiming:

- the dataset is clinically validated;
- the dataset is safe for medical decision-making;
- all answers are clinically correct;
- models trained on the dataset are safe for patient care;
- RAGAS scores prove medical correctness;
- synthetic generation is equivalent to expert authoring.

## 3. Quality Assessment Components

| Component | What it checks | Why it matters | Limitation |
|---|---|---|---|
| Source traceability | Each QA pair links to source context/chunk/PDF. | Enables audit and grounded evaluation. | Does not guarantee correctness if source extraction is flawed. |
| Document-level splits | Prevents same PDF from appearing in multiple splits. | Reduces leakage. | Does not eliminate all semantic overlap across different documents. |
| Deduplication | Removes exact/near duplicate chunks. | Improves corpus diversity. | Near-duplicate detection may miss paraphrases. |
| Grounding overlap | Measures lexical overlap between answer and context. | Simple signal of context dependence. | Low overlap can still be valid; high overlap can be superficial. |
| RAGAS faithfulness | Estimates whether answer is supported by context. | Useful source-consistency metric. | LLM-as-judge is imperfect. |
| RAGAS answer relevancy | Estimates whether answer responds to question. | Useful QA quality metric. | Does not ensure clinical correctness. |
| POC generator comparison | Selects stronger generation configuration. | Demonstrates design choice rationale. | Small sample; not final clinical validation. |

## 4. Current Quality Numbers

### RAGAS evaluation

| Split | Sample evaluated | Faithfulness | Answer relevancy |
|---|---:|---:|---:|
| Train | 300 / 5,093 | 0.7726 | 0.6466 |
| Validation | 100 / 306 | 0.7826 | 0.6812 |
| Test | 100 / 328 | 0.7132 | 0.5583 |

### POC generator/evaluator comparison

| Experiment | Generator | Evaluator | Faithfulness | Relevancy | Acceptance |
|---|---|---|---:|---:|---:|
| A | GPT-5.4 | GPT-5.5 | 0.9876 | 0.9829 | 100% |
| B | GPT-5.4-mini | GPT-5.5 | 0.9382 | 0.9447 | 94.1% |
| C | GPT-5.2 | GPT-5.5 | 0.9924 | 0.9947 | 100% |
| D | GPT-5.4 | GPT-5.4-mini | 0.9235 | 0.9359 | 94.1% |

Important explanation:

> Although the POC indicated strong generation quality under a generator/evaluator configuration, the final full dataset generation disabled intermediate evaluator scoring for cost efficiency. Therefore, final reported quality should rely on post-hoc sampling, grounding analysis, and explicit limitations rather than implying every record received LLM-judge validation during generation.

## 5. Recommended Validation Subsections

```text
8. Quality Assessment
  8.1 Source traceability and split integrity
  8.2 Grounding analysis
  8.3 RAGAS-based faithfulness and relevancy estimation
  8.4 Generator configuration pilot study
  8.5 Error analysis and limitations of automated validation
```

## 6. Error Analysis Categories

When reviewing examples manually, classify errors as:

| Error category | Description |
|---|---|
| Unsupported answer | Answer includes information not present in context. |
| Overgeneralization | Answer turns context-specific statement into broad medical rule. |
| Ambiguous question | Question lacks enough information to answer safely. |
| Extraction artifact | Source chunk contains broken text, table fragments, headers/footers. |
| Redundant QA | Question duplicates another example. |
| Weak clinical relevance | QA is technically answerable but not useful. |
| Terminology issue | Incorrect or inconsistent medical terminology. |
| Risky recommendation | Answer sounds like direct clinical instruction without sufficient context. |

## 7. Ethics and Responsible Use

### Main ethical position

MaternaQA-es is a research dataset, not a clinical device.

### Include in paper

- The dataset is synthetic but derived from medical documents.
- The dataset may inherit limitations, biases, or outdated recommendations from source documents.
- The dataset may include generated simplifications or paraphrases that require expert validation before clinical use.
- The dataset should not be used for patient-facing medical advice without independent validation, safety testing, and clinical oversight.
- Redistribution rights of source documents and derived text should be checked and documented.

## 8. Intended Uses

| Use | Appropriate? | Notes |
|---|---|---|
| Research in Spanish clinical NLP | Yes | Primary intended use. |
| Supervised fine-tuning experiments | Yes | Especially in Paper 2. |
| RAG evaluation | Yes | Use grounded variant and source metadata. |
| Prompting/evaluation studies | Yes | Useful for maternal-health QA behavior. |
| Educational prototypes | Cautiously | Must include disclaimers. |
| Direct medical advice | No | Not clinically validated. |
| Diagnosis or treatment decisions | No | Requires clinician oversight and independent validation. |

## 9. Limitations to Acknowledge

### Data limitations

- Source corpus may not cover all maternal-health topics uniformly.
- Some documents may reflect local guidelines or time-specific practices.
- OCR-needed pages were not recovered in the current pipeline.
- Tables and figures may be underrepresented due to PDF text extraction constraints.

### Generation limitations

- QA pairs are synthetic.
- LLMs may paraphrase, simplify, or overgeneralize.
- Prompt constraints reduce but do not eliminate hallucination risk.

### Evaluation limitations

- RAGAS is automated and judge-model dependent.
- Faithfulness and relevancy are not equivalent to clinical correctness.
- Stratified samples estimate quality but do not verify every record.
- Human expert review remains necessary for high-stakes use.

### Release limitations

- Licensing and redistribution constraints must be verified.
- Dataset users need clear citation and use restrictions.
- Maintenance/versioning must be documented.

## 10. Future Work

Strong future work items:

1. Clinician-led validation of a representative sample.
2. Full error taxonomy with expert adjudication.
3. OCR integration for excluded pages.
4. Expansion to additional maternal-health topics and countries.
5. Comparison between human-authored and LLM-generated QA.
6. Fine-tuning benchmark study using open models and QLoRA.
7. RAG benchmark using source documents as retrieval corpus.
8. Dataset card and datasheet publication with versioned releases.
