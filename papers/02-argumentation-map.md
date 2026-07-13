# Argumentation Map

This file maps the main claims of the MaternaQA-es dataset paper to evidence, project artifacts, and supporting literature.

## 1. Core Argument

The paper should defend the following argument:

> MaternaQA-es is a scientifically useful dataset because it transforms curated Spanish maternal-health documents into traceable QA examples using an explicit construction methodology, validates their quality through source-aware metrics and dataset analysis, and releases variants that support multiple downstream NLP scenarios.

## 2. Claim-Evidence Map

| Claim to defend | Evidence from project | Papers / frameworks to cite | Where in paper |
|---|---|---|---|
| Spanish maternal-health QA is under-resourced. | Dataset focuses on pregnancy/maternal-health Spanish documents; existing resources are mostly English or broad biomedical. | PubMedQA, MedQuAD, MedMCQA, BioASQ, RealMedQA, Data Statements for NLP. | Introduction, Related Work. |
| Dataset papers should document motivation, composition, collection, preprocessing, use, distribution, and maintenance. | `papers/`, README, dataset schema, Hugging Face publication plan. | Datasheets for Datasets; Data Statements for NLP; Data Cards. | Dataset Documentation, Ethics, Availability. |
| Synthetic QA generation is a valid strategy when paired with filtering, traceability, and quality assessment. | GPT-based QA generation from accepted chunks; structured outputs; grounding fields; RAGAS evaluation. | Self-Instruct, Alpaca, WizardLM/Evol-Instruct, PubMedQA artificial subset, RealMedQA LLM generation. | Methodology, Related Work, Discussion. |
| Quality and curation matter, not only dataset size. | 5,727 QA pairs, source chunk traceability, document-level splits, filtered pages, deduplication, RAGAS samples. | LIMA, Datasheets for Datasets, The Pile, Dolma. | Introduction, Discussion. |
| Grounded QA is important in clinical domains because answers should be evidence-based. | `sft_grounded` variant includes context + question -> answer; `qa_flat_jsonl` retains source context metadata. | RAG; PubMedQA; Natural Questions; RealMedQA. | Dataset Variants, Intended Uses. |
| Closed-book and grounded variants support different research questions. | `sft_closed_book` tests parametric adaptation; `sft_grounded` tests evidence-conditioned answering. | FLAN, instruction tuning literature, RAG. | Dataset Design, Dataset Variants. |
| Document-level splitting reduces leakage. | Splits keep all chunks from a PDF in only one split; zero leakage reported. | Natural Questions/SQuAD methodology as QA precedent; contamination literature; dataset design best practices. | Methodology, Quality Assessment. |
| Automated metrics are useful but not sufficient in clinical QA. | RAGAS faithfulness/relevancy; LLM-as-judge; explicit limitation that clinical human validation is needed. | Med-PaLM / MultiMedQA; RealMedQA verification; clinical NLP evaluation literature. | Quality Assessment, Limitations. |
| The dataset should not be used as a direct clinical decision tool. | Synthetic data; no exhaustive clinician validation; source-document limitations. | Datasheets for Datasets, Data Cards, Med-PaLM safety framing. | Ethics, Limitations, Intended/Non-intended Uses. |

## 3. Narrative Flow

### Step 1: Establish the need

Biomedical QA has strong precedents, but many resources are English-centric, broad biomedical, or designed for exam/research-abstract settings. Pregnancy and maternal health require domain-sensitive language, clinically careful phrasing, and evidence-grounded answers.

### Step 2: Establish the dataset contribution

MaternaQA-es contributes a Spanish QA dataset focused on pregnancy and maternal health, derived from curated medical documents rather than uncontrolled web text.

### Step 3: Explain why synthetic generation is acceptable

Synthetic QA is not a weakness if it is documented and evaluated. Prior work has used artificial or LLM-generated subsets, including PubMedQA's artificial subset and RealMedQA's LLM-generated questions. Self-Instruct, Alpaca, and WizardLM/Evol-Instruct support the broader instruction-data generation paradigm.

### Step 4: Explain why curation matters

The dataset is not valuable only because of its size. It is valuable because examples are traceable, source-grounded, split at the document level, and documented. LIMA helps argue that high-quality curated examples can matter substantially in instruction tuning and alignment.

### Step 5: Explain downstream value without making it the central paper

The dataset supports downstream use in fine-tuning and RAG evaluation. However, the present paper focuses on dataset construction and documentation. Fine-tuning results belong to a second paper.

## 4. Reviewer Concerns and Responses

| Likely reviewer concern | Response strategy |
|---|---|
| "This is just synthetic data." | Cite Self-Instruct, Alpaca, WizardLM, PubMedQA artificial subset, and RealMedQA. Emphasize source grounding, traceability, filtering, and quality assessment. |
| "Where is the clinical validation?" | Be honest: automated validation is not equivalent to clinician review. Position the dataset as a research resource and propose clinical expert review as future work. Include safeguards and non-intended uses. |
| "Why not simply use MedQuAD/PubMedQA?" | They are not pregnancy/maternal-health Spanish-specific and may differ in source type, language, task framing, and intended use. |
| "Why use LLMs to generate QA?" | LLM generation enables scalable QA creation from domain documents, but it is constrained by structured prompts, source context, grounding checks, and evaluation. |
| "Is this a pipeline paper?" | No. The pipeline is reproducibility infrastructure; the contribution is the dataset, design rationale, documentation, and quality analysis. |
| "Can this be used clinically?" | No direct clinical use. It is for research, benchmarking, fine-tuning experiments, and educational NLP exploration pending human clinical validation. |

## 5. Strong Phrases to Use

- "documented synthetic clinical QA resource"
- "source-traceable QA construction"
- "pregnancy and maternal-health focus"
- "Spanish clinical NLP resource"
- "dataset creation methodology"
- "quality-aware synthetic QA generation"
- "document-level split to reduce leakage"
- "grounded and closed-book training variants"
- "intended for research, not direct clinical decision-making"

## 6. Phrases to Avoid

- "automatic medical expert"
- "clinically validated assistant"
- "diagnostic dataset"
- "safe for patient care"
- "pipeline for generating data" as the main description
- "obstetrics and gynecology" as the only framing if the desired focus is maternal health and pregnancy
