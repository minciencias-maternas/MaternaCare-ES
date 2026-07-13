# Related Work and Citation Strategy

This file lists the main references and how to use them in the MaternaQA-es dataset paper.

## 1. Dataset Documentation Frameworks

These references help justify the structure of the paper itself.

| Reference | Why it matters | How to use it |
|---|---|---|
| Gebru et al., **Datasheets for Datasets** | Establishes that datasets should document motivation, composition, collection process, preprocessing, uses, distribution, and maintenance. | Use in Methodology/Ethics to justify transparent dataset documentation. |
| Bender & Friedman, **Data Statements for NLP** | Tailored to language datasets; emphasizes curation rationale, language variety, source data, annotator demographics, text characteristics, limitations, and ethical review. | Use to explain why the paper documents Spanish language, source provenance, preprocessing, and limitations. |
| Pushkarna et al., **Data Cards** | Provides a practical framework for concise dataset documentation, intended use, risk, and stakeholder-facing transparency. | Use in Dataset Availability and Intended Uses. |

### Recommended wording

> We align the dataset description with established transparency practices for ML and NLP datasets, including datasheets, data statements, and data cards, which emphasize documenting dataset motivation, composition, collection, preprocessing, intended uses, distribution, maintenance, and limitations.

## 2. General QA Dataset Papers

| Reference | What to borrow | Relevance to MaternaQA-es |
|---|---|---|
| **SQuAD** | Dataset collection stages, QA format, dataset analysis, baseline framing. | Shows how QA dataset papers document creation and analyze question/answer properties. |
| **Natural Questions** | Realistic QA setup, annotation quality evaluation, robust metrics, human variability. | Supports quality assessment and the importance of realistic information-seeking QA. |
| **QASPER** | Questions anchored in scientific papers and evidence spans. | Useful for discussing evidence-grounded QA and source-backed answers. |

## 3. Biomedical and Clinical QA Datasets

| Reference | What it contributes | Use in paper |
|---|---|---|
| **PubMedQA** | Biomedical QA from PubMed abstracts; includes expert-labeled, unlabeled, and artificially generated subsets. | Strong precedent for biomedical QA and artificial QA instances. |
| **MedQuAD** | Medical question-answer pairs from NIH sources. | Compare against broad medical QA resources. |
| **MedMCQA** | Large-scale medical multiple-choice QA. | Contrast with multiple-choice exam-style datasets. |
| **BioASQ** | Biomedical semantic indexing and QA benchmark. | Use as broader biomedical QA benchmark context. |
| **RealMedQA** | Realistic biomedical/clinical QA from guidelines; includes human and LLM question generation with verification. | Very relevant precedent for clinical QA dataset creation using guideline/document sources and LLM generation. |
| **Med-PaLM / MultiMedQA** | Evaluation of LLMs on medical QA and clinical safety concerns. | Use to justify rigorous evaluation and caution in clinical settings. |

## 4. Synthetic Instruction and QA Generation

| Reference | Why it matters | Where to use |
|---|---|---|
| **Self-Instruct** | Shows how models can bootstrap instruction-following data from generated tasks. | Synthetic QA Generation Methodology. |
| **Alpaca** | Popular demonstration of synthetic instruction data for instruction tuning. | Related Work and downstream-use discussion. |
| **WizardLM / Evol-Instruct** | Shows instruction evolution/diversification strategies. | Justify diversity of question types and complexity. |
| **LIMA: Less Is More for Alignment** | Argues that small, carefully curated instruction datasets can produce strong alignment behavior. | Discussion: quality over raw scale; stronger for second fine-tuning paper but useful as dataset-design rationale. |
| **FLAN** | Shows value of instruction tuning across tasks. | Intended Uses and second-paper bridge. |

## 5. Grounded QA and Retrieval-Augmented Generation

| Reference | Why it matters | Where to use |
|---|---|---|
| **RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** | Establishes evidence-conditioned generation. | Explain `sft_grounded` variant and evidence-based QA. |
| **Natural Questions / QASPER / PubMedQA** | QA tasks with context/evidence. | Explain why source context is retained and evaluated. |

## 6. Corpus Curation References

| Reference | Why it matters | Where to use |
|---|---|---|
| **The Pile** | Documents large-scale corpus construction, source mixture, dataset documentation, ethics. | Corpus Construction and Ethics. |
| **Dolma** | Strong open-corpus example with design principles, filtering, deduplication, decontamination, and toolkit release. | Corpus Processing and Reproducibility. |

## 7. Efficient Fine-Tuning References: Mention but Do Not Center

These are more central to Paper 2, but can be cited briefly in Paper 1 when describing intended use.

| Reference | Use in Paper 1 |
|---|---|
| **LoRA** | Mention that one intended use is parameter-efficient fine-tuning. |
| **QLoRA** | Mention that the dataset variants are compatible with low-resource fine-tuning experiments. |
| **TRL / supervised fine-tuning literature** | Mention only as implementation context for downstream work. |

## 8. Suggested Related Work Subsections

A clean Related Work section could be organized as:

```text
2. Related Work
  2.1 Biomedical and clinical question-answering datasets
  2.2 Synthetic instruction and QA data generation
  2.3 Dataset documentation and responsible data release
  2.4 Evidence-grounded QA and downstream adaptation
```

## 9. Reference List with Links

- Rajpurkar et al. SQuAD: https://aclanthology.org/D16-1264.pdf
- Kwiatkowski et al. Natural Questions: https://aclanthology.org/Q19-1026.pdf
- Jin et al. PubMedQA: https://pubmedqa.github.io
- RealMedQA: https://pmc.ncbi.nlm.nih.gov/articles/PMC12099375
- Gebru et al. Datasheets for Datasets: https://arxiv.org/abs/1803.09010
- Bender & Friedman. Data Statements for NLP: https://aclanthology.org/Q18-1041.pdf
- Data Statements resources: https://techpolicylab.uw.edu/data-statements
- Pushkarna et al. Data Cards: https://www.datacentricai.org/neurips21/papers/112_CameraReady_Data_Cards.pdf
- The Pile: https://pile.eleuther.ai/paper.pdf
- Dolma: https://aclanthology.org/2024.acl-long.840.pdf
- Self-Instruct: https://arxiv.org/abs/2212.10560
- Alpaca: https://crfm.stanford.edu/2023/03/13/alpaca.html
- WizardLM / Evol-Instruct: https://arxiv.org/abs/2304.12244
- LIMA: https://arxiv.org/abs/2305.11206
- FLAN: https://arxiv.org/abs/2210.11416
- RAG: https://arxiv.org/abs/2005.11401
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- Med-PaLM: https://arxiv.org/abs/2212.13138
- MedMCQA: https://arxiv.org/abs/2203.14371
