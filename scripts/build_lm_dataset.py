from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from transformers import AutoTokenizer

from utils import (
    accepted_for_lm,
    assign_chunk_ids,
    chunk_records,
    dedupe_chunks,
    default_corpus_dir,
    default_datasets_dir,
    default_metadata_dir,
    default_reports_dir,
    enrich_chunks,
    filter_lm_chunks,
    read_jsonl,
    split_train_validation_test_by_document,
    to_lm_record,
    write_json,
    write_jsonl,
)


DEFAULT_MODEL_NAME = "google/gemma-4-E2B-it"
MODEL_TOKENIZERS = {
    "google/gemma-4-E2B-it": "google/gemma-4-E2B-it",
    "google/medgemma-1.5-4b-it": "google/medgemma-1.5-4b-it",
}


def resolve_tokenizer_name(model_name: str, tokenizer_name: str | None) -> str:
    if tokenizer_name:
        return tokenizer_name
    normalized = model_name.lower()
    if "medgemma" in normalized:
        return MODEL_TOKENIZERS["google/medgemma-1.5-4b-it"]
    return MODEL_TOKENIZERS[DEFAULT_MODEL_NAME]


def parse_args() -> argparse.Namespace:
    corpus_dir = default_corpus_dir()
    datasets_dir = default_datasets_dir()
    metadata_dir = default_metadata_dir()
    reports_dir = default_reports_dir()
    parser = argparse.ArgumentParser(description="Build LM train/validation/test JSONL from cleaned obstetrics pages.")
    parser.add_argument("--input", type=Path, default=corpus_dir / "clean_pages.jsonl")
    parser.add_argument("--inventory", type=Path, default=metadata_dir / "inventory.json")
    parser.add_argument("--chunks-output", type=Path, default=corpus_dir / "chunks.jsonl")
    parser.add_argument("--train-output", type=Path, default=datasets_dir / "lm" / "train_lm.jsonl")
    parser.add_argument("--validation-output", type=Path, default=datasets_dir / "lm" / "validation_lm.jsonl")
    parser.add_argument("--test-output", type=Path, default=datasets_dir / "lm" / "test_lm.jsonl")
    parser.add_argument("--build-report-output", type=Path, default=reports_dir / "build_report.json")
    parser.add_argument("--min-tokens", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base model name; used to select the matching tokenizer automatically.",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=100,
        help="Number of real tokenizer tokens to overlap between adjacent chunks.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Optional tokenizer override for overlap computation.",
    )
    parser.add_argument("--min-accepted-tokens", type=int, default=180)
    parser.add_argument("--min-clinical-score", type=int, default=5)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument(
        "--allowed-languages",
        type=str,
        default="es,unknown",
        help="Idiomas permitidos para export LM (coma-separados).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _pdf_set(rows: Iterable[Dict[str, Any]]) -> set[str]:
    return {str(row.get("pdf_id") or row.get("source_pdf", "")) for row in rows}


def _topic_distribution(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        topics = row.get("topics") or row.get("topic_tags") or []
        for topic in topics:
            counts[str(topic)] += 1
    return dict(sorted(counts.items()))


def _content_role_distribution(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter(str(row.get("content_role", "unknown")) for row in rows)
    return dict(sorted(counts.items()))


def _doc_type_distribution(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter(str(row.get("doc_type", "unknown")) for row in rows)
    return dict(sorted(counts.items()))


def _language_distribution(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter(str(row.get("language", "unknown")) for row in rows)
    return dict(sorted(counts.items()))


def _dedupe_audit(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    exact_owner: Dict[str, str] = {}
    near_owner: Dict[str, str] = {}
    exact_removed = 0
    near_removed = 0
    cross_pdf_exact = 0
    cross_pdf_near = 0
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        source_pdf = str(row.get("source_pdf", ""))
        exact_key = text
        near_key = " ".join(text.lower().split()[:240])
        if exact_key in exact_owner:
            exact_removed += 1
            if exact_owner[exact_key] != source_pdf:
                cross_pdf_exact += 1
        else:
            exact_owner[exact_key] = source_pdf
        if near_key in near_owner:
            near_removed += 1
            if near_owner[near_key] != source_pdf:
                cross_pdf_near += 1
        else:
            near_owner[near_key] = source_pdf
    return {
        "exact_duplicates_detected": exact_removed,
        "near_duplicates_detected": near_removed,
        "cross_pdf_exact_duplicates": cross_pdf_exact,
        "cross_pdf_near_duplicates": cross_pdf_near,
    }


def main() -> None:
    args = parse_args()
    tokenizer_name = resolve_tokenizer_name(args.model_name, args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    clean_rows = read_jsonl(args.input)
    kept_rows = [row for row in clean_rows if row.get("is_kept") is True]
    candidate_chunks = chunk_records(
        kept_rows,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        tokenizer=tokenizer,
    )

    accepted: List[Dict[str, Any]] = []
    rejected_reasons: Counter[str] = Counter()
    for chunk in candidate_chunks:
        is_accepted, reason = accepted_for_lm(
            chunk,
            min_tokens=args.min_accepted_tokens,
            min_score=args.min_clinical_score,
        )
        if is_accepted:
            accepted.append(chunk)
        else:
            rejected_reasons[reason] += 1

    chunks = assign_chunk_ids(dedupe_chunks(accepted))

    # Phase 3-5 enrichment metadata feeds the document-level split report.
    chunks = enrich_chunks(chunks)

    # Phase 6: document-level split over exportable LM chunks so the actual
    # exported train/validation/test files stay close to the requested ratios.
    allowed_languages = [x.strip() for x in args.allowed_languages.split(",") if x.strip()]
    exportable_chunks = filter_lm_chunks(chunks, allowed_languages=allowed_languages)
    train_chunks, validation_chunks, test_chunks = split_train_validation_test_by_document(
        exportable_chunks,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_by="doc_type",
    )

    output_train_chunks = train_chunks
    output_validation_chunks = validation_chunks
    output_test_chunks = test_chunks
    train_records = [to_lm_record({**chunk, "split": "train"}) for chunk in output_train_chunks]
    validation_records = [
        to_lm_record({**chunk, "split": "validation"})
        for chunk in output_validation_chunks
    ]
    test_records = [to_lm_record({**chunk, "split": "test"}) for chunk in output_test_chunks]

    write_jsonl(args.chunks_output, chunks)
    write_jsonl(args.train_output, train_records)
    write_jsonl(args.validation_output, validation_records)
    write_jsonl(args.test_output, test_records)

    by_pdf = Counter(str(chunk.get("source_pdf", "")) for chunk in chunks)

    train_pdfs = _pdf_set(train_chunks)
    validation_pdfs = _pdf_set(validation_chunks)
    test_pdfs = _pdf_set(test_chunks)
    leakage_pdfs = sorted((train_pdfs & validation_pdfs) | (train_pdfs & test_pdfs) | (validation_pdfs & test_pdfs))
    output_train_pdfs = _pdf_set(output_train_chunks)
    output_validation_pdfs = _pdf_set(output_validation_chunks)
    output_test_pdfs = _pdf_set(output_test_chunks)
    total_export_records = max(1, len(train_records) + len(validation_records) + len(test_records))

    report = {
        "input": str(args.input),
        "chunks_output": str(args.chunks_output),
        "train_output": str(args.train_output),
        "validation_output": str(args.validation_output),
        "test_output": str(args.test_output),
        "clean_pages_read": len(clean_rows),
        "kept_pages_read": len(kept_rows),
        "candidate_chunks": len(candidate_chunks),
        "accepted_chunks_before_dedupe": len(accepted),
        "final_chunks": len(chunks),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "test_records": len(test_records),
        "rejected_chunk_reasons": dict(sorted(rejected_reasons.items())),
        "chunks_by_pdf": dict(sorted(by_pdf.items())),
        "average_token_estimate": round(
            sum(int(chunk.get("token_estimate", 0)) for chunk in chunks) / max(1, len(chunks)),
            2,
        ),
        "split_method": "document_level_chunk_balanced",
        "stratify_by": "doc_type",
        "allowed_languages": allowed_languages,
        "requested_split_ratios": {
            "train": round(1 - args.validation_ratio - args.test_ratio, 4),
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "actual_split_ratios": {
            "train": round(len(train_records) / total_export_records, 4),
            "validation": round(len(validation_records) / total_export_records, 4),
            "test": round(len(test_records) / total_export_records, 4),
        },
        "train_pdfs": len(output_train_pdfs),
        "validation_pdfs": len(output_validation_pdfs),
        "test_pdfs": len(output_test_pdfs),
        "train_pdf_ids": sorted(output_train_pdfs),
        "validation_pdf_ids": sorted(output_validation_pdfs),
        "test_pdf_ids": sorted(output_test_pdfs),
        "leakage_detected": bool(leakage_pdfs),
        "leakage_pdf_count": len(leakage_pdfs),
        "leakage_pdfs": leakage_pdfs,
        "topic_distribution_train": _topic_distribution(output_train_chunks),
        "topic_distribution_validation": _topic_distribution(output_validation_chunks),
        "topic_distribution_test": _topic_distribution(output_test_chunks),
        "content_role_distribution_train": _content_role_distribution(output_train_chunks),
        "content_role_distribution_validation": _content_role_distribution(output_validation_chunks),
        "content_role_distribution_test": _content_role_distribution(output_test_chunks),
        "doc_type_distribution_train": _doc_type_distribution(output_train_chunks),
        "doc_type_distribution_validation": _doc_type_distribution(output_validation_chunks),
        "doc_type_distribution_test": _doc_type_distribution(output_test_chunks),
        "language_distribution_train": _language_distribution(output_train_chunks),
        "language_distribution_validation": _language_distribution(output_validation_chunks),
        "language_distribution_test": _language_distribution(output_test_chunks),
        "dedupe_audit": _dedupe_audit(accepted),
    }
    write_json(args.build_report_output, report)

    print(f"Candidate chunks: {len(candidate_chunks)}")
    print(f"Final chunks: {len(chunks)}")
    print(f"Train records: {len(train_records)} (from {report['train_pdfs']} PDFs)")
    print(f"Validation records: {len(validation_records)} (from {report['validation_pdfs']} PDFs)")
    print(f"Test records: {len(test_records)} (from {report['test_pdfs']} PDFs)")
    print(f"Leakage detected: {report['leakage_detected']}")
    print(f"Saved chunks to: {args.chunks_output}")
    print(f"Saved train dataset to: {args.train_output}")
    print(f"Saved validation dataset to: {args.validation_output}")
    print(f"Saved test dataset to: {args.test_output}")
    print(f"Saved build report to: {args.build_report_output}")


if __name__ == "__main__":
    main()
