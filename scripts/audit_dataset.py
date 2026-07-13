from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from utils import (
    default_corpus_dir,
    default_datasets_dir,
    default_metadata_dir,
    default_reports_dir,
    load_inventory_manifest,
    read_jsonl,
    write_json,
)


def parse_args() -> argparse.Namespace:
    corpus_dir = default_corpus_dir()
    datasets_dir = default_datasets_dir()
    metadata_dir = default_metadata_dir()
    reports_dir = default_reports_dir()
    parser = argparse.ArgumentParser(description="Audit the Spanish obstetrics LM dataset artifacts.")
    parser.add_argument("--raw-pages", type=Path, default=corpus_dir / "raw_pages.jsonl")
    parser.add_argument("--clean-pages", type=Path, default=corpus_dir / "clean_pages.jsonl")
    parser.add_argument("--chunks", type=Path, default=corpus_dir / "chunks.jsonl")
    parser.add_argument("--train", type=Path, default=datasets_dir / "lm" / "train_lm.jsonl")
    parser.add_argument("--validation", type=Path, default=datasets_dir / "lm" / "validation_lm.jsonl")
    parser.add_argument("--test", type=Path, default=datasets_dir / "lm" / "test_lm.jsonl")
    parser.add_argument("--inventory", type=Path, default=metadata_dir / "inventory.json")
    parser.add_argument("--table-report", "--tables-report", dest="table_report", type=Path, default=reports_dir / "table_extraction_report.json")
    parser.add_argument("--output", type=Path, default=reports_dir / "audit_report.json")
    parser.add_argument("--samples-per-pdf", type=int, default=5)
    parser.add_argument("--sample-chars", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def preview(text: str, max_chars: int) -> str:
    value = " ".join(str(text).split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def validate_lm_records(rows: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for idx, row in enumerate(rows, start=1):
        if "messages" in row:
            errors.append(f"line {idx}: unexpected messages field")
        if not isinstance(row.get("text"), str) or not row.get("text", "").strip():
            errors.append(f"line {idx}: empty text")
        if not isinstance(row.get("metadata"), dict):
            errors.append(f"line {idx}: missing metadata")
    return errors[:50]


def build_chunk_index(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(chunk.get("chunk_id", "")): chunk
        for chunk in chunks
        if str(chunk.get("chunk_id", ""))
    }


def merged_metadata(row: Dict[str, Any], chunk_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    metadata = dict(row.get("metadata", {})) if isinstance(row.get("metadata"), dict) else {}
    chunk = chunk_index.get(str(metadata.get("chunk_id", "")))
    if chunk:
        for key in ("source_pdf", "doc_type", "section", "section_type", "content_role", "topics"):
            if key not in metadata and key in chunk:
                metadata[key] = chunk.get(key)
    return metadata


def source_balance(rows: List[Dict[str, Any]], chunk_index: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    by_pdf: Counter[str] = Counter()
    by_doc_type: Counter[str] = Counter()
    for row in rows:
        metadata = merged_metadata(row, chunk_index)
        by_pdf[str(metadata.get("source_pdf", "unknown"))] += 1
        by_doc_type[str(metadata.get("doc_type", "unknown"))] += 1
    return {
        "by_pdf": dict(sorted(by_pdf.items())),
        "by_doc_type": dict(sorted(by_doc_type.items())),
    }


def topic_distribution(rows: List[Dict[str, Any]], chunk_index: Dict[str, Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        metadata = merged_metadata(row, chunk_index)
        topics = metadata.get("topics", [])
        if isinstance(topics, list):
            for topic in topics:
                counts[str(topic)] += 1
    return counts


def language_distribution(rows: List[Dict[str, Any]], chunk_index: Dict[str, Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        metadata = merged_metadata(row, chunk_index)
        counts[str(metadata.get("language", "unknown"))] += 1
    return counts


def pdf_set(rows: List[Dict[str, Any]], chunk_index: Dict[str, Dict[str, Any]]) -> set[str]:
    pdfs: set[str] = set()
    for row in rows:
        source_pdf = str(merged_metadata(row, chunk_index).get("source_pdf", ""))
        if source_pdf:
            pdfs.add(source_pdf)
    return pdfs


def build_exclusion_summary(manifest: Dict[str, Any]) -> Dict[str, Any]:
    entries = manifest.get("pdfs", [])
    excluded = [
        entry
        for entry in entries
        if entry.get("inclusion_status") == "excluded_with_reason" or entry.get("exclusion_reason")
    ]
    reasons = Counter(str(entry.get("exclusion_reason", "unspecified")) for entry in excluded)
    return {
        "has_exclusion_info": bool(excluded or manifest.get("exclusion_manifest")),
        "excluded_pdf_count": len(excluded),
        "excluded_by_reason": dict(sorted(reasons.items())),
        "excluded_pdfs": [
            {
                "pdf_id": entry.get("pdf_id"),
                "source_pdf": entry.get("source_pdf"),
                "reason": entry.get("exclusion_reason"),
            }
            for entry in excluded
        ],
        "manifest_exclusion_manifest": manifest.get("exclusion_manifest", {}),
    }


def build_table_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    if not report:
        return {"report_found": False}

    pdf_reports = report.get("pdf_reports", []) if isinstance(report.get("pdf_reports"), list) else []
    top_pdfs = sorted(
        (
            {
                "source_pdf": row.get("source_pdf"),
                "total_tables": row.get("total_tables", 0),
                "pages_with_tables": row.get("pages_with_tables", 0),
            }
            for row in pdf_reports
            if row.get("total_tables", 0) > 0
        ),
        key=lambda row: (-int(row.get("total_tables", 0)), str(row.get("source_pdf", ""))),
    )[:10]
    errors = [row for row in pdf_reports if row.get("error")]
    return {
        "report_found": True,
        "total_pdfs": report.get("total_pdfs", 0),
        "pdfs_with_tables": report.get("pdfs_with_tables", 0),
        "total_tables_extracted": report.get("total_tables_extracted", 0),
        "tables_written": report.get("tables_written", 0),
        "strategy": report.get("strategy", "unknown"),
        "pdf_errors": len(errors),
        "top_pdfs_by_table_count": top_pdfs,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    raw_pages = load_if_exists(args.raw_pages)
    clean_pages = load_if_exists(args.clean_pages)
    chunks = load_if_exists(args.chunks)
    train = load_if_exists(args.train)
    validation = load_if_exists(args.validation)
    test = load_if_exists(args.test)
    manifest = load_inventory_manifest(args.inventory)
    table_report = load_json_if_exists(args.table_report)
    chunk_index = build_chunk_index(chunks)

    raw_by_pdf = Counter(str(row.get("source_pdf", "")) for row in raw_pages)
    kept_pages = [row for row in clean_pages if row.get("is_kept") is True]
    dropped_pages = [row for row in clean_pages if row.get("is_kept") is False]
    dropped_by_reason = Counter(str(row.get("drop_reason", "")) for row in dropped_pages)
    needs_ocr = [row for row in raw_pages if row.get("needs_ocr") is True]
    chunks_by_pdf = Counter(str(row.get("source_pdf", "")) for row in chunks)

    manifest_pdfs = manifest.get("pdfs", [])
    doc_type_counts = Counter(str(e.get("doc_type", "unknown")) for e in manifest_pdfs)
    inclusion_counts = Counter(str(e.get("inclusion_status", "unknown")) for e in manifest_pdfs)
    exclusion_reason_counts = Counter(
        str(e.get("exclusion_reason", ""))
        for e in manifest_pdfs
        if e.get("inclusion_status") == "excluded_with_reason"
    )
    ocr_status_counts = Counter(str(e.get("ocr_status", "unknown")) for e in manifest_pdfs)

    train_balance = source_balance(train, chunk_index)
    validation_balance = source_balance(validation, chunk_index)
    test_balance = source_balance(test, chunk_index)
    combined_balance = source_balance(train + validation + test, chunk_index)
    train_pdfs = pdf_set(train, chunk_index)
    validation_pdfs = pdf_set(validation, chunk_index)
    test_pdfs = pdf_set(test, chunk_index)
    overlapping_pdfs = sorted((train_pdfs & validation_pdfs) | (train_pdfs & test_pdfs) | (validation_pdfs & test_pdfs))
    train_topics = topic_distribution(train, chunk_index)
    validation_topics = topic_distribution(validation, chunk_index)
    test_topics = topic_distribution(test, chunk_index)
    train_lang = language_distribution(train, chunk_index)
    validation_lang = language_distribution(validation, chunk_index)
    test_lang = language_distribution(test, chunk_index)
    train_topic_set = set(train_topics)
    validation_topic_set = set(validation_topics)
    test_topic_set = set(test_topics)
    exclusion_summary = build_exclusion_summary(manifest)
    table_summary = build_table_summary(table_report)

    samples_by_pdf: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    chunks_grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_grouped[str(chunk.get("source_pdf", ""))].append(chunk)

    for source_pdf, pdf_chunks in sorted(chunks_grouped.items()):
        sample_size = min(args.samples_per_pdf, len(pdf_chunks))
        for chunk in rng.sample(pdf_chunks, sample_size):
            samples_by_pdf[source_pdf].append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "pages": chunk.get("pages"),
                    "section": chunk.get("section"),
                    "section_type": chunk.get("section_type"),
                    "content_role": chunk.get("content_role"),
                    "doc_type": chunk.get("doc_type"),
                    "topics": chunk.get("topics", []) or chunk.get("topic_tags", []),
                    "token_estimate": chunk.get("token_estimate"),
                    "clinical_score": chunk.get("clinical_score"),
                    "text_preview": preview(str(chunk.get("text", "")), args.sample_chars),
                }
            )

    accepted_examples = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "source_pdf": chunk.get("source_pdf"),
            "doc_type": chunk.get("doc_type"),
            "section_type": chunk.get("section_type"),
            "content_role": chunk.get("content_role"),
            "topics": chunk.get("topics", []) or chunk.get("topic_tags", []),
            "pages": chunk.get("pages"),
            "text_preview": preview(str(chunk.get("text", "")), args.sample_chars),
        }
        for chunk in chunks[:10]
    ]
    discarded_examples = [
        {
            "source_pdf": row.get("source_pdf"),
            "page": row.get("page"),
            "drop_reason": row.get("drop_reason"),
            "text_preview": preview(str(row.get("text", "")), args.sample_chars),
        }
        for row in dropped_pages[:10]
    ]

    report = {
        "raw_pages_path": str(args.raw_pages),
        "clean_pages_path": str(args.clean_pages),
        "chunks_path": str(args.chunks),
        "train_path": str(args.train),
        "validation_path": str(args.validation),
        "test_path": str(args.test),
        "inventory_path": str(args.inventory),
        "table_report_path": str(args.table_report),
        "pdfs_processed": len(raw_by_pdf),
        "pages_total": len(raw_pages),
        "pages_kept": len(kept_pages),
        "pages_discarded": len(dropped_pages),
        "pages_needing_ocr": len(needs_ocr),
        "drop_reasons": dict(sorted(dropped_by_reason.items())),
        "chunks_generated": len(chunks),
        "train_records": len(train),
        "validation_records": len(validation),
        "test_records": len(test),
        "distribution_by_pdf": dict(sorted(chunks_by_pdf.items())),
        "source_balance": {
            "combined": combined_balance,
            "train": train_balance,
            "validation": validation_balance,
            "test": test_balance,
        },
        "leakage_audit": {
            "train_pdf_count": len(train_pdfs),
            "validation_pdf_count": len(validation_pdfs),
            "test_pdf_count": len(test_pdfs),
            "overlap_pdf_count": len(overlapping_pdfs),
            "overlap_pdfs": overlapping_pdfs,
            "leakage_detected": bool(overlapping_pdfs),
        },
        "topic_coverage": {
            "train": dict(sorted(train_topics.items())),
            "validation": dict(sorted(validation_topics.items())),
            "test": dict(sorted(test_topics.items())),
            "shared_topics_all_splits": sorted(train_topic_set & validation_topic_set & test_topic_set),
            "train_only_topics": sorted(train_topic_set - validation_topic_set),
            "validation_only_topics": sorted(validation_topic_set - train_topic_set),
            "test_only_topics": sorted(test_topic_set - train_topic_set - validation_topic_set),
        },
        "language_distribution": {
            "train": dict(sorted(train_lang.items())),
            "validation": dict(sorted(validation_lang.items())),
            "test": dict(sorted(test_lang.items())),
            "combined": dict(sorted((train_lang + validation_lang + test_lang).items())),
        },
        "average_chunk_tokens": round(
            sum(int(chunk.get("token_estimate", 0)) for chunk in chunks) / max(1, len(chunks)),
            2,
        ),
        "lm_validation_errors": {
            "train": validate_lm_records(train),
            "validation": validate_lm_records(validation),
            "test": validate_lm_records(test),
        },
        "accepted_examples": accepted_examples,
        "discarded_page_examples": discarded_examples,
        "manual_review_samples_by_pdf": samples_by_pdf,
        "manifest_version": manifest.get("version", "1.0"),
        "manifest_pdf_count": manifest.get("pdf_count", 0),
        "manifest_included_count": manifest.get("included_count", 0),
        "manifest_excluded_count": manifest.get("excluded_count", 0),
        "doc_type_distribution": dict(sorted(doc_type_counts.items())),
        "inclusion_status_distribution": dict(sorted(inclusion_counts.items())),
        "exclusion_reason_distribution": dict(sorted(exclusion_reason_counts.items())),
        "ocr_status_distribution": dict(sorted(ocr_status_counts.items())),
        "exclusion_manifest": manifest.get("exclusion_manifest", {}),
        "exclusion_manifest_summary": exclusion_summary,
        "table_extraction_summary": table_summary,
    }
    write_json(args.output, report)

    print(f"PDFs processed: {report['pdfs_processed']}")
    print(f"Pages total: {report['pages_total']}")
    print(f"Pages kept: {report['pages_kept']}")
    print(f"Pages discarded: {report['pages_discarded']}")
    print(f"Pages needing OCR: {report['pages_needing_ocr']}")
    print(f"Chunks generated: {report['chunks_generated']}")
    print(f"Train records: {report['train_records']}")
    print(f"Validation records: {report['validation_records']}")
    print(f"Test records: {report['test_records']}")
    print(
        f"Leakage audit: {'DETECTED' if report['leakage_audit']['leakage_detected'] else 'clean'} "
        f"(overlap_pdfs={report['leakage_audit']['overlap_pdf_count']})"
    )
    print(
        f"Topic coverage: train={len(report['topic_coverage']['train'])} "
        f"validation={len(report['topic_coverage']['validation'])} "
        f"test={len(report['topic_coverage']['test'])}"
    )
    if table_summary.get("report_found"):
        print(
            f"Table extraction: {table_summary['total_tables_extracted']} tables "
            f"across {table_summary['pdfs_with_tables']} PDFs"
        )
    print(
        f"Manifest PDFs: {report['manifest_pdf_count']} "
        f"(included={report['manifest_included_count']}, excluded={report['manifest_excluded_count']})"
    )
    print(f"Doc types: {dict(doc_type_counts)}")
    print(f"Saved audit report to: {args.output}")


if __name__ == "__main__":
    main()
