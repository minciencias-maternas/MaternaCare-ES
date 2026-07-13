"""Backward-compatible wrappers for obstetrics enrichment helpers.

Single source of truth lives in utils.py. This module re-exports the
Phase 3-6 helpers so existing imports keep working.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from utils import (
    classify_content_role,
    classify_section_type,
    enrich_chunks as _enrich_chunks,
    find_duplicate_documents,
    split_train_validation_by_document,
    tag_topics,
)


CLINICAL_CONTENT_ROLES = {"evidence", "recommendation", "procedure", "diagnostic", "treatment"}


def extract_topic_tags(text: str, min_matches: int = 1) -> List[str]:
    del min_matches
    return tag_topics(text)


def enrich_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    return _enrich_chunks([chunk])[0]


def enrich_chunks(chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _enrich_chunks(chunks)


def split_by_document(
    chunks: Sequence[Dict[str, Any]],
    validation_ratio: float = 0.10,
    seed: int = 42,
    stratify_by: str = "doc_type",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    train_chunks, validation_chunks = split_train_validation_by_document(
        chunks,
        validation_ratio=validation_ratio,
        seed=seed,
        stratify_by=stratify_by,
    )
    train_pdfs = {str(chunk.get("pdf_id") or chunk.get("source_pdf", "")) for chunk in train_chunks}
    validation_pdfs = {str(chunk.get("pdf_id") or chunk.get("source_pdf", "")) for chunk in validation_chunks}
    overlap = sorted(train_pdfs & validation_pdfs)
    report = {
        "total_documents": len(train_pdfs | validation_pdfs),
        "train_documents": len(train_pdfs),
        "validation_documents": len(validation_pdfs),
        "train_chunks": len(train_chunks),
        "validation_chunks": len(validation_chunks),
        "validation_ratio": validation_ratio,
        "seed": seed,
        "stratify_by": stratify_by,
        "document_overlap": len(overlap),
        "overlap_documents": overlap,
    }
    return train_chunks, validation_chunks, report


def deduplicate_documents(
    manifest_entries: Sequence[Dict[str, Any]],
    chunks: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    duplicates = find_duplicate_documents(manifest_entries)
    duplicate_ids = {pdf_id_b for _, pdf_id_b, _ in duplicates}
    filtered_entries = [entry for entry in manifest_entries if str(entry.get("pdf_id", "")) not in duplicate_ids]
    filtered_chunks = [chunk for chunk in chunks if str(chunk.get("pdf_id") or chunk.get("source_pdf", "")) not in duplicate_ids]
    report = {
        "duplicate_documents_found": len(duplicate_ids),
        "duplicate_details": [
            {"pdf_id_a": pdf_id_a, "pdf_id_b": pdf_id_b, "similarity": similarity}
            for pdf_id_a, pdf_id_b, similarity in duplicates
        ],
        "remaining_documents": len(filtered_entries),
        "remaining_chunks": len(filtered_chunks),
    }
    return filtered_entries, filtered_chunks, report


def is_clinical_content(chunk: Dict[str, Any]) -> bool:
    return str(chunk.get("content_role", "")) in CLINICAL_CONTENT_ROLES


def filter_clinical_chunks(chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [chunk for chunk in chunks if is_clinical_content(chunk)]
