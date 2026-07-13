"""
document_manifest.py
======================
Document manifest system for the obstetrics pipeline.
Handles document type classification, inclusion/exclusion criteria,
and metadata extracted from PDFs.
"""
from __future__ import annotations

import json
from collections import Counter
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


class DocumentType(str, Enum):
    """Medical document types in the corpus."""

    GPC = "gpc"  # Clinical practice guideline.
    PROTOCOL = "protocol"  # Hospital protocol or algorithm.
    MANUAL = "manual"  # Academic manual or university textbook.
    ARTICLE = "article"  # Scientific article.
    BOOK = "book"  # Book or book chapter.
    UNKNOWN = "unknown"  # Unclassified.


class InclusionStatus(str, Enum):
    """Inclusion status for a document in the corpus."""

    INCLUDED = "included"
    EXCLUDED = "excluded"


class ExclusionReason(str, Enum):
    """Objective exclusion reasons."""

    OCR_DOMINATED = "ocr_dominated"  # >50% of pages require OCR.
    NON_CLINICAL = "non_clinical"  # <10% clinical-term density.
    DUPLICATE = "duplicate"  # Duplicate document.
    CORRUPTED = "corrupted"  # Corrupted or unreadable PDF.
    WRONG_LANGUAGE = "wrong_language"  # Document is not in Spanish.
    EMPTY = "empty"  # No extractable content.
    METADATA_ONLY = "metadata_only"  # Cover/table-of-contents/metadata only.
    EXCLUDED_BY_USER = "excluded_by_user"  # Manually excluded.


@dataclass
class DocumentMetadata:
    """Metadata extracted from a PDF."""

    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "keywords": self.keywords,
            "creator": self.creator,
            "producer": self.producer,
            "creation_date": self.creation_date,
            "modification_date": self.modification_date,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        return cls(
            title=data.get("title"),
            author=data.get("author"),
            subject=data.get("subject"),
            keywords=data.get("keywords"),
            creator=data.get("creator"),
            producer=data.get("producer"),
            creation_date=data.get("creation_date"),
            modification_date=data.get("modification_date"),
        )


@dataclass
class DocumentEntry:
    """A single document entry in the manifest."""

    pdf_id: str
    source_pdf: str
    source_path: str
    file_size: int
    page_count: int
    doc_type: DocumentType = DocumentType.UNKNOWN
    inclusion_status: InclusionStatus = InclusionStatus.INCLUDED
    exclusion_reason: Optional[ExclusionReason] = None
    exclusion_details: Optional[str] = None
    ocr_status: Dict[str, Any] = field(default_factory=dict)
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    fallback_pages: int = 0
    needs_ocr_pages: int = 0
    clinical_page_count: int = 0
    clinical_term_ratio: float = 0.0
    ocr_failed_pages: int = 0
    text_sample: str = ""
    page_ocr_status: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        inclusion_status = self.inclusion_status.value
        if self.inclusion_status == InclusionStatus.EXCLUDED:
            inclusion_status = "excluded_with_reason"
        return {
            "pdf_id": self.pdf_id,
            "source_pdf": self.source_pdf,
            "source_path": self.source_path,
            "file_size": self.file_size,
            "page_count": self.page_count,
            "doc_type": self.doc_type.value,
            "inclusion_status": inclusion_status,
            "exclusion_reason": self.exclusion_reason.value if self.exclusion_reason else None,
            "exclusion_details": self.exclusion_details,
            "ocr_status": self.ocr_status,
            "metadata": self.metadata.to_dict(),
            "fallback_pages": self.fallback_pages,
            "needs_ocr_pages": self.needs_ocr_pages,
            "clinical_page_count": self.clinical_page_count,
            "clinical_term_ratio": round(self.clinical_term_ratio, 4),
            "ocr_failed_pages": self.ocr_failed_pages,
            "text_sample": self.text_sample,
            "page_ocr_status": self.page_ocr_status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentEntry":
        inclusion_status = data.get("inclusion_status", "included")
        if inclusion_status == "excluded_with_reason":
            inclusion_status = InclusionStatus.EXCLUDED.value
        return cls(
            pdf_id=data["pdf_id"],
            source_pdf=data["source_pdf"],
            source_path=data["source_path"],
            file_size=data["file_size"],
            page_count=data["page_count"],
            doc_type=DocumentType(data.get("doc_type", "unknown")),
            inclusion_status=InclusionStatus(inclusion_status),
            exclusion_reason=ExclusionReason(data["exclusion_reason"]) if data.get("exclusion_reason") else None,
            exclusion_details=data.get("exclusion_details"),
            ocr_status=data.get("ocr_status", {}),
            metadata=DocumentMetadata.from_dict(data.get("metadata", {})),
            fallback_pages=data.get("fallback_pages", 0),
            needs_ocr_pages=data.get("needs_ocr_pages", 0),
            clinical_page_count=data.get("clinical_page_count", 0),
            clinical_term_ratio=data.get("clinical_term_ratio", 0.0),
            ocr_failed_pages=data.get("ocr_failed_pages", 0),
            text_sample=data.get("text_sample", ""),
            page_ocr_status=data.get("page_ocr_status", []),
        )


class DocumentManifest:
    """Document manifest for the corpus."""

    def __init__(self, entries: Optional[List[DocumentEntry]] = None) -> None:
        self.entries: List[DocumentEntry] = entries or []
        self._by_pdf_id: Dict[str, DocumentEntry] = {}
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._by_pdf_id = {e.pdf_id: e for e in self.entries}

    def add(self, entry: DocumentEntry) -> None:
        """Add a new entry or update an existing one."""
        self._by_pdf_id[entry.pdf_id] = entry
        self.entries = list(self._by_pdf_id.values())

    def get(self, pdf_id: str) -> Optional[DocumentEntry]:
        return self._by_pdf_id.get(pdf_id)

    def included(self) -> List[DocumentEntry]:
        return [e for e in self.entries if e.inclusion_status == InclusionStatus.INCLUDED]

    def excluded(self) -> List[DocumentEntry]:
        return [e for e in self.entries if e.inclusion_status == InclusionStatus.EXCLUDED]

    def by_type(self, doc_type: DocumentType) -> List[DocumentEntry]:
        return [e for e in self.entries if e.doc_type == doc_type]

    def to_dict(self) -> Dict[str, Any]:
        doc_type_counts = Counter(e.doc_type.value for e in self.entries)
        ocr_status_counts = Counter(
            str(e.ocr_status.get("status", "unknown")) for e in self.entries
        )
        return {
            "version": "2.1",
            "pdf_count": len(self.entries),
            "included_count": len(self.included()),
            "excluded_count": len(self.excluded()),
            "doc_type_distribution": dict(sorted(doc_type_counts.items())),
            "ocr_status_distribution": dict(sorted(ocr_status_counts.items())),
            "exclusion_manifest": generate_exclusion_report(self),
            "pdfs": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentManifest":
        entries = [DocumentEntry.from_dict(e) for e in data.get("pdfs", [])]
        return cls(entries=entries)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            f.write("\n")

    @classmethod
    def load(cls, path: Path) -> "DocumentManifest":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def normalize_for_compare(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


_SPANISH_MARKERS = {
    "el",
    "la",
    "los",
    "las",
    "una",
    "para",
    "con",
    "embarazo",
    "gestacion",
    "mujer",
    "parto",
    "salud",
}

_ENGLISH_MARKERS = {
    "the",
    "and",
    "with",
    "pregnancy",
    "women",
    "birth",
    "health",
    "guideline",
}

_CLINICAL_TERMS = {
    "embarazo",
    "obstetricia",
    "ginecologia",
    "gestacion",
    "parto",
    "cesarea",
    "preeclampsia",
    "eclampsia",
    "feto",
    "fetal",
    "prenatal",
    "hemorragia",
    "lactancia",
    "recien",
    "nacido",
    "uterino",
    "materna",
}


def detect_language(text: str) -> str:
    normalized = normalize_for_compare(text)
    if not normalized:
        return "unknown"
    words = normalized.split()
    spanish_hits = sum(1 for word in words if word in _SPANISH_MARKERS)
    english_hits = sum(1 for word in words if word in _ENGLISH_MARKERS)
    if spanish_hits == english_hits == 0:
        return "unknown"
    return "es" if spanish_hits >= english_hits else "en"


def compute_clinical_density(text: str) -> float:
    normalized = normalize_for_compare(text)
    words = normalized.split()
    if not words:
        return 0.0
    clinical_hits = sum(1 for word in words if word in _CLINICAL_TERMS)
    return round(clinical_hits / len(words), 4)


def classify_doc_type_from_text(text: str, pdf_name: str = "") -> DocumentType:
    return classify_document_type(text, pdf_name)


def classify_document_type(text: str, pdf_name: str) -> DocumentType:
    """Classify document type from filename and content."""
    name_lower = pdf_name.lower()
    text_lower = text.lower()
    # Strong signals in the filename.
    if any(k in name_lower for k in ["gpc", "guía de práctica", "guia de practica", "clinical practice"]):
        return DocumentType.GPC
    if any(k in name_lower for k in ["protocolo", "protocol", "algoritmo", "algorithm", "procedimiento"]):
        return DocumentType.PROTOCOL
    if any(k in name_lower for k in ["manual", "texto", "universidad", "académico", "academico"]):
        return DocumentType.MANUAL
    if any(k in name_lower for k in ["artículo", "articulo", "article", "paper", "pubmed"]):
        return DocumentType.ARTICLE

    # Content signals from the first 5000 characters.
    sample = text_lower[:5000]
    gpc_markers = ["guía de práctica clínica", "evidencia científica", "nivel de evidencia",
                   "recomendación formal", "grado de recomendación", "consenso"]
    protocol_markers = ["protocolo de", "algoritmo de", "flujo de", "procedimiento de",
                        "manejo de", "pasos a seguir", "indicaciones"]
    article_markers = ["resumen", "abstract", "objetivo", "métodos", "resultados",
                       "conclusiones", "palabras clave", "doi", "introducción"]
    book_markers = ["editorial", "isbn", "primera edición", "segunda edición", "derechos reservados",
                    "tabla de contenido", "prólogo", "prefacio"]

    scores = {
        DocumentType.GPC: sum(1 for m in gpc_markers if m in sample),
        DocumentType.PROTOCOL: sum(1 for m in protocol_markers if m in sample),
        DocumentType.ARTICLE: sum(1 for m in article_markers if m in sample),
        DocumentType.BOOK: sum(1 for m in book_markers if m in sample),
    }

    # Prefer direct signals when confidence is high.
    if scores:
        best_type, best_score = max(scores.items(), key=lambda x: x[1])
        if best_score >= 2:
            return best_type
    # Fallback when the filename suggests a chapter.
    if any(k in name_lower for k in ["cap", "capítulo", "capitulo", "chapter"]):
        return DocumentType.BOOK
    # Keep UNKNOWN when no reliable evidence is available.
    return DocumentType.UNKNOWN


def evaluate_inclusion(
    entry: DocumentEntry,
    min_clinical_ratio: float = 0.10,
    max_ocr_ratio: float = 0.50,
    min_pages: int = 3,
) -> Tuple[InclusionStatus, Optional[ExclusionReason], Optional[str]]:
    """Evaluate whether a document should be included in the corpus.

    Returns:
        Tuple of (inclusion_status, exclusion_reason, exclusion_details).
    """
    # 1) Exclude OCR-dominated documents.
    if entry.page_count > 0:
        ocr_ratio = entry.needs_ocr_pages / entry.page_count
        if ocr_ratio > max_ocr_ratio:
            return (
                InclusionStatus.EXCLUDED,
                ExclusionReason.OCR_DOMINATED,
                f"{entry.needs_ocr_pages}/{entry.page_count} páginas ({ocr_ratio:.1%}) requieren OCR",
            )

    # 2) Exclude documents with low clinical density.
    if entry.clinical_term_ratio < min_clinical_ratio:
        return (
            InclusionStatus.EXCLUDED,
            ExclusionReason.NON_CLINICAL,
            f"Ratio de términos clínicos: {entry.clinical_term_ratio:.1%} (mínimo: {min_clinical_ratio:.0%})",
        )

    # 3) Exclude documents below the minimum page threshold.
    if entry.page_count < min_pages:
        return (
            InclusionStatus.EXCLUDED,
            ExclusionReason.EMPTY,
            f"Solo {entry.page_count} páginas (mínimo: {min_pages})",
        )

    # 4-5) Duplicate and language checks are handled in upstream stages.

    return InclusionStatus.INCLUDED, None, None


def generate_exclusion_report(manifest: DocumentManifest) -> Dict[str, Any]:
    """Generate an exclusion summary for manifest entries."""
    excluded = manifest.excluded()
    by_reason: Dict[str, List[DocumentEntry]] = {}
    for e in excluded:
        reason = e.exclusion_reason.value if e.exclusion_reason else "unknown"
        by_reason.setdefault(reason, []).append(e)

    return {
        "total_documents": len(manifest.entries),
        "excluded_count": len(excluded),
        "included_count": len(manifest.included()),
        "exclusion_rate": round(len(excluded) / max(1, len(manifest.entries)), 4),
        "by_reason": {
            reason: {
                "count": len(entries),
                "documents": [
                    {
                        "pdf_id": e.pdf_id,
                        "source_pdf": e.source_pdf,
                        "details": e.exclusion_details,
                    }
                    for e in entries
                ],
            }
            for reason, entries in sorted(by_reason.items())
        },
    }


# Compatibility alias used by extract_pdfs.py.
Manifest = DocumentManifest


def classify_doc_type_from_filename(pdf_name: str) -> DocumentType:
    """Classify document type using only the filename."""
    return classify_document_type("", pdf_name)


def extract_pdf_metadata(document: Any) -> DocumentMetadata:
    """Extract metadata from a PyMuPDF document."""
    try:
        meta = document.metadata
        if meta:
            return DocumentMetadata(
                title=meta.get("title") or None,
                author=meta.get("author") or None,
                subject=meta.get("subject") or None,
                keywords=meta.get("keywords") or None,
                creator=meta.get("creator") or None,
                producer=meta.get("producer") or None,
                creation_date=meta.get("creationDate") or None,
                modification_date=meta.get("modDate") or None,
            )
    except Exception:
        pass
    return DocumentMetadata()


def compute_document_ocr_status(
    page_count: int,
    needs_ocr_pages: int,
    ocr_failed_pages: int = 0,
) -> Dict[str, Any]:
    """Compute OCR status metrics for a document."""
    if page_count == 0:
        return {"status": "unknown", "ratio": 0.0}

    ocr_ratio = needs_ocr_pages / page_count
    if ocr_ratio == 0:
        status = "clean"
    elif ocr_ratio <= 0.2:
        status = "minor_ocr"
    elif ocr_ratio <= 0.5:
        status = "needs_ocr"
    else:
        status = "ocr_dominated"

    return {
        "status": status,
        "needs_ocr_pages": needs_ocr_pages,
        "ocr_failed_pages": ocr_failed_pages,
        "ocr_ratio": round(ocr_ratio, 4),
    }


def determine_inclusion_status(
    doc_type: DocumentType,
    page_count: int,
    sample_text: str,
    file_size: int,
    min_pages: int = 3,
    max_empty_ratio: float = 0.8,
) -> Tuple[InclusionStatus, Optional[ExclusionReason]]:
    """Determine the inclusion status for a document.

    Returns:
        Tuple of (inclusion_status, exclusion_reason).
    """
    # Enforce a minimum page count.
    if page_count < min_pages:
        return InclusionStatus.EXCLUDED, ExclusionReason.EMPTY
    # Enforce a minimum file size (tiny files are often corrupted).
    if file_size < 1024:
        return InclusionStatus.EXCLUDED, ExclusionReason.CORRUPTED
    # Reject near-empty extracted text.
    if sample_text:
        empty_ratio = sample_text.count("\n\n") / max(1, len(sample_text))
        if empty_ratio > max_empty_ratio:
            return InclusionStatus.EXCLUDED, ExclusionReason.EMPTY
    # Basic language check using Spanish markers.
    spanish_markers = ["el", "la", "de", "y", "en", "que", "a", "los", "del"]
    text_lower = sample_text.lower()
    spanish_count = sum(1 for m in spanish_markers if m in text_lower)
    if len(sample_text) > 200 and spanish_count < 3:
        # The content is likely not Spanish.
        return InclusionStatus.EXCLUDED, ExclusionReason.WRONG_LANGUAGE

    return InclusionStatus.INCLUDED, None


def write_manifest_json(path: Path, manifest: DocumentManifest) -> None:
    """Write the manifest to a JSON file."""
    manifest.save(path)
