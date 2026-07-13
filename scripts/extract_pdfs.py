from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from document_manifest import (
    DocumentEntry,
    DocumentManifest,
    classify_doc_type_from_filename,
    classify_doc_type_from_text,
    compute_document_ocr_status,
    determine_inclusion_status,
    extract_pdf_metadata,
    write_manifest_json,
)
from utils import (
    clean_extracted_text,
    default_corpus_dir,
    default_metadata_dir,
    default_pdfs_dir,
    project_root,
    slugify,
    word_count,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    corpus_dir = default_corpus_dir()
    metadata_dir = default_metadata_dir()
    pdfs_dir = default_pdfs_dir()
    parser = argparse.ArgumentParser(description="Extract raw text pages from Spanish obstetrics PDFs.")
    parser.add_argument("--input-dir", type=Path, default=pdfs_dir)
    parser.add_argument("--output", type=Path, default=corpus_dir / "raw_pages.jsonl")
    parser.add_argument("--inventory-output", type=Path, default=metadata_dir / "inventory.json")
    parser.add_argument("--min-fallback-chars", type=int, default=120)
    parser.add_argument("--min-ocr-chars", type=int, default=80)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--pdf",
        type=Path,
        action="append",
        default=[],
        help="Specific PDF to process. Can be passed multiple times. Overrides --input-dir discovery.",
    )
    return parser.parse_args()


def require_extractors() -> Tuple[Any, Optional[Any]]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: pymupdf. Install with `pip install -r requirements.txt`.") from exc

    try:
        import pdfplumber  # type: ignore
    except ImportError:
        pdfplumber = None
    return fitz, pdfplumber


def extract_with_pymupdf(page: Any) -> str:
    blocks = page.get_text("blocks")
    if blocks:
        sorted_blocks = sorted(blocks, key=lambda block: (round(block[1], 1), round(block[0], 1)))
        parts = [str(block[4]).strip() for block in sorted_blocks if len(block) >= 5 and str(block[4]).strip()]
        if parts:
            return "\n\n".join(parts)
    return str(page.get_text("text") or "")


def extract_with_pdfplumber(pdf_path: Path, page_index: int, pdfplumber: Any) -> str:
    if pdfplumber is None:
        return ""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_index >= len(pdf.pages):
                return ""
            return pdf.pages[page_index].extract_text(layout=True) or ""
    except Exception:
        return ""


def extract_pdf(
    pdf_path: Path,
    fitz: Any,
    pdfplumber: Optional[Any],
    min_fallback_chars: int,
    min_ocr_chars: int,
    root: Path,
) -> Tuple[List[Dict[str, Any]], DocumentEntry]:
    rows: List[Dict[str, Any]] = []
    document = fitz.open(str(pdf_path))
    page_count = len(document)
    pdf_id = slugify(pdf_path.name)
    try:
        source_path = str(pdf_path.relative_to(root))
    except ValueError:
        source_path = str(pdf_path.resolve())

    pdf_meta = extract_pdf_metadata(document)
    doc_type = classify_doc_type_from_filename(pdf_path.name)

    fallback_pages = 0
    needs_ocr_pages = 0
    ocr_failed_pages = 0
    text_sample_parts: List[str] = []
    page_ocr_status_list: List[Dict[str, Any]] = []

    for page_index in range(page_count):
        page = document[page_index]
        raw_text = extract_with_pymupdf(page)
        cleaned_for_metrics = clean_extracted_text(raw_text)
        method = "pymupdf"

        if len(cleaned_for_metrics) < min_fallback_chars:
            fallback_text = extract_with_pdfplumber(pdf_path, page_index, pdfplumber)
            fallback_clean = clean_extracted_text(fallback_text)
            if len(fallback_clean) > len(cleaned_for_metrics):
                raw_text = fallback_text
                cleaned_for_metrics = fallback_clean
                method = "pdfplumber"
                fallback_pages += 1

        needs_ocr = len(cleaned_for_metrics) < min_ocr_chars
        if needs_ocr:
            needs_ocr_pages += 1

        # Track per-page OCR status
        page_ocr_status_list.append(
            {
                "page": page_index + 1,  # 1-indexed
                "needs_ocr": needs_ocr,
                "char_count": len(cleaned_for_metrics),
                "status": "needs_ocr" if needs_ocr else "clean",
            }
        )

        # Collect text sample from early pages for duplicate/clinical checks
        if len(" ".join(text_sample_parts)) < 3000 and cleaned_for_metrics.strip():
            text_sample_parts.append(cleaned_for_metrics.strip())

        rows.append(
            {
                "pdf_id": pdf_id,
                "source_pdf": pdf_path.name,
                "source_path": source_path,
                "file_size": pdf_path.stat().st_size,
                "page_count": page_count,
                "page": page_index + 1,
                "text": raw_text,
                "char_count": len(cleaned_for_metrics),
                "word_count": word_count(cleaned_for_metrics),
                "extraction_method": method,
                "needs_ocr": needs_ocr,
                "doc_type": doc_type,
            }
        )
    document.close()

    text_sample = " ".join(text_sample_parts)[:2000]
    if doc_type.value == "unknown" and text_sample.strip():
        inferred_doc_type = classify_doc_type_from_text(text_sample, pdf_path.name)
        if inferred_doc_type.value != "unknown":
            doc_type = inferred_doc_type
            for row in rows:
                row["doc_type"] = doc_type

    inclusion_status, exclusion_reason = determine_inclusion_status(
        doc_type=doc_type,
        page_count=page_count,
        sample_text=text_sample,
        file_size=pdf_path.stat().st_size,
    )
    ocr_status = compute_document_ocr_status(
        page_count=page_count,
        needs_ocr_pages=needs_ocr_pages,
        ocr_failed_pages=ocr_failed_pages,
    )

    entry = DocumentEntry(
        pdf_id=pdf_id,
        source_pdf=pdf_path.name,
        source_path=source_path,
        file_size=pdf_path.stat().st_size,
        page_count=page_count,
        doc_type=doc_type,
        inclusion_status=inclusion_status,
        exclusion_reason=exclusion_reason,
        ocr_status=ocr_status,
        fallback_pages=fallback_pages,
        needs_ocr_pages=needs_ocr_pages,
        ocr_failed_pages=ocr_failed_pages,
        metadata=pdf_meta,
        text_sample=text_sample,
        page_ocr_status=page_ocr_status_list,
    )
    return rows, entry


def main() -> None:
    args = parse_args()
    root = project_root()
    fitz, pdfplumber = require_extractors()
    if args.pdf:
        pdfs = sorted(path.resolve() for path in args.pdf)
    else:
        pdfs = sorted(args.input_dir.rglob("*.pdf") if args.recursive else args.input_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {args.input_dir}")

    all_rows: List[Dict[str, Any]] = []
    manifest = DocumentManifest()
    for pdf_path in tqdm(pdfs, desc="Extracting PDFs"):
        rows, entry = extract_pdf(
            pdf_path=pdf_path,
            fitz=fitz,
            pdfplumber=pdfplumber,
            min_fallback_chars=args.min_fallback_chars,
            min_ocr_chars=args.min_ocr_chars,
            root=root,
        )
        all_rows.extend(rows)
        manifest.add(entry)

    count = write_jsonl(args.output, all_rows)
    write_manifest_json(args.inventory_output, manifest)

    included_count = len(manifest.included())
    excluded_count = len(manifest.excluded())
    print(f"PDFs discovered: {len(manifest.entries)}")
    print(f"  Included: {included_count}")
    print(f"  Excluded: {excluded_count}")
    print(f"Pages extracted: {count}")
    print(f"Saved raw pages to: {args.output}")
    print(f"Saved manifest to: {args.inventory_output}")


if __name__ == "__main__":
    main()
