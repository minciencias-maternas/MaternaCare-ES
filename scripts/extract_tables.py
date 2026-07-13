"""
extract_tables.py
=================
Extract structured tables from obstetrics PDFs using pdfplumber.

Tables are especially relevant in CPGs because they often contain:
- Evidence tables (evidence level, recommendation grade)
- Clinical management algorithm tables
- Dosing tables
- Diagnostic criteria tables

Usage:
    python scripts/extract_tables.py --input-dir pdfs/obstetrics --output artifacts/obstetrics/tables/tables.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from utils import default_reports_dir, default_tables_dir, default_pdfs_dir, project_root, slugify, write_jsonl


# Optimized settings for medical tables (CPGs, protocols).
# Many medical tables rely on explicit lines or text alignment.
TABLE_SETTINGS_MEDICAL = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 2,
    "min_words_horizontal": 1,
    "intersection_tolerance": 3,
    "text_x_tolerance": 5,
    "text_y_tolerance": 5,
}

# Settings for tables without visible lines (text-aligned tables).
TABLE_SETTINGS_TEXT = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 5,
    "join_tolerance": 5,
    "edge_min_length": 3,
    "min_words_vertical": 2,
    "min_words_horizontal": 1,
    "intersection_tolerance": 5,
    "text_x_tolerance": 10,
    "text_y_tolerance": 10,
}

NUMERIC_RE = re.compile(r"\d")
SHORT_TOKEN_RE = re.compile(r"^[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{1,3}$")


def parse_args() -> argparse.Namespace:
    tables_dir = default_tables_dir()
    reports_dir = default_reports_dir()
    parser = argparse.ArgumentParser(
        description="Extract structured tables from Spanish obstetrics PDFs."
    )
    parser.add_argument("--input-dir", type=Path, default=default_pdfs_dir())
    parser.add_argument("--output", type=Path, default=tables_dir / "tables.jsonl")
    parser.add_argument("--report-output", type=Path, default=reports_dir / "table_extraction_report.json")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--pdf",
        type=Path,
        action="append",
        default=[],
        help="Specific PDF to process.",
    )
    parser.add_argument(
        "--strategy",
        choices=["lines", "text", "both"],
        default="lines",
        help="Table detection strategy",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=2,
        help="Minimum rows to consider a valid table",
    )
    parser.add_argument(
        "--min-cols",
        type=int,
        default=2,
        help="Minimum columns to consider a valid table",
    )
    parser.add_argument(
        "--strict-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply stricter heuristics to avoid layout/noise false positives.",
    )
    return parser.parse_args()


def extract_tables_from_page(
    page: Any,
    page_index: int,
    pdf_path: Path,
    strategy: str,
    min_rows: int,
    min_cols: int,
    strict_mode: bool,
) -> List[Dict[str, Any]]:
    """Extract tables from a page using pdfplumber.

    Returns:
        Extracted table records with page, strategy, and quality metadata.
    """
    tables_found: List[Dict[str, Any]] = []

    strategies = []
    if strategy in ("lines", "both"):
        strategies.append(("lines", TABLE_SETTINGS_MEDICAL))
    if strategy in ("text", "both"):
        strategies.append(("text", TABLE_SETTINGS_TEXT))

    def normalize_data(data: List[List[Any]]) -> List[List[str]]:
        if not data:
            return []
        max_cols = max(len(row) for row in data)
        normalized: List[List[str]] = []
        for row in data:
            new_row = [str(cell).strip() if cell is not None else "" for cell in row]
            if len(new_row) < max_cols:
                new_row.extend([""] * (max_cols - len(new_row)))
            normalized.append(new_row)
        return normalized

    def table_quality_metrics(data: List[List[str]]) -> Dict[str, float]:
        cells = [cell for row in data for cell in row]
        non_empty = [c for c in cells if c]
        total = max(1, len(cells))
        non_empty_density = len(non_empty) / total
        numeric_cells = sum(1 for c in non_empty if NUMERIC_RE.search(c))
        numeric_ratio = numeric_cells / max(1, len(non_empty))
        words_per_cell = [len(c.split()) for c in non_empty]
        avg_words_per_cell = sum(words_per_cell) / max(1, len(words_per_cell))
        max_words_cell = max(words_per_cell) if words_per_cell else 0
        short_cells = sum(1 for c in non_empty if SHORT_TOKEN_RE.match(c))
        short_cell_ratio = short_cells / max(1, len(non_empty))
        empty_rows = sum(1 for row in data if not any(row))
        empty_row_ratio = empty_rows / max(1, len(data))
        return {
            "non_empty_density": round(non_empty_density, 4),
            "numeric_ratio": round(numeric_ratio, 4),
            "avg_words_per_cell": round(avg_words_per_cell, 4),
            "max_words_cell": float(max_words_cell),
            "short_cell_ratio": round(short_cell_ratio, 4),
            "empty_row_ratio": round(empty_row_ratio, 4),
        }

    def header_like(first_row: List[str]) -> bool:
        non_empty = [c for c in first_row if c]
        if len(non_empty) < 2:
            return False
        # Headers tend to be concise.
        return all(len(c.split()) <= 6 for c in non_empty)

    def seems_layout_fragment(
        row_count: int,
        col_count: int,
        metrics: Dict[str, float],
    ) -> bool:
        # Large sparse grids coming from multi-column narrative layout.
        if col_count >= 6 and row_count >= 30 and metrics["numeric_ratio"] < 0.08:
            return True
        if metrics["avg_words_per_cell"] > 8 and metrics["numeric_ratio"] < 0.1:
            return True
        if metrics["max_words_cell"] > 45:
            return True
        if metrics["short_cell_ratio"] > 0.55 and metrics["numeric_ratio"] < 0.1:
            return True
        if metrics["non_empty_density"] < 0.35:
            return True
        return False

    for strat_name, settings in strategies:
        try:
            # Use find_tables to access table metadata and cell data.
            table_objects = page.find_tables(table_settings=settings)

            for tbl_idx, table in enumerate(table_objects):
                try:
                    data = table.extract()
                    if not data or len(data) < min_rows:
                        continue

                    data_norm = normalize_data(data)

                    # Ensure the candidate has enough columns.
                    first_row_cols = len(data_norm[0]) if data_norm else 0
                    if first_row_cols < min_cols:
                        continue

                    metrics = table_quality_metrics(data_norm)
                    first_row = data_norm[0] if data_norm else []
                    looks_like_header = header_like(first_row)

                    # Strict heuristics to avoid layout-driven false positives.
                    if strict_mode and seems_layout_fragment(len(data_norm), first_row_cols, metrics):
                        continue

                    # Extra strict when using text strategy because it over-detects narrative pages.
                    if strat_name == "text":
                        informative_signal = (
                            metrics["numeric_ratio"] >= 0.12
                            or (looks_like_header and metrics["avg_words_per_cell"] <= 4.0)
                        )
                        if strict_mode and not informative_signal:
                            continue
                        if strict_mode and metrics["numeric_ratio"] < 0.18:
                            continue

                    # Treat the first row as the header candidate.
                    headers = first_row
                    rows = data_norm[1:] if len(data_norm) > 1 else []
                    # Skip empty or whitespace-only table candidates.
                    non_empty_cells = sum(1 for row in data_norm for cell in row if cell and str(cell).strip())
                    if non_empty_cells < min_rows * min_cols:
                        continue

                    table_record = {
                        "pdf_id": slugify(pdf_path.name),
                        "source_pdf": pdf_path.name,
                        "page": page_index + 1,
                        "table_index": tbl_idx,
                        "strategy": strat_name,
                        "bbox": list(table.bbox) if table.bbox else None,
                        "row_count": len(data),
                        "col_count": first_row_cols,
                        "headers": headers,
                        "rows": rows,
                        "raw_data": data_norm,  # Full backup of normalized cell data.
                        "quality_metrics": metrics,
                        "is_header_like": looks_like_header,
                    }
                    tables_found.append(table_record)

                except Exception as e:
                    # Keep processing if a single table extraction fails.
                    continue

        except Exception as e:
            # Keep processing if one strategy fails.
            continue

    return tables_found


def extract_pdf_tables(
    pdf_path: Path,
    strategy: str,
    min_rows: int,
    min_cols: int,
    strict_mode: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract all tables from a PDF.

    Returns:
        Tuple of (table_records, per_pdf_report).
    """
    import pdfplumber

    all_tables: List[Dict[str, Any]] = []
    pages_with_tables = 0
    total_tables = 0
    strategy_counts: Dict[str, int] = {"lines": 0, "text": 0}
    pages_with_table_numbers: List[int] = []

    def table_signature(table: Dict[str, Any]) -> Tuple[Any, ...]:
        bbox = table.get("bbox") or []
        bbox_key = tuple(round(float(value), 1) for value in bbox) if bbox else ()
        headers = tuple(str(cell or "").strip() for cell in table.get("headers", []))
        first_row = tuple(
            str(cell or "").strip()
            for cell in (table.get("rows", [])[0] if table.get("rows") else [])
        )
        return (
            table.get("page"),
            table.get("row_count"),
            table.get("col_count"),
            bbox_key,
            headers,
            first_row,
        )

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            page_count = len(pdf.pages)

            for page_index, page in enumerate(pdf.pages):
                tables = extract_tables_from_page(
                    page=page,
                    page_index=page_index,
                    pdf_path=pdf_path,
                    strategy=strategy,
                    min_rows=min_rows,
                    min_cols=min_cols,
                    strict_mode=strict_mode,
                )
                if tables:
                    deduped: List[Dict[str, Any]] = []
                    seen_signatures = set()
                    for table in tables:
                        signature = table_signature(table)
                        if signature in seen_signatures:
                            continue
                        seen_signatures.add(signature)
                        deduped.append(table)

                    if deduped:
                        pages_with_tables += 1
                        pages_with_table_numbers.append(page_index + 1)
                        total_tables += len(deduped)
                        for table in deduped:
                            strategy = str(table.get("strategy", "unknown"))
                            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                        all_tables.extend(deduped)

    except Exception as e:
        return [], {
            "pdf_id": slugify(pdf_path.name),
            "source_pdf": pdf_path.name,
            "error": str(e),
            "page_count": 0,
            "pages_with_tables": 0,
            "total_tables": 0,
        }

    report = {
        "pdf_id": slugify(pdf_path.name),
        "source_pdf": pdf_path.name,
        "page_count": page_count,
        "pages_with_tables": pages_with_tables,
        "pages_with_table_numbers": pages_with_table_numbers,
        "total_tables": total_tables,
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "strict_mode": strict_mode,
    }

    return all_tables, report


def main() -> None:
    args = parse_args()
    root = project_root()

    if args.pdf:
        pdfs = sorted(path.resolve() for path in args.pdf)
    else:
        pdfs = sorted(args.input_dir.rglob("*.pdf") if args.recursive else args.input_dir.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {args.input_dir}")

    all_tables: List[Dict[str, Any]] = []
    pdf_reports: List[Dict[str, Any]] = []

    for pdf_path in tqdm(pdfs, desc="Extracting tables"):
        tables, report = extract_pdf_tables(
            pdf_path=pdf_path,
            strategy=args.strategy,
            min_rows=args.min_rows,
            min_cols=args.min_cols,
            strict_mode=args.strict_mode,
        )
        all_tables.extend(tables)
        pdf_reports.append(report)

    # Persist extracted tables.
    count = write_jsonl(args.output, all_tables)
    # Build summary report.
    total_pdfs = len(pdfs)
    pdfs_with_tables = sum(1 for r in pdf_reports if r.get("total_tables", 0) > 0)
    total_tables = sum(r.get("total_tables", 0) for r in pdf_reports)

    report = {
        "total_pdfs": total_pdfs,
        "pdfs_with_tables": pdfs_with_tables,
        "total_tables_extracted": total_tables,
        "tables_written": count,
        "strategy": args.strategy,
        "strict_mode": args.strict_mode,
        "min_rows": args.min_rows,
        "min_cols": args.min_cols,
        "pdf_reports": pdf_reports,
    }

    with open(args.report_output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"PDFs processed: {total_pdfs}")
    print(f"PDFs with tables: {pdfs_with_tables}")
    print(f"Total tables extracted: {total_tables}")
    print(f"Tables written: {count}")
    print(f"Saved tables to: {args.output}")
    print(f"Saved report to: {args.report_output}")


if __name__ == "__main__":
    main()
