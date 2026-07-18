from __future__ import annotations

import json
import math
import random
import re
import unicodedata
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
MULTISPACE_RE = re.compile(r"[ \t]+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
BROKEN_URL_RE = re.compile(r"https?\s*:\s*/\s*/\s*\S+|www\s*\.\s*\S+", re.IGNORECASE)
DOI_RE = re.compile(r"\b(?:doi\s*:?\s*)?10\.\d{4,9}/\S+", re.IGNORECASE)
PHONE_RE = re.compile(
    r"\b(?:tel(?:e?fono)?|tel[eé]fono|fax|lada sin costo|conmutador)\b|"
    r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?){2,}\d{2,4}",
    re.IGNORECASE,
)
PAGE_NUMBER_RE = re.compile(r"^\s*(?:pag(?:ina)?\.?\s*)?\d{1,4}\s*$", re.IGNORECASE)
PAGE_LABEL_RE = re.compile(r"^\s*p\s*[aá]\s*g\s*i\s*n\s*a\s*\|?\s*\d{1,4}\s*$", re.IGNORECASE)
SECTION_NUMBER_RE = re.compile(r"^\s*(?:\d{1,2}(?:\.\d{1,3}){0,4}|[IVXLCDM]{1,8})[\). -]+")
REFERENCE_LINE_RE = re.compile(
    r"\b(?:doi|isbn|issn|pmid|revista|journal|vol\.?|ed\.?|editorial)\b|"
    r"\(\d{4}\)|\b\d{4};\d+|\bet\s+al\b|\bdisponible en\b|\bconsultad[oa]\b|\bcitad[oa]\b",
    re.IGNORECASE,
)
ANSWER_KEY_RE = re.compile(r"\bRESPUESTA\s+CORRECTA\s*:\s*[A-E]\b", re.IGNORECASE)
TOC_DOT_LEADER_RE = re.compile(r"\.{4,}")
ANSWER_WITH_COMMENT_RE = re.compile(
    r"(?:^|\s)\d{1,3}\.\s*RESPUESTA\s+CORRECTA\s*:\s*[A-E]\s*(?:COMENTARIO\s*:)?\s*",
    re.IGNORECASE,
)
ANSWER_PLAIN_RE = re.compile(r"(?:^|\s)RESPUESTA\s+CORRECTA\s*:\s*[A-E]\s*(?:COMENTARIO\s*:)?\s*", re.IGNORECASE)
ENARM_HEADER_RE = re.compile(r"\bRESPUESTAS?\s+ENARM\b", re.IGNORECASE)
ADMIN_LINE_RE = re.compile(
    r"\b(?:isbn|issn|copyright|derechos reservados|agradecimientos?|directorio|"
    r"relacion general de autores|relación general de autores|autores|afiliaci[oó]n|"
    r"correspondencia|primera edici[oó]n|segunda edici[oó]n|editorial|"
    r"comisionado|director(?:a)?|subdirector(?:a)?|grupo de trabajo|"
    r"correo electr[oó]nico|e-?mail|tel[eé]fono|fax|lada sin costo|"
    r"conflicto de inter[eé]s|financiamiento)\b",
    re.IGNORECASE,
)
WEIRD_EXTRACTION_CHARS = set("‡ƒ¦„¥‚┫�")

DROP_SECTION_HEADINGS = {
    "referencias",
    "bibliografia",
    "bibliography",
    "conflicto de interes",
    "conflictos de interes",
    "financiamiento",
    "agradecimientos",
    "autores",
    "author contributions",
    "correspondencia",
}

INDEX_TERMS = {
    "indice",
    "tabla de contenido",
    "contenido",
    "sumario",
}

CLINICAL_TERMS = {
    "aborto",
    "amenaza",
    "anemia",
    "anticoncepcion",
    "atencion",
    "cesarea",
    "contraccion",
    "contracciones",
    "anticonceptivo",
    "anticonceptivos",
    "control prenatal",
    "diagnostico",
    "diabetes gestacional",
    "dilatacion",
    "eclampsia",
    "endometrial",
    "embarazada",
    "embarazo",
    "emergencia obstetrica",
    "feto",
    "fetal",
    "gestacion",
    "gestacional",
    "gestante",
    "ginecologia",
    "ginecologico",
    "hemorragia",
    "hipertension",
    "infeccion",
    "lactancia",
    "magnesio",
    "materna",
    "materno",
    "mortalidad materna",
    "neonato",
    "obstetricia",
    "obstetrico",
    "parto",
    "placenta",
    "preeclampsia",
    "prenatal",
    "posparto",
    "puerperio",
    "recien nacido",
    "riesgo",
    "ruptura",
    "sangrado",
    "sulfato de magnesio",
    "trabajo de parto",
    "tratamiento",
    "trabajo de parto",
    "utero",
    "vaginal",
    "amenorrea",
    "endometrio",
    "infertilidad",
    "menopausia",
    "ovulacion",
    "birth",
    "breastfeeding",
    "cesarean",
    "delivery",
    "fetus",
    "gestational",
    "hemorrhage",
    "hypertension",
    "labor",
    "maternal",
    "newborn",
    "obstetric",
    "obstetrics",
    "placenta",
    "postpartum",
    "pregnancy",
    "pregnant",
    "prenatal",
    "preeclampsia",
    "risk management",
    "vaginal birth",
}

CLINICAL_ACTION_TERMS = {
    "administrar",
    "controlar",
    "derivar",
    "diagnosticar",
    "evaluar",
    "identificar",
    "indicar",
    "manejar",
    "monitorizar",
    "prevenir",
    "realizar",
    "recomendar",
    "tratar",
    "vigilar",
    "administer",
    "assess",
    "diagnose",
    "evaluate",
    "identify",
    "manage",
    "monitor",
    "prevent",
    "recommend",
    "refer",
    "treat",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_pdfs_dir() -> Path:
    return project_root() / "pdfs" / "obstetrics"


def default_artifacts_dir() -> Path:
    return project_root() / "artifacts" / "obstetrics"


def default_corpus_dir() -> Path:
    return default_artifacts_dir() / "corpus"


def default_tables_dir() -> Path:
    return default_artifacts_dir() / "tables"


def default_metadata_dir() -> Path:
    return default_artifacts_dir() / "metadata"


def default_reports_dir() -> Path:
    return default_artifacts_dir() / "reports"


def default_datasets_dir() -> Path:
    return project_root() / "datasets" / "obstetrics"


def default_data_dir() -> Path:
    """Backward-compatible alias for the main artifacts directory."""
    return default_artifacts_dir()


# ---------------------------------------------------------------------------
# Document classification helpers (Phase 1) - Backward Compatibility
# ---------------------------------------------------------------------------

# Import from document_manifest for backward compatibility
try:
    from document_manifest import (
        classify_doc_type_from_text,
        detect_language,
        compute_clinical_density,
    )
except ImportError:
    # Fallback if document_manifest not available
    classify_doc_type_from_text = None  # type: ignore
    detect_language = None  # type: ignore
    compute_clinical_density = None  # type: ignore

DOC_TYPES = ("gpc", "protocol", "manual", "article", "book", "guide", "unknown")


def classify_doc_type_from_filename(filename: str) -> str:
    """Infer document type from filename using keyword heuristics.

    Use this fallback when content text is unavailable. Prefer
    classify_doc_type_from_text() from document_manifest for content-aware
    classification.
    """
    normalized = normalize_for_compare(filename)
    normalized = re.sub(r"[_\-+.]", " ", normalized)
    # Also split camelCase-like transitions (e.g., guiaPracClin -> guia Prac Clin)
    normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if re.search(r"\bgpc\b|\bguia\s+de\s+prac?tica\s+clinica|\bprac?tica\s+clinica\b", normalized):
        return "gpc"
    if re.search(r"\bprotocolo\b|\bprocedimiento\b|\bman\s*proc\s*serv\b", normalized):
        return "protocol"
    if re.search(r"\bmanual\b|\bmanuales\b", normalized):
        return "manual"
    if re.search(
        r"\brev(ista)?\b|\brechog\b|\barticle\b|\bjournal\b|\bs\d{7,}[\dx]\b|\bvol\s*\d+\b|\bno\.?\s*\d+|\bstamped\b",
        normalized,
    ):
        return "article"
    if re.search(
        r"\bbook\b|\blibro\b|\bcap[ií]tulo\b|\bcap\s*\d+\b|\btomo\b|"
        r"\bginecologia\b.*\bobstetricia\b|\bobstetricia\b.*\bginecologia\b",
        normalized,
    ):
        return "book"
    if re.search(
        r"\bgu[ií]a\b|\bguide\b|\brecomendaciones\b|\brecom\b|\bgap\b|"
        r"\bembarazo\b.*\bparto\b|\bparto\b.*\bembarazo\b",
        normalized,
    ):
        return "guide"
    return "unknown"


def load_inventory_manifest(path: Path) -> Dict[str, Any]:
    """Load inventory.json manifest and return as dict."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_excluded_pdf_ids(manifest: Dict[str, Any]) -> set[str]:
    """Return set of pdf_ids that are excluded from the manifest."""
    excluded: set[str] = set()
    for entry in manifest.get("pdfs", []):
        if entry.get("inclusion_status") in {"excluded", "excluded_with_reason"}:
            excluded.add(str(entry.get("pdf_id", "")))
    return excluded


def normalize_for_compare(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slugify(value: str) -> str:
    normalized = normalize_for_compare(Path(value).stem)
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return slug or "document"


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(word_count(text) * 1.35))


def clean_extracted_text(text: Any) -> str:
    if text is None:
        return ""
    value = str(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\u2028", "\n").replace("\u2029", "\n")
    value = value.replace("\ufeff", "").replace("\u200b", "")
    value = CONTROL_CHARS_RE.sub("", value)
    value = BROKEN_URL_RE.sub("", value)
    value = URL_RE.sub("", value)
    value = DOI_RE.sub("", value)
    value = EMAIL_RE.sub("", value)
    value = re.sub(r"([A-Za-zÁÉÍÓÚÜÑáéíóúüñ])- *\n *([A-Za-zÁÉÍÓÚÜÑáéíóúüñ])", r"\1\2", value)
    value = re.sub(r"[\u2022\u25e6\u25aa\u00b7]", "- ", value)
    value = re.sub(r"[ \t]+", " ", value)

    cleaned_lines: List[str] = []
    for line in value.split("\n"):
        line = MULTISPACE_RE.sub(" ", line).strip()
        if not line:
            cleaned_lines.append("")
            continue
        if PAGE_NUMBER_RE.match(line) or PAGE_LABEL_RE.match(line):
            continue
        if is_noise_line(line):
            continue
        cleaned_lines.append(line)

    value = "\n".join(cleaned_lines)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def normalize_lm_text(text: Any) -> str:
    return re.sub(r"\s+", " ", clean_extracted_text(text)).strip()


def collapse_whitespace(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    normalized = normalize_for_compare(stripped)
    if normalized in DROP_SECTION_HEADINGS:
        return True
    if BROKEN_URL_RE.search(stripped) or URL_RE.search(stripped) or EMAIL_RE.search(stripped) or DOI_RE.search(stripped):
        return True
    if PHONE_RE.search(stripped):
        return True
    if ADMIN_LINE_RE.search(stripped):
        return True
    if re.match(r"^\s*\d{1,3}\.\s+", stripped) and REFERENCE_LINE_RE.search(stripped):
        return True
    if re.match(r"^\s*[A-ZÁÉÍÓÚÜÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ' -]{2,},\s*[A-ZÁÉÍÓÚÜÑ]", stripped):
        if len(stripped.split()) > 5 and not any(term in normalized for term in CLINICAL_TERMS):
            return True
    return False


def line_signature(line: str) -> str:
    signature = normalize_for_compare(line)
    if len(signature) < 4 or signature.isdigit():
        return ""
    return signature


def find_repeated_lines(rows: Sequence[Dict[str, Any]], threshold_ratio: float = 0.28) -> Dict[str, List[str]]:
    by_pdf: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pdf[str(row.get("source_pdf", ""))].append(row)

    repeated: Dict[str, List[str]] = {}
    for source_pdf, pdf_rows in by_pdf.items():
        page_total = max(1, len(pdf_rows))
        line_pages: Dict[str, set] = defaultdict(set)
        display_by_signature: Dict[str, str] = {}
        for row in pdf_rows:
            page = int(row.get("page", 0))
            for line in str(row.get("text", "")).splitlines():
                stripped = line.strip()
                sig = line_signature(stripped)
                if not sig or len(stripped) > 120:
                    continue
                line_pages[sig].add(page)
                display_by_signature.setdefault(sig, stripped)

        min_pages = max(3, math.ceil(page_total * threshold_ratio))
        repeated[source_pdf] = sorted(
            display_by_signature[sig]
            for sig, pages in line_pages.items()
            if len(pages) >= min_pages
        )
    return repeated


def remove_repeated_lines(text: str, repeated_lines: Sequence[str]) -> str:
    signatures = {line_signature(line) for line in repeated_lines}
    signatures.discard("")
    kept = []
    for line in text.splitlines():
        sig = line_signature(line)
        if sig and sig in signatures:
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def is_heading(line: str) -> bool:
    stripped = line.strip(" :-")
    if not stripped or len(stripped) > 110 or len(stripped.split()) > 14:
        return False
    normalized = normalize_for_compare(stripped)
    if normalized in DROP_SECTION_HEADINGS or normalized in INDEX_TERMS:
        return True
    if re.match(r"^(capitulo|cap|tema|seccion|anexo)\b", normalized):
        return True
    if SECTION_NUMBER_RE.match(stripped) and len(stripped.split()) <= 12:
        return True

    letters = [ch for ch in stripped if ch.isalpha()]
    if len(letters) >= 6:
        uppercase = sum(1 for ch in letters if ch.upper() == ch)
        if uppercase / len(letters) >= 0.72:
            return True
    return False


def extract_page_section(text: str, previous_section: str = "") -> str:
    for line in text.splitlines()[:10]:
        if is_heading(line):
            heading = line.strip(" :-")
            normalized = normalize_for_compare(heading)
            if normalized not in INDEX_TERMS:
                return heading
    return previous_section


def reference_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    reference_lines = sum(1 for line in lines if REFERENCE_LINE_RE.search(line))
    return reference_lines / len(lines)


def reference_marker_count(text: str) -> int:
    markers = 0
    markers += len(REFERENCE_LINE_RE.findall(text))
    markers += len(URL_RE.findall(text))
    markers += len(BROKEN_URL_RE.findall(text))
    markers += len(DOI_RE.findall(text))
    markers += len(re.findall(r"\b(?:referencias|bibliografia|bibliografía)\b", text, flags=re.IGNORECASE))
    return markers


def suspicious_extraction_ratio(text: str) -> float:
    if not text:
        return 0.0
    weird = sum(1 for ch in text if ch in WEIRD_EXTRACTION_CHARS)
    excessive_quotes = max(0, text.count('"') - 4)
    return (weird + excessive_quotes) / max(1, len(text))


def numeric_token_ratio(text: str) -> float:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return 0.0
    numeric = sum(1 for token in tokens if re.search(r"\d", token))
    return numeric / len(tokens)


def decimal_number_count(text: str) -> int:
    return len(re.findall(r"\b\d+[.,]\d+\b", text))


def clinical_score(text: str) -> int:
    normalized = normalize_for_compare(text)
    score = 0
    for term in CLINICAL_TERMS:
        if term in normalized:
            score += 2 if " " in term else 1
    for term in CLINICAL_ACTION_TERMS:
        if term in normalized:
            score += 1
    words = word_count(text)
    if words >= 120:
        score += 2
    if words >= 350:
        score += 2
    if reference_line_ratio(text) >= 0.35:
        score -= 3
    return max(0, score)


def classify_page(text: str, page_number: int) -> Tuple[bool, str, Dict[str, Any]]:
    cleaned = clean_extracted_text(text)
    words = word_count(cleaned)
    chars = len(cleaned)
    normalized = normalize_for_compare(cleaned[:2500])
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    metrics = {
        "char_count": chars,
        "word_count": words,
        "line_count": len(lines),
        "reference_line_ratio": reference_line_ratio(cleaned),
        "clinical_score": clinical_score(cleaned),
        "suspicious_char_ratio": suspicious_extraction_ratio(cleaned),
        "reference_marker_count": reference_marker_count(cleaned),
    }

    if chars < 180 or words < 25:
        return False, "too_short", metrics

    if metrics["suspicious_char_ratio"] >= 0.018:
        return False, "corrupt_extraction", metrics

    first_page_noise = page_number <= 2 and any(term in normalized for term in ("isbn", "copyright", "editorial"))
    if first_page_noise and metrics["clinical_score"] < 3:
        return False, "cover_or_credits", metrics

    first_lines = [normalize_for_compare(line) for line in lines[:15]]
    if any(line in DROP_SECTION_HEADINGS for line in first_lines):
        return False, "non_clinical_section", metrics

    if any(term in normalized[:800] for term in INDEX_TERMS):
        dot_leaders = sum(1 for line in lines if re.search(r"\.{3,}\s*\d+$", line))
        if dot_leaders >= 3 or metrics["clinical_score"] < 4:
            return False, "index_or_table_of_contents", metrics

    if any(normalized.startswith(term) for term in DROP_SECTION_HEADINGS):
        return False, "non_clinical_section", metrics

    if metrics["reference_line_ratio"] >= 0.45 and metrics["clinical_score"] < 6:
        return False, "reference_heavy", metrics

    if metrics["reference_marker_count"] >= 5 and metrics["clinical_score"] < 10:
        return False, "reference_heavy", metrics

    short_lines = sum(1 for line in lines if len(line) <= 28)
    if len(lines) >= 12 and short_lines / len(lines) >= 0.72 and metrics["clinical_score"] < 5:
        return False, "fragmented_text", metrics

    return True, "", metrics


def quality_flags(text: str, token_estimate: int, score: int) -> List[str]:
    flags: List[str] = []
    if token_estimate < 180:
        flags.append("short_chunk")
    if token_estimate > 1500:
        flags.append("long_chunk")
    if score < 5:
        flags.append("low_clinical_score")
    if reference_line_ratio(text) >= 0.3:
        flags.append("reference_heavy")
    if reference_marker_count(text) >= 4:
        flags.append("reference_markers")
    if suspicious_extraction_ratio(text) >= 0.012:
        flags.append("corrupt_extraction")
    if numeric_token_ratio(text) >= 0.38:
        flags.append("numeric_or_table_heavy")
    if decimal_number_count(text) >= 12:
        flags.append("decimal_table_like")
    if len(set(normalize_for_compare(text).split())) < 35:
        flags.append("low_vocabulary_variety")
    return flags


def split_paragraphs(text: str) -> List[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blocks) > 1:
        return blocks

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    paragraphs: List[str] = []
    current: List[str] = []
    for line in lines:
        if is_heading(line) and current:
            paragraphs.append(" ".join(current).strip())
            current = [line]
            continue
        current.append(line)
        if len(" ".join(current).split()) >= 120:
            paragraphs.append(" ".join(current).strip())
            current = []
    if current:
        paragraphs.append(" ".join(current).strip())
    return paragraphs


def tail_words(text: str, n: int) -> str:
    words = text.split()
    if not words:
        return ""
    return " ".join(words[-n:])


def tail_tokens(text: str, n: int, tokenizer) -> str:
    """Return the last n real tokenizer tokens from text as a string.

    Tokenizes the input, keeps the last n token ids, and decodes them back
    to text. Handles empty input and non-positive n by returning an empty
    string.

    Args:
        text: Source text to tokenize.
        n: Number of tokens to keep from the end.
        tokenizer: A tokenizer with ``encode`` and ``decode`` methods.

    Returns:
        Decoded string of the last n tokens, or an empty string if no content.
    """
    if not text or n <= 0 or tokenizer is None:
        return ""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids[-n:], skip_special_tokens=True)


def chunk_records(
    rows: Sequence[Dict[str, Any]],
    min_tokens: int = 500,
    max_tokens: int = 1200,
    overlap_tokens: int = 100,
    tokenizer: Any = None,
) -> List[Dict[str, Any]]:
    if overlap_tokens > 0 and tokenizer is None:
        warnings.warn(
            "No tokenizer provided; overlap_tokens is approximate and measured in "
            "whitespace-separated words, not real tokenizer tokens.",
            UserWarning,
            stacklevel=2,
        )
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("is_kept") is False:
            continue
        key = (str(row.get("source_pdf", "")), str(row.get("section", "") or "Sin seccion"))
        grouped[key].append(row)

    chunks: List[Dict[str, Any]] = []
    for (source_pdf, section), group in grouped.items():
        group = sorted(group, key=lambda r: int(r.get("page", 0)))
        source_path = str(group[0].get("source_path", "")) if group else ""
        doc_type = str(group[0].get("doc_type", "unknown")) if group else "unknown"
        buffer_parts: List[str] = []
        buffer_pages: List[int] = []

        def flush(force: bool = False) -> None:
            nonlocal buffer_parts, buffer_pages
            text = normalize_lm_text(" ".join(buffer_parts))
            tokens = estimate_tokens(text)
            if not text:
                buffer_parts, buffer_pages = [], []
                return
            if force or tokens >= min_tokens:
                score = clinical_score(text)
                chunks.append(
                    {
                        "text": text,
                        "source_pdf": source_pdf,
                        "source_path": source_path,
                        "doc_type": doc_type,
                        "pages": sorted(set(buffer_pages)),
                        "section": section,
                        "token_estimate": tokens,
                        "clinical_score": score,
                        "quality_flags": quality_flags(text, tokens, score),
                    }
                )
                if overlap_tokens > 0 and tokens > max_tokens:
                    if tokenizer is not None:
                        overlap = tail_tokens(text, overlap_tokens, tokenizer)
                    else:
                        overlap = tail_words(text, overlap_tokens)
                    buffer_parts = [overlap] if overlap else []
                    buffer_pages = buffer_pages[-1:] if buffer_pages else []
                else:
                    buffer_parts, buffer_pages = [], []

        for row in group:
            page_text = str(row.get("text", "")).strip()
            if not page_text:
                continue
            for paragraph in split_paragraphs(page_text):
                if not paragraph:
                    continue
                candidate = normalize_lm_text(" ".join(buffer_parts + [paragraph]))
                if estimate_tokens(candidate) > max_tokens and buffer_parts:
                    flush(force=True)
                buffer_parts.append(paragraph)
                page = int(row.get("page", 0))
                if page:
                    buffer_pages.append(page)
            if estimate_tokens("\n\n".join(buffer_parts)) >= max_tokens:
                flush(force=True)
        flush(force=True)
    return chunks


def dedupe_chunks(chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_exact = set()
    seen_near = set()
    deduped: List[Dict[str, Any]] = []
    ordered = sorted(
        chunks,
        key=lambda row: (
            -int(row.get("clinical_score", 0)),
            -int(row.get("token_estimate", 0)),
            str(row.get("source_pdf", "")),
        ),
    )
    for row in ordered:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        exact_key = text
        near_key = " ".join(normalize_for_compare(text).split()[:240])
        if exact_key in seen_exact or near_key in seen_near:
            continue
        seen_exact.add(exact_key)
        seen_near.add(near_key)
        deduped.append(row)
    return sorted(deduped, key=lambda r: (str(r.get("source_pdf", "")), min(r.get("pages", [0]) or [0])))


def accepted_for_lm(row: Dict[str, Any], min_tokens: int = 180, min_score: int = 4) -> Tuple[bool, str]:
    tokens = int(row.get("token_estimate", 0))
    score = int(row.get("clinical_score", 0))
    text = str(row.get("text", ""))
    cleaned_text = strip_lm_noise(text)
    cleaned_tokens = estimate_tokens(cleaned_text)
    if cleaned_tokens < min_tokens:
        return False, "too_short_after_noise_cleanup"
    if tokens < min_tokens:
        return False, "too_short"
    if score < min_score:
        return False, "low_clinical_score"
    if reference_line_ratio(text) >= 0.45:
        return False, "reference_heavy"
    if reference_marker_count(text) >= 4:
        return False, "reference_heavy"
    if suspicious_extraction_ratio(text) >= 0.012:
        return False, "corrupt_extraction"
    normalized = normalize_for_compare(text)
    admin_terms = (
        "agradecimiento",
        "agradece",
        "directorio",
        "relacion general de autores",
        "telefono",
        "fax",
        "isbn",
        "issn",
        "conflicto de interes",
        "propiedad intelectual",
        "reproduccion comercial",
        "ministerio del poder popular",
        "comision nacional de arbitraje medico",
        "comisionado nacional",
        "proceso de validacion",
        "oficializacion",
        "formato forms",
        "metodologia anteriormente descrita",
        "viceministra",
        "viceministro",
        "ninguna parte de este libro",
        "puede ser reproducido",
        "diseno de portada",
        "diseño de portada",
        "derechos reservados",
    )
    if any(term in normalized for term in admin_terms):
        return False, "admin_or_contact_text"
    if numeric_token_ratio(text) >= 0.38 and score < 9:
        return False, "numeric_or_table_heavy"
    if decimal_number_count(text) >= 12 and score < 9:
        return False, "numeric_or_table_heavy"
    if "role" in row or "messages" in row:
        return False, "qa_like_record"
    if ANSWER_KEY_RE.search(cleaned_text):
        return False, "exam_answer_key"
    if TOC_DOT_LEADER_RE.search(text) and score < 10:
        return False, "toc_or_layout_noise"
    return True, ""


def assign_chunk_ids(chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counters: Counter[str] = Counter()
    assigned: List[Dict[str, Any]] = []
    for row in chunks:
        source_pdf = str(row.get("source_pdf", "document"))
        pdf_id = slugify(source_pdf)
        counters[pdf_id] += 1
        new_row = dict(row)
        new_row["chunk_id"] = f"{pdf_id}_{counters[pdf_id]:05d}"
        assigned.append(new_row)
    return assigned


def split_train_validation(
    chunks: Sequence[Dict[str, Any]],
    validation_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_pdf: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in chunks:
        by_pdf[str(row.get("source_pdf", ""))].append(row)

    rng = random.Random(seed)
    train: List[Dict[str, Any]] = []
    validation: List[Dict[str, Any]] = []
    for _, rows in sorted(by_pdf.items()):
        ordered = sorted(rows, key=lambda r: min(r.get("pages", [0]) or [0]))
        if len(ordered) <= 2:
            train.extend(ordered)
            continue
        block_size = max(1, round(1 / validation_ratio))
        offset = rng.randrange(block_size)
        for idx, row in enumerate(ordered):
            if (idx + offset) % block_size == 0:
                validation.append(row)
            else:
                train.append(row)
    return train, validation


def to_lm_record(chunk: Dict[str, Any]) -> Dict[str, Any]:
    topics = list(chunk.get("topics") or chunk.get("topic_tags") or [])
    metadata = {
        "source": "obstetrics_spanish",
        "pdf_id": chunk.get("pdf_id", ""),
        "source_pdf": chunk.get("source_pdf"),
        "source_path": chunk.get("source_path", ""),
        "doc_type": chunk.get("doc_type", "unknown"),
        "pages": chunk.get("pages", []),
        "section": chunk.get("section", ""),
        "chunk_id": chunk.get("chunk_id", ""),
        "token_estimate": chunk.get("token_estimate", 0),
        "clinical_score": chunk.get("clinical_score", 0),
        "section_type": chunk.get("section_type", "unknown"),
        "content_role": chunk.get("content_role", "unknown"),
        "language": chunk.get("language", "unknown"),
        "topics": topics,
        "topic_tags": topics,
        "topic_count": chunk.get("topic_count", len(topics)),
        "split": chunk.get("split", ""),
    }
    cleaned_text = strip_lm_noise(str(chunk.get("text", "")))
    return {"text": collapse_whitespace(cleaned_text), "metadata": metadata}


def strip_lm_noise(text: str) -> str:
    """Remove localized exam/index boilerplate while preserving clinical content."""
    value = str(text or "")
    value = ENARM_HEADER_RE.sub(" ", value)
    value = ANSWER_WITH_COMMENT_RE.sub(" ", value)
    value = ANSWER_PLAIN_RE.sub(" ", value)

    cleaned_lines: List[str] = []
    for line in value.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        # Remove table-of-contents leaders made of headings, dot runs, and page numbers.
        if TOC_DOT_LEADER_RE.search(line_stripped):
            if len(line_stripped) >= 40:
                continue
            dense_text = re.sub(r"[.\d\s]", "", line_stripped)
            if len(dense_text) < 40:
                continue
        cleaned_lines.append(line_stripped)

    return "\n".join(cleaned_lines)


# ---------------------------------------------------------------------------
# Phase 3: Section Classification (section_type, content_role)
# ---------------------------------------------------------------------------

SECTION_TYPE_KEYWORDS = {
    "index": ["indice", "tabla de contenido", "contenido", "sumario", "lista de figuras", "lista de tablas"],
    "introduction": ["introduccion", "introducción", "presentacion", "presentación", "antecedentes", "justificacion", "justificación"],
    "methodology": ["metodos", "métodos", "metodologia", "metodología", "diseno", "diseño", "muestra", "estadistica", "estadística", "analisis", "análisis"],
    "clinical_content": ["resultados", "discusion", "discusión", "hallazgos", "caso clinico", "caso clínico", "presentacion del caso"],
    "recommendations": ["recomendaciones", "conclusiones", "sintesis", "síntesis", "resumen ejecutivo", "implicaciones"],
    "references": ["referencias", "bibliografia", "bibliografía", "bibliographic", "references"],
    "appendix": ["anexo", "apendice", "apéndice", "anexos", "apendices", "apéndices", "material suplementario"],
    "glossary": ["glosario", "abreviaturas", "siglas", "terminos", "términos", "definiciones"],
}

CONTENT_ROLE_KEYWORDS = {
    "navigation": ["indice", "tabla de contenido", "sumario", "pagina", "página", "capitulo", "capítulo"],
    "background": ["introduccion", "introducción", "antecedentes", "historia", "epidemiologia", "epidemiología", "contexto"],
    "evidence": ["evidencia", "estudio", "estudios", "metaanalisis", "metaanálisis", "revision sistematica", "revisión sistemática", "ensayo clinico", "ensayo clínico", "cohorte", "caso control"],
    "recommendation": ["recomendacion", "recomendación", "recomendaciones", "sugerencia", "sugerencias", "indicacion", "indicación", "contraindicacion", "contraindicación", "grado a", "grado b", "grado c", "nivel i", "nivel ii", "nivel iii"],
    "procedure": ["procedimiento", "tecnica", "técnica", "protocolo", "algoritmo", "pasos", "manejo", "tratamiento", "intervencion", "intervención", "cirugia", "cirugía"],
    "diagnostic": ["diagnostico", "diagnóstico", "diagnostico diferencial", "criterios", "clasificacion", "clasificación", "estadificacion", "estadificación", "tamizaje", "screening", "prueba", "pruebas"],
    "treatment": ["tratamiento", "terapia", "farmacoterapia", "medicamento", "medicamentos", "dosis", "posologia", "posología", "administracion", "administración", "efectos adversos"],
    "administrative": ["agradecimiento", "financiamiento", "conflicto de interes", "autores", "correspondencia", "derechos reservados", "isbn", "issn", "copyright"],
}

SECTION_TYPE_TO_CONTENT_ROLE = {
    "index": "navigation",
    "introduction": "background",
    "methodology": "evidence",
    "clinical_content": "evidence",
    "recommendations": "recommendation",
    "references": "administrative",
    "appendix": "background",
    "glossary": "background",
}


def classify_section_type(section_heading: str, text_sample: str = "") -> str:
    """Classify a section from its heading first, then from its opening text."""
    normalized_heading = normalize_for_compare(section_heading)
    normalized_text = normalize_for_compare(text_sample[:1500])

    # Headings are the strongest signal because they usually state the section role directly.
    for section_type, keywords in SECTION_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized_heading:
                return section_type

    # Fall back to the first paragraphs; later text can drift into unrelated subtopics.
    for section_type, keywords in SECTION_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized_text[:500]:
                return section_type

    # A high clinical score is still useful when headings are missing or noisy.
    if clinical_score(text_sample[:1000]) >= 5:
        return "clinical_content"

    return "unknown"


def classify_content_role(text: str, section_type: str = "") -> str:
    """Classify the content role, such as evidence, recommendation, or procedure."""
    normalized = normalize_for_compare(text[:2000])
    default_role = SECTION_TYPE_TO_CONTENT_ROLE.get(section_type, "")

    # Score role-specific keywords instead of stopping at the first match; medical
    # sections often contain several overlapping concepts.
    scores: Dict[str, int] = {role: 0 for role in CONTENT_ROLE_KEYWORDS}
    for role, keywords in CONTENT_ROLE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized:
                scores[role] += 1

    if default_role:
        scores[default_role] += 2

    # Require a clear winner so a single incidental keyword does not overclassify text.
    if scores:
        best_role = max(scores, key=scores.get)
        best_score = scores[best_role]
        if best_score >= 2 or (default_role and best_role == default_role and best_score >= 1):
            return best_role

    # Use clinical density as a final fallback when keyword evidence is weak.
    score = clinical_score(text)
    if score >= 8:
        return "evidence"
    elif score >= 4:
        return "background"

    return "unknown"


# ---------------------------------------------------------------------------
# Phase 4: Topic Tagging
# ---------------------------------------------------------------------------

TOPIC_TAXONOMY = {
    "hemorrhage": ["hemorragia", "sangrado", "sangrar", "perdida de sangre", "hematoma", "hematocrito", "transfusion"],
    "preeclampsia": ["preeclampsia", "eclampsia", "hipertension gestacional", "hipertensión gestacional", "proteinuria", "edema", "convulsion", "convulsión", "magnesio"],
    "diabetes_gestational": ["diabetes gestacional", "glucosa", "hipoglucemia", "hiperglucemia", "insulina", "glicemia", "glicosilada", "hgba1c", "macrosomia"],
    "preterm_labor": ["parto prematuro", "trabajo de parto pretermino", "contracciones prematuras", "amenaza de parto prematuro", "tocolisis", "corticoides", "madurez fetal"],
    "infection": ["infeccion", "infección", "sepsis", "corioamnionitis", "endometritis", "fiebre", "puerperal", "antibiotico", "antibiótico", "infeccion puerperal"],
    "cesarean": ["cesarea", "cesárea", "cesarean", "csection", "c section", "operacion cesarea", "parto cesareo"],
    "vaginal_birth": ["parto vaginal", "parto normal", "parto eutocico", "distocia", "prolapso", "episiotomia", "episiotomía", "forceps", "ventosa"],
    "newborn_care": ["recien nacido", "recién nacido", "neonato", "neonatal", "apgar", "reanimacion neonatal", "reanimación neonatal", "hipoxia", "ictericia"],
    "contraception": ["anticoncepcion", "anticoncepción", "planificacion familiar", "metodos anticonceptivos", "diu", "pildora", "pastilla", "esterilizacion", "vasectomia"],
    "menopause": ["menopausia", "climaterio", "terapia hormonal", "sustitucion hormonal", "soja", "isoflavonas", "osteoporosis"],
    "infertility": ["infertilidad", "esterilidad", "fecundacion", "fecundación", "fiv", "fertilizacion in vitro", "espermatozoide", "ovocito", "ovulacion", "ovulación"],
    "prenatal_care": ["control prenatal", "consulta prenatal", "primer trimestre", "segundo trimestre", "tercer trimestre", "ecografia", "ecografía", "screening prenatal", "tamizaje"],
    "postpartum": ["puerperio", "posparto", "lactancia", "mastitis", "depresion postparto", "puerperio inmediato"],
    "fetal_monitoring": ["feto", "fetal", "monitoreo fetal", "cardiotocografia", "cardiotocografía", "ritmo fetal", "frecuencia cardiaca fetal", "biometria", "biometría"],
    "labor_induction": ["induccion", "inducción", "maduracion cervical", "maduración cervical", "prostaglandinas", "oxitocina", "amniotomia", "amniotomía", "desgarro membranas"],
    "gynecologic_oncology": ["cancer ginecologico", "cáncer ginecológico", "tumor", "neoplasia", "biopsia", "histologia", "histología", "cervix", "cervical", "vph", "hpv"],
    "ultrasound": ["ultrasonido", "ecografia", "ecografía", "doppler", "transvaginal", "morfologica", "morfológica", "nuchal translucency"],
    "genetics": ["genetica", "genética", "cariotipo", "mutacion", "mutación", "creening genetico", "screening genético", "amniocentesis", "villosidad corial"],
}


def tag_topics(text: str) -> List[str]:
    """Tag a text with relevant clinical topics from the taxonomy."""
    normalized = normalize_for_compare(text)
    tags: List[str] = []
    for topic, keywords in TOPIC_TAXONOMY.items():
        for keyword in keywords:
            if keyword in normalized:
                tags.append(topic)
                break  # Only add topic once
    return tags


# ---------------------------------------------------------------------------
# Phase 5: Document-Level Deduplication
# ---------------------------------------------------------------------------

def compute_document_fingerprint(text: str) -> str:
    """Compute a fingerprint for deduplication based on content."""
    normalized = normalize_for_compare(text)
    # Take first 100 "significant" words as fingerprint
    words = [w for w in normalized.split() if len(w) > 3]
    return " ".join(words[:100])


def find_duplicate_documents(
    entries: Sequence[Dict[str, Any]],
    similarity_threshold: float = 0.85,
) -> List[Tuple[str, str, float]]:
    """Find potential duplicate documents based on content similarity.

    Returns:
        List of (pdf_id_a, pdf_id_b, similarity) tuples.
    """
    from difflib import SequenceMatcher

    # Build fingerprints
    fingerprints: Dict[str, str] = {}
    for entry in entries:
        pdf_id = str(entry.get("pdf_id", ""))
        text_sample = str(entry.get("text_sample", ""))
        if pdf_id and text_sample:
            fingerprints[pdf_id] = compute_document_fingerprint(text_sample)

    duplicates: List[Tuple[str, str, float]] = []
    pdf_ids = sorted(fingerprints.keys())

    for i, pdf_a in enumerate(pdf_ids):
        fp_a = fingerprints[pdf_a]
        for pdf_b in pdf_ids[i + 1:]:
            fp_b = fingerprints[pdf_b]
            similarity = SequenceMatcher(None, fp_a, fp_b).ratio()
            if similarity >= similarity_threshold:
                duplicates.append((pdf_a, pdf_b, round(similarity, 4)))

    return duplicates


# ---------------------------------------------------------------------------
# Phase 6: Document-Level Train/Validation Split (NO LEAKAGE)
# ---------------------------------------------------------------------------

def _select_holdout_documents(
    by_pdf: Dict[str, List[Dict[str, Any]]],
    candidates: Sequence[str],
    target_chunks: int,
    rng: random.Random,
    min_docs: int = 1,
    max_single_pdf_ratio: float = 0.75,
) -> set[str]:
    """Choose whole documents whose chunk count approximates a target size."""
    if target_chunks <= 0 or not candidates:
        return set()

    remaining = list(candidates)
    rng.shuffle(remaining)
    selected: set[str] = set()
    selected_chunks = 0
    max_single_pdf_chunks = max(1, round(target_chunks * max_single_pdf_ratio))

    while remaining:
        preferred_pool = remaining
        if not selected:
            smaller_candidates = [
                pdf for pdf in remaining
                if len(by_pdf[pdf]) <= max_single_pdf_chunks
            ]
            if smaller_candidates:
                preferred_pool = smaller_candidates

        current_gap = abs(target_chunks - selected_chunks)
        best_pdf = min(
            preferred_pool,
            key=lambda pdf: (
                abs(target_chunks - (selected_chunks + len(by_pdf[pdf]))),
                len(by_pdf[pdf]),
                pdf,
            ),
        )
        next_gap = abs(target_chunks - (selected_chunks + len(by_pdf[best_pdf])))
        if len(selected) >= min_docs and selected and next_gap >= current_gap:
            break
        selected.add(best_pdf)
        selected_chunks += len(by_pdf[best_pdf])
        remaining.remove(best_pdf)

    if not selected:
        selected.add(min(candidates, key=lambda pdf: (len(by_pdf[pdf]), pdf)))
    while len(selected) < min_docs and remaining:
        next_pdf = min(
            remaining,
            key=lambda pdf: (
                len(by_pdf[pdf]),
                abs(target_chunks - (selected_chunks + len(by_pdf[pdf]))),
                pdf,
            ),
        )
        selected.add(next_pdf)
        selected_chunks += len(by_pdf[next_pdf])
        remaining.remove(next_pdf)
    return selected


def split_train_validation_test_by_document(
    chunks: Sequence[Dict[str, Any]],
    validation_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    stratify_by: str = "doc_type",
    min_validation_pdfs: int = 3,
    min_test_pdfs: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split chunks by document while approximating target chunk volume.

    Keeping every PDF in exactly one split prevents leakage. Ratios target
    exported chunk volume rather than document count so a few large PDFs do not
    dominate validation or test.

    Args:
        chunks: Chunks to split.
        validation_ratio: Approximate chunk-volume fraction for validation.
        test_ratio: Approximate chunk-volume fraction for test.
        seed: Random seed for reproducibility.
        stratify_by: Kept for API compatibility and report semantics.

    Returns:
        Tuple of (train_chunks, validation_chunks, test_chunks).
    """
    if validation_ratio < 0 or test_ratio < 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be >= 0 and sum to < 1")

    rng = random.Random(seed)

    # Group chunks by document
    by_pdf: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in chunks:
        document_key = str(row.get("pdf_id") or row.get("source_pdf", ""))
        by_pdf[document_key].append(row)

    pdf_ids = sorted(by_pdf)
    total_chunks = len(chunks)
    validation_target = round(total_chunks * validation_ratio)
    test_target = round(total_chunks * test_ratio)

    validation_pdfs = _select_holdout_documents(
        by_pdf,
        pdf_ids,
        validation_target,
        rng,
        min_docs=min_validation_pdfs,
    )
    remaining_pdf_ids = [pdf for pdf in pdf_ids if pdf not in validation_pdfs]
    test_pdfs = _select_holdout_documents(
        by_pdf,
        remaining_pdf_ids,
        test_target,
        rng,
        min_docs=min_test_pdfs,
    )
    train_pdfs = set(pdf_ids) - validation_pdfs - test_pdfs

    # Assign chunks based on PDF membership
    train: List[Dict[str, Any]] = []
    validation: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for pdf_name, pdf_chunks in by_pdf.items():
        if pdf_name in validation_pdfs:
            validation.extend(pdf_chunks)
        elif pdf_name in test_pdfs:
            test.extend(pdf_chunks)
        else:
            train.extend(pdf_chunks)

    return train, validation, test


def split_train_validation_by_document(
    chunks: Sequence[Dict[str, Any]],
    validation_ratio: float = 0.10,
    seed: int = 42,
    stratify_by: str = "doc_type",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Backward-compatible wrapper around the chunk-balanced splitter."""
    train, validation, _ = split_train_validation_test_by_document(
        chunks,
        validation_ratio=validation_ratio,
        test_ratio=0.0,
        seed=seed,
        stratify_by=stratify_by,
        min_validation_pdfs=3,
        min_test_pdfs=0,
    )
    return train, validation


# ---------------------------------------------------------------------------
# Pipeline Integration: Enhanced Chunk Processing
# ---------------------------------------------------------------------------

def enrich_chunks(chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enrich chunks with section_type, content_role, and topic tags."""
    enriched: List[Dict[str, Any]] = []
    for chunk in chunks:
        new_chunk = dict(chunk)
        text = str(chunk.get("text", ""))
        section = str(chunk.get("section", ""))
        new_chunk["pdf_id"] = str(chunk.get("pdf_id") or slugify(str(chunk.get("source_pdf", "document"))))

        # Phase 3: Section classification
        section_type = classify_section_type(section, text[:500])
        content_role = classify_content_role(text, section_type)

        # Phase 4: Topic tagging
        topics = tag_topics(text)
        lang = detect_language(text[:4000]) if callable(detect_language) else "unknown"

        new_chunk["section_type"] = section_type
        new_chunk["content_role"] = content_role
        new_chunk["language"] = lang
        new_chunk["topics"] = topics
        new_chunk["topic_tags"] = topics
        new_chunk["topic_count"] = len(topics)

        enriched.append(new_chunk)
    return enriched


def filter_lm_chunks(
    chunks: Sequence[Dict[str, Any]],
    allowed_languages: Sequence[str] = ("es", "unknown"),
) -> List[Dict[str, Any]]:
    """Drop navigation/administrative chunks from final LM outputs."""
    excluded_roles = {"navigation", "administrative", "excluded"}
    allowed = {str(x).strip().lower() for x in allowed_languages if str(x).strip()}
    filtered: List[Dict[str, Any]] = []
    for chunk in chunks:
        role = str(chunk.get("content_role", "unknown"))
        lang = str(chunk.get("language", "unknown")).lower()
        if role in excluded_roles:
            continue
        if allowed and lang not in allowed:
            continue
        filtered.append(chunk)
    return filtered
