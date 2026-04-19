from pathlib import Path
from typing import Any, Dict, List, Tuple


from pypdf import PdfReader


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def load_pdf_file(file_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extract text from a PDF.
    """
    reader = PdfReader(file_path)

    pages: List[Dict[str, Any]] = []
    combined_parts: List[str] = []

    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        cleaned = normalize_text(text)

        pages.append(
            {
                "page_number": idx,
                "text": cleaned,
            }
        )

        if cleaned.strip():
            combined_parts.append(f"--- PAGE {idx} ---\n{cleaned}")

    full_text = "\n\n".join(combined_parts)
    return full_text, pages


def load_text_file(file_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Load content from .pdf, .txt, or .md.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf_file(file_path)

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
        cleaned = normalize_text(text)
        return cleaned, [{"page_number": 1, "text": cleaned}]

    raise ValueError("Supported file types: .pdf, .txt, .md")