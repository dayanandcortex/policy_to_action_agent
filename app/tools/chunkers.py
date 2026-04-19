from typing import Any, Dict, List


def build_chunks(
    page_texts: List[Dict[str, Any]],
    max_chars: int,
) -> List[Dict[str, Any]]:
    """
    Build page-aware chunks.
    """
    chunks: List[Dict[str, Any]] = []
    current_text_parts: List[str] = []
    current_pages: List[int] = []
    current_len = 0

    for item in page_texts:
        page_number = item["page_number"]
        text = item["text"].strip()

        if not text:
            continue

        page_block = f"--- PAGE {page_number} ---\n{text}\n"
        block_len = len(page_block)

        if current_text_parts and current_len + block_len > max_chars:
            chunks.append(
                {
                    "chunk_id": len(chunks) + 1,
                    "page_numbers": current_pages[:],
                    "text": "\n".join(current_text_parts),
                }
            )
            current_text_parts = []
            current_pages = []
            current_len = 0

        current_text_parts.append(page_block)
        current_pages.append(page_number)
        current_len += block_len

    if current_text_parts:
        chunks.append(
            {
                "chunk_id": len(chunks) + 1,
                "page_numbers": current_pages[:],
                "text": "\n".join(current_text_parts),
            }
        )

    return chunks