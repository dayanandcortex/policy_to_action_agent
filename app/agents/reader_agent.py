from app.config import get_max_chars_per_chunk
from app.state import AgentState
from app.tools.chunkers import build_chunks
from app.tools.file_loader import load_text_file


def read_document(state: AgentState) -> AgentState:
    """
    Read source document and create chunks.
    """
    raw_text, page_texts = load_text_file(state["input_path"])
    chunks = build_chunks(page_texts, max_chars=get_max_chars_per_chunk())

    return {
        **state,
        "raw_text": raw_text,
        "page_texts": page_texts,
        "chunks": chunks,
    }