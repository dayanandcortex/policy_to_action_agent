from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict):
    input_path: str
    raw_text: str
    page_texts: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]

    document_type: str
    extracted_entities: Dict[str, Any]
    extracted_rules: List[Dict[str, Any]]
    verified_rules: List[Dict[str, Any]]
    ambiguities: List[str]

    final_decision: Dict[str, Any]
    critic_feedback: str
    confidence: float

    revision_count: int
    max_revisions: int