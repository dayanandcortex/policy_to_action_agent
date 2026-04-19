from app.config import get_llm
from app.schemas import ChunkExtractionOutput
from app.state import AgentState
from app.utils import (
    dedupe_rules,
    parse_model_output,
    repair_chunk_extraction_payload,
    safe_merge_entities,
)


def extract_rules(state: AgentState) -> AgentState:
    """
    Extract rules and entities chunk by chunk.
    """
    llm = get_llm()

    merged_entities = dict(state["extracted_entities"])
    all_rules = []
    all_ambiguities = list(state["ambiguities"])

    for chunk in state["chunks"]:
        prompt = f"""
You are a policy analysis agent.

Document type:
{state["document_type"]}

Analyze the chunk below and extract:
1. important entities
2. policy rules
3. ambiguities

For each rule include:
- condition
- action
- exception
- evidence
- page_number
- confidence

For extracted_entities:
- use a string for single-value fields
- use a list of strings for naturally multi-value fields
- examples of multi-value fields: exclusions, covered_items, required_documents, triggers, exceptions

Use only page numbers that appear in the chunk.
Do not invent facts.
Return JSON only in this shape:

{{
  "extracted_entities": {{
    "key": "value or list of strings"
  }},
  "extracted_rules": [
    {{
      "condition": "string",
      "action": "string",
      "exception": "string or null",
      "evidence": "string",
      "page_number": 1,
      "confidence": 0.0
    }}
  ],
  "ambiguities": ["string"]
}}

Chunk:
{chunk["text"]}
"""

        response = llm.invoke(prompt)

        print("\n[EXTRACTOR] Processing chunk:", chunk["chunk_id"])
        print("[EXTRACTOR] Pages:", chunk["page_numbers"])
        print("[EXTRACTOR] Raw response:")
        print(response.content)

        try:
            parsed = parse_model_output(response.content, ChunkExtractionOutput)
        except Exception:
            repaired = repair_chunk_extraction_payload(response.content)
            parsed = ChunkExtractionOutput(**repaired)

        merged_entities = safe_merge_entities(
            merged_entities,
            parsed.extracted_entities,
        )
        all_rules.extend([rule.model_dump() for rule in parsed.extracted_rules])
        all_ambiguities.extend(parsed.ambiguities)

    return {
        **state,
        "extracted_entities": merged_entities,
        "extracted_rules": dedupe_rules(all_rules),
        "ambiguities": sorted(set(a for a in all_ambiguities if a)),
    }