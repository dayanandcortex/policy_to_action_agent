import json
import re
from typing import Any, Dict, Type, Union, List

from pydantic import BaseModel

EntityValue = Union[str, List[str]]



def extract_json_block(text: str) -> str:
    """
    Extract JSON from raw LLM output.
    Handles plain JSON and triple-backtick fenced JSON.
    """
    stripped = text.strip()

    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
        if match:
            return match.group(1).strip()

    return stripped


def parse_model_output(text: str, schema: Type[BaseModel]) -> BaseModel:
    """
    Parse LLM text into a pydantic schema.
    """
    json_text = extract_json_block(text)
    parsed: Dict[str, Any] = json.loads(json_text)
    return schema(**parsed)


def safe_merge_entities(
    base: Dict[str, EntityValue],
    incoming: Dict[str, EntityValue],
) -> Dict[str, EntityValue]:
    """
    Merge extracted entities conservatively.
    Supports both string and list values.
    """
    merged = dict(base)

    for key, value in incoming.items():
        if key not in merged or not merged[key]:
            merged[key] = value
            continue

        existing = merged[key]

        if isinstance(existing, list) and isinstance(value, list):
            merged[key] = list(dict.fromkeys(existing + value))
        elif isinstance(existing, list) and isinstance(value, str):
            if value not in existing:
                merged[key] = existing + [value]
        elif isinstance(existing, str) and isinstance(value, list):
            items = [existing] + [v for v in value if v != existing]
            merged[key] = list(dict.fromkeys(items))
        elif isinstance(existing, str) and isinstance(value, str):
            if existing != value:
                merged[key] = [existing, value]

    return merged


def dedupe_rules(rules: list[dict]) -> list[dict]:
    seen = set()
    output = []

    for rule in rules:
        signature = (
            rule.get("condition", "").strip().lower(),
            rule.get("action", "").strip().lower(),
            str(rule.get("page_number", "")),
        )
        if signature not in seen:
            seen.add(signature)
            output.append(rule)

    return output



def repair_chunk_extraction_payload(text: str) -> Dict[str, Any]:
    """
    Repair extractor output when local/free models return slightly off shapes.
    """
    json_text = extract_json_block(text)
    parsed = json.loads(json_text)

    raw_entities = parsed.get("extracted_entities", {})
    repaired_entities: Dict[str, EntityValue] = {}

    for key, value in raw_entities.items():
        if isinstance(value, list):
            repaired_entities[key] = [str(v) for v in value]
        elif value is None:
            repaired_entities[key] = ""
        else:
            repaired_entities[key] = str(value)

    repaired_rules = []
    for rule in parsed.get("extracted_rules", []):
        repaired_rules.append(
            {
                "condition": str(rule.get("condition", "")),
                "action": str(rule.get("action", "")),
                "exception": rule.get("exception"),
                "evidence": str(rule.get("evidence", "")),
                "page_number": int(rule.get("page_number", 1) or 1),
                "confidence": float(rule.get("confidence", 0.5) or 0.5),
            }
        )

    return {
        "extracted_entities": repaired_entities,
        "extracted_rules": repaired_rules,
        "ambiguities": [str(a) for a in parsed.get("ambiguities", [])],
    }