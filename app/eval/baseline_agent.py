import json
from typing import Any, Dict

from app.config import get_llm
from app.schemas import OneShotAnalysisOutput
from app.tools.file_loader import load_text_file
from app.utils import extract_json_block, parse_model_output


def _normalize_entity_value(value: Any) -> str | list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return ""
    return str(value)


def _repair_one_shot_payload(text: str) -> Dict[str, Any]:
    """
    Repair common one-shot schema drift without hiding real structural errors.
    """
    parsed = json.loads(extract_json_block(text))
    entities = parsed.get("extracted_entities", {})

    if isinstance(entities, dict):
        parsed["extracted_entities"] = {
            str(key): _normalize_entity_value(value)
            for key, value in entities.items()
        }

    return parsed


def run_one_shot_baseline(input_path: str) -> dict:
    """
    Run a baseline where one LLM prompt performs the full analysis.
    """
    raw_text, _ = load_text_file(input_path)
    llm = get_llm()

    prompt = f"""
You are a policy analysis agent.

Analyze the full policy document in one pass. Do all of the following:
1. classify the document type
2. extract important entities
3. extract policy rules
4. verify each rule against the source text
5. list ambiguities
6. produce a final decision about extraction quality, not a claim outcome

For final_decision:
- use "Pass" when the extracted and verified rules are well grounded
- use "Needs Manual Review" only when there are major unsupported rules,
  missing evidence, broken schema, or serious unresolved uncertainty
- do not fail only because no claim/application scenario was provided

Return JSON only in this exact shape:

{{
  "document_type": "health_insurance_policy / tax_policy / hr_leave_policy / reimbursement_policy / compliance_policy / generic_policy",
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
  "verified_rules": [
    {{
      "condition": "string",
      "action": "string",
      "exception": "string or null",
      "evidence": "string",
      "page_number": 1,
      "supported": true,
      "support_reason": "string",
      "confidence": 0.0
    }}
  ],
  "ambiguities": ["string"],
  "final_decision": {{
    "recommendation": "Pass / Needs Manual Review",
    "reasoning_summary": "short extraction-quality assessment",
    "confidence": 0.0,
    "manual_review_needed": false
  }}
}}

Use only facts supported by the document.
Every rule must include evidence and a page_number.

Document:
{raw_text[:30000]}
"""

    response = llm.invoke(prompt)
    try:
        parsed = parse_model_output(response.content, OneShotAnalysisOutput)
    except Exception:
        parsed = OneShotAnalysisOutput(**_repair_one_shot_payload(response.content))

    output = parsed.model_dump()
    output["confidence"] = parsed.final_decision.confidence
    output["critic_feedback"] = ""
    output["revision_count"] = 0
    return output
