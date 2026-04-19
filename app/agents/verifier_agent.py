import json

from app.config import get_llm
from app.schemas import VerificationOutput
from app.state import AgentState
from app.utils import parse_model_output


def verify_rules(state: AgentState) -> AgentState:
    """
    Verify extracted rules against source text.
    """
    llm = get_llm()

    prompt = f"""
You are a strict policy verifier.

You are given:
1. original document text
2. extracted rules

Verify each rule.
A rule is supported only if the source text clearly backs it.

Return JSON only:

{{
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
  "ambiguities": ["string"]
}}

Document:
{state["raw_text"][:15000]}

Extracted rules:
{json.dumps(state["extracted_rules"], indent=2)}
"""

    response = llm.invoke(prompt)
    parsed = parse_model_output(response.content, VerificationOutput)

    return {
        **state,
        "verified_rules": [rule.model_dump() for rule in parsed.verified_rules],
        "ambiguities": sorted(
            set(state["ambiguities"] + [a for a in parsed.ambiguities if a])
        ),
    }