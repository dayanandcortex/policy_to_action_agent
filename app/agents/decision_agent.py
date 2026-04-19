import json

from app.config import get_llm
from app.schemas import FinalDecisionOutput
from app.state import AgentState
from app.utils import parse_model_output


def make_decision(state: AgentState) -> AgentState:
    """
    Make a final recommendation from verified rules.
    """
    llm = get_llm()

    supported_rules = [
        rule for rule in state["verified_rules"] if rule.get("supported") is True
    ]

    prompt = f"""
You are a decision agent.

Use only supported verified rules.
If there are meaningful ambiguities, prefer 'Needs Manual Review'.

Return JSON only:

{{
  "recommendation": "Approved / Denied / Needs Manual Review / Eligible / Not Eligible",
  "reasoning_summary": "short explanation",
  "confidence": 0.0,
  "manual_review_needed": true
}}

Document type:
{state["document_type"]}

Supported verified rules:
{json.dumps(supported_rules, indent=2)}

Ambiguities:
{json.dumps(state["ambiguities"], indent=2)}
"""

    response = llm.invoke(prompt)
    parsed = parse_model_output(response.content, FinalDecisionOutput)

    return {
        **state,
        "final_decision": parsed.model_dump(),
        "confidence": parsed.confidence,
    }