import json

from app.config import get_llm
from app.schemas import CriticOutput
from app.state import AgentState
from app.utils import parse_model_output


def critique_output(state: AgentState) -> AgentState:
    """
    Critique the final output.
    """
    llm = get_llm()

    prompt = f"""
You are a critic agent.

Review:
- verified rules
- ambiguities
- final decision

Fail the output if:
- the recommendation is too strong for the evidence
- many rules are unsupported
- confidence is too high relative to uncertainty
- manual review should obviously be required

Return JSON only:

{{
  "passed": true,
  "feedback": "string"
}}

Verified rules:
{json.dumps(state["verified_rules"], indent=2)}

Ambiguities:
{json.dumps(state["ambiguities"], indent=2)}

Final decision:
{json.dumps(state["final_decision"], indent=2)}
"""

    response = llm.invoke(prompt)
    parsed = parse_model_output(response.content, CriticOutput)

    final_decision = dict(state["final_decision"])
    final_decision["critic_passed"] = parsed.passed

    revision_count = state["revision_count"]
    if not parsed.passed:
        revision_count += 1

    return {
        **state,
        "critic_feedback": parsed.feedback,
        "revision_count": revision_count,
        "final_decision": final_decision,
    }