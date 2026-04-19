from app.config import get_llm
from app.state import AgentState


def classify_document(state: AgentState) -> AgentState:
    """
    Classify the document type.
    """
    llm = get_llm()

    sample_text = state["raw_text"][:5000]

    prompt = f"""
You are a document classifier.

Classify the following document into exactly one label:
- health_insurance_policy
- tax_policy
- hr_leave_policy
- reimbursement_policy
- compliance_policy
- generic_policy

Return only the label. No explanation.

Document:
{sample_text}
"""

    response = llm.invoke(prompt)
    document_type = response.content.strip().splitlines()[0].strip()

    return {
        **state,
        "document_type": document_type,
    }