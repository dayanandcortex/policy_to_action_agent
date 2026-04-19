from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.agents.classifier_agent import classify_document
from app.agents.critic_agent import critique_output
from app.agents.decision_agent import make_decision
from app.agents.extractor_agent import extract_rules
from app.agents.reader_agent import read_document
from app.agents.verifier_agent import verify_rules
from app.state import AgentState


def route_after_critic(state: AgentState) -> Literal["extract_rules", "__end__"]:
    """
    Route after critic.
    """
    critic_passed = state["final_decision"].get("critic_passed", False)

    if critic_passed:
        return END

    if state["revision_count"] >= state["max_revisions"]:
        return END

    return "extract_rules"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("read_document", read_document)
    builder.add_node("classify_document", classify_document)
    builder.add_node("extract_rules", extract_rules)
    builder.add_node("verify_rules", verify_rules)
    builder.add_node("make_decision", make_decision)
    builder.add_node("critique_output", critique_output)

    builder.add_edge(START, "read_document")
    builder.add_edge("read_document", "classify_document")
    builder.add_edge("classify_document", "extract_rules")
    builder.add_edge("extract_rules", "verify_rules")
    builder.add_edge("verify_rules", "make_decision")
    builder.add_edge("make_decision", "critique_output")

    builder.add_conditional_edges("critique_output", route_after_critic)

    return builder.compile()