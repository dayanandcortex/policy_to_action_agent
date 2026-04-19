import json
from pathlib import Path

from app.config import get_max_revisions
from app.graph import build_graph
from app.state import AgentState


def save_result_to_file(result: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)


def main() -> None:
    input_path = input("Enter path to policy file (.pdf, .txt, .md): ").strip()

    app = build_graph()

    initial_state: AgentState = {
        "input_path": input_path,
        "raw_text": "",
        "page_texts": [],
        "chunks": [],
        "document_type": "",
        "extracted_entities": {},
        "extracted_rules": [],
        "verified_rules": [],
        "ambiguities": [],
        "final_decision": {},
        "critic_feedback": "",
        "confidence": 0.0,
        "revision_count": 0,
        "max_revisions": get_max_revisions(),
    }

    result = app.invoke(initial_state)

    final_output = {
        "document_type": result["document_type"],
        "extracted_entities": result["extracted_entities"],
        "extracted_rules": result["extracted_rules"],
        "verified_rules": result["verified_rules"],
        "ambiguities": result["ambiguities"],
        "final_decision": result["final_decision"],
        "critic_feedback": result["critic_feedback"],
        "confidence": result["confidence"],
        "revision_count": result["revision_count"],
    }

    print("\n" + "=" * 80)
    print("FINAL OUTPUT")
    print("=" * 80)
    print(json.dumps(final_output, indent=2))

    save_result_to_file(final_output, "outputs/result.json")
    print("\nSaved to outputs/result.json")


if __name__ == "__main__":
    main()