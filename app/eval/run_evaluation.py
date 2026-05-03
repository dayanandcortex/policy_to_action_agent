import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.eval.scoring import compare_scores, score_result


def _initial_state(input_path: str) -> Dict[str, Any]:
    from app.config import get_max_revisions

    return {
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


def _finalize_multi_agent_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
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


def _timed_call(fn, *args) -> Tuple[Optional[Dict[str, Any]], float, Optional[str]]:
    start = time.perf_counter()
    try:
        return fn(*args), time.perf_counter() - start, None
    except Exception as exc:
        return None, time.perf_counter() - start, f"{type(exc).__name__}: {exc}"


def run_multi_agent(input_path: str) -> Dict[str, Any]:
    from app.graph import build_graph

    app = build_graph()
    result = app.invoke(_initial_state(input_path))
    return _finalize_multi_agent_result(result)


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_evaluation(input_path: str, output_dir: str = "outputs/eval") -> Dict[str, Any]:
    from app.eval.baseline_agent import run_one_shot_baseline

    output_path = Path(output_dir)

    multi_result, multi_runtime, multi_error = _timed_call(run_multi_agent, input_path)
    baseline_result, baseline_runtime, baseline_error = _timed_call(
        run_one_shot_baseline,
        input_path,
    )

    multi_score = score_result(
        "multi_agent",
        multi_result,
        multi_runtime,
        multi_error,
    )
    baseline_score = score_result(
        "one_shot_baseline",
        baseline_result,
        baseline_runtime,
        baseline_error,
    )

    report = {
        "input_path": input_path,
        "scores": {
            "multi_agent": multi_score,
            "one_shot_baseline": baseline_score,
        },
        "comparison": compare_scores(multi_score, baseline_score),
    }

    if multi_result is not None:
        save_json(multi_result, output_path / "multi_agent_result.json")
    if baseline_result is not None:
        save_json(baseline_result, output_path / "baseline_result.json")
    save_json(report, output_path / "comparison_report.json")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the multi-agent pipeline against a one-shot LLM baseline.",
    )
    parser.add_argument("input_path", help="Path to a .pdf, .txt, or .md policy document.")
    parser.add_argument(
        "--output-dir",
        default="outputs/eval",
        help="Directory where evaluation JSON files will be saved.",
    )
    args = parser.parse_args()

    report = run_evaluation(args.input_path, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
