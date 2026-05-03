from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Optional


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _avg_confidence(items: List[Dict[str, Any]]) -> float:
    values = [
        item.get("confidence")
        for item in items
        if isinstance(item.get("confidence"), (int, float))
    ]
    return round(mean(values), 4) if values else 0.0


def score_result(
    name: str,
    result: Optional[Dict[str, Any]],
    runtime_seconds: float,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute deterministic extraction and grounding metrics.
    """
    if error or result is None:
        return {
            "name": name,
            "schema_valid": False,
            "error": error or "No result produced",
            "runtime_seconds": round(runtime_seconds, 3),
            "rules_extracted": 0,
            "verified_rules": 0,
            "supported_rules": 0,
            "unsupported_rules": 0,
            "support_rate": 0.0,
            "rules_with_evidence": 0,
            "rules_with_page_number": 0,
            "avg_verified_confidence": 0.0,
            "ambiguities": 0,
            "manual_review_needed": None,
            "final_recommendation": None,
        }

    extracted_rules = _as_list(result.get("extracted_rules"))
    verified_rules = _as_list(result.get("verified_rules"))
    supported_rules = [
        rule for rule in verified_rules if isinstance(rule, dict) and rule.get("supported") is True
    ]
    unsupported_rules = [
        rule for rule in verified_rules if isinstance(rule, dict) and rule.get("supported") is False
    ]
    evidence_source = verified_rules if verified_rules else extracted_rules
    rules_with_evidence = sum(
        1 for rule in evidence_source if isinstance(rule, dict) and _has_text(rule.get("evidence"))
    )
    rules_with_page_number = sum(
        1
        for rule in evidence_source
        if isinstance(rule, dict) and isinstance(rule.get("page_number"), int) and rule["page_number"] >= 1
    )
    support_rate = len(supported_rules) / len(verified_rules) if verified_rules else 0.0
    final_decision = result.get("final_decision") or {}

    return {
        "name": name,
        "schema_valid": True,
        "error": None,
        "runtime_seconds": round(runtime_seconds, 3),
        "rules_extracted": len(extracted_rules),
        "verified_rules": len(verified_rules),
        "supported_rules": len(supported_rules),
        "unsupported_rules": len(unsupported_rules),
        "support_rate": round(support_rate, 4),
        "rules_with_evidence": rules_with_evidence,
        "rules_with_page_number": rules_with_page_number,
        "avg_verified_confidence": _avg_confidence(verified_rules),
        "ambiguities": len(_as_list(result.get("ambiguities"))),
        "manual_review_needed": final_decision.get("manual_review_needed"),
        "final_recommendation": final_decision.get("recommendation"),
    }


def compare_scores(
    multi_agent_score: Dict[str, Any],
    baseline_score: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a compact qualitative/quantitative comparison summary.
    """
    rule_delta = multi_agent_score["rules_extracted"] - baseline_score["rules_extracted"]
    supported_delta = multi_agent_score["supported_rules"] - baseline_score["supported_rules"]
    support_rate_delta = multi_agent_score["support_rate"] - baseline_score["support_rate"]
    evidence_delta = (
        multi_agent_score["rules_with_evidence"] - baseline_score["rules_with_evidence"]
    )

    observations = []
    if rule_delta > 0:
        observations.append(
            f"Multi-agent extracted {rule_delta} more rules than the one-shot baseline."
        )
    elif rule_delta < 0:
        observations.append(
            f"One-shot baseline extracted {abs(rule_delta)} more rules than the multi-agent pipeline."
        )
    else:
        observations.append("Both approaches extracted the same number of rules.")

    if supported_delta > 0:
        observations.append(
            f"Multi-agent produced {supported_delta} more supported verified rules."
        )
    elif supported_delta < 0:
        observations.append(
            f"One-shot baseline produced {abs(supported_delta)} more supported verified rules."
        )

    if evidence_delta > 0:
        observations.append(
            f"Multi-agent attached evidence to {evidence_delta} more rules."
        )
    elif evidence_delta < 0:
        observations.append(
            f"One-shot baseline attached evidence to {abs(evidence_delta)} more rules."
        )

    observations.append(
        f"Support-rate delta: {support_rate_delta:+.2%} in favor of multi-agent when positive."
    )

    return {
        "rule_delta": rule_delta,
        "supported_rule_delta": supported_delta,
        "support_rate_delta": round(support_rate_delta, 4),
        "evidence_delta": evidence_delta,
        "observations": observations,
    }
