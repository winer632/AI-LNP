"""Freeze reviewed Day 5 G1 metrics without overstating unreviewed correctness."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.output_paths import artifact_path


ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = ROOT / "reports" / "extraction" / "day5_g1_evidence_audit.json"
REVIEW_PATH = ROOT / "data" / "review" / "day5_g1_human_review.jsonl"

# Canonical locations of the frozen artifacts. The actual write target is
# resolved per call through `src.output_paths`, so a test run redirects the
# writes instead of rewriting these tracked files. See `finalize`.
OUTPUT_JSON = ROOT / "reports" / "extraction" / "day5_g1_final_metrics.json"
OUTPUT_MD = ROOT / "reports" / "extraction" / "day5_g1_final_decision.md"
OUTPUT_PARTS = ("reports", "extraction")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finalize(*, output_root: Path | str | None = None) -> dict[str, Any]:
    """Freeze the reviewed Day 5 G1 metrics.

    ``output_root`` overrides where the two artifacts are written. When it is
    omitted the target comes from :func:`src.output_paths.output_root`, which
    resolves to the repository for a real run and to a scratch directory under
    pytest so the tracked reports are never rewritten by the test suite.
    """
    output_json = artifact_path(*OUTPUT_PARTS, OUTPUT_JSON.name, root=output_root)
    output_md = artifact_path(*OUTPUT_PARTS, OUTPUT_MD.name, root=output_root)
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    reviews = load_jsonl(REVIEW_PATH)
    incomplete = [
        row["review_id"]
        for row in reviews
        if row.get("human_decision") not in {"correct", "incorrect", "ambiguous_or_absent"}
        or not row.get("reviewer_reason", "").strip()
        or not row.get("reviewer")
        or not row.get("reviewed_at")
    ]
    if incomplete:
        raise ValueError(f"Incomplete review rows: {incomplete}")

    decisions = Counter(row["human_decision"] for row in reviews)
    invalid_response_reviews = [
        row for row in reviews if row.get("preliminary_classification") == "invalid_model_response"
    ]
    semantic_reviews = [
        row for row in reviews if row.get("preliminary_classification") != "invalid_model_response"
    ]
    semantic_decisions = Counter(row["human_decision"] for row in semantic_reviews)
    literal_supported = audit["counts"]["source_supported"]
    extracted_fields = len(audit["extracted_field_audit"])
    best_case_correct = literal_supported + semantic_decisions["correct"]
    incorrect = semantic_decisions["incorrect"]
    precision_denominator = best_case_correct + incorrect
    best_case_precision = best_case_correct / precision_denominator if precision_denominator else None
    reviewed_semantic_precision = (
        semantic_decisions["correct"] / (semantic_decisions["correct"] + semantic_decisions["incorrect"])
        if semantic_decisions["correct"] + semantic_decisions["incorrect"]
        else None
    )
    threshold = 0.90
    result = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "review_completion": {
            "rows": len(reviews),
            "decisions": dict(decisions),
            "all_have_reasons_reviewers_and_timestamps": True,
        },
        "metrics": {
            "reviewed_semantic_precision": reviewed_semantic_precision,
            "best_case_overall_precision": best_case_precision,
            "best_case_correct_fields": best_case_correct,
            "incorrect_fields": incorrect,
            "abstract_extracted_fields_audited": extracted_fields,
            "traceable_abstract_evidence_coverage": audit["metrics"]["traceable_abstract_evidence_coverage"],
            "valid_bundle_rate": 6 / 9,
            "invalid_model_response_papers": len(invalid_response_reviews),
            "abstract_omissions_deferred_to_full_text": audit["counts"]["abstract_omission"],
            "abstract_available_gold_values": audit["counts"]["gold_value_literal_in_abstract"],
            "critical_field_recall": None,
            "recall_status": "not_established_due_to_invalid_responses_and_unresolved_entity_linkage",
        },
        "gate": {
            "name": "G1_extraction_quality",
            "required_precision": threshold,
            "decision": "FAIL",
            "reason": "Best-case precision is below 90%; three papers produced no valid extraction; critical-field recall is not established.",
        },
        "required_remediation": [
            "Remove payload records from formulation components.",
            "Separate payload type, encoded product, formulation description, and lipid composition.",
            "Preserve reported recipient-cell text alongside controlled categories.",
            "Prevent disease or tissue sites from being stored as cell types.",
            "Create separate outcome records for distinct endpoints.",
            "Require the cited excerpt to directly support each accepted value.",
            "Replace or repair the unreliable model response path and rerun all gold papers.",
        ],
    }
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md.write_text(
        "# Day 5 G1 final decision\n\n"
        "**Decision: FAIL**\n\n"
        f"- Reviewed semantic precision: {reviewed_semantic_precision:.1%}\n"
        f"- Best-case overall precision: {best_case_precision:.1%}\n"
        f"- Required precision: {threshold:.0%}\n"
        f"- Traceable abstract-evidence coverage: {audit['metrics']['traceable_abstract_evidence_coverage']:.1%}\n"
        "- Valid schema bundle rate: 6/9 (66.7%)\n"
        "- Critical-field recall: not established\n\n"
        "The best-case calculation assumes every automatically literal-supported field is correctly typed and linked. Even under that favorable assumption, precision remains below the gate. Abstract omissions are deferred to targeted full-text retrieval and are not counted as incorrect extractions.\n\n"
        "## Required remediation\n\n"
        + "\n".join(f"- {item}" for item in result["required_remediation"])
        + "\n",
        encoding="utf-8",
    )
    return result



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Required: this rewrites tracked files in place.",
    )
    args = parser.parse_args()
    if not args.confirm_write:
        parser.error("--confirm-write is required; this rewrites tracked files")
    print(json.dumps(finalize(), indent=2))


if __name__ == "__main__":
    main()
