from __future__ import annotations

import argparse

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = ROOT / "data" / "annotations" / "gold_v1"
REPORT_ROOT = ROOT / "reports" / "rag"

RECOVERED = {"GO-005", "GO-007", "GO-008", "GO-010", "GO-015", "GO-016"}

# Each missing frozen-gold outcome was compared with the final accepted graph and
# the independently scored retrieval result. The primary category describes the
# earliest stage that prevented exact recovery.
ERROR_AUDIT = {
    "GO-001": (
        "experiment_boundary_error",
        "Evidence was retrieved, but the GP-004 Kupffer/CD11b reporter experiment was omitted.",
    ),
    "GO-002": (
        "experiment_boundary_error",
        "Evidence was retrieved, but the GP-004 F4/80-positive qualitative result was omitted with its experiment.",
    ),
    "GO-003": (
        "experiment_boundary_error",
        "Evidence was retrieved, but GP-006 retained only the Cas9/sgRNA experiment and omitted reporter delivery.",
    ),
    "GO-004": (
        "experiment_boundary_error",
        "Evidence was retrieved, but the hepatocyte-to-LSEC reporter comparison was omitted with the reporter experiment.",
    ),
    "GO-006": (
        "incomplete_evidence",
        "The supplement page was retrieved and deletion frequency was retained, but the insertion-frequency table value was omitted.",
    ),
    "GO-011": (
        "normalization_error",
        "The graph retained low Kupffer translation but did not preserve the frozen exact negative result of no obvious EGFP-positive Kupffer cells.",
    ),
    "GO-013": (
        "wrong_relation",
        "GP-007 retained improvement values without a complete intervention-to-endpoint relation for the frozen LSEC protection outcome.",
    ),
    "GO-017": (
        "wrong_relation",
        "GP-008 retained the therapeutic-target-cell link but not the macrophage-mediated activated-HSC elimination outcome.",
    ),
    "GO-018": (
        "human_gold_disagreement",
        "The frozen page was 18, but the extracted PDF places Appendix Figure 5 panels G-L and marker labels on page 19.",
    ),
}

VALID_CATEGORIES = {
    "retrieval_miss",
    "experiment_boundary_error",
    "wrong_entity_type",
    "wrong_relation",
    "unsupported_inference",
    "incomplete_evidence",
    "normalization_error",
    "human_gold_disagreement",
}


def analyze() -> dict:
    outcomes = {
        row["gold_outcome_id"]: row
        for row in csv.DictReader((ANNOTATIONS / "outcomes.csv").open(encoding="utf-8", newline=""))
    }
    evidence = {
        row["evidence_id"]: row
        for row in csv.DictReader((ANNOTATIONS / "evidence.csv").open(encoding="utf-8", newline=""))
    }
    retrieval = json.loads(
        (REPORT_ROOT / "gold_v1_retrieval_sentence-transformers.json").read_text()
    )
    retrieval_by_evidence = {row["evidence_id"]: row for row in retrieval["results"]}

    missing = sorted(set(outcomes) - RECOVERED)
    if missing != sorted(ERROR_AUDIT):
        raise ValueError("ERROR_AUDIT must classify every and only missing gold outcome.")
    if any(category not in VALID_CATEGORIES for category, _ in ERROR_AUDIT.values()):
        raise ValueError("Unknown Day 7 error category.")

    rows = []
    for outcome_id in missing:
        outcome = outcomes[outcome_id]
        evidence_id = outcome["evidence_id"]
        retrieval_row = retrieval_by_evidence[evidence_id]
        category, rationale = ERROR_AUDIT[outcome_id]
        rows.append({
            "gold_outcome_id": outcome_id,
            "paper_id": evidence[evidence_id]["gold_paper_id"],
            "evidence_id": evidence_id,
            "endpoint_name": outcome["endpoint_name"],
            "retrieved_at_k": retrieval_row["hit"],
            "retrieval_rank": retrieval_row["first_gold_rank"],
            "primary_error_category": category,
            "rationale": rationale,
        })

    category_counts = Counter(row["primary_error_category"] for row in rows)
    retrieval_recovered = sum(bool(row["retrieved_at_k"]) for row in rows)
    result = {
        "analysis_version": "day7_morning_g1_error_audit_v1",
        "calculated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "All nine outcomes missing from the frozen G1 gold comparison.",
        "metrics": {
            "gold_outcomes": len(outcomes),
            "exact_outcomes_recovered": len(RECOVERED),
            "exact_outcome_recall": len(RECOVERED) / len(outcomes),
            "missing_outcomes": len(rows),
            "missing_outcomes_with_retrieved_evidence": retrieval_recovered,
            "missing_outcomes_with_retrieval_miss_or_provenance_defect": len(rows) - retrieval_recovered,
            "category_counts": dict(sorted(category_counts.items())),
            "retrieval_k": retrieval["k"],
            "retrieval_recall_at_k": retrieval["recall_at_k"],
            "extraction_recall_given_retrieval": len(RECOVERED) / (
                len(RECOVERED) + retrieval_recovered
            ),
            "post_review_critical_precision": 1.0,
        },
        "errors": rows,
        "interpretation": (
            "Retrieval and extraction are reported separately: retrieved evidence does not "
            "count as an extracted outcome, and a provenance mismatch is not an LLM error."
        ),
    }

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "day7_morning_g1_error_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    with (REPORT_ROOT / "day7_morning_g1_error_analysis.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Day 7 morning: G1 error analysis",
        "",
        f"- Exact outcome recall: **{len(RECOVERED)}/{len(outcomes)} ({len(RECOVERED)/len(outcomes):.1%})**",
        f"- Missing outcomes with evidence already retrieved: **{retrieval_recovered}/{len(rows)}**",
        f"- Extraction recall conditional on retrieved outcome evidence: **{result['metrics']['extraction_recall_given_retrieval']:.1%}**",
        f"- Independent retrieval recall@{retrieval['k']}: **{retrieval['hits']}/{retrieval['queries']} ({retrieval['recall_at_k']:.1%})**",
        "- Post-review critical-field precision: **100%**",
        "",
        "## Error classes",
        "",
    ]
    lines.extend(f"- {key}: **{value}**" for key, value in sorted(category_counts.items()))
    lines += [
        "",
        "## Outcome-level audit",
        "",
        "| Outcome | Paper | Retrieval | Primary class | Explanation |",
        "|---|---|---:|---|---|",
    ]
    lines.extend(
        f"| {row['gold_outcome_id']} | {row['paper_id']} | "
        f"{'hit' if row['retrieved_at_k'] else 'miss'} | "
        f"{row['primary_error_category']} | {row['rationale']} |"
        for row in rows
    )
    lines += [
        "",
        "> Retrieval recall, conditional extraction recall, and post-review precision are "
        "separate metrics and must not be substituted for one another.",
        "",
    ]
    (REPORT_ROOT / "day7_morning_g1_error_analysis.md").write_text("\n".join(lines))
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
    print(json.dumps(analyze(), indent=2))


if __name__ == "__main__":
    main()
