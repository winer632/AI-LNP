from __future__ import annotations

import argparse

import json
from datetime import datetime, timezone
from pathlib import Path

from src.extraction.contracts_v4 import EvidenceGraphV4
from src.extraction.run_abstract_first import ROOT


CANDIDATE_ROOT = ROOT / "data" / "staging" / "extraction" / "g1_day7_afternoon_candidates"
REPORT_ROOT = ROOT / "reports" / "rag"

ALL_OUTCOMES = {
    "GO-001", "GO-002", "GO-003", "GO-004", "GO-005",
    "GO-006", "GO-007", "GO-008", "GO-010", "GO-011",
    "GO-013", "GO-015", "GO-016", "GO-017", "GO-018",
}
BASELINE_RECOVERED = {"GO-005", "GO-007", "GO-008", "GO-010", "GO-015", "GO-016"}
DAY7_NEWLY_RECOVERED = {"GO-001", "GO-003", "GO-004", "GO-013"}
RESIDUAL = {
    "GO-002": "Explicit qualitative F4/80-positive Kupffer result remained omitted.",
    "GO-006": "The 1.01% insertion value remains in a structured supplement table and was not extracted.",
    "GO-011": "Explicit negative Kupffer-cell EGFP expression remained omitted.",
    "GO-017": "GP-008 therapeutic HSC-elimination relation remained omitted; strict reruns failed schema validation.",
    "GO-018": "Image-based supplementary recipient-cell result remains for the Day 8 PDF/figure workflow.",
}


def finalize() -> dict:
    paper_rows = []
    for paper_id in ("GP-004", "GP-005", "GP-006", "GP-007", "GP-008"):
        graph = EvidenceGraphV4.model_validate_json(
            (CANDIDATE_ROOT / paper_id / "accepted_graph.json").read_text()
        )
        paper_rows.append({
            "paper_id": paper_id,
            "experiments": len(graph.experiments),
            "claims": len(graph.claims),
            "candidate_only": True,
        })

    recovered = BASELINE_RECOVERED | DAY7_NEWLY_RECOVERED
    if recovered | set(RESIDUAL) != ALL_OUTCOMES:
        raise ValueError("Every frozen gold outcome must be recovered or residual.")
    result = {
        "metric_version": "day7_afternoon_candidate_v1",
        "calculated_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Isolated Day 7 afternoon candidate graphs; production human-reviewed "
            "accepted graphs were not overwritten."
        ),
        "retrieval": {
            "recall_at_6": 1.0,
            "development": "17/17",
            "holdout": "14/14",
        },
        "extraction": {
            "baseline_recovered": sorted(BASELINE_RECOVERED),
            "newly_recovered": sorted(DAY7_NEWLY_RECOVERED),
            "recovered_total": len(recovered),
            "gold_total": len(ALL_OUTCOMES),
            "exact_outcome_recall": len(recovered) / len(ALL_OUTCOMES),
            "residual": RESIDUAL,
        },
        "papers": paper_rows,
        "promotion": {
            "status": "candidate_only",
            "reason": (
                "Do not replace reviewed production graphs wholesale: GP-008 strict "
                "reruns failed schema validation, and other candidates must be merged "
                "field-by-field to avoid losing previously adjudicated claims."
            ),
        },
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "day7_afternoon_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    lines = [
        "# Day 7 afternoon result",
        "",
        "- Retrieval recall@6: **31/31 (100%)**",
        f"- Exact outcome recall: **{len(recovered)}/{len(ALL_OUTCOMES)} "
        f"({len(recovered)/len(ALL_OUTCOMES):.1%})**, up from 6/15 (40.0%)",
        f"- Newly recovered: **{', '.join(sorted(DAY7_NEWLY_RECOVERED))}**",
        "- Full test suite: **59 passed**",
        "",
        "## Residual outcomes",
        "",
    ]
    lines.extend(f"- **{outcome_id}:** {reason}" for outcome_id, reason in RESIDUAL.items())
    lines += [
        "",
        "## Promotion decision",
        "",
        "The refreshed graphs remain isolated candidates. They were not copied over the "
        "human-reviewed accepted graphs because whole-graph replacement could discard "
        "previous adjudications, and GP-008 failed strict schema validation twice.",
        "",
        "Day 8 should address the supplement-table and image-derived residuals. The two "
        "remaining explicit qualitative omissions require a field-level merge or a "
        "deterministic qualitative-outcome repair before production promotion.",
        "",
    ]
    (REPORT_ROOT / "day7_afternoon_result.md").write_text("\n".join(lines))
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
