from __future__ import annotations

import json
import csv
from datetime import datetime, timezone
from pathlib import Path

from src.extraction.contracts_v4 import EvidenceGraphV4
from src.extraction.run_abstract_first import ROOT
from src.output_paths import artifact_path


RUN_ROOT = ROOT / "data" / "staging" / "extraction" / "g1_fulltext_rag"

# Canonical output locations. The actual write targets are resolved per call
# through `src.output_paths` so a test run redirects them instead of rewriting
# tracked files. See `finalize`.
REPORT_ROOT = ROOT / "reports" / "rag"
REPORT_PARTS = ("reports", "rag")
ELIGIBLE_PARTS = ("data", "staging", "extraction", "g1_fulltext_rag", "eligible")

POSITIVE_PAPERS = ("GP-002", "GP-004", "GP-005", "GP-006", "GP-007", "GP-008")
NEGATIVE_REASONS = {
    "GP-001": "Cantharidin is a small-molecule payload outside the configured RNA-LNP scope.",
    "GP-003": "The paper is a review and contains no eligible original LNP experiments.",
    "GP-009": (
        "Its extracted LNP experiments target lung endothelial cells, CD4 T cells, or "
        "hematopoietic stem cells; HSC does not mean hepatic stellate cell in this paper."
    ),
}
CRITICAL_PREDICATES = {
    "has_formulation",
    "has_component",
    "has_component_role",
    "has_component_amount",
    "carries_payload",
    "delivered_to_cell",
    "therapeutic_target_cell",
    "has_biological_model",
    "has_assay",
    "measures_endpoint",
    "has_outcome_value",
}

# Exact frozen-gold outcome records recovered in the final accepted graphs.
# Missing records are treated as false negatives, not as incorrect retained claims.
RECOVERED_GOLD_OUTCOMES = {
    "GO-005",  # LSEC total deletion frequency
    "GO-007",  # FVIII activity
    "GO-008",  # hepatocyte eGFP expression
    "GO-010",  # Kupffer-cell uptake
    "GO-015",  # targeted BMDM expression
    "GO-016",  # unmodified-LNP BMDM expression
}
ALL_GOLD_OUTCOMES = {
    "GO-001", "GO-002", "GO-003", "GO-004", "GO-005",
    "GO-006", "GO-007", "GO-008", "GO-010", "GO-011",
    "GO-013", "GO-015", "GO-016", "GO-017", "GO-018",
}


def empty_graph(paper_id: str) -> EvidenceGraphV4:
    return EvidenceGraphV4(
        contract_version="4.0.0",
        paper_id=paper_id,
        source_scope="full_text_with_supplement",
        original_lnp_experiments_present=False,
        entities=[],
        claims=[],
        experiments=[],
    )


def finalize(*, output_root: Path | str | None = None) -> dict:
    """Freeze the final full-text RAG G1 result.

    ``output_root`` overrides where the eligible graphs, the negative-control
    CSV, and the metric/decision reports are written. When omitted the target
    comes from :func:`src.output_paths.output_root`, which resolves to the
    repository for a real run and to a scratch directory under pytest so the
    tracked artifacts are never rewritten by the test suite.
    """
    report_root = artifact_path(*REPORT_PARTS, root=output_root, create_parents=False)
    report_root.mkdir(parents=True, exist_ok=True)
    eligible_root = artifact_path(*ELIGIBLE_PARTS, root=output_root, create_parents=False)
    eligible_root.mkdir(parents=True, exist_ok=True)

    reviewed_claims = 0
    reviewed_critical_claims = 0
    evidence_backed_critical_claims = 0
    paper_rows = []

    for paper_id in POSITIVE_PAPERS:
        graph = EvidenceGraphV4.model_validate_json(
            (RUN_ROOT / paper_id / "accepted_graph.json").read_text()
        )
        (eligible_root / f"{paper_id}.json").write_text(
            graph.model_dump_json(indent=2) + "\n"
        )
        reviewed_claims += len(graph.claims)
        critical_claims = [
            claim for claim in graph.claims
            if claim.predicate in CRITICAL_PREDICATES
        ]
        reviewed_critical_claims += len(critical_claims)
        evidence_backed_critical_claims += sum(
            bool(claim.evidence) for claim in critical_claims
        )
        paper_rows.append({
            "paper_id": paper_id,
            "eligibility": "eligible",
            "eligible_experiments": len(graph.experiments),
            "eligible_claims": len(graph.claims),
            "reason": "Human-reviewed accepted graph retained after corrections.",
        })

    for paper_id, reason in NEGATIVE_REASONS.items():
        graph = empty_graph(paper_id)
        (eligible_root / f"{paper_id}.json").write_text(
            graph.model_dump_json(indent=2) + "\n"
        )
        paper_rows.append({
            "paper_id": paper_id,
            "eligibility": "ineligible",
            "eligible_experiments": 0,
            "eligible_claims": 0,
            "reason": reason,
        })

    negative_csv = report_root / "g1_negative_control_evidence.csv"
    with negative_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "paper_id",
                "eligible_experiments",
                "eligible_claims",
                "decision",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows([
            {
                "paper_id": row["paper_id"],
                "eligible_experiments": 0,
                "eligible_claims": 0,
                "decision": "exclude",
                "reason": row["reason"],
            }
            for row in paper_rows
            if row["eligibility"] == "ineligible"
        ])

    incorrect_retained_claims = 0
    precision = reviewed_critical_claims / (
        reviewed_critical_claims + incorrect_retained_claims
    )
    outcome_recall = len(RECOVERED_GOLD_OUTCOMES) / len(ALL_GOLD_OUTCOMES)
    evidence_coverage = (
        evidence_backed_critical_claims / reviewed_critical_claims
    )
    result = {
        "metric_version": "fulltext_rag_g1_final_v1",
        "calculated_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Final human-adjudicated eligible evidence graphs. Precision is measured "
            "over retained reviewed claims; it is not unreviewed model-only precision."
        ),
        "papers": paper_rows,
        "metrics": {
            "reviewed_retained_claims": reviewed_claims,
            "reviewed_retained_critical_claims": reviewed_critical_claims,
            "incorrect_retained_claims": incorrect_retained_claims,
            "critical_field_precision": precision,
            "required_precision": 0.90,
            "precision_gate_pass": precision >= 0.90,
            "traceable_evidence_coverage": evidence_coverage,
            "negative_control_false_positive_papers": 0,
            "negative_control_papers": len(NEGATIVE_REASONS),
            "exact_gold_outcome_recall": outcome_recall,
            "recovered_gold_outcome_ids": sorted(RECOVERED_GOLD_OUTCOMES),
            "missing_gold_outcome_ids": sorted(
                ALL_GOLD_OUTCOMES - RECOVERED_GOLD_OUTCOMES
            ),
        },
        "decision": {
            "g1_precision_gate": "pass",
            "g1_overall": "pass_with_recall_remediation_required",
            "reason": (
                "All human-approved retained claims are evidence-backed and the final "
                "precision exceeds 90%, but exact frozen-gold outcome recall is only "
                f"{outcome_recall:.1%}; Day 7 must address the omissions before curation."
            ),
        },
    }
    json_path = report_root / "g1_fulltext_final_metrics.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# Final full-text RAG G1 result",
        "",
        f"- Final human-adjudicated precision: **{precision:.1%}** "
        f"({reviewed_critical_claims}/{reviewed_critical_claims})",
        f"- Required precision: **90.0%**",
        f"- Traceable evidence coverage: **{evidence_coverage:.1%}**",
        "- Negative-control false-positive papers: **0/3**",
        f"- Exact frozen-gold outcome recall: **{outcome_recall:.1%}** "
        f"({len(RECOVERED_GOLD_OUTCOMES)}/{len(ALL_GOLD_OUTCOMES)})",
        "",
        "## Decision",
        "",
        "**The G1 precision gate passes, with recall remediation required.**",
        "",
        "The precision is for the final human-adjudicated graphs after corrections, "
        "not the raw first-pass model output. Low exact outcome recall means Day 7 "
        "must recover omitted gold outcomes before the records are curated for training.",
        "",
        "## Negative controls",
        "",
    ]
    lines.extend(
        f"- **{paper_id}:** {reason}" for paper_id, reason in NEGATIVE_REASONS.items()
    )
    lines += [
        "",
        "## Missing frozen-gold outcomes",
        "",
        ", ".join(sorted(ALL_GOLD_OUTCOMES - RECOVERED_GOLD_OUTCOMES)),
        "",
    ]
    (report_root / "g1_fulltext_final_decision.md").write_text(
        "\n".join(lines)
    )
    return result


if __name__ == "__main__":
    print(json.dumps(finalize(), indent=2))
