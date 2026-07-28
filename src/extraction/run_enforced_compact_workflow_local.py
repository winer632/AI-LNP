"""Run the enforced complexity, inventory, coverage, and routing path locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.extraction.assess_outcome_complexity import assess
from src.extraction.build_full_outcome_inventory import (
    CORPUS_ROOT,
    full_corpus_view,
    load_and_build,
)
from src.extraction.build_missing_record_tasks import (
    build_text_tasks,
    build_visual_referrals,
)
from src.extraction.check_outcome_coverage import check
from src.extraction.compact_validation import validate_candidate
from src.extraction.route_compact_findings import route
from src.rag.compact_packet import CompactEvidencePacket


ROOT = Path(__file__).resolve().parents[2]
PACKET_ROOT = ROOT / "data/staging/rag/compact_packets_v1"
RESULT_ROOT = ROOT / "data/staging/extraction/compact_one_call_v1"
OUTPUT_ROOT = ROOT / "reports/extraction/enforced_compact_workflow_v1"


def result_path(paper_id: str) -> Path:
    preferred = [
        ROOT
        / "data/staging/extraction/compact_merged_v1_1"
        / paper_id
        / "final_result.json",
        ROOT
        / "data/staging/extraction/compact_merged_v1"
        / paper_id
        / "final_result.json",
        RESULT_ROOT / paper_id / "result.json",
    ]
    return next(path for path in preferred if path.exists())


def run(
    *,
    packet_root: Path = PACKET_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> dict:
    papers = []
    for number in range(1, 10):
        paper_id = f"GP-{number:03d}"
        packet_path = packet_root / f"{paper_id}.json"
        packet = CompactEvidencePacket.model_validate_json(
            packet_path.read_text(encoding="utf-8")
        )
        full_view = full_corpus_view(
            packet, CORPUS_ROOT / f"{paper_id}.blocks.jsonl"
        )
        inventory = load_and_build(packet_path)
        complexity = assess(full_view)
        current_result_path = result_path(paper_id)
        result_text = current_result_path.read_text(encoding="utf-8")
        result = json.loads(result_text)
        parsed, validation = validate_candidate(
            result_text,
            paper_id=paper_id,
            allowed_evidence_ids={
                row.evidence_id for row in full_view.evidence
            },
        )
        eligible = (
            parsed is not None
            and parsed.eligibility.decision == "eligible"
        )
        coverage = (
            check(
                full_view,
                parsed.model_dump(mode="json"),
                assessment=complexity,
                candidates=inventory.retained_candidates,
            )
            if parsed is not None
            else None
        )
        routing = route(
            paper_id=paper_id,
            complexity_route=complexity.route,
            validation=validation,
            coverage=coverage,
            inventory=inventory if eligible else None,
        )
        text_tasks = (
            build_text_tasks(
                routing=routing,
                inventory=inventory,
                evidence_packet=full_view,
                result_path=current_result_path,
            )
            if eligible
            else []
        )
        visual_referrals = (
            build_visual_referrals(
                routing=routing,
                inventory=inventory,
                evidence_packet=full_view,
            )
            if eligible
            else []
        )
        paper_root = output_root / paper_id
        paper_root.mkdir(parents=True, exist_ok=True)
        (paper_root / "complexity.json").write_text(
            complexity.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (paper_root / "inventory.json").write_text(
            inventory.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (paper_root / "validation.json").write_text(
            validation.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        if coverage:
            (paper_root / "coverage.json").write_text(
                coverage.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
        (paper_root / "routing.json").write_text(
            routing.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        task_root = paper_root / "text_tasks"
        task_root.mkdir(exist_ok=True)
        for index, task in enumerate(text_tasks, 1):
            (task_root / f"task_{index:02d}.json").write_text(
                task.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
        referral_root = paper_root / "vision_referrals"
        referral_root.mkdir(exist_ok=True)
        for index, referral in enumerate(visual_referrals, 1):
            (referral_root / f"referral_{index:02d}.json").write_text(
                referral.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
        papers.append(
            {
                "paper_id": paper_id,
                "complexity_route": complexity.route,
                "eligibility": (
                    parsed.eligibility.decision if parsed else "invalid"
                ),
                "raw_candidates": inventory.raw_candidate_count,
                "retained_candidates": len(inventory.retained_candidates),
                "inventory_review": len(inventory.unresolved_dispositions),
                "coverage_status": coverage.status if coverage else None,
                "matched": (
                    len(coverage.matched_candidates) if coverage else 0
                ),
                "unmatched_text": (
                    sum(
                        row.route_hint == "text"
                        for row in coverage.unmatched_candidates
                    )
                    if coverage
                    else 0
                ),
                "unmatched_vision": (
                    sum(
                        row.route_hint == "vision"
                        for row in coverage.unmatched_candidates
                    )
                    if coverage
                    else 0
                ),
                "review": (
                    len(coverage.review_candidates) if coverage else 0
                ),
                "text_tasks": len(text_tasks),
                "vision_referrals": len(visual_referrals),
                "finalization_allowed": routing.finalization_allowed,
                "blockers": routing.finalization_blockers,
            }
        )
    summary = {
        "workflow_version": "enforced-compact-workflow-1.0.0",
        "scope": "local cached results; no API calls",
        "paid_api_requests": 0,
        "papers": papers,
        "counts": {
            "finalization_allowed": sum(
                row["finalization_allowed"] for row in papers
            ),
            "finalization_blocked": sum(
                not row["finalization_allowed"] for row in papers
            ),
            "text_tasks": sum(row["text_tasks"] for row in papers),
            "vision_referrals": sum(
                row["vision_referrals"] for row in papers
            ),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help=(
            "Required: this regenerates committed artifacts in place. Without "
            "it the command explains itself and writes nothing."
        ),
    )
    args = parser.parse_args()
    if not args.confirm_write:
        parser.error("--confirm-write is required; this rewrites tracked files")
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
