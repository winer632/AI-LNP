"""Run the frozen nine-paper local outcome-coverage benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.extraction.assess_outcome_complexity import assess
from src.extraction.build_outcome_candidates import build_candidates
from src.extraction.check_outcome_coverage import check
from src.rag.compact_api_packet import CompactApiPacket


ROOT = Path(__file__).resolve().parents[2]
PACKET_ROOT = ROOT / "data" / "staging" / "rag" / "compact_api_packets_v1"
RESULT_ROOT = ROOT / "data" / "staging" / "extraction" / "compact_one_call_v1"
OUTPUT_ROOT = ROOT / "reports" / "extraction" / "day5_outcome_coverage"


def _result_path(paper_id: str, result_root: Path) -> Path:
    preferred = [
        ROOT
        / "data/staging/extraction/compact_merged_v1_1"
        / paper_id
        / "final_result.json",
        ROOT
        / "data/staging/extraction/compact_merged_v1"
        / paper_id
        / "final_result.json",
        result_root / paper_id / "result.json",
    ]
    return next(path for path in preferred if path.exists())


def run(
    output_root: Path = OUTPUT_ROOT,
    *,
    packet_root: Path = PACKET_ROOT,
    result_root: Path = RESULT_ROOT,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    papers = []
    for number in range(1, 10):
        paper_id = f"GP-{number:03d}"
        packet = CompactApiPacket.model_validate_json(
            (packet_root / f"{paper_id}.json").read_text(encoding="utf-8")
        )
        result = json.loads(
            _result_path(paper_id, result_root).read_text(encoding="utf-8")
        )
        assessment = assess(packet)
        candidates = build_candidates(packet) if assessment.route == "complex" else []
        coverage = check(
            packet, result, assessment=assessment, candidates=candidates
        )
        paper_dir = output_root / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        (paper_dir / "complexity.json").write_text(
            assessment.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (paper_dir / "coverage.json").write_text(
            coverage.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        papers.append(
            {
                "paper_id": paper_id,
                "route": assessment.route,
                "complexity_score": assessment.complexity_score,
                "candidate_groups": coverage.candidate_groups,
                "extracted_outcomes": coverage.extracted_outcomes,
                "matched_candidate_groups": len(coverage.matched_candidates),
                "unmatched_candidate_groups": len(coverage.unmatched_candidates),
                "unmatched_text_groups": sum(
                    row.route_hint == "text"
                    for row in coverage.unmatched_candidates
                ),
                "unmatched_vision_groups": sum(
                    row.route_hint == "vision"
                    for row in coverage.unmatched_candidates
                ),
                "review_only_groups": len(coverage.review_candidates),
                "status": coverage.status,
                "merge_allowed": (
                    assessment.route == "simple"
                    or coverage.status in {"complete", "not_applicable"}
                ),
            }
        )
    summary = {
        "benchmark_version": "day5-outcome-coverage-1.1.0",
        "rules_status": "calibrated_local_gate_v1",
        "safety_policy": {
            "actionable_unmatched_groups_block_merge": True,
            "medium_confidence_groups_trigger_paid_calls": False,
            "overlapping_duplicate_groups_trigger_paid_calls": False,
            "automatic_paid_repair_calls": False,
        },
        "paid_api_requests": 0,
        "first_call_tokens_added": 0,
        "papers": papers,
        "counts": {
            "simple": sum(row["route"] == "simple" for row in papers),
            "complex": sum(row["route"] == "complex" for row in papers),
            "unmatched_text_groups": sum(
                row["unmatched_text_groups"] for row in papers
            ),
            "unmatched_vision_groups": sum(
                row["unmatched_vision_groups"] for row in papers
            ),
            "review_only_groups": sum(row["review_only_groups"] for row in papers),
            "merge_blocked": sum(not row["merge_allowed"] for row in papers),
        },
    }
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
