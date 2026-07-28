"""Prepare the evidence-bounded GP-008 repair required by the Day 5 G1 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.extraction.assess_outcome_complexity import assess
from src.extraction.build_outcome_candidates import build_candidates
from src.extraction.check_outcome_coverage import check
from src.rag.compact_api_packet import CompactApiPacket


ROOT = Path(__file__).resolve().parents[2]
PACKET_PATH = (
    ROOT / "data/staging/rag/compact_api_packets_v1_1/GP-008.json"
)
RESULT_PATH = (
    ROOT / "data/staging/extraction/compact_merged_v1/GP-008/final_result.json"
)
OUTPUT_ROOT = ROOT / "reports/extraction/day5_afternoon_g1"


def run() -> dict:
    packet = CompactApiPacket.model_validate_json(
        PACKET_PATH.read_text(encoding="utf-8")
    )
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assessment = assess(packet)
    candidates = build_candidates(packet)
    coverage = check(
        packet,
        result,
        assessment=assessment,
        candidates=candidates,
    )
    actionable_ids = {
        row.candidate_id for row in coverage.unmatched_candidates
    }
    actionable = [
        row for row in candidates if row.candidate_id in actionable_ids
    ]
    evidence_by_id = {row.evidence_id: row for row in packet.evidence}

    request = {
        "task_version": "day5-g1-text-repair-1.0.0",
        "paper_id": packet.paper_id,
        "status": "ready_for_human_authorization",
        "paid_api_call_made": False,
        "route": "narrow_text_repair",
        "source_packet": str(PACKET_PATH),
        "source_packet_checksum": packet.packet_checksum,
        "base_result": str(RESULT_PATH),
        "why": (
            "Restore two frozen-gold in-vitro BMDM percentage outcomes "
            "that were omitted by compact packet v1 ranking."
        ),
        "required_fragment": {
            "experiments": 1,
            "outcomes": 2,
            "instructions": [
                "Create a separate in-vitro BMDM experiment; do not attach these outcomes to the existing in-vivo E1 experiment.",
                "Return the targeted alpha-CD163 LNP expression/delivery result above 80%.",
                "Return the unmodified-LNP comparator expression/delivery result below 20%.",
                "Use only the supplied evidence IDs and preserve the inequality signs.",
            ],
        },
        "candidates": [
            {
                **row.model_dump(mode="json"),
                "evidence": [
                    evidence_by_id[evidence_id].model_dump(
                        mode="json", exclude_none=True
                    )
                    for evidence_id in row.evidence_ids
                ],
            }
            for row in actionable
        ],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    coverage_path = OUTPUT_ROOT / "GP-008-v1_1-coverage.json"
    request_path = OUTPUT_ROOT / "GP-008-repair-request.json"
    coverage_path.write_text(
        coverage.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    request_path.write_text(
        json.dumps(request, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "coverage_path": str(coverage_path),
        "request_path": str(request_path),
        "actionable_groups": len(actionable),
        "paid_api_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Required: this rewrites tracked report files in place.",
    )
    args = parser.parse_args()
    if not args.confirm_write:
        parser.error("--confirm-write is required; this rewrites tracked files")
    print(json.dumps(run(), indent=2, default=str))


if __name__ == "__main__":
    main()
