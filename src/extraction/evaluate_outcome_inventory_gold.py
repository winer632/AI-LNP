"""Evaluate full-evidence candidate recall against the frozen outcome set."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from src.extraction.build_full_outcome_inventory import (
    CORPUS_ROOT,
    full_corpus_view,
    load_and_build,
)
from src.rag.compact_api_packet import _same_gold_location
from src.rag.compact_packet import CompactEvidencePacket


ROOT = Path(__file__).resolve().parents[2]
GOLD_ROOT = ROOT / "data/annotations/gold_v1"
PACKET_ROOT = ROOT / "data/staging/rag/compact_packets_v1"
OUTPUT_ROOT = ROOT / "reports/extraction/full_outcome_inventory_gold_v1"
STOP = {
    "the", "and", "of", "in", "to", "a", "an", "was", "were", "with",
    "from", "for", "after", "outcome", "cells", "cell", "reported",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def tokens(text: str) -> set[str]:
    return {
        value
        for value in re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
        if len(value) >= 2 and value not in STOP
    }


def evaluate(
    *,
    gold_root: Path = GOLD_ROOT,
    packet_root: Path = PACKET_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> dict:
    outcomes = rows(gold_root / "outcomes.csv")
    evidence = {
        row["evidence_id"]: row for row in rows(gold_root / "evidence.csv")
    }
    experiments = {
        row["gold_experiment_id"]: row
        for row in rows(gold_root / "experiments.csv")
    }
    paper_cache = {}
    results = []
    for gold in outcomes:
        paper_id = experiments[gold["gold_experiment_id"]]["gold_paper_id"]
        if paper_id not in paper_cache:
            packet_path = packet_root / f"{paper_id}.json"
            packet = CompactEvidencePacket.model_validate_json(
                packet_path.read_text(encoding="utf-8")
            )
            inventory = load_and_build(packet_path)
            full_view = full_corpus_view(
                packet, CORPUS_ROOT / f"{paper_id}.blocks.jsonl"
            )
            source_by_id = {row.source_id: row for row in full_view.sources}
            paper_cache[paper_id] = (inventory, source_by_id)
        inventory, source_by_id = paper_cache[paper_id]
        gold_evidence = evidence[gold["evidence_id"]]
        expected_tokens = tokens(
            " ".join(
                [
                    gold["endpoint_name"],
                    gold["qualitative_outcome"],
                    gold_evidence["evidence_text"],
                ]
            )
        )
        matches = []
        for candidate in inventory.retained_candidates:
            candidate_locations = [
                source_by_id[source_id].model_dump(mode="json", exclude_none=True)
                for source_id in candidate.source_ids
                if source_id in source_by_id
            ]
            location_match = any(
                _same_gold_location(gold_evidence, location)
                for location in candidate_locations
            )
            overlap = sorted(expected_tokens & tokens(candidate.evidence_text))
            distinctive_overlap = len(overlap) >= 2
            value_tokens = tokens(gold["outcome_value"])
            strong_text_match = bool(
                len(overlap) >= 4
                and value_tokens
                and value_tokens <= set(overlap)
            )
            if (location_match and distinctive_overlap) or strong_text_match:
                matches.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "endpoint_family": candidate.endpoint_family,
                        "confidence": candidate.confidence,
                        "route_hint": candidate.route_hint,
                        "overlap_terms": overlap,
                        "evidence_ids": candidate.evidence_ids,
                    }
                )
        results.append(
            {
                "gold_outcome_id": gold["gold_outcome_id"],
                "paper_id": paper_id,
                "evidence_id": gold["evidence_id"],
                "recalled": bool(matches),
                "candidate_matches": matches,
            }
        )
    recalled = sum(row["recalled"] for row in results)
    summary = {
        "evaluation_version": "full-outcome-inventory-gold-1.0.0",
        "scope": "local full-evidence candidate recall; no API calls",
        "recalled": recalled,
        "total": len(results),
        "rate": recalled / len(results),
        "missing_gold_outcome_ids": [
            row["gold_outcome_id"] for row in results if not row["recalled"]
        ],
        "results": results,
        "paid_api_requests": 0,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary



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
    print(json.dumps(evaluate(), indent=2))


if __name__ == "__main__":
    main()
