"""Locally normalize, validate, and merge the completed GP-008 repair response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.extraction.compact_contracts import CompactExtractionResponse
from src.extraction.run_day5_gp008_repair import (
    BASE_PATH,
    MERGED_PATH,
    OUTPUT_ROOT,
    RepairFragment,
)
from src.rag.compact_api_packet import CompactApiPacket


ROOT = Path(__file__).resolve().parents[2]
PACKET_PATH = ROOT / "data/staging/rag/compact_api_packets_v1_1/GP-008.json"


def run() -> dict:
    response_paths = sorted(OUTPUT_ROOT.glob("*/response.json"))
    if len(response_paths) != 1:
        raise ValueError(f"Expected one completed response, found {len(response_paths)}")
    response_path = response_paths[0]
    response = json.loads(response_path.read_text(encoding="utf-8"))
    output_text = next(
        content["text"]
        for output in response["output"]
        if output.get("type") == "message"
        for content in output.get("content", [])
        if content.get("type") == "output_text"
    )
    raw = json.loads(output_text)

    # The contract represents non-exact inequalities in qualitative_outcome.
    # A field with status=missing may not cite value-support evidence.
    for outcome in raw["outcomes"]:
        if outcome["outcome_value"]["status"] == "missing":
            outcome["outcome_value"]["evidence_ids"] = []

    # The base result establishes that F2 is alpha-CD163/LNP-FAPCAR. The request
    # supplied only opaque IDs, so resolve the returned targeted experiment to F2.
    raw["experiment"]["formulation_id"] = "F2"
    fragment = RepairFragment.model_validate(raw)
    if fragment.experiment.experimental_context.value != "in_vitro":
        raise ValueError("Repair experiment must remain in vitro")
    if len(fragment.outcomes) != 2:
        raise ValueError("Repair must contain exactly two outcomes")
    qualitative = [
        row.qualitative_outcome.value or "" for row in fragment.outcomes
    ]
    if not any(">80%" in value for value in qualitative):
        raise ValueError("Repair lost the >80% inequality")
    if not any("<20%" in value for value in qualitative):
        raise ValueError("Repair lost the <20% inequality")

    base = CompactExtractionResponse.model_validate_json(
        BASE_PATH.read_text(encoding="utf-8")
    )
    merged_dict = base.model_dump(mode="json")
    merged_dict["experiments"].append(fragment.experiment.model_dump(mode="json"))
    merged_dict["outcomes"].extend(
        row.model_dump(mode="json") for row in fragment.outcomes
    )
    merged = CompactExtractionResponse.model_validate(merged_dict)
    packet = CompactApiPacket.model_validate_json(
        PACKET_PATH.read_text(encoding="utf-8")
    )
    merged.validate_evidence_ids({row.evidence_id for row in packet.evidence})

    run_dir = response_path.parent
    result_path = run_dir / "result.json"
    result_path.write_text(fragment.model_dump_json(indent=2) + "\n", encoding="utf-8")
    MERGED_PATH.parent.mkdir(parents=True, exist_ok=True)
    MERGED_PATH.write_text(merged.model_dump_json(indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "completed_validated_merged",
        "paper_id": "GP-008",
        "model_returned": response["model"],
        "response_id": response["id"],
        "paid_api_requests": 1,
        "usage": response.get("usage"),
        "normalizations": [
            "Cleared evidence_ids from missing inequality-valued fields; evidence remains on unit and qualitative outcome.",
            "Resolved the targeted experiment's opaque formulation reference to existing alpha-CD163 formulation F2.",
        ],
        "result_path": str(result_path),
        "merged_path": str(MERGED_PATH),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest



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
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
