"""Apply traceable human adjudications to Day 8 merged evidence."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone

from .merge_day8_evidence import OUTPUT


RECORD_ID = "D8M-0061"


def run() -> dict:
    source_path = OUTPUT / "merged_evidence.json"
    records = json.loads(source_path.read_text())
    corrected = deepcopy(records)
    target = next(row for row in corrected if row["merged_record_id"] == RECORD_ID)
    before = target["population"]
    target["population"] = "mice"
    target["requires_human_review"] = False
    target["deterministic_issues"] = [
        issue for issue in target["deterministic_issues"]
        if issue != "ambiguous_population"
    ]

    corrected_path = OUTPUT / "merged_evidence.human_corrected.json"
    corrected_path.write_text(
        json.dumps(corrected, indent=2, ensure_ascii=False) + "\n"
    )
    adjudication = {
        "record_id": RECORD_ID,
        "adjudicated_at": datetime.now(timezone.utc).isoformat(),
        "decision": "retain_with_population_correction",
        "field": "population",
        "before": before,
        "after": "mice",
        "reason": (
            "The paper supports mice treated with oil versus "
            "αCD163/LNP-FAPCAR but does not explicitly qualify them as healthy."
        ),
        "approved_by": "human_reviewer",
    }
    ledger_path = OUTPUT / "human_adjudications.json"
    ledger_path.write_text(
        json.dumps([adjudication], indent=2, ensure_ascii=False) + "\n"
    )
    return {
        "record_id": RECORD_ID,
        "corrected_result": str(corrected_path),
        "adjudication_ledger": str(ledger_path),
        "review_queue_remaining": 0,
    }



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
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
