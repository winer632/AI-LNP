"""Run the unpaid half of the repair route over an existing extraction result.

complexity -> validation -> candidate inventory -> coverage -> routing ->
missing-record tasks. Nothing here calls a model; the point is to see how much
paid work the routing actually asks for before committing to any of it.

Usage::

    python -m src.extraction.build_repair_route_local \
        <result-root> <packet-root> <output-root>

``run_enforced_compact_workflow_local`` covers the same ground for the frozen
nine-paper gold set with hard-coded roots. This variant is parameterised so it
can run over any extraction output, which is what makes it usable for a harness
comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.extraction.assess_outcome_complexity import assess  # noqa: E402
from src.extraction.build_missing_record_tasks import (  # noqa: E402
    build_text_tasks,
    build_visual_referrals,
)
from src.extraction.build_outcome_candidates import build_candidates  # noqa: E402
from src.extraction.check_outcome_coverage import check  # noqa: E402
from src.extraction.compact_validation import validate_candidate  # noqa: E402
from src.extraction.consolidate_outcome_candidates import consolidate  # noqa: E402
from src.extraction.route_compact_findings import route  # noqa: E402
from src.extraction.run_codex_one_call import load_packet  # noqa: E402

def run(
    result_root: Path,
    packet_root: Path,
    output_root: Path,
    *,
    repair_confidence_levels: tuple[str, ...] | None = None,
    max_candidates_per_task: int | None = None,
    only_papers: set[str] | None = None,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    totals = {"text_tasks": 0, "visual_referrals": 0, "papers": 0}
    for number in range(1, 10):
        paper_id = f"GP-{number:03d}"
        if only_papers and paper_id not in only_papers:
            continue
        result_path = result_root / paper_id / "result.json"
        if not result_path.exists():
            continue
        packet_path = packet_root / f"{paper_id}.json"
        if not packet_path.exists():
            continue

        packet = load_packet(paper_id, packet_root)
        complexity = assess(packet)
        result_text = result_path.read_text(encoding="utf-8")
        parsed, validation = validate_candidate(
            result_text,
            paper_id=paper_id,
            allowed_evidence_ids={row.evidence_id for row in packet.evidence},
        )
        if parsed is None:
            print(f"{paper_id}: result does not validate, skipping")
            continue

        raw = build_candidates(packet) if complexity.route == "complex" else []
        inventory = consolidate(
            paper_id=paper_id,
            source_packet_checksum=packet.packet_checksum,
            candidates=raw,
            packet=packet,
        )
        coverage = (
            check(
                packet,
                parsed.model_dump(mode="json"),
                assessment=complexity,
                candidates=inventory.retained_candidates,
                repair_confidence_levels=repair_confidence_levels,
            )
            if complexity.route == "complex"
            else None
        )
        routing = route(
            paper_id=paper_id,
            complexity_route=complexity.route,
            validation=validation,
            coverage=coverage,
            inventory=inventory,
        )
        tasks = build_text_tasks(
            routing=routing,
            inventory=inventory,
            evidence_packet=packet,
            result_path=result_path,
            max_candidates_per_task=max_candidates_per_task,
        )
        referrals = build_visual_referrals(routing=routing, inventory=inventory, evidence_packet=packet)

        paper_dir = output_root / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        (paper_dir / "routing.json").write_text(
            routing.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (paper_dir / "inventory.json").write_text(
            inventory.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        for index, task in enumerate(tasks, start=1):
            (paper_dir / f"task_{index:02d}.json").write_text(
                task.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )

        totals["papers"] += 1
        totals["text_tasks"] += len(tasks)
        totals["visual_referrals"] += len(referrals)
        unmatched = len(getattr(coverage, "unmatched_candidates", []) or []) if coverage else 0
        print(
            f"{paper_id}: route={complexity.route} validation={validation.status} "
            f"unmatched={unmatched} text_tasks={len(tasks)} visual={len(referrals)}"
        )

    return totals


def main() -> None:
    levels = tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    cap = int(sys.argv[5]) if len(sys.argv) > 5 else None
    only = set(sys.argv[6].split(",")) if len(sys.argv) > 6 else None
    totals = run(
        Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]),
        repair_confidence_levels=levels,
        max_candidates_per_task=cap,
        only_papers=only,
    )
    print("\n" + json.dumps(totals))


if __name__ == "__main__":
    main()
