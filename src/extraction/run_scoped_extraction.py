"""Extract one endpoint family per call instead of a whole paper per call.

Why
---
Three independent measurements this round all point the same way: the smaller
the unit a single turn has to account for, the more it recovers.

    candidate slots      39 answered cleanly        168 fabricated or unanswered
    repair tasks         1 candidate recovers GO-007   7 candidates recover nothing
    architecture         v4 per-experiment 12/15    whole-paper compact 10/15

Whole-paper extraction is the failure mode behind "recovers GO-006 but drops
GO-004 and GO-007": those three are different kinds of measurement --
cell_type_selectivity, gene_editing and therapeutic_function -- competing for
attention in one turn. Splitting by endpoint family gives each its own turn.

``experiment_candidate_ids`` would have been the natural boundary, but the field
is empty in every packet the pipeline produces, so the endpoint family carried
by the outcome candidates is used instead. It is the strongest boundary signal
that is actually populated.

Every sub-call receives the same packet contract and the same strict schema, so
nothing downstream needs to know the extraction was split.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.extraction.assess_outcome_complexity import assess
from src.extraction.build_outcome_candidates import build_candidates
from src.extraction.compact_prompt_v1 import candidate_slot_payload
from src.extraction.compact_validation import validate_candidate
from src.extraction.run_codex_one_call import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TIMEOUT_SECONDS,
    HARNESS,
    PACKET_ROOT_BY_VIEW,
    build_prompt,
    codex_available,
    load_packet,
    run_codex,
    strict_schema,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "staging" / "extraction" / "codex_scoped_v1"


def group_candidates_by_family(candidates) -> dict[str, list]:
    """Group outcome candidates into one bucket per endpoint family."""
    groups: dict[str, list] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.endpoint_family or "unspecified"].append(candidate)
    return dict(groups)


def _scope_note(family: str, total: int) -> str:
    return (
        f"\n\nSCOPE: this call covers the {family} endpoint family only "
        f"({total} candidate slots). Other families are handled by separate "
        "calls, so do not report outcomes outside this family, and do not "
        "treat their absence as a missing value."
    )


def run_paper(
    paper_id: str,
    *,
    packet_root: Path,
    output_root: Path,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_slots_per_call: int = 40,
) -> dict[str, Any]:
    packet = load_packet(paper_id, packet_root)
    complexity = assess(packet)
    candidates = build_candidates(packet) if complexity.route == "complex" else []
    groups = group_candidates_by_family(candidates)

    # A family larger than the working range is chunked, for the same reason the
    # families are split at all.
    scopes: list[tuple[str, list]] = []
    for family, rows in sorted(groups.items()):
        for index in range(0, len(rows), max_slots_per_call):
            scopes.append((family, rows[index : index + max_slots_per_call]))
    if not scopes:
        scopes = [("all", [])]

    run_dir = output_root / paper_id
    run_dir.mkdir(parents=True, exist_ok=True)
    schema = strict_schema(bool(candidates))

    merged_outcomes: list[dict[str, Any]] = []
    merged_experiments: list[dict[str, Any]] = []
    seen_outcomes: set[tuple] = set()
    seen_experiments: set[str] = set()
    base: dict[str, Any] | None = None
    per_scope = []

    for index, (family, rows) in enumerate(scopes, start=1):
        slots = candidate_slot_payload(rows) if rows else None
        prompt = build_prompt(packet, slots) + _scope_note(family, len(rows))
        response = run_codex(
            prompt, schema, model=model, reasoning_effort=reasoning_effort, timeout=timeout
        )
        parsed, validation = validate_candidate(
            response["text"],
            paper_id=paper_id,
            allowed_evidence_ids={row.evidence_id for row in packet.evidence},
            # Only this scope's slots are sent, so only they may be required.
            # Passing them also selects the response contract that permits
            # candidate_dispositions; omitting it parses against the contract
            # that forbids the field and rejects every reply.
            required_candidate_ids=(
                [row.candidate_id for row in rows] if slots else None
            ),
        )
        scope_dir = run_dir / f"scope_{index:02d}_{family}"
        scope_dir.mkdir(parents=True, exist_ok=True)
        (scope_dir / "response.json").write_text(
            json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (scope_dir / "validation_report.json").write_text(
            validation.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        record = {
            "family": family,
            "slots": len(rows),
            "validation_status": validation.status,
            "outcomes": 0,
            "elapsed_seconds": response["elapsed_seconds"],
        }
        if parsed is not None:
            data = parsed.model_dump(mode="json")
            if base is None:
                base = {k: v for k, v in data.items() if k not in {"outcomes", "experiments"}}
            for experiment in data.get("experiments", []):
                key = str(experiment.get("experiment_id"))
                if key in seen_experiments:
                    continue
                seen_experiments.add(key)
                merged_experiments.append(experiment)
            for outcome in data.get("outcomes", []):
                def field(name):
                    row = outcome.get(name)
                    return row.get("value") if isinstance(row, dict) else row

                key = (
                    str(field("endpoint") or "").strip().lower(),
                    str(field("outcome_value") or "").strip().lower(),
                )
                if key in seen_outcomes:
                    continue
                seen_outcomes.add(key)
                row = dict(outcome)
                row["source_scope"] = family
                merged_outcomes.append(row)
            record["outcomes"] = len(data.get("outcomes", []))
        per_scope.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    if base is None:
        base = {
            "contract_version": "compact-1.1.0",
            "paper_id": paper_id,
            "eligibility": {
                "decision": "ineligible",
                "reason_codes": ["ORIGINAL_EXPERIMENT"],
                "evidence_ids": [],
                "explanation": "No scope returned a parsable result.",
            },
            "formulations": [],
            "components": [],
            "unresolved_items": [],
        }
    base["experiments"] = merged_experiments
    base["outcomes"] = merged_outcomes
    (run_dir / "final_result.json").write_text(
        json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "harness": HARNESS,
        "extraction": "endpoint-family-scoped",
        "paper_id": paper_id,
        "scopes": per_scope,
        "total_scopes": len(scopes),
        "merged_outcomes": len(merged_outcomes),
        "merged_experiments": len(merged_experiments),
        "openai_api_requests": 0,
        "codex_exec_turns": len(scopes),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract one endpoint family per call instead of one paper per call."
    )
    parser.add_argument("--paper-id", action="append", dest="paper_ids")
    parser.add_argument("--evidence-view", choices=sorted(PACKET_ROOT_BY_VIEW), default="compact")
    parser.add_argument("--packet-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-slots-per-call", type=int, default=40)
    parser.add_argument("--confirm-codex-quota", action="store_true")
    args = parser.parse_args()

    if not args.confirm_codex_quota:
        parser.error("--confirm-codex-quota is required")
    if not codex_available():
        parser.error("`codex` was not found on PATH")

    packet_root = args.packet_root or PACKET_ROOT_BY_VIEW[args.evidence_view]
    for paper_id in args.paper_ids or [f"GP-{n:03d}" for n in range(1, 10)]:
        try:
            manifest = run_paper(
                paper_id,
                packet_root=packet_root,
                output_root=args.output_root,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout=args.timeout,
                max_slots_per_call=args.max_slots_per_call,
            )
        except Exception as error:  # noqa: BLE001
            manifest = {"paper_id": paper_id, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
