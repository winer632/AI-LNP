"""Append validated omitted records without mutating the original result."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from src.config_flags import is_enabled
from src.extraction.assess_outcome_complexity import assess
from src.extraction.check_outcome_coverage import check
from src.extraction.compact_validation import validate_candidate
from src.extraction.missing_record_contracts import (
    MissingRecordFragment,
    MissingRecordVisionResponse,
    MissingRecordVisionTask,
)
from src.extraction.outcome_inventory_contracts import OutcomeInventory
from src.extraction.run_missing_record_repair import load_task, validate_response
from src.extraction.run_missing_record_vision import (
    load_task as load_vision_task,
    validate as validate_vision,
)
from src.idempotent_write import IDEMPOTENT_UPSERT_FLAG, upsert_json
from src.rag.compact_api_packet import CompactApiPacket


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def merge(
    *,
    result_path: Path,
    packet_path: Path,
    pairs: list[tuple[Path, Path]],
    vision_pairs: list[tuple[Path, Path]] | None = None,
    inventory_path: Path,
    output_path: Path,
) -> dict:
    # With the flag off this is the original guard: an existing output ends the
    # run before anything is read. With it on the guard moves to the write
    # below, which can tell "you already ran this" from "you are about to
    # replace it with something else" -- the merge is pure and local, so the
    # answer is computable rather than guessable. Nothing here is paid for, and
    # nothing on disk is touched until the comparison has been made.
    idempotent = is_enabled(IDEMPOTENT_UPSERT_FLAG)
    if output_path.exists() and not idempotent:
        raise FileExistsError("Refusing to overwrite an existing recovered result")
    source_bytes = result_path.read_bytes()
    merged = deepcopy(json.loads(source_bytes))
    packet = CompactApiPacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    )
    packet_unsigned = packet.model_dump(
        mode="json", exclude={"packet_checksum"}, exclude_none=True
    )
    packet_canonical = json.dumps(
        packet_unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if _sha(packet_canonical) != packet.packet_checksum:
        raise ValueError("Evidence packet checksum mismatch")
    if merged.get("paper_id") != packet.paper_id:
        raise ValueError("Result and evidence packet paper IDs differ")
    existing_experiments = {
        row["experiment_id"] for row in merged.get("experiments", [])
    }
    existing_outcomes = {row["outcome_id"] for row in merged.get("outcomes", [])}
    recovered_candidates: set[str] = set()
    unresolved_candidates: set[str] = set()
    inputs = []
    prepared = []
    for task_path, fragment_path in pairs:
        task = load_task(task_path)
        fragment = MissingRecordFragment.model_validate_json(
            fragment_path.read_text(encoding="utf-8")
        )
        validate_response(fragment, task)
        prepared.append((task, fragment, task_path, fragment_path, None))
    for task_path, response_path in vision_pairs or []:
        task = load_vision_task(task_path)
        response = MissingRecordVisionResponse.model_validate_json(
            response_path.read_text(encoding="utf-8")
        )
        validate_vision(response, task)
        prepared.append(
            (task, response.fragment, task_path, response_path, task.crop_evidence_id)
        )
    allowed_evidence = {row.evidence_id for row in packet.evidence}
    for task, fragment, task_path, fragment_path, crop_evidence_id in prepared:
        if task.paper_id != packet.paper_id:
            raise ValueError("Task belongs to a different paper")
        if task.source_result_sha256 != _sha(source_bytes):
            raise ValueError("Task was built from a different source result")
        new_experiment_ids = {row.experiment_id for row in fragment.experiments}
        new_outcome_ids = {row.outcome_id for row in fragment.outcomes}
        if existing_experiments & new_experiment_ids:
            raise ValueError("Experiment ID collision across fragments")
        if existing_outcomes & new_outcome_ids:
            raise ValueError("Outcome ID collision across fragments")
        merged["experiments"].extend(
            row.model_dump(mode="json") for row in fragment.experiments
        )
        merged["outcomes"].extend(
            row.model_dump(mode="json") for row in fragment.outcomes
        )
        existing_experiments |= new_experiment_ids
        existing_outcomes |= new_outcome_ids
        recovered_candidates |= set(fragment.recovered_candidate_ids)
        unresolved_candidates |= set(fragment.unresolved_candidate_ids)
        if crop_evidence_id:
            allowed_evidence.add(crop_evidence_id)
        inputs.append(
            {
                "task_path": str(task_path),
                "fragment_path": str(fragment_path),
                "candidate_ids": task.candidate_ids,
                "disposition": fragment.disposition,
            }
        )
    if recovered_candidates & unresolved_candidates:
        raise ValueError("Conflicting candidate dispositions across fragments")
    parsed, validation = validate_candidate(
        json.dumps(merged, ensure_ascii=False),
        paper_id=packet.paper_id,
        allowed_evidence_ids=allowed_evidence,
    )
    if parsed is None or validation.status != "valid":
        raise ValueError(
            "Merged missing records failed compact validation: "
            + "; ".join(row.message for row in validation.findings)
        )
    inventory = OutcomeInventory.model_validate_json(
        inventory_path.read_text(encoding="utf-8")
    )
    if inventory.paper_id != packet.paper_id:
        raise ValueError("Inventory belongs to a different paper")
    inventory_sha = hashlib.sha256(
        inventory.model_dump_json(exclude_none=True).encode()
    ).hexdigest()
    if any(task.source_inventory_sha256 != inventory_sha for task, *_ in prepared):
        raise ValueError("A task was built from a different outcome inventory")
    coverage = check(
        packet,
        parsed.model_dump(mode="json"),
        assessment=assess(packet),
        candidates=inventory.retained_candidates,
    )
    coverage_complete = coverage.status in {
        "complete",
        "not_applicable",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Same inputs converge: byte-identical content is not rewritten and the
    # rerun is a no-op. Different inputs are refused, because a recovered
    # result that disagrees with the one already on disk is exactly the case
    # the original FileExistsError was protecting.
    upsert_json(
        output_path,
        parsed.model_dump(mode="json"),
        on_conflict="refuse",
        hint=(
            "A recovered result for this paper already exists and differs. "
            "Re-run with the same inputs, or write to a new --output-path."
        ),
        enabled=idempotent,
    )
    report = {
        "merge_version": "missing-record-merge-1.0.0",
        "paper_id": packet.paper_id,
        "source_result_sha256": _sha(source_bytes),
        "result_path": str(output_path),
        "recovered_candidate_ids": sorted(recovered_candidates),
        "unresolved_candidate_ids": sorted(unresolved_candidates),
        "inputs": inputs,
        "validation_status": validation.status,
        "coverage_status": coverage.status if coverage else None,
        "coverage_unresolved_candidate_ids": (
            sorted(
                {
                    row.candidate_id
                    for row in [
                        *coverage.unmatched_candidates,
                        *coverage.review_candidates,
                    ]
                }
            )
            if coverage
            else []
        ),
        "finalization_allowed": not unresolved_candidates and coverage_complete,
    }
    upsert_json(
        output_path.parent / "missing_record_merge_report.json",
        report,
        on_conflict="refuse",
        hint="The merge report for this paper already exists and differs.",
        enabled=idempotent,
    )
    return report
