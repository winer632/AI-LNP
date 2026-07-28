"""Build grouped text-recovery tasks and visual referrals without API calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.extraction.compact_routing_contracts import CompactRoutingDecision
from src.extraction.missing_record_contracts import (
    MissingRecordTask,
    MissingRecordVisionReferral,
)
from src.extraction.outcome_inventory_contracts import OutcomeInventory
from src.extraction.repair_contracts import RepairEvidence
from src.rag.compact_api_packet import CompactApiPacket


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def _components(candidate_ids: list[str], inventory: OutcomeInventory) -> list[list[str]]:
    candidate_by_id = {
        row.candidate_id: row for row in inventory.retained_candidates
    }
    remaining = set(candidate_ids)
    groups = []
    while remaining:
        seed = remaining.pop()
        group = {seed}
        sources = set(candidate_by_id[seed].source_ids)
        changed = True
        while changed:
            changed = False
            for candidate_id in list(remaining):
                candidate = candidate_by_id[candidate_id]
                if sources & set(candidate.source_ids):
                    remaining.remove(candidate_id)
                    group.add(candidate_id)
                    sources |= set(candidate.source_ids)
                    changed = True
        groups.append(sorted(group))
    return groups


def build_text_tasks(
    *,
    routing: CompactRoutingDecision,
    inventory: OutcomeInventory,
    evidence_packet: CompactApiPacket,
    result_path: Path,
    max_candidates_per_task: int | None = None,
) -> list[MissingRecordTask]:
    """Build the missing-record repair tasks for one paper.

    ``max_candidates_per_task`` caps how many candidates share a task. Measured
    on GP-006: a 7-candidate task failed twice -- invalid, then all-unresolved --
    while the same seven candidates over the same twelve evidence rows, one per
    task, returned five recovered fragments including GO-007's 3.3% FVIII
    activity. The evidence was never the problem; how much the model had to
    account for in one turn was. Splitting multiplies the call count by the
    component size, so it is opt-in.
    """
    candidate_by_id = {
        row.candidate_id: row for row in inventory.retained_candidates
    }
    evidence_by_id = {row.evidence_id: row for row in evidence_packet.evidence}
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    route_by_candidate = {
        row.source_id: row
        for row in routing.routes
        if row.route == "missing_record_text"
    }
    components = _components(list(route_by_candidate), inventory)
    if max_candidates_per_task is not None:
        if max_candidates_per_task < 1:
            raise ValueError("max_candidates_per_task must be >= 1")
        components = [
            component[index : index + max_candidates_per_task]
            for component in components
            for index in range(0, len(component), max_candidates_per_task)
        ]

    tasks = []
    for candidate_ids in components:
        evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for candidate_id in candidate_ids
                for evidence_id in candidate_by_id[candidate_id].evidence_ids
            )
        )[:12]
        unsigned = {
            "task_version": "missing-record-task-1.0.0",
            "paper_id": routing.paper_id,
            "route_ids": [
                route_by_candidate[candidate_id].route_id
                for candidate_id in candidate_ids
            ],
            "candidate_ids": candidate_ids,
            "evidence": [
                RepairEvidence(
                    evidence_id=evidence_by_id[evidence_id].evidence_id,
                    text=evidence_by_id[evidence_id].text,
                    source_ids=evidence_by_id[evidence_id].source_ids,
                ).model_dump(mode="json")
                for evidence_id in evidence_ids
            ],
            "existing_formulation_ids": [
                row["formulation_id"] for row in result.get("formulations", [])
            ],
            "existing_experiment_ids": [
                row["experiment_id"] for row in result.get("experiments", [])
            ],
            "existing_outcome_ids": [
                row["outcome_id"] for row in result.get("outcomes", [])
            ],
            "permitted_new_experiments": 1,
            "permitted_new_outcomes": min(8, max(1, len(candidate_ids) * 2)),
            "source_result_sha256": _sha(result_bytes),
            "source_inventory_sha256": _sha(
                inventory.model_dump_json(exclude_none=True)
            ),
        }
        tasks.append(
            MissingRecordTask.model_validate(
                {**unsigned, "task_checksum": _sha(_canonical(unsigned))}
            )
        )
    return tasks


def build_visual_referrals(
    *,
    routing: CompactRoutingDecision,
    inventory: OutcomeInventory,
    evidence_packet: CompactApiPacket,
) -> list[MissingRecordVisionReferral]:
    candidate_by_id = {
        row.candidate_id: row for row in inventory.retained_candidates
    }
    source_by_id = {row.source_id: row for row in evidence_packet.sources}
    referrals = []
    for route in routing.routes:
        if route.route != "selective_vision" or route.source != "outcome_coverage":
            continue
        candidate = candidate_by_id[route.source_id]
        visual_sources = [
            source_by_id[source_id]
            for source_id in candidate.source_ids
            if source_id in source_by_id
            and source_by_id[source_id].page_number is not None
            and source_by_id[source_id].source_path.lower().endswith(".pdf")
        ]
        unique = {
            (row.source_path, row.page_number): row for row in visual_sources
        }
        if len(unique) != 1:
            continue
        source = next(iter(unique.values()))
        referrals.append(
            MissingRecordVisionReferral(
                referral_version="missing-record-vision-referral-1.0.0",
                paper_id=routing.paper_id,
                route_ids=[route.route_id],
                candidate_ids=[candidate.candidate_id],
                evidence_ids=candidate.evidence_ids,
                source_path=source.source_path,
                page_number=source.page_number,
                figure_or_table=candidate.figure_or_table or "visual object",
                reason=route.reason,
            )
        )
    return referrals
