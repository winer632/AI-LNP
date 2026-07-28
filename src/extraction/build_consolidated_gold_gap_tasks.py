"""Build three benchmark gap tasks without exposing frozen gold answers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.extraction.build_full_outcome_inventory import (
    CORPUS_ROOT,
    full_corpus_view,
)
from src.extraction.build_selective_vision_tasks import render_pdf_region
from src.extraction.consolidated_recovery_contracts import (
    ConsolidatedRecoveryTask,
    RecoveryVisualAsset,
)
from src.extraction.outcome_inventory_contracts import OutcomeInventory
from src.extraction.repair_contracts import RepairEvidence
from src.rag.compact_packet import CompactEvidencePacket


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "data/staging/extraction/consolidated_gold_gap_tasks_v1"

# Selection is benchmark-specific, but contains no expected gold values.
SELECTION = {
    "GP-004": {
        "candidate_ids": [
            "OC-50a8a10d7440e225",
            "OC-88f65d75fdf21989",
        ],
        "purpose": (
            "Recover distinct cell-population eGFP outcomes represented in the "
            "selected Results evidence; do not combine biologically different "
            "cell populations into one outcome."
        ),
        "assets": [
            {
                "label": "Figure 2",
                "path": (
                    "data/raw/fulltext/oa_packages/PMC7840919/"
                    "41467_2021_20903_Fig2_HTML.jpg"
                ),
            }
        ],
    },
    "GP-006": {
        "candidate_ids": [
            "OC-be694177da34d91d",
            "OC-e7802a64c3c71f07",
        ],
        "purpose": (
            "Recover the LSEC GFP/marker outcome and the distinct insertion "
            "frequency outcome. Treat them as separate records."
        ),
        "assets": [
            {
                "label": "Figure 2",
                "path": "data/raw/fulltext/oa_packages/PMC11617921/gr2.jpg",
            },
            {
                "label": "Table S2",
                "path": "data/raw/fulltext/oa_packages/PMC11617921/mmc1.pdf",
                "page_number": 2,
            },
        ],
    },
    "GP-008": {
        "candidate_ids": [
            "OC-e2e938ef42d4499b",
            "OC-acf6d2762cb95d99",
            "OC-9b05bda97af13efb",
        ],
        "purpose": (
            "Recover the activated-HSC elimination outcome and the distinct "
            "reporter/FAPCAR recipient-cell localization outcome."
        ),
        "assets": [
            {
                "label": "Figure 2",
                "path": (
                    "data/raw/fulltext/oa_packages/PMC13229182/"
                    "pnas.2534673123fig02.jpg"
                ),
            },
            {
                "label": "Figure 6",
                "path": (
                    "data/raw/fulltext/oa_packages/PMC13229182/"
                    "pnas.2534673123fig06.jpg"
                ),
            },
            {
                "label": "Supplementary Figure 5, page 19",
                "path": (
                    "data/raw/fulltext/oa_packages/PMC13229182/"
                    "pnas.2534673123.sapp.pdf"
                ),
                "page_number": 19,
            },
        ],
    },
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def _portable(path: Path) -> str:
    """Record a path as repository-relative so signed tasks survive a fresh clone.

    Absolute developer paths baked into a committed task file make the task
    unloadable on every other machine, and the path is covered by the task
    checksum so it cannot be patched afterwards without re-signing. Paths
    outside the repository (an explicit temporary output root, for instance)
    are recorded verbatim.
    """
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _result_path(paper_id: str) -> Path:
    choices = [
        ROOT / f"data/staging/extraction/compact_merged_v1_1/{paper_id}/final_result.json",
        ROOT / f"data/staging/extraction/compact_merged_v1/{paper_id}/final_result.json",
        ROOT / f"data/staging/extraction/compact_one_call_v1/{paper_id}/result.json",
    ]
    return next(path for path in choices if path.exists())


def _next_ids(existing: list[str], prefix: str, count: int) -> list[str]:
    used = {
        int(value[len(prefix):])
        for value in existing
        if value.startswith(prefix) and value[len(prefix):].isdigit()
    }
    values = []
    number = 1
    while len(values) < count:
        if number not in used:
            values.append(f"{prefix}{number}")
        number += 1
    return values


def build_one(paper_id: str, *, output_root: Path = OUTPUT_ROOT) -> ConsolidatedRecoveryTask:
    config = SELECTION[paper_id]
    packet_path = ROOT / f"data/staging/rag/compact_packets_v1/{paper_id}.json"
    packet = CompactEvidencePacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    )
    full_view = full_corpus_view(packet, CORPUS_ROOT / f"{paper_id}.blocks.jsonl")
    evidence_by_id = {row.evidence_id: row for row in full_view.evidence}
    inventory_path = (
        ROOT / f"reports/extraction/enforced_compact_workflow_v1/{paper_id}/inventory.json"
    )
    inventory = OutcomeInventory.model_validate_json(
        inventory_path.read_text(encoding="utf-8")
    )
    candidates = {row.candidate_id: row for row in inventory.retained_candidates}
    missing = set(config["candidate_ids"]) - set(candidates)
    if missing:
        raise ValueError(f"Selected candidates absent from inventory: {sorted(missing)}")
    evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for candidate_id in config["candidate_ids"]
            for evidence_id in candidates[candidate_id].evidence_ids
        )
    )
    task_dir = output_root / paper_id
    task_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    for index, item in enumerate(config["assets"], 1):
        source_path = ROOT / item["path"]
        page_number = item.get("page_number")
        if source_path.suffix.lower() == ".pdf":
            image_path = task_dir / f"asset_{index:02d}.png"
            render_pdf_region(source_path, page_number, None, image_path)
        else:
            image_path = source_path
        image_sha = _sha(image_path.read_bytes())
        assets.append(
            RecoveryVisualAsset(
                label=item["label"],
                image_path=_portable(image_path),
                image_sha256=image_sha,
                crop_evidence_id=f"V-{image_sha[:16]}",
                source_path=_portable(source_path),
                page_number=page_number,
            )
        )
    result_path = _result_path(paper_id)
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    existing_outcome_ids = [
        row["outcome_id"] for row in result.get("outcomes", [])
    ]
    unsigned = {
        "task_version": "consolidated-recovery-task-1.0.0",
        "paper_id": paper_id,
        "purpose": config["purpose"],
        "candidate_ids": config["candidate_ids"],
        "evidence": [
            RepairEvidence(
                evidence_id=evidence_by_id[evidence_id].evidence_id,
                text=evidence_by_id[evidence_id].text,
                source_ids=evidence_by_id[evidence_id].source_ids,
            ).model_dump(mode="json")
            for evidence_id in evidence_ids
        ],
        "visual_assets": [row.model_dump(mode="json") for row in assets],
        "existing_formulation_ids": [
            row["formulation_id"] for row in result.get("formulations", [])
        ],
        "existing_experiment_ids": [
            row["experiment_id"] for row in result.get("experiments", [])
        ],
        "existing_outcome_ids": existing_outcome_ids,
        "existing_experiments": result.get("experiments", []),
        "existing_outcomes": result.get("outcomes", []),
        "permitted_new_outcome_ids": _next_ids(existing_outcome_ids, "O", 2),
        "permitted_new_experiments": 0,
        "permitted_new_outcomes": 2,
        "source_result_sha256": _sha(result_bytes),
        "source_inventory_sha256": _sha(inventory.model_dump_json(exclude_none=True)),
    }
    task = ConsolidatedRecoveryTask.model_validate(
        {**unsigned, "task_checksum": _sha(_canonical(unsigned))}
    )
    (task_dir / "task.json").write_text(
        task.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return task


def build_all(*, output_root: Path = OUTPUT_ROOT) -> list[ConsolidatedRecoveryTask]:
    tasks = [build_one(paper_id, output_root=output_root) for paper_id in SELECTION]
    summary = {
        "task_set_version": "consolidated-gold-gap-tasks-1.0.0",
        "benchmark_specific_selection": True,
        "gold_answer_values_in_tasks": False,
        "planned_paid_calls": len(tasks),
        "tasks": [
            {
                "paper_id": row.paper_id,
                "candidate_ids": row.candidate_ids,
                "evidence_passages": len(row.evidence),
                "visual_assets": len(row.visual_assets),
                "permitted_new_outcomes": row.permitted_new_outcomes,
                "permitted_new_outcome_ids": row.permitted_new_outcome_ids,
                "task_path": _portable(output_root / row.paper_id / "task.json"),
            }
            for row in tasks
        ],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return tasks


if __name__ == "__main__":
    print(
        json.dumps(
            [row.model_dump(mode="json") for row in build_all()],
            ensure_ascii=False,
            indent=2,
        )
    )
