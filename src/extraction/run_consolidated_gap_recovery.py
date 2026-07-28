"""Run one cached multimodal missing-record request for one paper."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema

from src.extraction.consolidated_recovery_contracts import ConsolidatedRecoveryTask
from src.extraction.missing_record_contracts import (
    MissingRecordFragment,
    MissingRecordTask,
)
from src.extraction.repair_contracts import RepairEvidence
from src.extraction.run_missing_record_repair import validate_response


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "data/staging/extraction/consolidated_gold_gap_results_v1"
PROMPT_VERSION = "consolidated-gap-recovery-prompt-1.0.0"
PROMPT = """Recover omitted biomedical outcome records for exactly one paper.
Use only the supplied evidence passages and images. Candidate IDs identify
possibly duplicated evidence groups, not expected answers. Account for every
candidate ID as recovered or unresolved. Multiple duplicate candidates may
support the same outcome. Keep biologically distinct endpoints or cell
populations as distinct outcome records. Do not repeat existing outcomes.
Use the supplied existing experiment records to select a valid experiment_id,
and use the existing outcome records to avoid duplicate recovery.
Exact printed and explicitly derived visual values may be recovered; visually
estimated axis values must remain unresolved. Never invent values, evidence
labels, comparisons, units, or record links. New outcome IDs must come from
permitted_new_outcome_ids."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def _image(path: Path) -> str:
    mime = "jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "png"
    return "data:image/" + mime + ";base64," + (
        base64.b64encode(path.read_bytes()).decode()
    )


def asset_path(value: str) -> Path:
    """Resolve a recorded visual-asset path against the repository.

    Signed task files record repository-relative paths so that a task stays
    valid in any checkout instead of only on the machine that generated it. The
    recorded string is part of the task checksum, so it must not be rewritten
    here; only the lookup is resolved. Absolute paths are honoured unchanged for
    backward compatibility with older task files.
    """
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def load_task(path: Path) -> ConsolidatedRecoveryTask:
    task = ConsolidatedRecoveryTask.model_validate_json(path.read_text(encoding="utf-8"))
    unsigned = task.model_dump(mode="json", exclude={"task_checksum"})
    if _sha(_canonical(unsigned)) != task.task_checksum:
        raise ValueError("Consolidated recovery task checksum mismatch")
    for asset in task.visual_assets:
        if _sha(asset_path(asset.image_path).read_bytes()) != asset.image_sha256:
            raise ValueError(f"Visual asset checksum mismatch: {asset.label}")
    return task


def _validation_task(task: ConsolidatedRecoveryTask) -> MissingRecordTask:
    evidence = list(task.evidence)
    evidence.extend(
        RepairEvidence(
            evidence_id=asset.crop_evidence_id,
            text=f"Visual asset: {asset.label}",
            source_ids=[],
        )
        for asset in task.visual_assets
    )
    return MissingRecordTask(
        task_version="missing-record-task-1.1.0",
        paper_id=task.paper_id,
        route_ids=[f"consolidated:{candidate_id}" for candidate_id in task.candidate_ids],
        candidate_ids=task.candidate_ids,
        evidence=evidence,
        existing_formulation_ids=task.existing_formulation_ids,
        existing_experiment_ids=task.existing_experiment_ids,
        existing_outcome_ids=task.existing_outcome_ids,
        permitted_new_experiments=task.permitted_new_experiments,
        permitted_new_outcomes=task.permitted_new_outcomes,
        source_result_sha256=task.source_result_sha256,
        source_inventory_sha256=task.source_inventory_sha256,
        task_checksum="consolidated-adapter",
    )


def validate(result: MissingRecordFragment, task: ConsolidatedRecoveryTask) -> None:
    validate_response(result, _validation_task(task))
    returned_ids = {row.outcome_id for row in result.outcomes}
    unknown_ids = returned_ids - set(task.permitted_new_outcome_ids)
    if unknown_ids:
        raise ValueError(
            f"Response used unpermitted new outcome IDs: {sorted(unknown_ids)}"
        )


def fingerprint(task: ConsolidatedRecoveryTask, model: str) -> str:
    return _sha(
        _canonical(
            {
                "task_checksum": task.task_checksum,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": _sha(PROMPT),
                "schema": to_strict_json_schema(MissingRecordFragment),
                "model": model,
            }
        )
    )


def run(
    task: ConsolidatedRecoveryTask,
    *,
    model: str,
    client: OpenAI,
    output_root: Path = OUTPUT_ROOT,
    max_output_tokens: int = 5_000,
) -> dict:
    run_fingerprint = fingerprint(task, model)
    run_dir = output_root / task.paper_id / run_fingerprint
    result_path = run_dir / "result.json"
    manifest_path = run_dir / "manifest.json"
    if result_path.exists() and manifest_path.exists():
        result = MissingRecordFragment.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        validate(result, task)
        return {
            **json.loads(manifest_path.read_text(encoding="utf-8")),
            "cache_hit": True,
            "paid_api_requests_this_run": 0,
        }
    if run_dir.exists():
        raise FileExistsError("Incomplete run exists; refusing an automatic paid retry")
    run_dir.mkdir(parents=True)
    text_payload = task.model_dump(
        mode="json",
        exclude={"visual_assets": {"__all__": {"image_path", "image_sha256"}}},
    )
    (run_dir / "request.json").write_text(
        json.dumps(
            {
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "prompt": PROMPT,
                "task": text_payload,
                "visual_assets": [
                    {
                        "label": asset.label,
                        "image_path": asset.image_path,
                        "image_sha256": asset.image_sha256,
                    }
                    for asset in task.visual_assets
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    content = [{"type": "input_text", "text": _canonical(text_payload)}]
    content.extend(
        {
            "type": "input_image",
            "image_url": _image(asset_path(asset.image_path)),
            "detail": "original",
        }
        for asset in task.visual_assets
    )
    started = datetime.now(timezone.utc)
    api_response = client.responses.create(
        model=model,
        reasoning={"effort": "low"},
        store=False,
        service_tier="default",
        max_output_tokens=max_output_tokens,
        input=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": content},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "ConsolidatedMissingRecordFragment",
                "schema": to_strict_json_schema(MissingRecordFragment),
                "strict": True,
            }
        },
    )
    if not api_response.output_text:
        raise RuntimeError("Consolidated recovery returned no structured output")
    result = MissingRecordFragment.model_validate_json(api_response.output_text)
    validate(result, task)
    (run_dir / "response.json").write_text(
        json.dumps(api_response.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "completed_pending_merge",
        "paper_id": task.paper_id,
        "candidate_ids": task.candidate_ids,
        "fingerprint": run_fingerprint,
        "response_id": api_response.id,
        "model_requested": model,
        "model_returned": api_response.model,
        "paid_api_requests": 1,
        "cache_hit": False,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "usage": (
            api_response.usage.model_dump(mode="json") if api_response.usage else None
        ),
        "disposition": result.disposition,
        "recovered_candidate_ids": result.recovered_candidate_ids,
        "unresolved_candidate_ids": result.unresolved_candidate_ids,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--confirm-paid-call", action="store_true")
    args = parser.parse_args()
    if not args.confirm_paid_call:
        parser.error("--confirm-paid-call is required")
    load_dotenv(ROOT / ".env")
    model = os.getenv("CONSOLIDATED_RECOVERY_MODEL", "gpt-5.6-terra")
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        timeout=300,
        max_retries=0,
    )
    print(json.dumps(run(load_task(args.task), model=model, client=client), indent=2))


if __name__ == "__main__":
    main()
