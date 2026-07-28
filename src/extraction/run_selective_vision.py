"""Run one independently cached OpenAI request against one PDF crop."""

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
from pydantic import TypeAdapter

from src.extraction.build_repair_tasks import COLLECTION_MODELS
from src.extraction.selective_vision_contracts import (
    SelectiveVisionResponse,
    SelectiveVisionTask,
    p2_prompt_suffix,
    p2_response_properties,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (
    ROOT / "data" / "staging" / "extraction" / "selective_vision_v1"
)
PROMPT_VERSION = "selective-vision-prompt-1.0.0"
VISION_PROMPT = """Inspect exactly one targeted scientific figure or table crop.
Use only the crop, caption, referring results passage, and methods context.
Never treat an axis position as an exact reported value. Distinguish exact
printed values, explicitly derived values, and visually estimated values.
Return the panel or table-cell location. Any visually estimated value must be
routed to human review. Do not alter any field other than the requested field."""


def vision_prompt() -> str:
    """System prompt for the current flag state.

    With ``local_vlm_vision`` off this returns :data:`VISION_PROMPT` unchanged,
    so the prompt hash inside :func:`vision_fingerprint` -- and therefore every
    cached run -- is untouched.
    """
    return VISION_PROMPT + p2_prompt_suffix()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def load_task(path: Path) -> SelectiveVisionTask:
    task = SelectiveVisionTask.model_validate_json(path.read_text(encoding="utf-8"))
    actual = _sha256(_canonical_json(task.unsigned_payload()))
    if actual != task.task_checksum:
        raise ValueError("Selective-vision task checksum mismatch")
    crop_path = Path(task.crop_path)
    if _sha256(crop_path.read_bytes()) != task.crop_sha256:
        raise ValueError("Selective-vision crop checksum mismatch")
    return task


def response_schema(task: SelectiveVisionTask) -> dict[str, Any]:
    field_name = task.finding.field_name
    assert field_name is not None
    fragment = task.expected_schema_fragment
    # Strict mode requires every declared property to be required, so the P2
    # block is spliced into both maps or neither. With the flag off both splices
    # are empty and the schema is byte-identical to the pre-P2 one.
    extra_properties = p2_response_properties()
    schema = {
        "type": "object",
        "properties": {
            "finding_id": {"type": "string"},
            "disposition": {
                "type": "string",
                "enum": ["resolved", "missing", "ambiguous", "human_review"],
            },
            "field_name": {"type": "string"},
            "corrected_fragment": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {field_name: fragment["schema"]},
                        "required": [field_name],
                        "additionalProperties": False,
                    },
                    {"type": "null"},
                ]
            },
            "value_status": {
                "type": "string",
                "enum": [
                    "exact_reported",
                    "derived",
                    "visually_estimated",
                    "not_resolved",
                ],
            },
            "supporting_evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "figure_or_table": {"type": "string"},
            "panel_or_table_cell": {"type": ["string", "null"]},
            "visible_support": {"type": "string"},
            "derivation": {"type": ["string", "null"]},
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "requires_human_review": {"type": "boolean"},
            **extra_properties,
        },
        "required": [
            "finding_id",
            "disposition",
            "field_name",
            "corrected_fragment",
            "value_status",
            "supporting_evidence_ids",
            "figure_or_table",
            "panel_or_table_cell",
            "visible_support",
            "derivation",
            "confidence",
            "requires_human_review",
            *extra_properties,
        ],
        "additionalProperties": False,
        "$defs": fragment.get("$defs", {}),
    }
    return schema


def _validate_panel_provenance(
    response: SelectiveVisionResponse, task: SelectiveVisionTask
) -> None:
    """Reject an observation that claims to have looked at a different object.

    Only meaningful once a task carries a uniparse panel, which only happens
    with ``local_vlm_vision`` on; a pre-P2 task has no panel and this is a
    no-op, so the pre-P2 acceptance set is unchanged.
    """
    panel = task.panel
    if panel is None:
        return
    if response.object_id and response.object_id != panel.object_id:
        raise ValueError("Vision response reports a different panel object_id")
    if response.page_id and response.page_id != panel.page_id:
        raise ValueError("Vision response reports a different panel page")
    if (
        response.image_sha256
        and panel.image_sha256
        and response.image_sha256 != panel.image_sha256
    ):
        raise ValueError("Vision response reports a different panel image digest")


def validate_response(
    response: SelectiveVisionResponse, task: SelectiveVisionTask
) -> None:
    finding = task.finding
    if response.finding_id != finding.finding_id:
        raise ValueError("Vision response finding_id does not match its task")
    if response.field_name != finding.field_name:
        raise ValueError("Vision response changed the requested field name")
    if response.figure_or_table != task.figure_or_table:
        raise ValueError("Vision response location does not match its task")
    allowed_evidence = {
        task.crop_evidence_id,
        task.caption.evidence_id,
        *(row.evidence_id for row in task.referring_results_passages),
        *(row.evidence_id for row in task.methods_context),
    }
    unknown = set(response.supporting_evidence_ids) - allowed_evidence
    if unknown:
        raise ValueError(f"Vision response cites unknown evidence: {sorted(unknown)}")
    _validate_panel_provenance(response, task)
    if response.corrected_fragment is None:
        return
    field_name = finding.field_name
    collection = finding.record_collection
    assert field_name is not None and collection is not None
    if set(response.corrected_fragment) != {field_name}:
        raise ValueError("Vision response may correct exactly one field")
    annotation = COLLECTION_MODELS[collection].model_fields[field_name].annotation
    TypeAdapter(annotation).validate_python(response.corrected_fragment[field_name])
    corrected_value = response.corrected_fragment[field_name]
    corrected_evidence_ids = (
        set(corrected_value.get("evidence_ids", []))
        if isinstance(corrected_value, dict)
        else set()
    )
    unknown_corrected = corrected_evidence_ids - allowed_evidence
    if unknown_corrected:
        raise ValueError(
            "Vision correction cites unknown evidence: "
            f"{sorted(unknown_corrected)}"
        )


def _image_data(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode(
        "ascii"
    )


def vision_fingerprint(task: SelectiveVisionTask, model: str) -> str:
    return _sha256(
        _canonical_json(
            {
                "task_checksum": task.task_checksum,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": _sha256(vision_prompt()),
                "response_schema": response_schema(task),
                "model": model,
            }
        )
    )


def run_vision(
    task: SelectiveVisionTask,
    *,
    model: str,
    client: OpenAI,
    output_root: Path = OUTPUT_ROOT,
    max_output_tokens: int = 2_000,
) -> dict[str, Any]:
    fingerprint = vision_fingerprint(task, model)
    run_dir = output_root / task.paper_id / task.finding.finding_id / fingerprint
    result_path = run_dir / "result.json"
    manifest_path = run_dir / "manifest.json"
    if result_path.exists() and manifest_path.exists():
        result = SelectiveVisionResponse.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        validate_response(result, task)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {**manifest, "cache_hit": True, "paid_api_requests_this_run": 0}
    if run_dir.exists():
        raise FileExistsError("Incomplete vision directory exists; refusing paid retry")
    run_dir.mkdir(parents=True)

    text_payload = task.text_payload()
    system_prompt = vision_prompt()
    request_snapshot = {
        "model": model,
        "reasoning_effort": "low",
        "store": False,
        "max_output_tokens": max_output_tokens,
        "vision_fingerprint": fingerprint,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": system_prompt,
        "text_payload": text_payload,
        "image": {
            "crop_path": task.crop_path,
            "crop_sha256": task.crop_sha256,
            "detail": "original",
        },
    }
    (run_dir / "request.json").write_text(
        json.dumps(request_snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    started_at = datetime.now(timezone.utc)
    api_response = client.responses.create(
        model=model,
        reasoning={"effort": "low"},
        store=False,
        service_tier="default",
        max_output_tokens=max_output_tokens,
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _canonical_json(text_payload),
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_data(Path(task.crop_path)),
                        "detail": "original",
                    },
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "SelectiveVisionResponse",
                "schema": response_schema(task),
                "strict": True,
            }
        },
    )
    completed_at = datetime.now(timezone.utc)
    (run_dir / "response.json").write_text(
        json.dumps(api_response.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    if not api_response.output_text:
        raise RuntimeError("Selective vision returned no structured output")
    result = SelectiveVisionResponse.model_validate_json(api_response.output_text)
    validate_response(result, task)
    result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    status = (
        "human_review_required"
        if result.requires_human_review
        else "completed_pending_merge"
    )
    manifest = {
        "status": status,
        "paper_id": task.paper_id,
        "finding_id": task.finding.finding_id,
        "field_name": task.finding.field_name,
        "vision_fingerprint": fingerprint,
        "model_requested": model,
        "model_returned": api_response.model,
        "response_id": api_response.id,
        "paid_api_requests": 1,
        "cache_hit": False,
        "source_images_sent": 1,
        "full_pdf_sent": False,
        "page_number": task.page_number,
        "figure_or_table": task.figure_or_table,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "usage": (
            api_response.usage.model_dump(mode="json")
            if api_response.usage
            else None
        ),
        "value_status": result.value_status,
        "requires_human_review": result.requires_human_review,
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
    model = os.getenv("SELECTIVE_VISION_MODEL", "gpt-5.6-terra")
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        timeout=300.0,
        max_retries=0,
    )
    print(
        json.dumps(
            run_vision(load_task(args.task), model=model, client=client),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
