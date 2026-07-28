"""Read one figure panel with a vision model, under the P2 abstain rule.

The four outcomes that survived every text-side fix -- GO-002, GO-003, GO-017,
GO-018 -- are qualitative marker colocalisations that live in immunofluorescence
panels. No amount of evidence selection, parsing or slot enforcement reaches
them, because the claim is only visible in the image.

uniparse already supplies what a vision call needs: per-panel crops with
bounding boxes and the caption attached to the object. This sends one crop plus
its text context to the same model the text path uses, through ``codex exec
--image``, and requires the reply to satisfy SelectiveVisionResponse.

The abstain rule is the point. ``exact_reported`` requires printed_labels
transcribed from the image, and a visually estimated value must route to human
review. A vision model that cannot ground a claim is expected to say so rather
than produce a plausible sentence.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.extraction.run_codex_one_call import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TIMEOUT_SECONDS,
    HARNESS,
    codex_available,
)

ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = ROOT / "data" / "raw" / "fulltext" / "uniparse"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "staging" / "extraction" / "codex_vision_v1"

PROMPT = (
    "You are reading one panel from a scientific figure. Report only what is "
    "visible in the image and supported by the caption text supplied below.\n\n"
    "Rules:\n"
    "- printed_labels must list, verbatim, the text you can actually read in "
    "the image: channel names, marker names, axis labels, printed numbers. If "
    "you cannot read any text, return an empty list.\n"
    "- Use value_status 'exact_reported' only for a number printed in the "
    "image. Use 'derived' only when you can state the derivation. Anything you "
    "judged by eye is 'visually_estimated' and must set disposition "
    "'human_review' and requires_human_review true.\n"
    "- If the panel does not support the claim under review, use disposition "
    "'missing' with value_status 'not_resolved'. Saying so is correct; "
    "inventing a plausible sentence is not.\n"
    "- visible_support must describe what in the image backs your answer.\n"
)


def _strict_vision_schema() -> dict[str, Any]:
    from openai.lib._pydantic import to_strict_json_schema

    from src.extraction.selective_vision_contracts import SelectiveVisionResponse

    return to_strict_json_schema(SelectiveVisionResponse)


def run_panel(
    *,
    paper_id: str,
    finding_id: str,
    image_path: Path,
    claim: str,
    caption: str,
    field_name: str = "qualitative_outcome",
    figure_or_table: str = "figure",
    evidence_ids: list[str] | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    run_dir = output_root / paper_id / finding_id
    if (run_dir / "response.json").exists():
        raise FileExistsError(
            f"{paper_id}/{finding_id} already has a response; use a new output root."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt = (
        f"{PROMPT}\n"
        f"CLAIM UNDER REVIEW: {claim}\n\n"
        f"CAPTION AND NEARBY TEXT:\n{caption}\n\n"
        f"finding_id to echo back: {finding_id}\n"
        f"field_name to echo back: {field_name}\n"
        f"figure_or_table to echo back: {figure_or_table}\n"
        f"supporting_evidence_ids to echo back: "
        f"{json.dumps(evidence_ids or [], ensure_ascii=False)}\n"
    )

    workdir = Path(tempfile.mkdtemp(prefix="codex_vision_"))
    try:
        schema_path = workdir / "schema.json"
        schema_path.write_text(
            json.dumps(_strict_vision_schema(), ensure_ascii=False), encoding="utf-8"
        )
        output_path = workdir / "last_message.txt"
        local_image = workdir / image_path.name
        shutil.copy(image_path, local_image)

        command = [
            "codex", "exec",
            "-m", model,
            "-c", f"model_reasoning_effort={reasoning_effort}",
            "--output-schema", str(schema_path),
            "--image", str(local_image),
            "--sandbox", "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "-o", str(output_path),
            "-",
        ]
        started = time.time()
        completed = subprocess.run(
            command, cwd=workdir, input=prompt, capture_output=True, text=True,
            timeout=timeout,
            env={**os.environ, "OPENAI_API_KEY": "", "SENSENOVA_API_KEY": ""},
        )
        elapsed = time.time() - started
        if completed.returncode != 0 or not output_path.exists():
            raise RuntimeError(
                f"codex exec failed (exit={completed.returncode}): "
                f"{completed.stderr[-600:]}"
            )
        raw = output_path.read_text(encoding="utf-8").strip()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    (run_dir / "response.json").write_text(
        json.dumps({"text": raw, "elapsed_seconds": round(elapsed, 3)},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    parsed = None
    error = None
    try:
        from src.extraction.selective_vision_contracts import SelectiveVisionResponse

        parsed = SelectiveVisionResponse.model_validate_json(raw)
        (run_dir / "observation.json").write_text(
            parsed.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001 - contract rejection is a real outcome
        error = f"{type(exc).__name__}: {str(exc)[:400]}"

    manifest = {
        "harness": HARNESS,
        "stage": "vision",
        "paper_id": paper_id,
        "finding_id": finding_id,
        "image": str(image_path.relative_to(ROOT)) if image_path.is_relative_to(ROOT) else str(image_path),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "contract_accepted": parsed is not None,
        "contract_error": error,
        "disposition": parsed.disposition if parsed else None,
        "value_status": parsed.value_status if parsed else None,
        "printed_labels": list(parsed.printed_labels) if parsed else None,
        "requires_human_review": parsed.requires_human_review if parsed else None,
        "openai_api_requests": 0,
        "codex_exec_turns": 1,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Read one figure panel with a vision model.")
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--finding-id", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--caption", default="")
    parser.add_argument("--field-name", default="qualitative_outcome")
    parser.add_argument("--figure-or-table", default="figure")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--confirm-codex-quota", action="store_true")
    args = parser.parse_args()

    if not args.confirm_codex_quota:
        parser.error("--confirm-codex-quota is required")
    if not codex_available():
        parser.error("`codex` was not found on PATH")

    print(json.dumps(run_panel(
        paper_id=args.paper_id, finding_id=args.finding_id, image_path=args.image,
        claim=args.claim, caption=args.caption, field_name=args.field_name,
        figure_or_table=args.figure_or_table, output_root=args.output_root,
        model=args.model, reasoning_effort=args.reasoning_effort, timeout=args.timeout,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
