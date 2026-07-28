"""Run missing-record repair through the Codex CLI instead of the OpenAI SDK.

Why this exists
---------------
``run_missing_record_repair`` is the audited paid implementation of the one
mechanism that has ever moved gold recall on this project: route the first-call
result, build evidence-bounded missing-record tasks, ask the model for *only*
the omitted experiments/outcomes, then merge the fragments back. On the native
API that chain took recall from 7/15 to 10/15.

The codex control group cannot use that module: it needs ``OPENAI_API_KEY`` and
bills per token. This module drives the *same* task contract, the *same* prompt,
and the *same* strict response schema through a ``codex exec`` subprocess, so
the repair half of the chain becomes reachable for the control group too.

What is shared and what is not
------------------------------
Shared with the paid path (imported, not copied):

* :func:`~src.extraction.run_missing_record_repair.load_task` -- checksum-verified
  task loading
* :func:`~src.extraction.run_missing_record_repair.validate_response` -- the
  evidence/ID/quota guard that decides whether a fragment is mergeable
* :func:`~src.extraction.run_missing_record_repair.fingerprint` -- the paid
  run identity, folded into this harness's own fingerprint
* ``PROMPT`` -- byte-identical instructions

Not shared, and therefore not comparable in absolute terms:

* the request is shaped by the Codex harness, not by ``client.responses.create``
* no ``store=False`` / ``service_tier`` / ``max_output_tokens`` control
* no ``response.usage``; only what Codex reports

Every manifest written here carries ``harness: "codex-exec"`` and
``openai_api_requests: 0`` so a downstream reader cannot mistake these fragments
for paid-path artifacts. Compare deltas *within* this harness (control first
call vs. control after merge), never across harnesses.

Isolation
---------
The actual subprocess call is delegated to
:func:`~src.extraction.run_codex_one_call.run_codex`: a scratch working
directory holding only the schema, ``--sandbox read-only``, ``--ephemeral``, and
blanked API keys. The model therefore cannot reach ``data/annotations/gold_v1/``
(the answer key) while answering, which is what makes a recovered record
evidence rather than recall of the gold file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.extraction.missing_record_contracts import (
    MissingRecordFragment,
    MissingRecordTask,
)
from src.extraction.run_codex_one_call import HARNESS, codex_available, run_codex
from src.extraction.run_missing_record_repair import (
    PROMPT,
    PROMPT_VERSION as PAID_PROMPT_VERSION,
    fingerprint as paid_fingerprint,
    load_task,
    validate_response,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "staging" / "extraction" / "codex_missing_record_v1"
DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_TIMEOUT_SECONDS = 900
PROMPT_VERSION = "codex-missing-record-prompt-1.0.0"
# The only text this harness adds to the paid prompt. `codex exec` is an agent
# turn, so it needs to be told that the turn's whole deliverable is the JSON
# object; the paid path gets that for free from the structured-output request.
SCHEMA_INSTRUCTION = "Return only the JSON object required by the output schema."


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def strict_schema() -> dict[str, Any]:
    """Return the strict fragment schema handed to ``codex exec``.

    ``openai`` is imported lazily: it is needed for the schema transform only,
    never for a network call.
    """
    from openai.lib._pydantic import to_strict_json_schema

    return to_strict_json_schema(MissingRecordFragment)


def build_prompt(task: MissingRecordTask) -> str:
    """Compose the paid prompt and the task payload into one codex turn.

    The paid path sends ``PROMPT`` as the system message and the canonical task
    JSON as the user message. A codex turn has one input channel, so the two are
    concatenated in that order with the payload clearly delimited.
    """
    return (
        f"{PROMPT}\n\n"
        f"{SCHEMA_INSTRUCTION}\n\n"
        "MISSING RECORD TASK:\n" + _canonical(task.model_dump(mode="json"))
    )


def fingerprint(
    task: MissingRecordTask,
    *,
    model: str,
    reasoning_effort: str,
) -> str:
    """Identity of one codex repair attempt.

    Built on top of the paid fingerprint so a codex run can never land in a
    directory a paid run would claim, and so a changed prompt wrapper, schema,
    or reasoning effort produces a new directory instead of a silent cache hit.
    """
    return _sha(
        _canonical(
            {
                "paid_fingerprint": paid_fingerprint(task, model),
                "paid_prompt_version": PAID_PROMPT_VERSION,
                "harness": HARNESS,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": _sha(build_prompt(task)),
                "schema_sha256": _sha(_canonical(strict_schema())),
                "reasoning_effort": reasoning_effort,
            }
        )
    )


def run_dir_for(
    task: MissingRecordTask,
    *,
    output_root: Path,
    model: str,
    reasoning_effort: str,
) -> Path:
    """Content-addressed output directory: ``<paper>/<task key>/<fingerprint>``."""
    return (
        output_root
        / task.paper_id
        / _sha(task.task_checksum)[:16]
        / fingerprint(task, model=model, reasoning_effort=reasoning_effort)
    )


def run_task(
    task_path: Path,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retry_failed: bool = False,
    codex_runner=run_codex,
) -> dict[str, Any]:
    """Execute one missing-record task through ``codex exec``.

    Writes ``request.json``, ``response.json``, ``manifest.json`` always, and
    ``fragment.json`` only when the response both parses against
    :class:`MissingRecordFragment` and survives
    :func:`validate_response`. An unvalidated fragment is never written, so the
    merge step cannot pick one up by accident.
    """
    task = load_task(task_path)
    run_dir = run_dir_for(
        task, output_root=output_root, model=model, reasoning_effort=reasoning_effort
    )
    fragment_path = run_dir / "fragment.json"
    manifest_path = run_dir / "manifest.json"

    if fragment_path.exists() and manifest_path.exists():
        # A completed, already-validated attempt. Re-check it against the task
        # rather than trusting the file, then refuse to spend a second turn.
        cached = MissingRecordFragment.model_validate_json(
            fragment_path.read_text(encoding="utf-8")
        )
        validate_response(cached, task)
        return {
            **json.loads(manifest_path.read_text(encoding="utf-8")),
            "cache_hit": True,
            "codex_exec_turns_this_run": 0,
        }
    if fragment_path.exists():
        raise FileExistsError(
            f"{run_dir} holds a fragment without a manifest; refusing to overwrite."
        )
    if run_dir.exists():
        if not retry_failed:
            raise FileExistsError(
                f"An incomplete or failed attempt already exists at {run_dir}; "
                "refusing an automatic Codex retry. Pass --retry-failed to "
                "deliberately spend another turn, or use a new --output-root."
            )
        # Only ever discards an attempt that produced no validated fragment.
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    prompt = build_prompt(task)
    schema = strict_schema()
    run_fingerprint = fingerprint(
        task, model=model, reasoning_effort=reasoning_effort
    )
    request_snapshot = {
        "harness": HARNESS,
        "paper_id": task.paper_id,
        "task_path": str(task_path),
        "task_checksum": task.task_checksum,
        "candidate_ids": task.candidate_ids,
        "route_ids": task.route_ids,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "prompt_version": PROMPT_VERSION,
        "paid_prompt_version": PAID_PROMPT_VERSION,
        "prompt_sha256": _sha(prompt),
        "prompt_characters": len(prompt),
        "schema_sha256": _sha(_canonical(schema)),
        "fingerprint": run_fingerprint,
        "permitted_new_experiments": task.permitted_new_experiments,
        "permitted_new_outcomes": task.permitted_new_outcomes,
        "evidence_ids": [row.evidence_id for row in task.evidence],
    }
    (run_dir / "request.json").write_text(
        json.dumps(request_snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    started = datetime.now(timezone.utc)
    response = codex_runner(
        prompt,
        schema,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )
    completed = datetime.now(timezone.utc)
    (run_dir / "response.json").write_text(
        json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fragment: MissingRecordFragment | None = None
    status = "completed_pending_merge"
    error: str | None = None
    try:
        fragment = MissingRecordFragment.model_validate_json(response["text"])
        validate_response(fragment, task)
    except Exception as failure:  # noqa: BLE001 - recorded, never merged
        fragment = None
        status = "rejected"
        error = f"{type(failure).__name__}: {failure}"
    if fragment is not None:
        fragment_path.write_text(
            fragment.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        "status": status,
        "harness": HARNESS,
        "harness_note": (
            "Driven through `codex exec`, not client.responses.create. "
            "Compare deltas within this harness only."
        ),
        "paper_id": task.paper_id,
        "task_path": str(task_path),
        "task_checksum": task.task_checksum,
        "candidate_ids": task.candidate_ids,
        "fingerprint": run_fingerprint,
        "model_requested": model,
        "reasoning_effort": reasoning_effort,
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "elapsed_seconds": response.get("elapsed_seconds"),
        "response_validated": fragment is not None,
        "validation_error": error,
        "disposition": fragment.disposition if fragment else None,
        "recovered_candidate_ids": (
            list(fragment.recovered_candidate_ids) if fragment else []
        ),
        "unresolved_candidate_ids": (
            list(fragment.unresolved_candidate_ids) if fragment else []
        ),
        "record_counts": {
            "experiments": len(fragment.experiments) if fragment else 0,
            "outcomes": len(fragment.outcomes) if fragment else 0,
        },
        "fragment_path": str(fragment_path) if fragment else None,
        "openai_api_requests": 0,
        "codex_exec_turns": 1,
        "cache_hit": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def discover_tasks(task_root: Path) -> list[Path]:
    """Every ``task_*.json`` under ``task_root``, in stable paper/task order."""
    return sorted(task_root.rglob("task_*.json"))


def run_tasks(
    task_paths: list[Path],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retry_failed: bool = False,
    codex_runner=run_codex,
    on_manifest=None,
) -> dict[str, Any]:
    """Run a batch; one failing task must not abort the sweep."""
    manifests: list[dict[str, Any]] = []
    for task_path in task_paths:
        try:
            manifest = run_task(
                task_path,
                output_root=output_root,
                model=model,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                retry_failed=retry_failed,
                codex_runner=codex_runner,
            )
        except Exception as failure:  # noqa: BLE001
            manifest = {
                "status": "error",
                "harness": HARNESS,
                "task_path": str(task_path),
                "error": f"{type(failure).__name__}: {failure}",
                "openai_api_requests": 0,
            }
        manifests.append(manifest)
        if on_manifest is not None:
            on_manifest(manifest)
    batch = {
        "harness": HARNESS,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "prompt_version": PROMPT_VERSION,
        "tasks": len(manifests),
        "validated": sum(1 for row in manifests if row.get("response_validated")),
        "recovered": sum(
            1 for row in manifests if row.get("disposition") == "recovered"
        ),
        "unresolved": sum(
            1 for row in manifests if row.get("disposition") == "unresolved"
        ),
        "failed": sum(1 for row in manifests if not row.get("response_validated")),
        "openai_api_requests": 0,
        "codex_exec_turns": sum(row.get("codex_exec_turns", 0) for row in manifests),
        "results": manifests,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "batch_manifest.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run evidence-bounded missing-record repair through the Codex CLI "
            "(no OpenAI API key, no paid request)."
        )
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        type=Path,
        help="Path to one missing-record task JSON. Repeatable.",
    )
    parser.add_argument(
        "--task-root",
        type=Path,
        default=None,
        help="Directory searched recursively for task_*.json.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Re-run tasks whose previous attempt produced no validated fragment. "
            "Never discards a validated fragment."
        ),
    )
    parser.add_argument(
        "--confirm-codex-quota",
        action="store_true",
        help=(
            "Required guard acknowledging that each task consumes one Codex turn "
            "from the operator's Codex plan."
        ),
    )
    args = parser.parse_args()

    if not args.confirm_codex_quota:
        parser.error("--confirm-codex-quota is required")
    if not args.tasks and not args.task_root:
        parser.error("provide --task and/or --task-root")
    if not codex_available():
        parser.error("`codex` was not found on PATH")

    task_paths = list(args.tasks or [])
    if args.task_root:
        task_paths.extend(discover_tasks(args.task_root))
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in task_paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(path)
    if not ordered:
        parser.error("no missing-record tasks found")

    batch = run_tasks(
        ordered,
        output_root=args.output_root,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout=args.timeout,
        retry_failed=args.retry_failed,
        on_manifest=lambda row: print(
            json.dumps(row, ensure_ascii=False), flush=True
        ),
    )
    print(
        json.dumps(
            {key: value for key, value in batch.items() if key != "results"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
