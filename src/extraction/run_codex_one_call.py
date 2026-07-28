"""Run one compact extraction through the Codex CLI instead of the OpenAI SDK.

Why this exists
---------------
``run_compact_one_call`` needs ``OPENAI_API_KEY`` and bills per token. This
module drives the *same* model (``gpt-5.6-terra`` by default), at the *same*
reasoning effort, against the *same* strict response schema, through a
``codex exec`` subprocess that authenticates with the operator's existing Codex
login. That makes a gold-set recall measurement reachable without an API key.

What this is NOT
----------------
This is not a drop-in replacement for the audited paid path. A ``codex exec``
turn is an agent turn, so the provider wraps the request differently and the
per-request artifacts differ:

* no ``prompt_cache_key`` / ``store=False`` / ``service_tier`` control
* no ``response.usage`` object; only the token total Codex reports
* the request is shaped by the Codex harness, not by ``client.responses.create``

Absolute numbers from this harness therefore must NOT be compared against
numbers produced by ``run_compact_one_call``. Run a *control* through this same
harness (for example the compact view) and compare deltas within the harness,
so the harness effect cancels. Every manifest written here carries
``harness: "codex-exec"`` so a downstream reader cannot mistake it for the
native path.

Isolation
---------
Each call runs in its own scratch directory containing only the prompt and the
schema, with ``--sandbox read-only``. The model cannot reach
``data/annotations/gold_v1/`` (the answer key), which is what makes the result
usable as evidence rather than as a memory test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.extraction.assess_outcome_complexity import assess
from src.extraction.build_outcome_candidates import build_candidates
from src.extraction.check_outcome_coverage import check
from src.extraction.compact_prompt_v1 import (
    COMPACT_EXTRACTION_PROMPT,
    PROMPT_VERSION,
    prompt_sha256,
)
from src.extraction.compact_validation import validate_candidate
from src.rag.compact_api_packet import CompactApiPacket

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_TIMEOUT_SECONDS = 1800
HARNESS = "codex-exec"

PACKET_ROOT_BY_VIEW = {
    "compact": ROOT / "data" / "staging" / "rag" / "compact_api_packets_v1",
    "full": ROOT / "data" / "staging" / "rag" / "full_api_packets_v1",
}
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "staging" / "extraction" / "codex_one_call_v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def strict_schema() -> dict[str, Any]:
    """Return the same strict schema the paid path sends.

    Imported lazily: ``openai`` is only needed for the schema transform, not for
    any network call, and keeping it out of module import lets the rest of this
    module load in environments without the SDK.
    """
    from openai.lib._pydantic import to_strict_json_schema

    from src.extraction.compact_contracts import CompactExtractionResponse

    return to_strict_json_schema(CompactExtractionResponse)


def load_packet(paper_id: str, packet_root: Path) -> CompactApiPacket:
    packet_path = packet_root / f"{paper_id}.json"
    packet = CompactApiPacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    )
    unsigned = packet.model_dump(
        mode="json", exclude={"packet_checksum"}, exclude_none=True
    )
    recomputed = _sha256(_canonical_json(unsigned).encode("utf-8"))
    if recomputed != packet.packet_checksum:
        raise ValueError(f"{paper_id}: packet checksum mismatch")
    return packet


def build_prompt(packet: CompactApiPacket) -> str:
    payload = packet.model_dump(mode="json", exclude_none=True)
    return (
        f"{COMPACT_EXTRACTION_PROMPT}\n\n"
        "Return only the JSON object required by the output schema.\n\n"
        "INPUT PACKET:\n" + _canonical_json(payload)
    )


def codex_available() -> bool:
    return shutil.which("codex") is not None


def run_codex(
    prompt: str,
    schema: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    timeout: int,
) -> dict[str, Any]:
    """Execute one isolated ``codex exec`` turn and return its raw text output."""
    workdir = Path(tempfile.mkdtemp(prefix="codex_extract_"))
    try:
        schema_path = workdir / "schema.json"
        schema_path.write_text(
            json.dumps(schema, ensure_ascii=False), encoding="utf-8"
        )
        output_path = workdir / "last_message.txt"

        command = [
            "codex", "exec",
            "-m", model,
            "-c", f"model_reasoning_effort={reasoning_effort}",
            "--output-schema", str(schema_path),
            "--sandbox", "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "-o", str(output_path),
            "-",
        ]
        started = time.time()
        completed = subprocess.run(
            command,
            cwd=workdir,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            # Blank the keys so a misconfiguration cannot silently fall back to
            # the billed path.
            env={**os.environ, "OPENAI_API_KEY": "", "SENSENOVA_API_KEY": ""},
        )
        elapsed = time.time() - started

        if completed.returncode != 0 or not output_path.exists():
            raise RuntimeError(
                f"codex exec failed (exit={completed.returncode}): "
                f"{completed.stderr[-800:]}"
            )
        return {
            "text": output_path.read_text(encoding="utf-8").strip(),
            "elapsed_seconds": round(elapsed, 3),
            "stdout_tail": completed.stdout[-2000:],
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_one(
    paper_id: str,
    *,
    evidence_view: str = "compact",
    packet_root: Path | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if evidence_view not in PACKET_ROOT_BY_VIEW:
        raise ValueError(f"unknown evidence_view: {evidence_view}")
    resolved_root = packet_root or PACKET_ROOT_BY_VIEW[evidence_view]

    run_dir = output_root / paper_id
    result_path = run_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(
            f"A completed result already exists for {paper_id}; "
            "refusing to overwrite. Use a new --output-root."
        )

    packet = load_packet(paper_id, resolved_root)
    prompt = build_prompt(packet)
    schema = strict_schema()

    run_dir.mkdir(parents=True, exist_ok=True)

    complexity = assess(packet)
    (run_dir / "complexity.json").write_text(
        complexity.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    outcome_candidates = None
    if complexity.route == "complex":
        outcome_candidates = build_candidates(packet)
        (run_dir / "outcome_candidates.json").write_text(
            json.dumps(
                [row.model_dump(mode="json") for row in outcome_candidates],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    (run_dir / "request.json").write_text(
        json.dumps(
            {
                "harness": HARNESS,
                "paper_id": paper_id,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "evidence_view": evidence_view,
                "packet_root": str(resolved_root),
                "prompt_version": PROMPT_VERSION,
                "prompt_checksum": prompt_sha256(),
                "schema_checksum": _sha256(_canonical_json(schema).encode("utf-8")),
                "packet_checksum": packet.packet_checksum,
                "prompt_characters": len(prompt),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    started_at = datetime.now(timezone.utc)
    response = run_codex(
        prompt,
        schema,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )
    completed_at = datetime.now(timezone.utc)

    (run_dir / "response.json").write_text(
        json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    parsed, validation = validate_candidate(
        response["text"],
        paper_id=paper_id,
        allowed_evidence_ids={row.evidence_id for row in packet.evidence},
    )
    (run_dir / "validation_report.json").write_text(
        validation.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    if parsed is not None:
        (run_dir / "result.json").write_text(
            json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    coverage = None
    if parsed is not None and complexity.route == "complex":
        coverage = check(
            packet,
            parsed.model_dump(mode="json"),
            assessment=complexity,
            candidates=outcome_candidates,
        )
        (run_dir / "outcome_coverage.json").write_text(
            coverage.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        "harness": HARNESS,
        "harness_note": (
            "Driven through `codex exec`, not client.responses.create. "
            "Compare deltas within this harness only."
        ),
        "paper_id": paper_id,
        "evidence_view": evidence_view,
        "model_requested": model,
        "reasoning_effort": reasoning_effort,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": response["elapsed_seconds"],
        "validation_status": validation.status,
        "eligibility": (
            parsed.eligibility.model_dump(mode="json") if parsed else None
        ),
        "record_counts": {
            "formulations": len(parsed.formulations) if parsed else 0,
            "components": len(parsed.components) if parsed else 0,
            "experiments": len(parsed.experiments) if parsed else 0,
            "outcomes": len(parsed.outcomes) if parsed else 0,
            "unresolved_items": len(parsed.unresolved_items) if parsed else 0,
        },
        "openai_api_requests": 0,
        "codex_exec_turns": 1,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run compact extraction through the Codex CLI (no API key)."
    )
    parser.add_argument("--paper-id", action="append", dest="paper_ids")
    parser.add_argument(
        "--evidence-view", choices=sorted(PACKET_ROOT_BY_VIEW), default="compact"
    )
    parser.add_argument("--packet-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--confirm-codex-quota",
        action="store_true",
        help=(
            "Required guard acknowledging that each paper consumes one Codex "
            "turn from the operator's Codex plan."
        ),
    )
    args = parser.parse_args()

    if not args.confirm_codex_quota:
        parser.error("--confirm-codex-quota is required")
    if not codex_available():
        parser.error("`codex` was not found on PATH")

    paper_ids = args.paper_ids or [f"GP-{n:03d}" for n in range(1, 10)]
    manifests = []
    for paper_id in paper_ids:
        try:
            manifest = run_one(
                paper_id,
                evidence_view=args.evidence_view,
                packet_root=args.packet_root,
                output_root=args.output_root,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout=args.timeout,
            )
        except Exception as error:  # noqa: BLE001 - one bad paper must not abort the sweep
            manifest = {
                "paper_id": paper_id,
                "harness": HARNESS,
                "error": f"{type(error).__name__}: {error}",
            }
        manifests.append(manifest)
        print(json.dumps(manifest, ensure_ascii=False), flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "batch_manifest.json").write_text(
        json.dumps(
            {
                "harness": HARNESS,
                "evidence_view": args.evidence_view,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "papers": manifests,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
