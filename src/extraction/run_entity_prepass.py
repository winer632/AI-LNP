"""Run the entity-resolution pre-pass over one paper, through the codex harness.

One ``codex exec`` turn per paper. The turn sees only the methods/reagent-table
slice of the paper's own evidence packet -- selected deterministically by
:func:`src.extraction.entity_resolution.select_identity_evidence` -- and is asked
what each named cell line is, with citations.

Whatever it returns is then put through
:func:`src.extraction.entity_resolution.ground_entity_table` before anything is
written as an entity table. The raw reply is kept alongside, so a reader can
always see what was proposed and what was refused.

This writes an artifact. It changes no extraction by itself: the table travels
into a run only when ``entity_resolution_prepass`` is on and the run is pointed
at the table's root.
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

from src.extraction.entity_resolution import (
    ENTITY_TABLE_VERSION,
    EntityResolutionResponse,
    GroundedEntityTable,
    evidence_payload,
    ground_entity_table,
    select_identity_evidence,
)
from src.extraction.run_codex_one_call import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TIMEOUT_SECONDS,
    HARNESS,
    PACKET_ROOT_BY_VIEW,
    _canonical_json,
    _sha256,
    codex_available,
    load_packet,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "staging" / "extraction" / "entity_tables_v1"
DEFAULT_EVIDENCE_VIEW = "full"
PREPASS_VERSION = "entity-prepass-1.0.0"

# Written as a rule about *evidence*, not about biology. It names no cell line,
# no cell type, no supplier and no species; the two-place join it describes is
# the shape reagent tables and methods sections have in general, and the example
# is given in the abstract ("a table row and a sentence that share a supplier or
# a species") precisely so that it cannot be read as a hint about one paper.
PREPASS_PROMPT = (
    "You are reading the methods, cell-culture text and reagent/resource tables "
    "of one scientific paper. Your only task is to say what each cell line the "
    "text names actually IS.\n\n"
    "For every cell line named in the evidence below, return one entry:\n"
    "- line_name: the line's name exactly as the evidence writes it.\n"
    "- population: the cell type or population that line represents, in the "
    "evidence's own words. null when the evidence does not say.\n"
    "- state: the state that line was in when it was used or measured -- how it "
    "was treated, induced, differentiated, selected or enriched -- in the "
    "evidence's own words. null when the evidence does not say.\n"
    "- species: the organism the line comes from, in the evidence's own words. "
    "null when the evidence does not say.\n"
    "- evidence_ids: every evidence_id you used, and only ids from the list "
    "below.\n"
    "- derivation: one sentence saying how those ids give that answer.\n\n"
    "Rules:\n"
    "- Use no outside knowledge. If you recognise a line but this evidence does "
    "not say what it is, return null. That is the correct answer here, not a "
    "failure.\n"
    "- Every word you write in population, state and species must appear in the "
    "evidence you cite for that line. This is checked mechanically and an entry "
    "that fails is discarded.\n"
    "- The answer is often spread across two places that are not next to each "
    "other: a table row naming a line, a supplier and a species, and a sentence "
    "elsewhere saying what that supplier provided or how those cells were "
    "treated. When the answer needs both, cite both ids and say so in "
    "derivation.\n"
    "- Do not return a line the evidence does not name. Do not return a "
    "population or state for a line the evidence only lists in a catalogue row "
    "with nothing else said about it.\n"
    "- An entry is a statement about identity, not about results. Do not report "
    "experimental findings here.\n"
)


def _seal(node: Any) -> Any:
    """Force ``additionalProperties: false`` everywhere; the provider requires it."""
    if isinstance(node, dict):
        node = {key: _seal(value) for key, value in node.items()}
        if node.get("type") == "object":
            node["additionalProperties"] = False
            node.setdefault("required", sorted(node.get("properties", {})))
        return node
    if isinstance(node, list):
        return [_seal(item) for item in node]
    return node


def strict_prepass_schema() -> dict[str, Any]:
    from openai.lib._pydantic import to_strict_json_schema

    return _seal(to_strict_json_schema(EntityResolutionResponse))


def build_prepass_prompt(paper_id: str, rendered_evidence: list[dict[str, Any]]) -> str:
    return (
        f"{PREPASS_PROMPT}\n"
        "Return only the JSON object required by the output schema.\n\n"
        f"paper_id to echo back: {paper_id}\n\n"
        "EVIDENCE:\n" + _canonical_json({"evidence": rendered_evidence})
    )


def run_codex_prepass(
    prompt: str,
    schema: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    timeout: int,
) -> dict[str, Any]:
    """One isolated ``codex exec`` turn. Keys blanked so no billed call can happen."""
    workdir = Path(tempfile.mkdtemp(prefix="codex_entity_"))
    try:
        schema_path = workdir / "schema.json"
        schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
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
    evidence_view: str = DEFAULT_EVIDENCE_VIEW,
    packet_root: Path | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_items: int = 260,
) -> dict[str, Any]:
    if evidence_view not in PACKET_ROOT_BY_VIEW:
        raise ValueError(f"unknown evidence_view: {evidence_view}")
    resolved_root = packet_root or PACKET_ROOT_BY_VIEW[evidence_view]
    packet = load_packet(paper_id, resolved_root)

    run_dir = output_root / paper_id
    if (run_dir / "entity_table.json").exists():
        raise FileExistsError(
            f"{paper_id} already has an entity table under {output_root}; "
            "use a new --output-root."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    sources = {row.source_id: row for row in packet.sources}
    selected = select_identity_evidence(packet, max_items=max_items)
    rendered = evidence_payload(selected, sources)
    prompt = build_prepass_prompt(paper_id, rendered)
    schema = strict_prepass_schema()

    request = {
        "harness": HARNESS,
        "stage": "entity_prepass",
        "prepass_version": PREPASS_VERSION,
        "paper_id": paper_id,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "evidence_view": evidence_view,
        "packet_root": str(resolved_root),
        "packet_checksum": packet.packet_checksum,
        "prompt_checksum": _sha256(prompt.encode("utf-8")),
        "schema_checksum": _sha256(_canonical_json(schema).encode("utf-8")),
        "selected_evidence_ids": [row["evidence_id"] for row in rendered],
        "prompt_characters": len(prompt),
    }
    (run_dir / "request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    raw = run_codex_prepass(
        prompt,
        schema,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )
    (run_dir / "response.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    parsed: EntityResolutionResponse | None = None
    contract_error: str | None = None
    try:
        parsed = EntityResolutionResponse.model_validate_json(raw["text"])
    except Exception as exc:  # noqa: BLE001 - a rejected reply is a real outcome
        contract_error = f"{type(exc).__name__}: {str(exc)[:400]}"

    if parsed is None:
        table = GroundedEntityTable(paper_id=paper_id)
        report = None
    else:
        # The model is asked to echo paper_id; a mismatch would silently file one
        # paper's entities under another, so the packet wins.
        parsed = parsed.model_copy(update={"paper_id": paper_id})
        table, report = ground_entity_table(parsed, packet)

    (run_dir / "entity_table.json").write_text(
        table.model_dump_json(indent=2, exclude_none=False) + "\n", encoding="utf-8"
    )
    if report is not None:
        (run_dir / "grounding_report.json").write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        "harness": HARNESS,
        "harness_note": (
            "Driven through `codex exec`, not client.responses.create. Compare "
            "deltas within this harness only."
        ),
        "stage": "entity_prepass",
        "prepass_version": PREPASS_VERSION,
        "entity_table_version": ENTITY_TABLE_VERSION,
        "paper_id": paper_id,
        "model_requested": model,
        "reasoning_effort": reasoning_effort,
        "evidence_view": evidence_view,
        "packet_checksum": packet.packet_checksum,
        "packet_evidence_items": len(packet.evidence),
        "selected_evidence_items": len(rendered),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": raw["elapsed_seconds"],
        "contract_accepted": parsed is not None,
        "contract_error": contract_error,
        "mappings_proposed": report.proposed if report else 0,
        "mappings_kept": report.kept if report else 0,
        "openai_api_requests": 0,
        "codex_exec_turns": 1,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", action="append", dest="paper_ids")
    parser.add_argument(
        "--evidence-view",
        choices=sorted(PACKET_ROOT_BY_VIEW),
        default=DEFAULT_EVIDENCE_VIEW,
    )
    parser.add_argument("--packet-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-items", type=int, default=260)
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

    paper_ids = args.paper_ids or [f"GP-{index:03d}" for index in range(1, 10)]
    manifests = []
    for paper_id in paper_ids:
        try:
            manifests.append(
                run_one(
                    paper_id,
                    evidence_view=args.evidence_view,
                    packet_root=args.packet_root,
                    output_root=args.output_root,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    timeout=args.timeout,
                    max_items=args.max_items,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad paper must not abort the sweep
            manifests.append(
                {"paper_id": paper_id, "harness": HARNESS, "error": str(exc)[:400]}
            )
        print(json.dumps(manifests[-1], ensure_ascii=False, indent=2))

    batch = {
        "harness": HARNESS,
        "stage": "entity_prepass",
        "prepass_version": PREPASS_VERSION,
        "evidence_view": args.evidence_view,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "papers": manifests,
        "codex_exec_turns": sum(
            int(row.get("codex_exec_turns", 0)) for row in manifests
        ),
        "openai_api_requests": 0,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "batch_manifest.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
