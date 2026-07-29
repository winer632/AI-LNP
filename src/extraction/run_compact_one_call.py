"""Run exactly one compact OpenAI extraction request for human review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema
from src.config_flags import is_enabled
from src.extraction.compact_contracts import active_response_contract
from src.extraction.compact_validation import validate_candidate
from src.extraction.assess_outcome_complexity import assess
from src.extraction.build_outcome_candidates import build_candidates
from src.extraction.check_outcome_coverage import check
from src.extraction.compact_prompt_v1 import (
    COMPACT_EXTRACTION_PROMPT,
    PROMPT_VERSION,
    active_prompt,
    candidate_slot_payload,
    prompt_sha256,
)
from src.rag.compact_api_packet import CompactApiPacket, estimate_tokens


ROOT = Path(__file__).resolve().parents[2]
PACKET_ROOT = ROOT / "data" / "staging" / "rag" / "compact_api_packets_v1"
FULL_PACKET_ROOT = ROOT / "data" / "staging" / "rag" / "full_api_packets_v1"
OUTPUT_ROOT = ROOT / "data" / "staging" / "extraction" / "compact_one_call_v1"
SCHEMA_PATH = (
    ROOT
    / "docs"
    / "extraction"
    / "schemas"
    / "compact_v1"
    / "compact_extraction_response.schema.json"
)
DEFAULT_EVIDENCE_VIEW = "compact"
PACKET_ROOT_BY_VIEW = {
    "compact": PACKET_ROOT,
    "full": FULL_PACKET_ROOT,
}
# The full view ships every parsed block, so an oversized or wrongly built
# packet must fail locally instead of becoming a very expensive request.
DEFAULT_MAX_INPUT_TOKENS = 150_000
# P3. Off by default; when on, the locally enumerated outcome candidates are
# sent with the packet and the response must account for every one of them.
CANDIDATE_SLOT_FLAG = "candidate_slot_enforcement"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def load_packet(paper_id: str, packet_root: Path = PACKET_ROOT) -> CompactApiPacket:
    packet_path = packet_root / f"{paper_id}.json"
    packet = CompactApiPacket.model_validate_json(packet_path.read_text(encoding="utf-8"))
    unsigned = packet.model_dump(mode="json", exclude={"packet_checksum"}, exclude_none=True)
    actual_checksum = _sha256(_canonical_json(unsigned))
    if actual_checksum != packet.packet_checksum:
        raise ValueError(
            f"Packet checksum mismatch for {paper_id}: "
            f"expected {packet.packet_checksum}, calculated {actual_checksum}"
        )
    return packet


def packet_root_for(evidence_view: str = DEFAULT_EVIDENCE_VIEW) -> Path:
    try:
        return PACKET_ROOT_BY_VIEW[evidence_view]
    except KeyError:
        raise ValueError(
            f"Unknown evidence view {evidence_view!r}; "
            f"expected one of {sorted(PACKET_ROOT_BY_VIEW)}"
        ) from None


def estimated_input_tokens(
    packet_payload: dict[str, Any],
    *,
    prompt: str = COMPACT_EXTRACTION_PROMPT,
    candidate_slots: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Estimate the request's input size with the packet builder's own method.

    The candidate-slot term is added only when slots are actually sent, so the
    baseline estimate -- and the manifest it is written into -- is unchanged.
    """
    prompt_tokens = estimate_tokens(prompt)
    schema_tokens = estimate_tokens(
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    )
    packet_tokens = estimate_tokens(packet_payload)
    counts = {
        "prompt": prompt_tokens,
        "response_schema": schema_tokens,
        "evidence_packet": packet_tokens,
    }
    if candidate_slots is not None:
        counts["candidate_slots"] = estimate_tokens(candidate_slots)
    counts["total"] = sum(counts.values())
    return counts


def request_fingerprint(
    packet: CompactApiPacket,
    model: str,
    evidence_view: str = DEFAULT_EVIDENCE_VIEW,
    *,
    prompt_version: str = PROMPT_VERSION,
    prompt_checksum: str | None = None,
    candidate_slots_checksum: str | None = None,
) -> str:
    """Identify one request, and never let two views share a prompt cache key.

    The compact view's fingerprint payload is frozen: the view key is added only
    for other views, and the candidate-slot key only when slots are sent, so
    already-cached compact requests keep their cache key. A slot-enforced call
    carries a different prompt version and a checksum of the slots themselves,
    which is what stops it from reusing a baseline cache entry.
    """
    payload = {
        "paper_id": packet.paper_id,
        "packet_checksum": packet.packet_checksum,
        "prompt_version": prompt_version,
        "prompt_checksum": prompt_checksum or prompt_sha256(),
        "schema_checksum": _sha256(SCHEMA_PATH.read_bytes()),
        "model": model,
    }
    if evidence_view != DEFAULT_EVIDENCE_VIEW:
        payload["evidence_view"] = evidence_view
    if candidate_slots_checksum is not None:
        payload["candidate_slots"] = candidate_slots_checksum
    return _sha256(_canonical_json(payload))


def _refusals(response: Any) -> list[str]:
    return [
        item.refusal
        for output in response.output
        if output.type == "message"
        for item in output.content
        if item.type == "refusal"
    ]


def run_one(
    paper_id: str,
    *,
    model: str,
    client: OpenAI,
    packet_root: Path | None = None,
    output_root: Path = OUTPUT_ROOT,
    max_output_tokens: int = 12_000,
    evidence_view: str = DEFAULT_EVIDENCE_VIEW,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
) -> dict[str, Any]:
    """Make one paid request and persist its request, response, result, and usage."""
    if packet_root is None:
        packet_root = packet_root_for(evidence_view)
    elif evidence_view not in PACKET_ROOT_BY_VIEW:
        raise ValueError(
            f"Unknown evidence view {evidence_view!r}; "
            f"expected one of {sorted(PACKET_ROOT_BY_VIEW)}"
        )
    packet = load_packet(paper_id, packet_root)
    run_dir = output_root / paper_id
    result_path = run_dir / "result.json"
    raw_response_path = run_dir / "response.json"
    if result_path.exists() or raw_response_path.exists():
        raise FileExistsError(
            f"A completed response already exists for {paper_id}; refusing a duplicate paid call."
        )
    packet_payload = packet.model_dump(mode="json", exclude_none=True)
    # Both are pure functions of the packet; computing them before the size
    # guard lets the guard also cover the candidate slots, without writing
    # anything to disk any earlier than before.
    complexity = assess(packet)
    outcome_candidates = (
        build_candidates(packet) if complexity.route == "complex" else None
    )
    candidate_slots = (
        candidate_slot_payload(outcome_candidates)
        if outcome_candidates and is_enabled(CANDIDATE_SLOT_FLAG)
        else None
    )
    prompt = active_prompt(candidate_slots is not None)
    response_model = active_response_contract(candidate_slots is not None)
    input_tokens = estimated_input_tokens(
        packet_payload, prompt=prompt.text, candidate_slots=candidate_slots
    )
    if input_tokens["total"] > max_input_tokens:
        raise ValueError(
            f"Estimated input for {paper_id} in the {evidence_view} view is "
            f"{input_tokens['total']:,} tokens, above the "
            f"{max_input_tokens:,} token limit; refusing to send the request."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "complexity.json").write_text(
        complexity.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    if outcome_candidates is not None:
        (run_dir / "outcome_candidates.json").write_text(
            json.dumps(
                [row.model_dump(mode="json") for row in outcome_candidates],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if candidate_slots is not None:
        (run_dir / "candidate_slots.json").write_text(
            json.dumps(candidate_slots, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    fingerprint = request_fingerprint(
        packet,
        model,
        evidence_view,
        prompt_version=prompt.version,
        prompt_checksum=prompt.checksum,
        candidate_slots_checksum=(
            None if candidate_slots is None else _sha256(_canonical_json(candidate_slots))
        ),
    )
    request_snapshot = {
        "paper_id": paper_id,
        "model": model,
        "evidence_view": evidence_view,
        "packet_root": str(packet_root),
        "estimated_input_tokens": input_tokens,
        "max_input_tokens": max_input_tokens,
        "reasoning_effort": "low",
        "store": False,
        "service_tier": "default",
        "max_output_tokens": max_output_tokens,
        "prompt_version": prompt.version,
        "prompt_checksum": prompt.checksum,
        "schema_checksum": _sha256(SCHEMA_PATH.read_bytes()),
        "packet_checksum": packet.packet_checksum,
        "request_fingerprint": fingerprint,
        "system_prompt": prompt.text,
        "packet": packet_payload,
    }
    if candidate_slots is not None:
        request_snapshot["candidate_slot_enforcement"] = True
        request_snapshot["candidate_slots"] = candidate_slots
    (run_dir / "request.json").write_text(
        json.dumps(request_snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # The packet stays its own message in its original position, so the slot
    # message can only ever be appended after it.
    request_input: list[dict[str, str]] = [
        {"role": "system", "content": prompt.text},
        {"role": "user", "content": _canonical_json(packet_payload)},
    ]
    if candidate_slots is not None:
        request_input.append(
            {
                "role": "user",
                "content": _canonical_json({"candidate_slots": candidate_slots}),
            }
        )
    started_at = datetime.now(timezone.utc)
    response = client.responses.create(
        model=model,
        reasoning={"effort": "low"},
        store=False,
        service_tier="default",
        max_output_tokens=max_output_tokens,
        prompt_cache_key=fingerprint,
        input=request_input,
        text={
            "format": {
                "type": "json_schema",
                "name": response_model.__name__,
                "schema": to_strict_json_schema(response_model),
                "strict": True,
            }
        },
    )
    completed_at = datetime.now(timezone.utc)

    raw_response_path.write_text(
        json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not response.output_text:
        detail = "; ".join(_refusals(response)) or "no parsed output"
        raise RuntimeError(f"OpenAI structured extraction failed: {detail}")
    candidate_path = run_dir / "candidate.json"
    candidate_path.write_text(response.output_text + "\n", encoding="utf-8")
    parsed, validation_report = validate_candidate(
        response.output_text,
        paper_id=paper_id,
        allowed_evidence_ids={row.evidence_id for row in packet.evidence},
        required_candidate_ids=(
            None
            if candidate_slots is None
            else [str(slot["candidate_id"]) for slot in candidate_slots]
        ),
    )
    (run_dir / "validation_report.json").write_text(
        validation_report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    if parsed is None:
        raise ValueError(
            f"Compact candidate failed {len(validation_report.findings)} "
            "deterministic validation check(s); see validation_report.json"
        )

    # The dispositions are a record of this run, not part of the extraction
    # contract, so result.json stays exactly what every downstream consumer of
    # CompactExtractionResponse already expects.
    dispositions = [
        row.model_dump(mode="json")
        for row in getattr(parsed, "candidate_dispositions", [])
    ]
    if candidate_slots is not None:
        (run_dir / "candidate_dispositions.json").write_text(
            json.dumps(dispositions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    result_payload = parsed.model_dump(
        mode="json", exclude={"candidate_dispositions"}
    )
    result_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    coverage = None
    if complexity.route == "complex":
        coverage = check(
            packet,
            result_payload,
            assessment=complexity,
            candidates=outcome_candidates,
        )
        (run_dir / "outcome_coverage.json").write_text(
            coverage.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    needs_coverage_review = bool(
        coverage and coverage.status == "review_unmatched_groups"
    )
    manifest = {
        "status": (
            "completed_pending_outcome_coverage_review"
            if needs_coverage_review
            else "completed_pending_human_verification"
        ),
        "paper_id": paper_id,
        "paid_api_requests": 1,
        "model_requested": model,
        "model_returned": response.model,
        "response_id": response.id,
        "request_fingerprint": fingerprint,
        "evidence_view": evidence_view,
        "packet_root": str(packet_root),
        "estimated_input_tokens": input_tokens,
        "max_input_tokens": max_input_tokens,
        "packet_checksum": packet.packet_checksum,
        "prompt_version": prompt.version,
        "prompt_checksum": prompt.checksum,
        "schema_checksum": _sha256(SCHEMA_PATH.read_bytes()),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": (completed_at - started_at).total_seconds(),
        "usage": response.usage.model_dump(mode="json") if response.usage else None,
        "eligibility": parsed.eligibility.model_dump(mode="json"),
        "record_counts": {
            "formulations": len(parsed.formulations),
            "components": len(parsed.components),
            "experiments": len(parsed.experiments),
            "outcomes": len(parsed.outcomes),
            "unresolved_items": len(parsed.unresolved_items),
        },
        "checks": {
            "structured_output_valid": True,
            "paper_id_matches": True,
            "all_evidence_ids_exist_in_packet": True,
            "validation_report_status": validation_report.status,
            "narrative_requested": False,
            "complexity_route": complexity.route,
            "outcome_coverage_status": (
                coverage.status if coverage else "ordinary_validation_only"
            ),
        },
    }
    if candidate_slots is not None:
        manifest["candidate_slot_enforcement"] = True
        manifest["candidate_slots"] = len(candidate_slots)
        manifest["candidate_disposition_counts"] = {
            code: sum(1 for row in dispositions if row["disposition"] == code)
            for code in ("extracted", "not_an_outcome", "unresolved")
        }
        manifest["checks"]["candidate_slot_coverage"] = "complete"
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send exactly one compact paper packet to OpenAI."
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument(
        "--evidence-view",
        choices=sorted(PACKET_ROOT_BY_VIEW),
        default=DEFAULT_EVIDENCE_VIEW,
        help=(
            "Which evidence view to send: the budgeted compact packet, or the "
            "full local view containing every parsed block. Selects the default "
            "--packet-root."
        ),
    )
    parser.add_argument(
        "--packet-root",
        type=Path,
        default=None,
        help=(
            "Relative or absolute directory holding {paper_id}.json packets; "
            "overrides the directory implied by --evidence-view."
        ),
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help="Refuse to send a request whose estimated input exceeds this size.",
    )
    parser.add_argument(
        "--confirm-paid-call",
        action="store_true",
        help="Required guard acknowledging that exactly one paid API request may be made.",
    )
    args = parser.parse_args()
    if not args.confirm_paid_call:
        parser.error("--confirm-paid-call is required")

    packet_root = packet_root_for(args.evidence_view)
    if args.packet_root is not None:
        packet_root = args.packet_root.expanduser()
        if not packet_root.is_absolute():
            packet_root = (Path.cwd() / packet_root).resolve()

    load_dotenv(ROOT / ".env")
    model = os.getenv("COMPACT_EXTRACTION_MODEL", "gpt-5.6-terra")
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        timeout=300.0,
        max_retries=0,
    )
    manifest = run_one(
        args.paper_id,
        model=model,
        client=client,
        packet_root=packet_root,
        evidence_view=args.evidence_view,
        max_input_tokens=args.max_input_tokens,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
