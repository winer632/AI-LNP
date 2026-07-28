"""Budget the full evidence view down with structure-aware priority.

Why this exists
---------------
Two measured facts motivated this packet:

* The full evidence view recovers one extra gold outcome over the 16k compact
  packet, but costs 4-8x the tokens (GP-006: 15,993 -> 128,234).
* Enlarging the evidence pool is not what did the work. Structuring the same
  content through uniparse recovered GO-003 on its own, and candidate-slot
  enforcement recovered GO-004 and GO-007 at the compact size. The candidate
  count is also what breaks slot enforcement: 39 candidates validate cleanly,
  168 do not.

So the useful move is not "send everything", it is "never let the ranker drop a
table row or a caption". This module keeps the full view's structured evidence
unconditionally and then fills the remaining budget by retrieval priority,
producing a packet the same size as the compact one.

The output is a normal :class:`CompactApiPacket`, signed with the same checksum
rule, so ``load_packet`` and every downstream consumer accept it unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.rag.compact_api_packet import (
    DEFAULT_PACKET_BUDGET_TOKENS,
    API_PACKET_VERSION,
    ApiEvidence,
    CompactApiPacket,
    estimate_tokens,
)
from src.rag.full_api_packet import (
    CORPUS_ROOT,
    REVIEW_PACKET_ROOT,
    build_full_packet,
    resolve_directory,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "data" / "staging" / "rag" / "structured_compact_packets_v1"

# Block types that carry a value in a place the sentence ranker reads poorly.
# A table row is one grid line; dropping it separates a number from the column
# header that gives it meaning.
STRUCTURED_BLOCK_TYPES = frozenset(
    {"table", "table_row", "caption", "figure_caption", "figure"}
)


def _source_block_types(packet: CompactApiPacket) -> dict[str, str]:
    return {
        source.source_id: (source.block_type or "")
        for source in packet.sources
    }


def is_structured(evidence: ApiEvidence, block_types: dict[str, str]) -> bool:
    """Report whether this evidence came from a structured block."""
    return any(
        block_types.get(source_id, "") in STRUCTURED_BLOCK_TYPES
        for source_id in (evidence.source_ids or ())
    )


def _retrieval_rank(evidence: ApiEvidence) -> int:
    """Higher is kept first. Retrieval-tagged evidence outranks corpus filler.

    The full view tags corpus-expanded passages only ``outcomes`` or
    ``full_corpus`` and gives them no experiment anchors, so they are weaker
    than evidence that a field-specific query actually retrieved.
    """
    tags = set(evidence.retrieval_field_tags or ())
    if tags - {"full_corpus"}:
        score = 20
    else:
        score = 0
    if "outcomes" in tags:
        score += 10
    if evidence.experiment_candidate_ids:
        score += 5
    return score


def select_evidence(
    packet: CompactApiPacket,
    *,
    budget_tokens: int = DEFAULT_PACKET_BUDGET_TOKENS,
) -> tuple[list[ApiEvidence], dict[str, Any]]:
    """Keep every structured passage, then fill the budget by retrieval rank."""
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")

    block_types = _source_block_types(packet)
    order = {row.evidence_id: index for index, row in enumerate(packet.evidence)}

    structured = [row for row in packet.evidence if is_structured(row, block_types)]
    rest = [row for row in packet.evidence if not is_structured(row, block_types)]
    rest.sort(key=lambda row: (-_retrieval_rank(row), order[row.evidence_id]))

    selected: list[ApiEvidence] = list(structured)
    used = sum(estimate_tokens(row.model_dump(mode="json", exclude_none=True)) for row in selected)
    structured_tokens = used
    overflow = used > budget_tokens

    for row in rest:
        cost = estimate_tokens(row.model_dump(mode="json", exclude_none=True))
        if used + cost > budget_tokens:
            continue
        selected.append(row)
        used += cost

    selected.sort(key=lambda row: order[row.evidence_id])
    report = {
        "budget_tokens": budget_tokens,
        "selected_evidence": len(selected),
        "structured_evidence": len(structured),
        "filler_evidence": len(selected) - len(structured),
        "excluded_evidence": len(packet.evidence) - len(selected),
        "estimated_evidence_tokens": used,
        "structured_evidence_tokens": structured_tokens,
        # Structured evidence is never dropped, so a paper whose tables alone
        # exceed the budget overshoots on purpose rather than losing a row.
        "structured_evidence_over_budget": overflow,
    }
    return selected, report


def sign(paper_id: str, blocked_fields: list[str], sources, evidence) -> CompactApiPacket:
    """Rebuild and sign a packet using the same rule as compact_api_packet."""
    payload = {
        "packet_version": API_PACKET_VERSION,
        "paper_id": paper_id,
        "blocked_fields": sorted(blocked_fields),
        "sources": [row.model_dump(mode="json", exclude_none=True) for row in sources],
        "evidence": [row.model_dump(mode="json", exclude_none=True) for row in evidence],
    }
    checksum = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CompactApiPacket.model_validate({**payload, "packet_checksum": checksum})


def build_structured_compact_packet(
    paper_id: str,
    *,
    review_packet_root: Path = REVIEW_PACKET_ROOT,
    corpus_root: Path = CORPUS_ROOT,
    budget_tokens: int = DEFAULT_PACKET_BUDGET_TOKENS,
) -> tuple[CompactApiPacket, dict[str, Any]]:
    full = build_full_packet(
        paper_id, review_packet_root=review_packet_root, corpus_root=corpus_root
    )
    evidence, report = select_evidence(full, budget_tokens=budget_tokens)

    kept_source_ids = {sid for row in evidence for sid in (row.source_ids or ())}
    sources = [row for row in full.sources if row.source_id in kept_source_ids]

    # Context pointers may reference evidence the budget dropped.
    kept_ids = {row.evidence_id for row in evidence}
    cleaned = []
    for row in evidence:
        data = row.model_dump(mode="json", exclude_none=True)
        for key in ("context_before_evidence_id", "context_after_evidence_id"):
            if data.get(key) not in kept_ids:
                data.pop(key, None)
        cleaned.append(ApiEvidence.model_validate(data))

    packet = sign(paper_id, full.blocked_fields, sources, cleaned)
    report.update(
        {
            "paper_id": paper_id,
            "full_view_evidence": len(full.evidence),
            "full_view_sources": len(full.sources),
            "selected_sources": len(sources),
            "packet_checksum": packet.packet_checksum,
        }
    )
    return packet, report


def build_all(
    paper_ids: list[str],
    *,
    output_dir: Path,
    review_packet_root: Path = REVIEW_PACKET_ROOT,
    corpus_root: Path = CORPUS_ROOT,
    budget_tokens: int = DEFAULT_PACKET_BUDGET_TOKENS,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for paper_id in paper_ids:
        packet, report = build_structured_compact_packet(
            paper_id,
            review_packet_root=review_packet_root,
            corpus_root=corpus_root,
            budget_tokens=budget_tokens,
        )
        (output_dir / f"{paper_id}.json").write_text(
            packet.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
        )
        reports.append(report)
    manifest = {
        "packet_version": "structured-compact-1.0.0",
        "budget_tokens": budget_tokens,
        "papers": reports,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build budget-sized packets that never drop structured evidence."
    )
    parser.add_argument("--paper-id", action="append", dest="paper_ids")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--review-packet-root", type=Path, default=REVIEW_PACKET_ROOT)
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    parser.add_argument(
        "--budget-tokens", type=int, default=DEFAULT_PACKET_BUDGET_TOKENS
    )
    args = parser.parse_args()

    paper_ids = args.paper_ids or [f"GP-{n:03d}" for n in range(1, 10)]
    manifest = build_all(
        paper_ids,
        output_dir=resolve_directory(args.output_dir),
        review_packet_root=resolve_directory(args.review_packet_root),
        corpus_root=resolve_directory(args.corpus_root),
        budget_tokens=args.budget_tokens,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
