"""Serialize the full local evidence view as an extraction-ready API packet.

The compact API packet (``src/rag/compact_api_packet.py``) selects a small,
token-budgeted slice of the retrieval evidence.  The full view rejoins that
slice with every locally parsed block so the extraction call can see the whole
paper.  Both views are the same ``CompactApiPacket`` type and use the same
checksum rule, so ``run_compact_one_call.load_packet`` reads either one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.extraction.build_full_outcome_inventory import (
    CORPUS_ROOT,
    full_api_view,
    full_corpus_view,
)
from src.extraction.compact_prompt_v1 import COMPACT_EXTRACTION_PROMPT
from src.rag.compact_api_packet import (
    OUTPUT_ROOT as COMPACT_PACKET_ROOT,
    SCHEMA_PATH,
    TOKEN_ESTIMATION_METHOD,
    CompactApiPacket,
    estimate_tokens,
)
from src.rag.compact_packet import (
    OUTPUT_ROOT as REVIEW_PACKET_ROOT,
    ROOT,
    CompactEvidencePacket,
    display_path,
)


OUTPUT_ROOT = ROOT / "data" / "staging" / "rag" / "full_api_packets_v1"
FULL_VIEW_VERSION = "full-api-packet-1.0.0"
FULL_CORPUS_TAG = "full_corpus"
CORPUS_EVIDENCE_MARKER = "-FC-"


def resolve_directory(path: Path | str) -> Path:
    """Accept relative and absolute paths from a CLI without exploding later.

    ``compact_api_packet.build_all`` reports ``path.relative_to(ROOT)``, which
    raises for any ``--output-dir`` outside the repository or given relative to
    a different working directory.  Resolving here plus :func:`display_path`
    keeps both forms working.
    """
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return Path(path).resolve()


def load_review_packet(
    paper_id: str,
    review_packet_root: Path = REVIEW_PACKET_ROOT,
) -> CompactEvidencePacket:
    packet_path = review_packet_root / f"{paper_id}.json"
    return CompactEvidencePacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    )


def corpus_path_for(paper_id: str, corpus_root: Path = CORPUS_ROOT) -> Path:
    return corpus_root / f"{paper_id}.blocks.jsonl"


def build_full_packet(
    paper_id: str,
    *,
    review_packet_root: Path = REVIEW_PACKET_ROOT,
    corpus_root: Path = CORPUS_ROOT,
    allow_missing_corpus: bool = False,
) -> CompactApiPacket:
    """Return the full evidence view for one paper as an API packet.

    Missing parsed blocks are an error by default: a retrieval-only packet must
    never be shipped under the ``full`` label.
    """
    review_packet = load_review_packet(paper_id, review_packet_root)
    corpus_path = corpus_path_for(paper_id, corpus_root)
    if corpus_path.exists():
        return full_corpus_view(review_packet, corpus_path)
    if not allow_missing_corpus:
        raise FileNotFoundError(
            f"No parsed corpus blocks for {paper_id} at {corpus_path}; "
            "the full evidence view would silently degrade to retrieval-only "
            "evidence. Pass allow_missing_corpus=True to accept that."
        )
    return full_api_view(review_packet)


def _is_corpus_evidence(evidence_id: str, paper_id: str) -> bool:
    return evidence_id.startswith(f"{paper_id}{CORPUS_EVIDENCE_MARKER}")


def packet_counts(packet: CompactApiPacket) -> dict[str, int]:
    corpus_evidence = [
        row
        for row in packet.evidence
        if _is_corpus_evidence(row.evidence_id, packet.paper_id)
    ]
    return {
        "evidence": len(packet.evidence),
        "sources": len(packet.sources),
        "retrieval_evidence": len(packet.evidence) - len(corpus_evidence),
        "corpus_expanded_evidence": len(corpus_evidence),
        "outcome_tagged_evidence": sum(
            "outcomes" in row.retrieval_field_tags for row in packet.evidence
        ),
        "full_corpus_only_evidence": sum(
            set(row.retrieval_field_tags) == {FULL_CORPUS_TAG}
            for row in packet.evidence
        ),
    }


def estimated_input_tokens(packet: CompactApiPacket) -> dict[str, int]:
    """Estimate the same three input parts the compact packet manifest reports."""
    prompt_tokens = estimate_tokens(COMPACT_EXTRACTION_PROMPT)
    schema_tokens = estimate_tokens(
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    )
    packet_tokens = estimate_tokens(
        packet.model_dump(mode="json", exclude_none=True)
    )
    return {
        "prompt": prompt_tokens,
        "response_schema": schema_tokens,
        "evidence_packet": packet_tokens,
        "total": prompt_tokens + schema_tokens + packet_tokens,
    }


def _compact_comparison(
    packet: CompactApiPacket,
    compact_packet_root: Path,
) -> dict[str, Any] | None:
    compact_path = compact_packet_root / f"{packet.paper_id}.json"
    if not compact_path.exists():
        return None
    compact_packet = CompactApiPacket.model_validate_json(
        compact_path.read_text(encoding="utf-8")
    )
    compact_tokens = estimate_tokens(
        compact_packet.model_dump(mode="json", exclude_none=True)
    )
    full_tokens = estimate_tokens(
        packet.model_dump(mode="json", exclude_none=True)
    )
    return {
        "compact_packet_path": display_path(compact_path),
        "compact_packet_checksum": compact_packet.packet_checksum,
        "compact_evidence": len(compact_packet.evidence),
        "compact_sources": len(compact_packet.sources),
        "compact_evidence_packet_tokens": compact_tokens,
        "added_evidence": len(packet.evidence) - len(compact_packet.evidence),
        "added_sources": len(packet.sources) - len(compact_packet.sources),
        "added_evidence_packet_tokens": full_tokens - compact_tokens,
        "evidence_multiple": round(
            len(packet.evidence) / len(compact_packet.evidence), 2
        )
        if compact_packet.evidence
        else None,
        "evidence_packet_token_multiple": round(full_tokens / compact_tokens, 2)
        if compact_tokens
        else None,
    }


def build_manifest(
    packet: CompactApiPacket,
    *,
    review_packet: CompactEvidencePacket | None = None,
    corpus_path: Path | None = None,
    compact_packet_root: Path = COMPACT_PACKET_ROOT,
) -> dict[str, Any]:
    return {
        "paper_id": packet.paper_id,
        "full_view_version": FULL_VIEW_VERSION,
        "packet_version": packet.packet_version,
        "evidence_view": "full",
        "full_packet_checksum": packet.packet_checksum,
        "source_review_packet_checksum": (
            review_packet.packet_checksum if review_packet else None
        ),
        "corpus_path": display_path(corpus_path) if corpus_path else None,
        "corpus_available": bool(corpus_path and corpus_path.exists()),
        "token_estimation_method": TOKEN_ESTIMATION_METHOD,
        "estimated_input_tokens": estimated_input_tokens(packet),
        "counts": packet_counts(packet),
        "blocked_fields": packet.blocked_fields,
        "compact_packet_comparison": _compact_comparison(
            packet, compact_packet_root
        ),
    }


def write_outputs(
    packet: CompactApiPacket,
    manifest: dict[str, Any],
    output_root: Path = OUTPUT_ROOT,
) -> tuple[Path, Path]:
    output_root = resolve_directory(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    packet_path = output_root / f"{packet.paper_id}.json"
    manifest_path = output_root / f"{packet.paper_id}.manifest.json"
    packet_path.write_text(
        json.dumps(
            packet.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return packet_path, manifest_path


def build_all(
    *,
    review_packet_root: Path = REVIEW_PACKET_ROOT,
    corpus_root: Path = CORPUS_ROOT,
    output_root: Path = OUTPUT_ROOT,
    compact_packet_root: Path = COMPACT_PACKET_ROOT,
    paper_ids: list[str] | None = None,
    allow_missing_corpus: bool = False,
) -> dict[str, Any]:
    output_root = resolve_directory(output_root)
    selected = set(paper_ids or [])
    papers = []
    for review_path in sorted(review_packet_root.glob("GP-*.json")):
        paper_id = review_path.stem
        if "." in paper_id:
            continue
        if selected and paper_id not in selected:
            continue
        review_packet = load_review_packet(paper_id, review_packet_root)
        packet = build_full_packet(
            paper_id,
            review_packet_root=review_packet_root,
            corpus_root=corpus_root,
            allow_missing_corpus=allow_missing_corpus,
        )
        manifest = build_manifest(
            packet,
            review_packet=review_packet,
            corpus_path=corpus_path_for(paper_id, corpus_root),
            compact_packet_root=compact_packet_root,
        )
        packet_path, manifest_path = write_outputs(packet, manifest, output_root)
        papers.append(
            {
                "paper_id": paper_id,
                "packet_path": display_path(packet_path),
                "manifest_path": display_path(manifest_path),
                "packet_checksum": packet.packet_checksum,
                **manifest["counts"],
                "estimated_input_tokens": manifest["estimated_input_tokens"],
                "compact_packet_comparison": manifest["compact_packet_comparison"],
            }
        )

    run_manifest = {
        "full_view_version": FULL_VIEW_VERSION,
        "evidence_view": "full",
        "token_estimation_method": TOKEN_ESTIMATION_METHOD,
        "review_packet_root": display_path(review_packet_root),
        "corpus_root": display_path(corpus_root),
        "output_root": display_path(output_root),
        "papers": papers,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "manifest.md").write_text(
        _manifest_markdown(papers) + "\n",
        encoding="utf-8",
    )
    return run_manifest


def _manifest_markdown(papers: list[dict[str, Any]]) -> str:
    lines = [
        "# Full API packet v1 - complete local evidence view",
        "",
        "Every locally parsed block is present; no token budget is applied here.",
        f"- Estimation method: `{TOKEN_ESTIMATION_METHOD}`",
        "",
        "| Paper | Evidence | Retrieval | Corpus-expanded | Sources | "
        "Estimated packet tokens | Estimated total input tokens | vs compact evidence | vs compact tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in papers:
        comparison = row.get("compact_packet_comparison") or {}
        evidence_multiple = comparison.get("evidence_multiple")
        token_multiple = comparison.get("evidence_packet_token_multiple")
        lines.append(
            f"| {row['paper_id']} | {row['evidence']} | "
            f"{row['retrieval_evidence']} | {row['corpus_expanded_evidence']} | "
            f"{row['sources']} | "
            f"{row['estimated_input_tokens']['evidence_packet']:,} | "
            f"{row['estimated_input_tokens']['total']:,} | "
            f"{'-' if evidence_multiple is None else f'{evidence_multiple}x'} | "
            f"{'-' if token_multiple is None else f'{token_multiple}x'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description=(
            "Write the complete local evidence view for each paper as an API "
            "packet that run_compact_one_call can send with --evidence-view full."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Relative or absolute output directory.",
    )
    parser.add_argument("--review-packet-root", type=Path, default=REVIEW_PACKET_ROOT)
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    parser.add_argument(
        "--compact-packet-root",
        type=Path,
        default=COMPACT_PACKET_ROOT,
        help="Compact packets used only to report the size delta.",
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        dest="paper_ids",
        help="Restrict the run to one paper; repeatable.",
    )
    parser.add_argument(
        "--allow-missing-corpus",
        action="store_true",
        help="Accept retrieval-only evidence when parsed blocks are missing.",
    )
    args = parser.parse_args(argv)
    run_manifest = build_all(
        review_packet_root=resolve_directory(args.review_packet_root),
        corpus_root=resolve_directory(args.corpus_root),
        output_root=resolve_directory(args.output_dir),
        compact_packet_root=resolve_directory(args.compact_packet_root),
        paper_ids=args.paper_ids,
        allow_missing_corpus=args.allow_missing_corpus,
    )
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2))
    return run_manifest


if __name__ == "__main__":
    main()
