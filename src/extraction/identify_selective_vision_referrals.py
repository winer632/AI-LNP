"""Conservatively identify unresolved table/figure findings for selective vision.

P1 (uniparse ingestion) is the first producer this detector has ever had for
``figure`` / ``table`` / ``table_row`` blocks. Two things follow, and both are
gated behind the ``local_vlm_vision`` flag so the pre-P2 detector is unchanged
when the flag is off:

* a uniparse *image* element folds its caption into the ``figure`` block itself
  rather than emitting a separate ``caption`` block (see
  ``src/rag/ingestion.py::uniparse_blocks``), so requiring a standalone caption
  block would reject every uniparse figure; and
* the block corpus carries the panel geometry and the already-cropped PNG, so
  the referral can name the panel object instead of leaving the crop to be
  re-derived downstream.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from src.config_flags import is_enabled
from src.extraction.compact_validation import ValidationReport
from src.extraction.selective_vision_contracts import (
    P2_FLAG,
    VisionReferral,
    VisualPanel,
)
from src.rag.compact_api_packet import ApiEvidence, ApiSource, CompactApiPacket
from src.rag.compact_packet import CORPUS_ROOT, load_corpus_blocks


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (
    ROOT / "data" / "staging" / "extraction" / "selective_vision_referrals_v1"
)
UNRESOLVED_RE = re.compile(
    r"\b(?:unresolved|image[- ]only|not (?:available|reported) in text|"
    r"requires? (?:visual|vision)|not extracted)\b",
    re.I,
)
VISUAL_RE = re.compile(r"\b(?:fig(?:ure)?|table|graph|panel)\b", re.I)


def _source_visual(source: ApiSource) -> tuple[str, str] | None:
    if source.table_number or source.block_type in {"table", "table_row"}:
        return "unresolved_table", source.table_number or "Table"
    if source.figure_number or source.block_type in {
        "figure",
        "figure_caption",
    }:
        return "unresolved_figure", source.figure_number or "Figure"
    return None


def _self_captioned_evidence(
    packet: CompactApiPacket, visual_source: ApiSource
) -> list[ApiEvidence]:
    """Evidence attached to a uniparse ``figure`` block, which is its own caption.

    ``uniparse_blocks`` writes the image's caption text as the ``figure``
    block's body precisely so that no duplicate caption block is emitted, so
    the figure block *is* the caption evidence for that object.
    """
    if visual_source.block_type != "figure":
        return []
    return [
        evidence
        for evidence in packet.evidence
        if visual_source.source_id in evidence.source_ids
    ]


def identify(
    report: ValidationReport,
    packet: CompactApiPacket,
    corpus_blocks: Mapping[str, Any] | None = None,
) -> tuple[list[VisionReferral], list[dict[str, Any]]]:
    """Return safe referrals and explicit skip reasons.

    ``corpus_blocks`` maps ``block_id`` -> ``DocumentBlock`` for this paper, as
    returned by ``src.rag.compact_packet.load_corpus_blocks``. It is consulted
    only when P2 is on, and only to attach the uniparse panel; the referral
    decision itself never depends on it.
    """
    p2 = is_enabled(P2_FLAG)
    corpus_blocks = corpus_blocks or {}
    evidence_by_id = {row.evidence_id: row for row in packet.evidence}
    source_by_id = {row.source_id: row for row in packet.sources}
    referrals: list[VisionReferral] = []
    skipped: list[dict[str, Any]] = []

    for finding in report.findings:
        if not finding.repairable:
            skipped.append(
                {"finding_id": finding.finding_id, "reason": "not_field_level"}
            )
            continue
        cited = [
            evidence_by_id[evidence_id]
            for evidence_id in finding.cited_evidence_ids
            if evidence_id in evidence_by_id
        ]
        trigger_text = " ".join(
            [finding.code, finding.message, *(row.text for row in cited)]
        )
        if not (UNRESOLVED_RE.search(trigger_text) and VISUAL_RE.search(trigger_text)):
            skipped.append(
                {
                    "finding_id": finding.finding_id,
                    "reason": "no_explicit_unresolved_visual_signal",
                }
            )
            continue

        visual_sources: dict[tuple[str, int, str], ApiSource] = {}
        for evidence in cited:
            for source_id in evidence.source_ids:
                source = source_by_id.get(source_id)
                if source is None or source.page_number is None:
                    continue
                visual = _source_visual(source)
                if visual:
                    trigger, label = visual
                    visual_sources[(trigger, source.page_number, label)] = source
        if len(visual_sources) != 1:
            skipped.append(
                {
                    "finding_id": finding.finding_id,
                    "reason": "visual_source_missing_or_ambiguous",
                    "candidate_visual_sources": len(visual_sources),
                }
            )
            continue

        (trigger, page_number, label), visual_source = next(
            iter(visual_sources.items())
        )
        same_object_source_ids = {
            source.source_id
            for source in packet.sources
            if source.page_number == page_number
            and _source_visual(source) == (trigger, label)
        }
        caption_candidates = [
            evidence
            for evidence in packet.evidence
            if set(evidence.source_ids) & same_object_source_ids
            and any(
                source_by_id[source_id].block_type
                in {"caption", "figure_caption"}
                for source_id in evidence.source_ids
                if source_id in source_by_id
            )
        ]
        if p2 and len(caption_candidates) != 1:
            caption_candidates = _self_captioned_evidence(packet, visual_source)
        results_candidates = [
            evidence
            for evidence in cited
            if any(
                "result" in source_by_id[source_id].section.lower()
                for source_id in evidence.source_ids
                if source_id in source_by_id
            )
        ][:3]
        if len(caption_candidates) != 1 or not results_candidates:
            skipped.append(
                {
                    "finding_id": finding.finding_id,
                    "reason": "caption_or_results_context_missing_or_ambiguous",
                    "caption_candidates": len(caption_candidates),
                    "results_candidates": len(results_candidates),
                }
            )
            continue

        field_name = finding.field_name or ""
        methods_candidates: list[ApiEvidence] = []
        for evidence in packet.evidence:
            if evidence.evidence_id in finding.cited_evidence_ids:
                continue
            is_methods = any(
                "method" in source_by_id[source_id].section.lower()
                for source_id in evidence.source_ids
                if source_id in source_by_id
            )
            field_match = any(
                field_name.lower() in tag.lower()
                or tag.lower() in field_name.lower()
                for tag in evidence.retrieval_field_tags
            )
            object_match = label.lower() in evidence.text.lower()
            if is_methods and (field_match or object_match):
                methods_candidates.append(evidence)

        panel = None
        if p2:
            block = corpus_blocks.get(visual_source.chunk_id)
            if block is not None:
                panel = VisualPanel.from_block(block)
                if panel is not None and panel.page_id != page_number:
                    # Geometry that disagrees with the packet is not provenance.
                    panel = None
        referrals.append(
            VisionReferral(
                referral_version="selective-vision-referral-1.0.0",
                paper_id=report.paper_id,
                finding_id=finding.finding_id,
                trigger=trigger,
                reason=(
                    "Deterministic text review found an unresolved field, an "
                    "explicit visual reference, and one matching visual source."
                ),
                source_id=visual_source.source_id,
                page_number=page_number,
                figure_or_table=label,
                crop_box=None,
                caption_evidence_id=caption_candidates[0].evidence_id,
                referring_results_evidence_ids=[
                    row.evidence_id for row in results_candidates
                ],
                methods_evidence_ids=[
                    row.evidence_id for row in methods_candidates[:3]
                ],
                panel=panel,
            )
        )
    return referrals, skipped


def write_referrals(
    report: ValidationReport,
    packet: CompactApiPacket,
    output_root: Path = OUTPUT_ROOT,
    corpus_root: Path = CORPUS_ROOT,
) -> dict[str, Any]:
    blocks = (
        load_corpus_blocks(corpus_root, report.paper_id)
        if is_enabled(P2_FLAG)
        else {}
    )
    referrals, skipped = identify(report, packet, blocks)
    paper_root = output_root / report.paper_id
    paper_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for referral in referrals:
        path = paper_root / f"{referral.finding_id}.json"
        path.write_text(referral.model_dump_json(indent=2) + "\n", encoding="utf-8")
        paths.append(str(path))
    manifest = {
        "paper_id": report.paper_id,
        "referrals": paths,
        "skipped": skipped,
    }
    (paper_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            write_referrals(
                ValidationReport.model_validate_json(
                    args.validation_report.read_text(encoding="utf-8")
                ),
                CompactApiPacket.model_validate_json(
                    args.packet.read_text(encoding="utf-8")
                ),
                args.output_dir,
                args.corpus_root,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
