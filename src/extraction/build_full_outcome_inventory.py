"""Build outcome candidates from the full local compact packet, before API budgeting."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.extraction.build_outcome_candidates import build_candidates
from src.extraction.consolidate_outcome_candidates import consolidate
from src.extraction.outcome_inventory_contracts import OutcomeInventory
from src.rag.compact_api_packet import (
    ApiEvidence,
    ApiSource,
    CompactApiPacket,
    _prepare,
    _selection_payload,
)
from src.rag.compact_packet import (
    CompactEvidencePacket,
    sentence_clauses,
)
from src.rag.evidence_locators import clause_locator
from src.rag.models import DocumentBlock


ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = ROOT / "data/staging/rag/gold_v1"
OCR_NORMALIZATION = str.maketrans(
    {
        "Ɵ": "ti",
        "Ʃ": "tt",
        "Ō": "ft",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
    }
)
CORPUS_OUTCOME_SIGNAL = re.compile(
    r"(?:"
    r"\b(?:express|transfect|deliver|uptake|internaliz|edit|insert|delet|"
    r"indel|knockdown|silenc|activity|efficacy|survival|fibrosis|steatosis|"
    r"necrosis|phagocyt|eliminat|locali[sz]|colocali[sz]|co-stain|"
    r"protect|improv|reduce|increase|decrease|absent|few)\w*|"
    r"\d+(?:\.\d+)?\s*(?:%|fold)|"
    r"\b(?:table|fig(?:ure)?|panel)\s+[A-Za-z0-9.-]+"
    r")",
    re.I,
)
RESULT_SECTION = re.compile(
    r"\b(?:results?|findings?|supplement|figure|table)\b", re.I
)


def full_api_view(packet: CompactEvidencePacket) -> CompactApiPacket:
    """Losslessly adapt the local packet to the candidate builder's API view."""
    prepared, sources, _, original_order = _prepare(packet)
    payload = _selection_payload(
        paper_id=packet.paper_id,
        blocked_fields=sorted(packet.blocked_fields),
        selected_ids=set(prepared),
        prepared=prepared,
        sources=sources,
        original_order=original_order,
    )
    checksum = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return CompactApiPacket.model_validate(
        {**payload, "packet_checksum": checksum}
    )


def _source_id(block: DocumentBlock) -> str:
    return "FC-S-" + hashlib.sha256(
        json.dumps(
            [
                block.block_id,
                block.source_path,
                block.page_number,
                block.xml_element_id,
                block.table_number,
                block.figure_number,
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]


def full_corpus_view(
    packet: CompactEvidencePacket,
    corpus_path: Path,
) -> CompactApiPacket:
    """Combine retrieval-selected evidence with every parsed full-text block."""
    base = full_api_view(packet)
    sources = {row.source_id: row for row in base.sources}
    evidence = {row.evidence_id: row for row in base.evidence}
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        block = DocumentBlock.model_validate_json(line)
        source_id = _source_id(block)
        section, _, subsection = block.section_path.partition(">")
        sources[source_id] = ApiSource(
            source_id=source_id,
            chunk_id=block.block_id,
            source_path=block.source_path,
            source_kind=block.source_kind,
            block_type=block.block_type,
            section=section.strip() or "Unknown",
            subsection=subsection.strip() or None,
            page_number=block.page_number,
            xml_element_id=block.xml_element_id,
            table_number=block.table_number,
            figure_number=block.figure_number,
            locator_id=block.locator_id,
        )
        normalized_block_text = block.text.translate(OCR_NORMALIZATION)
        clauses = sentence_clauses(normalized_block_text)
        for index, text in enumerate(clauses, 1):
            if (
                index > 1
                and re.fullmatch(
                    r"(?:Table|Fig(?:ure)?\.?)\s+[A-Za-z0-9.-]+\.",
                    clauses[index - 2],
                    re.I,
                )
            ):
                text = f"{clauses[index - 2]} {text}"
            digest = hashlib.sha256(
                f"{block.block_id}:{index}:{text}".encode()
            ).hexdigest()[:16]
            evidence_id = f"{packet.paper_id}-FC-{digest}"
            evidence.setdefault(
                evidence_id,
                ApiEvidence(
                    evidence_id=evidence_id,
                    locator_id=(
                        clause_locator(block.locator_id, index)
                        if block.locator_id
                        else None
                    ),
                    text=text,
                    retrieval_field_tags=(
                        ["outcomes"]
                        if (
                            CORPUS_OUTCOME_SIGNAL.search(text)
                            and (
                                RESULT_SECTION.search(block.section_path)
                                or block.block_type
                                in {
                                    "figure",
                                    "figure_caption",
                                    "caption",
                                    "table",
                                    "table_row",
                                    "pdf_page",
                                }
                            )
                        )
                        else ["full_corpus"]
                    ),
                    experiment_candidate_ids=[],
                    source_ids=[source_id],
                ),
            )
    unsigned = {
        "packet_version": base.packet_version,
        "paper_id": packet.paper_id,
        "blocked_fields": base.blocked_fields,
        "sources": [
            row.model_dump(mode="json", exclude_none=True)
            for row in sorted(sources.values(), key=lambda item: item.source_id)
        ],
        "evidence": [
            row.model_dump(mode="json", exclude_none=True)
            for row in sorted(evidence.values(), key=lambda item: item.evidence_id)
        ],
    }
    checksum = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return CompactApiPacket.model_validate(
        {**unsigned, "packet_checksum": checksum}
    )


def build_inventory(
    packet: CompactEvidencePacket,
    *,
    corpus_path: Path | None = None,
) -> OutcomeInventory:
    corpus_path = corpus_path or CORPUS_ROOT / f"{packet.paper_id}.blocks.jsonl"
    full_view = (
        full_corpus_view(packet, corpus_path)
        if corpus_path.exists()
        else full_api_view(packet)
    )
    candidates = build_candidates(full_view)
    return consolidate(
        paper_id=packet.paper_id,
        source_packet_checksum=packet.packet_checksum,
        candidates=candidates,
        packet=full_view,
    )


def load_and_build(path: Path) -> OutcomeInventory:
    packet = CompactEvidencePacket.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    return build_inventory(packet)
