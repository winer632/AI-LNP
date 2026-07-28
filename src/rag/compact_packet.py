"""Build one deduplicated, provenance-preserving evidence packet per paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import Field

from src.config_flags import is_enabled
from src.idempotent_write import IDEMPOTENT_UPSERT_FLAG, upsert_json

from .evidence_locators import clause_locator
from .models import DocumentBlock, StrictModel


ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_ROOT = ROOT / "data" / "staging" / "rag" / "retrieval_packets"
CORPUS_ROOT = ROOT / "data" / "staging" / "rag" / "gold_v1"
BOUNDARY_ROOT = ROOT / "data" / "staging" / "extraction" / "g1_v3_frozen_boundaries"
OUTPUT_ROOT = ROOT / "data" / "staging" / "rag" / "compact_packets_v1"
PACKET_VERSION = "compact-packet-1.0.0"

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
ANAPHORIC_START = re.compile(
    r"^(?:this|these|those|it|they|such|both|the former|the latter|"
    r"however|therefore|thus|in contrast|respectively)\b",
    re.I,
)
EXPERIMENT_IDENTIFIER = re.compile(
    r"\b(?:"
    r"LNP[\s-]?[A-Z0-9][A-Z0-9-]*|"
    r"formulation\s+[A-Z0-9][A-Z0-9-]*|"
    r"(?:table|figure|fig\.?)\s+[A-Z0-9][A-Z0-9.-]*|"
    r"(?:group|cohort|experiment)\s+[A-Z0-9][A-Z0-9-]*"
    r")\b",
    re.I,
)
WORD = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
CONTEXT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "was",
    "were",
    "with",
}


class SourceLocation(StrictModel):
    chunk_id: str
    source_path: str
    source_kind: str
    block_type: str
    section: str
    subsection: str | None = None
    page_number: int | None = None
    xml_element_id: str | None = None
    table_number: str | None = None
    figure_number: str | None = None
    # The block's design-document section 9 locator, carried through unchanged.
    # `compact_api_packet._source_id` explicitly excludes it, so populating this
    # field cannot re-key an S- id that a committed packet already contains.
    locator_id: str | None = None


class CompactEvidence(StrictModel):
    evidence_id: str
    # The legible counterpart of evidence_id: the locator of the block this
    # clause came from plus the clause's ordinal, e.g.
    # "GP-006-mmc1-p01-tabS2-r2-s1". Additive -- evidence_id remains the key
    # every committed extraction result cites and every validator checks.
    locator_id: str | None = None
    clause_ids: list[str]
    chunk_ids: list[str]
    text: str
    normalized_text_sha256: str
    field_tags: list[str]
    field_groups: list[str]
    query_ids: list[str]
    entity_types: list[str]
    experiment_candidate_ids: list[str]
    source_locations: list[SourceLocation]
    context_before: str | None = None
    context_after: str | None = None


class DeduplicationSummary(StrictModel):
    input_hits: int
    unique_chunks: int
    removed_chunk_duplicates: int
    removed_normalized_passages: int
    input_clauses: int
    unique_evidence_items: int


class CompactEvidencePacket(StrictModel):
    packet_version: str = PACKET_VERSION
    paper_id: str
    source_retrieval_packet: str
    blocked_fields: dict[str, list[str]]
    evidence: list[CompactEvidence]
    deduplication: DeduplicationSummary
    packet_checksum: str


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


def display_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def split_section(section_path: str) -> tuple[str, str | None]:
    parts = [part.strip() for part in section_path.split(">") if part.strip()]
    if not parts:
        return "Unknown", None
    return parts[0], " > ".join(parts[1:]) or None


def sentence_clauses(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    return [value.strip() for value in SENTENCE_BOUNDARY.split(compact) if value.strip()]


def _experiment_identifiers(text: str) -> set[str]:
    return {
        normalize_text(match.group(0))
        for match in EXPERIMENT_IDENTIFIER.finditer(text)
    }


def _content_terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in WORD.findall(text)
        if len(token) >= 4 and token.casefold() not in CONTEXT_STOPWORDS
    }


def _content_bigrams(text: str) -> set[tuple[str, str]]:
    terms = [
        token.casefold()
        for token in WORD.findall(text)
        if len(token) >= 4 and token.casefold() not in CONTEXT_STOPWORDS
    ]
    return set(zip(terms, terms[1:]))


def sentences_are_contextually_related(
    first_sentence: str,
    second_sentence: str,
) -> bool:
    """Return whether adjacent sentences should travel together as evidence.

    The rule is deliberately conservative and deterministic. It keeps context
    for explicit anaphora/continuations, shared experiment identifiers, or
    unusually strong content-term overlap. It is not intended to decide
    whether the sentences describe the same experiment.
    """
    first = re.sub(r"\s+", " ", first_sentence).strip()
    second = re.sub(r"\s+", " ", second_sentence).strip()
    if not first or not second:
        return False
    if ANAPHORIC_START.search(second) or first.endswith((":", ";")):
        return True

    shared_identifiers = _experiment_identifiers(first) & _experiment_identifiers(
        second
    )
    if shared_identifiers:
        return True
    if _content_bigrams(first) & _content_bigrams(second):
        return True

    first_terms = _content_terms(first)
    second_terms = _content_terms(second)
    shared_terms = first_terms & second_terms
    smaller_term_count = min(len(first_terms), len(second_terms))
    return (
        len(shared_terms) >= 2
        and smaller_term_count > 0
        and len(shared_terms) / smaller_term_count >= 0.5
    )


def load_corpus_blocks(corpus_root: Path, paper_id: str) -> dict[str, DocumentBlock]:
    path = corpus_root / f"{paper_id}.blocks.jsonl"
    if not path.exists():
        return {}
    return {
        block.block_id: block
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
        for block in [DocumentBlock.model_validate_json(line)]
    }


def load_experiment_candidates(
    boundary_root: Path,
    paper_id: str,
) -> list[dict[str, str]]:
    path = boundary_root / f"{paper_id}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "experiment_id": row["experiment_id"],
            "anchor": normalize_text(row.get("experiment_anchor_quote") or ""),
        }
        for row in payload.get("experiments", [])
        if row.get("experiment_anchor_quote")
    ]


def source_location(block: DocumentBlock) -> SourceLocation:
    section, subsection = split_section(block.section_path)
    return SourceLocation(
        chunk_id=block.block_id,
        source_path=block.source_path,
        source_kind=block.source_kind,
        block_type=block.block_type,
        section=section,
        subsection=subsection,
        page_number=block.page_number,
        xml_element_id=block.xml_element_id,
        table_number=block.table_number,
        figure_number=block.figure_number,
        locator_id=block.locator_id,
    )


def block_from_hit(hit: dict[str, Any]) -> DocumentBlock:
    text = hit["text"]
    return DocumentBlock(
        block_id=hit["block_id"],
        paper_id=hit["paper_id"],
        source_path=hit["source_path"],
        source_kind=hit.get("source_kind") or "pmc_xml",
        section_path=hit["section_path"],
        block_type=hit.get("block_type") or "paragraph",
        text=text,
        page_number=hit.get("page_number"),
        xml_element_id=hit.get("xml_element_id"),
        table_number=hit.get("table_number"),
        figure_number=hit.get("figure_number"),
        locator_id=hit.get("locator_id"),
        char_start=0,
        char_end=len(text),
        parser="retrieval_packet_fallback",
        parser_confidence=0.5,
    )


def build_packet(
    retrieval_path: Path,
    *,
    corpus_root: Path = CORPUS_ROOT,
    boundary_root: Path = BOUNDARY_ROOT,
) -> CompactEvidencePacket:
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    paper_id = retrieval["paper_id"]
    corpus = load_corpus_blocks(corpus_root, paper_id)
    experiment_candidates = load_experiment_candidates(boundary_root, paper_id)

    input_hits = 0
    by_chunk: dict[str, dict[str, Any]] = {}
    for field_tag, field_packet in retrieval["packets"].items():
        field_group = field_packet["query"]["field_group"]
        query_id = field_packet["query"]["query_id"]
        for hit in field_packet["hits"]:
            input_hits += 1
            chunk_id = hit["block_id"]
            if chunk_id not in by_chunk:
                by_chunk[chunk_id] = {
                    "block": corpus.get(chunk_id) or block_from_hit(hit),
                    "field_tags": set(),
                    "field_groups": set(),
                    "query_ids": set(),
                    "entity_types": set(),
                    "chunk_ids": set(),
                    "locations": {},
                }
            row = by_chunk[chunk_id]
            row["field_tags"].add(field_tag)
            row["field_groups"].add(field_group)
            row["query_ids"].add(query_id)
            row["entity_types"].update(hit.get("entity_types", []))
            row["chunk_ids"].add(chunk_id)
            location = source_location(row["block"])
            row["locations"][location.model_dump_json()] = location

    by_passage_text: dict[str, dict[str, Any]] = {}
    for row in by_chunk.values():
        normalized = normalize_text(row["block"].text)
        if normalized not in by_passage_text:
            by_passage_text[normalized] = row
            continue
        existing = by_passage_text[normalized]
        for key in (
            "field_tags",
            "field_groups",
            "query_ids",
            "entity_types",
            "chunk_ids",
        ):
            existing[key].update(row[key])
        existing["locations"].update(row["locations"])

    clause_rows: dict[str, dict[str, Any]] = {}
    input_clauses = 0
    for passage in by_passage_text.values():
        block = passage["block"]
        clauses = sentence_clauses(block.text)
        for index, clause in enumerate(clauses, 1):
            input_clauses += 1
            normalized_clause = normalize_text(clause)
            candidate_ids = {
                row["experiment_id"]
                for row in experiment_candidates
                if row["anchor"] and row["anchor"] in normalized_clause
            }
            clause_id = f"{block.block_id}:C{index:03d}"
            if normalized_clause not in clause_rows:
                clause_rows[normalized_clause] = {
                    "text": clause,
                    "clause_ids": set(),
                    "chunk_ids": set(),
                    "field_tags": set(),
                    "field_groups": set(),
                    "query_ids": set(),
                    "entity_types": set(),
                    "experiment_candidate_ids": set(),
                    "locations": {},
                    # clause_id -> legible locator, so the evidence's locator is
                    # picked by the same rule that picks its first clause_id
                    # rather than by whichever block happened to be seen first.
                    "locators": {},
                    "context_before": clauses[index - 2]
                    if index > 1
                    and sentences_are_contextually_related(
                        clauses[index - 2],
                        clause,
                    )
                    else None,
                    "context_after": clauses[index]
                    if index < len(clauses)
                    and sentences_are_contextually_related(
                        clause,
                        clauses[index],
                    )
                    else None,
                }
            evidence = clause_rows[normalized_clause]
            evidence["clause_ids"].add(clause_id)
            if block.locator_id:
                evidence["locators"][clause_id] = clause_locator(
                    block.locator_id, index
                )
            evidence["chunk_ids"].update(passage["chunk_ids"])
            evidence["field_tags"].update(passage["field_tags"])
            evidence["field_groups"].update(passage["field_groups"])
            evidence["query_ids"].update(passage["query_ids"])
            evidence["entity_types"].update(passage["entity_types"])
            evidence["experiment_candidate_ids"].update(candidate_ids)
            evidence["locations"].update(passage["locations"])

    evidence_items = []
    for normalized_clause, row in clause_rows.items():
        text_hash = hashlib.sha256(normalized_clause.encode("utf-8")).hexdigest()
        clause_ids = sorted(row["clause_ids"])
        evidence_items.append(
            CompactEvidence(
                evidence_id=f"{paper_id}-E-{text_hash[:16]}",
                locator_id=row["locators"].get(clause_ids[0]),
                clause_ids=clause_ids,
                chunk_ids=sorted(row["chunk_ids"]),
                text=row["text"],
                normalized_text_sha256=text_hash,
                field_tags=sorted(row["field_tags"]),
                field_groups=sorted(row["field_groups"]),
                query_ids=sorted(row["query_ids"]),
                entity_types=sorted(row["entity_types"]),
                experiment_candidate_ids=sorted(row["experiment_candidate_ids"]),
                source_locations=sorted(
                    row["locations"].values(),
                    key=lambda location: (
                        location.source_path,
                        location.page_number or 0,
                        location.chunk_id,
                    ),
                ),
                context_before=row["context_before"],
                context_after=row["context_after"],
            )
        )
    evidence_items.sort(key=lambda item: (item.source_locations[0].source_path, item.clause_ids[0]))

    summary = DeduplicationSummary(
        input_hits=input_hits,
        unique_chunks=len(by_chunk),
        removed_chunk_duplicates=input_hits - len(by_chunk),
        removed_normalized_passages=len(by_chunk) - len(by_passage_text),
        input_clauses=input_clauses,
        unique_evidence_items=len(evidence_items),
    )
    checksum_payload = {
        "packet_version": PACKET_VERSION,
        "paper_id": paper_id,
        "evidence": [
            item.model_dump(mode="json", exclude_none=True)
            for item in evidence_items
        ],
    }
    checksum = hashlib.sha256(
        json.dumps(
            checksum_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return CompactEvidencePacket(
        paper_id=paper_id,
        source_retrieval_packet=display_path(retrieval_path),
        blocked_fields=retrieval.get("blocked_fields", {}),
        evidence=evidence_items,
        deduplication=summary,
        packet_checksum=checksum,
    )


def write_packet(packet: CompactEvidencePacket, output_root: Path = OUTPUT_ROOT) -> Path:
    """Write one review packet, converging on a rerun that changes nothing.

    A packet is a pure function of its retrieval packet and corpus, so the
    normal rerun produces identical bytes and must leave the file -- and
    therefore ``git status`` -- alone. When it does differ the inputs or the
    code moved, which is a real event: ``version`` keeps the previous content
    under ``<name>.versions/`` and records it, rather than the previous
    behaviour of overwriting with no trace. Section 9 of the design document
    exists because that trace was missing: 6 of 9 committed packets no longer
    matched what the code produced and nothing said when they diverged.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{packet.paper_id}.json"
    upsert_json(
        path,
        packet.model_dump(mode="json", exclude_none=True),
        on_conflict="version",
        enabled=is_enabled(IDEMPOTENT_UPSERT_FLAG),
    )
    return path


def build_all(
    retrieval_root: Path = RETRIEVAL_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    rows = []
    for retrieval_path in sorted(retrieval_root.glob("GP-*.json")):
        packet = build_packet(retrieval_path)
        output_path = write_packet(packet, output_root)
        rows.append(
            {
                "paper_id": packet.paper_id,
                "output_path": str(output_path.relative_to(ROOT)),
                "evidence_items": len(packet.evidence),
                "packet_checksum": packet.packet_checksum,
                "blocked_fields": sorted(packet.blocked_fields),
            }
        )
    manifest = {
        "packet_version": PACKET_VERSION,
        "papers": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-id", action="append")
    args = parser.parse_args()
    if args.paper_id:
        results = []
        for paper_id in args.paper_id:
            packet = build_packet(RETRIEVAL_ROOT / f"{paper_id}.json")
            output = write_packet(packet)
            results.append(
                {
                    "paper_id": paper_id,
                    "output_path": str(output.relative_to(ROOT)),
                    "evidence_items": len(packet.evidence),
                    "packet_checksum": packet.packet_checksum,
                }
            )
        print(json.dumps({"packet_version": PACKET_VERSION, "papers": results}, indent=2))
    else:
        print(json.dumps(build_all(), indent=2))
