"""Budget and serialize compact evidence packets for one-call extraction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from pydantic import Field

from src.config_flags import is_enabled
from src.extraction.compact_prompt_v1 import COMPACT_EXTRACTION_PROMPT
from src.idempotent_write import IDEMPOTENT_UPSERT_FLAG, upsert_json

from .compact_packet import (
    OUTPUT_ROOT as REVIEW_PACKET_ROOT,
    RETRIEVAL_ROOT,
    ROOT,
    CompactEvidence,
    CompactEvidencePacket,
    SourceLocation,
    normalize_text,
)
from .models import StrictModel


OUTPUT_ROOT = ROOT / "data" / "staging" / "rag" / "compact_api_packets_v1"
SCHEMA_PATH = (
    ROOT
    / "docs"
    / "extraction"
    / "schemas"
    / "compact_v1"
    / "compact_extraction_response.schema.json"
)
GOLD_EVIDENCE_PATH = ROOT / "data" / "annotations" / "gold_v1" / "evidence.csv"
API_PACKET_VERSION = "compact-api-packet-1.0.0"
DEFAULT_PACKET_BUDGET_TOKENS = 16_000
BUDGET_STATUS = "human_approved"
TOKEN_ESTIMATION_METHOD = "ceil(compact UTF-8 JSON bytes / 4)"

DIRECT_EVIDENCE = re.compile(
    r"(?:"
    r"\bmol(?:ar)?\s*%|\bmolar ratio\b|\bN/P\b|"
    r"\b(?:LNP|formulation)[-\s]?[A-Za-z0-9]+\b|"
    r"\b(?:ionizable|cationic|helper|PEG(?:ylated)?)[-\s]lipid\b|"
    r"\b(?:mRNA|siRNA|sgRNA|RNA|DNA|CRISPR|Cas9)\b|"
    r"\b(?:administered|injected|treated|transfected|delivered|encapsulated)\b|"
    r"\b(?:mg/kg|micrograms?|µg|hours?|days?|weeks?)\b|"
    r"\b(?:increased|decreased|reduced|knockdown|expression|editing|uptake)\b|"
    r"(?:\d+(?:\.\d+)?\s*(?:%|nm|mV))"
    r")",
    re.I,
)
BACKGROUND_SECTIONS = re.compile(
    r"\b(?:introduction|background|discussion|review|references?)\b",
    re.I,
)
DIRECT_SECTIONS = re.compile(
    r"\b(?:methods?|materials?|results?|experimental|supplement)\b",
    re.I,
)
QUANTITATIVE_OUTCOME = re.compile(
    r"(?:"
    r"\b(?:over|under|more than|fewer than|less than|greater than)?\s*"
    r"\d+(?:\.\d+)?\s*%|"
    r"\d+(?:\.\d+)?\s*(?:±|\+/-)\s*\d+(?:\.\d+)?|"
    r"\d+(?:\.\d+)?\s*[- ]?fold\b"
    r")",
    re.I,
)
BIOLOGICAL_OUTCOME_SIGNAL = re.compile(
    r"\b(?:express(?:ed|ion)?|transfect(?:ed|ion)?|deliver(?:ed|y)?|uptake|"
    r"internaliz(?:ed|ation)?|edit(?:ed|ing)?|insertion|deletion|indel|"
    r"knockdown|silenc(?:ed|ing)?|activity|efficacy|survival|fibrosis|"
    r"steatosis|necrosis|phagocyt(?:osed|osis)?|eliminat(?:ed|ion)?)\b",
    re.I,
)


class ApiSource(StrictModel):
    source_id: str
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
    # Design-document section 9 locator, carried alongside source_id. Excluded
    # from the source_id digest below, so a packet rebuilt with locators on has
    # exactly the S- ids it had before.
    locator_id: str | None = None


class ApiEvidence(StrictModel):
    evidence_id: str
    # The legible counterpart of evidence_id. Additive: evidence_id stays the
    # id the extraction prompt asks the model to cite and the validator checks
    # against, so no committed result stops resolving.
    locator_id: str | None = None
    text: str
    retrieval_field_tags: list[str]
    experiment_candidate_ids: list[str] = Field(default_factory=list)
    source_ids: list[str]
    context_before_evidence_id: str | None = None
    context_after_evidence_id: str | None = None


class CompactApiPacket(StrictModel):
    packet_version: str = API_PACKET_VERSION
    paper_id: str
    blocked_fields: list[str]
    sources: list[ApiSource]
    evidence: list[ApiEvidence]
    packet_checksum: str


def estimate_tokens(value: Any) -> int:
    """Return a deterministic tokenizer-free estimate suitable for budgeting."""
    if not isinstance(value, str):
        value = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return math.ceil(len(value.encode("utf-8")) / 4)


# Fields that have ever contributed to an S- id. Frozen: the S- ids in every
# committed packet were computed over exactly this set, and results cite them.
# Anything added to SourceLocation later is excluded by omission rather than by
# a rule someone has to remember, which is why this is a literal tuple and not
# `set(SourceLocation.model_fields) - {...}`.
_SOURCE_ID_FIELDS = (
    "chunk_id",
    "source_path",
    "source_kind",
    "block_type",
    "section",
    "subsection",
    "page_number",
    "xml_element_id",
    "table_number",
    "figure_number",
)


def _source_id(location: SourceLocation) -> str:
    """The frozen content hash. Unchanged, including when a locator is present.

    Previously this dumped the whole model, so adding any field to
    SourceLocation would have silently re-keyed every source in every packet.
    Projecting onto the frozen field list first makes that impossible; the
    digest for a location without a locator is byte-for-byte what it always
    was, which `test_source_id_digest_is_frozen` pins.
    """
    canonical = location.model_dump_json(
        exclude_none=True,
        include=set(_SOURCE_ID_FIELDS),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"S-{digest}"


def _api_source(location: SourceLocation) -> ApiSource:
    return ApiSource(
        source_id=_source_id(location),
        **location.model_dump(mode="json"),
    )


def evidence_priority(evidence: CompactEvidence) -> tuple[int, list[str]]:
    """Score direct scientific evidence above background discussion."""
    score = 0
    reasons: list[str] = []
    text = evidence.text
    sections = " ".join(
        filter(
            None,
            (
                location.section
                for location in evidence.source_locations
            ),
        )
    )
    subsections = " ".join(
        filter(
            None,
            (
                location.subsection
                for location in evidence.source_locations
            ),
        )
    )
    section_text = f"{sections} {subsections}"

    if DIRECT_EVIDENCE.search(text):
        score += 60
        reasons.append("direct_scientific_signal")
    if QUANTITATIVE_OUTCOME.search(text) and (
        "outcome" in evidence.field_groups
        or BIOLOGICAL_OUTCOME_SIGNAL.search(text)
    ):
        score += 120
        reasons.append("direct_quantitative_outcome")
    if evidence.experiment_candidate_ids:
        score += 30
        reasons.append("experiment_anchor")
    if any(
        group in {"formulation", "experiment_boundary", "outcome"}
        for group in evidence.field_groups
    ):
        score += 20
        reasons.append("priority_retrieval_group")
    if any(
        location.block_type in {"table", "table_row", "caption", "figure_caption"}
        for location in evidence.source_locations
    ):
        score += 15
        reasons.append("structured_or_caption_evidence")
    if DIRECT_SECTIONS.search(section_text):
        score += 10
        reasons.append("methods_or_results")
    if BACKGROUND_SECTIONS.search(section_text) and not DIRECT_EVIDENCE.search(text):
        score -= 25
        reasons.append("background_without_direct_signal")
    return score, reasons or ["retrieval_candidate"]


def _prepare(
    packet: CompactEvidencePacket,
) -> tuple[
    dict[str, ApiEvidence],
    dict[str, ApiSource],
    dict[str, tuple[int, list[str]]],
    dict[str, int],
]:
    evidence_id_by_text = {
        normalize_text(evidence.text): evidence.evidence_id
        for evidence in packet.evidence
    }
    sources: dict[str, ApiSource] = {}
    prepared: dict[str, ApiEvidence] = {}
    priorities: dict[str, tuple[int, list[str]]] = {}
    original_order: dict[str, int] = {}

    for index, evidence in enumerate(packet.evidence):
        source_ids = []
        for location in evidence.source_locations:
            source = _api_source(location)
            sources[source.source_id] = source
            source_ids.append(source.source_id)
        prepared[evidence.evidence_id] = ApiEvidence(
            evidence_id=evidence.evidence_id,
            locator_id=evidence.locator_id,
            text=evidence.text,
            retrieval_field_tags=evidence.field_tags,
            experiment_candidate_ids=evidence.experiment_candidate_ids,
            source_ids=sorted(set(source_ids)),
            context_before_evidence_id=evidence_id_by_text.get(
                normalize_text(evidence.context_before)
            )
            if evidence.context_before
            else None,
            context_after_evidence_id=evidence_id_by_text.get(
                normalize_text(evidence.context_after)
            )
            if evidence.context_after
            else None,
        )
        priorities[evidence.evidence_id] = evidence_priority(evidence)
        original_order[evidence.evidence_id] = index
    return prepared, sources, priorities, original_order


def _selection_payload(
    *,
    paper_id: str,
    blocked_fields: list[str],
    selected_ids: set[str],
    prepared: dict[str, ApiEvidence],
    sources: dict[str, ApiSource],
    original_order: dict[str, int],
    include_checksum_reserve: bool = False,
) -> dict[str, Any]:
    ordered_ids = sorted(selected_ids, key=original_order.__getitem__)
    selected_sources = {
        source_id
        for evidence_id in ordered_ids
        for source_id in prepared[evidence_id].source_ids
    }
    evidence_rows = []
    for evidence_id in ordered_ids:
        row = prepared[evidence_id].model_dump(mode="json", exclude_none=True)
        for key in (
            "context_before_evidence_id",
            "context_after_evidence_id",
        ):
            if row.get(key) not in selected_ids:
                row.pop(key, None)
        evidence_rows.append(row)
    payload = {
        "packet_version": API_PACKET_VERSION,
        "paper_id": paper_id,
        "blocked_fields": blocked_fields,
        "sources": [
            sources[source_id].model_dump(mode="json", exclude_none=True)
            for source_id in sorted(selected_sources)
        ],
        "evidence": evidence_rows,
    }
    if include_checksum_reserve:
        payload["packet_checksum"] = "0" * 64
    return payload


def build_api_packet(
    packet: CompactEvidencePacket,
    *,
    packet_budget_tokens: int = DEFAULT_PACKET_BUDGET_TOKENS,
) -> tuple[CompactApiPacket, dict[str, Any]]:
    if packet_budget_tokens <= 0:
        raise ValueError("packet_budget_tokens must be positive")

    prepared, sources, priorities, original_order = _prepare(packet)
    blocked_fields = sorted(packet.blocked_fields)
    ranked_ids = sorted(
        prepared,
        key=lambda evidence_id: (
            -priorities[evidence_id][0],
            original_order[evidence_id],
        ),
    )
    selected_ids: set[str] = set()
    for evidence_id in ranked_ids:
        evidence = prepared[evidence_id]
        bundle = {evidence_id}
        for context_id in (
            evidence.context_before_evidence_id,
            evidence.context_after_evidence_id,
        ):
            if context_id:
                bundle.add(context_id)
        proposed = selected_ids | bundle
        payload = _selection_payload(
            paper_id=packet.paper_id,
            blocked_fields=blocked_fields,
            selected_ids=proposed,
            prepared=prepared,
            sources=sources,
            original_order=original_order,
            include_checksum_reserve=True,
        )
        if estimate_tokens(payload) <= packet_budget_tokens:
            selected_ids = proposed

    payload = _selection_payload(
        paper_id=packet.paper_id,
        blocked_fields=blocked_fields,
        selected_ids=selected_ids,
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
        ).encode("utf-8")
    ).hexdigest()
    api_packet = CompactApiPacket.model_validate(
        {**payload, "packet_checksum": checksum}
    )

    selected_rows = []
    excluded_rows = []
    for evidence_id in sorted(prepared, key=original_order.__getitem__):
        evidence = prepared[evidence_id]
        score, reasons = priorities[evidence_id]
        row = {
            "evidence_id": evidence_id,
            "may_support_fields": evidence.retrieval_field_tags,
            "priority_score": score,
            "priority_reasons": reasons,
            "estimated_item_tokens": estimate_tokens(
                evidence.model_dump(mode="json", exclude_none=True)
            ),
            "source_ids": evidence.source_ids,
            "text": evidence.text,
        }
        if evidence_id in selected_ids:
            selected_rows.append(row)
        else:
            excluded_rows.append(
                {
                    **row,
                    "exclusion_reason": "evidence_budget_exceeded",
                }
            )

    packet_tokens = estimate_tokens(
        api_packet.model_dump(mode="json", exclude_none=True)
    )
    schema_tokens = estimate_tokens(
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    )
    prompt_tokens = estimate_tokens(COMPACT_EXTRACTION_PROMPT)
    manifest = {
        "paper_id": packet.paper_id,
        "source_review_packet_checksum": packet.packet_checksum,
        "api_packet_checksum": api_packet.packet_checksum,
        "packet_budget_tokens": packet_budget_tokens,
        "budget_status": BUDGET_STATUS,
        "token_estimation_method": TOKEN_ESTIMATION_METHOD,
        "estimated_input_tokens": {
            "prompt": prompt_tokens,
            "response_schema": schema_tokens,
            "evidence_packet": packet_tokens,
            "total": prompt_tokens + schema_tokens + packet_tokens,
        },
        "counts": {
            "candidate_evidence": len(prepared),
            "selected_evidence": len(selected_rows),
            "excluded_evidence": len(excluded_rows),
            "selected_sources": len(api_packet.sources),
        },
        "removed_duplicates": packet.deduplication.model_dump(mode="json"),
        "blocked_fields_with_diagnostics": packet.blocked_fields,
        "selected_passages": selected_rows,
        "excluded_passages": excluded_rows,
    }
    return api_packet, manifest


def _same_gold_location(
    row: dict[str, str],
    location: dict[str, Any],
) -> bool:
    gold_path = row["xml_file"]
    gold_pmcid = re.search(r"PMC\d+", gold_path)
    source_path = location.get("source_path") or ""
    source_pmcid = re.search(r"PMC\d+", source_path)
    same_source = bool(gold_path) and (
        source_path == gold_path
        or Path(source_path).name == Path(gold_path).name
        or (
            gold_pmcid
            and source_pmcid
            and gold_pmcid.group() == source_pmcid.group()
            and Path(gold_path).suffix.lower() in {".xml", ".nxml"}
            and Path(source_path).suffix.lower() in {".xml", ".nxml"}
        )
    )
    xml_id = row["xml_element_id"].strip()
    page = row["page_number"].strip()
    exact_xml = bool(xml_id and location.get("xml_element_id") == xml_id)
    exact_page = bool(page and location.get("page_number") == int(page))
    return bool(
        exact_xml
        or (same_source and (exact_page or (not xml_id and not page)))
    )


def frozen_gold_preservation_report(
    *,
    review_packet_root: Path = REVIEW_PACKET_ROOT,
    retrieval_root: Path = RETRIEVAL_ROOT,
    api_packet_root: Path | None = None,
    gold_evidence_path: Path = GOLD_EVIDENCE_PATH,
) -> dict[str, Any]:
    rows = list(
        csv.DictReader(gold_evidence_path.open(encoding="utf-8", newline=""))
    )
    results = []
    for row in rows:
        paper_id = row["gold_paper_id"]
        retrieval_path = retrieval_root / f"{paper_id}.json"
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
        before_locations = [
            hit
            for field_packet in retrieval["packets"].values()
            for hit in field_packet["hits"]
        ]

        review_path = review_packet_root / f"{paper_id}.json"
        review_packet = json.loads(review_path.read_text(encoding="utf-8"))
        after_locations = [
            location
            for evidence in review_packet["evidence"]
            for location in evidence["source_locations"]
        ]
        before = any(
            _same_gold_location(row, location)
            for location in before_locations
        )
        after = any(
            _same_gold_location(row, location)
            for location in after_locations
        )
        selected = None
        if api_packet_root is not None:
            api_path = api_packet_root / f"{paper_id}.json"
            api_packet = json.loads(api_path.read_text(encoding="utf-8"))
            selected = any(
                _same_gold_location(row, location)
                for location in api_packet["sources"]
            )
        results.append(
            {
                "evidence_id": row["evidence_id"],
                "paper_id": paper_id,
                "available_before_deduplication": before,
                "available_after_deduplication": after,
                "available_in_budgeted_packet": selected,
                "lost_during_deduplication": before and not after,
                "lost_to_evidence_budget": bool(
                    after and selected is False
                ),
            }
        )
    available_before = sum(
        row["available_before_deduplication"]
        for row in results
    )
    available_after = sum(
        row["available_after_deduplication"]
        for row in results
    )
    selected_count = (
        sum(bool(row["available_in_budgeted_packet"]) for row in results)
        if api_packet_root is not None
        else None
    )
    return {
        "frozen_gold_locations": len(results),
        "available_before_deduplication": available_before,
        "available_after_deduplication": available_after,
        "unavailable_before_deduplication": len(results) - available_before,
        "lost_during_deduplication": sum(
            row["lost_during_deduplication"] for row in results
        ),
        "available_in_budgeted_packet": selected_count,
        "lost_to_evidence_budget": sum(
            row["lost_to_evidence_budget"] for row in results
        ),
        "results": results,
    }


def write_outputs(
    api_packet: CompactApiPacket,
    manifest: dict[str, Any],
    output_root: Path = OUTPUT_ROOT,
) -> tuple[Path, Path]:
    """Write the packet and its manifest, converging on an unchanged rebuild.

    Both files are pure functions of the review packet and the budget, so a
    rebuild that changes nothing must write nothing; a rebuild that changes
    something keeps the superseded content under ``<name>.versions/`` instead
    of overwriting it silently.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    packet_path = output_root / f"{api_packet.paper_id}.json"
    manifest_path = output_root / f"{api_packet.paper_id}.manifest.json"
    enabled = is_enabled(IDEMPOTENT_UPSERT_FLAG)
    upsert_json(
        packet_path,
        api_packet.model_dump(mode="json", exclude_none=True),
        on_conflict="version",
        enabled=enabled,
    )
    upsert_json(manifest_path, manifest, on_conflict="version", enabled=enabled)
    return packet_path, manifest_path


def build_all(
    *,
    review_packet_root: Path = REVIEW_PACKET_ROOT,
    output_root: Path = OUTPUT_ROOT,
    packet_budget_tokens: int = DEFAULT_PACKET_BUDGET_TOKENS,
) -> dict[str, Any]:
    papers = []
    for review_path in sorted(review_packet_root.glob("GP-*.json")):
        review_packet = CompactEvidencePacket.model_validate_json(
            review_path.read_text(encoding="utf-8")
        )
        api_packet, manifest = build_api_packet(
            review_packet,
            packet_budget_tokens=packet_budget_tokens,
        )
        packet_path, manifest_path = write_outputs(
            api_packet,
            manifest,
            output_root,
        )
        papers.append(
            {
                "paper_id": api_packet.paper_id,
                "packet_path": str(packet_path.relative_to(ROOT)),
                "manifest_path": str(manifest_path.relative_to(ROOT)),
                **manifest["counts"],
                "estimated_input_tokens": manifest["estimated_input_tokens"],
                "blocked_fields": api_packet.blocked_fields,
            }
        )

    gold_report = frozen_gold_preservation_report(
        review_packet_root=review_packet_root,
        api_packet_root=output_root,
    )
    run_manifest = {
        "packet_version": API_PACKET_VERSION,
        "packet_budget_tokens": packet_budget_tokens,
        "budget_status": BUDGET_STATUS,
        "token_estimation_method": TOKEN_ESTIMATION_METHOD,
        "papers": papers,
        "frozen_gold_preservation": gold_report,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Compact API packet v1 - approved evidence budget",
        "",
        f"- Packet evidence budget: {packet_budget_tokens:,} estimated tokens",
        f"- Estimation method: `{TOKEN_ESTIMATION_METHOD}`",
        (
            "- Frozen gold locations: "
            f"{gold_report['available_before_deduplication']}/"
            f"{gold_report['frozen_gold_locations']} available before dedup; "
            f"{gold_report['lost_during_deduplication']} lost during dedup; "
            f"{gold_report['lost_to_evidence_budget']} lost to budget"
        ),
        "",
        "| Paper | Candidates | Selected | Excluded | Sources | Estimated total input tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in papers:
        lines.append(
            f"| {row['paper_id']} | {row['candidate_evidence']} | "
            f"{row['selected_evidence']} | {row['excluded_evidence']} | "
            f"{row['selected_sources']} | "
            f"{row['estimated_input_tokens']['total']:,} |"
        )
    (output_root / "manifest.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return run_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet-budget-tokens",
        type=int,
        default=DEFAULT_PACKET_BUDGET_TOKENS,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            build_all(
                output_root=args.output_dir,
                packet_budget_tokens=args.packet_budget_tokens,
            ),
            indent=2,
        )
    )
