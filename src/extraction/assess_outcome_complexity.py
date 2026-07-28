"""Cheaply classify evidence packets before candidate building or the first LLM call."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.extraction.build_outcome_candidates import (
    DIRECT_SIGNAL,
    ENDPOINT_PATTERNS,
    INTERVENTION_SIGNAL,
    OUTCOME_VALUE,
    QUANTITATIVE_BIOLOGICAL_RESULT,
    RESULT_SIGNAL,
    TARGET_CONTEXT_SIGNAL,
    VISUAL_REFERENCE,
    VISUAL_OUTCOME_SIGNAL,
)
from src.extraction.outcome_coverage_contracts import ComplexityAssessment
from src.rag.compact_api_packet import CompactApiPacket


FULL_CORPUS_TAG = "full_corpus"
COMPACT_VIEW = "compact"
FULL_CORPUS_VIEW = "full_corpus"
FULL_VIEW_VOLUME_SCALE = 3
FULL_CORPUS_ONLY_WEIGHT = 0.5


@dataclass(frozen=True)
class ComplexityThresholds:
    """Counting rules for one evidence view.

    The full evidence view carries every locally parsed block, so raw passage
    counts are several times larger than in a budgeted compact packet even when
    the paper reports the same handful of outcomes.  Volume-based thresholds are
    therefore scaled per view, and passages that exist only because of the
    corpus expansion (tagged solely ``full_corpus``) count at a reduced weight.
    Diversity thresholds (endpoint families, distinct visual locations) are
    bounded by the paper itself and are not scaled.
    """

    view: str
    many_outcome_groups: int = 8
    some_outcome_groups: int = 4
    many_endpoint_families: int = 4
    some_endpoint_families: int = 2
    multiple_visual_locations: int = 2
    many_contextual_passages: int = 4
    many_contextual_families: int = 2
    many_outcome_tagged_passages: int = 60
    full_corpus_only_weight: float = 1.0


COMPACT_THRESHOLDS = ComplexityThresholds(view=COMPACT_VIEW)
FULL_CORPUS_THRESHOLDS = ComplexityThresholds(
    view=FULL_CORPUS_VIEW,
    many_outcome_groups=COMPACT_THRESHOLDS.many_outcome_groups
    * FULL_VIEW_VOLUME_SCALE,
    some_outcome_groups=COMPACT_THRESHOLDS.some_outcome_groups
    * FULL_VIEW_VOLUME_SCALE,
    many_contextual_passages=COMPACT_THRESHOLDS.many_contextual_passages
    * FULL_VIEW_VOLUME_SCALE,
    many_outcome_tagged_passages=COMPACT_THRESHOLDS.many_outcome_tagged_passages
    * FULL_VIEW_VOLUME_SCALE,
    full_corpus_only_weight=FULL_CORPUS_ONLY_WEIGHT,
)
THRESHOLDS_BY_VIEW = {
    COMPACT_VIEW: COMPACT_THRESHOLDS,
    FULL_CORPUS_VIEW: FULL_CORPUS_THRESHOLDS,
}


def detect_evidence_view(packet: CompactApiPacket) -> str:
    """Return the view a packet was built for.

    Only ``build_full_outcome_inventory.full_corpus_view`` emits the
    ``full_corpus`` retrieval tag, so its presence identifies the full view
    without needing a new packet field.
    """
    if any(
        FULL_CORPUS_TAG in evidence.retrieval_field_tags
        for evidence in packet.evidence
    ):
        return FULL_CORPUS_VIEW
    return COMPACT_VIEW


def _passage_weight(evidence, thresholds: ComplexityThresholds) -> float:
    if set(evidence.retrieval_field_tags) == {FULL_CORPUS_TAG}:
        return thresholds.full_corpus_only_weight
    return 1.0


def assess(
    packet: CompactApiPacket,
    *,
    evidence_view: str | None = None,
    thresholds: ComplexityThresholds | None = None,
) -> ComplexityAssessment:
    """Use a fast signal census; never construct detailed outcome candidates."""
    if thresholds is None:
        view = evidence_view or detect_evidence_view(packet)
        if view not in THRESHOLDS_BY_VIEW:
            raise ValueError(
                f"Unknown evidence view {view!r}; "
                f"expected one of {sorted(THRESHOLDS_BY_VIEW)}"
            )
        thresholds = THRESHOLDS_BY_VIEW[view]
    sources = {
        row.source_id: row.model_dump(mode="json", exclude_none=True)
        for row in packet.sources
    }
    signal_passages = []
    weighted_signal_passages = 0.0
    families: set[str] = set()
    contextual_passages = 0.0
    contextual_families: set[str] = set()
    visual_locations: set[str] = set()
    outcome_tagged_passages = sum(
        "outcomes" in evidence.retrieval_field_tags for evidence in packet.evidence
    )
    for evidence in packet.evidence:
        outcome_tagged = "outcomes" in evidence.retrieval_field_tags
        tag_robust_outcome = bool(
            QUANTITATIVE_BIOLOGICAL_RESULT.search(evidence.text)
            or (
                RESULT_SIGNAL.search(evidence.text)
                and INTERVENTION_SIGNAL.search(evidence.text)
                and TARGET_CONTEXT_SIGNAL.search(evidence.text)
            )
            or (
                VISUAL_REFERENCE.search(evidence.text)
                and VISUAL_OUTCOME_SIGNAL.search(evidence.text)
            )
        )
        if not outcome_tagged and not tag_robust_outcome:
            continue
        if not DIRECT_SIGNAL.search(evidence.text):
            continue
        passage_families = {
            name
            for name, pattern in ENDPOINT_PATTERNS.items()
            if re.search(pattern, evidence.text, re.I)
        }
        if not passage_families:
            continue
        weight = _passage_weight(evidence, thresholds)
        if INTERVENTION_SIGNAL.search(evidence.text):
            contextual_passages += weight
            contextual_families |= passage_families
        text_visual = VISUAL_REFERENCE.search(evidence.text)
        strong = bool(
            OUTCOME_VALUE.search(evidence.text)
            or (
                RESULT_SIGNAL.search(evidence.text)
                and INTERVENTION_SIGNAL.search(evidence.text)
            )
            or (
                text_visual
                and VISUAL_OUTCOME_SIGNAL.search(evidence.text)
                and (
                    INTERVENTION_SIGNAL.search(evidence.text)
                    or "gene_editing" in passage_families
                )
            )
        )
        if not strong:
            continue
        signal_passages.append(evidence)
        weighted_signal_passages += weight
        families |= passage_families
        if text_visual:
            visual_locations.add(text_visual.group(1).lower())
        for source_id in evidence.source_ids:
            source = sources.get(source_id, {})
            location = (
                source.get("table_number")
                or source.get("figure_number")
                or source.get("supplement_identifier")
            )
            if location:
                visual_locations.add(str(location).lower())
    reasons: list[str] = []
    score = 0
    if weighted_signal_passages >= thresholds.many_outcome_groups:
        score += 2
        reasons.append("eight_or_more_direct_outcome_groups")
    elif weighted_signal_passages >= thresholds.some_outcome_groups:
        score += 1
        reasons.append("four_to_seven_direct_outcome_groups")
    if len(families) >= thresholds.many_endpoint_families:
        score += 2
        reasons.append("four_or_more_endpoint_families")
    elif len(families) >= thresholds.some_endpoint_families:
        score += 1
        reasons.append("multiple_endpoint_families")
    if visual_locations:
        score += 2
        reasons.append("visual_outcome_evidence")
    if len(visual_locations) >= thresholds.multiple_visual_locations:
        score += 1
        reasons.append("multiple_visual_outcome_locations")
    if contextual_passages >= thresholds.many_contextual_passages:
        score += 2
        reasons.append("four_or_more_intervention_endpoint_passages")
    if len(contextual_families) >= thresholds.many_contextual_families:
        score += 2
        reasons.append("multiple_intervention_endpoint_families")
    if outcome_tagged_passages >= thresholds.many_outcome_tagged_passages:
        score += 2
        reasons.append("sixty_or_more_outcome_tagged_passages")
    if thresholds.view != COMPACT_VIEW:
        reasons.append(f"{thresholds.view}_view_thresholds")
    route = "complex" if score >= 4 else "simple"
    return ComplexityAssessment(
        assessment_version="outcome-complexity-1.1.0",
        paper_id=packet.paper_id,
        complexity_score=score,
        route=route,
        reasons=reasons,
        candidate_groups=len(signal_passages),
        endpoint_families=len(families),
        visual_candidate_groups=len(visual_locations),
    )
