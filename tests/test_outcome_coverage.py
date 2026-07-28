import hashlib
import json
from unittest.mock import patch

from src.config_flags import override
from src.extraction.assess_outcome_complexity import assess
from src.extraction.build_outcome_candidates import build_candidates
from src.extraction.check_outcome_coverage import check
from src.rag.compact_api_packet import ApiEvidence, ApiSource, CompactApiPacket


def _packet():
    sources = [
        ApiSource(
            source_id="S-TEXT",
            chunk_id="C1",
            source_path="paper.xml",
            source_kind="pmc_xml",
            block_type="paragraph",
            section="Results",
        ),
        ApiSource(
            source_id="S-TABLE",
            chunk_id="C2",
            source_path="supp.pdf",
            source_kind="pdf",
            block_type="table",
            section="Results",
            table_number="Table S2",
        ),
    ]
    evidence = [
        ApiEvidence(
            evidence_id="E-FVIII",
            text="FVIII activity reached 3.30 ± 0.68% and was maintained for 26 weeks.",
            retrieval_field_tags=["outcomes"],
            source_ids=["S-TEXT"],
        ),
        ApiEvidence(
            evidence_id="E-INSERT",
            text="Total insertion frequency is shown in Table S2.",
            retrieval_field_tags=["outcomes"],
            source_ids=["S-TABLE"],
        ),
    ]
    unsigned = {
        "packet_version": "compact-api-packet-1.0.0",
        "paper_id": "GP-X",
        "blocked_fields": [],
        "sources": [row.model_dump(mode="json", exclude_none=True) for row in sources],
        "evidence": [row.model_dump(mode="json", exclude_none=True) for row in evidence],
    }
    canonical = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return CompactApiPacket.model_validate(
        {
            **unsigned,
            "packet_checksum": hashlib.sha256(canonical.encode()).hexdigest(),
        }
    )


def _result():
    missing = {
        "value": None,
        "status": "missing",
        "evidence_ids": [],
        "missing_reason": "Not reported.",
    }
    return {
        "eligibility": {"decision": "eligible"},
        "outcomes": [
            {
                "outcome_id": "O1",
                "endpoint": {
                    "value": "FVIII activity",
                    "status": "reported",
                    "evidence_ids": ["E-FVIII"],
                    "missing_reason": None,
                },
                "assay": missing,
                "qualitative_outcome": missing,
                "outcome_value": {
                    "value": 3.3,
                    "status": "reported",
                    "evidence_ids": ["E-FVIII"],
                    "missing_reason": None,
                },
                "outcome_unit": {
                    "value": "percent",
                    "status": "reported",
                    "evidence_ids": ["E-FVIII"],
                    "missing_reason": None,
                },
            }
        ],
    }


def test_candidates_group_text_and_visual_outcomes():
    candidates = build_candidates(_packet())
    assert {row.endpoint_family for row in candidates} == {
        "gene_editing",
        "therapeutic_function",
    }
    assert {row.route_hint for row in candidates} == {"text", "vision"}


def test_coverage_matches_text_and_routes_unmatched_table_to_vision():
    report = check(_packet(), _result())
    assert len(report.matched_candidates) == 1
    assert len(report.unmatched_candidates) == 1
    assert report.unmatched_candidates[0].route_hint == "vision"


def test_complexity_is_local_and_deterministic():
    first = assess(_packet())
    second = assess(_packet())
    assert first == second
    assert first.paper_id == "GP-X"


def test_complexity_does_not_build_detailed_candidates():
    with patch(
        "src.extraction.build_outcome_candidates.build_candidates",
        side_effect=AssertionError("detailed builder must not run"),
    ):
        assessment = assess(_packet())
    assert assessment.assessment_version == "outcome-complexity-1.1.0"


def test_prebuilt_candidates_are_reused_by_coverage_checker():
    candidates = build_candidates(_packet())
    with patch(
        "src.extraction.check_outcome_coverage.build_candidates",
        side_effect=AssertionError("candidates should be reused"),
    ):
        report = check(
            _packet(),
            _result(),
            assessment=assess(_packet()),
            candidates=candidates,
        )
    assert report.coverage_version == "outcome-coverage-1.1.0"


def test_medium_confidence_warning_blocks_completion_for_adjudication():
    packet = _packet()
    candidates = build_candidates(packet)
    visual = next(row for row in candidates if row.route_hint == "vision")
    visual.confidence = "medium"
    report = check(
        packet,
        _result(),
        assessment=assess(packet),
        candidates=candidates,
    )
    assert report.status == "review_unmatched_groups"
    assert report.unmatched_candidates == []
    assert len(report.review_candidates) == 1


def test_one_outcome_cannot_satisfy_two_candidate_groups():
    packet = _packet()
    packet.evidence.append(
        ApiEvidence(
            evidence_id="E-FVIII-SECOND",
            text="FVIII activity reached 6.0% after a second intervention.",
            retrieval_field_tags=["outcomes"],
            source_ids=["S-TEXT"],
        )
    )
    report = check(packet, _result())
    matched_outcome_ids = [row.outcome_id for row in report.matched_candidates]
    assert matched_outcome_ids.count("O1") == 1
    assert report.status == "review_unmatched_groups"


def _two_row_group_packet():
    """One candidate group holding two evidence rows; the result cites one."""
    packet = _packet()
    packet.evidence.append(
        ApiEvidence(
            evidence_id="E-FVIII-SECOND",
            text="FVIII activity reached 6.0% after a second intervention.",
            retrieval_field_tags=["outcomes"],
            source_ids=["S-TEXT"],
        )
    )
    group = next(
        row for row in build_candidates(packet) if "E-FVIII" in row.evidence_ids
    )
    assert set(group.evidence_ids) == {"E-FVIII", "E-FVIII-SECOND"}
    return packet, group


def test_partially_covered_group_is_discharged_whole_by_default():
    packet, group = _two_row_group_packet()
    report = check(packet, _result())
    assert group.candidate_id in {
        row.candidate_id for row in report.matched_candidates
    }
    assert group.candidate_id not in {
        row.candidate_id for row in report.unmatched_candidates
    }


def test_residue_targeting_reopens_a_group_holding_uncited_evidence():
    packet, group = _two_row_group_packet()
    with override(coverage_residue_targeting=True):
        report = check(packet, _result())
    # Still a real match -- the flag adds a second chance, it does not undo one.
    assert group.candidate_id in {
        row.candidate_id for row in report.matched_candidates
    }
    assert group.candidate_id in {
        row.candidate_id for row in report.unmatched_candidates
    }


def test_residue_targeting_leaves_a_fully_accounted_group_alone():
    packet = _packet()
    shipped = check(packet, _result())
    with override(coverage_residue_targeting=True):
        flagged = check(packet, _result())
    # Every row of the matched group is cited, so nothing is re-opened.
    assert flagged == shipped


def test_reopened_group_below_the_repair_threshold_says_why():
    packet, group = _two_row_group_packet()
    candidates = build_candidates(packet)
    next(
        row for row in candidates if row.candidate_id == group.candidate_id
    ).confidence = "medium"
    with override(coverage_residue_targeting=True):
        report = check(
            packet, _result(), assessment=assess(packet), candidates=candidates
        )
    reason = next(
        row.reason
        for row in report.review_candidates
        if row.candidate_id == group.candidate_id
    )
    assert reason == "partially_covered_group_below_repair_threshold"


def test_complexity_detects_mistagged_quantitative_biological_outcome():
    packet = _packet()
    for number in range(4):
        packet.evidence.append(
            ApiEvidence(
                evidence_id=f"E-MISTAG-{number}",
                text=(
                    f"Over {80 + number}% of BMDMs expressed GFP after "
                    "alpha-CD163 LNP treatment."
                ),
                retrieval_field_tags=["payload"],
                source_ids=["S-TEXT"],
            )
        )
    assessment = assess(packet)
    assert assessment.route == "complex"
    assert assessment.candidate_groups >= 4


def test_quantitative_outcome_survives_imperfect_retrieval_tags():
    packet = _packet()
    packet.evidence.append(
        ApiEvidence(
            evidence_id="E-BMDM",
            text=(
                "Over 80% of BMDMs expressed GFP after exposure to "
                "alpha-CD163 LNP-GFP."
            ),
            retrieval_field_tags=[
                "payload",
                "delivery_recipient_cell_reported",
            ],
            source_ids=["S-TEXT"],
        )
    )
    candidates = build_candidates(packet)
    candidate = next(
        row for row in candidates if "E-BMDM" in row.evidence_ids
    )
    assert candidate.endpoint_family == "expression"
    assert candidate.confidence == "high"


def test_husbandry_percentage_is_not_an_outcome():
    packet = _packet()
    packet.evidence.append(
        ApiEvidence(
            evidence_id="E-HUMIDITY",
            text=(
                "Mice were maintained at 65% humidity before receiving "
                "the fibrosis-targeting LNP."
            ),
            retrieval_field_tags=["payload"],
            source_ids=["S-TEXT"],
        )
    )
    candidates = build_candidates(packet)
    assert all("E-HUMIDITY" not in row.evidence_ids for row in candidates)


def test_quantified_mrna_delivery_is_an_expression_outcome():
    packet = _packet()
    packet.evidence.append(
        ApiEvidence(
            evidence_id="E-DELIVERY",
            text=(
                "Unmodified LNPs delivered mRNA to fewer than 20% "
                "of target BMDMs in vitro."
            ),
            retrieval_field_tags=["payload"],
            source_ids=["S-TEXT"],
        )
    )
    candidates = build_candidates(packet)
    candidate = next(
        row for row in candidates if "E-DELIVERY" in row.evidence_ids
    )
    assert candidate.endpoint_family == "expression"
    assert candidate.confidence == "high"
