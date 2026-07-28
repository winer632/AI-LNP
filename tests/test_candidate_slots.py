"""P3: candidate slots are sent with the first call and must all be answered.

The gap this closes: run_compact_one_call already computed the outcome
candidates before the call, but sent only the packet. A response that silently
omitted an experiment still passed schema validation, so GP-001/GP-003/GP-009
returning zero outcomes was indistinguishable from "the paper really has none".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config_flags import is_enabled, override
from src.extraction.compact_contracts import (
    CandidateSlotExtractionResponse,
    CompactExtractionResponse,
)
from src.extraction.compact_prompt_v1 import active_prompt, candidate_slot_payload
from src.extraction.compact_validation import validate_candidate

ROOT = Path(__file__).resolve().parents[1]
FLAG = "candidate_slot_enforcement"


def _allowed_evidence_ids(paper_id: str = "GP-001") -> set[str]:
    """The real packet ids, so validation fails on slots rather than on evidence."""
    from src.extraction.run_codex_one_call import PACKET_ROOT_BY_VIEW, load_packet

    packet = load_packet(paper_id, PACKET_ROOT_BY_VIEW["compact"])
    return {row.evidence_id for row in packet.evidence}


def _response_payload(paper_id: str = "GP-001", dispositions=None) -> dict:
    """Start from a real ineligible response so the fixture cannot drift from the contract."""
    source = json.loads(
        (ROOT / "data/staging/extraction/codex_control_compact_v1/GP-001/result.json")
        .read_text(encoding="utf-8")
    )
    payload = {
        key: source[key]
        for key in (
            "contract_version", "paper_id", "eligibility", "formulations",
            "components", "experiments", "outcomes", "unresolved_items",
        )
    }
    payload["paper_id"] = paper_id
    if dispositions is not None:
        payload["candidate_dispositions"] = dispositions
    return payload


# ---------------------------------------------------------------------------
# The flag is off by default and off means bit-identical
# ---------------------------------------------------------------------------


def test_flag_is_off_by_default():
    assert is_enabled(FLAG) is False


def test_prompt_version_is_frozen_when_the_flag_is_off():
    # A changed prompt version would silently invalidate every cached response
    # keyed on the old one, so pin it rather than merely asserting inequality.
    assert active_prompt(False).version == "compact-prompt-1.1.0"
    assert active_prompt(True).version == "compact-prompt-1.2.0"


def test_enabling_the_flag_does_not_mutate_the_disabled_prompt():
    baseline = active_prompt(False).text
    with override(candidate_slot_enforcement=True):
        assert is_enabled(FLAG) is True
        assert active_prompt(False).text == baseline


def test_disabled_response_contract_has_no_disposition_field():
    assert "candidate_dispositions" not in CompactExtractionResponse.model_fields
    assert "candidate_dispositions" in CandidateSlotExtractionResponse.model_fields


# ---------------------------------------------------------------------------
# Slot payload
# ---------------------------------------------------------------------------


def _real_candidates(paper_id: str = "GP-006"):
    from src.extraction.build_outcome_candidates import build_candidates
    from src.extraction.run_codex_one_call import PACKET_ROOT_BY_VIEW, load_packet

    return build_candidates(load_packet(paper_id, PACKET_ROOT_BY_VIEW["compact"]))


def test_slot_payload_exposes_candidate_ids_and_is_json_serialisable():
    candidates = _real_candidates()
    slots = candidate_slot_payload(candidates)
    assert len(slots) == len(candidates)
    assert [slot["candidate_id"] for slot in slots] == [
        candidate.candidate_id for candidate in candidates
    ]
    json.dumps(slots)  # must survive canonical serialisation into the prompt


def test_slot_payload_is_empty_for_no_candidates():
    assert candidate_slot_payload([]) == []


# ---------------------------------------------------------------------------
# Validation: silence is not a valid disposition
# ---------------------------------------------------------------------------


def test_unanswered_slot_is_rejected():
    text = json.dumps(_response_payload(dispositions=[]))
    parsed, report = validate_candidate(
        text,
        paper_id="GP-001",
        allowed_evidence_ids=_allowed_evidence_ids(),
        required_candidate_ids=["OC-1"],
    )
    codes = {finding.code for finding in report.findings}
    assert "missing_candidate_disposition" in codes
    assert report.status != "valid"


def test_fully_answered_slots_pass():
    text = json.dumps(
        _response_payload(
            dispositions=[
                {
                    "candidate_id": "OC-1",
                    "disposition": "not_an_outcome",
                    "reason": "The row reports an assay setting, not a measured outcome.",
                }
            ]
        )
    )
    parsed, report = validate_candidate(
        text,
        paper_id="GP-001",
        allowed_evidence_ids=_allowed_evidence_ids(),
        required_candidate_ids=["OC-1"],
    )
    assert parsed is not None
    assert {f.code for f in report.findings} == set()
    assert report.status == "valid"


def test_disposition_for_an_unsent_candidate_is_rejected():
    text = json.dumps(
        _response_payload(
            dispositions=[
                {
                    "candidate_id": "OC-1",
                    "disposition": "not_an_outcome",
                    "reason": "The row reports an assay setting, not a measured outcome.",
                },
                {
                    "candidate_id": "OC-INVENTED",
                    "disposition": "unresolved",
                    "reason": "Fabricated candidate id.",
                },
            ]
        )
    )
    _, report = validate_candidate(
        text,
        paper_id="GP-001",
        allowed_evidence_ids=_allowed_evidence_ids(),
        required_candidate_ids=["OC-1"],
    )
    assert report.status != "valid"


def test_no_required_ids_means_no_slot_findings():
    """With the flag off no slots are sent, so slot checks must stay silent."""
    text = json.dumps(_response_payload())
    parsed, report = validate_candidate(
        text, paper_id="GP-001", allowed_evidence_ids=_allowed_evidence_ids()
    )
    assert parsed is not None
    slot_codes = {
        f.code for f in report.findings if "candidate_disposition" in (f.code or "")
    }
    assert slot_codes == set()


# ---------------------------------------------------------------------------
# Contract: a non-extracted disposition must carry a reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("disposition", ["not_an_outcome", "unresolved"])
def test_non_extracted_disposition_requires_a_reason(disposition):
    text = json.dumps(
        _response_payload(
            dispositions=[{"candidate_id": "OC-1", "disposition": disposition}]
        )
    )
    parsed, report = validate_candidate(
        text,
        paper_id="GP-001",
        allowed_evidence_ids=_allowed_evidence_ids(),
        required_candidate_ids=["OC-1"],
    )
    assert parsed is None or report.status != "valid"


def test_strict_schema_requires_dispositions_only_when_enabled():
    from src.extraction.run_codex_one_call import strict_schema

    off = strict_schema(False)
    on = strict_schema(True)
    assert "candidate_dispositions" not in off.get("required", [])
    assert "candidate_dispositions" in on.get("required", [])


# ---------------------------------------------------------------------------
# "extracted" is a claim, not proof
# ---------------------------------------------------------------------------


def test_extracted_without_a_citing_outcome_is_rejected():
    """Measured on GP-006: the model marked all four candidates covering Table
    S2's insertion-frequency row as extracted while returning no record that
    carried 1.01. An unverified self-report reintroduces the silent omission
    the slots exist to prevent."""
    text = json.dumps(
        _response_payload(
            dispositions=[
                {"candidate_id": "OC-1", "disposition": "extracted", "reason": None}
            ]
        )
    )
    _, report = validate_candidate(
        text,
        paper_id="GP-001",
        allowed_evidence_ids=_allowed_evidence_ids(),
        required_candidate_ids=["OC-1"],
        candidate_evidence_ids={"OC-1": ["GP-001-E-not-cited-anywhere"]},
    )
    codes = {finding.code for finding in report.findings}
    assert "unsupported_extracted_disposition" in codes
    assert report.status != "valid"


def test_extracted_is_accepted_when_an_outcome_cites_the_candidate_evidence():
    payload = _response_payload(
        dispositions=[
            {"candidate_id": "OC-1", "disposition": "extracted", "reason": None}
        ]
    )
    evidence_id = sorted(_allowed_evidence_ids())[0]
    outcome = json.loads(
        (ROOT / "data/staging/extraction/codex_control_compact_v1/GP-004/result.json")
        .read_text(encoding="utf-8")
    )["outcomes"][0]
    payload["outcomes"] = [outcome]
    payload["experiments"] = json.loads(
        (ROOT / "data/staging/extraction/codex_control_compact_v1/GP-004/result.json")
        .read_text(encoding="utf-8")
    )["experiments"]
    cited = sorted(
        {
            eid
            for field in outcome.values()
            if isinstance(field, dict)
            for eid in (field.get("evidence_ids") or [])
        }
    )
    _, report = validate_candidate(
        json.dumps(payload),
        paper_id="GP-004",
        allowed_evidence_ids=set(cited) | _allowed_evidence_ids("GP-004"),
        required_candidate_ids=["OC-1"],
        candidate_evidence_ids={"OC-1": cited},
    )
    codes = {finding.code for finding in report.findings}
    assert "unsupported_extracted_disposition" not in codes


def test_extracted_check_is_inert_without_candidate_evidence_map():
    """Callers that do not supply the map keep the previous behaviour."""
    text = json.dumps(
        _response_payload(
            dispositions=[
                {"candidate_id": "OC-1", "disposition": "extracted", "reason": None}
            ]
        )
    )
    _, report = validate_candidate(
        text,
        paper_id="GP-001",
        allowed_evidence_ids=_allowed_evidence_ids(),
        required_candidate_ids=["OC-1"],
    )
    codes = {finding.code for finding in report.findings}
    assert "unsupported_extracted_disposition" not in codes
