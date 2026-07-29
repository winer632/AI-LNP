"""An interpretive summary the paper asserts as a finding may be an outcome.

The annotation side already works this way. The frozen gold set records claims
a paper states about its own results in prose, so a rule that admits only a
measured value describes something the gold set is not. The extraction prompt
drew the line at "measured" -- "Do not convert a mechanism, hypothesis, or
interpretation into a measured outcome", and `not_an_outcome when the cited
evidence is a method, hypothesis, or interpretation rather than a measured or
reported result" -- and the two therefore disagreed about what an outcome is.

`interpretive_outcome_admission` moves the extraction to the annotation side.
These tests pin the three things that make that a policy change rather than a
loosening:

* the flag-off request stays byte-identical, prompt text, version and checksum;
* the evidence requirement survives the change unweakened;
* the rule stays general -- it describes a kind of statement, and introduces no
  vocabulary from the answer key.

The flag ships off. That is a measurement result, recorded in its rationale,
not a statement about the policy.
"""

from __future__ import annotations

# Version numbering note: these prompts are 1.4.x rather than 1.3.x. Two
# independent prompt amendments were developed in parallel and both claimed
# 1.3.0/1.3.1; cell_line_identity kept them and this one was renumbered on
# merge, so a version still identifies exactly one prompt text.
import csv
import json
from pathlib import Path

import pytest

from src.config_flags import describe_flag, is_enabled, override
from src.extraction.compact_prompt_v1 import (
    CANDIDATE_SLOT_EXTRACTION_PROMPT,
    CANDIDATE_SLOT_INTERPRETIVE_EXTRACTION_PROMPT,
    CANDIDATE_SLOT_PROMPT_VERSION,
    COMPACT_EXTRACTION_PROMPT,
    INTERPRETIVE_DISPOSITION_RULES,
    INTERPRETIVE_EXTRACTION_PROMPT,
    INTERPRETIVE_OUTCOME_RULES,
    PROMPT_VERSION,
    active_prompt,
    candidate_slot_prompt_sha256,
    prompt_sha256,
)
from src.extraction.compact_validation import validate_candidate
from src.extraction.evaluate_final_gold_dynamic import DISTINCTIVE_TERMS, _tokens

ROOT = Path(__file__).resolve().parents[1]
FLAG = "interpretive_outcome_admission"
GOLD = ROOT / "data/annotations/gold_v1"
# A committed, contract-valid, eligible response. Using a real one keeps the
# fixture from drifting away from the contract it is supposed to exercise.
FIXTURE = ROOT / "data/staging/extraction/codex_full_view_v2/GP-008/result.json"
ADDED_RULES = INTERPRETIVE_OUTCOME_RULES + INTERPRETIVE_DISPOSITION_RULES


# ---------------------------------------------------------------------------
# Off by default, and off means bit-identical
# ---------------------------------------------------------------------------


def test_flag_ships_off_until_a_measurement_justifies_it():
    assert is_enabled(FLAG) is False
    assert describe_flag(FLAG).rationale, "an available flag must say what it measured"


def test_flag_off_leaves_both_frozen_prompts_untouched():
    # Versions and checksums, not just text: a changed version silently
    # invalidates every cached response keyed on the old one, and the baseline
    # checksum is pinned in config/extraction/compact_route_v1.yaml.
    assert active_prompt(False) == (
        COMPACT_EXTRACTION_PROMPT,
        PROMPT_VERSION,
        prompt_sha256(),
    )
    assert active_prompt(True) == (
        CANDIDATE_SLOT_EXTRACTION_PROMPT,
        CANDIDATE_SLOT_PROMPT_VERSION,
        candidate_slot_prompt_sha256(),
    )


def test_enabling_the_flag_does_not_mutate_the_frozen_prompts():
    baseline, slots = COMPACT_EXTRACTION_PROMPT, CANDIDATE_SLOT_EXTRACTION_PROMPT
    with override(**{FLAG: True}):
        assert COMPACT_EXTRACTION_PROMPT == baseline
        assert CANDIDATE_SLOT_EXTRACTION_PROMPT == slots
        assert prompt_sha256() == (
            "b1589a1e37b5e0d9cc5c5aa250109da68a91c2f6f5ff40f51b021187172a8eb7"
        )


def test_flag_on_selects_its_own_versions_and_checksums():
    with override(**{FLAG: True}):
        plain, slotted = active_prompt(False), active_prompt(True)
    assert plain.version == "compact-prompt-1.4.0"
    assert slotted.version == "compact-prompt-1.4.1"
    assert plain.text == INTERPRETIVE_EXTRACTION_PROMPT
    assert slotted.text == CANDIDATE_SLOT_INTERPRETIVE_EXTRACTION_PROMPT
    assert len({plain.checksum, slotted.checksum,
                prompt_sha256(), candidate_slot_prompt_sha256()}) == 4


def test_an_explicit_argument_beats_the_flag_in_both_directions():
    """A measurement must be able to pin the prompt regardless of deployment."""
    with override(**{FLAG: True}):
        assert active_prompt(True, interpretive_outcome_admission=False).version == (
            CANDIDATE_SLOT_PROMPT_VERSION
        )
    with override(**{FLAG: False}):
        assert active_prompt(True, interpretive_outcome_admission=True).version == (
            "compact-prompt-1.4.1"
        )


# ---------------------------------------------------------------------------
# What the new rule says, and what it still refuses
# ---------------------------------------------------------------------------


def test_the_new_prompts_are_the_old_ones_plus_an_amendment():
    assert INTERPRETIVE_EXTRACTION_PROMPT.startswith(COMPACT_EXTRACTION_PROMPT)
    assert CANDIDATE_SLOT_INTERPRETIVE_EXTRACTION_PROMPT.startswith(
        CANDIDATE_SLOT_EXTRACTION_PROMPT
    )
    # The disposition sentence only means anything next to the text that
    # defines the three codes, so it must not reach the slot-free prompt.
    assert INTERPRETIVE_DISPOSITION_RULES not in INTERPRETIVE_EXTRACTION_PROMPT
    assert INTERPRETIVE_DISPOSITION_RULES in (
        CANDIDATE_SLOT_INTERPRETIVE_EXTRACTION_PROMPT
    )


def test_the_amendment_admits_a_finding_and_still_refuses_a_non_finding():
    admitted = [
        "stating what this paper's own experiments showed is a reportable outcome",
        "summary",
        "conclusion",
    ]
    for phrase in admitted:
        assert phrase in INTERPRETIVE_OUTCOME_RULES
    # Everything the earlier rule was really guarding is named again, so the
    # amendment narrows the prohibition rather than deleting it.
    for refused in ("attributed to earlier work", "an aim", "a hypothesis",
                    "a proposal", "a method", "an assay"):
        assert refused in INTERPRETIVE_OUTCOME_RULES


def test_the_evidence_requirement_is_restated_not_relaxed():
    with override(**{FLAG: True}):
        text = active_prompt(True).text
    # Inherited verbatim from the frozen baseline.
    assert (
        "return either a reported value with valid packet evidence IDs or "
        "missing with a short reason and no evidence IDs"
    ) in text
    assert (
        "a candidate marked extracted whose evidence you never cite is rejected"
    ) in text
    # And restated for the newly admitted kind of record.
    assert "Cite the packet evidence that carries the statement" in text
    assert "assert no more than that evidence says" in text
    assert "outcome_value missing unless the cited evidence itself states a value" in (
        text
    )


# ---------------------------------------------------------------------------
# The contract is untouched: a record still cites evidence that exists
# ---------------------------------------------------------------------------


def _payload() -> dict:
    source = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {
        key: source[key]
        for key in (
            "contract_version", "paper_id", "eligibility", "formulations",
            "components", "experiments", "outcomes", "unresolved_items",
        )
    }


def _allowed_ids(payload: dict) -> set[str]:
    from src.extraction.run_codex_one_call import PACKET_ROOT_BY_VIEW, load_packet

    packet = load_packet(payload["paper_id"], PACKET_ROOT_BY_VIEW["full"])
    return {row.evidence_id for row in packet.evidence}


def test_the_fixture_is_valid_to_begin_with():
    payload = _payload()
    parsed, report = validate_candidate(
        json.dumps(payload),
        paper_id=payload["paper_id"],
        allowed_evidence_ids=_allowed_ids(payload),
    )
    assert parsed is not None and report.status == "valid"


@pytest.mark.parametrize("flag_on", [False, True])
def test_an_invented_evidence_id_is_rejected_on_both_sides_of_the_flag(flag_on):
    payload = _payload()
    allowed = _allowed_ids(payload)
    payload["outcomes"][0]["qualitative_outcome"]["evidence_ids"] = [
        "GP-008-E-000000000000dead"
    ]
    with override(**{FLAG: flag_on}):
        parsed, report = validate_candidate(
            json.dumps(payload),
            paper_id=payload["paper_id"],
            allowed_evidence_ids=allowed,
        )
    assert parsed is None or report.status != "valid"


@pytest.mark.parametrize("flag_on", [False, True])
def test_a_reported_claim_citing_nothing_is_rejected_on_both_sides(flag_on):
    payload = _payload()
    allowed = _allowed_ids(payload)
    payload["outcomes"][0]["qualitative_outcome"]["evidence_ids"] = []
    with override(**{FLAG: flag_on}):
        parsed, report = validate_candidate(
            json.dumps(payload),
            paper_id=payload["paper_id"],
            allowed_evidence_ids=allowed,
        )
    assert parsed is None or report.status != "valid"


# ---------------------------------------------------------------------------
# The rule is general: it describes a kind of statement, not an answer
# ---------------------------------------------------------------------------


def _gold_rows() -> list[dict[str, str]]:
    with (GOLD / "outcomes.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _newly_introduced() -> set[str]:
    """Tokens the amendment adds that the two frozen prompts did not have."""
    return _tokens(ADDED_RULES) - _tokens(CANDIDATE_SLOT_EXTRACTION_PROMPT)


def test_the_amendment_names_no_gold_endpoint():
    leaked = {
        token
        for row in _gold_rows()
        for token in _tokens(row["endpoint_name"]) & _newly_introduced()
    }
    assert not leaked, f"the prompt names the answer key: {sorted(leaked)}"


def test_the_amendment_carries_no_distinctive_claim_vocabulary():
    """DISTINCTIVE_TERMS is the evaluator's own list of claim-specific words.

    A prompt that seeded any of them would be teaching the matcher's lexicon
    rather than stating a rule.
    """
    assert not (_newly_introduced() & DISTINCTIVE_TERMS)


def test_no_gold_claim_is_recognisable_from_the_amendment():
    """At most one ordinary word per gold claim, so no claim is reconstructible."""
    introduced = _newly_introduced()
    for row in _gold_rows():
        shared = _tokens(row["qualitative_outcome"]) & introduced
        assert len(shared) <= 1, (
            f"{row['gold_outcome_id']} shares {sorted(shared)} with the prompt"
        )
