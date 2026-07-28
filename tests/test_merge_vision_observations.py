"""Turning panel observations into outcome records, and the abstain rule.

The rule this module has to hold is narrow and load-bearing: only a
``resolved`` observation becomes an outcome record. ``missing``, ``ambiguous``
and ``human_review`` are the vision route saying it could not ground the claim,
and a record built from one of those is a fabricated claim wearing the same
shape as a measured one.

Fixtures are the two observations this route actually produced, kept under
``data/staging/extraction/codex_vision_v1/``, and a real extraction result. The
abstaining variants are derived from the real observation and re-validated
through :class:`SelectiveVisionResponse`, so they are shapes the contract
genuinely accepts rather than hand-written dicts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.extraction.merge_vision_observations import (
    merge_into_result,
    observation_to_outcome,
)
from src.extraction.selective_vision_contracts import SelectiveVisionResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
RESOLVED_OBSERVATION = (
    REPO_ROOT / "data/staging/extraction/codex_vision_v1/GP-004/observation.json"
)
REAL_RESULT = (
    REPO_ROOT
    / "data/staging/extraction/codex_control_compact_v1/GP-004/result.json"
)


def _resolved() -> dict:
    return json.loads(RESOLVED_OBSERVATION.read_text(encoding="utf-8"))


def _abstained(disposition: str) -> dict:
    """A contract-valid observation that declined to answer."""
    payload = _resolved()
    payload["disposition"] = disposition
    if disposition == "human_review":
        # The model judged the value by eye; the contract routes that to a human.
        payload["value_status"] = "visually_estimated"
        payload["requires_human_review"] = True
    else:
        payload["value_status"] = "not_resolved"
        payload["corrected_fragment"] = None
        payload["requires_human_review"] = False
    # Re-validate so the fixture is a shape the contract really accepts.
    return SelectiveVisionResponse.model_validate(payload).model_dump(mode="json")


@pytest.mark.parametrize("disposition", ["missing", "ambiguous", "human_review"])
def test_an_observation_that_abstained_never_becomes_an_outcome(disposition):
    record = observation_to_outcome(
        _abstained(disposition), outcome_id="V1", experiment_id="E1"
    )
    assert record is None


def test_a_resolved_observation_carries_its_value_unit_and_printed_labels():
    observation = _resolved()
    record = observation_to_outcome(observation, outcome_id="V1", experiment_id="E1")

    assert record is not None
    assert record["outcome_id"] == "V1"
    assert record["experiment_id"] == "E1"
    assert record["source_stage"] == "vision"
    assert record["outcome_value"]["value"] == 41.5
    assert record["outcome_value"]["status"] == "reported"
    assert record["outcome_unit"]["value"] == "%"
    # The marker names the claim is about only exist in printed_labels.
    assert record["printed_labels"] == observation["printed_labels"]
    assert "F4-80" in record["endpoint"]["value"]
    assert record["vision_panel"] == observation["panel_or_table_cell"]


def test_the_endpoint_carries_what_was_observed_not_the_claim():
    """The claim and the provenance are not scored as endpoint text.

    ``_result_text`` scores endpoint alongside qualitative_outcome, so putting
    the claim sentence in both counted it twice, and visible_support -- a
    description of the picture rather than a measurement -- was matched as
    though it were one. Vision records ran 55-62 tokens against 8-19 for text
    records, and ``lexical`` divides by ``min(len(expected), len(actual))``,
    so bulk past the gold length costs nothing while every incidental overlap
    is gain.
    """
    observation = _resolved()
    record = observation_to_outcome(observation, outcome_id="V1", experiment_id="E1")
    endpoint = record["endpoint"]["value"]

    assert observation["visible_support"] not in endpoint
    assert observation["corrected_fragment"]["qualitative_outcome"] not in endpoint
    # what it does carry: the panel observed, and the markers read off it
    assert observation["panel_or_table_cell"] in endpoint
    assert "F4-80" in endpoint
    # and the claim is still present, once, in its own field
    assert (
        record["qualitative_outcome"]["value"]
        == observation["corrected_fragment"]["qualitative_outcome"]
    )


def test_a_field_the_panel_did_not_print_is_marked_missing_not_guessed():
    """GP-006's real observation resolved qualitatively with no printed number."""
    observation = json.loads(
        (
            REPO_ROOT
            / "data/staging/extraction/codex_vision_v1/GP-006/observation.json"
        ).read_text(encoding="utf-8")
    )
    record = observation_to_outcome(observation, outcome_id="V1", experiment_id="E1")

    assert record is not None
    assert record["outcome_value"]["value"] is None
    assert record["outcome_value"]["status"] == "missing"
    assert record["outcome_value"]["missing_reason"]
    assert record["outcome_value"]["evidence_ids"] == []
    assert record["qualitative_outcome"]["status"] == "reported"


def test_merge_keeps_the_abstention_out_of_the_result_entirely(tmp_path):
    abstained = _abstained("human_review")
    output = tmp_path / "final_result.json"

    report = merge_into_result(REAL_RESULT, [abstained], output)

    merged = json.loads(output.read_text(encoding="utf-8"))
    original = json.loads(REAL_RESULT.read_text(encoding="utf-8"))
    assert report["added_outcome_ids"] == []
    assert report["skipped_observations"] == [
        {"finding_id": abstained["finding_id"], "disposition": "human_review"}
    ]
    assert len(merged["outcomes"]) == len(original["outcomes"])
    # Not merely absent as a record: the sentence must not leak into the result
    # in any shape a downstream reader could mistake for an extracted claim.
    claim = abstained["corrected_fragment"]["qualitative_outcome"]
    assert claim not in json.dumps(merged, ensure_ascii=False)


def test_merge_adds_a_resolved_observation_without_disturbing_the_source(tmp_path):
    original_bytes = REAL_RESULT.read_bytes()
    output = tmp_path / "final_result.json"

    report = merge_into_result(REAL_RESULT, [_resolved()], output)

    merged = json.loads(output.read_text(encoding="utf-8"))
    original = json.loads(original_bytes)
    assert len(report["added_outcome_ids"]) == 1
    assert report["total_outcomes"] == len(original["outcomes"]) + 1
    added = merged["outcomes"][-1]
    assert added["outcome_id"] == report["added_outcome_ids"][0]
    assert added["experiment_id"] == original["experiments"][0]["experiment_id"]
    # A test must never rewrite committed data.
    assert REAL_RESULT.read_bytes() == original_bytes


def test_merged_outcome_ids_do_not_collide_with_records_already_present(tmp_path):
    """Two resolved observations must not be handed the same generated id."""
    output = tmp_path / "final_result.json"
    seed = json.loads(REAL_RESULT.read_text(encoding="utf-8"))
    # A prior vision merge already took V1.
    seed["outcomes"].append({**seed["outcomes"][0], "outcome_id": "V1"})
    seeded = tmp_path / "seeded.json"
    seeded.write_text(json.dumps(seed), encoding="utf-8")

    report = merge_into_result(seeded, [_resolved(), _resolved()], output)

    merged = json.loads(output.read_text(encoding="utf-8"))
    ids = [row["outcome_id"] for row in merged["outcomes"]]
    assert len(ids) == len(set(ids))
    assert "V1" not in report["added_outcome_ids"]
    assert len(report["added_outcome_ids"]) == 2
