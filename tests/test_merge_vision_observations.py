"""Turning panel reads into outcome records.

The load-bearing property here is that the observation's ``relationship``
survives as a structured field and never as text. The vocabulary spells its
negative member with the positive one inside it, so any text encoding of
"not_colocalized" reads to a bag-of-words matcher as an assertion of
"colocalized" -- the exact inversion the evaluator's polarity gate cannot see.
"""

import json

from src.extraction.merge_vision_observations import (
    merge_into_result,
    observation_to_outcome,
)
from src.extraction.selective_vision_contracts import (
    RELATIONSHIP_POLARITY,
    VisualRelationship,
)


def _observation(**overrides):
    observation = {
        "finding_id": "GO-X-fig1",
        "disposition": "resolved",
        "field_name": "qualitative_outcome",
        "corrected_fragment": {
            "qualitative_outcome": "GFP signal is present in marker-positive cells.",
            "outcome_value": None,
            "outcome_unit": None,
            "relationship": "colocalized",
        },
        "value_status": "exact_reported",
        "supporting_evidence_ids": ["V-1"],
        "figure_or_table": "Figure 2B",
        "panel_or_table_cell": "Merge",
        "visible_support": "Overlapping green and red signal in the merge panel.",
        "derivation": None,
        "confidence": "high",
        "requires_human_review": False,
        "printed_labels": ["100 um"],
        "relationship": "colocalized",
    }
    observation.update(overrides)
    return observation


def test_resolved_observation_carries_the_relationship_structurally():
    record = observation_to_outcome(
        _observation(), outcome_id="V1", experiment_id="E1"
    )
    assert record["vision_relationship"] == "colocalized"
    assert record["source_stage"] == "vision"


def test_relationship_is_never_written_into_a_text_field():
    """A denial folded into prose would read as its own affirmation."""
    record = observation_to_outcome(
        _observation(
            relationship="not_colocalized",
            corrected_fragment={
                "qualitative_outcome": "Signals occupy separate cells.",
                "outcome_value": None,
                "outcome_unit": None,
                "relationship": "not_colocalized",
            },
            visible_support="Green and red signal are in separate cells.",
        ),
        outcome_id="V1",
        experiment_id="E1",
    )
    assert record["vision_relationship"] == "not_colocalized"
    for field in ("assay", "endpoint", "comparator", "outcome_unit",
                  "qualitative_outcome"):
        assert "colocalized" not in str(record[field]["value"] or "").lower()


def test_every_vocabulary_member_has_a_declared_polarity():
    """The evaluator reads polarity from the table, so it must be total."""
    assert set(RELATIONSHIP_POLARITY) == set(VisualRelationship.__args__)
    for value, (base, affirmed) in RELATIONSHIP_POLARITY.items():
        assert isinstance(base, str) and isinstance(affirmed, bool)
    assert RELATIONSHIP_POLARITY["colocalized"] == ("colocalized", True)
    assert RELATIONSHIP_POLARITY["not_colocalized"] == ("colocalized", False)


def test_an_abstaining_observation_never_becomes_a_record():
    """The abstain rule, restated against the new field.

    An observation that did not resolve must not acquire a relationship,
    because a record carrying one is a record making a claim.
    """
    for disposition in ("missing", "ambiguous", "human_review"):
        assert (
            observation_to_outcome(
                _observation(disposition=disposition),
                outcome_id="V1",
                experiment_id="E1",
            )
            is None
        )


def test_merge_adds_resolved_records_and_reports_the_abstentions(tmp_path):
    result_path = tmp_path / "final_result.json"
    result_path.write_text(
        json.dumps(
            {
                "paper_id": "GP-T01",
                "experiments": [{"experiment_id": "E1"}],
                "outcomes": [{"outcome_id": "O1"}],
            }
        ),
        encoding="utf-8",
    )
    report = merge_into_result(
        result_path,
        [
            _observation(),
            _observation(finding_id="GO-Y", disposition="ambiguous",
                         corrected_fragment=None, value_status="not_resolved"),
        ],
        result_path,
    )
    assert report["added_outcome_ids"] == ["V1"]
    assert report["skipped_observations"] == [
        {"finding_id": "GO-Y", "disposition": "ambiguous"}
    ]
    merged = json.loads(result_path.read_text(encoding="utf-8"))
    added = [row for row in merged["outcomes"] if row["outcome_id"] == "V1"]
    assert added[0]["vision_relationship"] == "colocalized"
    assert all(
        "vision_relationship" not in row
        for row in merged["outcomes"]
        if row["outcome_id"] == "O1"
    )
