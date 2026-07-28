"""P2: a visual value is only "exact" if the labels backing it were transcribed.

The abstain rule (visually_estimated must go to human review) was already
enforced. What was missing is the other half: nothing stopped a response from
claiming ``exact_reported`` without recording what it actually read off the
image, which is indistinguishable from a confident guess.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config_flags import is_enabled, override
from src.extraction.selective_vision_contracts import (
    P2_FLAG,
    SelectiveVisionResponse,
    p2_enabled,
)


def _response(**overrides) -> dict:
    payload = {
        "finding_id": "F-1",
        "disposition": "resolved",
        "field_name": "outcome_value",
        "corrected_fragment": {"value": 1.01},
        "value_status": "exact_reported",
        "supporting_evidence_ids": ["GP-006-E-1"],
        "figure_or_table": "Table S2",
        "panel_or_table_cell": "row LSEC / col total insertion frequency",
        "visible_support": "The LSEC row reads 1.01 +/- 0.38 % in the final column.",
        "derivation": None,
        "confidence": "high",
        "requires_human_review": False,
        "printed_labels": ["LSEC", "total insertion frequency", "1.01 ± 0.38 %"],
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Flag
# ---------------------------------------------------------------------------


def test_flag_is_off_by_default():
    assert is_enabled(P2_FLAG) is False
    assert p2_enabled() is False


def test_exact_reported_without_labels_is_accepted_while_the_flag_is_off():
    """Responses recorded before the rule existed must stay loadable."""
    model = SelectiveVisionResponse(**_response(printed_labels=[]))
    assert model.value_status == "exact_reported"


def test_exact_reported_without_labels_is_rejected_once_enabled():
    with override(local_vlm_vision=True):
        with pytest.raises(ValidationError, match="printed_labels"):
            SelectiveVisionResponse(**_response(printed_labels=[]))


def test_exact_reported_with_labels_passes_when_enabled():
    with override(local_vlm_vision=True):
        model = SelectiveVisionResponse(**_response())
        assert model.printed_labels[0] == "LSEC"


# ---------------------------------------------------------------------------
# The pre-existing abstain rule must not be weakened by the new one
# ---------------------------------------------------------------------------


def test_visual_estimate_still_requires_human_review():
    for enabled in (False, True):
        with override(local_vlm_vision=enabled):
            with pytest.raises(ValidationError):
                SelectiveVisionResponse(
                    **_response(
                        value_status="visually_estimated",
                        disposition="resolved",
                        requires_human_review=False,
                    )
                )


def test_visual_estimate_routed_to_human_review_is_accepted():
    for enabled in (False, True):
        with override(local_vlm_vision=enabled):
            model = SelectiveVisionResponse(
                **_response(
                    value_status="visually_estimated",
                    disposition="human_review",
                    corrected_fragment=None,
                    requires_human_review=True,
                )
            )
            assert model.requires_human_review is True


def test_derived_value_still_requires_a_derivation():
    with override(local_vlm_vision=True):
        with pytest.raises(ValidationError, match="derivation"):
            SelectiveVisionResponse(
                **_response(value_status="derived", derivation=None)
            )


def test_derived_value_does_not_require_printed_labels():
    """A derivation documents its own provenance; labels are for transcription."""
    with override(local_vlm_vision=True):
        model = SelectiveVisionResponse(
            **_response(
                value_status="derived",
                derivation="Sum of the five insertion columns.",
                printed_labels=[],
            )
        )
        assert model.value_status == "derived"


@pytest.mark.parametrize("disposition", ["missing", "ambiguous"])
def test_unresolved_dispositions_carry_no_correction(disposition):
    with override(local_vlm_vision=True):
        with pytest.raises(ValidationError):
            SelectiveVisionResponse(
                **_response(
                    disposition=disposition,
                    value_status="not_resolved",
                    corrected_fragment={"value": 1.01},
                )
            )


def test_abstained_reports_itself_as_abstaining():
    with override(local_vlm_vision=True):
        model = SelectiveVisionResponse(
            **_response(
                disposition="human_review",
                value_status="not_resolved",
                corrected_fragment=None,
                requires_human_review=True,
                printed_labels=[],
            )
        )
        assert model.abstained is True
