"""Per-record salvage of contract-invalid extraction responses.

The property under test is narrow and it is the whole point: salvage may lower
the *blast radius* of the evidence-citation rule and may never lower the rule.
A record citing an id the packet does not contain has to be rejected here
exactly as the document-level contract rejects it, and everything the salvage
refused has to appear in the report rather than quietly vanishing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config_flags import describe_flag, is_enabled, override
from src.extraction.compact_validation import validate_candidate
from src.extraction.salvage_invalid_response import (
    SALVAGE_CONTRACT_VERSION,
    SALVAGE_FLAG,
    build,
    salvage_response,
    salvage_run_dir,
)


ROOT = Path(__file__).resolve().parents[1]
FULL_VIEW_RUN = ROOT / "data/staging/extraction/codex_treatment_full_v1"
STRUCTURED_RUN = ROOT / "data/staging/extraction/structured_compact_one_call_v1"
FULL_PACKETS = ROOT / "data/staging/rag/full_api_packets_v1"

KNOWN = {"GP-TEST-E-0001", "GP-TEST-E-0002", "GP-TEST-E-0003"}
HALLUCINATED = "GP-TEST-FC-31eb0348edb2aaf828"


def field(value, evidence_ids):
    return {
        "value": value,
        "status": "reported",
        "evidence_ids": list(evidence_ids),
        "missing_reason": None,
    }


def missing(reason="not reported"):
    return {
        "value": None,
        "status": "missing",
        "evidence_ids": [],
        "missing_reason": reason,
    }


def outcome(outcome_id="O1", evidence_id="GP-TEST-E-0001", experiment_id="E1"):
    return {
        "outcome_id": outcome_id,
        "experiment_id": experiment_id,
        "assay": field("immunohistochemistry", [evidence_id]),
        "endpoint": field("eGFP expression in hepatocytes", [evidence_id]),
        "comparator": field("Poly(C) RNA-LNP", [evidence_id]),
        "outcome_value": missing("no numeric value reported"),
        "outcome_unit": missing("no numeric value reported"),
        "qualitative_outcome": field(
            "Virtually all hepatocytes expressed eGFP.", [evidence_id]
        ),
    }


def experiment(experiment_id="E1", evidence_id="GP-TEST-E-0002"):
    record = {
        "experiment_id": experiment_id,
        "formulation_id": "F1",
        "experimental_context": field("in_vivo", [evidence_id]),
    }
    for name in (
        "payload_type",
        "payload_name",
        "encoded_product",
        "molecular_target",
        "delivery_recipient_cell",
        "therapeutic_target_cell",
        "tissue_or_organ",
        "species",
        "disease_model",
        "dose_unit",
        "route",
        "timepoint_unit",
    ):
        record[name] = field("mouse" if name == "species" else "reported", [evidence_id])
    record["dose"] = missing()
    record["timepoint"] = missing()
    return record


def formulation(evidence_id="GP-TEST-E-0003"):
    return {
        "formulation_id": "F1",
        "formulation_name": field("mRNA-LNP", [evidence_id]),
        "composition": field("ionizable lipid / cholesterol", [evidence_id]),
        "composition_basis": field("reported lipid classes", [evidence_id]),
        "np_ratio": missing("no N/P ratio reported"),
    }


def response_body(**overrides):
    body = {
        "contract_version": "compact-1.1.0",
        "paper_id": "GP-TEST",
        "eligibility": {
            "decision": "eligible",
            "reason_codes": [
                "ORIGINAL_EXPERIMENT",
                "IDENTIFIABLE_LNP",
                "SUPPORTED_PAYLOAD",
                "TARGET_CELL_EVIDENCE",
                "USABLE_FORMULATION_OUTCOME_LINKAGE",
            ],
            "evidence_ids": ["GP-TEST-E-0001"],
            "explanation": "Original mouse experiment with an identifiable LNP.",
        },
        "formulations": [formulation()],
        "components": [],
        "experiments": [experiment()],
        "outcomes": [outcome()],
        "unresolved_items": ["component ratios not reported"],
    }
    body.update(overrides)
    return json.dumps(body)


def records_only(document):
    """The document without its salvage report.

    The report names the ids that were dropped -- that is what makes the result
    auditable -- so "the bad id does not appear" has to be asserted about the
    records, not about the file.
    """
    return {key: value for key, value in document.items() if key != "salvage"}


def salvage(text, *, paper_id="GP-TEST", allowed=KNOWN):
    _, validation = validate_candidate(
        text, paper_id=paper_id, allowed_evidence_ids=set(allowed)
    )
    return salvage_response(
        text,
        paper_id=paper_id,
        allowed_evidence_ids=allowed,
        validation=validation,
    )


# --------------------------------------------------------------------------- #
# The property the contract exists to protect
# --------------------------------------------------------------------------- #


def test_a_record_citing_a_hallucinated_evidence_id_is_still_rejected():
    """The rule loses its blast radius, not its teeth.

    The outcome here cites an id no packet contains. Salvage must drop it, and
    must drop it *whole* -- keeping the record with the bad id filtered out of
    its evidence_ids would leave a claim standing on nothing, which is the
    fabrication the contract exists to stop.
    """
    text = response_body(
        outcomes=[
            outcome("O1", "GP-TEST-E-0001"),
            outcome("O2", HALLUCINATED),
        ]
    )
    with override(**{SALVAGE_FLAG: True}):
        document, report = salvage(text)

    assert report.status == "salvaged"
    assert [row["outcome_id"] for row in document["outcomes"]] == ["O1"]
    assert report.rejected["outcomes"] == 1
    assert HALLUCINATED not in json.dumps(records_only(document))

    dropped = [item for item in report.dropped if item.reason == "unknown_evidence_id"]
    assert [item.location for item in dropped] == [["outcomes", 1]]
    assert dropped[0].record_id == "O2"
    assert dropped[0].unknown_evidence_ids == [HALLUCINATED]


def test_no_salvaged_record_cites_an_id_outside_the_packet():
    """Stated as the invariant rather than as one example.

    Every id reachable in the salvaged document has to be in the packet. This
    is the check that would catch a salvage that kept a record because some
    *other* record's citations were clean.
    """
    text = response_body(
        outcomes=[outcome("O1", "GP-TEST-E-0001"), outcome("O2", HALLUCINATED)],
        experiments=[experiment("E1", "GP-TEST-E-0002"), experiment("E2", HALLUCINATED)],
        formulations=[formulation("GP-TEST-E-0003")],
    )
    with override(**{SALVAGE_FLAG: True}):
        document, _ = salvage(text)

    cited = set()
    for collection in ("formulations", "components", "experiments", "outcomes"):
        for record in document[collection]:
            for value in record.values():
                if isinstance(value, dict):
                    cited |= set(value.get("evidence_ids") or [])
    assert cited <= KNOWN
    assert cited


def test_an_unknown_citation_in_eligibility_costs_eligibility_not_the_records():
    """The finding this module was built for, in miniature.

    One bad id in a field no outcome record cites used to discard every record
    in the response. It now costs the field that carried it.
    """
    text = response_body(
        eligibility={
            "decision": "eligible",
            "reason_codes": [
                "ORIGINAL_EXPERIMENT",
                "IDENTIFIABLE_LNP",
                "SUPPORTED_PAYLOAD",
                "TARGET_CELL_EVIDENCE",
                "USABLE_FORMULATION_OUTCOME_LINKAGE",
            ],
            "evidence_ids": ["GP-TEST-E-0001", HALLUCINATED],
            "explanation": "Original mouse experiment with an identifiable LNP.",
        }
    )
    parsed, validation = validate_candidate(
        text, paper_id="GP-TEST", allowed_evidence_ids=KNOWN
    )
    assert parsed is None and validation.status == "invalid"

    with override(**{SALVAGE_FLAG: True}):
        document, report = salvage(text)

    assert report.status == "salvaged"
    assert len(document["outcomes"]) == 1
    # Recorded as unusable, never re-cited: the decision does not come back
    # with the bad id quietly removed.
    assert document["eligibility"] is None
    assert report.eligibility_kept is False
    assert HALLUCINATED not in json.dumps(records_only(document))
    # ...but the drop list says which id was refused, so the loss is legible.
    assert HALLUCINATED in json.dumps(document["salvage"])


# --------------------------------------------------------------------------- #
# What salvage keeps, and what it says about what it dropped
# --------------------------------------------------------------------------- #


def test_a_response_whose_outcomes_are_all_clean_is_salvaged_whole():
    text = response_body(
        components=[
            {
                "component_id": "C1",
                "formulation_id": "F1",
                "identity": field("ionizable lipid", [HALLUCINATED]),
                "role": field("ionizable_lipid", ["GP-TEST-E-0003"]),
                "amount": missing(),
                "amount_unit": missing(),
            }
        ]
    )
    with override(**{SALVAGE_FLAG: True}):
        document, report = salvage(text)

    assert report.kept["outcomes"] == 1
    assert report.kept["experiments"] == 1
    assert report.kept["formulations"] == 1
    assert report.rejected == {
        "formulations": 0,
        "components": 1,
        "experiments": 0,
        "outcomes": 0,
    }
    assert document["outcomes"] == json.loads(text)["outcomes"]
    assert document["unresolved_items"] == ["component ratios not reported"]


def test_the_dropped_fields_are_recorded_not_silently_omitted():
    """A partial result that does not say it is partial is worse than none."""
    text = response_body(
        outcomes=[outcome("O1", "GP-TEST-E-0001"), outcome("O2", HALLUCINATED)],
        unresolved_items=["a real note", 17],
    )
    with override(**{SALVAGE_FLAG: True}):
        document, report = salvage(text)

    reasons = {item.reason for item in report.dropped}
    assert "unknown_evidence_id" in reasons
    assert "contract_invalid" in reasons
    locations = [item.location for item in report.dropped]
    assert ["outcomes", 1] in locations
    assert ["unresolved_items", 1] in locations
    assert all(item.detail for item in report.dropped)

    # The drop list travels *with* the document, not only beside it.
    assert document["salvage"]["dropped"] == [
        item.model_dump(mode="json") for item in report.dropped
    ]
    assert document["contract_version"] == SALVAGE_CONTRACT_VERSION
    assert document["contract_version"] != "compact-1.1.0"


def test_a_kept_record_whose_parent_was_dropped_is_reported_as_dangling():
    text = response_body(experiments=[experiment("E1", HALLUCINATED)])
    with override(**{SALVAGE_FLAG: True}):
        document, report = salvage(text)

    assert report.kept["outcomes"] == 1
    assert report.rejected["experiments"] == 1
    assert [row.references for row in report.dangling_references] == ["E1"]
    # The reference is left as the model wrote it rather than re-pointed.
    assert document["outcomes"][0]["experiment_id"] == "E1"


def test_a_response_about_another_paper_is_refused_whole():
    text = response_body(paper_id="GP-OTHER")
    with override(**{SALVAGE_FLAG: True}):
        document, report = salvage(text)

    assert document is None
    assert report.status == "unsalvageable"
    assert [item.reason for item in report.dropped] == ["paper_id_mismatch"]


def test_an_unparseable_body_is_reported_rather_than_guessed_at():
    with override(**{SALVAGE_FLAG: True}):
        document, report = salvage("{not json")

    assert document is None
    assert report.status == "unsalvageable"
    assert [item.reason for item in report.dropped] == ["invalid_json"]


def test_a_response_with_no_surviving_record_yields_no_document():
    text = response_body(
        formulations=[formulation(HALLUCINATED)],
        experiments=[experiment("E1", HALLUCINATED)],
        outcomes=[outcome("O1", HALLUCINATED)],
    )
    with override(**{SALVAGE_FLAG: True}):
        document, report = salvage(text)

    assert document is None
    assert report.status == "unsalvageable"
    assert report.kept_records == 0
    assert report.rejected_records == 3


def test_a_field_the_salvage_has_no_rule_for_is_recorded_as_dropped():
    text = response_body(candidate_dispositions=[{"candidate_id": "OC-1"}])
    with override(**{SALVAGE_FLAG: True}):
        document, report = salvage(text)

    unhandled = [item for item in report.dropped if item.reason == "unhandled_field"]
    assert [item.location for item in unhandled] == [["candidate_dispositions"]]
    assert "candidate_dispositions" not in document


# --------------------------------------------------------------------------- #
# The flag
# --------------------------------------------------------------------------- #


def test_salvage_is_inert_when_the_flag_is_off():
    text = response_body(
        outcomes=[outcome("O1", "GP-TEST-E-0001"), outcome("O2", HALLUCINATED)]
    )
    with override(**{SALVAGE_FLAG: False}):
        document, report = salvage(text)

    assert document is None
    assert report.status == "disabled"
    assert report.kept == {}
    assert report.dropped == []


def test_the_flag_ships_off():
    flag = describe_flag(SALVAGE_FLAG)
    assert flag.default is False
    assert flag.status == "available"
    assert flag.rationale
    assert list(flag.integration_points) == [
        "src/extraction/salvage_invalid_response.py"
    ]
    assert is_enabled(SALVAGE_FLAG, env={}) is False


def test_a_valid_response_is_not_salvaged_at_all():
    text = response_body()
    parsed, validation = validate_candidate(
        text, paper_id="GP-TEST", allowed_evidence_ids=KNOWN
    )
    assert parsed is not None and validation.status == "valid"

    with override(**{SALVAGE_FLAG: True}):
        document, report = salvage(text)

    assert document is None
    assert report.status == "not_attempted"


# --------------------------------------------------------------------------- #
# Against the responses the model actually produced
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not (FULL_VIEW_RUN / "GP-007" / "response.json").exists(),
    reason="the committed full-view run is not present",
)
def test_the_committed_full_view_gp007_response_salvages_its_four_outcomes():
    """A real response, a real rejection, a real per-record answer.

    ``codex_treatment_full_v1/GP-007`` cites ``GP-007-FC-b360a234fb68377d`` in
    one experiment field. That id is not in the full packet as it stands today,
    so the document-level contract rejects the whole response and its four
    outcome records with it. Per-record salvage drops the experiment and keeps
    the four outcomes, none of which cite the bad id.
    """
    with override(**{SALVAGE_FLAG: True}):
        document, report, context = salvage_run_dir(FULL_VIEW_RUN / "GP-007")

    assert context["validation_status"] == "invalid"
    assert report.status == "salvaged"
    assert report.kept["outcomes"] == 4
    assert report.rejected["experiments"] == 1
    assert [item.unknown_evidence_ids for item in report.dropped] == [
        ["GP-007-FC-b360a234fb68377d"]
    ]

    allowed = {
        row["evidence_id"]
        for row in json.loads((FULL_PACKETS / "GP-007.json").read_text())["evidence"]
    }
    cited = set()
    for collection in ("formulations", "components", "experiments", "outcomes"):
        for record in document[collection]:
            for value in record.values():
                if isinstance(value, dict):
                    cited |= set(value.get("evidence_ids") or [])
    assert cited <= allowed


@pytest.mark.skipif(
    not (STRUCTURED_RUN / "GP-001" / "response.json").exists(),
    reason="the committed structured run is not present",
)
def test_the_one_response_this_repository_actually_discarded_carries_no_records():
    """Honest about what salvage does not buy here.

    ``structured_compact_one_call_v1/GP-001`` is the only committed response
    that was rejected at run time and left without a ``result.json``. It
    hallucinated ``GP-001-FC-62738f4303a2f`` in ``eligibility``, exactly the
    shape the salvage was built for -- and it returned zero records, because
    GP-001 is ineligible. So the salvage recovers nothing from it, and says so
    rather than emitting an empty document.
    """
    assert not (STRUCTURED_RUN / "GP-001" / "result.json").exists()
    with override(**{SALVAGE_FLAG: True}):
        document, report, context = salvage_run_dir(STRUCTURED_RUN / "GP-001")

    assert context["validation_status"] == "invalid"
    assert document is None
    assert report.status == "unsalvageable"
    assert report.kept_records == 0
    assert [item.unknown_evidence_ids for item in report.dropped] == [
        ["GP-001-FC-62738f4303a2f"]
    ]


# --------------------------------------------------------------------------- #
# Wired into the extraction run
# --------------------------------------------------------------------------- #


@pytest.fixture
def replayed_gp007(monkeypatch):
    """Replay the committed GP-007 full-view response through ``run_one``."""
    import src.extraction.run_codex_one_call as module

    response = json.loads((FULL_VIEW_RUN / "GP-007" / "response.json").read_text())
    monkeypatch.setattr(
        module,
        "run_codex",
        lambda prompt, schema, **kwargs: {
            "text": response["text"],
            "elapsed_seconds": 0.0,
            "stdout_tail": "",
        },
    )
    return module.run_one


@pytest.mark.skipif(
    not (FULL_VIEW_RUN / "GP-007" / "response.json").exists(),
    reason="the committed full-view run is not present",
)
def test_a_rejected_run_keeps_its_records_when_the_flag_is_on(
    tmp_path, replayed_gp007
):
    with override(**{SALVAGE_FLAG: True, "candidate_slot_enforcement": False}):
        manifest = replayed_gp007(
            "GP-007",
            evidence_view="full",
            packet_root=FULL_PACKETS,
            output_root=tmp_path,
        )

    run_dir = tmp_path / "GP-007"
    # Still invalid, and still no result.json: salvage does not make a
    # contract-invalid response valid, and nothing that reads result.json sees
    # a partial document.
    assert manifest["validation_status"] == "invalid"
    assert not (run_dir / "result.json").exists()
    assert manifest["salvage"]["status"] == "salvaged"
    assert manifest["salvage"]["kept"]["outcomes"] == 4
    assert manifest["salvage"]["rejected"]["experiments"] == 1

    document = json.loads((run_dir / "salvaged_result.json").read_text())
    assert len(document["outcomes"]) == 4
    assert (run_dir / "salvage_report.json").exists()


@pytest.mark.skipif(
    not (FULL_VIEW_RUN / "GP-007" / "response.json").exists(),
    reason="the committed full-view run is not present",
)
def test_a_rejected_run_writes_nothing_extra_when_the_flag_is_off(
    tmp_path, replayed_gp007
):
    with override(**{SALVAGE_FLAG: False, "candidate_slot_enforcement": False}):
        manifest = replayed_gp007(
            "GP-007",
            evidence_view="full",
            packet_root=FULL_PACKETS,
            output_root=tmp_path,
        )

    run_dir = tmp_path / "GP-007"
    assert manifest["validation_status"] == "invalid"
    assert "salvage" not in manifest
    assert not (run_dir / "salvaged_result.json").exists()
    assert not (run_dir / "salvage_report.json").exists()
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "complexity.json",
        "manifest.json",
        "outcome_candidates.json",
        "request.json",
        "response.json",
        "validation_report.json",
    ]


# --------------------------------------------------------------------------- #
# The materialised root
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not (FULL_VIEW_RUN / "GP-007" / "response.json").exists(),
    reason="the committed full-view run is not present",
)
def test_build_writes_a_readable_root_only_for_the_papers_it_salvaged(tmp_path):
    with override(**{SALVAGE_FLAG: True}):
        manifest = build([FULL_VIEW_RUN], output_root=tmp_path)

    written = sorted(path.name for path in tmp_path.glob("GP-*"))
    assert written == ["GP-006", "GP-007"]
    for paper_id in written:
        document = json.loads(
            (tmp_path / paper_id / "final_result.json").read_text()
        )
        assert document["contract_version"] == SALVAGE_CONTRACT_VERSION
        assert document["salvage"]["status"] == "salvaged"
        assert (tmp_path / paper_id / "salvage_report.json").exists()
    assert manifest["flag_enabled"] is True
    # Every paper is accounted for, including the seven that needed nothing.
    assert len(manifest["papers"]) == 9


@pytest.mark.skipif(
    not (FULL_VIEW_RUN / "GP-007" / "response.json").exists(),
    reason="the committed full-view run is not present",
)
def test_build_writes_no_result_at_all_when_the_flag_is_off(tmp_path):
    with override(**{SALVAGE_FLAG: False}):
        manifest = build([FULL_VIEW_RUN], output_root=tmp_path)

    assert list(tmp_path.glob("GP-*")) == []
    assert manifest["flag_enabled"] is False
    assert {entry["salvage_status"] for entry in manifest["papers"]} == {"disabled"}
