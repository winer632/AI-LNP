import pytest

from src.extraction.compact_contracts import ExperimentRecord, OutcomeRecord
from src.extraction.compact_validation import ValidationFinding, ValidationReport
from src.extraction.missing_record_contracts import (
    MissingRecordFragment,
    MissingRecordTask,
)
from src.extraction.repair_contracts import RepairEvidence
from src.extraction.run_missing_record_repair import validate_response
from src.extraction.route_compact_findings import route


def _reported(value):
    return {
        "value": value,
        "status": "reported",
        "evidence_ids": ["E-1"],
        "missing_reason": None,
    }


def _missing():
    return {
        "value": None,
        "status": "missing",
        "evidence_ids": [],
        "missing_reason": "Not reported.",
    }


def _experiment():
    return ExperimentRecord(
        experiment_id="E2",
        formulation_id="F1",
        payload_type=_reported("mRNA"),
        payload_name=_reported("GFP mRNA"),
        encoded_product=_reported("GFP"),
        molecular_target=_missing(),
        delivery_recipient_cell=_reported("hepatocyte"),
        therapeutic_target_cell=_reported("hepatocyte"),
        tissue_or_organ=_reported("liver"),
        species=_reported("mouse"),
        disease_model=_missing(),
        experimental_context=_reported("in_vivo"),
        dose=_missing(),
        dose_unit=_missing(),
        route=_missing(),
        timepoint=_missing(),
        timepoint_unit=_missing(),
    )


def _outcome():
    return OutcomeRecord(
        outcome_id="O2",
        experiment_id="E2",
        assay=_reported("microscopy"),
        endpoint=_reported("GFP expression"),
        comparator=_missing(),
        outcome_value=_reported(80.0),
        outcome_unit=_reported("%"),
        qualitative_outcome=_reported("More than 80% expressed GFP."),
    )


def _task():
    return MissingRecordTask(
        task_version="missing-record-task-1.1.0",
        paper_id="GP-X",
        route_ids=["RT-1", "RT-2"],
        candidate_ids=["OC-1", "OC-2"],
        evidence=[
            RepairEvidence(evidence_id="E-1", text="More than 80%.", source_ids=["S1"])
        ],
        existing_formulation_ids=["F1"],
        existing_experiment_ids=["E1"],
        existing_outcome_ids=["O1"],
        permitted_new_experiments=1,
        permitted_new_outcomes=2,
        source_result_sha256="a" * 64,
        source_inventory_sha256="b" * 64,
        task_checksum="c" * 64,
    )


def test_missing_record_response_accounts_for_every_candidate():
    result = MissingRecordFragment(
        disposition="recovered",
        recovered_candidate_ids=["OC-1"],
        unresolved_candidate_ids=["OC-2"],
        experiments=[_experiment()],
        outcomes=[_outcome()],
        unresolved_reason="OC-2 requires a figure.",
    )
    validate_response(result, _task())


def test_missing_record_response_cannot_silently_drop_candidate():
    result = MissingRecordFragment(
        disposition="recovered",
        recovered_candidate_ids=["OC-1"],
        unresolved_candidate_ids=[],
        experiments=[_experiment()],
        outcomes=[_outcome()],
        unresolved_reason=None,
    )
    with pytest.raises(ValueError, match="every candidate"):
        validate_response(result, _task())


def test_missing_record_response_rejects_existing_id_collision():
    outcome = _outcome()
    outcome.outcome_id = "O1"
    result = MissingRecordFragment(
        disposition="recovered",
        recovered_candidate_ids=["OC-1"],
        unresolved_candidate_ids=["OC-2"],
        experiments=[_experiment()],
        outcomes=[outcome],
        unresolved_reason="OC-2 requires a figure.",
    )
    with pytest.raises(ValueError, match="existing outcome"):
        validate_response(result, _task())


def test_missing_record_response_rejects_made_up_evidence():
    outcome = _outcome()
    outcome.endpoint.evidence_ids = ["E-MADE-UP"]
    result = MissingRecordFragment(
        disposition="recovered",
        recovered_candidate_ids=["OC-1"],
        unresolved_candidate_ids=["OC-2"],
        experiments=[_experiment()],
        outcomes=[outcome],
        unresolved_reason="OC-2 requires a figure.",
    )
    with pytest.raises(ValueError, match="unknown evidence"):
        validate_response(result, _task())


def test_whole_response_schema_failure_requires_first_call_not_field_repair():
    finding = ValidationFinding(
        finding_id="VF-schema",
        code="pydantic.literal_error",
        message="Wrong contract version",
        location=["contract_version"],
        record_collection=None,
        record_index=None,
        field_name=None,
        cited_evidence_ids=[],
        repairable=False,
    )
    decision = route(
        paper_id="GP-X",
        complexity_route="complex",
        validation=ValidationReport(
            paper_id="GP-X", status="invalid", findings=[finding]
        ),
        coverage=None,
        inventory=None,
    )
    assert decision.routes[0].route == "first_call_required"
