"""Evidence-bounded contracts for adding omitted experiments and outcomes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.extraction.compact_contracts import ExperimentRecord, OutcomeRecord
from src.extraction.repair_contracts import RepairEvidence


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MissingRecordTask(StrictModel):
    """A repair task, bounded by the evidence it carries.

    Version 1.1.0 adds ``existing_experiments``. Until then the task named
    the experiments only by id, and an outcome record must cite an experiment,
    so a model that had genuinely recovered an outcome could not attach it to
    anything without inventing that experiment's required fields. It correctly
    refused instead: "does not identify the associated experiment ID ...
    without inventing required experiment fields." Three repair calls returned
    unresolved for exactly this reason and produced no records.

    The ids stay for readers that only need identity.
    """

    # Both versions parse. 1.0.0 tasks are committed as the historical record
    # of runs that produced measurements; rejecting them would invalidate that
    # evidence to gain nothing. New tasks are built at 1.1.0.
    task_version: Literal["missing-record-task-1.0.0", "missing-record-task-1.1.0"]
    paper_id: str
    route_ids: list[str] = Field(min_length=1)
    candidate_ids: list[str] = Field(min_length=1)
    evidence: list[RepairEvidence] = Field(min_length=1, max_length=12)
    existing_formulation_ids: list[str]
    existing_experiment_ids: list[str]
    # The records themselves, so a recovered outcome has something legal to
    # attach to. Empty when the source result declares no experiments.
    existing_experiments: list[ExperimentRecord] = Field(default_factory=list)
    existing_outcome_ids: list[str]
    permitted_new_experiments: int = Field(ge=0, le=2)
    permitted_new_outcomes: int = Field(ge=1, le=8)
    source_result_sha256: str
    source_inventory_sha256: str
    task_checksum: str


class MissingRecordFragment(StrictModel):
    disposition: Literal["recovered", "unresolved"]
    recovered_candidate_ids: list[str]
    unresolved_candidate_ids: list[str]
    experiments: list[ExperimentRecord] = Field(max_length=2)
    outcomes: list[OutcomeRecord] = Field(max_length=8)
    unresolved_reason: str | None

    @model_validator(mode="after")
    def validate_disposition(self) -> "MissingRecordFragment":
        if len(set(self.recovered_candidate_ids)) != len(self.recovered_candidate_ids):
            raise ValueError("recovered_candidate_ids must be unique")
        if len(set(self.unresolved_candidate_ids)) != len(self.unresolved_candidate_ids):
            raise ValueError("unresolved_candidate_ids must be unique")
        if self.disposition == "recovered" and not self.outcomes:
            raise ValueError("Recovered fragment requires at least one outcome")
        if self.unresolved_candidate_ids and not self.unresolved_reason:
            raise ValueError("Unresolved candidates require a reason")
        if not self.unresolved_candidate_ids and self.unresolved_reason:
            raise ValueError("unresolved_reason requires an unresolved candidate")
        if self.disposition == "unresolved":
            if self.recovered_candidate_ids:
                raise ValueError("Unresolved fragment cannot claim recovered candidates")
            if self.experiments or self.outcomes:
                raise ValueError("Unresolved fragment cannot contain records")
            if not self.unresolved_reason:
                raise ValueError("Unresolved fragment requires a reason")
        return self


class MissingRecordVisionReferral(StrictModel):
    referral_version: Literal["missing-record-vision-referral-1.0.0"]
    paper_id: str
    route_ids: list[str] = Field(min_length=1)
    candidate_ids: list[str] = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    source_path: str
    page_number: int = Field(ge=1)
    figure_or_table: str
    reason: str


class MissingRecordVisionTask(StrictModel):
    task_version: Literal["missing-record-vision-task-1.0.0"]
    paper_id: str
    route_ids: list[str] = Field(min_length=1)
    candidate_ids: list[str] = Field(min_length=1)
    evidence: list[RepairEvidence] = Field(min_length=1, max_length=12)
    existing_formulation_ids: list[str]
    existing_experiment_ids: list[str]
    existing_outcome_ids: list[str]
    permitted_new_experiments: int = Field(ge=0, le=2)
    permitted_new_outcomes: int = Field(ge=1, le=8)
    source_result_sha256: str
    source_inventory_sha256: str
    source_pdf: str
    source_pdf_sha256: str
    page_number: int = Field(ge=1)
    figure_or_table: str
    crop_path: str
    crop_sha256: str
    crop_evidence_id: str
    task_checksum: str


class MissingRecordVisionResponse(StrictModel):
    fragment: MissingRecordFragment
    value_status: Literal[
        "exact_reported", "derived", "visually_estimated", "not_resolved"
    ]
    panel_or_table_cell: str | None
    visible_support: str = Field(min_length=1, max_length=500)
    derivation: str | None
    requires_human_review: bool

    @model_validator(mode="after")
    def enforce_visual_safety(self) -> "MissingRecordVisionResponse":
        if self.value_status == "visually_estimated":
            if not self.requires_human_review:
                raise ValueError("Visual estimates require human review")
            if self.fragment.disposition != "unresolved":
                raise ValueError("Visual estimates cannot be automatically recovered")
        if self.fragment.disposition == "recovered":
            if self.value_status not in {"exact_reported", "derived"}:
                raise ValueError("Recovered records require exact or derived support")
            if self.requires_human_review:
                raise ValueError("Human-review records cannot be auto-merged")
            if not self.panel_or_table_cell:
                raise ValueError("Recovered visual records require a cell/panel location")
        if self.value_status == "derived" and not self.derivation:
            raise ValueError("Derived records require a derivation")
        return self
