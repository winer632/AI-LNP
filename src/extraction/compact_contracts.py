"""Compact, evidence-ID-only extraction contract for the v7 pipeline.

The response intentionally contains no evidence quotations. Evidence text and
coordinates remain in the local evidence packet and are joined by
``evidence_ids`` after the model response is validated.

Two response shapes live here:

``CompactExtractionResponse``
    The frozen baseline contract. Its JSON Schema is exported to
    ``docs/extraction/schemas/compact_v1/`` and its checksum is part of the
    request fingerprint, so this class must not gain or lose fields.

``CandidateSlotExtractionResponse``
    The same contract plus ``candidate_dispositions``, used only when the
    ``candidate_slot_enforcement`` (P3) flag is on. Keeping it a subclass is
    what lets the flag-off request stay byte-identical: the baseline schema,
    prompt, and fingerprint are untouched, and a slot-aware response still
    validates as -- and is an instance of -- the baseline contract.

``EndpointDefinedExtractionResponse`` / ``EndpointDefinedSlotExtractionResponse``
    Whichever of the two above applies, with ``endpoint`` given the definition
    it has never had. Used only when the ``endpoint_definition`` flag is on.
    Same discipline once more, and here it is the whole reason for the shape:
    a description added to :class:`OutcomeRecord` in place would change the
    exported baseline schema, and therefore the schema checksum inside every
    request fingerprint, for runs that never asked for it. Call
    :func:`active_response_contract` rather than importing one of these.
"""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config_flags import is_enabled


T = TypeVar("T")


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReportedField(StrictContract, Generic[T]):
    """A directly reported value or an explicit evidence-based abstention."""

    value: T | None
    status: Literal["reported", "missing"]
    evidence_ids: list[str] = Field(default_factory=list)
    missing_reason: str | None

    @model_validator(mode="after")
    def require_evidence_or_missing(self) -> "ReportedField[T]":
        if self.status == "reported":
            if self.value is None:
                raise ValueError("reported fields must contain a value")
            if not self.evidence_ids:
                raise ValueError("reported fields require at least one evidence_id")
            if self.missing_reason is not None:
                raise ValueError("reported fields cannot contain a missing_reason")
        else:
            if self.value is not None:
                raise ValueError("missing fields must contain a null value")
            if self.evidence_ids:
                raise ValueError("missing fields cannot cite evidence as value support")
            if not self.missing_reason:
                raise ValueError("missing fields require a missing_reason")
        return self


TextField = ReportedField[str]
NumberField = ReportedField[float]


EligibilityReason = Literal[
    "ORIGINAL_EXPERIMENT",
    "IDENTIFIABLE_LNP",
    "SUPPORTED_PAYLOAD",
    "TARGET_CELL_EVIDENCE",
    "USABLE_FORMULATION_OUTCOME_LINKAGE",
    "NOT_ORIGINAL_RESEARCH",
    "NOT_ELIGIBLE_LNP",
    "UNSUPPORTED_PAYLOAD",
    "NO_TARGET_CELL_EVIDENCE",
    "NO_FORMULATION_OUTCOME_LINKAGE",
    "DUPLICATE_SCIENTIFIC_REPORT",
    "RETRACTED_OR_INVALID",
    "FULL_TEXT_REQUIRED",
    "PUBLICATION_TYPE_AMBIGUOUS",
    "LNP_IDENTITY_AMBIGUOUS",
    "PAYLOAD_AMBIGUOUS",
    "TARGET_CELL_AMBIGUOUS",
    "FORMULATION_INCOMPLETE",
    "OUTCOME_LINKAGE_AMBIGUOUS",
    "CHEMISTRY_AMBIGUOUS",
]

ELIGIBLE_REASONS = {
    "ORIGINAL_EXPERIMENT",
    "IDENTIFIABLE_LNP",
    "SUPPORTED_PAYLOAD",
    "TARGET_CELL_EVIDENCE",
    "USABLE_FORMULATION_OUTCOME_LINKAGE",
}
INELIGIBLE_REASONS = {
    "NOT_ORIGINAL_RESEARCH",
    "NOT_ELIGIBLE_LNP",
    "UNSUPPORTED_PAYLOAD",
    "NO_TARGET_CELL_EVIDENCE",
    "NO_FORMULATION_OUTCOME_LINKAGE",
    "DUPLICATE_SCIENTIFIC_REPORT",
    "RETRACTED_OR_INVALID",
}
UNCERTAIN_REASONS = {
    "FULL_TEXT_REQUIRED",
    "PUBLICATION_TYPE_AMBIGUOUS",
    "LNP_IDENTITY_AMBIGUOUS",
    "PAYLOAD_AMBIGUOUS",
    "TARGET_CELL_AMBIGUOUS",
    "FORMULATION_INCOMPLETE",
    "OUTCOME_LINKAGE_AMBIGUOUS",
    "CHEMISTRY_AMBIGUOUS",
}


class EligibilityRecord(StrictContract):
    """Evidence-grounded final paper disposition for the extraction task."""

    decision: Literal["eligible", "ineligible", "uncertain"]
    reason_codes: list[EligibilityReason] = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_decision_reasons(self) -> "EligibilityRecord":
        reasons = set(self.reason_codes)
        if len(reasons) != len(self.reason_codes):
            raise ValueError("eligibility reason_codes must be unique")
        if self.decision == "eligible":
            if not ELIGIBLE_REASONS <= reasons:
                raise ValueError(
                    "eligible requires all five positive eligibility reason codes"
                )
            if not self.evidence_ids:
                raise ValueError("eligible requires at least one evidence_id")
        elif self.decision == "ineligible":
            if not reasons & INELIGIBLE_REASONS:
                raise ValueError("ineligible requires an exclusion reason code")
            if not self.evidence_ids:
                raise ValueError("ineligible requires at least one evidence_id")
        elif not reasons & UNCERTAIN_REASONS:
            raise ValueError("uncertain requires a manual-review reason code")
        return self


class FormulationRecord(StrictContract):
    formulation_id: str
    formulation_name: TextField
    composition: TextField
    composition_basis: TextField
    np_ratio: NumberField


class ComponentRecord(StrictContract):
    component_id: str
    formulation_id: str
    identity: TextField
    role: ReportedField[
        Literal[
            "ionizable_lipid",
            "helper_lipid",
            "cholesterol",
            "peg_lipid",
            "targeting_ligand",
            "sort_lipid",
            "other",
        ]
    ]
    amount: NumberField
    amount_unit: TextField


class ExperimentRecord(StrictContract):
    experiment_id: str
    formulation_id: str
    payload_type: TextField
    payload_name: TextField
    encoded_product: TextField
    molecular_target: TextField
    delivery_recipient_cell: TextField
    therapeutic_target_cell: TextField
    tissue_or_organ: TextField
    species: TextField
    disease_model: TextField
    experimental_context: ReportedField[
        Literal["in_vitro", "ex_vivo", "in_vivo"]
    ]
    dose: NumberField
    dose_unit: TextField
    route: TextField
    timepoint: NumberField
    timepoint_unit: TextField


class OutcomeRecord(StrictContract):
    outcome_id: str
    experiment_id: str
    assay: TextField
    endpoint: TextField
    comparator: TextField
    outcome_value: NumberField
    outcome_unit: TextField
    qualitative_outcome: TextField


ENDPOINT_DEFINITION_FLAG = "endpoint_definition"

# `endpoint` is the only scientific field in this contract that has never been
# defined anywhere. The class above says `endpoint: TextField` and stops, and
# the extraction prompt does not mention the field at all, so what belongs in
# it has been left entirely to the model's own reading of the word.
#
# The definition below is taken from how the field is actually written on the
# annotation side, where it has a settled convention: an endpoint names the
# thing measured *and* the population it was measured in, and it names both
# even though the experiment row it hangs off already carries a
# recipient-population column of its own. A record that carries only half of
# that has dropped a fact its author had, so this asks for both halves.
#
# Deliberately general: it names no population, no cell type and no marker,
# because a definition that listed them would stop being a definition of the
# field and start being a list of expected answers. The second half is the
# part that keeps it honest -- a paper that states no population must not have
# one supplied for it.
ENDPOINT_DEFINITION = (
    "What this record measures, in the paper's own terms: the quantity, "
    "signal or event that was measured, together with the population it was "
    "measured in whenever the paper states one. Name both parts here even "
    "when the experiment this outcome belongs to also names that population, "
    "so that the endpoint identifies the measurement on its own. When the "
    "paper reports the measurement without saying what population it was made "
    "in, name the measurement alone; do not supply a population the cited "
    "evidence does not give."
)


class DefinedEndpointOutcomeRecord(OutcomeRecord):
    """:class:`OutcomeRecord` with ``endpoint`` actually defined.

    Sent only when ``endpoint_definition`` is enabled. It adds, removes and
    renames nothing: the sole difference from the baseline record is that the
    exported schema for ``endpoint`` carries a ``description``. A response to
    this contract therefore still validates as -- and is an instance of -- the
    baseline one, so nothing downstream needs to know which was sent.
    """

    endpoint: TextField = Field(description=ENDPOINT_DEFINITION)


class CompactExtractionResponse(StrictContract):
    """One paper response joined to a local evidence packet after validation."""

    contract_version: Literal["compact-1.1.0"]
    paper_id: str
    eligibility: EligibilityRecord
    formulations: list[FormulationRecord]
    components: list[ComponentRecord]
    experiments: list[ExperimentRecord]
    outcomes: list[OutcomeRecord]
    unresolved_items: list[str]

    @model_validator(mode="after")
    def validate_record_links(self) -> "CompactExtractionResponse":
        records = [
            *self.formulations,
            *self.components,
            *self.experiments,
            *self.outcomes,
        ]
        if self.eligibility.decision != "eligible" and records:
            raise ValueError(
                "ineligible or uncertain papers must return empty extraction lists"
            )
        if self.eligibility.decision == "eligible" and not (
            self.formulations and self.experiments and self.outcomes
        ):
            raise ValueError(
                "eligible papers require formulation, experiment, and outcome records"
            )
        formulation_ids = {item.formulation_id for item in self.formulations}
        experiment_ids = {item.experiment_id for item in self.experiments}

        if len(formulation_ids) != len(self.formulations):
            raise ValueError("formulation_id values must be unique")
        if len(experiment_ids) != len(self.experiments):
            raise ValueError("experiment_id values must be unique")
        if any(item.formulation_id not in formulation_ids for item in self.components):
            raise ValueError("component references an unknown formulation_id")
        if any(item.formulation_id not in formulation_ids for item in self.experiments):
            raise ValueError("experiment references an unknown formulation_id")
        if any(item.experiment_id not in experiment_ids for item in self.outcomes):
            raise ValueError("outcome references an unknown experiment_id")
        return self

    def candidate_disposition_ids(self) -> set[str]:
        """Candidate IDs this response accounts for; empty on the baseline contract."""

        return set()

    def validate_evidence_ids(self, allowed_evidence_ids: set[str]) -> None:
        """Reject model citations that do not exist in the local packet."""

        unknown_eligibility = (
            set(self.eligibility.evidence_ids) - allowed_evidence_ids
        )
        if unknown_eligibility:
            raise ValueError(
                "EligibilityRecord references unknown evidence IDs: "
                f"{sorted(unknown_eligibility)}"
            )

        for record in [
            *self.formulations,
            *self.components,
            *self.experiments,
            *self.outcomes,
        ]:
            for field_name in record.__class__.model_fields:
                value = getattr(record, field_name)
                if not isinstance(value, ReportedField):
                    continue
                unknown = set(value.evidence_ids) - allowed_evidence_ids
                if unknown:
                    raise ValueError(
                        f"{record.__class__.__name__}.{field_name} references "
                        f"unknown evidence IDs: {sorted(unknown)}"
                    )


CandidateDispositionCode = Literal["extracted", "not_an_outcome", "unresolved"]


class CandidateDisposition(StrictContract):
    """How one locally enumerated outcome candidate was accounted for.

    Mirrors the repair-stage rule in
    :mod:`src.extraction.missing_record_contracts`: a response must account for
    every candidate ID, and every disposition other than ``extracted`` has to
    say why. Silence is not a valid disposition.
    """

    candidate_id: str
    disposition: CandidateDispositionCode
    reason: str | None

    @model_validator(mode="after")
    def require_reason_unless_extracted(self) -> "CandidateDisposition":
        if self.disposition != "extracted" and not (self.reason or "").strip():
            raise ValueError(
                f"candidate {self.candidate_id} is {self.disposition} and "
                "therefore requires a reason"
            )
        return self


class CandidateSlotExtractionResponse(CompactExtractionResponse):
    """Baseline response plus an explicit answer for every candidate slot.

    Sent only when ``candidate_slot_enforcement`` is enabled. ``result.json``
    is still written from the inherited baseline fields, so every downstream
    consumer of the compact contract keeps working unchanged.
    """

    candidate_dispositions: list[CandidateDisposition]

    @model_validator(mode="after")
    def validate_candidate_dispositions(self) -> "CandidateSlotExtractionResponse":
        seen = [row.candidate_id for row in self.candidate_dispositions]
        if len(set(seen)) != len(seen):
            raise ValueError("candidate_dispositions must not repeat a candidate_id")
        return self

    def candidate_disposition_ids(self) -> set[str]:
        return {row.candidate_id for row in self.candidate_dispositions}


class EndpointDefinedExtractionResponse(CompactExtractionResponse):
    """The baseline response whose outcome records define ``endpoint``."""

    outcomes: list[DefinedEndpointOutcomeRecord]


class EndpointDefinedSlotExtractionResponse(CandidateSlotExtractionResponse):
    """The slot-aware response whose outcome records define ``endpoint``."""

    outcomes: list[DefinedEndpointOutcomeRecord]


def active_response_contract(
    candidate_slot_enforcement: bool = False,
    *,
    endpoint_definition: bool | None = None,
) -> type[CompactExtractionResponse]:
    """Pick the response contract for this request from the flags that amend it.

    Both amendments are subclasses rather than edits, for the same reason the
    prompt amendments are new versions rather than edits: with every flag off
    the request is byte-identical to what shipped, down to the schema checksum
    in its fingerprint, and the exported baseline schema in
    ``docs/extraction/schemas/compact_v1/`` -- whose sha256 is pinned in
    ``config/extraction/compact_route_v1.yaml`` -- does not move.

    An explicit argument beats the flag in both directions so a test can
    exercise either side without touching the environment.
    """

    defined = (
        is_enabled(ENDPOINT_DEFINITION_FLAG)
        if endpoint_definition is None
        else endpoint_definition
    )
    if candidate_slot_enforcement:
        return (
            EndpointDefinedSlotExtractionResponse
            if defined
            else CandidateSlotExtractionResponse
        )
    return (
        EndpointDefinedExtractionResponse if defined else CompactExtractionResponse
    )
