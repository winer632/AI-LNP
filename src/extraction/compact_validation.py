"""Normalize compact-extraction validation failures into repairable findings."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.extraction.compact_contracts import (
    CandidateSlotExtractionResponse,
    CompactExtractionResponse,
    ReportedField,
)


COLLECTIONS = {"formulations", "components", "experiments", "outcomes"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ValidationFinding(StrictModel):
    """One deterministic validation failure, with a field-level repair boundary."""

    finding_id: str
    code: str
    message: str
    location: list[str | int]
    record_collection: str | None = None
    record_index: int | None = Field(default=None, ge=0)
    field_name: str | None = None
    cited_evidence_ids: list[str] = Field(default_factory=list)
    repairable: bool


class ValidationReport(StrictModel):
    report_version: str = "compact-validation-1.0.0"
    paper_id: str
    status: Literal["valid", "invalid"]
    findings: list[ValidationFinding]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _finding_id(
    paper_id: str, code: str, location: list[str | int], message: str
) -> str:
    digest = hashlib.sha256(
        _canonical_json([paper_id, code, location, message]).encode("utf-8")
    ).hexdigest()[:16]
    return f"VF-{digest}"


def _evidence_ids(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "evidence_ids" and isinstance(child, list):
                found.extend(item for item in child if isinstance(item, str))
            else:
                found.extend(_evidence_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_evidence_ids(child))
    return list(dict.fromkeys(found))


def _record_at(candidate: dict[str, Any], location: list[str | int]) -> Any:
    if len(location) < 2:
        return {}
    collection, index = location[0], location[1]
    if collection not in COLLECTIONS or not isinstance(index, int):
        return {}
    rows = candidate.get(collection)
    if not isinstance(rows, list) or index >= len(rows):
        return {}
    return rows[index]


def _finding(
    *,
    paper_id: str,
    code: str,
    message: str,
    location: list[str | int],
    candidate: dict[str, Any],
) -> ValidationFinding:
    collection = location[0] if location and location[0] in COLLECTIONS else None
    index = (
        location[1]
        if collection and len(location) > 1 and isinstance(location[1], int)
        else None
    )
    field_name = (
        location[2]
        if index is not None and len(location) > 2 and isinstance(location[2], str)
        else None
    )
    repairable = collection is not None and index is not None and field_name is not None
    record = _record_at(candidate, location)
    evidence_scope = (
        record.get(field_name)
        if field_name is not None and isinstance(record, dict)
        else record
    )
    return ValidationFinding(
        finding_id=_finding_id(paper_id, code, location, message),
        code=code,
        message=message,
        location=location,
        record_collection=collection,
        record_index=index,
        field_name=field_name,
        cited_evidence_ids=_evidence_ids(evidence_scope),
        repairable=repairable,
    )


def _slot_coverage_findings(
    parsed: CompactExtractionResponse,
    candidate: dict[str, Any],
    *,
    paper_id: str,
    required_candidate_ids: Iterable[str],
    candidate_evidence_ids: Mapping[str, Iterable[str]] | None = None,
) -> list[ValidationFinding]:
    """Reject a response that leaves a supplied candidate slot unanswered.

    This is the first-call half of the rule the repair stage already enforces:
    every candidate ID that was sent must come back with a disposition, and no
    disposition may name an ID that was never sent.

    ``extracted`` is additionally checked against the returned records. Measured
    on GP-006, a response marked all four candidates covering Table S2's
    insertion-frequency row as ``extracted`` while returning no record carrying
    1.01, so an unverified self-report reintroduces exactly the silent omission
    the slots exist to prevent.
    """

    required = list(dict.fromkeys(required_candidate_ids))
    answered = parsed.candidate_disposition_ids()
    findings: list[ValidationFinding] = []

    for candidate_id in required:
        if candidate_id in answered:
            continue
        findings.append(
            _finding(
                paper_id=paper_id,
                code="missing_candidate_disposition",
                message=(
                    f"Candidate {candidate_id} was sent as an answerable slot but "
                    "the response returned no disposition for it; silence is not "
                    "a valid disposition"
                ),
                location=["candidate_dispositions", candidate_id],
                candidate=candidate,
            )
        )

    for candidate_id in sorted(answered - set(required)):
        findings.append(
            _finding(
                paper_id=paper_id,
                code="unknown_candidate_id",
                message=(
                    f"Response dispositions candidate {candidate_id}, which was "
                    "not supplied as a slot"
                ),
                location=["candidate_dispositions", candidate_id],
                candidate=candidate,
            )
        )

    if candidate_evidence_ids:
        cited = _evidence_ids(
            [row.model_dump(mode="json") for row in parsed.outcomes]
        )
        cited_set = set(cited)
        for disposition in getattr(parsed, "candidate_dispositions", []) or []:
            if disposition.disposition != "extracted":
                continue
            expected = set(candidate_evidence_ids.get(disposition.candidate_id, ()))
            if not expected or expected & cited_set:
                continue
            findings.append(
                _finding(
                    paper_id=paper_id,
                    code="unsupported_extracted_disposition",
                    message=(
                        f"Candidate {disposition.candidate_id} is marked extracted "
                        "but none of its evidence is cited by any returned outcome"
                    ),
                    location=["candidate_dispositions", disposition.candidate_id],
                    candidate=candidate,
                )
            )
    return findings


def validate_candidate(
    candidate_text: str,
    *,
    paper_id: str,
    allowed_evidence_ids: set[str],
    required_candidate_ids: Iterable[str] | None = None,
    candidate_evidence_ids: Mapping[str, Iterable[str]] | None = None,
) -> tuple[CompactExtractionResponse | None, ValidationReport]:
    """Validate one first-call candidate and return machine-readable findings.

    ``required_candidate_ids`` switches on candidate-slot enforcement (P3): the
    response is parsed against :class:`CandidateSlotExtractionResponse` and must
    account for every ID that was sent. Leave it ``None`` -- the default -- for
    the baseline contract and the baseline findings.
    """
    response_model: type[CompactExtractionResponse] = (
        CompactExtractionResponse
        if required_candidate_ids is None
        else CandidateSlotExtractionResponse
    )
    try:
        candidate = json.loads(candidate_text)
    except json.JSONDecodeError as error:
        finding = _finding(
            paper_id=paper_id,
            code="invalid_json",
            message=str(error),
            location=[],
            candidate={},
        )
        return None, ValidationReport(
            paper_id=paper_id, status="invalid", findings=[finding]
        )

    try:
        parsed = response_model.model_validate(candidate)
    except ValidationError as error:
        findings = [
            _finding(
                paper_id=paper_id,
                code=f"pydantic.{item['type']}",
                message=item["msg"],
                location=list(item["loc"]),
                candidate=candidate,
            )
            for item in error.errors(include_url=False)
        ]
        return None, ValidationReport(
            paper_id=paper_id, status="invalid", findings=findings
        )

    findings: list[ValidationFinding] = []
    if parsed.paper_id != paper_id:
        findings.append(
            _finding(
                paper_id=paper_id,
                code="paper_id_mismatch",
                message=(
                    f"Response paper_id {parsed.paper_id!r} does not match "
                    f"requested {paper_id!r}"
                ),
                location=["paper_id"],
                candidate=candidate,
            )
        )

    unknown_eligibility = sorted(
        set(parsed.eligibility.evidence_ids) - allowed_evidence_ids
    )
    if unknown_eligibility:
        findings.append(
            _finding(
                paper_id=paper_id,
                code="unknown_evidence_id",
                message=f"Unknown evidence IDs: {unknown_eligibility}",
                location=["eligibility", "evidence_ids"],
                candidate=candidate,
            )
        )

    for collection in COLLECTIONS:
        for index, record in enumerate(getattr(parsed, collection)):
            for field_name in record.__class__.model_fields:
                value = getattr(record, field_name)
                if not isinstance(value, ReportedField):
                    continue
                unknown = sorted(set(value.evidence_ids) - allowed_evidence_ids)
                if unknown:
                    findings.append(
                        _finding(
                            paper_id=paper_id,
                            code="unknown_evidence_id",
                            message=f"Unknown evidence IDs: {unknown}",
                            location=[collection, index, field_name],
                            candidate=candidate,
                        )
                    )

    if required_candidate_ids is not None:
        findings.extend(
            _slot_coverage_findings(
                parsed,
                candidate,
                paper_id=paper_id,
                required_candidate_ids=required_candidate_ids,
                candidate_evidence_ids=candidate_evidence_ids,
            )
        )

    report = ValidationReport(
        paper_id=paper_id,
        status="valid" if not findings else "invalid",
        findings=findings,
    )
    return (parsed if not findings else None), report
