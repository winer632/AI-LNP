"""Keep the records a contract-invalid extraction response got right.

Why this exists
---------------
``compact_validation.validate_candidate`` is all-or-nothing at the document
level: it returns a parsed response only when the finding list is empty, so one
hallucinated evidence id anywhere -- in ``eligibility``, in a component's
``identity``, in a formulation's ``composition`` -- discards every outcome
record in the same response, including records that cited nothing but real
evidence.

That is the right *default*, because the contract exists to stop fabricated
citations entering the results. But it throws away more than the fabrication:
the unit whose citations went bad is a field of one record, and the contract
punishes the document.

What this module does, and what it refuses to do
------------------------------------------------
It re-derives the answer one record at a time. A record is kept only when

1. it validates on its own against its own contract model, and
2. every evidence id **it itself** cites is present in the packet.

Rule 2 is the whole point, and it is not weaker than the document rule -- it is
the same rule with the blast radius reduced to the record that broke it. A
record citing an id the packet does not contain is rejected here exactly as it
is rejected there. What changes is that its neighbours no longer die with it.

Nothing is repaired. A field whose citation is unknown is never "fixed" by
dropping the bad id and keeping the value: the value's only support was that
citation, so editing it would manufacture an unsupported claim, which is the
failure mode the contract exists to prevent. The record is dropped whole, and
:class:`SalvageReport` records which record, at which location, and why.

The output is deliberately *not* a ``CompactExtractionResponse``. It carries its
own ``contract_version`` (``compact-salvage-1.0.0``), it may hold a null
``eligibility``, and it embeds the drop list, so no reader can mistake a
partial result for a response that validated whole.

Flag
----
Everything here is gated by ``record_level_salvage``, which is off by default.
The gate is inside :func:`salvage_response`, so a caller cannot integrate this
and accidentally bypass it: with the flag off the function returns a report with
status ``disabled``, keeps nothing, and writes nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config_flags import is_enabled
from src.extraction.compact_contracts import (
    ComponentRecord,
    EligibilityRecord,
    ExperimentRecord,
    FormulationRecord,
    OutcomeRecord,
    ReportedField,
)
from src.extraction.compact_validation import ValidationReport


ROOT = Path(__file__).resolve().parents[2]
SALVAGE_FLAG = "record_level_salvage"
SALVAGE_CONTRACT_VERSION = "compact-salvage-1.0.0"
SALVAGE_REPORT_VERSION = "record-salvage-1.0.0"

# Ordered so a salvaged document lists its collections the way the contract
# does, and so `experiments` is resolved before `outcomes` when dangling
# references are noted.
RECORD_MODELS: dict[str, type[BaseModel]] = {
    "formulations": FormulationRecord,
    "components": ComponentRecord,
    "experiments": ExperimentRecord,
    "outcomes": OutcomeRecord,
}
ID_FIELDS = {
    "formulations": "formulation_id",
    "components": "component_id",
    "experiments": "experiment_id",
    "outcomes": "outcome_id",
}
# Everything the salvage knows how to read. A response carrying anything else --
# `candidate_dispositions` from a slot-enforced run, a field a later contract
# added -- has that content recorded as dropped rather than silently ignored,
# because "the salvage did not look at this" and "this was not there" are
# different facts and only one of them is true.
KNOWN_TOP_LEVEL = {
    "contract_version",
    "paper_id",
    "eligibility",
    "unresolved_items",
    *RECORD_MODELS,
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DroppedItem(StrictModel):
    """One thing the salvage refused to keep, and the reason it refused."""

    location: list[str | int]
    collection: str | None = None
    index: int | None = Field(default=None, ge=0)
    record_id: str | None = None
    reason: str
    detail: str
    unknown_evidence_ids: list[str] = Field(default_factory=list)


class DanglingReference(StrictModel):
    """A kept record whose parent record did not survive the salvage.

    Recorded rather than resolved. The id is left exactly as the model wrote
    it -- re-pointing it at a surviving record would be a guess -- so a reader
    can see that the join is broken instead of finding a join that quietly
    goes somewhere else.
    """

    location: list[str | int]
    record_id: str | None = None
    field_name: str
    references: str


class SalvageReport(StrictModel):
    salvage_version: str = SALVAGE_REPORT_VERSION
    paper_id: str
    flag: str = SALVAGE_FLAG
    status: Literal["disabled", "not_attempted", "unsalvageable", "salvaged"]
    source_validation_status: str | None = None
    source_finding_codes: list[str] = Field(default_factory=list)
    kept: dict[str, int] = Field(default_factory=dict)
    rejected: dict[str, int] = Field(default_factory=dict)
    eligibility_kept: bool = False
    dropped: list[DroppedItem] = Field(default_factory=list)
    dangling_references: list[DanglingReference] = Field(default_factory=list)
    note: str | None = None

    @property
    def kept_records(self) -> int:
        return sum(self.kept.values())

    @property
    def rejected_records(self) -> int:
        return sum(self.rejected.values())


def _record_evidence_ids(record: BaseModel) -> list[str]:
    """Every evidence id this record cites in its own fields.

    Only the record's own ``ReportedField`` values are read, which is what makes
    the check per-record: a sibling's bad citation is not this record's
    citation, and this record's bad citation is not excused by a clean sibling.
    """
    found: list[str] = []
    for field_name in type(record).model_fields:
        value = getattr(record, field_name)
        if isinstance(value, ReportedField):
            found.extend(value.evidence_ids)
    return list(dict.fromkeys(found))


def _validation_messages(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
        for item in error.errors(include_url=False)
    )


def _salvage_eligibility(
    payload: Any,
    allowed_evidence_ids: set[str],
    dropped: list[DroppedItem],
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        dropped.append(
            DroppedItem(
                location=["eligibility"],
                reason="not_an_object",
                detail=f"eligibility is {type(payload).__name__}, not an object",
            )
        )
        return None
    try:
        record = EligibilityRecord.model_validate(payload)
    except ValidationError as error:
        dropped.append(
            DroppedItem(
                location=["eligibility"],
                reason="contract_invalid",
                detail=_validation_messages(error),
            )
        )
        return None
    unknown = sorted(set(record.evidence_ids) - allowed_evidence_ids)
    if unknown:
        dropped.append(
            DroppedItem(
                location=["eligibility", "evidence_ids"],
                reason="unknown_evidence_id",
                detail=(
                    "eligibility cites evidence the packet does not contain; "
                    "the decision is recorded as unusable rather than "
                    "re-cited"
                ),
                unknown_evidence_ids=unknown,
            )
        )
        return None
    return record.model_dump(mode="json")


def _salvage_collection(
    collection: str,
    payload: Any,
    allowed_evidence_ids: set[str],
    dropped: list[DroppedItem],
) -> tuple[list[dict[str, Any]], int]:
    model = RECORD_MODELS[collection]
    id_field = ID_FIELDS[collection]
    kept: list[dict[str, Any]] = []
    rejected = 0
    if payload is None:
        return kept, rejected
    if not isinstance(payload, list):
        dropped.append(
            DroppedItem(
                location=[collection],
                collection=collection,
                reason="not_a_list",
                detail=f"{collection} is {type(payload).__name__}, not a list",
            )
        )
        return kept, rejected
    for index, row in enumerate(payload):
        record_id = row.get(id_field) if isinstance(row, dict) else None
        record_id = record_id if isinstance(record_id, str) else None
        try:
            record = model.model_validate(row)
        except ValidationError as error:
            rejected += 1
            dropped.append(
                DroppedItem(
                    location=[collection, index],
                    collection=collection,
                    index=index,
                    record_id=record_id,
                    reason="contract_invalid",
                    detail=_validation_messages(error),
                )
            )
            continue
        unknown = sorted(set(_record_evidence_ids(record)) - allowed_evidence_ids)
        if unknown:
            rejected += 1
            dropped.append(
                DroppedItem(
                    location=[collection, index],
                    collection=collection,
                    index=index,
                    record_id=record_id,
                    reason="unknown_evidence_id",
                    detail=(
                        "record cites evidence the packet does not contain; "
                        "dropped whole rather than re-cited"
                    ),
                    unknown_evidence_ids=unknown,
                )
            )
            continue
        kept.append(record.model_dump(mode="json"))
    return kept, rejected


def _note_dangling(
    records: list[dict[str, Any]],
    collection: str,
    field_name: str,
    available: set[str],
    dangling: list[DanglingReference],
) -> None:
    for index, row in enumerate(records):
        reference = row.get(field_name)
        if isinstance(reference, str) and reference not in available:
            dangling.append(
                DanglingReference(
                    location=[collection, index],
                    record_id=row.get(ID_FIELDS[collection]),
                    field_name=field_name,
                    references=reference,
                )
            )


def salvage_response(
    candidate_text: str,
    *,
    paper_id: str,
    allowed_evidence_ids: Iterable[str],
    validation: ValidationReport | None = None,
) -> tuple[dict[str, Any] | None, SalvageReport]:
    """Salvage the records a contract-invalid response got right.

    Returns ``(document, report)``. ``document`` is ``None`` whenever nothing
    survived, whenever the flag is off, and whenever the response is not
    salvageable at all -- an unparseable body, or one that answers about a
    different paper. The report is always returned, so a caller can record why
    it got nothing.
    """
    allowed = set(allowed_evidence_ids)
    source_status = validation.status if validation is not None else None
    source_codes = (
        sorted({finding.code for finding in validation.findings})
        if validation is not None
        else []
    )

    if not is_enabled(SALVAGE_FLAG):
        return None, SalvageReport(
            paper_id=paper_id,
            status="disabled",
            source_validation_status=source_status,
            source_finding_codes=source_codes,
            note=(
                f"{SALVAGE_FLAG} is off; the response was discarded whole, "
                "which is the unchanged behaviour"
            ),
        )

    if validation is not None and validation.status == "valid":
        return None, SalvageReport(
            paper_id=paper_id,
            status="not_attempted",
            source_validation_status=source_status,
            note="the response validated whole; there is nothing to salvage",
        )

    dropped: list[DroppedItem] = []
    try:
        candidate = json.loads(candidate_text)
    except json.JSONDecodeError as error:
        return None, SalvageReport(
            paper_id=paper_id,
            status="unsalvageable",
            source_validation_status=source_status,
            source_finding_codes=source_codes,
            dropped=[
                DroppedItem(
                    location=[],
                    reason="invalid_json",
                    detail=str(error),
                )
            ],
            note="the response body is not JSON, so it has no records to read",
        )

    if not isinstance(candidate, dict):
        return None, SalvageReport(
            paper_id=paper_id,
            status="unsalvageable",
            source_validation_status=source_status,
            source_finding_codes=source_codes,
            dropped=[
                DroppedItem(
                    location=[],
                    reason="not_an_object",
                    detail=f"response body is {type(candidate).__name__}",
                )
            ],
        )

    # A response that answers about another paper is refused whole rather than
    # re-labelled. Its records may be perfectly well formed and still be about
    # the wrong document, and the only way to keep them would be to overwrite
    # the paper_id the model wrote -- a guess, and the most consequential one
    # available here.
    body_paper_id = candidate.get("paper_id")
    if body_paper_id != paper_id:
        return None, SalvageReport(
            paper_id=paper_id,
            status="unsalvageable",
            source_validation_status=source_status,
            source_finding_codes=source_codes,
            dropped=[
                DroppedItem(
                    location=["paper_id"],
                    reason="paper_id_mismatch",
                    detail=(
                        f"response reports paper_id {body_paper_id!r}, requested "
                        f"{paper_id!r}; refused whole rather than re-labelled"
                    ),
                )
            ],
        )

    declared_version = candidate.get("contract_version")
    if declared_version != "compact-1.1.0":
        dropped.append(
            DroppedItem(
                location=["contract_version"],
                reason="contract_version_mismatch",
                detail=(
                    f"response declares {declared_version!r}, not 'compact-1.1.0'; "
                    "records are still checked against the compact models, so "
                    "anything shaped differently is dropped below"
                ),
            )
        )
    for key in sorted(set(candidate) - KNOWN_TOP_LEVEL):
        dropped.append(
            DroppedItem(
                location=[key],
                reason="unhandled_field",
                detail=(
                    f"{key!r} is not part of the compact record contract, so the "
                    "salvage carries no rule for validating it and does not keep it"
                ),
            )
        )

    eligibility = _salvage_eligibility(
        candidate.get("eligibility"), allowed, dropped
    )

    kept: dict[str, list[dict[str, Any]]] = {}
    kept_counts: dict[str, int] = {}
    rejected_counts: dict[str, int] = {}
    for collection in RECORD_MODELS:
        rows, rejected = _salvage_collection(
            collection, candidate.get(collection), allowed, dropped
        )
        kept[collection] = rows
        kept_counts[collection] = len(rows)
        rejected_counts[collection] = rejected

    unresolved_items: list[str] = []
    raw_unresolved = candidate.get("unresolved_items")
    if isinstance(raw_unresolved, list):
        for index, item in enumerate(raw_unresolved):
            if isinstance(item, str):
                unresolved_items.append(item)
            else:
                dropped.append(
                    DroppedItem(
                        location=["unresolved_items", index],
                        reason="contract_invalid",
                        detail=f"expected a string, got {type(item).__name__}",
                    )
                )
    elif raw_unresolved is not None:
        dropped.append(
            DroppedItem(
                location=["unresolved_items"],
                reason="not_a_list",
                detail=f"unresolved_items is {type(raw_unresolved).__name__}",
            )
        )

    dangling: list[DanglingReference] = []
    formulation_ids = {row["formulation_id"] for row in kept["formulations"]}
    experiment_ids = {row["experiment_id"] for row in kept["experiments"]}
    _note_dangling(
        kept["components"], "components", "formulation_id", formulation_ids, dangling
    )
    _note_dangling(
        kept["experiments"], "experiments", "formulation_id", formulation_ids, dangling
    )
    _note_dangling(
        kept["outcomes"], "outcomes", "experiment_id", experiment_ids, dangling
    )

    report = SalvageReport(
        paper_id=paper_id,
        status="salvaged" if sum(kept_counts.values()) else "unsalvageable",
        source_validation_status=source_status,
        source_finding_codes=source_codes,
        kept=kept_counts,
        rejected=rejected_counts,
        eligibility_kept=eligibility is not None,
        dropped=dropped,
        dangling_references=dangling,
        # "nothing was offered" and "nothing survived" are different verdicts
        # and only one of them is about this module.
        note=(
            None
            if sum(kept_counts.values())
            else (
                "no record survived per-record validation"
                if sum(rejected_counts.values())
                else "the response returned no records to salvage"
            )
        ),
    )
    if not sum(kept_counts.values()):
        return None, report

    document = {
        "contract_version": SALVAGE_CONTRACT_VERSION,
        "paper_id": paper_id,
        # Embedded, not merely written beside the document: a reader holding
        # only this file must be able to see that it is partial and what is
        # missing from it.
        "salvage": report.model_dump(mode="json"),
        "eligibility": eligibility,
        **kept,
        "unresolved_items": unresolved_items,
    }
    return document, report


# --------------------------------------------------------------------------- #
# Retrospective salvage over completed run directories
# --------------------------------------------------------------------------- #


def _evidence_ids_of(payload: dict[str, Any]) -> set[str]:
    return {
        row["evidence_id"]
        for row in payload.get("evidence") or []
        if isinstance(row, dict) and isinstance(row.get("evidence_id"), str)
    }


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _resolve_packet(
    request: dict[str, Any], packet_root: Path | None
) -> tuple[set[str], str | None, str]:
    """Return ``(allowed ids, packet checksum, where it came from)``.

    Several request shapes exist on disk. The paid path embeds the whole packet
    under ``packet``, which is better evidence than any path because it cannot
    have drifted. Otherwise the packet is a file: the caller's override first,
    then the ``packet_root`` the request recorded -- which on older runs is an
    absolute path into a working directory that no longer exists -- then the
    view map, imported lazily because ``run_codex_one_call`` imports this
    module. Candidates that are not on disk are skipped rather than raising, so
    a stale recorded path degrades to the packet root the same view resolves to
    today instead of making the run unsalvageable.
    """
    paper_id = request["paper_id"]
    embedded = request.get("packet")
    if packet_root is None and isinstance(embedded, dict):
        return (
            _evidence_ids_of(embedded),
            embedded.get("packet_checksum"),
            "request.json:packet",
        )

    from src.extraction.run_codex_one_call import PACKET_ROOT_BY_VIEW

    candidates: list[Path] = []
    if packet_root is not None:
        candidates.append(_absolute(packet_root) / f"{paper_id}.json")
    if isinstance(request.get("packet_root"), str):
        candidates.append(
            _absolute(Path(request["packet_root"])) / f"{paper_id}.json"
        )
    if isinstance(embedded, str):
        candidates.append(_absolute(Path(embedded)))
    view = request.get("evidence_view")
    if view in PACKET_ROOT_BY_VIEW:
        candidates.append(PACKET_ROOT_BY_VIEW[view] / f"{paper_id}.json")

    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _evidence_ids_of(payload), payload.get("packet_checksum"), str(path)
    raise ValueError(
        f"{paper_id}: no packet on disk among {[str(p) for p in candidates]}; "
        "pass --packet-root"
    )


def _response_text(payload: dict[str, Any]) -> str:
    """The model's answer text, from either harness's ``response.json``.

    The codex harness writes ``{"text": "..."}``; the paid path writes the whole
    Responses object, whose answer is the ``output_text`` parts of its message
    items. Both are read here so a salvage can be run over any committed run
    without the caller having to know which harness produced it.
    """
    text = payload.get("text")
    if isinstance(text, str):
        return text
    parts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                parts.append(content["text"])
    if not parts:
        raise ValueError("response.json carries no answer text")
    return "".join(parts)


def salvage_run_dir(
    run_dir: Path,
    *,
    packet_root: Path | None = None,
) -> tuple[dict[str, Any] | None, SalvageReport, dict[str, Any]]:
    """Re-run validation over one completed run directory and salvage it.

    The packet is the one ``request.json`` names, read as it exists on disk. A
    packet that has since been rebuilt is reported as ``packet_drift`` rather
    than silently accepted: the ids a response could legally cite are a
    property of the packet it was sent, and if that has moved, the reader has
    to know the check was run against different bytes.
    """
    from src.extraction.compact_validation import validate_candidate

    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    paper_id = request["paper_id"]
    allowed, packet_checksum, packet_source = _resolve_packet(request, packet_root)
    response = json.loads((run_dir / "response.json").read_text(encoding="utf-8"))
    answer = _response_text(response)

    # A slot-enforced run answered a different contract, so it is re-validated
    # against that one; validating it as the baseline would reject the whole
    # response for carrying candidate_dispositions, which is a fact about this
    # reader rather than about the response.
    slots = request.get("candidate_slots") if request.get(
        "candidate_slot_enforcement"
    ) else None
    required_candidate_ids = (
        [str(slot["candidate_id"]) for slot in slots] if slots else None
    )
    _, validation = validate_candidate(
        answer,
        paper_id=paper_id,
        allowed_evidence_ids=allowed,
        required_candidate_ids=required_candidate_ids,
    )
    document, report = salvage_response(
        answer,
        paper_id=paper_id,
        allowed_evidence_ids=allowed,
        validation=validation,
    )
    context = {
        "run_dir": str(run_dir),
        "paper_id": paper_id,
        "packet": packet_source,
        "packet_drift": packet_checksum != request.get("packet_checksum"),
        "validation_status": validation.status,
        "validation_findings": [
            {"code": finding.code, "location": finding.location}
            for finding in validation.findings
        ],
    }
    return document, report, context


def build(
    run_roots: list[Path],
    *,
    output_root: Path,
    packet_root: Path | None = None,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Materialise salvaged documents for every invalid run under ``run_roots``.

    Writes ``<paper>/final_result.json`` so the evaluator and the union builder
    can read the root the same way they read any other, plus
    ``<paper>/salvage_report.json`` and a root-level manifest. A paper whose
    response validated whole contributes nothing: this root holds salvage, not
    a copy of results that never needed it.

    A root holds one document per paper, so when two run roots each salvage the
    same paper the first one wins and the second is recorded as
    ``skipped_conflict``. Root order is therefore part of the definition, the
    same way it is for the union builder.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "salvage_version": SALVAGE_REPORT_VERSION,
        "flag": SALVAGE_FLAG,
        "flag_enabled": is_enabled(SALVAGE_FLAG),
        "run_roots": [str(root) for root in run_roots],
        "papers": [],
    }
    for run_root in run_roots:
        for run_dir in sorted(run_root.glob("GP-*")):
            if paper_ids and run_dir.name not in paper_ids:
                continue
            if not (run_dir / "request.json").exists():
                continue
            if not (run_dir / "response.json").exists():
                continue
            document, report, context = salvage_run_dir(
                run_dir, packet_root=packet_root
            )
            entry = {
                **context,
                "salvage_status": report.status,
                "kept": report.kept,
                "rejected": report.rejected,
                "eligibility_kept": report.eligibility_kept,
                "dropped": len(report.dropped),
            }
            manifest["papers"].append(entry)
            if document is None:
                continue
            paper_dir = output_root / report.paper_id
            if (paper_dir / "final_result.json").exists():
                entry["skipped_conflict"] = True
                continue
            paper_dir.mkdir(parents=True, exist_ok=True)
            (paper_dir / "final_result.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (paper_dir / "salvage_report.json").write_text(
                report.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
    (output_root / "salvage_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Salvage the per-record content of contract-invalid extraction "
            "responses. Inert unless record_level_salvage is enabled."
        )
    )
    parser.add_argument("--run-root", action="append", dest="run_roots", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--packet-root",
        type=Path,
        default=None,
        help="Override the packet root request.json names.",
    )
    parser.add_argument("--paper-id", action="append", dest="paper_ids")
    args = parser.parse_args()

    manifest = build(
        [Path(root) for root in args.run_roots],
        output_root=args.output_root,
        packet_root=args.packet_root,
        paper_ids=args.paper_ids,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
