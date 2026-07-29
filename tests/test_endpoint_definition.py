"""The outcome contract's ``endpoint`` field, finally given a definition.

``OutcomeRecord`` declares ``endpoint: TextField`` and stops there, and no
version of the extraction prompt mentions the field at all, so what belongs in
it has always been left to the model's reading of the word. The annotation side
has no such silence: it names the thing measured and the population it was
measured in, in the endpoint itself, even though the experiment row it hangs off
carries a recipient-population column of its own.

This file covers the amendment that closes that gap and, mostly, what the
amendment must not do:

* with the flag off, the prompt text, the prompt version, the strict request
  schema and the exported baseline schema are byte-identical to what shipped --
  the schema checked against the checksum nine committed requests recorded, not
  against a constant written here;
* the pinned ``schema_sha256`` in ``config/extraction/compact_route_v1.yaml``
  does not move, which is the whole reason the definition lives on a subclass;
* with the flag on the definition reaches the model on both surfaces, from one
  text, so the schema and the instruction cannot drift apart;
* and the definition names no entity, no claim and no gold vocabulary. That is
  checked against the frozen gold set itself rather than a hand-written list,
  so it cannot be satisfied by renaming.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from src.config_flags import describe_flag, is_enabled, override
from src.extraction.compact_contracts import (
    ENDPOINT_DEFINITION,
    ENDPOINT_DEFINITION_FLAG,
    CandidateSlotExtractionResponse,
    CompactExtractionResponse,
    DefinedEndpointOutcomeRecord,
    EndpointDefinedExtractionResponse,
    EndpointDefinedSlotExtractionResponse,
    OutcomeRecord,
    active_response_contract,
)
from src.extraction.compact_prompt_v1 import (
    CANDIDATE_SLOT_EXTRACTION_PROMPT,
    CANDIDATE_SLOT_PROMPT_VERSION,
    COMPACT_EXTRACTION_PROMPT,
    ENDPOINT_DEFINITION_PROMPT_VERSION,
    ENDPOINT_DEFINITION_RULE,
    ENDPOINT_DEFINITION_SLOT_PROMPT_VERSION,
    PROMPT_VERSION,
    active_prompt,
)
from src.extraction.evaluate_final_gold_dynamic import DISTINCTIVE_TERMS, _tokens
from src.extraction.export_compact_contract_schema import OUTPUT as SCHEMA_PATH

ROOT = Path(__file__).resolve().parents[1]
FLAG = ENDPOINT_DEFINITION_FLAG
GOLD = ROOT / "data/annotations/gold_v1"
ROUTE_CONFIG = ROOT / "config/extraction/compact_route_v1.yaml"
# Nine requests this repository has already sent and committed, every one of
# them recording the schema checksum it used.
SHIPPED_REQUESTS = sorted(
    (ROOT / "data/staging/extraction/codex_full_view_v2").glob("GP-*/request.json")
)


# --------------------------------------------------------------------------- #
# The flag
# --------------------------------------------------------------------------- #


def test_the_flag_is_registered_off_and_declares_the_files_that_read_it():
    flag = describe_flag(FLAG)
    assert flag.default is False
    assert flag.status == "available"
    assert flag.description
    assert flag.rationale
    assert set(flag.integration_points) == {
        "src/extraction/compact_contracts.py",
        "src/extraction/compact_prompt_v1.py",
    }
    assert not is_enabled(FLAG)
    for alias in ("endpoint_names_its_population", "defined_endpoint"):
        assert describe_flag(alias).name == FLAG


# --------------------------------------------------------------------------- #
# Flag off: the request must stay byte-identical to what shipped
# --------------------------------------------------------------------------- #


def test_flag_off_leaves_every_shipped_prompt_byte_identical():
    with override(**{FLAG: False}):
        assert active_prompt(False).text == COMPACT_EXTRACTION_PROMPT
        assert active_prompt(False).version == PROMPT_VERSION
        assert active_prompt(True).text == CANDIDATE_SLOT_EXTRACTION_PROMPT
        assert active_prompt(True).version == CANDIDATE_SLOT_PROMPT_VERSION


def test_flag_off_selects_the_shipped_response_contracts():
    with override(**{FLAG: False}):
        assert active_response_contract(False) is CompactExtractionResponse
        assert active_response_contract(True) is CandidateSlotExtractionResponse


@pytest.mark.parametrize("request_path", SHIPPED_REQUESTS, ids=lambda p: p.parent.name)
def test_flag_off_reproduces_the_schema_checksum_of_a_request_already_sent(
    request_path,
):
    """The strongest available statement that nothing moved under the flag.

    Every committed codex request records the checksum of the strict schema it
    was sent with. Rebuilding that schema today with the flag off has to
    reproduce it exactly, or a run made after this change is no longer
    comparable with the runs the repository's headline numbers come from.
    """
    pytest.importorskip("openai")
    from src.extraction.run_codex_one_call import (
        _canonical_json,
        _sha256,
        strict_schema,
    )

    recorded = json.loads(request_path.read_text(encoding="utf-8"))
    slots = bool(recorded.get("candidate_slot_enforcement"))
    with override(**{FLAG: False}):
        schema = strict_schema(slots)
    assert (
        _sha256(_canonical_json(schema).encode("utf-8"))
        == recorded["schema_checksum"]
    )


def test_the_exported_baseline_schema_and_its_pinned_checksum_do_not_move():
    """The pin in compact_route_v1.yaml stays put, and not vacuously.

    A ``description`` on ``OutcomeRecord.endpoint`` itself would have been the
    obvious way to write this definition, and it would have changed the exported
    baseline schema -- and so the schema checksum inside every request
    fingerprint -- for runs that never asked for it. The second half of this
    test is what stops the first half from passing because nothing was written
    at all.
    """
    regenerated = (
        json.dumps(
            CompactExtractionResponse.model_json_schema(), indent=2, sort_keys=True
        )
        + "\n"
    )
    assert SCHEMA_PATH.read_text(encoding="utf-8") == regenerated

    pinned = yaml.safe_load(ROUTE_CONFIG.read_text(encoding="utf-8"))
    assert (
        pinned["response_contract"]["schema_sha256"]
        == hashlib.sha256(regenerated.encode("utf-8")).hexdigest()
    )

    baseline = OutcomeRecord.model_fields["endpoint"]
    defined = DefinedEndpointOutcomeRecord.model_fields["endpoint"]
    assert baseline.description is None
    assert defined.description == ENDPOINT_DEFINITION


# --------------------------------------------------------------------------- #
# Flag on: the definition, on both surfaces, from one text
# --------------------------------------------------------------------------- #


def test_flag_on_appends_the_definition_and_versions_it():
    with override(**{FLAG: True}):
        plain, slotted = active_prompt(False), active_prompt(True)
    assert plain.text == COMPACT_EXTRACTION_PROMPT + ENDPOINT_DEFINITION_RULE
    assert plain.version == ENDPOINT_DEFINITION_PROMPT_VERSION
    assert slotted.text == CANDIDATE_SLOT_EXTRACTION_PROMPT + ENDPOINT_DEFINITION_RULE
    assert slotted.version == ENDPOINT_DEFINITION_SLOT_PROMPT_VERSION
    assert plain.version != PROMPT_VERSION
    assert slotted.version != CANDIDATE_SLOT_PROMPT_VERSION
    assert plain.checksum == hashlib.sha256(plain.text.encode("utf-8")).hexdigest()


def test_flag_on_selects_the_contracts_that_define_endpoint():
    with override(**{FLAG: True}):
        assert active_response_contract(False) is EndpointDefinedExtractionResponse
        assert active_response_contract(True) is EndpointDefinedSlotExtractionResponse


def test_the_definition_travels_on_both_surfaces_from_one_text():
    """One definition, two carriers. Two copies of a definition drift."""
    assert ENDPOINT_DEFINITION in ENDPOINT_DEFINITION_RULE
    assert (
        DefinedEndpointOutcomeRecord.model_fields["endpoint"].description
        == ENDPOINT_DEFINITION
    )
    schema = EndpointDefinedSlotExtractionResponse.model_json_schema()
    described = schema["$defs"]["DefinedEndpointOutcomeRecord"]["properties"][
        "endpoint"
    ]["description"]
    assert described == ENDPOINT_DEFINITION


def test_an_explicit_argument_beats_the_flag_in_both_directions():
    with override(**{FLAG: True}):
        assert active_prompt(True, endpoint_definition=False).version == (
            CANDIDATE_SLOT_PROMPT_VERSION
        )
        assert active_response_contract(True, endpoint_definition=False) is (
            CandidateSlotExtractionResponse
        )
    with override(**{FLAG: False}):
        assert active_prompt(True, endpoint_definition=True).version == (
            ENDPOINT_DEFINITION_SLOT_PROMPT_VERSION
        )
        assert active_response_contract(True, endpoint_definition=True) is (
            EndpointDefinedSlotExtractionResponse
        )


# --------------------------------------------------------------------------- #
# The amendment adds no field, so nothing downstream needs to know
# --------------------------------------------------------------------------- #


def _reported(value: str) -> dict:
    return {
        "value": value,
        "status": "reported",
        "evidence_ids": ["E-1"],
        "missing_reason": None,
    }


def _missing() -> dict:
    return {
        "value": None,
        "status": "missing",
        "evidence_ids": [],
        "missing_reason": "not reported",
    }


def _response_payload() -> dict:
    return {
        "contract_version": "compact-1.1.0",
        "paper_id": "GP-000",
        "eligibility": {
            "decision": "eligible",
            "reason_codes": [
                "ORIGINAL_EXPERIMENT",
                "IDENTIFIABLE_LNP",
                "SUPPORTED_PAYLOAD",
                "TARGET_CELL_EVIDENCE",
                "USABLE_FORMULATION_OUTCOME_LINKAGE",
            ],
            "evidence_ids": ["E-1"],
            "explanation": "fixture",
        },
        "formulations": [
            {
                "formulation_id": "F1",
                "formulation_name": _reported("F1"),
                "composition": _missing(),
                "composition_basis": _missing(),
                "np_ratio": _missing(),
            }
        ],
        "components": [],
        "experiments": [
            {
                "experiment_id": "X1",
                "formulation_id": "F1",
                "payload_type": _reported("mRNA"),
                "payload_name": _missing(),
                "encoded_product": _missing(),
                "molecular_target": _missing(),
                "delivery_recipient_cell": _missing(),
                "therapeutic_target_cell": _missing(),
                "tissue_or_organ": _missing(),
                "species": _missing(),
                "disease_model": _missing(),
                "experimental_context": {
                    "value": "in_vivo",
                    "status": "reported",
                    "evidence_ids": ["E-1"],
                    "missing_reason": None,
                },
                "dose": _missing(),
                "dose_unit": _missing(),
                "route": _missing(),
                "timepoint": _missing(),
                "timepoint_unit": _missing(),
            }
        ],
        "outcomes": [
            {
                "outcome_id": "O1",
                "experiment_id": "X1",
                "assay": _missing(),
                "endpoint": _reported("signal in the population it was measured in"),
                "comparator": _missing(),
                "outcome_value": _missing(),
                "outcome_unit": _missing(),
                "qualitative_outcome": _missing(),
            }
        ],
        "unresolved_items": [],
    }


def test_a_defined_endpoint_response_is_still_a_baseline_response():
    payload = _response_payload()
    payload["candidate_dispositions"] = [
        {"candidate_id": "C1", "disposition": "extracted", "reason": None}
    ]
    parsed = EndpointDefinedSlotExtractionResponse.model_validate(payload)
    assert isinstance(parsed, CompactExtractionResponse)
    assert isinstance(parsed.outcomes[0], OutcomeRecord)

    baseline = CandidateSlotExtractionResponse.model_validate(payload)
    assert parsed.model_dump(mode="json") == baseline.model_dump(mode="json")


def test_the_definition_adds_no_field_to_the_record():
    assert list(DefinedEndpointOutcomeRecord.model_fields) == list(
        OutcomeRecord.model_fields
    )
    assert list(EndpointDefinedSlotExtractionResponse.model_fields) == list(
        CandidateSlotExtractionResponse.model_fields
    )


def test_the_endpoint_field_is_still_required_and_still_validated():
    payload = _response_payload()
    payload["outcomes"][0].pop("endpoint")
    with pytest.raises(Exception):
        EndpointDefinedExtractionResponse.model_validate(payload)

    payload = _response_payload()
    payload["outcomes"][0]["endpoint"] = {
        "value": "something",
        "status": "reported",
        "evidence_ids": [],
        "missing_reason": None,
    }
    with pytest.raises(Exception):
        EndpointDefinedExtractionResponse.model_validate(payload)


# --------------------------------------------------------------------------- #
# The measurement: the treatment run and the control it is paired with
# --------------------------------------------------------------------------- #

TREATMENT_RUN = ROOT / "data/staging/extraction/codex_endpoint_definition_v1"
CONTROL_RUN = ROOT / "data/staging/extraction/codex_full_view_v2"
BASE_ROOT = ROOT / "data/staging/extraction/codex_union_vision_v3"
MEASURED_ROOT = (
    ROOT / "data/staging/extraction/codex_union_vision_v3_endpoint_definition"
)

needs_measurement = pytest.mark.skipif(
    not (TREATMENT_RUN.exists() and CONTROL_RUN.exists() and BASE_ROOT.exists()),
    reason="the measured run, its control or its base is not present",
)


@needs_measurement
@pytest.mark.parametrize("paper_id", [f"GP-{index:03d}" for index in range(1, 10)])
def test_the_treatment_run_differs_from_its_control_only_in_the_definition(paper_id):
    """The whole claim rests on this being a paired comparison.

    The control is a flag-off nine-paper run this repository already committed,
    not a run made for the occasion, so it costs nothing and cannot have been
    tuned. What makes it a control is that everything except the amendment is
    the same request: same evidence view, same packet down to its checksum,
    same candidate slots, same model and reasoning effort. If that stops being
    true the measurement is comparing two different things and says nothing
    about the definition.
    """
    treatment = json.loads(
        (TREATMENT_RUN / paper_id / "request.json").read_text(encoding="utf-8")
    )
    control = json.loads(
        (CONTROL_RUN / paper_id / "request.json").read_text(encoding="utf-8")
    )
    for field in (
        "evidence_view",
        "packet_checksum",
        "candidate_slots",
        "model",
        "reasoning_effort",
    ):
        assert treatment[field] == control[field], field
    # By name, not by path: both runs were made from a git worktree, so the
    # absolute prefix differs and says nothing. The packet checksum above is
    # what proves the bytes were the same.
    assert Path(treatment["packet_root"]).name == Path(control["packet_root"]).name

    assert control["prompt_version"] == CANDIDATE_SLOT_PROMPT_VERSION
    assert treatment["prompt_version"] == ENDPOINT_DEFINITION_SLOT_PROMPT_VERSION
    assert treatment["schema_checksum"] != control["schema_checksum"]
    assert treatment["prompt_characters"] - control["prompt_characters"] == len(
        ENDPOINT_DEFINITION_RULE
    )


# --------------------------------------------------------------------------- #
# The measurement overlay: merge, never swap
# --------------------------------------------------------------------------- #


def _outcome(outcome_id: str, endpoint: str) -> dict:
    return {
        "outcome_id": outcome_id,
        "experiment_id": "E1",
        "assay": _missing(),
        "endpoint": _reported(endpoint),
        "comparator": _missing(),
        "outcome_value": _missing(),
        "outcome_unit": _missing(),
        "qualitative_outcome": _missing(),
    }


def _result(outcomes: list[dict]) -> dict:
    return {
        "paper_id": "GP-001",
        "experiments": [{"experiment_id": "E1", "formulation_id": "F1"}],
        "outcomes": outcomes,
    }


def _write(root: Path, paper_id: str, payload: dict) -> None:
    (root / paper_id).mkdir(parents=True, exist_ok=True)
    (root / paper_id / "final_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def test_the_overlay_merges_and_never_drops_a_base_record(tmp_path):
    """A swap can lose a gold row the base already recovers.

    That loss would be caused by not sending a record, not by the definition,
    so the arm that measures the definition must not be able to produce it.
    """
    from src.extraction.build_union_endpoint_definition import build

    base, overlay = tmp_path / "base", tmp_path / "overlay"
    _write(base, "GP-001", _result([_outcome("O1", "base only")]))
    _write(base, "GP-002", _result([_outcome("O1", "untouched")]))
    _write(overlay, "GP-001", _result([_outcome("O1", "overlay only")]))

    manifest = build(
        output_root=tmp_path / "out",
        base_root=base,
        overlay_root=overlay,
        paper_ids=["GP-001", "GP-002"],
    )
    merged = json.loads(
        (tmp_path / "out/GP-001/final_result.json").read_text(encoding="utf-8")
    )
    endpoints = [row["endpoint"]["value"] for row in merged["outcomes"]]
    assert endpoints == ["base only", "overlay only"]
    assert [row["experiment_id"] for row in merged["experiments"]] == ["E1", "E1-e"]
    assert merged["outcomes"][1]["experiment_id"] == "E1-e"

    untouched = (base / "GP-002/final_result.json").read_text(encoding="utf-8")
    assert (
        tmp_path / "out/GP-002/final_result.json"
    ).read_text(encoding="utf-8") == untouched
    assert manifest["papers"][1]["note"].startswith("overlay absent")


@needs_measurement
def test_the_overlay_root_is_rebuildable_and_scores_what_was_recorded(tmp_path):
    """The measured numbers, and the root they came from, reproduce from data.

    Pinned because they are the whole answer: the definition is obeyed, it
    bought no recall, and it cost precision. A rebuild that quietly scored
    differently would make that statement unverifiable.
    """
    from src.extraction.build_union_endpoint_definition import build
    from src.extraction.evaluate_final_gold_dynamic import evaluate

    build(output_root=tmp_path / "overlay")
    with override(vision_relationship_polarity=True):
        rebuilt = evaluate(result_roots=[tmp_path / "overlay"])
        base = evaluate(result_roots=[BASE_ROOT])

    assert base["recovered"] == rebuilt["recovered"] == 13
    assert rebuilt["missing_gold_outcome_ids"] == ["GO-017", "GO-018"]
    assert round(base["precision"], 6) == 0.220339
    assert round(rebuilt["precision"], 6) == 0.136842
    assert base["false_additions"]["count"] == 46
    assert rebuilt["false_additions"]["count"] == 82
    assert base["evidence_accuracy"] == {
        "checked": 12,
        "supported": 7,
        "rate": 7 / 12,
    }
    assert rebuilt["evidence_accuracy"] == {
        "checked": 12,
        "supported": 8,
        "rate": 8 / 12,
    }

    if MEASURED_ROOT.exists():
        with override(vision_relationship_polarity=True):
            committed = evaluate(result_roots=[MEASURED_ROOT])
        assert round(committed["precision"], 9) == round(rebuilt["precision"], 9)


@needs_measurement
def test_the_control_arm_shows_the_merge_causes_most_of_what_moved(tmp_path):
    """Attribution, not just a delta.

    Merging any fresh nine-paper run adds records, lowers precision and can
    move a match, whether or not the definition did anything. Overlaying the
    flag-off run this same way is what separates the two, and it costs no
    Codex turn because that run was already committed. It reproduces the
    recall, the evidence-accuracy gain and the one improved assignment, which
    is why none of those is claimed for the definition.
    """
    from src.extraction.build_union_endpoint_definition import build
    from src.extraction.evaluate_final_gold_dynamic import evaluate

    build(output_root=tmp_path / "control", overlay_root=CONTROL_RUN)
    with override(vision_relationship_polarity=True):
        control = evaluate(result_roots=[tmp_path / "control"])
        treatment = evaluate(result_roots=[MEASURED_ROOT])

    assert control["recovered"] == treatment["recovered"] == 13
    assert round(control["precision"], 6) == 0.166667
    assert control["evidence_accuracy"]["supported"] == 8
    # The definition's own contribution: more records, less precision.
    assert control["false_additions"]["count"] == 65
    assert treatment["false_additions"]["count"] == 82
    assert treatment["precision"] < control["precision"]


@needs_measurement
def test_no_matched_row_moved_to_a_record_its_evidence_does_not_carry():
    """The veto condition, checked rather than asserted.

    Three matched rows changed assignment. A move is only acceptable if the
    record moved to is at least as well evidenced as the one it replaced, so
    this pins each move's direction: the literal evidence check must not go
    from supported to unsupported, and the match score must not fall.
    """
    from src.extraction.evaluate_final_gold_dynamic import evaluate

    with override(vision_relationship_polarity=True):
        before = {
            row["gold_outcome_id"]: row
            for row in evaluate(result_roots=[BASE_ROOT])["results"]
        }
        after = {
            row["gold_outcome_id"]: row
            for row in evaluate(result_roots=[MEASURED_ROOT])["results"]
        }

    moved = sorted(
        gold_id
        for gold_id, row in after.items()
        if (row.get("match") or {}).get("outcome_index")
        != (before[gold_id].get("match") or {}).get("outcome_index")
    )
    assert moved == ["GO-011", "GO-013", "GO-016"]
    for gold_id in moved:
        assert after[gold_id]["match"]["score"] > before[gold_id]["match"]["score"]
        assert not (
            before[gold_id]["evidence_supported"]
            and not after[gold_id]["evidence_supported"]
        ), f"{gold_id} moved to a record whose evidence stopped supporting it"
    assert before["GO-016"]["evidence_supported"] is False
    assert after["GO-016"]["evidence_supported"] is True


@needs_measurement
def test_the_rule_was_obeyed_and_the_control_was_mostly_obeying_it_already():
    """Why this is a negative result rather than a bug report.

    Counted against each record's own experiment row rather than a list of
    populations written here: an endpoint counts as naming its population when
    it shares a word with the recipient-population or tissue field of the
    experiment that record belongs to. The rule moved that from 16 of 21 to 29
    of 36 -- and moved the record count from 21 to 36, which is the cost.
    """
    from src.extraction.evaluate_final_gold_dynamic import _tokens

    def survey(root: Path) -> tuple[int, int]:
        named = total = 0
        for path in sorted(root.glob("GP-*/result.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            experiments = {
                row["experiment_id"]: row for row in result.get("experiments", [])
            }
            for outcome in result.get("outcomes", []):
                total += 1
                endpoint = _tokens((outcome.get("endpoint") or {}).get("value") or "")
                experiment = experiments.get(outcome.get("experiment_id")) or {}
                population: set[str] = set()
                for field in (
                    "delivery_recipient_cell",
                    "therapeutic_target_cell",
                    "tissue_or_organ",
                ):
                    population |= _tokens(
                        (experiment.get(field) or {}).get("value") or ""
                    )
                if population and (endpoint & population):
                    named += 1
        return named, total

    assert survey(TREATMENT_RUN) == (29, 36)
    assert survey(CONTROL_RUN) == (16, 21)


# --------------------------------------------------------------------------- #
# Generality: the definition may name nothing from the answer key
# --------------------------------------------------------------------------- #

AMENDMENT_TEXTS = (ENDPOINT_DEFINITION, ENDPOINT_DEFINITION_RULE)


def _gold_rows() -> list[dict[str, str]]:
    with (GOLD / "outcomes.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _gold_experiment_rows() -> list[dict[str, str]]:
    with (GOLD / "experiments.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _rare_gold_endpoint_tokens() -> set[str]:
    """The evaluator's own definition of a distinguishing endpoint word.

    Frequency <= 2 across the frozen gold set, two-letter connectives dropped
    because they name nothing and occur in any English sentence.
    """
    frequency = Counter(
        token for row in _gold_rows() for token in _tokens(row["endpoint_name"])
    )
    return {
        token for token, count in frequency.items() if count <= 2 and len(token) > 2
    }


def _gold_population_terms() -> set[str]:
    """Every population the gold set names, taken from the gold set.

    Derived from its own cell columns rather than written down here, so the
    test cannot be satisfied by naming a population this list forgot.
    """
    terms: set[str] = set()
    for row in _gold_experiment_rows():
        for column in ("cell_type", "delivery_recipient_cell", "therapeutic_target_cell"):
            terms |= {token for token in _tokens(row[column] or "") if len(token) > 2}
    for row in _gold_rows():
        terms |= {
            token
            for token in _tokens(row["normalization_basis"] or "")
            if len(token) > 2
        }
    return terms


@pytest.mark.parametrize("text", AMENDMENT_TEXTS)
def test_the_definition_names_no_distinguishing_gold_endpoint(text):
    leaked = _tokens(text) & _rare_gold_endpoint_tokens()
    assert not leaked, f"the definition names the answer key: {sorted(leaked)}"


@pytest.mark.parametrize("text", AMENDMENT_TEXTS)
def test_the_definition_names_no_population_the_gold_set_names(text):
    """No subtraction of the shipped prompt here, deliberately.

    The other generality tests forgive a word the baseline already puts in
    front of the model, which is right for ordinary vocabulary. It is wrong
    for populations: the baseline names four of them in its eligibility
    criterion, so subtracting it would license the one edit this rule must
    never make -- telling the model which populations to write into an
    endpoint. The rule is general or it is an answer key.
    """
    leaked = _tokens(text) & _gold_population_terms()
    assert not leaked, f"the definition names a gold population: {sorted(leaked)}"


@pytest.mark.parametrize("text", AMENDMENT_TEXTS)
def test_the_definition_seeds_no_distinctive_claim_vocabulary(text):
    """DISTINCTIVE_TERMS is the matcher's own list of claim-specific words.

    Subtracting the shipped prompt first: a word the baseline already puts in
    front of the model is not introduced by an amendment.
    """
    already = _tokens(CANDIDATE_SLOT_EXTRACTION_PROMPT)
    assert not ((_tokens(text) - already) & DISTINCTIVE_TERMS)


@pytest.mark.parametrize("text", AMENDMENT_TEXTS)
def test_no_gold_claim_is_reconstructible_from_the_definition(text):
    """At most one ordinary word per gold claim, so no claim is recoverable."""
    already = _tokens(CANDIDATE_SLOT_EXTRACTION_PROMPT)
    introduced = _tokens(text) - already
    for row in _gold_rows():
        for column in ("qualitative_outcome", "endpoint_name"):
            shared = _tokens(row[column] or "") & introduced
            assert len(shared) <= 1, (
                f"{row['gold_outcome_id']} shares {sorted(shared)} with the "
                "definition"
            )


@pytest.mark.parametrize("text", AMENDMENT_TEXTS)
def test_the_definition_says_nothing_about_how_records_are_scored(text):
    """It defines a field. It must not describe the thing that reads the field.

    A rule written towards the evaluator would be fitting the answer key even
    if every word of it were true, so the vocabulary of matching is banned
    outright rather than left to judgement.
    """
    forbidden = (
        "gate",
        "token",
        "match",
        "matched",
        "matching",
        "overlap",
        "score",
        "scoring",
        "recall",
        "precision",
        "gold",
        "evaluat",
        "grader",
    )
    lowered = text.lower()
    named = [word for word in forbidden if word in lowered]
    assert not named, f"the definition talks about scoring: {named}"
