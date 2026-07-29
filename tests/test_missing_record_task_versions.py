"""What ``missing-record-task-1.1.0`` was bumped for, and what 1.0.0 kept.

The bump added ``existing_experiments``: the experiment *records*, not just
their ids. Without them a model that had genuinely recovered an outcome had
nothing legal to attach it to -- an outcome must cite an experiment -- and
three repair calls returned unresolved saying exactly that. Every fixture in
the suite passed ``[]`` for the field, so the payload the bump exists for was
never exercised.

Both halves of the contract are pinned here:

* a 1.1.0 task built from a result that carries experiments carries those
  records, and its checksum covers them; and
* a committed 1.0.0 task, which has no such key at all, still validates and
  still verifies its checksum -- the historical record of runs that produced
  real measurements must not be invalidated by a later field.

Inputs are the committed control run and packets, so the record shapes are the
ones the pipeline really produces rather than shapes invented here. Nothing
calls a model; ``build_repair_route_local`` is the unpaid half of the route.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.extraction.build_repair_route_local import run
from src.extraction.run_missing_record_repair import load_task


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "data/staging/extraction/codex_control_compact_v1"
PACKET_ROOT = REPO_ROOT / "data/staging/rag/compact_api_packets_v1"

# GP-004's control result declares three experiments and four outcomes, and its
# routing still leaves an unmatched candidate, so it is the paper that produces
# a repair task with a non-empty experiment payload.
PAPER_ID = "GP-004"

# A committed 1.0.0 task: written before the field existed, so its checksum was
# computed over a body that has no `existing_experiments` key.
LEGACY_TASK = (
    REPO_ROOT
    / "data/staging/extraction/codex_control_repair_tasks_v1"
    / PAPER_ID
    / "task_01.json"
)


@pytest.fixture(scope="module")
def source_result() -> dict:
    return json.loads(
        (RESULT_ROOT / PAPER_ID / "result.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def built_task_path(tmp_path_factory) -> Path:
    output_root = tmp_path_factory.mktemp("route") / "out"
    totals = run(RESULT_ROOT, PACKET_ROOT, output_root, only_papers={PAPER_ID})
    assert totals["text_tasks"] == 1, totals
    return next((output_root / PAPER_ID).glob("task_*.json"))


def test_the_source_result_still_carries_experiments(source_result):
    """Guard the fixture: an empty list here would make the next test vacuous."""
    assert len(source_result["experiments"]) == 3
    assert [row["experiment_id"] for row in source_result["experiments"]] == [
        "E1",
        "E2",
        "E3",
    ]


def test_a_1_1_0_task_carries_the_experiment_records_not_only_their_ids(
    built_task_path, source_result
):
    task = load_task(built_task_path)

    assert task.task_version == "missing-record-task-1.1.0"
    assert task.existing_experiments, "1.1.0 task built with an empty payload"
    carried = [row.model_dump(mode="json") for row in task.existing_experiments]
    assert carried == source_result["experiments"]
    # The ids stay, for readers that only need identity.
    assert [row.experiment_id for row in task.existing_experiments] == (
        task.existing_experiment_ids
    )


def test_a_recovered_outcome_has_a_real_experiment_to_attach_to(built_task_path):
    """The point of the bump: the records are complete enough to cite.

    A model asked to add an outcome must name an existing experiment. The task
    must therefore hand it experiments whose required fields are already filled
    in, not ids it would have to invent bodies for.
    """
    task = load_task(built_task_path)

    experiment = task.existing_experiments[0]
    assert experiment.experiment_id in task.existing_experiment_ids
    assert experiment.formulation_id in task.existing_formulation_ids
    # A ReportedField that is present rather than a bare identifier.
    assert experiment.payload_type.status in {"reported", "missing", "not_applicable"}


def test_the_1_1_0_checksum_covers_the_experiment_payload(built_task_path, tmp_path):
    """Tampering with the records the model is handed must break the signature.

    The untampered task is loaded first so this cannot pass for the wrong
    reason: a ``load_task`` that dropped ``existing_experiments`` before hashing
    would also make the tampered file raise, while quietly breaking every real
    1.1.0 task.
    """
    load_task(built_task_path)

    body = json.loads(built_task_path.read_text(encoding="utf-8"))
    assert body["existing_experiments"], "nothing to tamper with"
    body["existing_experiments"] = body["existing_experiments"][:-1]
    tampered = tmp_path / "tampered_task.json"
    tampered.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_task(tampered)


def test_a_committed_1_0_0_task_has_no_such_key(built_task_path):
    """Guard the fixture: if the field appeared, the next test stops testing 1.0.0."""
    body = json.loads(LEGACY_TASK.read_text(encoding="utf-8"))

    assert body["task_version"] == "missing-record-task-1.0.0"
    assert "existing_experiments" not in body


def test_a_1_0_0_task_still_validates_and_still_verifies_its_checksum():
    """The default must not be folded into a signature computed without it."""
    task = load_task(LEGACY_TASK)

    assert task.task_version == "missing-record-task-1.0.0"
    assert task.existing_experiments == []
    assert task.existing_experiment_ids  # it still names them by id
    assert task.evidence  # and it is still evidence-bounded


def test_a_1_0_0_task_is_not_exempt_from_checksum_verification(tmp_path):
    """Popping the new field must not turn into "1.0.0 signatures are unchecked"."""
    body = json.loads(LEGACY_TASK.read_text(encoding="utf-8"))
    body["evidence"][0]["text"] += " and the effect was larger than reported."
    tampered = tmp_path / "tampered_legacy_task.json"
    tampered.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_task(tampered)
