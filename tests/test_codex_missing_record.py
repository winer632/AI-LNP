"""The codex missing-record executor: what it must and must not write.

The mechanism under test is the one that historically moved gold recall
(7/15 -> 10/15 on the paid path): repair -> merge -> re-evaluate. The risk it
carries is that a fragment the model returned but that does *not* survive
``validate_response`` still reaches the merge, inventing records the evidence
never supported. These tests pin the boundary: only a validated fragment is
written to disk, everything else is recorded as a rejection.

No test here calls ``codex``. The subprocess is injected as ``codex_runner``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.extraction.compact_contracts import ExperimentRecord, OutcomeRecord
from src.extraction.missing_record_contracts import (
    MissingRecordFragment,
    MissingRecordTask,
)
from src.extraction.repair_contracts import RepairEvidence
from src.extraction.run_codex_missing_record import (
    HARNESS,
    PROMPT_VERSION,
    build_prompt,
    discover_tasks,
    fingerprint,
    run_task,
    run_tasks,
    strict_schema,
)
from src.extraction.run_missing_record_repair import PROMPT, load_task


# ---------------------------------------------------------------------------
# Fixtures: a checksum-valid task and a fragment that answers it
# ---------------------------------------------------------------------------


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _reported(value, evidence="GP-T-E-1"):
    return {
        "value": value,
        "status": "reported",
        "evidence_ids": [evidence],
        "missing_reason": None,
    }


def _missing():
    return {
        "value": None,
        "status": "missing",
        "evidence_ids": [],
        "missing_reason": "Not reported.",
    }


def _task(**overrides) -> MissingRecordTask:
    payload = {
        "task_version": "missing-record-task-1.1.0",
        "paper_id": "GP-T",
        "route_ids": ["RT-1"],
        "candidate_ids": ["OC-1", "OC-2"],
        "evidence": [
            {
                "evidence_id": "GP-T-E-1",
                "text": "More than 80% of hepatocytes expressed GFP at 24 h.",
                "source_ids": ["S1"],
            }
        ],
        "existing_formulation_ids": ["F1"],
        "existing_experiment_ids": ["E1"],
        # Present in the payload so the checksum covers it: the model
        # supplies its default on load, and a checksum computed without
        # it would never verify.
        "existing_experiments": [],
        "existing_outcome_ids": ["O1"],
        "permitted_new_experiments": 1,
        "permitted_new_outcomes": 2,
        "source_result_sha256": "a" * 64,
        "source_inventory_sha256": "b" * 64,
    }
    payload.update(overrides)
    checksum = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return MissingRecordTask(**payload, task_checksum=checksum)


def _write_task(directory: Path, name: str = "task_01.json", **overrides) -> Path:
    task = _task(**overrides)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(task.model_dump_json(indent=2), encoding="utf-8")
    # The executor loads through the shared checksum-verifying loader, so a
    # fixture that drifted from the contract fails here rather than silently.
    load_task(path)
    return path


def _experiment(experiment_id="E2"):
    return ExperimentRecord(
        experiment_id=experiment_id,
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


def _outcome(outcome_id="O2", experiment_id="E2"):
    return OutcomeRecord(
        outcome_id=outcome_id,
        experiment_id=experiment_id,
        assay=_reported("microscopy"),
        endpoint=_reported("GFP expression"),
        comparator=_missing(),
        outcome_value=_reported(80.0),
        outcome_unit=_reported("%"),
        qualitative_outcome=_reported("More than 80% expressed GFP."),
    )


def _fragment(**overrides) -> MissingRecordFragment:
    payload = {
        "disposition": "recovered",
        "recovered_candidate_ids": ["OC-1"],
        "unresolved_candidate_ids": ["OC-2"],
        "experiments": [_experiment()],
        "outcomes": [_outcome()],
        "unresolved_reason": "OC-2 needs a figure.",
    }
    payload.update(overrides)
    return MissingRecordFragment(**payload)


def _runner(text: str, calls: list | None = None):
    """A codex stand-in that records its arguments and returns fixed text."""

    def run(prompt, schema, *, model, reasoning_effort, timeout):
        if calls is not None:
            calls.append(
                {
                    "prompt": prompt,
                    "schema": schema,
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "timeout": timeout,
                }
            )
        return {"text": text, "elapsed_seconds": 1.5, "stdout_tail": ""}

    return run


# ---------------------------------------------------------------------------
# The prompt and schema are the paid ones, not a rewrite
# ---------------------------------------------------------------------------


def test_prompt_carries_the_paid_instructions_verbatim():
    # A drifted prompt would make the codex control measure a different
    # mechanism than the paid run it is being compared against.
    prompt = build_prompt(_task())
    assert PROMPT in prompt


def test_prompt_carries_the_whole_task_payload():
    task = _task()
    prompt = build_prompt(task)
    assert _canonical(task.model_dump(mode="json")) in prompt
    assert task.evidence[0].text in prompt


def test_schema_is_the_strict_fragment_contract():
    schema = strict_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(MissingRecordFragment.model_fields)


# ---------------------------------------------------------------------------
# Fingerprint: distinct inputs must not share an output directory
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_for_the_same_inputs():
    task = _task()
    assert fingerprint(task, model="m", reasoning_effort="low") == fingerprint(
        task, model="m", reasoning_effort="low"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "other-model", "reasoning_effort": "low"},
        {"model": "m", "reasoning_effort": "high"},
    ],
)
def test_fingerprint_changes_with_model_or_effort(kwargs):
    task = _task()
    assert fingerprint(task, model="m", reasoning_effort="low") != fingerprint(
        task, **kwargs
    )


def test_fingerprint_changes_with_the_task():
    base = _task()
    other = _task(candidate_ids=["OC-1", "OC-3"])
    assert fingerprint(base, model="m", reasoning_effort="low") != fingerprint(
        other, model="m", reasoning_effort="low"
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_validated_fragment_writes_the_four_artifacts(tmp_path):
    task_path = _write_task(tmp_path / "tasks" / "GP-T")
    calls: list = []
    manifest = run_task(
        task_path,
        output_root=tmp_path / "out",
        model="gpt-5.6-terra",
        reasoning_effort="low",
        codex_runner=_runner(_fragment().model_dump_json(), calls),
    )

    run_dir = Path(manifest["fragment_path"]).parent
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "fragment.json",
        "manifest.json",
        "request.json",
        "response.json",
    ]
    assert manifest["status"] == "completed_pending_merge"
    assert manifest["response_validated"] is True
    assert manifest["disposition"] == "recovered"
    assert manifest["record_counts"] == {"experiments": 1, "outcomes": 1}
    assert calls[0]["model"] == "gpt-5.6-terra"
    assert calls[0]["reasoning_effort"] == "low"


def test_manifest_declares_the_harness_and_zero_paid_requests(tmp_path):
    task_path = _write_task(tmp_path / "tasks" / "GP-T")
    manifest = run_task(
        task_path,
        output_root=tmp_path / "out",
        codex_runner=_runner(_fragment().model_dump_json()),
    )
    # These two keys are what stops a downstream reader treating codex output as
    # a paid-path artifact and comparing the numbers across harnesses.
    assert manifest["harness"] == HARNESS == "codex-exec"
    assert manifest["openai_api_requests"] == 0
    assert manifest["codex_exec_turns"] == 1


def test_written_fragment_reloads_as_a_valid_fragment(tmp_path):
    task_path = _write_task(tmp_path / "tasks" / "GP-T")
    manifest = run_task(
        task_path,
        output_root=tmp_path / "out",
        codex_runner=_runner(_fragment().model_dump_json()),
    )
    # merge_missing_records reads exactly this way.
    reloaded = MissingRecordFragment.model_validate_json(
        Path(manifest["fragment_path"]).read_text(encoding="utf-8")
    )
    assert reloaded.recovered_candidate_ids == ["OC-1"]


def test_request_snapshot_records_the_task_identity(tmp_path):
    task_path = _write_task(tmp_path / "tasks" / "GP-T")
    manifest = run_task(
        task_path,
        output_root=tmp_path / "out",
        codex_runner=_runner(_fragment().model_dump_json()),
    )
    request = json.loads(
        (Path(manifest["fragment_path"]).parent / "request.json").read_text(
            encoding="utf-8"
        )
    )
    assert request["task_checksum"] == load_task(task_path).task_checksum
    assert request["candidate_ids"] == ["OC-1", "OC-2"]
    assert request["prompt_version"] == PROMPT_VERSION
    assert request["harness"] == HARNESS


def test_unresolved_fragment_is_accepted_and_adds_no_records(tmp_path):
    task_path = _write_task(tmp_path / "tasks" / "GP-T")
    unresolved = _fragment(
        disposition="unresolved",
        recovered_candidate_ids=[],
        unresolved_candidate_ids=["OC-1", "OC-2"],
        experiments=[],
        outcomes=[],
        unresolved_reason="The evidence gives no value.",
    )
    manifest = run_task(
        task_path,
        output_root=tmp_path / "out",
        codex_runner=_runner(unresolved.model_dump_json()),
    )
    assert manifest["response_validated"] is True
    assert manifest["disposition"] == "unresolved"
    assert manifest["record_counts"] == {"experiments": 0, "outcomes": 0}


# ---------------------------------------------------------------------------
# Rejection: an unvalidated fragment must never reach the merge
# ---------------------------------------------------------------------------


def test_fragment_citing_unknown_evidence_is_not_written(tmp_path):
    task_path = _write_task(tmp_path / "tasks" / "GP-T")
    outcome = _outcome()
    outcome.endpoint.evidence_ids = ["E-MADE-UP"]
    manifest = run_task(
        task_path,
        output_root=tmp_path / "out",
        codex_runner=_runner(_fragment(outcomes=[outcome]).model_dump_json()),
    )
    assert manifest["status"] == "rejected"
    assert manifest["response_validated"] is False
    assert "unknown evidence" in manifest["validation_error"]
    assert manifest["fragment_path"] is None
    assert not list((tmp_path / "out").rglob("fragment.json"))


def test_fragment_dropping_a_candidate_is_not_written(tmp_path):
    task_path = _write_task(tmp_path / "tasks" / "GP-T")
    silent = _fragment(unresolved_candidate_ids=[], unresolved_reason=None)
    manifest = run_task(
        task_path,
        output_root=tmp_path / "out",
        codex_runner=_runner(silent.model_dump_json()),
    )
    assert manifest["status"] == "rejected"
    assert "every candidate" in manifest["validation_error"]
    assert not list((tmp_path / "out").rglob("fragment.json"))


def test_fragment_reusing_an_existing_outcome_id_is_not_written(tmp_path):
    task_path = _write_task(tmp_path / "tasks" / "GP-T")
    manifest = run_task(
        task_path,
        output_root=tmp_path / "out",
        codex_runner=_runner(_fragment(outcomes=[_outcome("O1")]).model_dump_json()),
    )
    assert manifest["status"] == "rejected"
    assert "existing outcome" in manifest["validation_error"]


def test_non_json_output_is_recorded_not_raised(tmp_path):
    task_path = _write_task(tmp_path / "tasks" / "GP-T")
    manifest = run_task(
        task_path,
        output_root=tmp_path / "out",
        codex_runner=_runner("I could not find the value in the evidence."),
    )
    assert manifest["status"] == "rejected"
    assert manifest["response_validated"] is False
    # The raw text is still on disk for a human to read.
    response_path = next((tmp_path / "out" / "GP-T").rglob("response.json"))
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["text"].startswith("I could not find")


def test_rejected_attempt_still_writes_a_manifest(tmp_path):
    task_path = _write_task(tmp_path / "tasks" / "GP-T")
    run_task(
        task_path,
        output_root=tmp_path / "out",
        codex_runner=_runner("not json"),
    )
    assert list((tmp_path / "out").rglob("manifest.json"))


# ---------------------------------------------------------------------------
# The quota guard: no accidental second turn
# ---------------------------------------------------------------------------


def test_completed_task_is_served_from_disk_without_a_second_turn(tmp_path):
    task_path = _write_task(tmp_path / "tasks" / "GP-T")
    calls: list = []
    runner = _runner(_fragment().model_dump_json(), calls)
    run_task(task_path, output_root=tmp_path / "out", codex_runner=runner)
    second = run_task(task_path, output_root=tmp_path / "out", codex_runner=runner)
    assert len(calls) == 1
    assert second["cache_hit"] is True
    assert second["codex_exec_turns_this_run"] == 0


def test_failed_attempt_blocks_an_automatic_retry(tmp_path):
    task_path = _write_task(tmp_path / "tasks" / "GP-T")
    calls: list = []
    run_task(
        task_path,
        output_root=tmp_path / "out",
        codex_runner=_runner("not json", calls),
    )
    with pytest.raises(FileExistsError, match="refusing an automatic Codex retry"):
        run_task(
            task_path,
            output_root=tmp_path / "out",
            codex_runner=_runner("not json", calls),
        )
    assert len(calls) == 1


def test_retry_failed_reruns_only_the_failed_attempt(tmp_path):
    task_path = _write_task(tmp_path / "tasks" / "GP-T")
    calls: list = []
    run_task(
        task_path,
        output_root=tmp_path / "out",
        codex_runner=_runner("not json", calls),
    )
    manifest = run_task(
        task_path,
        output_root=tmp_path / "out",
        retry_failed=True,
        codex_runner=_runner(_fragment().model_dump_json(), calls),
    )
    assert len(calls) == 2
    assert manifest["response_validated"] is True


def test_retry_failed_never_discards_a_validated_fragment(tmp_path):
    task_path = _write_task(tmp_path / "tasks" / "GP-T")
    calls: list = []
    runner = _runner(_fragment().model_dump_json(), calls)
    first = run_task(task_path, output_root=tmp_path / "out", codex_runner=runner)
    second = run_task(
        task_path,
        output_root=tmp_path / "out",
        retry_failed=True,
        codex_runner=runner,
    )
    assert len(calls) == 1
    assert second["cache_hit"] is True
    assert Path(first["fragment_path"]).exists()


def test_a_tampered_task_file_is_refused(tmp_path):
    task_path = _write_task(tmp_path / "tasks" / "GP-T")
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    payload["permitted_new_outcomes"] = 8
    task_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        run_task(
            task_path,
            output_root=tmp_path / "out",
            codex_runner=_runner(_fragment().model_dump_json()),
        )


# ---------------------------------------------------------------------------
# Batch behaviour
# ---------------------------------------------------------------------------


def test_discover_tasks_finds_every_task_in_paper_order(tmp_path):
    _write_task(tmp_path / "GP-002", "task_02.json")
    _write_task(tmp_path / "GP-002", "task_01.json")
    _write_task(tmp_path / "GP-004", "task_01.json")
    (tmp_path / "GP-002" / "routing.json").write_text("{}", encoding="utf-8")
    found = discover_tasks(tmp_path)
    assert [path.parent.name + "/" + path.name for path in found] == [
        "GP-002/task_01.json",
        "GP-002/task_02.json",
        "GP-004/task_01.json",
    ]


def test_batch_continues_past_a_failing_task_and_counts_it(tmp_path):
    good = _write_task(tmp_path / "tasks" / "GP-T", "task_01.json")
    bad = tmp_path / "tasks" / "GP-T" / "task_02.json"
    bad.write_text("{not json", encoding="utf-8")
    batch = run_tasks(
        [good, bad],
        output_root=tmp_path / "out",
        codex_runner=_runner(_fragment().model_dump_json()),
    )
    assert batch["tasks"] == 2
    assert batch["validated"] == 1
    assert batch["failed"] == 1
    assert batch["results"][1]["status"] == "error"


def test_batch_manifest_is_written_and_declares_zero_paid_requests(tmp_path):
    task_path = _write_task(tmp_path / "tasks" / "GP-T")
    run_tasks(
        [task_path],
        output_root=tmp_path / "out",
        codex_runner=_runner(_fragment().model_dump_json()),
    )
    batch = json.loads(
        (tmp_path / "out" / "batch_manifest.json").read_text(encoding="utf-8")
    )
    assert batch["harness"] == HARNESS
    assert batch["openai_api_requests"] == 0
    assert batch["codex_exec_turns"] == 1
    assert batch["recovered"] == 1


# ---------------------------------------------------------------------------
# The real prepared tasks still load through this executor
# ---------------------------------------------------------------------------


def test_real_control_tasks_build_a_prompt_and_schema_if_present():
    root = Path(__file__).resolve().parents[1] / "data/staging/extraction"
    tasks = discover_tasks(root / "codex_control_repair_tasks_v1")
    if not tasks:
        pytest.skip("control repair tasks are not checked in")
    for path in tasks:
        prompt = build_prompt(load_task(path))
        assert PROMPT in prompt
