"""Recall-first union over extraction runs, and the key it deduplicates on.

Two runs number their outcome records independently, so ``outcome_id`` says
nothing about identity across runs. Both halves of that matter:

* the same ``O1`` in two runs is usually two different measurements, and keying
  on the id silently throws one away -- the recall the union exists to buy;
* the same measurement renumbered by a second run is one measurement, and not
  collapsing it double-counts a record the evaluator matches one-to-one.

The run roots here are the real committed control and treatment runs of GP-004,
whose outcome records genuinely collide on ``O1`` while measuring different
things.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.extraction.union_extraction_results import (
    _outcome_key,
    build,
    union_paper,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_RUN = REPO_ROOT / "data/staging/extraction/codex_control_compact_v1"
TREATMENT_RUN = REPO_ROOT / "data/staging/extraction/codex_treatment_full_v1"
PAPER_ID = "GP-004"


def _outcomes(root: Path) -> list[dict]:
    return json.loads(
        (root / PAPER_ID / "result.json").read_text(encoding="utf-8")
    )["outcomes"]


def _endpoint(outcome: dict) -> str:
    return (outcome.get("endpoint") or {}).get("value") or ""


def test_outcome_key_ignores_the_generated_outcome_id():
    control, treatment = _outcomes(CONTROL_RUN)[0], _outcomes(TREATMENT_RUN)[0]
    assert control["outcome_id"] == treatment["outcome_id"] == "O1"

    renumbered = {**control, "outcome_id": "O47"}
    assert _outcome_key(control) == _outcome_key(renumbered)
    assert _outcome_key(control) != _outcome_key(treatment)


def test_two_runs_that_both_number_a_record_o1_keep_both_measurements():
    merged = union_paper(PAPER_ID, [CONTROL_RUN, TREATMENT_RUN])

    assert merged is not None
    endpoints = {_endpoint(row) for row in merged["outcomes"]}
    control_endpoints = {_endpoint(row) for row in _outcomes(CONTROL_RUN)}
    treatment_endpoints = {_endpoint(row) for row in _outcomes(TREATMENT_RUN)}
    assert control_endpoints <= endpoints
    assert treatment_endpoints <= endpoints
    assert len(merged["outcomes"]) == len(_outcomes(CONTROL_RUN)) + len(
        _outcomes(TREATMENT_RUN)
    )


def test_the_same_measurement_renumbered_by_a_second_run_is_kept_once(tmp_path):
    """A second run reporting the same endpoint/value/unit adds nothing."""
    renumbered_root = tmp_path / "renumbered_run"
    paper_dir = renumbered_root / PAPER_ID
    paper_dir.mkdir(parents=True)
    result = json.loads(
        (CONTROL_RUN / PAPER_ID / "result.json").read_text(encoding="utf-8")
    )
    for index, outcome in enumerate(result["outcomes"], start=41):
        outcome["outcome_id"] = f"O{index}"
    (paper_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8"
    )

    merged = union_paper(PAPER_ID, [CONTROL_RUN, renumbered_root])

    assert merged is not None
    assert len(merged["outcomes"]) == len(_outcomes(CONTROL_RUN))
    assert {row["source_run"] for row in merged["outcomes"]} == {CONTROL_RUN.name}


def test_every_record_names_the_run_that_produced_it():
    merged = union_paper(PAPER_ID, [CONTROL_RUN, TREATMENT_RUN])

    assert merged is not None
    assert {row["source_run"] for row in merged["outcomes"]} == {
        CONTROL_RUN.name,
        TREATMENT_RUN.name,
    }
    # The union is an ensemble, not a new extraction: nothing is synthesised.
    originals = {
        json.dumps(row, sort_keys=True, ensure_ascii=False)
        for root in (CONTROL_RUN, TREATMENT_RUN)
        for row in _outcomes(root)
    }
    for row in merged["outcomes"]:
        stripped = {k: v for k, v in row.items() if k != "source_run"}
        assert json.dumps(stripped, sort_keys=True, ensure_ascii=False) in originals


def test_experiments_are_merged_by_id_without_duplicates():
    merged = union_paper(PAPER_ID, [CONTROL_RUN, TREATMENT_RUN])

    assert merged is not None
    ids = [row["experiment_id"] for row in merged["experiments"]]
    assert ids == sorted(set(ids), key=ids.index)
    assert len(ids) == len(set(ids))


def test_build_writes_one_result_per_paper_and_a_manifest(tmp_path):
    output_root = tmp_path / "union"

    report = build(
        [CONTROL_RUN, TREATMENT_RUN],
        output_root=output_root,
        paper_ids=[PAPER_ID],
    )

    written = json.loads(
        (output_root / PAPER_ID / "final_result.json").read_text(encoding="utf-8")
    )
    assert written["paper_id"] == PAPER_ID
    assert written["contract_version"] == "compact-1.1.0"
    entry = next(row for row in report["papers"] if row["paper_id"] == PAPER_ID)
    assert entry["outcomes"] == len(written["outcomes"])
    assert entry["by_source_run"] == {
        CONTROL_RUN.name: len(_outcomes(CONTROL_RUN)),
        TREATMENT_RUN.name: len(_outcomes(TREATMENT_RUN)),
    }
    manifest = json.loads(
        (output_root / "union_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["union_version"] == "union-extraction-1.0.0"


def test_a_paper_missing_from_every_run_is_skipped(tmp_path):
    assert union_paper("GP-999", [CONTROL_RUN, TREATMENT_RUN]) is None

    report = build(
        [CONTROL_RUN, TREATMENT_RUN],
        output_root=tmp_path / "union",
        paper_ids=["GP-999"],
    )
    assert report["papers"] == []
    assert not (tmp_path / "union" / "GP-999").exists()
