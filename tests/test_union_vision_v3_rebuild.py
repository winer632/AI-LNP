"""The 13/15 root must be rebuildable, and the rebuild must be checked.

13/15 is the only figure that meets the design target, and it had no scripted
rebuild: `codex_union_vision_v3` was referenced by nothing outside its own
directory. An auditor reconstructed the chain from a note inside the manifest.
It worked, which is the point -- nobody else would have found it.

These tests pin the rebuild and, separately, the two ways of combining the
salvage that silently do nothing. That distinction is the whole reason the
manifest alone was insufficient.
"""

from __future__ import annotations

import json

import pytest

from src.config_flags import override
from src.extraction.build_union_vision_v3 import BASE_ROOT, SALVAGE_ROOT, build
from src.extraction.evaluate_final_gold_dynamic import ROOT, evaluate
from src.extraction.union_extraction_results import union_paper

COMMITTED = ROOT / "data/staging/extraction/codex_union_vision_v3"

needs_inputs = pytest.mark.skipif(
    not (BASE_ROOT.exists() and SALVAGE_ROOT.exists()),
    reason="the union base or the salvage root is not present",
)


def _score(root):
    with override(vision_relationship_polarity=True):
        return evaluate(result_roots=[root])


@needs_inputs
def test_the_rebuild_reproduces_the_committed_root_s_measurement(tmp_path):
    with override(record_level_salvage=True):
        manifest = build(
            output_root=tmp_path / "v3",
            salvage_root=tmp_path / "salvage",
        )

    rebuilt = _score(tmp_path / "v3")
    assert rebuilt["recovered"] == 13, "the rebuild no longer reaches 13/15"
    assert round(rebuilt["precision"], 6) == 0.220339
    assert rebuilt["missing_gold_outcome_ids"] == ["GO-017", "GO-018"]
    assert manifest["total_outcomes"] == 59

    if COMMITTED.exists():
        committed = _score(COMMITTED)
        assert rebuilt["recovered"] == committed["recovered"]
        assert round(rebuilt["precision"], 9) == round(committed["precision"], 9)
        assert (
            rebuilt["evidence_accuracy"]["checked"]
            == committed["evidence_accuracy"]["checked"]
        )


@needs_inputs
def test_the_manifest_cites_only_paths_the_repository_holds(tmp_path):
    """A rebuild instruction that names a path nobody has is not one."""
    with override(record_level_salvage=True):
        manifest = build(
            output_root=tmp_path / "v3", salvage_root=tmp_path / "salvage"
        )
    for key in ("base_root", "salvage_source_runs", "vision_observations"):
        entries = manifest[key]
        for entry in [entries] if isinstance(entries, str) else entries:
            assert not entry.startswith("/"), f"{key} cites an absolute path"
            assert (ROOT / entry).exists(), f"{key} cites a missing {entry}"


@needs_inputs
def test_appending_the_salvage_as_another_root_changes_nothing():
    """The mistake the scripted rebuild exists to prevent.

    _result_path returns the first root holding a paper, so the salvage's
    GP-004 is never read when it is appended. Someone reading the manifest and
    reaching for --run-root would get 11/15 and no indication why.
    """
    with override(record_level_salvage=True):
        base = evaluate(result_roots=[BASE_ROOT])
        appended = evaluate(result_roots=[BASE_ROOT, SALVAGE_ROOT])
    assert appended["recovered"] == base["recovered"] == 11
    assert "GO-002" in appended["missing_gold_outcome_ids"]


@needs_inputs
def test_prepending_the_salvage_trades_one_outcome_for_another():
    """The other way to get it wrong: GP-004 is replaced rather than merged."""
    with override(record_level_salvage=True):
        prepended = evaluate(result_roots=[SALVAGE_ROOT, BASE_ROOT])
    assert prepended["recovered"] == 11
    missing = prepended["missing_gold_outcome_ids"]
    assert "GO-001" in missing and "GO-002" not in missing


@needs_inputs
def test_the_merge_is_what_recovers_go_002(tmp_path):
    """Union GP-004 without any vision merged: 11/15 becomes 12/15."""
    import shutil

    destination = tmp_path / "merged"
    shutil.copytree(BASE_ROOT, destination)
    merged = union_paper("GP-004", [BASE_ROOT, SALVAGE_ROOT])
    (destination / "GP-004" / "final_result.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with override(record_level_salvage=True):
        scored = evaluate(result_roots=[destination])
    assert scored["recovered"] == 12
    assert "GO-002" not in scored["missing_gold_outcome_ids"]


def test_a_colliding_outcome_id_is_renamed_when_runs_are_unioned():
    """The duplicate-id fix had no coverage at all.

    Runs number their own records from O1, so a union of two runs collides.
    Before cb3b3a7 the evaluator credited one record's match to every record
    sharing its id, which inflated precision on every union root. The fix
    survived all 637 tests when removed; this is the test that would have
    caught it.
    """
    merged = union_paper("GP-004", [BASE_ROOT, SALVAGE_ROOT])
    if merged is None:
        pytest.skip("GP-004 is not present in the union base")
    ids = [str(row.get("outcome_id")) for row in merged["outcomes"]]
    assert len(ids) == len(set(ids)), "the union emitted colliding outcome ids"

    renamed = [row for row in merged["outcomes"] if "source_outcome_id" in row]
    assert renamed, "no record was renamed, so this union cannot prove the rule"
    for row in renamed:
        assert row["outcome_id"] != row["source_outcome_id"]
        assert row["source_outcome_id"] in {
            "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9",
        } or row["source_outcome_id"].startswith("O")
