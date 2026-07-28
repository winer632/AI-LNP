"""Consolidated gold-gap recovery task checks.

`data/raw/fulltext/` is deliberately not tracked (see `.gitignore`), so the
figure images that two of these tasks are signed against are absent from a fresh
clone. Tests that need those bytes therefore skip when the assets have not been
retrieved locally, instead of failing on a missing file. Everything that can be
checked from tracked data alone - task identity, task-checksum validity, path
portability, and gold-answer leakage - always runs.
"""

from pathlib import Path

import pytest

from src.extraction.build_consolidated_gold_gap_tasks import OUTPUT_ROOT
from src.extraction.consolidated_recovery_contracts import ConsolidatedRecoveryTask
from src.extraction.run_consolidated_gap_recovery import (
    _canonical,
    _sha,
    asset_path,
    load_task,
)


PAPER_IDS = ("GP-004", "GP-006", "GP-008")

REQUIRES_LOCAL_ASSETS = pytest.mark.skipif(
    not all(
        asset_path(asset.image_path).is_file()
        for paper_id in PAPER_IDS
        for asset in ConsolidatedRecoveryTask.model_validate_json(
            (OUTPUT_ROOT / paper_id / "task.json").read_text(encoding="utf-8")
        ).visual_assets
    ),
    reason=(
        "Signed visual assets live under data/raw/fulltext/, which is not tracked "
        "by Git. Fetch the OA packages locally to run this check."
    ),
)


def _raw_task(paper_id: str) -> ConsolidatedRecoveryTask:
    """Parse a task without touching its visual assets on disk."""
    return ConsolidatedRecoveryTask.model_validate_json(
        (OUTPUT_ROOT / paper_id / "task.json").read_text(encoding="utf-8")
    )


def test_exactly_one_signed_task_per_affected_paper():
    tasks = [_raw_task(paper_id) for paper_id in PAPER_IDS]
    assert [task.paper_id for task in tasks] == list(PAPER_IDS)
    assert all(task.permitted_new_outcomes == 2 for task in tasks)


def test_task_checksums_are_valid_without_the_visual_assets():
    for paper_id in PAPER_IDS:
        task = _raw_task(paper_id)
        unsigned = task.model_dump(mode="json", exclude={"task_checksum"})
        assert _sha(_canonical(unsigned)) == task.task_checksum


def test_recorded_asset_paths_are_repository_relative():
    for paper_id in PAPER_IDS:
        for asset in _raw_task(paper_id).visual_assets:
            for recorded in (asset.image_path, asset.source_path):
                assert not Path(recorded).is_absolute(), (
                    f"{paper_id} records the absolute path {recorded!r}; signed "
                    "tasks must stay valid in any checkout."
                )
                assert recorded.startswith("data/"), recorded


@REQUIRES_LOCAL_ASSETS
def test_visual_assets_exist_and_are_checksum_validated():
    for paper_id in PAPER_IDS:
        task = load_task(OUTPUT_ROOT / paper_id / "task.json")
        assert task.visual_assets
        assert all(
            asset_path(asset.image_path).is_file() for asset in task.visual_assets
        )


def test_tasks_do_not_embed_frozen_gold_identifiers():
    for paper_id in PAPER_IDS:
        text = (OUTPUT_ROOT / paper_id / "task.json").read_text(encoding="utf-8")
        assert "GO-" not in text
        assert "EVID-" not in text
