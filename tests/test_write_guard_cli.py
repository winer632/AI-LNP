"""Asking an artifact-rewriting entry point what it does must not do it.

Every module listed here regenerates committed artifacts in place. Before the
``--confirm-write`` guards existed, running one of them to see what it was --
or by habit, with no arguments -- rewrote tracked files, and 41 of them were
committed unreviewed. The guards are the fix; these tests are what keeps them.

Two invocations are checked for each entry point:

``--help``      must print usage, exit 0, and touch nothing. Import happens
                before argparse sees anything, so this also catches a
                module-level ``mkdir`` or write sneaking back in.
``(no args)``   must fail with argparse's exit status 2 and say what is
                missing, having written nothing.

The subprocess environment deliberately has ``PYTEST_CURRENT_TEST`` and
``AI_LNP_OUTPUT_ROOT`` removed. Redirecting writes away from the checkout would
make a missing guard invisible to the tree comparison, which is the one thing
these tests exist to see.

The set of entry points is discovered from the source rather than typed out, so
a case cannot quietly disappear: :func:`test_every_confirm_write_entry_point_is_covered`
fails if a module gains or loses the guard without this list following.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

# The audited entry points, module path as ``python -m`` takes it.
CONFIRM_WRITE_ENTRY_POINTS = (
    "src.extraction.apply_day8_human_adjudication",
    "src.extraction.build_g1_review",
    "src.extraction.build_g1_v3_field_review",
    "src.extraction.build_object_vision_review",
    "src.extraction.build_union_entity_prepass",
    "src.extraction.build_union_vision_v3",
    "src.extraction.evaluate_day5_afternoon",
    "src.extraction.evaluate_final_gold_dynamic",
    "src.extraction.evaluate_outcome_inventory_gold",
    "src.extraction.finalize_day5_gp008_repair",
    "src.extraction.finalize_g1",
    "src.extraction.freeze_g1_v3_boundaries",
    "src.extraction.merge_structured_view_pass",
    "src.extraction.prepare_day5_g1_repair",
    "src.extraction.run_enforced_compact_workflow_local",
    "src.extraction.run_outcome_coverage_gold",
    "src.rag.analyze_g1_errors",
    "src.rag.finalize_day7_afternoon",
    "src.rag.finalize_day7_v41",
    "src.rag.finalize_fulltext_g1",
    "src.rag.ingestion",
    "src.screening.complete_day4_gold_annotations",
    "src.screening.retrieve_gold_oa_packages",
)

# Git-ignored tool and tooling state, none of it repository content. ``.claude``
# in particular can hold sibling git worktrees that other processes are writing
# to, which would make the comparison both slow and flaky.
_IGNORED_DIRS = {".git", ".claude", ".venv", "venv", "__pycache__",
                 ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}


def _discover_confirm_write_modules() -> set[str]:
    found = set()
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        if '"--confirm-write"' in path.read_text(encoding="utf-8"):
            found.add(".".join(path.relative_to(REPO_ROOT).with_suffix("").parts))
    return found


def _snapshot() -> dict[str, object]:
    """Size and mtime of every file in the checkout, plus every directory.

    Names are matched *relative to* the checkout. Matching against the absolute
    path would let a directory name that happens to appear above the checkout --
    a repository living under ``.claude/worktrees/...``, say -- exclude
    everything and leave a comparison that can never fail.
    """
    tree: dict[str, object] = {}
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [name for name in dirnames if name not in _IGNORED_DIRS]
        directory = Path(dirpath)
        for name in dirnames:
            tree[str((directory / name).relative_to(REPO_ROOT))] = "dir"
        for name in filenames:
            if name in _IGNORED_DIRS:  # a linked worktree's .git is a file
                continue
            path = directory / name
            try:
                stat = path.stat()
            except OSError:  # pragma: no cover - a race we do not need to model
                continue
            tree[str(path.relative_to(REPO_ROOT))] = (stat.st_size, stat.st_mtime_ns)
    return tree


def _describe(before: dict[str, object], after: dict[str, object]) -> str:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(key for key in set(before) & set(after) if before[key] != after[key])
    return f"added={added[:10]} removed={removed[:10]} changed={changed[:10]}"


def _invoke(module: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = ""
    # Keep the run from leaving .pyc files behind, so any new file is a real one.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("AI_LNP_OUTPUT_ROOT", None)
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_the_snapshot_actually_sees_the_repository():
    """Guard: an over-broad ignore rule makes every comparison below vacuous.

    ``after == before`` on two empty dicts passes against any implementation,
    correct or not. Assert the snapshot has content before trusting it.
    """
    tree = _snapshot()

    assert "pytest.ini" in tree
    assert "src/output_paths.py" in tree
    assert "reports" in tree and tree["reports"] == "dir"
    assert len(tree) > 500, f"snapshot saw only {len(tree)} entries"


def test_every_confirm_write_entry_point_is_covered():
    """The parametrised cases below must not be able to silently empty out."""
    assert _discover_confirm_write_modules() == set(CONFIRM_WRITE_ENTRY_POINTS)
    assert len(CONFIRM_WRITE_ENTRY_POINTS) == 23


@pytest.mark.parametrize("module", CONFIRM_WRITE_ENTRY_POINTS)
def test_help_explains_the_command_and_writes_nothing(module):
    before = _snapshot()

    completed = _invoke(module, "--help")

    after = _snapshot()
    assert completed.returncode == 0, completed.stderr
    assert "--confirm-write" in completed.stdout
    assert "usage:" in completed.stdout
    assert after == before, f"`{module} --help` wrote to the repo: {_describe(before, after)}"


@pytest.mark.parametrize("module", CONFIRM_WRITE_ENTRY_POINTS)
def test_running_with_no_arguments_refuses_and_writes_nothing(module):
    before = _snapshot()

    completed = _invoke(module)

    after = _snapshot()
    assert completed.returncode == 2, (
        f"`{module}` with no arguments should have been refused by argparse "
        f"(exit 2); got {completed.returncode}.\nstdout: {completed.stdout[:2000]}"
    )
    assert "--confirm-write is required" in completed.stderr
    assert completed.stdout == ""
    assert after == before, f"`{module}` wrote to the repo: {_describe(before, after)}"
