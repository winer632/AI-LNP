"""The unpaid half of the repair route, run over an arbitrary extraction output.

Two things are pinned here.

Importing this module must do nothing. It used to read ``sys.argv[1]`` and
``mkdir`` an output root at module level, which made it unimportable from a
test runner and made merely importing it write to the filesystem.

``run`` must stay parameterised. The value of this variant over
``run_enforced_compact_workflow_local`` is that its roots are arguments, so a
harness comparison can point it at any run; a version that reaches for a
hard-coded root instead is useless for that.

Nothing here calls a model -- that is the point of the module -- so the
integration test runs against the real committed control run and packets.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "data/staging/extraction/codex_control_compact_v1"
PACKET_ROOT = REPO_ROOT / "data/staging/rag/compact_api_packets_v1"
MODULE = "src.extraction.build_repair_route_local"


def _import_in_subprocess(argv: list[str]) -> subprocess.CompletedProcess:
    script = (
        "import sys, json\n"
        f"sys.argv = {argv!r}\n"
        f"import {MODULE} as m\n"
        "print(json.dumps({'callable': callable(m.run)}))\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_importing_the_module_does_not_read_argv():
    """``python -c "import ..."`` leaves argv at ``['-c']``; that must be enough."""
    completed = _import_in_subprocess(["-c"])

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip())["callable"] is True


def test_importing_the_module_creates_no_directories(tmp_path):
    """Import must not act on argv, including by creating an output root."""
    would_be_roots = [
        tmp_path / "results",
        tmp_path / "packets",
        tmp_path / "output",
    ]
    completed = _import_in_subprocess(
        ["build_repair_route_local", *[str(path) for path in would_be_roots]]
    )

    assert completed.returncode == 0, completed.stderr
    for path in would_be_roots:
        assert not path.exists(), f"import created {path}"
    assert list(tmp_path.iterdir()) == []


def _snapshot(root: Path) -> set[Path]:
    return set(root.rglob("*"))


def test_run_writes_the_route_under_the_output_root_it_was_given(tmp_path):
    from src.extraction.build_repair_route_local import run

    before_results = _snapshot(RESULT_ROOT)
    before_packets = _snapshot(PACKET_ROOT)
    output_root = tmp_path / "route"
    totals = run(RESULT_ROOT, PACKET_ROOT, output_root, only_papers={"GP-004"})

    # The inputs are committed data; the route reads them and writes elsewhere.
    assert _snapshot(RESULT_ROOT) == before_results
    assert _snapshot(PACKET_ROOT) == before_packets

    assert totals["papers"] == 1
    paper_dir = output_root / "GP-004"
    routing = json.loads((paper_dir / "routing.json").read_text(encoding="utf-8"))
    inventory = json.loads((paper_dir / "inventory.json").read_text(encoding="utf-8"))
    assert routing["paper_id"] == "GP-004"
    assert inventory["paper_id"] == "GP-004"
    tasks = sorted(paper_dir.glob("task_*.json"))
    assert len(tasks) == totals["text_tasks"]


def test_only_papers_restricts_the_sweep(tmp_path):
    from src.extraction.build_repair_route_local import run

    output_root = tmp_path / "route"
    run(RESULT_ROOT, PACKET_ROOT, output_root, only_papers={"GP-004"})

    assert sorted(p.name for p in output_root.iterdir()) == ["GP-004"]


def test_a_paper_with_no_result_is_skipped_rather_than_raising(tmp_path):
    from src.extraction.build_repair_route_local import run

    empty_results = tmp_path / "no_results"
    empty_results.mkdir()

    totals = run(empty_results, PACKET_ROOT, tmp_path / "route")

    assert totals == {"text_tasks": 0, "visual_referrals": 0, "papers": 0}
