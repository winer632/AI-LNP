"""Committed artifacts must not change when the code does not.

Repair tasks are written to ``reports/`` and committed, so a rerun that
rewrites them with no code change makes the clean-tree gate useless: a real
regression and an incidental reshuffle look identical in ``git status``.

``_components`` used ``set.pop()`` to choose each component's seed. Under
Python's randomised string hashing that pick differs between processes, so
which candidates were grouped into which task -- and therefore the contents of
every task file -- changed run to run. The tests below pin the ordering
guarantee rather than the specific grouping, so they keep holding if the
grouping rule itself is revised.
"""

from __future__ import annotations

import subprocess
import sys

from src.extraction.build_missing_record_tasks import _components


class _Row:
    def __init__(self, candidate_id: str, source_ids: list[str]):
        self.candidate_id = candidate_id
        self.source_ids = source_ids


class _Inventory:
    def __init__(self, rows):
        self.retained_candidates = rows


def _inventory():
    # Two disjoint components, plus a chain that only links transitively:
    # c1-c2 share s1, c3-c4 share s2, and c5 bridges nothing.
    return _Inventory(
        [
            _Row("c1", ["s1"]),
            _Row("c2", ["s1", "sx"]),
            _Row("c3", ["s2"]),
            _Row("c4", ["s2"]),
            _Row("c5", ["s9"]),
        ]
    )


def test_components_are_independent_of_input_order():
    inventory = _inventory()
    forward = _components(["c1", "c2", "c3", "c4", "c5"], inventory)
    reverse = _components(["c5", "c4", "c3", "c2", "c1"], inventory)
    assert forward == reverse


def test_components_group_transitively_shared_sources():
    groups = _components(["c1", "c2", "c3", "c4", "c5"], _inventory())
    assert ["c1", "c2"] in groups
    assert ["c3", "c4"] in groups
    assert ["c5"] in groups


def test_components_are_stable_across_hash_seeds():
    """The regression itself: same input, different PYTHONHASHSEED, same output.

    Run in subprocesses because PYTHONHASHSEED is fixed at interpreter start
    and cannot be changed from inside a running process.
    """
    program = (
        "from tests.test_artifact_determinism import _components, _inventory;"
        "print(_components(['c1','c2','c3','c4','c5'], _inventory()))"
    )
    outputs = set()
    for seed in ("0", "1", "12345"):
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        )
        outputs.add(completed.stdout.strip())
    assert len(outputs) == 1, f"grouping varied by hash seed: {outputs}"
