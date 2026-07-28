"""Resolve the root directory that generated artifacts are written to.

Reporting and freezing steps normally write next to their inputs, under the
repository root. That is correct for a real run, but it made the test suite
non-idempotent: importing and calling ``finalize()`` or ``freeze()`` rewrote
tracked files under ``reports/`` with a fresh ``frozen_at`` timestamp, so
``git status`` was dirty after every ``pytest`` run and CI could never assert a
clean tree.

This module gives those steps an injectable output root.

Resolution order (highest priority first):

1. an explicit ``root=`` argument -- the injection point production callers and
   tests should prefer, e.g. ``finalize(output_root=tmp_path)``;
2. the ``AI_LNP_OUTPUT_ROOT`` environment variable;
3. a per-process scratch directory when running under pytest, detected via
   ``PYTEST_CURRENT_TEST``, so a plain ``pytest -q`` never mutates the working
   tree even for tests that call the writers with no arguments;
4. the repository root.

Inputs are deliberately *not* redirected. Only writes move.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping


__all__ = [
    "OUTPUT_ROOT_ENV",
    "PYTEST_ENV",
    "REPO_ROOT",
    "artifact_path",
    "output_root",
    "repo_root",
]


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT_ENV = "AI_LNP_OUTPUT_ROOT"
PYTEST_ENV = "PYTEST_CURRENT_TEST"

_SCRATCH_NAME = "ai-lnp-test-artifacts"
_SCRATCH_REGISTERED = False


def repo_root() -> Path:
    """Absolute path of the repository checkout."""
    return REPO_ROOT


def _scratch_root() -> Path:
    """Per-process scratch tree, removed when the process exits.

    Keyed by PID so concurrent runs (pytest-xdist workers, for example) cannot
    overwrite each other. Only this auto-created directory is ever deleted; an
    output root the caller configured explicitly is left alone.
    """
    global _SCRATCH_REGISTERED
    path = Path(tempfile.gettempdir()) / f"{_SCRATCH_NAME}-{os.getpid()}"
    path.mkdir(parents=True, exist_ok=True)
    if not _SCRATCH_REGISTERED:
        atexit.register(shutil.rmtree, path, True)
        _SCRATCH_REGISTERED = True
    return path


def output_root(
    root: Path | str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Return the directory generated artifacts should be written under."""
    if root is not None:
        return Path(root)

    environ = os.environ if env is None else env

    configured = (environ.get(OUTPUT_ROOT_ENV) or "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path)

    if environ.get(PYTEST_ENV):
        return _scratch_root()

    return REPO_ROOT


def artifact_path(
    *parts: str,
    root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    create_parents: bool = True,
) -> Path:
    """Build a path under :func:`output_root`, creating parent directories.

    ``artifact_path("reports", "extraction", "metrics.json")`` returns
    ``<output root>/reports/extraction/metrics.json`` and makes sure
    ``<output root>/reports/extraction`` exists, because a redirected output
    root starts empty.
    """
    path = output_root(root, env=env).joinpath(*parts)
    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path
