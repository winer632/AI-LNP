"""The clean-tree mechanism, pinned by tests instead of only by a CI step.

``src/output_paths.py`` is the reason ``pytest`` no longer rewrites tracked
files under ``reports/`` with a fresh timestamp. CI asserts the *outcome* --
``git status --porcelain`` is empty after the suite -- but nothing asserted the
*rule* that produces it, so a regression in the resolution order would only
ever surface as a red CI job on some unrelated change, with the real cause a
directory away.

Every case injects ``env=`` rather than mutating ``os.environ``, so the
resolution order is tested independently of how this suite happens to be run.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.output_paths import (
    OUTPUT_ROOT_ENV,
    PYTEST_ENV,
    REPO_ROOT,
    artifact_path,
    output_root,
    repo_root,
)


# A plausible value for PYTEST_CURRENT_TEST; only its truthiness is consulted.
UNDER_PYTEST = {PYTEST_ENV: "tests/test_output_paths.py::test_x (call)"}


def _is_inside_repo(path: Path) -> bool:
    return path.resolve() == REPO_ROOT or REPO_ROOT in path.resolve().parents


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------


def test_explicit_root_beats_every_environment_setting(tmp_path):
    """``root=`` is the injection point; nothing in the environment overrides it."""
    env = {OUTPUT_ROOT_ENV: str(tmp_path / "from-env"), **UNDER_PYTEST}

    assert output_root(tmp_path / "explicit", env=env) == tmp_path / "explicit"


def test_environment_root_is_used_when_no_root_is_passed(tmp_path):
    env = {OUTPUT_ROOT_ENV: str(tmp_path / "configured")}

    assert output_root(env=env) == tmp_path / "configured"


def test_environment_root_beats_the_pytest_scratch_fallback(tmp_path):
    """Setting the variable must still work while running under pytest."""
    env = {OUTPUT_ROOT_ENV: str(tmp_path / "configured"), **UNDER_PYTEST}

    assert output_root(env=env) == tmp_path / "configured"


def test_relative_environment_root_resolves_under_the_repo_not_the_cwd(
    tmp_path, monkeypatch
):
    """A relative value means "inside the checkout", wherever the shell sits."""
    monkeypatch.chdir(tmp_path)

    resolved = output_root(env={OUTPUT_ROOT_ENV: "build/artifacts"})

    assert resolved == REPO_ROOT / "build/artifacts"
    assert resolved != Path("build/artifacts").resolve()


def test_home_relative_environment_root_is_expanded():
    resolved = output_root(env={OUTPUT_ROOT_ENV: "~/ai-lnp-artifacts"})

    assert resolved.is_absolute()
    assert resolved == Path.home() / "ai-lnp-artifacts"


def test_blank_environment_root_is_ignored_rather_than_pointing_at_the_repo():
    """``AI_LNP_OUTPUT_ROOT=" "`` must not resolve to a directory in the checkout.

    Without the strip-and-test, a blank value is truthy, becomes a relative
    path, and lands back inside ``REPO_ROOT`` -- silently re-enabling exactly
    the dirty-tree behaviour this module exists to prevent.
    """
    env = {OUTPUT_ROOT_ENV: "   ", **UNDER_PYTEST}

    resolved = output_root(env=env)

    assert not _is_inside_repo(resolved), f"{resolved} is inside the checkout"


def test_under_pytest_the_output_root_is_outside_the_checkout():
    """The headline guarantee: a writer called with no arguments cannot dirty the tree."""
    resolved = output_root(env=UNDER_PYTEST)

    assert not _is_inside_repo(resolved), f"{resolved} is inside the checkout"
    assert resolved.is_dir()


def test_the_pytest_scratch_root_is_keyed_by_process_so_workers_cannot_collide():
    resolved = output_root(env=UNDER_PYTEST)

    assert str(os.getpid()) in resolved.name
    assert output_root(env=UNDER_PYTEST) == resolved


def test_without_pytest_or_environment_the_root_is_the_repository():
    """A real run still writes next to its inputs; only tests are redirected."""
    assert output_root(env={}) == REPO_ROOT


def test_repo_root_is_the_checkout_that_holds_src_and_pytest_ini():
    root = repo_root()

    assert (root / "src" / "output_paths.py").is_file()
    assert (root / "pytest.ini").is_file()
    assert root == REPO_ROOT


# ---------------------------------------------------------------------------
# artifact_path
# ---------------------------------------------------------------------------


def test_artifact_path_joins_the_parts_under_the_output_root(tmp_path):
    path = artifact_path("reports", "extraction", "metrics.json", root=tmp_path)

    assert path == tmp_path / "reports" / "extraction" / "metrics.json"


def test_artifact_path_creates_the_parent_directory_but_not_the_file(tmp_path):
    """A redirected output root starts empty, so the writer needs its parents made."""
    path = artifact_path("reports", "extraction", "metrics.json", root=tmp_path)

    assert path.parent.is_dir()
    assert not path.exists()


def test_artifact_path_can_be_asked_not_to_create_parents(tmp_path):
    path = artifact_path(
        "reports", "extraction", "metrics.json", root=tmp_path, create_parents=False
    )

    assert not path.parent.exists()


def test_artifact_path_under_pytest_never_lands_in_the_repository():
    """The exact regression: ``reports/...`` resolved against the checkout."""
    path = artifact_path("reports", "extraction", "metrics.json", env=UNDER_PYTEST)

    assert not _is_inside_repo(path.parent), f"{path} is inside the checkout"
    assert path.name == "metrics.json"
