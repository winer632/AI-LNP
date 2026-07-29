"""Tests for the central feature-flag registry.

These cover the contract other modules rely on: configured defaults, the
environment override, alias/priority lookup, the compact-route gate actually
mirroring its YAML file, and the deliberate failure on unknown flag names.
"""

import textwrap

import pytest
import yaml

from src import config_flags
from src.config_flags import (
    CONFIG_PATH_ENV,
    FlagConfigError,
    UnknownFlagError,
    all_flags,
    describe_flag,
    env_var_for,
    is_enabled,
    known_flags,
    override,
    registry,
    reload_flags,
)


DESIGN_PRIORITIES = ("P0-a", "P0-b", "P1", "P2", "P3")


@pytest.fixture(autouse=True)
def _clean_registry_cache():
    reload_flags()
    yield
    reload_flags()


@pytest.fixture
def custom_registry(tmp_path, monkeypatch):
    """Point the registry at a throwaway YAML file."""

    def _write(body: str):
        path = tmp_path / "flags.yaml"
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        monkeypatch.setenv(CONFIG_PATH_ENV, str(path))
        reload_flags()
        return path

    return _write


# --------------------------------------------------------------------------- #
# Registry contents
# --------------------------------------------------------------------------- #


def test_every_design_priority_has_a_registered_flag():
    priorities = {flag.priority for flag in registry().values()}
    missing = set(DESIGN_PRIORITIES) - priorities
    assert not missing, f"No feature flag registered for {sorted(missing)}"


def test_named_capability_flags_are_registered():
    for name in ("uniparse_ingestion", "full_evidence_view", "precision_metrics"):
        assert name in known_flags()


def test_a_flag_may_only_ship_enabled_with_a_recorded_rationale():
    """Rollout policy, restated now that some flags are on.

    The original rule was that every design-priority flag ships disabled. That
    held while nothing had been measured. It cannot hold now without also
    meaning "never turn anything on", so the rule it becomes is: a flag that
    defaults on must say why, and a flag that is still planned must be off.
    """
    for flag in registry().values():
        if flag.default is True:
            assert getattr(flag, "rationale", None), (
                f"{flag.name} ships enabled but records no rationale"
            )
        if flag.status == "planned":
            assert flag.default is False, (
                f"{flag.name} is still planned and must not default on"
            )


def test_flags_are_documented_and_declare_integration_points():
    for flag in registry().values():
        assert flag.description, f"{flag.name} has no description"
        if flag.status == "planned":
            # Nothing reads a planned flag yet, and claiming otherwise is what
            # made this list untrustworthy in the first place. An empty list is
            # the honest answer; the companion test pins it to the real call
            # sites, so it cannot stay empty once code starts reading it.
            assert not flag.integration_points, (
                f"{flag.name} is planned but claims integration points"
            )
            continue
        assert flag.integration_points, f"{flag.name} has no integration points"


def test_every_flag_has_a_distinct_environment_variable():
    """No two flags share a variable, and the conventional name always works.

    The registry supports `env:` for a flag that a deployment already sets
    under its own name -- uniparse_ingestion declares UNIPARSE_ENABLED -- so
    requiring every env_var to carry the AI_LNP_FLAG_ prefix would forbid the
    feature. What must hold is that the prefixed name stays available on every
    flag, so a caller who knows only the convention can always reach one, and
    that no variable governs two flags.
    """
    every = [name for flag in registry().values() for name in flag.env_vars]
    assert len(every) == len(set(every)), "a variable governs more than one flag"
    # env_var_for returns the declared name when there is one, so build the
    # conventional name here rather than asking for "the" variable.
    for flag in registry().values():
        conventional = "AI_LNP_FLAG_" + flag.name.upper().replace("-", "_")
        assert conventional in flag.env_vars, (
            f"{flag.name} cannot be reached by the conventional variable"
        )


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


def test_defaults_apply_when_no_environment_override_is_present():
    # Compare against what each flag registers rather than a literal, so a
    # rollout decision changes one file instead of breaking this test.
    for name in known_flags():
        assert is_enabled(name, env={}) is describe_flag(name).default


def test_all_flags_reports_every_registered_flag():
    values = all_flags(env={})
    assert set(values) == set(known_flags())
    assert values == {name: describe_flag(name).default for name in known_flags()}


def test_compact_route_default_mirrors_its_yaml_gate():
    from src.config_flags import REPO_ROOT

    route = yaml.safe_load(
        (REPO_ROOT / "config" / "extraction" / "compact_route_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    flag = describe_flag("compact_route")
    assert flag.default is route["activation_gate"]["active"]
    assert flag.source.endswith("#activation_gate.active")
    assert is_enabled("compact_route", env={}) is route["activation_gate"]["active"]


def test_yaml_source_default_is_read_live(custom_registry, tmp_path, monkeypatch):
    gate = tmp_path / "gate.yaml"
    gate.write_text("activation_gate:\n  active: true\n", encoding="utf-8")
    custom_registry(
        f"""
        flags:
          gated:
            default: false
            description: mirrors another file
            integration_points: [somewhere]
            yaml_source:
              path: {gate}
              key: activation_gate.active
        """
    )
    assert is_enabled("gated", env={}) is True


# --------------------------------------------------------------------------- #
# Environment overrides
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "Yes", "on", "enabled", " y "])
def test_truthy_environment_values_enable_a_flag(raw):
    assert is_enabled("uniparse_ingestion", env={"AI_LNP_FLAG_UNIPARSE_INGESTION": raw})


@pytest.mark.parametrize("raw", ["0", "false", "No", "OFF", "disabled"])
def test_falsy_environment_values_disable_a_flag(raw):
    assert not is_enabled("compact_route", env={"AI_LNP_FLAG_COMPACT_ROUTE": raw})


def test_environment_override_beats_the_configured_default():
    for name in known_flags():
        variable = env_var_for(name)
        configured = describe_flag(name).default
        opposite = "false" if configured else "true"
        assert is_enabled(name, env={}) is configured
        assert is_enabled(name, env={variable: opposite}) is (not configured)


def test_empty_environment_value_falls_back_to_the_default():
    # `AI_LNP_FLAG_X= command` must not silently mean "off".
    for name, blank in (("compact_route", ""), ("uniparse_ingestion", "  ")):
        env = {env_var_for(name): blank}
        assert is_enabled(name, env=env) is describe_flag(name).default


def test_unparseable_environment_value_is_rejected():
    with pytest.raises(FlagConfigError) as excinfo:
        is_enabled("precision_metrics", env={"AI_LNP_FLAG_PRECISION_METRICS": "maybe"})
    assert "AI_LNP_FLAG_PRECISION_METRICS" in str(excinfo.value)


def test_real_process_environment_is_read_per_call(monkeypatch):
    assert is_enabled("local_vlm_vision") is False
    monkeypatch.setenv("AI_LNP_FLAG_LOCAL_VLM_VISION", "1")
    assert is_enabled("local_vlm_vision") is True
    monkeypatch.delenv("AI_LNP_FLAG_LOCAL_VLM_VISION")
    assert is_enabled("local_vlm_vision") is False


def test_declared_env_name_wins_over_the_conventional_one(custom_registry):
    custom_registry(
        """
        flags:
          special:
            default: false
            env: MY_SPECIAL_SWITCH
            description: uses its own variable
            integration_points: [somewhere]
        """
    )
    assert env_var_for("special") == "MY_SPECIAL_SWITCH"
    assert is_enabled("special", env={"MY_SPECIAL_SWITCH": "on"}) is True
    # The conventional name keeps working as a fallback.
    assert is_enabled("special", env={"AI_LNP_FLAG_SPECIAL": "on"}) is True
    assert is_enabled(
        "special",
        env={"MY_SPECIAL_SWITCH": "off", "AI_LNP_FLAG_SPECIAL": "on"},
    ) is False


# --------------------------------------------------------------------------- #
# Aliases and priority ids
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("P0-a", "full_evidence_view"),
        ("p0a", "full_evidence_view"),
        ("P0-b", "precision_metrics"),
        ("P1", "uniparse_ingestion"),
        ("P2", "local_vlm_vision"),
        ("P3", "candidate_slot_enforcement"),
        ("activation_gate", "compact_route"),
    ],
)
def test_flags_are_reachable_by_alias_and_priority_id(alias, canonical):
    assert describe_flag(alias).name == canonical
    assert is_enabled(alias, env={}) is is_enabled(canonical, env={})


def test_lookup_ignores_case_and_dash_underscore_differences():
    assert describe_flag("UNIPARSE_INGESTION").name == "uniparse_ingestion"
    assert describe_flag("uniparse-ingestion").name == "uniparse_ingestion"
    assert describe_flag(" P0_A ").name == "full_evidence_view"


# --------------------------------------------------------------------------- #
# Unknown flags
# --------------------------------------------------------------------------- #


def test_unknown_flag_raises_by_default():
    with pytest.raises(UnknownFlagError) as excinfo:
        is_enabled("definitely_not_a_flag")
    message = str(excinfo.value)
    assert "definitely_not_a_flag" in message
    assert "config/feature_flags.yaml" in message


def test_unknown_flag_is_a_keyerror_subclass():
    assert issubclass(UnknownFlagError, KeyError)
    with pytest.raises(KeyError):
        is_enabled("definitely_not_a_flag")


@pytest.mark.parametrize("fallback", [True, False])
def test_unknown_flag_uses_an_explicit_default_when_given(fallback):
    assert is_enabled("not_registered_yet", default=fallback) is fallback


def test_explicit_default_never_overrides_a_registered_flag():
    # A registered flag always wins, so a stale default= cannot silently mask it.
    for name in known_flags():
        configured = describe_flag(name).default
        assert is_enabled(name, default=not configured, env={}) is configured


def test_describe_flag_rejects_unknown_names():
    with pytest.raises(UnknownFlagError):
        describe_flag("definitely_not_a_flag")


# --------------------------------------------------------------------------- #
# override() helper
# --------------------------------------------------------------------------- #


def test_override_sets_and_restores_the_process_environment(monkeypatch):
    monkeypatch.delenv("AI_LNP_FLAG_FULL_EVIDENCE_VIEW", raising=False)
    with override(full_evidence_view=True):
        assert is_enabled("full_evidence_view") is True
    assert is_enabled("full_evidence_view") is False
    assert "AI_LNP_FLAG_FULL_EVIDENCE_VIEW" not in config_flags.os.environ


def test_override_restores_a_pre_existing_value(monkeypatch):
    monkeypatch.setenv("AI_LNP_FLAG_FULL_EVIDENCE_VIEW", "true")
    with override(full_evidence_view=False):
        assert is_enabled("full_evidence_view") is False
    assert is_enabled("full_evidence_view") is True


def test_override_accepts_priority_ids(monkeypatch):
    monkeypatch.delenv("AI_LNP_FLAG_UNIPARSE_INGESTION", raising=False)
    with override(p1=True):
        assert is_enabled("uniparse_ingestion") is True
    with override(p1=False):
        assert is_enabled("uniparse_ingestion") is False
    assert is_enabled("uniparse_ingestion") is describe_flag("uniparse_ingestion").default


def test_override_restores_after_an_exception(monkeypatch):
    monkeypatch.delenv("AI_LNP_FLAG_PRECISION_METRICS", raising=False)
    with pytest.raises(RuntimeError):
        with override(precision_metrics=True):
            raise RuntimeError("boom")
    assert is_enabled("precision_metrics") is describe_flag("precision_metrics").default


# --------------------------------------------------------------------------- #
# Malformed registries
# --------------------------------------------------------------------------- #


def test_missing_default_is_rejected(custom_registry):
    custom_registry(
        """
        flags:
          broken:
            description: no default declared
        """
    )
    with pytest.raises(FlagConfigError, match="no 'default'"):
        known_flags()


def test_duplicate_alias_is_rejected(custom_registry):
    custom_registry(
        """
        flags:
          first:
            default: false
            aliases: [shared]
          second:
            default: false
            aliases: [shared]
        """
    )
    with pytest.raises(FlagConfigError, match="maps to both"):
        known_flags()


def test_missing_registry_file_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv(CONFIG_PATH_ENV, str(tmp_path / "absent.yaml"))
    reload_flags()
    with pytest.raises(FlagConfigError, match="not found"):
        known_flags()


def test_registry_is_reloaded_when_the_file_changes(custom_registry):
    path = custom_registry(
        """
        flags:
          rolling:
            default: false
            description: starts off
            integration_points: [somewhere]
        """
    )
    assert is_enabled("rolling", env={}) is False
    path.write_text(
        "flags:\n  rolling:\n    default: true\n    description: now on\n",
        encoding="utf-8",
    )
    assert is_enabled("rolling", env={}) is True

# ---------------------------------------------------------------------------
# Design-document alignment
# ---------------------------------------------------------------------------


# docs/ai_lnp_recall_improvement_design_zh.html assigns these priorities. They
# were transposed once already (uniparse labelled P0-a), which silently pointed
# is_enabled("P0-a") at the wrong capability, so pin them explicitly.
DESIGN_PRIORITY_OWNER = {
    "P0-a": "full_evidence_view",
    "P0-b": "precision_metrics",
    "P1": "uniparse_ingestion",
    "P2": "local_vlm_vision",
    "P3": "candidate_slot_enforcement",
}


@pytest.mark.parametrize("priority,canonical", sorted(DESIGN_PRIORITY_OWNER.items()))
def test_priority_ids_match_the_design_document(priority, canonical):
    assert describe_flag(priority).name == canonical
    assert describe_flag(canonical).priority == priority


def test_declared_integration_points_really_read_the_flag():
    """A flag's integration_points must be files that call is_enabled for it.

    The list used to be aspirational: it named the files a capability was
    expected to touch, and 10 of 16 entries named a file that never read the
    flag, while every real consumer of full_evidence_view, local_vlm_vision
    and compact_route went undeclared. Anyone auditing which code a flag
    governs got the wrong answer from the registry.

    Constants count. Several call sites pass CANDIDATE_SLOT_FLAG or P2_FLAG
    rather than a string literal, so a literal-only search under-reports.
    """
    import re
    from pathlib import Path

    source_root = Path(__file__).resolve().parents[1] / "src"
    constants: dict[str, str] = {}
    for path in source_root.rglob("*.py"):
        for match in re.finditer(
            r'^([A-Z][A-Z0-9_]*)\s*=\s*"([a-z_]+)"',
            path.read_text(encoding="utf-8"),
            re.MULTILINE,
        ):
            constants[match.group(1)] = match.group(2)

    readers: dict[str, set[str]] = {}
    for path in source_root.rglob("*.py"):
        if path.name == "config_flags.py":
            continue
        for match in re.finditer(
            r"is_enabled\(\s*(?:\"([^\"]+)\"|'([^']+)'|([A-Z][A-Z0-9_]*))",
            path.read_text(encoding="utf-8"),
        ):
            name = match.group(1) or match.group(2) or constants.get(match.group(3))
            if name:
                readers.setdefault(name, set()).add(
                    str(path.relative_to(source_root.parent))
                )

    for flag in registry().values():
        declared = set(flag.integration_points)
        actual = readers.get(flag.name, set())
        assert declared == actual, (
            f"{flag.name}: declared {sorted(declared)} but "
            f"is_enabled is called from {sorted(actual)}"
        )


def test_a_rationale_that_quotes_a_recall_figure_reproduces_it():
    """Rationale content, not just its presence.

    test_a_flag_may_only_ship_enabled_with_a_recorded_rationale asserts the
    string is non-empty. An audit then found seventeen quoted numbers that no
    longer reproduced, including one rationale that stated the exact opposite
    of what the repository's own headline artifact demonstrates -- because the
    commit that reached 13/15 never updated the flag file. Every one of those
    survived a green suite.

    A rationale cannot be checked in general. What can be checked is that any
    "N/15" it quotes is a figure some committed configuration actually
    produces, which catches a stale headline -- the failure that happened.
    """
    import re

    from src.extraction.evaluate_final_gold_dynamic import ROOT, evaluate

    # Every committed extraction root, not just the headline ones: a rationale
    # may legitimately cite an ablation figure from a single run, and a set
    # narrowed to the union roots would reject those as unreachable. This test
    # is meant to catch a stale headline, not to forbid citing a measurement.
    achievable = set()
    extraction = ROOT / "data/staging/extraction"
    roots = [None] + [
        [path] for path in sorted(extraction.iterdir())
        if path.is_dir() and any(path.glob("GP-*/*result*.json"))
    ]
    for candidate in roots:
        for polarity in (False, True):
            with override(vision_relationship_polarity=polarity):
                try:
                    recovered = evaluate(result_roots=candidate)["recovered"]
                except Exception:  # noqa: BLE001 - an unscorable root is not this test's concern
                    continue
                achievable.add(f"{recovered}/15")

    # A rationale may legitimately cite a before-state: "the structured view
    # takes the default run from 10/15 to 11/15" names a figure no committed
    # root produces on its own, because it is the default with that stage
    # removed. Dropping each stage in turn covers those honestly, rather than
    # exempting any figure a rationale happens to contain.
    from src.extraction.evaluate_final_gold_dynamic import RESULT_ROOTS

    for dropped in RESULT_ROOTS:
        kept = [root for root in RESULT_ROOTS if root != dropped]
        if not kept:
            continue
        try:
            achievable.add(f"{evaluate(result_roots=kept)['recovered']}/15")
        except Exception:  # noqa: BLE001
            continue

    quoted: dict[str, set[str]] = {}
    for flag in registry().values():
        for figure in re.findall(r"\b(\d{1,2}/15)\b", flag.rationale or ""):
            quoted.setdefault(flag.name, set()).add(figure)

    for name, figures in quoted.items():
        unreachable = figures - achievable
        assert not unreachable, (
            f"{name}'s rationale quotes {sorted(unreachable)}, which no "
            f"committed configuration produces. Reachable: {sorted(achievable)}"
        )
