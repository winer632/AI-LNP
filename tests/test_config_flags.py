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
        assert flag.integration_points, f"{flag.name} has no integration points"


def test_every_flag_has_a_distinct_environment_variable():
    variables = [flag.env_var for flag in registry().values()]
    assert len(variables) == len(set(variables))
    assert all(name.startswith("AI_LNP_FLAG_") for name in variables)


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
