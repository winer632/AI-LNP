"""Central runtime feature-flag registry for AI-LNP.

Why this module exists
----------------------
Optional capabilities used to be gated by configuration that no runtime code
read -- ``config/extraction/compact_route_v1.yaml`` declared
``activation_gate.active`` but nothing in ``src/`` ever consulted it, so
flipping it changed nothing. This module turns that declaration, and every new
capability flag, into a real switch with a single, testable resolution order.

Where flags are declared
------------------------
``config/feature_flags.yaml`` is the only registry. Adding a capability means
adding an entry there; no code change in this module is required.

Resolution order (highest priority first)
-----------------------------------------
1. an explicit ``default=`` argument -- only consulted for *unknown* flag names,
   which is what makes ``is_enabled("not_registered", default=False)`` safe;
2. the flag's own environment variable, when the registry declares ``env:``;
3. the conventional environment variable ``AI_LNP_FLAG_<NAME_IN_UPPER_CASE>``;
4. ``yaml_source:`` -- the value of another config file's dotted key, used so
   ``compact_route`` keeps mirroring ``activation_gate.active``;
5. ``default:`` from the registry.

An empty or whitespace-only environment value counts as *unset* so that
``AI_LNP_FLAG_X= pytest`` falls back to the configured default rather than
silently meaning "off".

How to integrate a capability (for the agents building P0-a / P0-b / P1 ...)
----------------------------------------------------------------------------
Import the checker and branch at the *boundary* of the new behaviour. Do not
import the registry, do not read ``os.environ`` directly, and do not cache the
result at module import time -- flags are read per call so tests and gradual
rollouts can flip them::

    from src.config_flags import is_enabled

    def parse_document(path):
        if is_enabled("uniparse_ingestion"):
            return _parse_with_uniparse(path)
        return _parse_locally(path)

Every registered flag is also reachable by its design-document priority id, so
``is_enabled("P0-a")`` and ``is_enabled("uniparse_ingestion")`` are equivalent.
If the design document renames a capability, add the new name to that flag's
``aliases`` list in the YAML instead of touching call sites.

Unknown flag names raise :class:`UnknownFlagError` by default. That is
deliberate: a typo in a flag name must fail loudly rather than quietly disable a
feature. Pass an explicit ``default=`` when probing a flag that may legitimately
not be registered yet.

Testing
-------
Use :func:`override` to flip flags inside a test or a script; it sets and
restores the real environment variables, so code in other modules sees the
change::

    with override(uniparse_ingestion=True):
        assert is_enabled("uniparse_ingestion")

Point ``AI_LNP_FLAGS_CONFIG`` at another YAML file to test an alternate
registry. The parsed registry is cached and invalidated automatically when the
registry file -- or any file it mirrors through ``yaml_source`` -- changes path,
size, or mtime; call :func:`reload_flags` to force it.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml


__all__ = [
    "FeatureFlag",
    "UnknownFlagError",
    "FlagConfigError",
    "CONFIG_PATH_ENV",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_ENV_PREFIX",
    "all_flags",
    "describe_flag",
    "env_var_for",
    "is_enabled",
    "known_flags",
    "override",
    "reload_flags",
    "registry",
]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "feature_flags.yaml"
CONFIG_PATH_ENV = "AI_LNP_FLAGS_CONFIG"
DEFAULT_ENV_PREFIX = "AI_LNP_FLAG_"

_TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y", "on", "enable", "enabled"})
_FALSE_VALUES = frozenset({"0", "false", "f", "no", "n", "off", "disable", "disabled"})


class UnknownFlagError(KeyError):
    """Raised when a flag name is not present in the registry."""

    def __init__(self, name: str, known: tuple[str, ...]) -> None:
        self.name = name
        self.known = known
        super().__init__(
            f"Unknown feature flag {name!r}. "
            f"Registered flags: {', '.join(known) or '(none)'}. "
            "Add it to config/feature_flags.yaml, or pass an explicit "
            "default= to probe an unregistered flag."
        )

    def __str__(self) -> str:  # KeyError repr-quotes its argument otherwise.
        return self.args[0]


class FlagConfigError(ValueError):
    """Raised when the registry file or an environment value is malformed."""


@dataclass(frozen=True)
class FeatureFlag:
    """One registered capability switch."""

    name: str
    default: bool
    description: str = ""
    priority: str | None = None
    status: str | None = None
    # Why this flag ships at its current default. Required by the rollout test
    # for anything that defaults on, so "it is enabled" always comes with the
    # measurement that justified enabling it.
    rationale: str = ""
    env_var: str = ""
    aliases: tuple[str, ...] = ()
    integration_points: tuple[str, ...] = ()
    source: str = ""
    extra_env_vars: tuple[str, ...] = field(default=(), repr=False)

    @property
    def env_vars(self) -> tuple[str, ...]:
        """Environment variables consulted for this flag, highest priority first."""
        seen: list[str] = []
        for name in (self.env_var, *self.extra_env_vars):
            if name and name not in seen:
                seen.append(name)
        return tuple(seen)


# --------------------------------------------------------------------------- #
# Registry loading
# --------------------------------------------------------------------------- #

_CACHE: dict[str, Any] = {
    "stamp": None,
    "registry": None,
    "lookup": None,
    "referenced": (),
}


def _config_path() -> Path:
    """Locate the registry file.

    The registry location is process-level configuration and is always read from
    the real environment. The ``env`` argument accepted elsewhere in this module
    overrides *flag value* resolution only, so passing ``env={}`` isolates a
    lookup from ambient flag variables without also hiding the configured
    registry path.
    """
    raw = (os.environ.get(CONFIG_PATH_ENV) or "").strip()
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path)
    return DEFAULT_CONFIG_PATH


def _stamp(path: Path) -> tuple[str, int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (str(path), -1, -1)
    return (str(path), stat.st_size, stat.st_mtime_ns)


def _coerce_bool(value: Any, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _parse_env_value(value, context=context, allow_empty=False)
    raise FlagConfigError(f"{context}: expected a boolean, got {value!r}")


def _dotted(data: Any, key: str) -> Any:
    current = data
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _yaml_source_default(
    entry: Mapping[str, Any], name: str, referenced: list[Path]
) -> bool | None:
    source = entry.get("yaml_source")
    if source is None:
        return None
    if not isinstance(source, Mapping):
        raise FlagConfigError(f"flag {name!r}: yaml_source must be a mapping")
    raw_path = source.get("path")
    raw_key = source.get("key")
    if not raw_path or not raw_key:
        raise FlagConfigError(
            f"flag {name!r}: yaml_source needs both 'path' and 'key'"
        )
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = REPO_ROOT / path
    # Recorded even when absent so that creating the file invalidates the cache.
    referenced.append(path)
    if not path.is_file():
        return None
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    value = _dotted(document, str(raw_key))
    if value is None:
        return None
    return _coerce_bool(value, context=f"flag {name!r} yaml_source {raw_path}:{raw_key}")


def _string_tuple(value: Any, *, context: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raise FlagConfigError(f"{context}: expected a string or list of strings")


def _build_registry(
    path: Path,
) -> tuple[dict[str, FeatureFlag], dict[str, str], tuple[Path, ...]]:
    if not path.is_file():
        raise FlagConfigError(f"Feature flag registry not found: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise FlagConfigError(f"{path}: registry must be a mapping")
    prefix = str(document.get("env_prefix") or DEFAULT_ENV_PREFIX)
    raw_flags = document.get("flags") or {}
    if not isinstance(raw_flags, Mapping):
        raise FlagConfigError(f"{path}: 'flags' must be a mapping")

    try:
        relative_source = str(path.relative_to(REPO_ROOT))
    except ValueError:
        relative_source = str(path)

    flags: dict[str, FeatureFlag] = {}
    lookup: dict[str, str] = {}
    referenced: list[Path] = []

    for name, entry in raw_flags.items():
        name = str(name)
        if not isinstance(entry, Mapping):
            raise FlagConfigError(f"{path}: flag {name!r} must be a mapping")
        aliases = _string_tuple(entry.get("aliases"), context=f"flag {name!r} aliases")
        priority = entry.get("priority")
        priority = None if priority is None else str(priority)
        if priority and priority not in aliases:
            aliases = (*aliases, priority)

        yaml_default = _yaml_source_default(entry, name, referenced)
        if yaml_default is None:
            if "default" not in entry:
                raise FlagConfigError(f"{path}: flag {name!r} has no 'default'")
            default = _coerce_bool(entry["default"], context=f"flag {name!r} default")
            source = relative_source
        else:
            default = yaml_default
            yaml_source = entry["yaml_source"]
            source = f"{yaml_source['path']}#{yaml_source['key']}"

        declared_env = entry.get("env")
        conventional_env = prefix + name.upper().replace("-", "_")
        env_var = str(declared_env) if declared_env else conventional_env
        extra = () if env_var == conventional_env else (conventional_env,)

        flag = FeatureFlag(
            name=name,
            default=default,
            description=" ".join(str(entry.get("description") or "").split()),
            priority=priority,
            status=None if entry.get("status") is None else str(entry["status"]),
            rationale=" ".join(str(entry.get("rationale") or "").split()),
            env_var=env_var,
            aliases=aliases,
            integration_points=_string_tuple(
                entry.get("integration_points"),
                context=f"flag {name!r} integration_points",
            ),
            source=source,
            extra_env_vars=extra,
        )
        flags[name] = flag

        for key in (name, *aliases):
            normalised = _normalise(key)
            existing = lookup.get(normalised)
            if existing is not None and existing != name:
                raise FlagConfigError(
                    f"{path}: {key!r} maps to both {existing!r} and {name!r}"
                )
            lookup[normalised] = name

    return flags, lookup, tuple(referenced)


def _normalise(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def _stamps(path: Path, referenced: tuple[Path, ...]) -> tuple:
    return (_stamp(path), tuple(_stamp(item) for item in referenced))


def _load() -> tuple[dict[str, FeatureFlag], dict[str, str]]:
    path = _config_path()
    if (
        _CACHE["registry"] is None
        or _CACHE["stamp"] != _stamps(path, _CACHE["referenced"])
    ):
        flags, lookup, referenced = _build_registry(path)
        _CACHE["referenced"] = referenced
        _CACHE["stamp"] = _stamps(path, referenced)
        _CACHE["registry"] = flags
        _CACHE["lookup"] = lookup
    return _CACHE["registry"], _CACHE["lookup"]


def reload_flags() -> None:
    """Drop the cached registry so the next lookup re-reads the YAML files."""
    _CACHE["stamp"] = None
    _CACHE["registry"] = None
    _CACHE["lookup"] = None
    _CACHE["referenced"] = ()


def registry() -> Mapping[str, FeatureFlag]:
    """Return every registered flag keyed by its canonical name."""
    flags, _ = _load()
    return dict(flags)


def known_flags() -> tuple[str, ...]:
    """Canonical names of every registered flag, in registry order."""
    flags, _ = _load()
    return tuple(flags)


def _resolve(name: str) -> FeatureFlag | None:
    flags, lookup = _load()
    canonical = lookup.get(_normalise(name))
    return None if canonical is None else flags[canonical]


def describe_flag(name: str) -> FeatureFlag:
    """Return the :class:`FeatureFlag` registered under ``name`` or an alias."""
    flag = _resolve(name)
    if flag is None:
        raise UnknownFlagError(name, known_flags())
    return flag


def env_var_for(name: str) -> str:
    """Return the primary environment variable that overrides ``name``."""
    return describe_flag(name).env_var


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def _parse_env_value(raw: str, *, context: str, allow_empty: bool = True) -> bool | None:
    text = raw.strip().lower()
    if not text:
        if allow_empty:
            return None
        raise FlagConfigError(f"{context}: empty value is not a boolean")
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise FlagConfigError(
        f"{context}: {raw!r} is not a boolean. Use one of "
        f"{sorted(_TRUE_VALUES)} or {sorted(_FALSE_VALUES)}."
    )


def is_enabled(
    name: str,
    *,
    default: bool | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the capability ``name`` is currently switched on.

    ``name`` may be a canonical flag name, an alias, or a design-document
    priority id such as ``"P0-a"``.

    ``default`` is a fallback for flag names that are **not** registered. A
    registered flag always uses its own configured default, so passing
    ``default`` never silently masks a registry value. Omitting ``default`` for
    an unregistered name raises :class:`UnknownFlagError`.

    ``env`` overrides the environment mapping consulted for the *value*; it
    exists for tests and defaults to :data:`os.environ`. It does not affect
    where the registry itself is loaded from.
    """
    environ = os.environ if env is None else env
    flag = _resolve(name)
    if flag is None:
        if default is None:
            raise UnknownFlagError(name, known_flags())
        return bool(default)

    for variable in flag.env_vars:
        if variable not in environ:
            continue
        parsed = _parse_env_value(
            environ[variable], context=f"environment variable {variable}"
        )
        if parsed is not None:
            return parsed
    return flag.default


def all_flags(env: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Effective value of every registered flag, canonical name -> bool."""
    return {name: is_enabled(name, env=env) for name in known_flags()}


@contextmanager
def override(**values: bool) -> Iterator[None]:
    """Temporarily force flag values by setting their environment variables.

    Keys are flag names or aliases with ``-`` written as ``_`` (Python keyword
    syntax), for example ``override(uniparse_ingestion=True, p0b=False)``. The
    previous environment is restored on exit, including variables that were
    unset.
    """
    saved: list[tuple[str, str | None]] = []
    try:
        for name, value in values.items():
            variable = env_var_for(name)
            saved.append((variable, os.environ.get(variable)))
            os.environ[variable] = "true" if value else "false"
        yield
    finally:
        for variable, previous in reversed(saved):
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous


def _main() -> int:
    import json

    rows = [
        {
            "name": flag.name,
            "priority": flag.priority,
            "status": flag.status,
            "default": flag.default,
            "effective": is_enabled(flag.name),
            "env_var": flag.env_var,
            "source": flag.source,
            "description": flag.description,
        }
        for flag in registry().values()
    ]
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
