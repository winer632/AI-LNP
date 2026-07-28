"""Converge-or-refuse writes, so re-running a stage is safe.

Why this exists
---------------
Section 9 of the recall-improvement design document asks for one property::

    写入语义 | 覆盖式 -> 幂等 upsert，重跑结果一致
    write semantics | overwriting -> idempotent upsert, a rerun agrees

Today the repository has both failure modes at once. Sixteen entry points raise
``FileExistsError`` when their output already exists, so a rerun of a stage
that was interrupted after writing cannot be completed without deleting files
by hand. Every local packet builder does the opposite and clobbers, so a rerun
against changed inputs silently replaces a committed artifact and nothing
records that it happened.

What idempotence is NOT
-----------------------
It is not "overwrite freely". The ``FileExistsError`` guards were put there to
stop a rerun destroying a completed run -- several of them guard results that
cost a paid API call -- and deleting the guard in the name of idempotence
would throw away the only protection those runs have.

The property implemented here is the one that is actually useful:

* **Same inputs converge.** A rerun that would produce byte-identical content
  writes nothing at all -- not even the same bytes, so the mtime does not move
  and ``git status`` stays clean -- and reports ``unchanged``. For a stage
  whose output cannot be recomputed without paying for it (a model call), the
  caller compares a recorded *input fingerprint* instead and skips the call.
* **Different inputs are rejected, or recorded as a new version.** The default
  is rejection: :class:`UpsertConflict`, carrying both fingerprints, so the
  operator decides. ``on_conflict="version"`` is the opt-in that keeps the
  previous content under ``<name>.versions/`` and appends to a ledger, for
  stages whose outputs are cheap, derived and expected to move.

"Converge or refuse" is chosen over "always version" because a silent new
version turns destruction into something invisible rather than something
impossible. An operator who reruns a stage with a changed packet should be
told, not handed a directory that quietly grew a file.

Flag
----
``idempotent_upsert``. With the flag off every wired call site behaves exactly
as it did before -- clobber where it clobbered, raise where it raised -- so
turning it on is the only thing that changes behaviour.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.config_flags import is_enabled


__all__ = [
    "IDEMPOTENT_UPSERT_FLAG",
    "UpsertConflict",
    "UpsertResult",
    "canonical_json",
    "fingerprint",
    "upsert_enabled",
    "upsert_json",
    "upsert_text",
]


IDEMPOTENT_UPSERT_FLAG = "idempotent_upsert"

OnConflict = Literal["refuse", "version", "replace"]
UpsertStatus = Literal["created", "unchanged", "replaced", "versioned"]

VERSION_LEDGER_NAME = "versions.json"


class UpsertConflict(FileExistsError):
    """Existing content differs from what this run would write.

    Subclasses :class:`FileExistsError` on purpose: every call site wired to
    this module previously raised ``FileExistsError`` for the same situation,
    and callers that catch it -- including the tests that pin the paid-call
    guards -- keep working unchanged.
    """

    def __init__(
        self,
        path: Path,
        *,
        existing: str,
        proposed: str,
        hint: str = "",
    ) -> None:
        self.path = path
        self.existing_fingerprint = existing
        self.proposed_fingerprint = proposed
        message = (
            f"{path} already holds different content "
            f"(existing {existing[:12]}, proposed {proposed[:12]})."
        )
        super().__init__(f"{message} {hint}".strip())


@dataclass(frozen=True)
class UpsertResult:
    """What an upsert did, and to what."""

    path: Path
    status: UpsertStatus
    fingerprint: str
    previous_fingerprint: str | None = None
    version_path: Path | None = None

    @property
    def wrote(self) -> bool:
        """Whether anything was written. ``False`` for a converged rerun."""
        return self.status != "unchanged"

    @property
    def converged(self) -> bool:
        """Whether the target already held exactly this content."""
        return self.status == "unchanged"


def upsert_enabled() -> bool:
    """Whether converge-or-refuse semantics are switched on."""
    return is_enabled(IDEMPOTENT_UPSERT_FLAG)


def canonical_json(payload: Any) -> str:
    """Key-sorted, separator-tight JSON: the same rule the packet checksums use."""
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def fingerprint(value: Any) -> str:
    """A sha256 over ``value``, canonicalising anything that is not bytes/str."""
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _archive(path: Path, previous: bytes, previous_fingerprint: str) -> Path:
    """Keep ``previous`` beside ``path`` and record it in the ledger."""
    store = path.parent / f"{path.name}.versions"
    store.mkdir(parents=True, exist_ok=True)
    ledger_path = store / VERSION_LEDGER_NAME
    ledger: dict[str, Any] = {"path": path.name, "versions": []}
    if ledger_path.exists():
        loaded = json.loads(ledger_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("versions"), list):
            ledger = loaded
    ordinal = len(ledger["versions"]) + 1
    archived = store / f"{ordinal:03d}-{previous_fingerprint[:12]}{path.suffix}"
    archived.write_bytes(previous)
    ledger["versions"].append(
        {
            "version": ordinal,
            "fingerprint": previous_fingerprint,
            "archived_as": archived.name,
        }
    )
    ledger_path.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return archived


def upsert_text(
    path: Path,
    text: str,
    *,
    on_conflict: OnConflict = "refuse",
    hint: str = "",
    enabled: bool | None = None,
) -> UpsertResult:
    """Write ``text`` to ``path`` unless it is already exactly there.

    ``enabled`` overrides the flag; it exists so a call site can pass the value
    it already resolved rather than reading the registry twice, and so tests
    can exercise both semantics without touching the environment.
    """
    path = Path(path)
    payload = text.encode("utf-8")
    proposed = fingerprint(payload)

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return UpsertResult(path=path, status="created", fingerprint=proposed)

    previous = path.read_bytes()
    existing = fingerprint(previous)
    if existing == proposed:
        # The whole point: converge without touching the file, so a rerun
        # leaves no trace in the working tree at all.
        return UpsertResult(
            path=path,
            status="unchanged",
            fingerprint=proposed,
            previous_fingerprint=existing,
        )

    active = upsert_enabled() if enabled is None else enabled
    if not active or on_conflict == "replace":
        path.write_bytes(payload)
        return UpsertResult(
            path=path,
            status="replaced",
            fingerprint=proposed,
            previous_fingerprint=existing,
        )
    if on_conflict == "version":
        archived = _archive(path, previous, existing)
        path.write_bytes(payload)
        return UpsertResult(
            path=path,
            status="versioned",
            fingerprint=proposed,
            previous_fingerprint=existing,
            version_path=archived,
        )
    raise UpsertConflict(path, existing=existing, proposed=proposed, hint=hint)


def upsert_json(
    path: Path,
    payload: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    on_conflict: OnConflict = "refuse",
    hint: str = "",
    enabled: bool | None = None,
) -> UpsertResult:
    """:func:`upsert_text` for a JSON document, serialised the repository way.

    Defaults match what the writers being replaced already emitted: two-space
    indent, non-ASCII kept, key order preserved, one trailing newline. Any
    change here would rewrite every artifact on the first rerun, which is the
    opposite of the property this module exists to provide.
    """
    text = json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys)
    return upsert_text(
        path,
        text + "\n",
        on_conflict=on_conflict,
        hint=hint,
        enabled=enabled,
    )
