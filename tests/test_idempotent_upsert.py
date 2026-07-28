"""Idempotent upsert (design document, section 9).

    写入语义 | 覆盖式 -> 幂等 upsert，重跑结果一致
    write semantics | overwriting -> idempotent upsert, a rerun agrees

The property under test is deliberately narrower than "writes never fail":

* a rerun with the **same** inputs converges -- byte-identical content is not
  rewritten at all, so the mtime does not move and the tree stays clean;
* a rerun with **different** inputs is refused, or, where the call site opted
  in, recorded as a new version.

Refusal is the default because the FileExistsError guards this replaces exist
to stop a rerun destroying a completed run. The tests therefore pin the
refusal as hard as they pin the convergence, and pin that the flag off leaves
every wired call site behaving exactly as it did.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config_flags import override
from src.idempotent_write import (
    UpsertConflict,
    fingerprint,
    upsert_json,
    upsert_text,
)


# --------------------------------------------------------------------------- #
# The primitive
# --------------------------------------------------------------------------- #


def test_first_write_creates(tmp_path: Path):
    result = upsert_text(tmp_path / "a.json", "one\n", enabled=True)
    assert result.status == "created"
    assert (tmp_path / "a.json").read_text() == "one\n"


def test_identical_content_is_not_rewritten(tmp_path: Path):
    """Convergence, checked by the file's own mtime rather than its bytes.

    Writing the same bytes again would also "agree", but it dirties the working
    tree's timestamps and makes every rerun look like a change to anything
    watching the file. Not writing is the property.
    """
    path = tmp_path / "a.json"
    upsert_text(path, "one\n", enabled=True)
    before = path.stat().st_mtime_ns

    result = upsert_text(path, "one\n", enabled=True)

    assert result.status == "unchanged"
    assert result.converged and not result.wrote
    assert path.stat().st_mtime_ns == before


def test_different_content_is_refused_by_default(tmp_path: Path):
    path = tmp_path / "a.json"
    upsert_text(path, "one\n", enabled=True)

    with pytest.raises(UpsertConflict) as raised:
        upsert_text(path, "two\n", enabled=True, hint="use a new --output-root")

    # A FileExistsError subclass, so the handlers that caught the guards this
    # replaces keep catching it.
    assert isinstance(raised.value, FileExistsError)
    assert "use a new --output-root" in str(raised.value)
    assert raised.value.existing_fingerprint == fingerprint("one\n")
    assert raised.value.proposed_fingerprint == fingerprint("two\n")
    assert path.read_text() == "one\n", "a refused write must not have happened"


def test_different_content_can_be_recorded_as_a_new_version(tmp_path: Path):
    path = tmp_path / "a.json"
    upsert_text(path, "one\n", enabled=True)

    result = upsert_text(path, "two\n", on_conflict="version", enabled=True)

    assert result.status == "versioned"
    assert path.read_text() == "two\n"
    assert result.version_path is not None
    assert result.version_path.read_text() == "one\n"
    ledger = json.loads(
        (path.parent / "a.json.versions" / "versions.json").read_text()
    )
    assert [row["fingerprint"] for row in ledger["versions"]] == [
        fingerprint("one\n")
    ]

    # A third, different write appends rather than replacing the record.
    upsert_text(path, "three\n", on_conflict="version", enabled=True)
    ledger = json.loads(
        (path.parent / "a.json.versions" / "versions.json").read_text()
    )
    assert [row["version"] for row in ledger["versions"]] == [1, 2]


def test_with_the_flag_off_a_conflict_clobbers_exactly_as_before(tmp_path: Path):
    path = tmp_path / "a.json"
    upsert_text(path, "one\n", enabled=False)
    result = upsert_text(path, "two\n", enabled=False)
    assert result.status == "replaced"
    assert path.read_text() == "two\n"
    assert not (path.parent / "a.json.versions").exists()


def test_upsert_json_serialises_the_way_the_writers_it_replaces_did(tmp_path: Path):
    """Two-space indent, non-ASCII kept, key order preserved, trailing newline.

    Any drift here rewrites every artifact on the first rerun, which is the
    opposite of what this module is for.
    """
    path = tmp_path / "a.json"
    payload = {"b": 1, "a": "1.01 ± 0.38 %"}
    upsert_json(path, payload, enabled=True)
    assert path.read_text(encoding="utf-8") == (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


# --------------------------------------------------------------------------- #
# Wired: the local packet writers
# --------------------------------------------------------------------------- #


def _packet(evidence_text: str = "Total insertion frequency was 1.01 %."):
    from src.rag.compact_packet import (
        CompactEvidence,
        CompactEvidencePacket,
        DeduplicationSummary,
        SourceLocation,
    )

    return CompactEvidencePacket(
        paper_id="GP-X",
        source_retrieval_packet="retrieval.json",
        blocked_fields={},
        evidence=[
            CompactEvidence(
                evidence_id="GP-X-E-0000000000000001",
                clause_ids=["GP-X-B-1:C001"],
                chunk_ids=["GP-X-B-1"],
                text=evidence_text,
                normalized_text_sha256="0" * 64,
                field_tags=["outcomes"],
                field_groups=["outcome"],
                query_ids=["Q1"],
                entity_types=[],
                experiment_candidate_ids=[],
                source_locations=[
                    SourceLocation(
                        chunk_id="GP-X-B-1",
                        source_path="a.nxml",
                        source_kind="pmc_xml",
                        block_type="paragraph",
                        section="Results",
                    )
                ],
            )
        ],
        deduplication=DeduplicationSummary(
            input_hits=1,
            unique_chunks=1,
            removed_chunk_duplicates=0,
            removed_normalized_passages=0,
            input_clauses=1,
            unique_evidence_items=1,
        ),
        packet_checksum="0" * 64,
    )


def test_write_packet_converges_on_an_unchanged_rebuild(tmp_path: Path):
    from src.rag.compact_packet import write_packet

    with override(idempotent_upsert=True):
        path = write_packet(_packet(), tmp_path)
        before = path.stat().st_mtime_ns
        write_packet(_packet(), tmp_path)
    assert path.stat().st_mtime_ns == before


def test_write_packet_records_a_version_when_the_rebuild_differs(tmp_path: Path):
    from src.rag.compact_packet import write_packet

    with override(idempotent_upsert=True):
        path = write_packet(_packet(), tmp_path)
        write_packet(_packet("Total insertion frequency was 4.23 %."), tmp_path)

    assert "4.23" in path.read_text(encoding="utf-8")
    archive = tmp_path / "GP-X.json.versions"
    assert archive.exists(), "the superseded packet must still be recoverable"
    ledger = json.loads((archive / "versions.json").read_text(encoding="utf-8"))
    archived = archive / ledger["versions"][0]["archived_as"]
    assert "1.01" in archived.read_text(encoding="utf-8")


def test_write_packet_with_the_flag_off_still_just_overwrites(tmp_path: Path):
    from src.rag.compact_packet import write_packet

    with override(idempotent_upsert=False):
        path = write_packet(_packet(), tmp_path)
        write_packet(_packet("Total insertion frequency was 4.23 %."), tmp_path)

    assert "4.23" in path.read_text(encoding="utf-8")
    assert not (tmp_path / "GP-X.json.versions").exists()


def test_api_packet_outputs_converge(tmp_path: Path):
    from src.rag.compact_api_packet import build_api_packet, write_outputs

    api, manifest = build_api_packet(_packet())
    with override(idempotent_upsert=True):
        packet_path, manifest_path = write_outputs(api, manifest, tmp_path)
        stamps = (
            packet_path.stat().st_mtime_ns,
            manifest_path.stat().st_mtime_ns,
        )
        write_outputs(api, manifest, tmp_path)
    assert (
        packet_path.stat().st_mtime_ns,
        manifest_path.stat().st_mtime_ns,
    ) == stamps


# --------------------------------------------------------------------------- #
# Wired: the stage whose output costs a model call
# --------------------------------------------------------------------------- #


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPACT_PACKETS = REPO_ROOT / "data/staging/rag/compact_api_packets_v1"
COMMITTED_RUN = REPO_ROOT / "data/staging/extraction/codex_control_compact_v1/GP-004"


@pytest.fixture
def codex_stub(monkeypatch):
    """Replay a committed Codex response and count how often it is asked for."""
    import src.extraction.run_codex_one_call as module

    response = json.loads((COMMITTED_RUN / "response.json").read_text("utf-8"))
    calls = []

    def fake_run_codex(prompt, schema, **kwargs):
        calls.append(kwargs)
        return {
            "text": response["text"],
            "elapsed_seconds": 0.0,
            "stdout_tail": "",
        }

    monkeypatch.setattr(module, "run_codex", fake_run_codex)
    return calls


def _run(output_root: Path, **kwargs):
    """One replayed run against the committed GP-004 compact packet.

    ``candidate_slot_enforcement`` is off for every caller because the
    committed response being replayed was produced without slots, so with the
    flag on it fails schema validation, no ``result.json`` is written, and the
    guard under test is never reached -- the test would pass while checking
    nothing.
    """
    from src.extraction.run_codex_one_call import run_one

    return run_one(
        "GP-004",
        evidence_view="compact",
        packet_root=COMPACT_PACKETS,
        output_root=output_root,
        **kwargs,
    )


def test_a_rerun_with_the_same_request_returns_the_completed_run(
    tmp_path: Path, codex_stub
):
    """No second model call, and nothing on disk touched.

    This is the case the FileExistsError guard could not distinguish from a
    destructive rerun, so it made the operator delete the directory -- which
    threw away the completed run in order to redo it.
    """
    with override(idempotent_upsert=True, candidate_slot_enforcement=False):
        first = _run(tmp_path)
        assert len(codex_stub) == 1
        result_path = tmp_path / "GP-004" / "result.json"
        stamp = result_path.stat().st_mtime_ns

        second = _run(tmp_path)

    assert len(codex_stub) == 1, "a converged rerun must not call the model"
    assert second["upsert"] == "unchanged"
    assert second["codex_exec_turns"] == 0
    assert second["record_counts"] == first["record_counts"]
    assert result_path.stat().st_mtime_ns == stamp


def test_a_rerun_with_a_different_request_is_still_refused(
    tmp_path: Path, codex_stub
):
    with override(idempotent_upsert=True, candidate_slot_enforcement=False):
        _run(tmp_path)
        with pytest.raises(UpsertConflict, match="different inputs"):
            from src.extraction.run_codex_one_call import run_one

            run_one(
                "GP-004",
                evidence_view="compact",
                packet_root=COMPACT_PACKETS,
                output_root=tmp_path,
                # A different model is a different run; its answer may not be
                # assumed to be the one already on disk.
                model="some-other-model",
            )
    assert len(codex_stub) == 1


def test_a_result_whose_request_was_not_recorded_is_refused(
    tmp_path: Path, codex_stub
):
    """An unproven match is not a match."""
    with override(idempotent_upsert=True, candidate_slot_enforcement=False):
        _run(tmp_path)
        (tmp_path / "GP-004" / "request.json").unlink()
        with pytest.raises(UpsertConflict, match="request.json does not"):
            _run(tmp_path)
    assert len(codex_stub) == 1


def test_with_the_flag_off_a_rerun_raises_the_original_guard(
    tmp_path: Path, codex_stub
):
    with override(idempotent_upsert=False, candidate_slot_enforcement=False):
        _run(tmp_path)
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            _run(tmp_path)
    assert len(codex_stub) == 1


# --------------------------------------------------------------------------- #
# Wired: the recovered-result merge
# --------------------------------------------------------------------------- #


def _merge(output_path: Path):
    from src.extraction.merge_missing_records import merge

    absent = output_path.parent / "absent"
    return merge(
        result_path=absent / "result.json",
        packet_path=absent / "packet.json",
        pairs=[],
        inventory_path=absent / "inventory.json",
        output_path=output_path,
    )


def test_merge_keeps_the_original_guard_while_the_flag_is_off(tmp_path: Path):
    output = tmp_path / "recovered.json"
    output.write_text("{}\n", encoding="utf-8")
    with override(idempotent_upsert=False):
        with pytest.raises(
            FileExistsError, match="Refusing to overwrite an existing recovered"
        ):
            _merge(output)


def test_merge_defers_the_decision_to_the_write_when_the_flag_is_on(tmp_path: Path):
    """With the flag on, an existing output no longer ends the run up front.

    The merge is pure and local, so whether the rerun agrees is computable
    rather than guessable -- but only if the inputs are actually read. Reaching
    the missing input file is what proves the early guard stepped aside; the
    conflict decision itself is then made by the upsert, which the tests above
    pin.
    """
    output = tmp_path / "recovered.json"
    output.write_text("{}\n", encoding="utf-8")
    with override(idempotent_upsert=True):
        with pytest.raises(FileNotFoundError):
            _merge(output)
    assert output.read_text(encoding="utf-8") == "{}\n"
