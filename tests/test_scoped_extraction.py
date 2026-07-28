"""One endpoint family per call, and the candidate ids each call is judged on.

The bug this guards is quiet and total. ``validate_candidate`` uses
``required_candidate_ids`` for two things at once: it selects which response
contract the reply is parsed against, and it demands a disposition for every id
in it. So

* omitting it parses the reply against ``CompactExtractionResponse``, which
  forbids ``candidate_dispositions`` -- every reply dies of
  ``pydantic.extra_forbidden``; and
* passing every candidate in the paper demands dispositions for slots this call
  never sent -- every reply dies of ``missing_candidate_disposition``.

Either way no scope ever produces a record and the whole run comes back empty,
which looks like a model failure rather than a wiring failure.

``run_codex`` is replaced with a stub that answers exactly the slots it was
sent, the way a well-behaved reply does. No subprocess is started and ``codex``
is never invoked. The packet is the real committed compact packet for GP-004
and the reply body is that paper's real committed extraction result, so the
evidence ids and the contract version are the ones the pipeline really uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.extraction.run_scoped_extraction as scoped
from src.extraction.build_outcome_candidates import build_candidates
from src.extraction.run_codex_one_call import load_packet

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKET_ROOT = REPO_ROOT / "data/staging/rag/compact_api_packets_v1"
REAL_RESULT = (
    REPO_ROOT
    / "data/staging/extraction/codex_control_compact_v1/GP-004/result.json"
)
PAPER_ID = "GP-004"
SLOT_MARKER = "CANDIDATE SLOTS:\n"


def _slots_in_prompt(prompt: str) -> list[str]:
    """Read back the candidate ids this particular call was actually sent."""
    if SLOT_MARKER not in prompt:
        return []
    payload, _ = json.JSONDecoder().raw_decode(prompt.split(SLOT_MARKER, 1)[1])
    return [row["candidate_id"] for row in payload["candidate_slots"]]


def _all_candidate_ids() -> list[str]:
    packet = load_packet(PAPER_ID, PACKET_ROOT)
    return [row.candidate_id for row in build_candidates(packet)]


def _stub_codex(monkeypatch, *, answer):
    """Replace the codex subprocess with a reply builder. Records every prompt."""
    prompts: list[list[str]] = []

    def fake_run_codex(prompt, schema, *, model, reasoning_effort, timeout):
        sent = _slots_in_prompt(prompt)
        prompts.append(sent)
        body = json.loads(REAL_RESULT.read_text(encoding="utf-8"))
        body["candidate_dispositions"] = [
            {
                "candidate_id": candidate_id,
                "disposition": "unresolved",
                "reason": "No printed value for this slot.",
            }
            for candidate_id in answer(sent)
        ]
        return {
            "text": json.dumps(body, ensure_ascii=False),
            "elapsed_seconds": 0.0,
            "stdout_tail": "",
        }

    monkeypatch.setattr(scoped, "run_codex", fake_run_codex)
    return prompts


@pytest.fixture
def refuse_subprocess(monkeypatch):
    """Any real process launch from this module is a test failure."""

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("a scoped extraction test tried to start a process")

    monkeypatch.setattr("subprocess.run", explode)


def test_each_scope_is_judged_only_on_the_slots_it_was_sent(
    tmp_path, monkeypatch, refuse_subprocess
):
    prompts = _stub_codex(monkeypatch, answer=lambda sent: sent)

    manifest = scoped.run_paper(
        PAPER_ID, packet_root=PACKET_ROOT, output_root=tmp_path
    )

    assert manifest["total_scopes"] == len(prompts) > 1
    assert [scope["validation_status"] for scope in manifest["scopes"]] == [
        "valid"
    ] * len(prompts)

    for report_path in sorted((tmp_path / PAPER_ID).glob("scope_*/validation_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["findings"] == [], report_path.parent.name
        assert report["status"] == "valid"

    # Records really did survive validation, which is what the contract
    # selection buys; a rejected reply yields parsed=None and no outcomes.
    assert manifest["merged_outcomes"] > 0
    final = json.loads(
        (tmp_path / PAPER_ID / "final_result.json").read_text(encoding="utf-8")
    )
    assert final["contract_version"] == "compact-1.1.0"
    assert len(final["outcomes"]) == manifest["merged_outcomes"]


def test_no_call_is_sent_the_whole_papers_candidate_set(
    tmp_path, monkeypatch, refuse_subprocess
):
    prompts = _stub_codex(monkeypatch, answer=lambda sent: sent)

    scoped.run_paper(PAPER_ID, packet_root=PACKET_ROOT, output_root=tmp_path)

    everything = _all_candidate_ids()
    assert len(everything) > 1
    assert sorted(sum(prompts, [])) == sorted(everything)
    for sent in prompts:
        assert 0 < len(sent) < len(everything)
    # No candidate is asked for twice; the scopes partition the set.
    assert len(sum(prompts, [])) == len(set(sum(prompts, [])))


def test_scopes_are_split_by_endpoint_family(
    tmp_path, monkeypatch, refuse_subprocess
):
    _stub_codex(monkeypatch, answer=lambda sent: sent)
    packet = load_packet(PAPER_ID, PACKET_ROOT)
    families = {
        candidate.endpoint_family or "unspecified"
        for candidate in build_candidates(packet)
    }

    manifest = scoped.run_paper(
        PAPER_ID, packet_root=PACKET_ROOT, output_root=tmp_path
    )

    assert {scope["family"] for scope in manifest["scopes"]} == families
    assert sum(scope["slots"] for scope in manifest["scopes"]) == len(
        build_candidates(packet)
    )
    written = {path.name.split("_", 2)[2] for path in (tmp_path / PAPER_ID).glob("scope_*")}
    assert written == families


def test_a_reply_that_answers_out_of_scope_slots_is_rejected(
    tmp_path, monkeypatch, refuse_subprocess
):
    """The scope note is a contract, not a hint: answering more is an error.

    This is the other side of the same wiring. If the run were judged against
    every candidate, a reply covering all of them would validate, and the
    per-family split would stop meaning anything.
    """
    everything = _all_candidate_ids()
    _stub_codex(monkeypatch, answer=lambda sent: everything)

    manifest = scoped.run_paper(
        PAPER_ID, packet_root=PACKET_ROOT, output_root=tmp_path
    )

    assert {scope["validation_status"] for scope in manifest["scopes"]} == {"invalid"}
    assert manifest["merged_outcomes"] == 0
    codes = {
        finding["code"]
        for path in (tmp_path / PAPER_ID).glob("scope_*/validation_report.json")
        for finding in json.loads(path.read_text(encoding="utf-8"))["findings"]
    }
    assert codes == {"unknown_candidate_id"}


def test_max_slots_per_call_chunks_a_family_without_losing_a_candidate(
    tmp_path, monkeypatch, refuse_subprocess
):
    prompts = _stub_codex(monkeypatch, answer=lambda sent: sent)

    manifest = scoped.run_paper(
        PAPER_ID,
        packet_root=PACKET_ROOT,
        output_root=tmp_path,
        max_slots_per_call=2,
    )

    assert all(len(sent) <= 2 for sent in prompts)
    assert sorted(sum(prompts, [])) == sorted(_all_candidate_ids())
    assert manifest["codex_exec_turns"] == len(prompts)
    assert {scope["validation_status"] for scope in manifest["scopes"]} == {"valid"}
