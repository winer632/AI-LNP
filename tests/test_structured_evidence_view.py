"""The structured evidence view: selection rule, wiring, and the merge stage.

``structured_compact_packet`` existed and was measured, but nothing called it:
``run_codex_one_call.PACKET_ROOT_BY_VIEW`` knew only ``compact`` and ``full``,
so a default run could not reach it and GP-006's ``1.01`` stayed out of every
packet the pipeline actually sent. These tests pin the wiring that closed that
gap, so it cannot silently come undone again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config_flags import override
from src.extraction.merge_structured_view_pass import (
    build as build_merge,
    merge_paper,
    resolve_baseline,
)
from src.extraction.run_codex_one_call import (
    PACKET_ROOT_BY_VIEW,
    default_evidence_view,
    load_packet,
)
from src.rag.compact_api_packet import ApiEvidence, ApiSource, CompactApiPacket
from src.rag.structured_compact_packet import (
    STRUCTURED_BLOCK_TYPES,
    build_structured_compact_packet,
    is_structured,
    select_evidence,
    sign,
)

from tests.test_full_api_packet import PAPER_ID, _fixture_roots


REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_structured_view_is_reachable_and_is_what_a_default_run_uses():
    assert "structured" in PACKET_ROOT_BY_VIEW
    assert PACKET_ROOT_BY_VIEW["structured"] == (
        REPO_ROOT / "data/staging/rag/structured_compact_packets_v1"
    )
    assert default_evidence_view() == "structured"


def test_turning_the_flag_off_falls_back_to_the_compact_packet():
    with override(structured_evidence_view=False):
        assert default_evidence_view() == "compact"


def test_a_pinned_full_view_still_wins_over_the_structured_default():
    """full_evidence_view + compact_route off is the documented full-view pin."""
    with override(full_evidence_view=True, compact_route=False):
        assert default_evidence_view() == "full"
    # ... and it stays an opt-in: the compact route on is enough to ignore it.
    with override(full_evidence_view=True, compact_route=True):
        assert default_evidence_view() == "structured"


# --------------------------------------------------------------------------- #
# Selection rule
# --------------------------------------------------------------------------- #


def _packet(rows: list[tuple[str, str, str]]) -> CompactApiPacket:
    """Build a packet from (source_id, block_type, text) triples."""
    sources = [
        ApiSource(
            source_id=source_id,
            chunk_id=source_id,
            source_path="data/raw/paper.pdf",
            source_kind="uniparse",
            block_type=block_type,
            section="Supplement",
        )
        for source_id, block_type, _ in rows
    ]
    evidence = [
        ApiEvidence(
            evidence_id=f"{PAPER_ID}-FC-{index:04d}",
            text=text,
            source_ids=[source_id],
            retrieval_field_tags=["full_corpus"],
        )
        for index, (source_id, _, text) in enumerate(rows)
    ]
    return sign(PAPER_ID, [], sources, evidence)


def test_a_table_row_survives_a_budget_that_drops_the_prose_around_it():
    """The regression this view exists for, in miniature.

    The ranker treats a grid line as a weak passage, so a budget that cannot
    hold everything drops the row carrying the number and keeps the prose.
    """
    rows = [("s-table", "table_row", "Table S2 | LSEC | total insertion frequency: 1.01")]
    rows += [
        (f"s-prose-{index}", "paragraph", f"Filler sentence number {index}. " * 40)
        for index in range(30)
    ]
    packet = _packet(rows)

    selected, report = select_evidence(packet, budget_tokens=400)
    texts = " ".join(row.text for row in selected)

    assert "1.01" in texts
    assert report["structured_evidence"] == 1
    assert report["excluded_evidence"] > 0


def test_every_structured_block_type_is_kept_even_over_budget():
    rows = [
        (f"s-{block_type}", block_type, f"{block_type} carries 1.0{index}")
        for index, block_type in enumerate(sorted(STRUCTURED_BLOCK_TYPES))
    ]
    packet = _packet(rows)

    selected, report = select_evidence(packet, budget_tokens=1)

    assert len(selected) == len(STRUCTURED_BLOCK_TYPES)
    assert report["filler_evidence"] == 0
    assert report["structured_evidence_over_budget"] is True


def test_unstructured_evidence_is_ranked_not_truncated_in_packet_order():
    """Retrieval-tagged filler outranks corpus filler that appears before it."""
    packet = _packet(
        [("s-corpus", "paragraph", "Background prose. " * 30)]
        + [("s-retrieved", "paragraph", "Editing reached 45% in hepatocytes. " * 5)]
    )
    ranked = packet.evidence[1].model_copy(
        update={"retrieval_field_tags": ["outcomes"]}
    )
    packet = sign(PAPER_ID, [], packet.sources, [packet.evidence[0], ranked])

    # Wide enough for either passage on its own, too narrow for both.
    selected, _ = select_evidence(packet, budget_tokens=100)

    assert [row.evidence_id for row in selected] == [ranked.evidence_id]


def test_is_structured_reads_the_block_type_of_the_cited_source():
    packet = _packet([("s-caption", "caption", "Figure 3. Editing by cell type.")])
    block_types = {row.source_id: row.block_type or "" for row in packet.sources}
    assert is_structured(packet.evidence[0], block_types)
    assert not is_structured(packet.evidence[0], {"s-caption": "paragraph"})


def test_a_built_structured_packet_passes_the_load_packet_checksum(tmp_path):
    review_root, corpus_root = _fixture_roots(tmp_path)
    packet, report = build_structured_compact_packet(
        PAPER_ID, review_packet_root=review_root, corpus_root=corpus_root
    )
    output_root = tmp_path / "structured_packets"
    output_root.mkdir()
    (output_root / f"{PAPER_ID}.json").write_text(
        packet.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
    )

    assert load_packet(PAPER_ID, output_root) == packet
    assert report["paper_id"] == PAPER_ID


def test_dropped_context_pointers_do_not_dangle(tmp_path):
    """A kept row may point at a neighbour the budget excluded."""
    review_root, corpus_root = _fixture_roots(tmp_path)
    packet, _ = build_structured_compact_packet(
        PAPER_ID,
        review_packet_root=review_root,
        corpus_root=corpus_root,
        budget_tokens=40,
    )
    kept = {row.evidence_id for row in packet.evidence}
    for row in packet.evidence:
        for pointer in (row.context_before_evidence_id, row.context_after_evidence_id):
            assert pointer is None or pointer in kept


# --------------------------------------------------------------------------- #
# The merge stage
# --------------------------------------------------------------------------- #


def _result(path: Path, outcomes: list[tuple[str, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "paper_id": PAPER_ID,
                "experiments": [{"experiment_id": "EX1"}],
                "outcomes": [
                    {
                        "outcome_id": outcome_id,
                        "experiment_id": "EX1",
                        "endpoint": {"value": endpoint},
                        "outcome_value": {"value": value},
                    }
                    for outcome_id, endpoint, value in outcomes
                ],
            }
        ),
        encoding="utf-8",
    )


def test_the_second_pass_adds_records_without_dropping_the_first(tmp_path):
    baseline = tmp_path / "baseline"
    second = tmp_path / "structured_compact_one_call_v1"
    _result(baseline / PAPER_ID / "final_result.json", [("O1", "deletion frequency", 16.5)])
    _result(
        second / PAPER_ID / "result.json",
        [("O1", "deletion frequency", 16.5), ("O3", "insertion frequency", 1.01)],
    )

    merged, detail = merge_paper(
        PAPER_ID, baseline_roots=[baseline], pass_root=second
    )

    values = [row["outcome_value"]["value"] for row in merged["outcomes"]]
    assert values == [16.5, 1.01], "the shared record must not be duplicated"
    assert detail["outcomes_added_by_second_pass"] == 1
    assert detail["baseline_root"] == "baseline"


def test_an_earlier_stages_merge_report_travels_with_the_result(tmp_path):
    """The evaluation counts unresolved candidates from the report beside the
    result, and those ids are not in the second pass's id space."""
    baseline = tmp_path / "baseline"
    _result(baseline / PAPER_ID / "final_result.json", [("O1", "editing", 45.0)])
    (baseline / PAPER_ID / "merge_report.json").write_text(
        json.dumps({"unresolved_candidate_ids": ["OC-1", "OC-2"]}), encoding="utf-8"
    )

    build_merge(
        baseline_roots=[baseline],
        pass_root=tmp_path / "absent",
        output_root=tmp_path / "merged",
        paper_ids=[PAPER_ID],
    )

    carried = json.loads(
        (tmp_path / "merged" / PAPER_ID / "merge_report.json").read_text(encoding="utf-8")
    )
    assert carried["unresolved_candidate_ids"] == ["OC-1", "OC-2"]


def test_a_paper_with_no_second_pass_is_carried_through_unchanged(tmp_path):
    baseline = tmp_path / "baseline"
    _result(baseline / PAPER_ID / "final_result.json", [("O1", "editing", 45.0)])

    merged, detail = merge_paper(
        PAPER_ID, baseline_roots=[baseline], pass_root=tmp_path / "absent"
    )

    assert detail["second_pass_present"] is False
    assert [row["outcome_id"] for row in merged["outcomes"]] == ["O1"]


def test_the_deepest_baseline_stage_wins_per_paper(tmp_path):
    deep, shallow = tmp_path / "deep", tmp_path / "shallow"
    _result(deep / PAPER_ID / "final_result.json", [("O9", "deep", 1.0)])
    _result(shallow / PAPER_ID / "result.json", [("O1", "shallow", 2.0)])

    assert resolve_baseline(PAPER_ID, [deep, shallow]) == deep
    merged, _ = merge_paper(
        PAPER_ID, baseline_roots=[deep, shallow], pass_root=tmp_path / "absent"
    )
    assert [row["outcome_id"] for row in merged["outcomes"]] == ["O9"]


def test_build_records_every_paper_it_was_asked_about(tmp_path):
    baseline = tmp_path / "baseline"
    _result(baseline / PAPER_ID / "final_result.json", [("O1", "editing", 45.0)])

    report = build_merge(
        baseline_roots=[baseline],
        pass_root=tmp_path / "absent",
        output_root=tmp_path / "merged",
        paper_ids=[PAPER_ID, "GP-MISSING"],
    )

    assert [row["paper_id"] for row in report["papers"]] == [PAPER_ID, "GP-MISSING"]
    assert (tmp_path / "merged" / PAPER_ID / "final_result.json").exists()
    assert report["papers"][0]["carried_merge_report"] is False
    assert not (tmp_path / "merged" / "GP-MISSING").exists()
    manifest = json.loads(
        (tmp_path / "merged" / "merge_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["second_pass_root"].endswith("absent")


def test_select_evidence_rejects_a_non_positive_budget():
    with pytest.raises(ValueError, match="must be positive"):
        select_evidence(_packet([("s", "table", "x")]), budget_tokens=0)
