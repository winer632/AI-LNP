"""Budget-sized packets that never drop a table row or a caption.

The whole reason this packet exists is that the sentence ranker reads
structured blocks badly. A table row is one grid line; on its own it scores as
noise, and the retrieval tags that would rescue it are exactly what corpus-
expanded table content does not have. So the rule is unconditional: every
passage from a ``table`` / ``table_row`` / ``caption`` / ``figure_caption`` /
``figure`` block is kept, and only then is the remaining budget filled by
retrieval rank. A structured passage evicted by a budget heuristic is the
failure this module was written to prevent.

The real fixture is GP-005's committed full evidence view, whose 135 structured
passages include 53 that the ranker scores at the bottom -- the ones a
rank-first selection drops first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.extraction.run_codex_one_call import load_packet
from src.rag.compact_api_packet import (
    API_PACKET_VERSION,
    ApiEvidence,
    ApiSource,
    CompactApiPacket,
    estimate_tokens,
)
from src.rag.structured_compact_packet import (
    STRUCTURED_BLOCK_TYPES,
    _retrieval_rank,
    _source_block_types,
    build_structured_compact_packet,
    is_structured,
    select_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_PACKET = REPO_ROOT / "data/staging/rag/full_api_packets_v1/GP-005.json"
PAPER_ID = "GP-005"


@pytest.fixture(scope="module")
def full_packet() -> CompactApiPacket:
    return CompactApiPacket.model_validate_json(
        FULL_PACKET.read_text(encoding="utf-8")
    )


def _structured(packet: CompactApiPacket) -> list[ApiEvidence]:
    block_types = _source_block_types(packet)
    return [row for row in packet.evidence if is_structured(row, block_types)]


def _from_block_type(packet: CompactApiPacket, block_type: str) -> list[ApiEvidence]:
    source_ids = {
        source.source_id
        for source in packet.sources
        if source.block_type == block_type
    }
    return [
        row
        for row in packet.evidence
        if set(row.source_ids or ()) & source_ids
    ]


def test_the_fixture_still_contains_low_ranked_structured_evidence(full_packet):
    """Precondition: the real packet has table passages the ranker buries."""
    tables = _from_block_type(full_packet, "table")
    assert tables, "GP-005 no longer carries table-derived evidence"
    assert any(_retrieval_rank(row) == 0 for row in _structured(full_packet))


def test_no_structured_passage_is_dropped_by_a_budget_that_fits_nothing(full_packet):
    structured = _structured(full_packet)
    assert structured

    selected, report = select_evidence(full_packet, budget_tokens=200)

    selected_ids = {row.evidence_id for row in selected}
    missing = [row.evidence_id for row in structured if row.evidence_id not in selected_ids]
    assert missing == []
    assert report["structured_evidence"] == len(structured)
    assert report["filler_evidence"] == 0
    # The budget really did bite; it just was not allowed to touch structure.
    assert report["excluded_evidence"] > 0
    assert report["estimated_evidence_tokens"] > 200
    assert report["structured_evidence_over_budget"] is True


def test_a_table_row_the_ranker_scores_last_is_still_selected(full_packet):
    """A table passage with no retrieval tags is the first thing rank drops."""
    buried = [
        row
        for row in _from_block_type(full_packet, "table")
        if _retrieval_rank(row) == 0
    ]
    assert buried, "no bottom-ranked table evidence left to protect"
    top_rank = max(_retrieval_rank(row) for row in full_packet.evidence)

    selected, report = select_evidence(full_packet, budget_tokens=200)

    selected_ids = {row.evidence_id for row in selected}
    assert all(row.evidence_id in selected_ids for row in buried)
    # Meanwhile the budget did evict passages the ranker liked far better,
    # which is what makes keeping the table rows a decision rather than luck.
    excluded_top = [
        row
        for row in full_packet.evidence
        if row.evidence_id not in selected_ids and _retrieval_rank(row) == top_rank
    ]
    assert excluded_top
    assert report["excluded_evidence"] >= len(excluded_top)


# Spelled out rather than read off the module: a test that takes its cases
# from the constant it is checking cannot notice the constant shrinking.
PROTECTED_BLOCK_TYPES = ["caption", "figure", "figure_caption", "table", "table_row"]


def test_the_protected_block_types_are_the_five_that_carry_structure():
    assert sorted(STRUCTURED_BLOCK_TYPES) == PROTECTED_BLOCK_TYPES


@pytest.mark.parametrize("block_type", PROTECTED_BLOCK_TYPES)
def test_every_structured_block_type_is_protected(block_type):
    """All five types, including the ones GP-005 happens not to carry."""
    answer = "Table 3 row: LNP-7 editing efficiency 1.01% in LSECs."
    sources = [
        ApiSource(
            source_id="S-structured",
            chunk_id="c-structured",
            source_path="paper.xml",
            source_kind="pmc_xml",
            block_type=block_type,
            section="Results",
        ),
        ApiSource(
            source_id="S-prose",
            chunk_id="c-prose",
            source_path="paper.xml",
            source_kind="pmc_xml",
            block_type="paragraph",
            section="Results",
        ),
    ]
    evidence = [
        # Deliberately worst case: the answer is long, untagged and unanchored,
        # so every budget heuristic wants it gone.
        ApiEvidence(
            evidence_id="GP-X-E-answer",
            text=answer + " " + "padding text. " * 200,
            retrieval_field_tags=[],
            source_ids=["S-structured"],
        ),
    ] + [
        ApiEvidence(
            evidence_id=f"GP-X-E-{index:03d}",
            text="Highly ranked prose about the outcome. " * 20,
            retrieval_field_tags=["outcomes", "payload"],
            experiment_candidate_ids=["EC-1"],
            source_ids=["S-prose"],
        )
        for index in range(30)
    ]
    packet = CompactApiPacket(
        packet_version=API_PACKET_VERSION,
        paper_id="GP-X",
        blocked_fields=[],
        sources=sources,
        evidence=evidence,
        packet_checksum="0" * 64,
    )

    selected, report = select_evidence(packet, budget_tokens=1_500)

    assert "GP-X-E-answer" in {row.evidence_id for row in selected}
    assert report["structured_evidence"] == 1
    assert report["excluded_evidence"] > 0


def test_filler_is_chosen_by_retrieval_rank(full_packet):
    block_types = _source_block_types(full_packet)
    selected, report = select_evidence(full_packet, budget_tokens=16_000)

    selected_ids = {row.evidence_id for row in selected}
    kept_filler = [
        row
        for row in selected
        if not is_structured(row, block_types)
    ]
    dropped = [
        row
        for row in full_packet.evidence
        if row.evidence_id not in selected_ids and not is_structured(row, block_types)
    ]
    assert kept_filler and dropped
    assert report["filler_evidence"] == len(kept_filler)
    assert min(_retrieval_rank(row) for row in kept_filler) >= max(
        _retrieval_rank(row) for row in dropped
    )


def test_selection_keeps_the_packet_order(full_packet):
    selected, _ = select_evidence(full_packet, budget_tokens=16_000)
    order = {row.evidence_id: index for index, row in enumerate(full_packet.evidence)}
    positions = [order[row.evidence_id] for row in selected]
    assert positions == sorted(positions)


def test_a_non_positive_budget_is_refused(full_packet):
    with pytest.raises(ValueError):
        select_evidence(full_packet, budget_tokens=0)


def test_the_built_packet_is_signed_so_load_packet_accepts_it(tmp_path):
    packet, report = build_structured_compact_packet(PAPER_ID)

    (tmp_path / f"{PAPER_ID}.json").write_text(
        packet.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
    )
    reloaded = load_packet(PAPER_ID, tmp_path)

    assert reloaded.packet_checksum == packet.packet_checksum
    assert report["paper_id"] == PAPER_ID
    assert report["structured_evidence"] == len(_structured(packet))


@pytest.mark.parametrize(
    ("paper_id", "budget_tokens"),
    # Budgets chosen because they really do drop a passage some other passage
    # points at; a budget that drops nothing would not exercise the cleanup.
    [("GP-005", 8_000), ("GP-002", 3_000)],
)
def test_the_built_packet_has_no_dangling_references(paper_id, budget_tokens):
    packet, _ = build_structured_compact_packet(paper_id, budget_tokens=budget_tokens)

    evidence_ids = {row.evidence_id for row in packet.evidence}
    source_ids = {row.source_id for row in packet.sources}
    surviving_pointers = 0
    for row in packet.evidence:
        for pointer in (row.context_before_evidence_id, row.context_after_evidence_id):
            assert pointer is None or pointer in evidence_ids, row.evidence_id
            surviving_pointers += pointer is not None
        assert set(row.source_ids or ()) <= source_ids
    # Pointers into surviving neighbours are kept, not blanket-erased.
    assert surviving_pointers > 0
    # And nothing rides along: a source kept without evidence is dead weight
    # in a packet whose whole purpose is fitting a budget.
    used_source_ids = {
        source_id for row in packet.evidence for source_id in (row.source_ids or ())
    }
    assert source_ids == used_source_ids


def test_the_built_packet_stays_within_the_budget_when_structure_fits(tmp_path):
    packet, report = build_structured_compact_packet(PAPER_ID, budget_tokens=16_000)

    assert report["structured_evidence_over_budget"] is False
    used = sum(
        estimate_tokens(row.model_dump(mode="json", exclude_none=True))
        for row in packet.evidence
    )
    assert used <= 16_000
    assert report["full_view_evidence"] > len(packet.evidence)
