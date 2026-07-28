"""Deterministic, human-legible evidence ids (design document, section 9).

The design document names one defect and one target:

    Evidence ID | GP-001-E-0431f0f16a6f0571 (content hash, drifts the moment
                | the content moves)
                | -> structured deterministic ID, e.g.
                |    GP-006-mmc1-p01-tabS2-r2-c7

The constraint on implementing it is that evidence ids are cited by committed
extraction results and are what the evaluator resolves a record's claim
against, so the new scheme has to be *additive*: a block carries both ids, and
nothing that already exists is re-keyed. The tests below are split accordingly
-- the grammar, then the additivity, then the frozen digests.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config_flags import override
from src.extraction.build_full_outcome_inventory import (
    _source_id as _full_view_source_id,
)
from src.extraction.evaluate_final_gold_dynamic import _evidence_texts
from src.rag.compact_api_packet import _source_id, build_api_packet
from src.rag.compact_packet import (
    SourceLocation,
    build_packet,
    source_location,
)
from src.rag.evidence_locators import (
    LocatorMinter,
    cell_locator,
    clause_locator,
    file_token,
    label_token,
    parse_locator,
    row_locator,
)
from src.rag.ingestion import (
    table_cell_locators,
    table_grid_from_html,
    uniparse_blocks,
    xml_blocks,
)
from src.rag.models import DocumentBlock


MMC1 = "data/raw/fulltext/oa_packages/PMC11617921/mmc1.pdf"

XML_WITH_TABLE = """<?xml version="1.0"?>
<article>
  <front><article-meta><title-group><article-title>Editing outcomes</article-title>
  </title-group></article-meta></front>
  <body><sec><title>Results</title>
    <p>Total insertion frequency was measured per cell type.</p>
    <p>LSECs showed a lower frequency than hepatocytes.</p>
    <table-wrap id="tbl1">
      <label>Table 1</label>
      <caption><p>Editing outcomes per cell type.</p></caption>
      <table>
        <thead><tr><th>Cell</th><th>total insertion frequency</th></tr></thead>
        <tbody>
          <tr><td>Hepatocyte</td><td>4.23 &#177; 1.17 %</td></tr>
          <tr><td>Not measured</td><td></td></tr>
          <tr><td>LSEC</td><td>1.01 &#177; 0.38 %</td></tr>
        </tbody>
      </table>
    </table-wrap>
  </sec></body>
</article>
"""


def _write_xml(tmp_path: Path, body: str = XML_WITH_TABLE) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "article.nxml"
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# The grammar
# --------------------------------------------------------------------------- #


def test_locator_reproduces_the_design_document_example():
    """The literal id from section 9's table, built from its parts.

    Pinning the exact string is the point: paper, source file, page, table,
    row and column each have to land in the right segment and the right form
    (``p01`` zero padded, ``tabS2`` with the noun stripped, ``r2`` and ``c7``
    unpadded), and there is no way to assert that other than against the
    example the document gives.
    """
    minter = LocatorMinter("GP-006")
    base = minter.table(
        source_path=MMC1, page_number=1, table_number="Table S2"
    )
    assert base == "GP-006-mmc1-p01-tabS2"
    assert row_locator(base, 2) == "GP-006-mmc1-p01-tabS2-r2"
    assert cell_locator(base, 2, 7) == "GP-006-mmc1-p01-tabS2-r2-c7"


def test_locator_parses_back_into_its_parts():
    parsed = parse_locator("GP-006-mmc1-p01-tabS2-r2-c7")
    assert parsed is not None
    assert (parsed.paper_id, parsed.file, parsed.page) == ("GP-006", "mmc1", 1)
    assert (parsed.anchor, parsed.row, parsed.column) == ("tabS2", 2, 7)
    assert str(parsed) == "GP-006-mmc1-p01-tabS2-r2-c7"
    # A hash id is not a locator, and asking must not raise: both populations
    # travel together for as long as the flag is opt-in.
    assert parse_locator("GP-001-E-0431f0f16a6f0571") is None


def test_file_and_label_tokens_keep_the_file_and_drop_the_noun():
    assert file_token(MMC1) == "mmc1"
    # Only the final suffix is a suffix; the rest is the file's name.
    assert file_token("a/b/pnas.2534673123.sapp.pdf") == "pnas2534673123sapp"
    assert label_token("Table S2") == "S2"
    assert label_token("Supplementary Figure 3") == "3"
    assert label_token("Table") is None
    assert label_token(None) is None


def test_unlabelled_blocks_are_numbered_within_their_page_and_kind():
    minter = LocatorMinter("GP-006")
    first = minter.block(
        source_path=MMC1, page_number=2, block_type="paragraph"
    )
    second = minter.block(
        source_path=MMC1, page_number=2, block_type="paragraph"
    )
    other_page = minter.block(
        source_path=MMC1, page_number=3, block_type="paragraph"
    )
    heading = minter.block(source_path=MMC1, page_number=2, block_type="heading")
    assert first == "GP-006-mmc1-p02-par001"
    assert second == "GP-006-mmc1-p02-par002"
    assert other_page == "GP-006-mmc1-p03-par001"
    assert heading == "GP-006-mmc1-p02-hed001"


def test_a_reclaimed_anchor_is_disambiguated_rather_than_duplicated():
    """A table printed across a page break is labelled once and parsed twice."""
    minter = LocatorMinter("GP-006")
    first = minter.table(source_path=MMC1, page_number=1, table_number="Table S2")
    second = minter.table(source_path=MMC1, page_number=1, table_number="Table S2")
    assert first == "GP-006-mmc1-p01-tabS2"
    assert second == "GP-006-mmc1-p01-tabS2x2"


def test_an_unpaginated_source_is_page_zero():
    minter = LocatorMinter("GP-006")
    assert minter.block(
        source_path="a/b/PMC11617921.nxml",
        page_number=None,
        block_type="paragraph",
    ) == "GP-006-PMC11617921-p00-par001"


# --------------------------------------------------------------------------- #
# Additivity: the new id never displaces the old one
# --------------------------------------------------------------------------- #


def test_ingestion_mints_no_locator_while_the_flag_is_off(tmp_path: Path):
    """The default. Nothing anywhere gains an id, so nothing can be re-keyed."""
    with override(structured_evidence_ids=False):
        blocks = xml_blocks("GP-X", _write_xml(tmp_path))
    assert blocks
    assert all(block.locator_id is None for block in blocks)


def test_a_block_keeps_its_hash_id_and_gains_a_locator(tmp_path: Path):
    path = _write_xml(tmp_path)
    with override(structured_evidence_ids=False):
        without = xml_blocks("GP-X", path)
    with override(structured_evidence_ids=True):
        with_locators = xml_blocks("GP-X", path)

    assert [block.block_id for block in without] == [
        block.block_id for block in with_locators
    ]
    assert all(block.locator_id for block in with_locators)
    # Everything except the added field is identical, so a corpus rebuilt with
    # the flag on is the old corpus plus one column.
    for old, new in zip(without, with_locators):
        assert old.model_dump(exclude={"locator_id"}) == new.model_dump(
            exclude={"locator_id"}
        )


def test_a_locator_survives_an_edit_that_moves_the_hash_id(tmp_path: Path):
    """The defect section 9 names: "内容一动即漂移", drift on any content change.

    A parser fix that changes a character changes the whole block_id. The
    locator addresses the position, so it does not move -- which is the entire
    reason for having one.
    """
    edited = XML_WITH_TABLE.replace(
        "LSECs showed a lower frequency than hepatocytes.",
        "LSECs showed a lower frequency than hepatocytes overall.",
    )
    # The same path both times: block_id also hashes the file path, so writing
    # to two directories would make every id differ for the wrong reason.
    path = _write_xml(tmp_path)
    with override(structured_evidence_ids=True):
        before = xml_blocks("GP-X", path)
        path.write_text(edited, encoding="utf-8")
        after = xml_blocks("GP-X", path)

    changed = [
        (old, new)
        for old, new in zip(before, after)
        if old.text != new.text
    ]
    assert changed, "the fixture must actually change one block's text"
    for old, new in changed:
        assert old.block_id != new.block_id
        assert old.locator_id == new.locator_id


def test_locators_are_unique_within_a_paper(tmp_path: Path):
    with override(structured_evidence_ids=True):
        blocks = xml_blocks("GP-X", _write_xml(tmp_path))
    locators = [block.locator_id for block in blocks]
    assert len(locators) == len(set(locators))


def test_table_rows_are_numbered_by_the_grid_not_by_emission(tmp_path: Path):
    """A row the flattener skips must not renumber the rows after it.

    The fixture's grid is header, Hepatocyte, "Not measured" (a row label and
    no value, which the flattener drops because it has no column/value pair),
    LSEC. LSEC is grid row 4 and the second *emitted* row, so a counter over
    emitted rows would call it r3 and point an auditor at the wrong line.
    """
    with override(structured_evidence_ids=True):
        blocks = xml_blocks("GP-X", _write_xml(tmp_path))
    rows = {
        block.locator_id: block.text
        for block in blocks
        if block.block_type == "table_row"
    }
    assert set(rows) == {"GP-X-article-p00-tab1-r2", "GP-X-article-p00-tab1-r4"}
    assert "LSEC" in rows["GP-X-article-p00-tab1-r4"]


def test_table_cell_locators_name_every_cell_of_the_parsed_grid(tmp_path: Path):
    with override(structured_evidence_ids=True):
        blocks = xml_blocks("GP-X", _write_xml(tmp_path))
    table = next(block for block in blocks if block.block_type == "table")
    cells = table_cell_locators(table)
    assert cells[(1, 1)] == "GP-X-article-p00-tab1-r1-c1"
    assert cells[(4, 2)] == "GP-X-article-p00-tab1-r4-c2"
    # Safe on a block that has neither a locator nor a grid.
    assert table_cell_locators(table.model_copy(update={"locator_id": None})) == {}


# --------------------------------------------------------------------------- #
# Frozen digests: the committed artifacts must not move
# --------------------------------------------------------------------------- #


def _block_for_digest(fields: dict) -> DocumentBlock:
    """A DocumentBlock carrying only the fields the FC-S- digest reads."""
    return DocumentBlock(
        block_id=fields["chunk_id"],
        paper_id=fields["chunk_id"].split("-B-")[0],
        source_path=fields["source_path"],
        source_kind=fields["source_kind"],
        section_path=fields["section"],
        block_type=fields["block_type"],
        text="x",
        char_end=1,
        parser="digest-probe",
        parser_confidence=1.0,
        page_number=fields.get("page_number"),
        xml_element_id=fields.get("xml_element_id"),
        table_number=fields.get("table_number"),
        figure_number=fields.get("figure_number"),
    )


def _location(**overrides) -> SourceLocation:
    base = dict(
        chunk_id="GP-006-B-9f0a1b2c3d4e5f607182",
        source_path=MMC1,
        source_kind="uniparse",
        block_type="table_row",
        section="Supplement",
        subsection="Tables",
        page_number=1,
        table_number="Table S2",
    )
    return SourceLocation(**{**base, **overrides})


def test_source_id_digest_is_frozen():
    """The S- id of a known location, pinned as a literal.

    Committed packets carry these ids and committed results cite the evidence
    they hang off, so this digest is not allowed to move -- not when a field is
    added to SourceLocation, and not when the new locator is populated. The
    literal is what makes "we did not change it" checkable rather than
    asserted.
    """
    # Verified equal to what the pre-change implementation returned for this
    # location before _source_id was narrowed to its frozen field list.
    assert _source_id(_location()) == "S-8dfc5c14f203"
    assert _source_id(
        _location(locator_id="GP-006-mmc1-p01-tabS2-r2")
    ) == "S-8dfc5c14f203"


def test_every_committed_source_id_is_reproduced_by_the_current_code():
    """Recompute every S- id in every committed packet and compare.

    A pinned literal proves one location did not move; this proves the whole
    committed population did not. Deliberately recomputed from each packet's
    own source rows rather than by rebuilding the packet: the rebuild differs
    from the committed baseline for 6 of 9 papers -- the code/data drift
    section 9 measured, present before this change and unaffected by it -- and
    a test of the digest must not be answering a question about selection.
    """
    root = Path(__file__).resolve().parents[1] / "data/staging/rag"
    packets = sorted(
        path
        for directory in (
            "compact_api_packets_v1",
            "compact_api_packets_v1_1",
            "structured_compact_packets_v1",
        )
        # Not GP-*.json: that also matches GP-001.manifest.json, which has no
        # sources and would make the loop pass by skipping everything.
        for path in (root / directory).glob("GP-[0-9][0-9][0-9].json")
    )
    assert len(packets) >= 9
    checked = 0
    for path in packets:
        for row in json.loads(path.read_text(encoding="utf-8"))["sources"]:
            fields = {key: value for key, value in row.items() if key != "source_id"}
            # Two id schemes are committed. S- comes from the compact packet's
            # SourceLocation digest; FC-S- from the full corpus view's digest
            # over a DocumentBlock. Both are frozen, so both are checked.
            if row["source_id"].startswith("FC-S-"):
                recomputed = _full_view_source_id(_block_for_digest(fields))
            else:
                recomputed = _source_id(SourceLocation(**fields))
            assert recomputed == row["source_id"], path.name
            checked += 1
    assert checked > 1000, "the committed packets should hold many more than this"


# --------------------------------------------------------------------------- #
# End to end: a locator reaches the packet the model is sent
# --------------------------------------------------------------------------- #


def _corpus(tmp_path: Path, blocks: list[DocumentBlock]) -> Path:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir(parents=True, exist_ok=True)
    (corpus_root / f"{blocks[0].paper_id}.blocks.jsonl").write_text(
        "".join(block.model_dump_json() + "\n" for block in blocks),
        encoding="utf-8",
    )
    return corpus_root


def _retrieval(tmp_path: Path, blocks: list[DocumentBlock]) -> Path:
    payload = {
        "paper_id": blocks[0].paper_id,
        "blocked_fields": {},
        "packets": {
            "outcomes": {
                "query": {"query_id": "Q1", "field_group": "outcome"},
                "hits": [
                    {
                        "block_id": block.block_id,
                        "paper_id": block.paper_id,
                        "text": block.text,
                        "section_path": block.section_path,
                        "source_path": block.source_path,
                        "source_kind": block.source_kind,
                        "block_type": block.block_type,
                        "page_number": block.page_number,
                        "table_number": block.table_number,
                        "locator_id": block.locator_id,
                        "entity_types": [],
                        "fused_score": 1.0,
                    }
                    for block in blocks
                ],
            }
        },
    }
    path = tmp_path / "retrieval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_locators_reach_the_api_packet_beside_the_hash_ids(tmp_path: Path):
    with override(structured_evidence_ids=True):
        blocks = xml_blocks("GP-X", _write_xml(tmp_path))
    packet = build_packet(
        _retrieval(tmp_path, blocks),
        corpus_root=_corpus(tmp_path, blocks),
        boundary_root=tmp_path / "absent",
    )
    api, _ = build_api_packet(packet)

    assert all(row.evidence_id.startswith("GP-X-E-") for row in api.evidence)
    assert any(row.locator_id for row in api.evidence)
    assert any(row.locator_id for row in api.sources)
    # The flattened row, not the whole-table block that also mentions 1.01:
    # the row is the passage a value can be cited from, and its locator names
    # the line of the table it came off.
    row = next(row for row in api.evidence if row.text.startswith("Table 1 | LSEC"))
    assert row.locator_id == "GP-X-article-p00-tab1-r4-s1"


def test_a_locator_on_a_hit_survives_reverse_construction(tmp_path: Path):
    """A packet built from a retrieval hit alone still carries the locator."""
    with override(structured_evidence_ids=True):
        blocks = xml_blocks("GP-X", _write_xml(tmp_path))
    # No corpus: build_packet falls back to block_from_hit for every chunk.
    packet = build_packet(
        _retrieval(tmp_path, blocks),
        corpus_root=tmp_path / "absent",
        boundary_root=tmp_path / "absent",
    )
    assert any(
        location.locator_id
        for evidence in packet.evidence
        for location in evidence.source_locations
    )


def test_source_location_carries_the_locator_through(tmp_path: Path):
    with override(structured_evidence_ids=True):
        blocks = xml_blocks("GP-X", _write_xml(tmp_path))
    block = next(block for block in blocks if block.block_type == "table_row")
    assert source_location(block).locator_id == block.locator_id


def test_the_evaluator_resolves_a_record_that_cites_the_locator(tmp_path: Path):
    """A legible id resolves to exactly the text its hash id resolves to.

    Additive here too: the locator is registered with ``setdefault``, so it can
    make an id resolvable but can never move one that already did.
    """
    packet_root = tmp_path / "packets"
    packet_root.mkdir()
    (packet_root / "GP-X.json").write_text(
        json.dumps(
            {
                "evidence": [
                    {
                        "evidence_id": "GP-X-E-0123456789abcdef",
                        "locator_id": "GP-X-article-p00-tab1-r4-s1",
                        "text": "Table 1 | LSEC | total insertion frequency: 1.01",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    texts = _evidence_texts(
        "GP-X",
        packet_root=packet_root,
        task_root=tmp_path / "absent",
        structured_packet_root=tmp_path / "absent",
    )
    assert texts["GP-X-E-0123456789abcdef"] == texts["GP-X-article-p00-tab1-r4-s1"]


def test_clause_locator_appends_the_sentence_ordinal():
    assert clause_locator("GP-006-mmc1-p03-par007", 2) == "GP-006-mmc1-p03-par007-s2"


def test_the_real_gp006_supplement_yields_the_documented_shape():
    """The design document's own example, on the parse it was written about.

    ``mmc1_response`` is a verbatim live ``POST /api/v1/parse`` response for
    ``PMC11617921/mmc1.pdf``, so this is the production parse offline. The
    document writes the example as ``GP-006-mmc1-p01-tabS2-r2-c7``; the real
    parse puts Table S2 on page 2 and LSEC on grid row 3, so the ids the code
    mints are ``p02`` and ``r3``. The shape, and every segment's meaning, is
    the document's.
    """
    from src.rag.uniparse_client import UniparseDocument
    from tests.test_uniparse_ingestion import MMC1, mmc1_response

    document = UniparseDocument.model_validate(mmc1_response())
    with override(structured_evidence_ids=True):
        blocks = uniparse_blocks("GP-006", MMC1, document)

    by_locator = {block.locator_id: block for block in blocks}
    assert len(by_locator) == len(blocks), "locators must be unique"
    # The table, its rows and its caption are all addressable and distinct.
    assert "GP-006-mmc1-p02-tabS2" in by_locator
    assert "GP-006-mmc1-p02-capS2" in by_locator
    row = by_locator["GP-006-mmc1-p02-tabS2-r3"]
    assert "LSEC" in row.text and "1.01" in row.text

    table = by_locator["GP-006-mmc1-p02-tabS2"]
    cells = table_cell_locators(table)
    # Column 7 is "total insertion frequency"; row 3 is LSEC. That cell is the
    # value GO-006 turns on, and this is the id that names it.
    assert cells[(3, 7)] == "GP-006-mmc1-p02-tabS2-r3-c7"
    assert table_grid_from_html(table.table_html)[2][6] == "1.01 ± 0.38 %"
