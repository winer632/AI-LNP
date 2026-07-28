"""Structured supplement ingestion, anchored on the real GP-006 Table S2.

The uniparse payload below is copied verbatim from a live
`POST /api/v1/parse` response for
`data/raw/fulltext/oa_packages/PMC11617921/mmc1.pdf`, so these tests reproduce
the production parse offline.
"""

import base64
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from src.extraction.build_full_outcome_inventory import _source_id
from src.rag.compact_packet import block_from_hit
from src.rag.ingestion import (
    ROOT,
    grid_to_html,
    pdf_blocks,
    table_grid_from_html,
    table_row_entries,
    table_row_texts,
    uniparse_blocks,
    xml_blocks,
)
from src.rag.models import DocumentBlock

MMC1_FIXTURE = ROOT / "tests/fixtures/pdf_pages/GP-006_mmc1_p02.pdf"
from src.rag.uniparse_client import UniparseDocument


TABLE_S1_HTML = (
    "<table><tr><td></td><td>-1</td><td>-2</td><td>total deletion frequency</td></tr>"
    "<tr><td>Hepatocyte</td><td>34.40 ± 5.41 %</td><td>4.36 ± 1.29 %</td>"
    "<td>60.54 ± 11.65 %</td></tr>"
    "<tr><td>LSEC</td><td>7.68 ± 1.90 %</td><td>1.03 ± 0.28 %</td>"
    "<td>16.50 ± 2.96 %</td></tr></table>"
)
TABLE_S2_HTML = (
    "<table><tr><td></td><td>+1</td><td>+2</td><td>+3</td><td>+4</td><td>&gt;+4</td>"
    "<td>total insertion frequency</td></tr>"
    "<tr><td>Hepatocyte</td><td>3.39 ± 0.91 %</td><td>0.83 ± 0.27 %</td>"
    "<td>0.01 ± 0.01 %</td><td>0.00</td><td>0.00</td><td>4.23 ± 1.17 %</td></tr>"
    "<tr><td>LSEC</td><td>0.78 ± 0.27 %</td><td>0.21 ± 0.10 %</td>"
    "<td>0.01 ± 0.01 %</td><td>0.00</td><td>0.00</td><td>1.01 ± 0.38 %</td></tr></table>"
)
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\ntable-crop").decode()
MMC1 = ROOT / "data/raw/fulltext/oa_packages/PMC11617921/mmc1.pdf"


def mmc1_response() -> dict:
    return {
        "meta": {"filesize": 336145, "version": "1.1.0"},
        "pages": [
            {
                "page_id": 0,
                "content": [
                    {"type": "text", "bbox": [69.3, 72.0, 165.8, 84.5],
                     "text": "OMTN, Volume 35", "text_format": "markdown"},
                    {"type": "heading", "bbox": [69.3, 122.9, 243.0, 140.1],
                     "text": "Supplemental information", "text_format": "markdown",
                     "heading_level": 1},
                ],
                "others": [],
            },
            {
                "page_id": 1,
                "content": [
                    {"type": "heading", "bbox": [227.6, 73.6, 384.3, 90.2],
                     "text": "Supplemental Material", "text_format": "markdown",
                     "heading_level": 1},
                    {
                        "type": "image",
                        "bbox": [73.4, 125.1, 545.9, 264.5],
                        "text": None,
                        "text_format": None,
                        "img_path": "images/page_001_block_001_image_23f8b39c37.png",
                        "captions": [{
                            "bbox": [68.5, 276.4, 523.8, 304.9],
                            "type": "image_caption",
                            "text": (
                                "Figure S1. Representative flow cytometry of LSECs "
                                "after anti-mouse CD146 column isolation."
                            ),
                        }],
                        "footnotes": [],
                    },
                    {
                        "type": "table",
                        "bbox": [70.3, 379.3, 548.3, 444.3],
                        "text": TABLE_S1_HTML,
                        "text_format": "html",
                        "img_path": "images/page_001_block_004_table_f4760a44f1.png",
                        "captions": [{
                            "bbox": [69.7, 358.7, 422.8, 372.2],
                            "type": "table_caption",
                            "text": (
                                "Table S1. Deletion Frequency Patterns Following "
                                "Cas9/sgRNA LNP Treatment."
                            ),
                        }],
                        "footnotes": [],
                    },
                    {
                        "type": "table",
                        "bbox": [70.38, 536.976, 549.576, 602.712],
                        "text": TABLE_S2_HTML,
                        "text_format": "html",
                        "img_path": "images/page_001_block_006_table_6e37cb3b62.png",
                        "captions": [{
                            "bbox": [69.156, 515.592, 424.728, 529.056],
                            "type": "table_caption",
                            "text": (
                                "Table S2. Insertion Frequency Patterns Following "
                                "Cas9/sgRNA LNP Treatment."
                            ),
                        }],
                        "footnotes": [],
                    },
                ],
                "others": [
                    {"type": "page_number", "bbox": [302.3, 731.0, 308.4, 740.5],
                     "text": "1"},
                ],
            },
        ],
        "images": {"images/page_001_block_006_table_6e37cb3b62.png": PNG},
        "markdown": "",
    }


def mmc1_blocks(images=None) -> list[DocumentBlock]:
    document = UniparseDocument.model_validate(mmc1_response())
    return uniparse_blocks("GP-006", MMC1, document, images=images or {})


# --------------------------------------------------------------------------
# The acceptance case: Table S2 / LSEC / total insertion frequency
# --------------------------------------------------------------------------


def test_table_s2_lsec_insertion_frequency_becomes_a_citable_block():
    """The pre-uniparse PyMuPDF page dump lost this value entirely."""
    blocks = mmc1_blocks()
    rows = [
        block for block in blocks
        if block.block_type == "table_row" and "1.01 ± 0.38 %" in block.text
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row.table_number == "Table S2"
    assert "LSEC" in row.text
    assert "total insertion frequency: 1.01 ± 0.38 %" in row.text
    # A row block must be answerable on its own, without its neighbours.
    assert "Hepatocyte" not in row.text


def test_table_block_keeps_the_whole_table_as_html():
    table = next(
        block for block in mmc1_blocks()
        if block.block_type == "table" and block.table_number == "Table S2"
    )
    assert table.table_html is not None
    assert "1.01 ± 0.38 %" in table.table_html
    assert "total insertion frequency" in table.table_html
    assert table.caption_text.startswith("Table S2.")
    assert "1.01 ± 0.38 %" in table.text


# --------------------------------------------------------------------------
# Block inventory
# --------------------------------------------------------------------------


def test_uniparse_produces_structured_types_and_never_pdf_page():
    blocks = mmc1_blocks()
    kinds = {block.block_type for block in blocks}
    assert {"heading", "figure", "table", "table_row", "caption"} <= kinds
    assert "pdf_page" not in kinds
    assert {block.source_kind for block in blocks} == {"uniparse"}
    assert all(block.parser == "uniparse-1.1.0" for block in blocks)


def test_page_furniture_never_enters_the_evidence_pool():
    """`others` holds running headers, footers and page numbers."""
    blocks = mmc1_blocks()
    assert all(block.text.strip() != "1" for block in blocks)
    assert all("page_number" not in (block.block_type or "") for block in blocks)


def test_page_numbers_are_one_based():
    blocks = mmc1_blocks()
    assert {block.page_number for block in blocks} == {1, 2}
    table = next(block for block in blocks if block.table_number == "Table S2")
    assert table.page_number == 2


def test_headings_build_the_section_path():
    blocks = mmc1_blocks()
    table = next(block for block in blocks if block.table_number == "Table S2")
    assert table.section_path == "Supplement > Supplemental Material"
    heading = next(block for block in blocks if block.block_type == "heading")
    # A heading is filed under its parent, not under itself.
    assert heading.section_path == "Supplement"


def test_figure_block_carries_its_caption_and_image_reference():
    figure = next(block for block in mmc1_blocks() if block.block_type == "figure")
    assert figure.figure_number == "Figure S1"
    assert figure.text.startswith("Figure S1.")
    assert figure.image_ref == "images/page_001_block_001_image_23f8b39c37.png"


def test_bbox_and_image_digest_are_recorded():
    images = {
        "images/page_001_block_006_table_6e37cb3b62.png": {
            "path": "data/raw/fulltext/uniparse/GP-006/x.png",
            "bytes": 10,
            "sha256": "d" * 64,
        }
    }
    table = next(
        block for block in mmc1_blocks(images=images)
        if block.block_type == "table" and block.table_number == "Table S2"
    )
    assert table.image_sha256 == "d" * 64
    assert table.bbox is not None
    assert (table.bbox.x0, table.bbox.y1) == (70.38, 602.712)


def test_table_captions_become_their_own_caption_blocks():
    captions = [
        block for block in mmc1_blocks() if block.block_type == "caption"
    ]
    assert {caption.table_number for caption in captions} == {"Table S1", "Table S2"}


def test_caption_blocks_use_their_own_geometry_not_the_table_body():
    caption = next(
        block for block in mmc1_blocks()
        if block.block_type == "caption" and block.table_number == "Table S2"
    )
    # The caption sits above the table; vision crops depend on this bbox.
    assert (caption.bbox.x0, caption.bbox.y0) == (69.156, 515.592)


def test_block_ids_are_deterministic_and_unique():
    first = [block.block_id for block in mmc1_blocks()]
    second = [block.block_id for block in mmc1_blocks()]
    assert first == second
    assert len(first) == len(set(first))


# --------------------------------------------------------------------------
# Table helpers
# --------------------------------------------------------------------------


def test_table_grid_decodes_html_entities():
    grid = table_grid_from_html(TABLE_S2_HTML)
    assert grid[0][5] == ">+4"
    assert grid[2] == [
        "LSEC", "0.78 ± 0.27 %", "0.21 ± 0.10 %", "0.01 ± 0.01 %",
        "0.00", "0.00", "1.01 ± 0.38 %",
    ]


def test_row_text_avoids_sentence_boundaries_that_would_split_the_row():
    from src.rag.compact_packet import sentence_clauses

    row = table_row_texts(table_grid_from_html(TABLE_S2_HTML), "Table S2")[1]
    assert len(sentence_clauses(row)) == 1


def test_grid_to_html_round_trips():
    grid = table_grid_from_html(TABLE_S2_HTML)
    assert table_grid_from_html(grid_to_html(grid)) == grid


def test_single_row_table_produces_no_row_blocks():
    assert table_row_texts([["only", "header"]], "Table S9") == []


# --------------------------------------------------------------------------
# PMC XML tables keep their structure too
# --------------------------------------------------------------------------


XML_WITH_TABLE = """<article>
  <front><article-meta><title-group><article-title>T</article-title></title-group>
  </article-meta></front>
  <body><sec><title>Results</title>
    <table-wrap id="tbl1">
      <label>Table 1</label>
      <caption><p>Editing outcomes per cell type.</p></caption>
      <table>
        <thead><tr><th>Cell</th><th>total insertion frequency</th></tr></thead>
        <tbody>
          <tr><td>Hepatocyte</td><td>4.23 &#177; 1.17 %</td></tr>
          <tr><td>LSEC</td><td>1.01 &#177; 0.38 %</td></tr>
        </tbody>
      </table>
    </table-wrap>
  </sec></body>
</article>
"""


def test_xml_tables_keep_html_and_gain_row_blocks(tmp_path: Path):
    path = tmp_path / "article.nxml"
    path.write_text(XML_WITH_TABLE, encoding="utf-8")
    blocks = xml_blocks("GP-X", path)

    table = next(block for block in blocks if block.block_type == "table")
    assert table.table_html is not None
    assert "total insertion frequency" in table.table_html
    assert table.caption_text == "Editing outcomes per cell type."

    rows = [block for block in blocks if block.block_type == "table_row"]
    assert len(rows) == 2
    lsec = next(row for row in rows if "LSEC" in row.text)
    assert "total insertion frequency: 1.01 ± 0.38 %" in lsec.text
    assert lsec.table_number == "Table 1"
    assert lsec.xml_element_id == "tbl1"
    assert lsec.source_kind == "pmc_xml"


def test_xml_table_grid_reads_thead_and_tbody():
    from src.rag.ingestion import table_grid_from_xml

    root = ET.fromstring(XML_WITH_TABLE)
    wrap = next(node for node in root.iter() if node.tag == "table-wrap")
    grid = table_grid_from_xml(wrap)
    assert grid[0] == ["Cell", "total insertion frequency"]
    assert grid[2] == ["LSEC", "1.01 ± 0.38 %"]


# --------------------------------------------------------------------------
# Availability, fallback and backwards compatibility
# --------------------------------------------------------------------------


class _FailingClient:
    def parse_pdf(self, path, **kwargs):
        raise RuntimeError("uniparse is down")


@pytest.fixture
def uniparse_on(monkeypatch):
    """Pin the switch so the ambient environment cannot change the outcome."""
    monkeypatch.setenv("UNIPARSE_ENABLED", "1")


# The committed page fixture stands in when the untracked supplement is absent,
# so these run in a clone. See tests/fixtures/pdf_pages.
@pytest.mark.skipif(
    not (MMC1.exists() or MMC1_FIXTURE.exists()),
    reason="neither the GP-006 supplement nor its page fixture is present",
)
def test_pdf_blocks_falls_back_to_pymupdf_with_a_warning(tmp_path, uniparse_on):
    warnings: list[str] = []
    blocks = pdf_blocks(
        "GP-006", MMC1 if MMC1.exists() else MMC1_FIXTURE, client=_FailingClient(),
        image_root=tmp_path, warnings=warnings,
    )
    assert blocks
    assert {block.block_type for block in blocks} == {"pdf_page"}
    assert {block.source_kind for block in blocks} == {"pdf"}
    assert len(warnings) == 1
    assert "uniparse is down" in warnings[0]


# The committed page fixture stands in when the untracked supplement is absent,
# so these run in a clone. See tests/fixtures/pdf_pages.
@pytest.mark.skipif(
    not (MMC1.exists() or MMC1_FIXTURE.exists()),
    reason="neither the GP-006 supplement nor its page fixture is present",
)
def test_pdf_blocks_can_be_disabled_by_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("UNIPARSE_ENABLED", "0")

    class _Exploding:
        def parse_pdf(self, path, **kwargs):
            raise AssertionError("uniparse must not be called when disabled")

    blocks = pdf_blocks("GP-006", MMC1 if MMC1.exists() else MMC1_FIXTURE, client=_Exploding(), image_root=tmp_path)
    assert {block.block_type for block in blocks} == {"pdf_page"}


# The committed page fixture stands in when the untracked supplement is absent,
# so these run in a clone. See tests/fixtures/pdf_pages.
@pytest.mark.skipif(
    not (MMC1.exists() or MMC1_FIXTURE.exists()),
    reason="neither the GP-006 supplement nor its page fixture is present",
)
def test_fallback_can_be_switched_off_so_failures_surface(tmp_path, uniparse_on):
    with pytest.raises(RuntimeError, match="uniparse is down"):
        pdf_blocks(
            "GP-006", MMC1 if MMC1.exists() else MMC1_FIXTURE, client=_FailingClient(),
            image_root=tmp_path, allow_fallback=False,
        )


def test_committed_block_corpora_still_load():
    """New DocumentBlock fields must all be optional."""
    corpus = ROOT / "data/staging/rag/gold_v1"
    paths = sorted(corpus.glob("GP-*.blocks.jsonl"))
    assert len(paths) == 9
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                DocumentBlock.model_validate_json(line)


def test_block_from_hit_still_builds_a_block_without_the_new_fields():
    block = block_from_hit({
        "block_id": "B1",
        "paper_id": "GP-X",
        "source_path": "supplement.pdf",
        "section_path": "Supplement",
        "text": "Total insertion frequency was 1.01 ± 0.38 %.",
    })
    assert block.bbox is None
    assert block.table_html is None
    assert block.image_sha256 is None


def test_structured_fields_do_not_shift_existing_source_ids():
    """FC-S-* ids are frozen: adding schema fields must not re-key them."""
    base = dict(
        block_id="GP-006-B-abc", paper_id="GP-006",
        source_path="data/raw/fulltext/oa_packages/PMC11617921/mmc1.pdf",
        source_kind="uniparse", section_path="Supplement", block_type="table",
        text="Table S2 | LSEC | total insertion frequency: 1.01 ± 0.38 %",
        page_number=2, table_number="Table S2", char_end=10,
        parser="uniparse-1.1.0", parser_confidence=0.9,
    )
    plain = DocumentBlock(**base)
    enriched = DocumentBlock(
        **base,
        table_html=TABLE_S2_HTML,
        caption_text="Table S2. Insertion Frequency Patterns.",
        image_ref="images/page_001_block_006_table_6e37cb3b62.png",
        image_sha256="e" * 64,
        bbox={"x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0},
    )
    assert _source_id(plain) == _source_id(enriched)


def test_a_header_less_table_keeps_its_first_row_and_does_not_leak_it():
    """The bug that got published as a uniparse hallucination.

    GP-004's Supplementary Table 2 is a two-column label/sequence table with no
    header row. Treating grid[0] as a header consumed the Luciferase row
    entirely -- it received no block at all -- and pasted its 1.6 kb sequence
    into every other row as a pseudo column name. Truncated for inspection the
    eGFP row then read as though it carried Luciferase's sequence, and that
    reading was written into the design document, a flag rationale and a
    docstring as evidence that uniparse had fabricated data.

    uniparse had not. Its eGFP row matches the real eGFP sequence at 0.977 and
    Luciferase at 0.113. The parser did this.
    """
    grid = [
        ["Luciferase", "ATGGAGGACGCCAAGAACATCAAG"],
        ["eGFP", "ATGGTGAGCAAGGGCGAGGAGCTG"],
        ["HGF", "ATGTGGGTGACCAAGCTGCTGCCC"],
    ]
    entries = table_row_entries(grid, "Table 2")

    assert len(entries) == 3, "a header-less table lost a data row"
    rendered = [text for _, text in entries]
    assert any("Luciferase" in text for text in rendered)
    # No row may carry another row's sequence.
    for text, own in zip(rendered, ("ATGGAGGACGCCAAGAACATCAAG",
                                    "ATGGTGAGCAAGGGCGAGGAGCTG",
                                    "ATGTGGGTGACCAAGCTGCTGCCC")):
        assert own in text
        others = {"ATGGAGGACGCCAAGAACATCAAG", "ATGGTGAGCAAGGGCGAGGAGCTG",
                  "ATGTGGGTGACCAAGCTGCTGCCC"} - {own}
        for other in others:
            assert other not in text, "a row leaked another row's sequence"


def test_a_real_header_is_still_used_as_column_names():
    """The fix must not cost the capability it protects."""
    grid = [["Cell type", "Frequency"], ["LSEC", "1.01"], ["Hepatocyte", "16.5"]]
    entries = table_row_entries(grid, "Table S2")
    assert len(entries) == 2, "a genuine header row became a data row"
    assert "Frequency: 1.01" in entries[0][1]


def test_a_lone_header_row_still_produces_nothing():
    assert table_row_entries([["only", "header"]], "Table S9") == []


def test_a_lone_data_row_is_not_discarded_as_a_header():
    assert len(table_row_entries([["Luciferase", "ATGGAGGACGCCAAGAAC"]], "T")) == 1
