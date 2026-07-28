"""The section-十 cross-check over uniparse output.

uniparse is a VLM. Its table HTML is model output, and the design document says
in section 十 that leaving it unaudited "只是把幻觉从抽取阶段挪到解析阶段" -- it
merely moves hallucination from extraction to parsing, where it is harder to see
because it now looks like structured data.

Two real measurements anchor these tests, both taken with this module against
the committed corpus:

* **GP-006 Table S2 verifies.** The block that carries GO-006's ``1.01 ± 0.38 %``
  sits in a region with no text layer at all, so the parse was its only account.
  macOS Vision OCR of a 600 dpi crop of the recorded bbox reproduces every one
  of the row's twelve numbers, and a second uniparse read of the region cut out
  on its own returns byte-identical table HTML.
* **GP-004 Supplementary Tables 2 and 3 do not.** 42 of the 45 distinct
  sequence-length runs uniparse produced for pages 10-11 of
  ``41467_2021_20903_MOESM1_ESM.pdf`` do not appear in those pages' own text
  layer. It gave the eGFP and HGF rows the Luciferase row's sequence and
  corrupted the bases besides.

``VISION_LINES`` below is that OCR engine's real output, copied verbatim, so the
verdict logic can be exercised offline. It includes the two cells Vision got
*wrong* -- ``0.92`` for ``0.91``, ``0.02`` for ``0.01`` -- at confidence 0.5 and
0.3 while every cell it read correctly came back at 1.0. Those two are why
corroboration and contradiction are asymmetric, and the first cut of this module
failed exactly there: it condemned a correct table on a noisy reader's silence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config_flags import describe_flag, is_enabled, override
from src.rag import ingestion
from src.rag.ingestion import ROOT, pdf_blocks, uniparse_verification_enabled
from src.rag.models import DocumentBlock
from src.rag.uniparse_client import UniparseDocument
from src.rag.uniparse_verification import (
    HIGH_CONFIDENCE,
    IndependentRead,
    OcrLine,
    _verdict,
    block_body_text,
    canonical_number,
    crop_pdf_region,
    long_token_claims,
    numeric_claims,
    render_crop,
    verify_and_stamp,
    verify_blocks,
)


MMC1 = ROOT / "data/raw/fulltext/oa_packages/PMC11617921/mmc1.pdf"
GP004_SUPPLEMENT = ROOT / "data/raw/fulltext/oa_packages/PMC7840919/41467_2021_20903_MOESM1_ESM.pdf"
CORPUS = ROOT / "data/staging/rag/gold_v1"

# Table S2 on mmc1.pdf page 2, exactly as the committed corpus records it.
TABLE_S2_BBOX = (70.38, 536.976, 549.576, 602.712)
GO_006_BLOCK = "GP-006-B-098d79d1de1fdd6819a5"
# Located by content, not by id. A block id is a hash of its text, so pinning
# one makes any legitimate parser improvement look like a regression -- which
# it did: fixing the header-less-table flattener changed every row's text and
# broke these tests while the behaviour under test was unaffected.
GP004_SEQUENCE_MARKER = "ACGAGAACAAAGGACTACATCCGCAACTGC"

# Real macOS Vision output for a 600 dpi render of TABLE_S2_BBOX. See docstring.
VISION_LINES = (
    ("Hepatocyte", 1.0),
    ("LSEC", 1.0),
    ("+1", 1.0),
    ("3.39 $ 0.92%", 0.5),
    ("0.78 ÷ 0.27 %", 0.5),
    ("+2", 1.0),
    ("0.83 0.27 %", 0.3),
    ("0.21 ÷ 0.10%", 0.5),
    ("+3", 1.0),
    ("0.02 $ 0.02%", 0.3),
    ("0.01 ÷ 0.01 %", 0.5),
    ("+4", 1.0),
    ("0.00", 1.0),
    ("0.00", 1.0),
    (">+4", 1.0),
    ("0.00", 1.0),
    ("0.00", 1.0),
    ("total insertion", 1.0),
    ("frequency", 1.0),
    ("4.23 $ 1.17 %", 0.5),
    ("1.01 ÷ 0.38 %", 0.5),
)


class _RecordedOcr:
    """Replays a recorded read so the verdict logic is testable without OCR."""

    name = "recorded"

    def __init__(self, lines=VISION_LINES):
        self._lines = [OcrLine(text=text, confidence=score) for text, score in lines]
        self.calls = 0

    def read(self, png: bytes) -> list[OcrLine]:
        assert png[:4] == b"\x89PNG", "the engine must be handed a rendered crop"
        self.calls += 1
        return list(self._lines)


def on_readable_pdf(block):
    """Repoint a block at whichever copy of its page is actually present.

    verify_blocks resolves ROOT / block.source_path itself, so a test holding a
    committed block cannot redirect it from outside. Rewriting source_path and
    page_number on a copy keeps the fixture out of production code while
    letting these tests run in a clone. A no-op when the source is present.
    """
    # Only the two real supplements have fixtures. A synthetic block, or one
    # with no page, is returned untouched -- resolving it would turn a test
    # about missing geometry into a fixture lookup failure.
    known = {"mmc1.pdf", "41467_2021_20903_MOESM1_ESM.pdf"}
    if Path(block.source_path).name not in known or block.page_number is None:
        return block
    path, page = resolve_pdf(block.source_path, block.page_number)
    relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return block.model_copy(
        update={"source_path": str(relative), "page_number": page}
    )


def _find(paper_id: str, marker: str):
    """Return the one corpus block whose text contains `marker`."""
    import json

    path = CORPUS / f"{paper_id}.blocks.jsonl"
    matches = [
        DocumentBlock.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if marker in line
    ]
    assert matches, f"no block in {path.name} contains {marker!r}"
    # Prefer the row over the whole-table block: a table block spans several
    # rows, so feeding it one row's true text cannot verify it.
    rows = [block for block in matches if block.block_type == "table_row"]
    return (rows or matches)[0]


def _load(paper_id: str, block_id: str) -> DocumentBlock:
    path = CORPUS / f"{paper_id}.blocks.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and json.loads(line)["block_id"] == block_id:
            return DocumentBlock.model_validate_json(line)
    raise AssertionError(f"{block_id} is not in {path}")


def _read(method, tokens, *, decisive=False, deterministic=False, text=""):
    return IndependentRead(
        method=method,
        available=True,
        tokens=frozenset(tokens),
        decisive=decisive,
        deterministic=deterministic,
        text=text,
    )


# The source supplements live under data/raw/fulltext/, which is gitignored, so
# these tests used to skip in a fresh clone -- and they are the evidence for
# section 十, the one the design document calls mandatory. Eighteen of the
# nineteen skips in a clone were these. Committing the two pages they actually
# read makes the claim checkable by anyone who clones the repository, which is
# what section 九 asks of every claim here.
#
# The fixtures are single pages cut from PMC open-access packages: 315 KB and
# 98 KB against a 7.7 MB source.
FIXTURE_PAGES = ROOT / "tests/fixtures/pdf_pages"
MMC1_FIXTURE = FIXTURE_PAGES / "GP-006_mmc1_p02.pdf"
MMC1_REL = "data/raw/fulltext/oa_packages/PMC11617921/mmc1.pdf"
GP004_FIXTURE = FIXTURE_PAGES / "GP-004_MOESM1_p10-11.pdf"

# The fixtures keep the source's page numbering -- pages before the one we need
# are present but blank, which costs almost nothing and means no code anywhere
# has to translate a page number. A mapping would have been one more thing to
# get wrong, and it was: the first attempt renumbered the pages and every block
# came back unverified because the crop asked a one-page PDF for its page 2.


def resolve_pdf(source_relative: str, page_number: int) -> tuple[Path, int]:
    """Return a readable PDF and the page to read, preferring the real source.

    Falls back to the committed page fixture, translating the page number.
    Tests that use this run identically with or without the untracked source.
    """
    real = ROOT / source_relative
    if real.exists():
        return real, page_number
    name = Path(source_relative).name
    fixture = MMC1_FIXTURE if name == "mmc1.pdf" else GP004_FIXTURE
    assert fixture.exists(), (
        f"neither {source_relative} nor its page fixture is present"
    )
    return fixture, page_number


needs_mmc1 = pytest.mark.skipif(
    not (MMC1.exists() or MMC1_FIXTURE.exists()),
    reason="neither the GP-006 supplement nor its page fixture is present",
)
needs_gp004 = pytest.mark.skipif(
    not (GP004_SUPPLEMENT.exists() or GP004_FIXTURE.exists()),
    reason="neither the GP-004 supplement nor its page fixture is present",
)


# --------------------------------------------------------------------------- #
# Claim extraction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0.10", "0.1"), ("0.1", "0.1"), ("0.00", "0"), ("0", "0"), ("1.01", "1.01"),
     ("3.39", "3.39"), ("004", "4")],
)
def test_equivalent_spellings_of_a_number_compare_equal(raw, expected):
    """Two readers disagree about how to write a value, not about the value."""
    assert canonical_number(raw) == expected


def test_a_digit_glued_to_letters_is_not_a_numeric_claim():
    """"Table S2" and "Cas9" assert no measurement, and no reader reports one."""
    claims = numeric_claims("Table S2. Cas9/sgRNA in page_001, n = 5")
    assert "2" not in claims
    assert "9" not in claims
    assert "1" not in claims
    assert "5" in claims


def test_a_column_header_sign_does_not_split_the_number():
    assert numeric_claims("+1: 0.78 ± 0.27 %; >+4: 0.00") == ("1", "0.78", "0.27", "4", "0")


def test_the_synthesised_table_label_is_not_treated_as_a_claim_about_the_page():
    """Regression: it made every numbered table look contradicted.

    ``ingestion.table_row_texts`` prepends "Table 2 | " so a row can be cited on
    its own. That number comes from the caption, which sits outside the table's
    bbox, so no crop of the table body can ever contain it.
    """
    block = DocumentBlock(
        block_id="b", paper_id="GP-004", source_path="s.pdf", source_kind="uniparse",
        section_path="Supplement", block_type="table_row",
        text="Table 2 | eGFP | sequence: ATGGTG", table_number="Table 2",
        char_end=10, parser="uniparse-1.1.0", parser_confidence=0.9,
    )
    assert block_body_text(block) == "eGFP | sequence: ATGGTG"
    assert "2" not in numeric_claims(block_body_text(block))


def test_only_sequence_length_runs_count_as_long_tokens():
    tokens = long_token_claims("eGFP ATGGTGAGCAAGGGCGAGGAGCTGTTCACC and Hepatocyte")
    assert tokens == ("ATGGTGAGCAAGGGCGAGGAGCTGTTCACC",)


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #


def test_every_claim_corroborated_is_verified():
    verdict = _verdict(["1.01", "0.38"], [], [_read("ocr", {"1.01", "0.38", "0.27"})])
    assert verdict.status == "verified"
    assert verdict.uncorroborated == []
    assert verdict.corroborating == ["ocr"]


def test_a_hesitant_reader_cannot_condemn_the_parse():
    """The GP-006 ``0.91`` case, which is the whole reason for the asymmetry.

    Vision read that cell as 0.92 at confidence 0.5 and was wrong. If a reader
    that admits it could not resolve part of the region were allowed to
    contradict, this layer would cry wolf on a correctly transcribed table and
    everyone would learn to ignore it.
    """
    verdict = _verdict(["0.91"], [], [_read("ocr", {"0.92"}, decisive=False)])
    assert verdict.status == "unverified"
    assert verdict.uncorroborated == ["0.91"]
    assert verdict.contradicting == []


def test_a_confident_reader_that_never_saw_the_number_does_contradict():
    verdict = _verdict(["0.91"], [], [_read("ocr", {"0.92"}, decisive=True)])
    assert verdict.status == "contradicted"
    assert verdict.contradicting == ["ocr"]


def test_no_reader_at_all_is_never_reported_as_verified():
    """Silence is not corroboration. This is the failure mode section 十 names."""
    verdict = _verdict(
        ["1.01"], [], [IndependentRead(method="ocr", available=False)]
    )
    assert verdict.status == "unverified"


def test_a_sequence_absent_from_the_pages_own_text_layer_is_contradicted():
    """GP-004: the file itself disagrees with the parse, so there is no doubt."""
    verdict = _verdict(
        [],
        ["ATGGAGGACGCCAAGAACATCAAGAAGG"],
        [_read("pdf_text_layer", set(), deterministic=True, text="ATGGTGAGCAAGGGCGAGGAGCTGTTC")],
    )
    assert verdict.status == "contradicted"
    assert verdict.fabricated == ["ATGGAGGACGCCAAGAACATCAAGAAGG"]


def test_a_sequence_broken_across_printed_lines_still_matches():
    """The text layer wraps; the parse does not wrap in the same places."""
    verdict = _verdict(
        [],
        ["ATGGTGAGCAAGGGCGAGGAGCTGTTCACC"],
        [_read(
            "pdf_text_layer", set(), deterministic=True,
            text="ATGGTGAGCAAGGGC\nGAGGAGCTGTTCACC\n",
        )],
    )
    assert verdict.status == "verified"
    assert verdict.fabricated == []


def test_ocr_noise_can_never_condemn_a_sequence():
    """A 24-plus character exact match against OCR would fail on correct parses.

    Only a reader with no recognition step -- the PDF's own text layer -- is
    allowed to rule on these, so an OCR-only region falls to human review
    instead of being accused.
    """
    verdict = _verdict(
        [],
        ["ATGGTGAGCAAGGGCGAGGAGCTGTTCACC"],
        [_read("ocr:x", set(), decisive=True, deterministic=False, text="ATGGT6AGCAA")],
    )
    assert verdict.status != "contradicted"
    assert verdict.fabricated == []


def test_the_confidence_bar_is_the_one_the_measurement_justified():
    """Vision's two wrong cells were 0.5 and 0.3; every right one was 1.0."""
    wrong = [score for text, score in VISION_LINES if "0.92" in text or "0.02" in text]
    assert wrong and all(score < HIGH_CONFIDENCE for score in wrong)
    assert HIGH_CONFIDENCE <= 1.0


# --------------------------------------------------------------------------- #
# The real GP-006 block: the acceptance case for GO-006
# --------------------------------------------------------------------------- #


@needs_mmc1
def test_the_region_that_supplies_go_006_has_no_text_layer():
    """Why this layer exists at all: nothing deterministic can read that table.

    ``page.get_text()`` returns nothing inside the recorded bbox because Table S2
    is a rendered image, so before this module the sole evidence for GO-006's
    1.01 was one VLM's transcription of a picture.
    """
    crop = render_crop(*resolve_pdf("data/raw/fulltext/oa_packages/PMC11617921/mmc1.pdf", 2), TABLE_S2_BBOX)
    assert crop.text_layer.strip() == ""
    assert crop.png[:4] == b"\x89PNG"


@needs_mmc1
def test_rendering_the_same_region_twice_gives_the_same_bytes():
    """A crop is evidence, so it has to be reproducible and citable by digest."""
    first = render_crop(*resolve_pdf("data/raw/fulltext/oa_packages/PMC11617921/mmc1.pdf", 2), TABLE_S2_BBOX)
    second = render_crop(*resolve_pdf("data/raw/fulltext/oa_packages/PMC11617921/mmc1.pdf", 2), TABLE_S2_BBOX)
    assert first.sha256 == second.sha256


@needs_mmc1
def test_go_006_table_row_verifies_against_an_independent_read_of_the_crop():
    """The measurement this whole module was built to make.

    Every number in "Table S2 | LSEC | ... total insertion frequency:
    1.01 ± 0.38 %" is recovered from a fresh crop of the bbox the block records,
    by a reader that never saw the parse.
    """
    block = _load("GP-006", GO_006_BLOCK)
    assert "1.01 ± 0.38 %" in block.text
    engine = _RecordedOcr()
    report = verify_blocks([on_readable_pdf(block)], ocr_engine=engine)

    verdict = report.blocks[0]
    assert engine.calls == 1
    assert verdict.status == "verified"
    assert "1.01" in verdict.corroborated and "0.38" in verdict.corroborated
    assert verdict.uncorroborated == []
    assert verdict.accepted and not verdict.requires_human_review


@needs_mmc1
def test_a_misread_digit_in_that_row_would_not_have_verified():
    """The check has to be able to fail, or "verified" means nothing.

    If uniparse had read 1.01 as 7.01, no reader of the crop would produce it.
    """
    block = _load("GP-006", GO_006_BLOCK).model_copy(
        update={"text": _load("GP-006", GO_006_BLOCK).text.replace("1.01", "7.01")}
    )
    verdict = verify_blocks([on_readable_pdf(block)], ocr_engine=_RecordedOcr()).blocks[0]
    assert verdict.status != "verified"
    assert "7.01" in verdict.uncorroborated
    assert verdict.requires_human_review


@needs_mmc1
def test_the_whole_table_block_is_held_back_by_the_cell_ocr_could_not_resolve():
    """Honest partial result: 0.91 is real, but this reader cannot establish it.

    The row block that carries GO-006 verifies; the whole-table block does not,
    because it also asserts the Hepatocyte 0.91 that Vision misread. Reporting
    that as "unverified" rather than "verified" is the point -- and reporting it
    as "contradicted" would have been wrong, because uniparse was right.
    """
    block = _load("GP-006", "GP-006-B-023023d8f9c02eef0936")
    verdict = verify_blocks([on_readable_pdf(block)], ocr_engine=_RecordedOcr()).blocks[0]
    assert verdict.status == "unverified"
    assert verdict.uncorroborated == ["0.91"]
    assert verdict.contradicting_methods == []


# --------------------------------------------------------------------------- #
# The second read: re-parsing the region on its own
# --------------------------------------------------------------------------- #


class _FakeReread:
    """Stands in for uniparse re-parsing the isolated crop. No network."""

    def __init__(self, html: str):
        self._html = html
        self.seen: list[Path] = []

    def parse_pdf(self, path, **kwargs) -> UniparseDocument:
        self.seen.append(Path(path))
        assert Path(path).is_file(), "the re-read must be given a real PDF to parse"
        return UniparseDocument.model_validate({
            "meta": {"version": "1.1.0"},
            "pages": [{
                "page_id": 0,
                "content": [{"type": "table", "text": self._html, "text_format": "html"}],
                "others": [],
            }],
            "images": {},
            "markdown": "",
        })


@needs_mmc1
def test_the_second_read_is_handed_the_recorded_region_and_nothing_else(tmp_path):
    """Otherwise a matching transcription proves nothing.

    If the cut PDF carried the rest of the page, the second read would be
    looking at the same context as the first and agreement would be expected
    whether or not either is right. mmc1 page 2 has text-layer captions above
    and below Table S2; none of them may survive the cut.
    """
    import pymupdf

    destination = crop_pdf_region(*resolve_pdf("data/raw/fulltext/oa_packages/PMC11617921/mmc1.pdf", 2), TABLE_S2_BBOX, tmp_path / "crop.pdf")
    with pymupdf.open(destination) as cut:
        assert cut.page_count == 1
        page = cut[0]
        assert round(page.rect.width) == round(TABLE_S2_BBOX[2] - TABLE_S2_BBOX[0]) + 8
        text = page.get_text("text")
    for stray in ("Table S1", "Figure S1", "Supplemental"):
        assert stray not in text, f"{stray!r} lies outside the bbox and must not be cut in"


@needs_mmc1
def test_a_second_read_of_the_region_alone_settles_what_ocr_could_not():
    """This is what the live service actually did for the Hepatocyte 0.91.

    OCR left the whole-table block short one number. A second uniparse parse of
    the region cut out on its own returned the same table, and the block moves
    from human review to verified.
    """
    block = _load("GP-006", "GP-006-B-023023d8f9c02eef0936")
    reread = _FakeReread(block.table_html)
    verdict = verify_blocks(
        [on_readable_pdf(block)], ocr_engine=_RecordedOcr(), reread_client=reread
    ).blocks[0]

    assert reread.seen, "the re-read client was never called"
    assert verdict.status == "verified"
    assert "uniparse_reread" in verdict.corroborating_methods
    assert verdict.uncorroborated == []


@needs_mmc1
def test_a_second_read_that_disagrees_is_a_contradiction_not_a_shrug():
    """Two independent parses of the same pixels disagreeing is a real signal."""
    block = _load("GP-006", GO_006_BLOCK)
    reread = _FakeReread("<table><tr><td>LSEC</td><td>7.01 ± 0.38 %</td></tr></table>")
    verdict = verify_blocks([on_readable_pdf(block)], ocr_engine=None, reread_client=reread).blocks[0]

    assert verdict.status == "contradicted"
    assert "uniparse_reread" in verdict.contradicting_methods


# --------------------------------------------------------------------------- #
# The real GP-004 block: a transcription that does not verify
# --------------------------------------------------------------------------- #


@needs_gp004
def test_gp004_sequence_row_is_contradicted_by_the_pages_own_text_layer():
    """The finding this layer was supposed to be able to make, on committed data.

    uniparse's transcription of the EGF row of GP-004's Supplementary Table 3
    does not appear in the text layer of the page it claims to come from. This
    is not OCR noise and not a confidence judgement: the PDF's own content
    stream says something else.
    """
    block = _find("GP-004", GP004_SEQUENCE_MARKER)
    verdict = verify_blocks([on_readable_pdf(block)]).blocks[0]
    assert verdict.status == "contradicted"
    assert verdict.fabricated_long_tokens
    assert verdict.long_tokens == len(verdict.fabricated_long_tokens)
    assert "pdf_text_layer" in verdict.contradicting_methods
    assert verdict.requires_human_review


@needs_gp004
def test_a_correct_sequence_transcription_would_have_passed_the_same_check():
    """Guards against a check that just fails everything with a long token.

    Feed the block the text the PDF actually contains and the same code path
    returns verified, so the verdict above is about this parse, not about the
    rule being impossible to satisfy.
    """
    block = _find("GP-004", GP004_SEQUENCE_MARKER)
    truth = render_crop(
        *resolve_pdf(block.source_path, block.page_number),
        (block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1),
    ).text_layer
    honest = block.model_copy(update={"text": f"EGF | sequence: {truth}"})
    verdict = verify_blocks([on_readable_pdf(honest)]).blocks[0]
    assert verdict.status == "verified"
    assert verdict.fabricated_long_tokens == []


# --------------------------------------------------------------------------- #
# Stamping and non-destruction
# --------------------------------------------------------------------------- #


@needs_mmc1
def test_verification_stamps_the_verdict_and_never_drops_a_block():
    """An unverified block keeps its text. Deleting it would trade a
    hallucination risk for a silent-omission risk, which is the failure this
    project is already fighting."""
    blocks = [
        _load("GP-006", GO_006_BLOCK),
        _load("GP-006", "GP-006-B-023023d8f9c02eef0936"),
    ]
    warnings: list[str] = []
    stamped = verify_and_stamp([on_readable_pdf(b) for b in blocks], ocr_engine=_RecordedOcr(), warnings=warnings)

    assert [row.block_id for row in stamped] == [row.block_id for row in blocks]
    assert [row.text for row in stamped] == [row.text for row in blocks]
    assert stamped[0].verification_status == "verified"
    assert stamped[1].verification_status == "unverified"
    assert stamped[0].verification_method
    assert len(warnings) == 1 and "0.91" in warnings[0]


@needs_mmc1
def test_one_crop_and_one_read_serve_a_table_and_all_of_its_rows():
    """A nine-row table must not be rendered and re-read nine times."""
    blocks = [
        _load("GP-006", GO_006_BLOCK),
        _load("GP-006", "GP-006-B-023023d8f9c02eef0936"),
        _load("GP-006", "GP-006-B-8979d7c9e13813ada1f9"),
    ]
    engine = _RecordedOcr()
    verify_blocks([on_readable_pdf(b) for b in blocks], ocr_engine=engine)
    assert engine.calls == 1


def test_a_block_that_asserts_numbers_without_geometry_is_not_waved_through():
    block = DocumentBlock(
        block_id="b", paper_id="GP-X", source_path="s.pdf", source_kind="uniparse",
        section_path="Supplement", block_type="table_row",
        text="LSEC | total insertion frequency: 1.01 ± 0.38 %",
        char_end=10, parser="uniparse-1.1.0", parser_confidence=0.9,
    )
    verdict = verify_blocks([on_readable_pdf(block)]).blocks[0]
    assert verdict.status == "no_geometry"
    assert verdict.requires_human_review


def test_pmc_xml_tables_are_out_of_scope():
    """A JATS table is a deterministic parse of marked-up cells, not a VLM read."""
    block = DocumentBlock(
        block_id="b", paper_id="GP-X", source_path="a.nxml", source_kind="pmc_xml",
        section_path="Body", block_type="table_row", text="LSEC | 1.01 ± 0.38 %",
        char_end=10, parser="pmc_xml", parser_confidence=1.0,
    )
    assert verify_blocks([on_readable_pdf(block)]).blocks == []


def test_document_block_verification_fields_are_optional():
    """Corpora written before this layer existed must keep validating."""
    block = DocumentBlock(
        block_id="b", paper_id="GP-X", source_path="s.pdf", source_kind="uniparse",
        section_path="S", block_type="table", text="t", char_end=1,
        parser="uniparse-1.1.0", parser_confidence=0.9,
    )
    assert block.verification_status is None
    # None must not mean "passed": absent from an exclude_none dump entirely.
    assert "verification_status" not in block.model_dump(exclude_none=True)


# --------------------------------------------------------------------------- #
# The flag
# --------------------------------------------------------------------------- #


def test_the_flag_is_registered_with_the_metadata_the_registry_requires():
    flag = describe_flag("uniparse_crop_verification")
    assert flag.description and flag.rationale
    assert flag.integration_points == ("src/rag/ingestion.py",)
    assert flag.status != "planned"


def test_ingestion_reads_the_flag_rather_than_hard_coding_it():
    with override(uniparse_crop_verification=True):
        assert uniparse_verification_enabled() is True
    with override(uniparse_crop_verification=False):
        assert uniparse_verification_enabled() is False


def test_the_flag_is_separate_from_uniparse_ingestion():
    """Parsing and auditing the parser are two decisions."""
    with override(uniparse_ingestion=True, uniparse_crop_verification=False):
        assert ingestion.uniparse_enabled() is True
        assert uniparse_verification_enabled() is False


# --------------------------------------------------------------------------- #
# Wiring into ingestion
# --------------------------------------------------------------------------- #


def _mmc1_client():
    from tests.test_uniparse_ingestion import mmc1_response

    class _Client:
        def parse_pdf(self, path, **kwargs):
            return UniparseDocument.model_validate(mmc1_response())

    return _Client()


@needs_mmc1
def test_pdf_blocks_stamps_every_verifiable_block_when_the_flag_is_on(tmp_path):
    with override(uniparse_ingestion=True, uniparse_crop_verification=True):
        blocks = pdf_blocks(
            "GP-006", resolve_pdf(MMC1_REL, 2)[0], client=_mmc1_client(), image_root=tmp_path,
            ocr_engine=_RecordedOcr(),
        )
    verifiable = [
        block for block in blocks
        if block.block_type in {"table", "table_row", "caption"}
    ]
    assert verifiable
    assert all(block.verification_status is not None for block in verifiable)
    row = next(block for block in blocks if "1.01 ± 0.38 %" in block.text
               and block.block_type == "table_row")
    assert row.verification_status == "verified"


@needs_mmc1
def test_pdf_blocks_leaves_no_verdict_behind_when_the_flag_is_off(tmp_path):
    """Off must mean "not checked", not "checked and fine"."""
    with override(uniparse_ingestion=True, uniparse_crop_verification=False):
        blocks = pdf_blocks(
            "GP-006", resolve_pdf(MMC1_REL, 2)[0], client=_mmc1_client(), image_root=tmp_path,
            ocr_engine=_RecordedOcr(),
        )
    assert blocks
    assert all(block.verification_status is None for block in blocks)


@needs_mmc1
def test_a_broken_verifier_never_degrades_to_the_unstructured_page_dump(
    tmp_path, monkeypatch
):
    """Regression guard on where the try/except sits.

    Verification runs *outside* the uniparse try/except. If it ran inside, a
    defect in the audit layer would be indistinguishable from "uniparse is
    down", and would silently throw away a good structured parse for a
    PyMuPDF page dump -- losing GO-006 to a bug in the code meant to protect it.
    """
    def _explode(*args, **kwargs):
        raise RuntimeError("verifier defect")

    monkeypatch.setattr(ingestion, "verify_and_stamp", _explode)
    with override(uniparse_ingestion=True, uniparse_crop_verification=True):
        with pytest.raises(RuntimeError, match="verifier defect"):
            pdf_blocks(
                "GP-006", resolve_pdf(MMC1_REL, 2)[0], client=_mmc1_client(), image_root=tmp_path,
                ocr_engine=_RecordedOcr(),
            )


def test_a_contradicted_block_is_stamped_and_never_dropped():
    """The rule the module states, pinned for the verdict it matters most for.

    `test_verification_stamps_the_verdict_and_never_drops_a_block` exercises
    verified and unverified blocks only. Making verify_and_stamp drop
    contradicted blocks passed the entire suite -- the one verdict where
    dropping is tempting was the one nothing checked. Dropping an unverified
    block trades a hallucination risk for the silent-omission risk this project
    is already fighting, which is worse.
    """
    from src.rag.uniparse_verification import load_corpus, verify_and_stamp

    blocks = [on_readable_pdf(block) for block in load_corpus("GP-004")]
    stamped = verify_and_stamp(blocks)

    assert len(stamped) == len(blocks), "verification changed the block count"
    contradicted = [
        block for block in stamped
        if getattr(block, "verification_status", None) == "contradicted"
    ]
    assert contradicted, "GP-004 should still carry contradicted blocks"
    ids = {block.block_id for block in stamped}
    assert {block.block_id for block in contradicted} <= ids


def test_the_long_token_boundary_is_pinned():
    """24 characters is what caught GP-004; nothing asserted it.

    A run of 24+ alphanumerics is a sequence, an accession or an identifier --
    something a parser must transcribe exactly rather than paraphrase. Moving
    the boundary to 20 would sweep in ordinary long words, and to 28 would let
    a 24-mer through. Neither change failed a test before this one.
    """
    from src.rag.uniparse_verification import LONG_TOKEN_CHARS, long_token_claims

    assert LONG_TOKEN_CHARS == 24

    just_under = "A" * (LONG_TOKEN_CHARS - 1)
    exactly = "ATGGAGGACGCCAAGAACATCAAG"
    assert len(exactly) == LONG_TOKEN_CHARS
    assert exactly in long_token_claims(f"sequence {exactly} end")
    assert just_under not in long_token_claims(f"word {just_under} end")
