"""What a cell line IS, resolved before extraction and grounded in the paper.

The capability is a pre-pass: it reads a paper's methods, cell-culture text and
reagent/resource tables, and emits a table saying, for each named cell line, the
population it represents and the state it was in. That table then travels with
the evidence packet into extraction.

The reason it is worth having is that this fact is frequently a *join* -- a
reagent-table row on one page and a procurement or culture sentence on another,
tied together by a supplier and a species -- and a single reading pass does not
put the two side by side.

The reason it is dangerous is that a wrong "line -> population" mapping
contaminates every downstream record naming that line and is harder to see than
an omission, because it looks like structured data. So the tests below are mostly
about refusal, and they are the point of the file:

* a mapping that cites nothing is not written;
* a mapping citing an id the packet does not contain is not written;
* a field asserting a word its own cited text does not contain is dropped;
* a mapping left asserting nothing is discarded;
* the table is re-checked against the packet it is about to travel with, because
  the pre-pass may have read a different evidence view;
* with the flag off, every shipped prompt is byte-identical and no table travels.

And the generality tests: nothing here may name a cell line, a cell type or a
gold claim. Those are checked against data -- the line names the nine packets
actually contain, and the gold set's own rare endpoint vocabulary -- rather than
against a hand-written list, so they cannot be satisfied by renaming.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from src.config_flags import describe_flag, is_enabled, override
from src.extraction.compact_prompt_v1 import (
    CANDIDATE_SLOT_EXTRACTION_PROMPT,
    CANDIDATE_SLOT_PROMPT_VERSION,
    CELL_IDENTITY_RULE,
    COMPACT_EXTRACTION_PROMPT,
    ENTITY_TABLE_PROMPT_VERSION,
    ENTITY_TABLE_RULE,
    ENTITY_TABLE_SLOT_PROMPT_VERSION,
    PROMPT_VERSION,
    active_prompt,
)
from src.extraction.entity_resolution import (
    CellLineIdentity,
    EntityResolutionResponse,
    GroundedEntityTable,
    entity_table_payload,
    ground_entity_table,
    has_line_shaped_mention,
    identity_evidence_score,
    restrict_to_packet,
    select_identity_evidence,
    terms,
)
from src.extraction.evaluate_final_gold_dynamic import DISTINCTIVE_TERMS, _tokens
from src.extraction.run_codex_one_call import (
    PACKET_ROOT_BY_VIEW,
    build_prompt,
    load_entity_table,
    load_packet,
)
from src.extraction.run_entity_prepass import (
    PREPASS_PROMPT,
    build_prepass_prompt,
)
from src.rag.compact_api_packet import ApiEvidence, ApiSource, CompactApiPacket

ROOT = Path(__file__).resolve().parents[1]
FLAG = "entity_resolution_prepass"
GOLD = ROOT / "data/annotations/gold_v1"
PAPER_IDS = [f"GP-{index:03d}" for index in range(1, 10)]
PREPASS_SOURCES = (
    ROOT / "src/extraction/entity_resolution.py",
    ROOT / "src/extraction/run_entity_prepass.py",
)


# --------------------------------------------------------------------------- #
# Fixtures: a packet small enough to reason about, shaped like a real one.
# --------------------------------------------------------------------------- #


def _source(source_id: str, block_type: str, section: str, subsection=None):
    return ApiSource(
        source_id=source_id,
        chunk_id=f"chunk-{source_id}",
        source_path="data/raw/fulltext/example.pdf",
        source_kind="uniparse",
        block_type=block_type,
        section=section,
        subsection=subsection,
        page_number=1,
    )


def _evidence(evidence_id: str, text: str, source_ids: list[str]):
    return ApiEvidence(
        evidence_id=evidence_id,
        text=text,
        retrieval_field_tags=["full_corpus"],
        source_ids=source_ids,
    )


def _packet(paper_id: str, evidence: list[ApiEvidence], sources: list[ApiSource]):
    return CompactApiPacket(
        paper_id=paper_id,
        blocked_fields=[],
        sources=sources,
        evidence=evidence,
        packet_checksum="0" * 64,
    )


# A two-place join in the abstract: a catalogue row that names a line, a supplier
# and an organism, and a sentence elsewhere saying what that supplier provided.
# The vocabulary is invented on purpose -- no line, supplier, organism or cell
# type here occurs in any paper in this repository.
ROW_ID = "GP-000-E-row"
SENTENCE_ID = "GP-000-E-sentence"
CULTURE_ID = "GP-000-E-culture"
JOIN_PACKET = _packet(
    "GP-000",
    [
        _evidence(
            ROW_ID,
            "Experimental Models ; QQ-7 cells (r.fictus) | Vermillion Bioworks | VB4410",
            ["S-table"],
        ),
        _evidence(
            SENTENCE_ID,
            "Cell Culture. The rodent zamboni-cell lines were purchased from "
            "Vermillion Bioworks; the primate lines came from the Riverside Cell Bank.",
            ["S-methods"],
        ),
        _evidence(
            CULTURE_ID,
            "QQ-7 cells were differentiated with kappatropin before use.",
            ["S-methods"],
        ),
    ],
    [
        _source("S-table", "table_row", "Supplement", "Reagents and Tools Table"),
        _source("S-methods", "paragraph", "Materials and Methods", "Cell Culture"),
    ],
)


def _response(**kwargs):
    base = {
        "line_name": "QQ-7",
        "population": None,
        "state": None,
        "species": None,
        "evidence_ids": [ROW_ID, SENTENCE_ID],
        "derivation": "The row and the sentence share the supplier.",
    }
    base.update(kwargs)
    return EntityResolutionResponse(
        paper_id="GP-000", cell_lines=[CellLineIdentity(**base)]
    )


# --------------------------------------------------------------------------- #
# The flag
# --------------------------------------------------------------------------- #


def test_the_flag_is_registered_off_and_declares_the_files_that_read_it():
    flag = describe_flag(FLAG)
    assert flag.default is False
    assert flag.status == "available"
    assert flag.description
    assert flag.rationale
    assert set(flag.integration_points) == {
        "src/extraction/compact_prompt_v1.py",
        "src/extraction/run_codex_one_call.py",
    }
    assert not is_enabled(FLAG)
    for alias in ("entity_prepass", "paper_entity_resolution"):
        assert describe_flag(alias).name == FLAG


# --------------------------------------------------------------------------- #
# Prompt versioning: the flag-off request must stay byte-identical
# --------------------------------------------------------------------------- #


def test_flag_off_leaves_every_shipped_prompt_byte_identical():
    """Including the two amendments that already existed.

    A new appender that reordered or rewrote the earlier ones would change the
    prompt cache key of runs that have nothing to do with this capability.
    """
    with override(entity_resolution_prepass=False):
        assert active_prompt(False).text == COMPACT_EXTRACTION_PROMPT
        assert active_prompt(True).text == CANDIDATE_SLOT_EXTRACTION_PROMPT
        assert active_prompt(False).version == PROMPT_VERSION
        assert active_prompt(True).version == CANDIDATE_SLOT_PROMPT_VERSION
        assert (
            active_prompt(False, cell_line_identity=True).text
            == COMPACT_EXTRACTION_PROMPT + CELL_IDENTITY_RULE
        )


def test_flag_on_appends_the_rule_and_versions_it_separately():
    baseline = COMPACT_EXTRACTION_PROMPT
    slots = CANDIDATE_SLOT_EXTRACTION_PROMPT
    with override(entity_resolution_prepass=True):
        without_slots = active_prompt(False)
        with_slots = active_prompt(True)
    assert COMPACT_EXTRACTION_PROMPT == baseline
    assert CANDIDATE_SLOT_EXTRACTION_PROMPT == slots
    assert without_slots.text == baseline + ENTITY_TABLE_RULE
    assert with_slots.text == slots + ENTITY_TABLE_RULE
    assert without_slots.version == ENTITY_TABLE_PROMPT_VERSION == "compact-prompt-1.5.0"
    assert with_slots.version == ENTITY_TABLE_SLOT_PROMPT_VERSION == "compact-prompt-1.5.1"
    for selection in (without_slots, with_slots):
        assert selection.checksum == hashlib.sha256(
            selection.text.encode("utf-8")
        ).hexdigest()
    assert {without_slots.version, with_slots.version}.isdisjoint(
        {PROMPT_VERSION, CANDIDATE_SLOT_PROMPT_VERSION}
    )


# --------------------------------------------------------------------------- #
# Grounding: what is refused
# --------------------------------------------------------------------------- #


def test_a_mapping_that_cites_nothing_is_not_written():
    table, report = ground_entity_table(
        _response(population="zamboni-cell", evidence_ids=[]), JOIN_PACKET
    )
    assert table.cell_lines == []
    assert report.mappings[0].reason == "no_evidence_cited"


def test_a_mapping_citing_an_id_the_packet_does_not_contain_is_not_written():
    """The whole mapping, not just the bad id.

    The reasoning that produced it ran over text this packet cannot show, so
    nothing it concluded can be checked -- including the parts that happen to
    cite ids that do exist.
    """
    table, report = ground_entity_table(
        _response(
            population="zamboni-cell",
            evidence_ids=[ROW_ID, "GP-000-E-does-not-exist"],
        ),
        JOIN_PACKET,
    )
    assert table.cell_lines == []
    assert report.mappings[0].reason == "cited_evidence_id_not_in_packet"
    assert report.mappings[0].unknown_evidence_ids == ["GP-000-E-does-not-exist"]


def test_a_population_its_cited_text_does_not_contain_is_dropped():
    """This is the test that stops the model answering from what it knows.

    A real line has a real answer, and a model that recognises the name can
    supply it without reading anything. The only defence is that the answer has
    to point at a sentence, and a sentence that does not contain it is not one.
    """
    table, report = ground_entity_table(
        _response(population="cardiac fibroblast"), JOIN_PACKET
    )
    assert table.cell_lines == []
    assert report.mappings[0].reason == "nothing_grounded_beyond_the_line_name"
    dropped = report.mappings[0].dropped_fields[0]
    assert dropped.field == "population"
    assert set(dropped.missing_terms) == {"cardiac", "fibroblast"}


def test_a_two_place_join_is_grounded_only_when_both_places_are_cited():
    """The join is the whole reason this pass exists, so both halves are pinned.

    Citing the row alone cannot support a population that only the sentence
    states; citing both can. The check is over the union of the cited texts,
    which is exactly what makes a legitimate cross-sentence inference legal and
    an unsupported one impossible.
    """
    only_row = _response(population="rodent zamboni-cell", evidence_ids=[ROW_ID])
    table, report = ground_entity_table(only_row, JOIN_PACKET)
    assert table.cell_lines == []
    # "cell" is grounded by the row's own "cells"; the two words that make the
    # answer an answer are not.
    assert set(report.mappings[0].dropped_fields[0].missing_terms) == {
        "rodent",
        "zamboni",
    }

    both = _response(
        population="rodent zamboni-cell line",
        species="r.fictus",
        evidence_ids=[ROW_ID, SENTENCE_ID],
    )
    table, report = ground_entity_table(both, JOIN_PACKET)
    assert [row.population for row in table.cell_lines] == ["rodent zamboni-cell line"]
    assert [row.species for row in table.cell_lines] == ["r.fictus"]
    assert table.cell_lines[0].evidence_ids == [ROW_ID, SENTENCE_ID]
    assert report.kept == 1


def test_one_unsupported_field_is_dropped_without_discarding_a_supported_one():
    table, report = ground_entity_table(
        _response(
            population="rodent zamboni-cell",
            state="senescent",
            evidence_ids=[ROW_ID, SENTENCE_ID],
        ),
        JOIN_PACKET,
    )
    assert [row.population for row in table.cell_lines] == ["rodent zamboni-cell"]
    assert table.cell_lines[0].state is None
    assert [drop.field for drop in report.mappings[0].dropped_fields] == ["state"]
    assert report.mappings[0].kept is True


def test_a_state_the_culture_sentence_does_state_survives():
    table, _ = ground_entity_table(
        _response(
            population="rodent zamboni-cell",
            state="differentiated with kappatropin",
            evidence_ids=[ROW_ID, SENTENCE_ID, CULTURE_ID],
        ),
        JOIN_PACKET,
    )
    assert table.cell_lines[0].state == "differentiated with kappatropin"


def test_a_line_name_the_cited_text_does_not_contain_is_not_written():
    table, report = ground_entity_table(
        _response(line_name="ZZ-9", population="rodent zamboni-cell"), JOIN_PACKET
    )
    assert table.cell_lines == []
    assert report.mappings[0].reason == "line_name_not_in_cited_evidence"


def test_a_mapping_left_saying_nothing_about_the_line_is_discarded():
    """A bare line name is what the record already had."""
    table, report = ground_entity_table(_response(), JOIN_PACKET)
    assert table.cell_lines == []
    assert report.proposed == 1 and report.kept == 0
    assert report.mappings[0].reason == "nothing_grounded_beyond_the_line_name"


def test_plural_is_the_only_morphology_allowed():
    """Singular/plural is grammar; anything wider could launder a claim.

    "the rodent zamboni-cell lines were purchased" has to be able to support an
    answer about one line, and must not be able to support a different noun.
    """
    both = [ROW_ID, SENTENCE_ID]
    table, _ = ground_entity_table(
        _response(population="rodent zamboni-cell line", evidence_ids=both),
        JOIN_PACKET,
    )
    assert table.cell_lines[0].population == "rodent zamboni-cell line"
    table, _ = ground_entity_table(
        _response(population="rodent zamboni-cellular", evidence_ids=both), JOIN_PACKET
    )
    assert table.cell_lines == []


def test_grounding_ignores_case_and_unicode_form_but_not_content():
    """NFKC and casefold, the same normalisation the packet's own ids use."""
    assert terms("Rodent Zamboni-Cell") == terms("rodent zamboni cell")
    table, _ = ground_entity_table(
        _response(population="RODENT ZAMBONI-cell", evidence_ids=[ROW_ID, SENTENCE_ID]),
        JOIN_PACKET,
    )
    assert table.cell_lines[0].population == "RODENT ZAMBONI-cell"


# --------------------------------------------------------------------------- #
# Travelling with the packet
# --------------------------------------------------------------------------- #


def test_the_table_is_rechecked_against_the_packet_it_travels_with():
    """The pre-pass and the extraction may read different evidence views.

    A citation the extracting model cannot look up is a citation it must not be
    given, so the re-check happens at attach time and not only at write time.
    """
    table = GroundedEntityTable(
        paper_id="GP-000",
        cell_lines=[
            CellLineIdentity(
                line_name="QQ-7",
                population="rodent zamboni-cell",
                evidence_ids=[ROW_ID, SENTENCE_ID],
            )
        ],
    )
    assert restrict_to_packet(table, JOIN_PACKET).cell_lines
    narrower = _packet("GP-000", [JOIN_PACKET.evidence[0]], JOIN_PACKET.sources)
    assert restrict_to_packet(table, narrower).cell_lines == []


def test_no_entity_table_reaches_the_prompt_while_the_flag_is_off(tmp_path):
    run_dir = tmp_path / "GP-000"
    run_dir.mkdir()
    (run_dir / "entity_table.json").write_text(
        GroundedEntityTable(
            paper_id="GP-000",
            cell_lines=[
                CellLineIdentity(
                    line_name="QQ-7",
                    population="rodent zamboni-cell",
                    evidence_ids=[ROW_ID, SENTENCE_ID],
                )
            ],
        ).model_dump_json(),
        encoding="utf-8",
    )
    with override(entity_resolution_prepass=False):
        assert load_entity_table("GP-000", JOIN_PACKET, tmp_path) is None
    with override(entity_resolution_prepass=True):
        payload = load_entity_table("GP-000", JOIN_PACKET, tmp_path)
    assert payload is not None
    assert payload["cell_lines"][0]["line_name"] == "QQ-7"

    without = build_prompt(JOIN_PACKET, None, None)
    with_table = build_prompt(JOIN_PACKET, None, payload)
    assert "ENTITY TABLE" not in without
    assert "ENTITY TABLE" in with_table
    assert with_table.startswith(without)


def test_a_paper_with_no_table_sends_the_request_it_would_have_sent(tmp_path):
    with override(entity_resolution_prepass=True):
        assert load_entity_table("GP-000", JOIN_PACKET, tmp_path) is None
        assert load_entity_table("GP-000", JOIN_PACKET, None) is None


def test_the_payload_carries_the_citations_forward():
    """A record is asked to cite the table's ids, so the table has to carry them."""
    payload = entity_table_payload(
        GroundedEntityTable(
            paper_id="GP-000",
            cell_lines=[
                CellLineIdentity(
                    line_name="QQ-7",
                    population="rodent zamboni-cell",
                    evidence_ids=[ROW_ID, SENTENCE_ID],
                )
            ],
        )
    )
    assert payload["cell_lines"][0]["evidence_ids"] == [ROW_ID, SENTENCE_ID]


# --------------------------------------------------------------------------- #
# Selection: what the pre-pass reads, on the real corpus
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("paper_id", PAPER_IDS)
def test_the_selector_finds_methods_and_reagent_evidence_in_every_paper(paper_id):
    """A pass that only works on the paper it was designed against is not one."""
    packet = load_packet(paper_id, PACKET_ROOT_BY_VIEW["full"])
    sources = {row.source_id: row for row in packet.sources}
    selected = select_identity_evidence(packet)
    assert selected, f"{paper_id}: nothing selected"
    assert len(selected) < len(packet.evidence), f"{paper_id}: selected the whole packet"
    reasons = {
        reason
        for item in selected
        for reason in identity_evidence_score(item, sources)[1]
    }
    assert "names_line_shaped_entity" in reasons
    assert "provenance_or_culture_vocabulary" in reasons


def test_selection_is_deterministic():
    packet = load_packet("GP-004", PACKET_ROOT_BY_VIEW["full"])
    first = [row.evidence_id for row in select_identity_evidence(packet)]
    second = [row.evidence_id for row in select_identity_evidence(packet)]
    assert first == second


def test_a_line_shaped_mention_needs_a_cell_word_near_it():
    """Otherwise every figure number and catalogue code is a cell line."""
    assert has_line_shaped_mention("QQ-7 cells were seeded at low density.")
    assert has_line_shaped_mention("the clone AB12 was maintained in DMEM")
    assert not has_line_shaped_mention("Shown in Fig. 3B and Table S2 of the report.")
    assert not has_line_shaped_mention("Antibody AF3219 was diluted 1:200 overnight.")


def test_the_prepass_prompt_carries_the_selected_ids_and_nothing_else():
    packet = load_packet("GP-003", PACKET_ROOT_BY_VIEW["full"])
    selected = select_identity_evidence(packet)
    prompt = build_prepass_prompt(
        "GP-003",
        [{"evidence_id": row.evidence_id, "text": row.text} for row in selected],
    )
    chosen = {row.evidence_id for row in selected}
    for item in packet.evidence:
        if item.evidence_id in chosen:
            assert item.evidence_id in prompt
        else:
            assert item.evidence_id not in prompt


# --------------------------------------------------------------------------- #
# Generality: nothing here may name an entity or a gold claim
# --------------------------------------------------------------------------- #


def _gold_rows() -> list[dict[str, str]]:
    with (GOLD / "outcomes.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _rare_gold_endpoint_tokens() -> set[str]:
    """The evaluator's own definition of a distinguishing endpoint word.

    `evaluate_final_gold_dynamic` counts each endpoint token across the gold set
    and treats frequency <= 2 as rare; a rare token is what its endpoint gate
    keys on. Two-letter tokens are excluded because they are connectives ("or")
    that occur in any Python source and name nothing.
    """
    frequency = Counter(
        token for row in _gold_rows() for token in _tokens(row["endpoint_name"])
    )
    return {
        token
        for token, count in frequency.items()
        if count <= 2 and len(token) > 2
    }


def _packet_line_names() -> set[str]:
    """Every cell-line-shaped mention the nine packets contain.

    Derived from the corpus rather than written down, so the test cannot be
    satisfied by hard-coding a name it does not happen to list.
    """
    import re

    from src.extraction.entity_resolution import _CELL_WORD, _LINE_SHAPED, _LINE_WINDOW

    names: set[str] = set()
    for paper_id in PAPER_IDS:
        packet = load_packet(paper_id, PACKET_ROOT_BY_VIEW["full"])
        for item in packet.evidence:
            for match in _LINE_SHAPED.finditer(item.text):
                window = item.text[
                    max(0, match.start() - _LINE_WINDOW) : match.end() + _LINE_WINDOW
                ]
                if _CELL_WORD.search(window):
                    names.add(match.group(0))
    # Bare digits and one-letter stems are not names.
    return {name for name in names if re.search(r"[A-Za-z]{2}", name)}


@pytest.fixture(scope="module")
def prepass_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in PREPASS_SOURCES)


def test_the_prepass_names_no_cell_line_the_corpus_contains(prepass_text):
    """The veto: no entity from the papers may be written into the pass itself."""
    import re

    lowered = prepass_text.lower()
    named = sorted(
        name
        for name in _packet_line_names()
        if re.search(rf"(?<![\w-]){re.escape(name.lower())}(?![\w-])", lowered)
    )
    assert not named, f"the pre-pass hard-codes cell lines: {named}"


def test_the_prepass_names_no_distinguishing_gold_endpoint(prepass_text):
    leaked = _tokens(prepass_text) & _rare_gold_endpoint_tokens()
    assert not leaked, f"the pre-pass names the answer key: {sorted(leaked)}"


def test_the_prompts_seed_no_distinctive_claim_vocabulary():
    """DISTINCTIVE_TERMS is the matcher's own list of claim-specific words.

    Subtracting the frozen shipped prompt is what the interpretive amendment's
    test does too: a word the baseline already puts in front of the model is not
    introduced by an amendment.
    """
    already = _tokens(CANDIDATE_SLOT_EXTRACTION_PROMPT)
    for text in (ENTITY_TABLE_RULE, PREPASS_PROMPT):
        assert not ((_tokens(text) - already) & DISTINCTIVE_TERMS)


def test_no_gold_claim_is_reconstructible_from_either_prompt():
    """At most one ordinary word per gold claim, so no claim is recoverable."""
    already = _tokens(CANDIDATE_SLOT_EXTRACTION_PROMPT)
    for text in (ENTITY_TABLE_RULE, PREPASS_PROMPT):
        introduced = _tokens(text) - already
        for row in _gold_rows():
            shared = _tokens(row["qualitative_outcome"]) & introduced
            assert len(shared) <= 1, (
                f"{row['gold_outcome_id']} shares {sorted(shared)} with a prompt"
            )


def test_the_prepass_prompt_forbids_outside_knowledge_explicitly():
    lowered = PREPASS_PROMPT.lower()
    assert "no outside knowledge" in lowered
    assert "null" in lowered
    assert "only ids from the list" in lowered


# --------------------------------------------------------------------------- #
# The committed artifacts
# --------------------------------------------------------------------------- #

TABLE_ROOT = ROOT / "data/staging/extraction/entity_tables_v1"


@pytest.mark.parametrize("paper_id", PAPER_IDS)
def test_every_committed_mapping_cites_ids_its_packet_really_contains(paper_id):
    """The discipline, checked against what was actually produced and shipped.

    Not a re-test of `ground_entity_table`: this reads the committed table for
    each paper and re-derives the check from the packet on disk, so a table
    written by an older or a hand-edited path cannot pass.
    """
    path = TABLE_ROOT / paper_id / "entity_table.json"
    if not path.exists():
        pytest.skip(f"no committed entity table for {paper_id}")
    table = GroundedEntityTable.model_validate_json(path.read_text(encoding="utf-8"))
    packet = load_packet(paper_id, PACKET_ROOT_BY_VIEW["full"])
    known = {item.evidence_id for item in packet.evidence}
    for mapping in table.cell_lines:
        assert mapping.evidence_ids, f"{paper_id}/{mapping.line_name} cites nothing"
        assert set(mapping.evidence_ids) <= known, (
            f"{paper_id}/{mapping.line_name} cites ids not in the packet"
        )
    # Re-derive the grounding from the packet on disk rather than trusting the
    # file: feeding the committed table back through the checker must be a
    # fixed point. A hand-edited or stale table is not.
    regrounded, report = ground_entity_table(
        EntityResolutionResponse(paper_id=paper_id, cell_lines=table.cell_lines),
        packet,
    )
    assert regrounded.cell_lines == table.cell_lines, (
        f"{paper_id}: the committed table does not survive its own check: "
        f"{[row.model_dump() for row in report.mappings if not row.kept]}"
    )


def test_the_committed_grounding_reports_record_what_was_refused():
    """A pass that never refuses anything is not being checked."""
    if not TABLE_ROOT.exists():
        pytest.skip("the pre-pass has not been run")
    reports = sorted(TABLE_ROOT.glob("GP-*/grounding_report.json"))
    assert reports, "no grounding report was written"
    refusals = 0
    for path in reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["proposed"] >= payload["kept"]
        for mapping in payload["mappings"]:
            if not mapping["kept"]:
                refusals += 1
                assert mapping["reason"], f"{path}: a refusal with no reason"
    assert refusals, "nothing was ever refused; the check is not binding"


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #

OVERLAY_RUN = ROOT / "data/staging/extraction/codex_entity_prepass_gp008_v1"
MEASURED_ROOT = ROOT / "data/staging/extraction/codex_union_vision_v3_entity_prepass"
BASE_ROOT = ROOT / "data/staging/extraction/codex_union_vision_v3"

needs_measurement = pytest.mark.skipif(
    not (OVERLAY_RUN.exists() and BASE_ROOT.exists()),
    reason="the measured overlay run or its base is not present",
)


@needs_measurement
def test_the_table_actually_travelled_into_the_extraction_request():
    """Otherwise the measurement is of a run that never saw the capability."""
    request = json.loads(
        (OVERLAY_RUN / "GP-008/request.json").read_text(encoding="utf-8")
    )
    assert request["entity_resolution_prepass"] is True
    assert request["prompt_version"] == ENTITY_TABLE_SLOT_PROMPT_VERSION
    table = GroundedEntityTable.model_validate_json(
        (TABLE_ROOT / "GP-008/entity_table.json").read_text(encoding="utf-8")
    )
    assert request["entity_table"] == entity_table_payload(table)


@needs_measurement
def test_the_overlay_root_is_rebuildable_and_scores_what_was_recorded(tmp_path):
    """The measured numbers, and the root they came from, reproduce from data.

    Pinned because they are the whole answer: the capability produced a grounded
    table, and the gold row it was aimed at is still missed. A rebuild that
    quietly scored differently would make that statement unverifiable.
    """
    from src.extraction.build_union_entity_prepass import build
    from src.extraction.evaluate_final_gold_dynamic import evaluate

    manifest = build(output_root=tmp_path / "overlay")
    assert manifest["outcomes"]["merged"] == (
        manifest["outcomes"]["base"] + manifest["outcomes"]["overlay"]
    )
    with override(vision_relationship_polarity=True):
        rebuilt = evaluate(result_roots=[tmp_path / "overlay"])
        base = evaluate(result_roots=[BASE_ROOT])

    assert base["recovered"] == rebuilt["recovered"] == 13
    assert rebuilt["missing_gold_outcome_ids"] == ["GO-017", "GO-018"]
    assert round(base["precision"], 6) == 0.220339
    assert round(rebuilt["precision"], 6) == 0.209677
    assert base["evidence_accuracy"]["supported"] == 7
    assert rebuilt["evidence_accuracy"]["supported"] == 8
    assert base["evidence_accuracy"]["checked"] == 12
    assert rebuilt["evidence_accuracy"]["checked"] == 12

    if MEASURED_ROOT.exists():
        committed = evaluate(result_roots=[MEASURED_ROOT])
        assert round(committed["precision"], 9) == round(rebuilt["precision"], 9)


@needs_measurement
def test_exactly_one_matched_row_changed_assignment_and_it_improved():
    """The precision cost and the one reassignment, stated rather than implied."""
    from src.extraction.evaluate_final_gold_dynamic import evaluate

    with override(vision_relationship_polarity=True):
        before = {
            row["gold_outcome_id"]: row
            for row in evaluate(result_roots=[BASE_ROOT])["results"]
        }
        after = {
            row["gold_outcome_id"]: row
            for row in evaluate(result_roots=[MEASURED_ROOT])["results"]
        }
    moved = [
        gold_id
        for gold_id, row in after.items()
        if (row.get("match") or {}).get("outcome_index")
        != (before[gold_id].get("match") or {}).get("outcome_index")
    ]
    assert moved == ["GO-016"]
    assert before["GO-016"]["evidence_supported"] is False
    assert after["GO-016"]["evidence_supported"] is True
    assert (after["GO-016"]["match"]["score"]
            > before["GO-016"]["match"]["score"])


@needs_measurement
def test_the_experiment_rename_keeps_both_runs_experiments():
    """Without it the base's experiment wins and the treatment is discarded."""
    from src.extraction.build_union_entity_prepass import (
        RENAME_SUFFIX,
        rename_experiments,
    )

    overlay = json.loads(
        (OVERLAY_RUN / "GP-008/result.json").read_text(encoding="utf-8")
    )
    renamed = rename_experiments(overlay, RENAME_SUFFIX)
    original = {row["experiment_id"] for row in overlay["experiments"]}
    assert {row["experiment_id"] for row in renamed["experiments"]} == {
        f"{name}{RENAME_SUFFIX}" for name in original
    }
    assert all(
        row["experiment_id"].endswith(RENAME_SUFFIX) for row in renamed["outcomes"]
    )
    merged = json.loads(
        (MEASURED_ROOT / "GP-008/final_result.json").read_text(encoding="utf-8")
    )
    ids = {row["experiment_id"] for row in merged["experiments"]}
    assert original <= ids and {f"{n}{RENAME_SUFFIX}" for n in original} <= ids


@needs_measurement
def test_the_resolved_population_reached_a_record_but_not_the_gated_field():
    """The finding, pinned: right fact, wrong place for the matcher to see it.

    The gates read `_result_text(outcome, None)` -- the outcome's own endpoint,
    assay, comparator, unit and qualitative text -- and deliberately exclude the
    experiment's fields. The pre-pass's answer landed on the experiment. Both
    halves are asserted here because a change to either would change what this
    capability is measured to have done.
    """
    from src.extraction.evaluate_final_gold_dynamic import _result_text, _tokens

    table = GroundedEntityTable.model_validate_json(
        (TABLE_ROOT / "GP-008/entity_table.json").read_text(encoding="utf-8")
    )
    # Line names, not populations: a population's words ("human", "macrophage")
    # occur in outcome text for reasons that have nothing to do with the table,
    # so they cannot tell where the table's answer landed. A line name can.
    lines = {
        token for mapping in table.cell_lines for token in _tokens(mapping.line_name)
    }
    assert lines, "the committed table names no line"
    merged = json.loads(
        (MEASURED_ROOT / "GP-008/final_result.json").read_text(encoding="utf-8")
    )
    experiments = {row["experiment_id"]: row for row in merged["experiments"]}

    on_experiment = on_outcome = False
    for outcome in merged["outcomes"]:
        experiment = experiments.get(outcome.get("experiment_id"))
        if lines & _tokens(_result_text(outcome, experiment)):
            on_experiment = True
        if lines & _tokens(_result_text(outcome, None)):
            on_outcome = True
    assert on_experiment, "the table's answer never reached any record"
    assert not on_outcome, (
        "a line the table covers now reaches the gated field; the recorded "
        "explanation for GO-017 no longer holds and must be re-measured"
    )


def test_the_gates_read_the_outcome_alone_and_the_experiment_carries_the_population():
    """The structural fact the explanation rests on, pinned independently.

    `_score` builds its gate tokens from `_result_text(outcome, None)`. The
    experiment's target-cell field -- the field an extraction naturally puts a
    resolved population in -- reaches `actual_text` and never reaches a gate. If
    that ever changes, the recorded reason GO-017 is still missed stops being
    the reason.
    """
    from src.extraction.evaluate_final_gold_dynamic import _result_text

    outcome = {"endpoint": {"value": "an endpoint"}}
    experiment = {"therapeutic_target_cell": {"value": "a distinctive population"}}
    with_experiment = _result_text(outcome, experiment)
    outcome_only = _result_text(outcome, None)
    assert "a distinctive population" in with_experiment
    assert "a distinctive population" not in outcome_only

    source = (
        ROOT / "src/extraction/evaluate_final_gold_dynamic.py"
    ).read_text(encoding="utf-8")
    assert "outcome_only_text = _result_text(outcome, None)" in source
