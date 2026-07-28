import csv
import json

import pytest

from src.extraction.evaluate_final_gold_dynamic import (
    DISTINCTIVE_TERMS,
    ROOT,
    _DISTINCTIVE_SURFACE,
    _best_one_to_one_matches,
    _evidence_ids,
    _evidence_supports,
    _evidence_texts,
    _number_in_text,
    _outcome_claim,
    _score,
    _tokens,
    evaluate,
)
from src.extraction.selective_vision_contracts import RELATIONSHIP_POLARITY


SUMMARY_KEYS = {
    "evaluation_version",
    # Which result roots were scored. The default single-configuration run and
    # the cross-configuration union both write this report and differ by a
    # whole outcome, so the file has to say which one it is.
    "result_roots",
    "matching",
    "recovered",
    "total",
    "rate",
    "missing_gold_outcome_ids",
    "precision",
    "false_additions",
    "evidence_accuracy",
    "unresolved_declared",
    "results",
    "paid_api_requests",
}
# GO-006 left this list when the structured evidence view was wired in: its
# value (1.01) is a Table S2 cell that the compact packet's ranker dropped, so
# no packet the pipeline sent had ever contained it. See
# tests/test_structured_evidence_view.py for the wiring itself.
MISSING_GOLD_OUTCOME_IDS = ["GO-002", "GO-003", "GO-017", "GO-018"]


def _field(value, evidence_ids):
    return {
        "value": value,
        "status": "reported" if value is not None else "not_reported",
        "evidence_ids": list(evidence_ids),
        "missing_reason": None,
    }


def _outcome(outcome_id, *, value=None, qualitative=None, evidence_ids=("E-1",)):
    return {
        "outcome_id": outcome_id,
        "experiment_id": "E1",
        "assay": _field("flow cytometry", evidence_ids),
        "endpoint": _field("percentage of eGFP-positive cells", evidence_ids),
        "comparator": _field(None, evidence_ids),
        "outcome_value": _field(value, evidence_ids),
        "outcome_unit": _field("%" if value is not None else None, evidence_ids),
        "qualitative_outcome": _field(qualitative, evidence_ids),
    }


def test_paper_level_assignment_beats_greedy_record_stealing():
    gold = [{"gold_outcome_id": "G1"}, {"gold_outcome_id": "G2"}]
    outcomes = [{"outcome_id": "O1"}, {"outcome_id": "O2"}]
    scored = {
        (0, 0): (0.90, {"label": "broad"}),
        (0, 1): (0.80, {"label": "specific-g1"}),
        (1, 0): (0.85, {"label": "specific-g2"}),
        (1, 1): (0.00, {"label": "wrong"}),
    }
    matches = _best_one_to_one_matches(gold, outcomes, scored)
    assert matches["G1"]["outcome_id"] == "O2"
    assert matches["G2"]["outcome_id"] == "O1"


def test_matches_identify_the_record_by_position_not_by_id():
    """outcome_id is not unique on a union root, so the match records both."""
    gold = [{"gold_outcome_id": "G1"}]
    outcomes = [{"outcome_id": "O1"}, {"outcome_id": "O1"}]
    scored = {
        (0, 0): (0.10, {}),
        (0, 1): (0.90, {}),
    }
    matches = _best_one_to_one_matches(gold, outcomes, scored)
    assert matches["G1"]["outcome_index"] == 1


def test_duplicate_outcome_ids_do_not_inflate_precision(tmp_path):
    """Regression: precision was computed by excluding matched ids.

    A union root merges runs that each number their own records from O1, so
    ids repeat -- codex_union_v1 holds 51 records under 32 distinct ids. When
    one O1 matched, every record called O1 was dropped from false_additions,
    so matched_records overcounted and precision came out too high.

    Under one-to-one matching precision is exactly recovered/records, which
    is the invariant this asserts.
    """
    gold_root, result_root, packet_root = _write_fixture(tmp_path)
    result_path = result_root / "GP-T01" / "final_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    # a second, unrelated record that reuses the id of the one that matches
    duplicate = _outcome("O1", qualitative="unrelated spleen accumulation")
    payload["outcomes"].append(duplicate)
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = evaluate(
        gold_root=gold_root,
        output_root=tmp_path / "out",
        result_roots=[result_root],
        packet_root=packet_root,
        task_root=tmp_path / "missing-tasks",
    )
    assert summary["recovered"] == 1
    # 4 records, 1 matched -> 3 false additions, not 2
    assert summary["false_additions"]["count"] == 3
    assert summary["precision"] == 1 / 4


def test_precision_equals_recovered_over_records_on_the_real_roots(tmp_path):
    """The same invariant, on the data the reported numbers come from."""
    for index, roots in enumerate(
        [None, [ROOT / "data/staging/extraction/codex_union_v1"]]
    ):
        if roots is not None and not roots[0].exists():
            continue
        summary = evaluate(output_root=tmp_path / str(index), result_roots=roots)
        records = summary["recovered"] + summary["false_additions"]["count"]
        assert summary["precision"] == summary["recovered"] / records


def test_evidence_check_runs_against_the_record_that_matched(tmp_path):
    """Looking the match up by id picked whichever duplicate came last."""
    gold_root, result_root, packet_root = _write_fixture(tmp_path)
    result_path = result_root / "GP-T01" / "final_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    # same id as the matching record, but a claim its evidence cannot support
    payload["outcomes"].append(_outcome("O1", value=99.9))
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = evaluate(
        gold_root=gold_root,
        output_root=tmp_path / "out",
        result_roots=[result_root],
        packet_root=packet_root,
        task_root=tmp_path / "missing-tasks",
    )
    matched = [row for row in summary["results"] if row["recovered"]]
    assert len(matched) == 1
    # the record that matched claims 41.5, which its evidence does support
    assert matched[0]["evidence_check"]["claim"] == 41.5
    assert matched[0]["evidence_supported"] is True


def test_evidence_ids_are_collected_recursively():
    outcome = _outcome("O1", value=1.0, evidence_ids=("E-1", "E-2"))
    outcome["endpoint"]["evidence_ids"] = ["E-3"]
    assert _evidence_ids(outcome) == {"E-1", "E-2", "E-3"}


def test_number_in_text_tolerates_source_formatting():
    assert _number_in_text(16.5, "gene editing rates of 16.50% +- 2.96%")
    assert _number_in_text(1661.0, "an average of 1,661 counts")
    assert not _number_in_text(16.5, "gene editing rates of 60.54%")
    assert not _number_in_text(3.3, "average of 33.0% activity")


def test_outcome_claim_prefers_reported_value_then_qualitative():
    assert _outcome_claim(_outcome("O1", value=41.5)) == ("numeric", 41.5)
    assert _outcome_claim(
        _outcome("O2", qualitative="strong eGFP staining")
    ) == ("qualitative", "strong eGFP staining")
    assert _outcome_claim(_outcome("O3"))[0] == "none"


def test_evidence_supports_numeric_claim_present_in_cited_text():
    outcome = _outcome("O1", value=41.5)
    checked, supported, detail = _evidence_supports(
        outcome,
        {"E-1": "CD11b-positive cells expressed eGFP (41.50 +- 14.6%)."},
    )
    assert (checked, supported) == (True, True)
    assert detail["claim_type"] == "numeric"
    assert detail["resolved_evidence_ids"] == ["E-1"]


def test_evidence_supports_rejects_numeric_claim_absent_from_cited_text():
    outcome = _outcome("O1", value=41.5)
    checked, supported, _ = _evidence_supports(
        outcome,
        {"E-1": "CD11b-positive cells expressed eGFP (12.4 +- 3.1%)."},
    )
    assert (checked, supported) == (True, False)


def test_evidence_supports_flags_unsupported_qualitative_terms():
    outcome = _outcome(
        "O1",
        qualitative="LNP16 was taken up by hepatocytes and Kupffer cells.",
    )
    checked, supported, detail = _evidence_supports(
        outcome,
        {"E-1": "LNP16 was taken up by hepatocytes."},
    )
    assert (checked, supported) == (True, False)
    assert detail["unsupported_terms"] == ["kupffer"]


def test_evidence_supports_accepts_fully_grounded_qualitative_claim():
    outcome = _outcome("O1", qualitative=">80% of BMDMs expressed GFP.")
    checked, supported, detail = _evidence_supports(
        outcome,
        {"E-1": "More than 80% of BMDMs expressed GFP after treatment."},
    )
    assert (checked, supported) == (True, True)
    assert detail["unsupported_terms"] == []


def test_evidence_supports_is_unchecked_without_resolvable_evidence():
    outcome = _outcome("O1", value=41.5, evidence_ids=("V-image-only",))
    checked, supported, detail = _evidence_supports(outcome, {"E-1": "text"})
    assert (checked, supported) == (False, False)
    assert detail["resolved_evidence_ids"] == []


def test_evidence_supports_is_unchecked_without_any_claim():
    checked, supported, detail = _evidence_supports(
        _outcome("O1"),
        {"E-1": "some source text"},
    )
    assert (checked, supported) == (False, False)
    assert detail["claim_type"] == "none"


def _write_fixture(tmp_path):
    gold_root = tmp_path / "gold"
    gold_root.mkdir()
    with (gold_root / "outcomes.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "gold_outcome_id",
                "gold_experiment_id",
                "endpoint_name",
                "outcome_value",
                "outcome_unit",
                "normalization_basis",
                "qualitative_outcome",
                "value_status",
                "evidence_id",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "gold_outcome_id": "GO-T1",
                "gold_experiment_id": "GE-T1",
                "endpoint_name": "percentage of eGFP-positive endothelial cells",
                "outcome_value": "41.5",
                "outcome_unit": "%",
                "normalization_basis": "",
                "qualitative_outcome": "",
                "value_status": "reported",
                "evidence_id": "GE-1",
            }
        )
    with (gold_root / "experiments.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(
            out, fieldnames=["gold_experiment_id", "gold_paper_id"]
        )
        writer.writeheader()
        writer.writerow({"gold_experiment_id": "GE-T1", "gold_paper_id": "GP-T01"})
    with (gold_root / "evidence.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["evidence_id", "evidence_text"])
        writer.writeheader()
        writer.writerow(
            {
                "evidence_id": "GE-1",
                "evidence_text": (
                    "41.5% of eGFP-positive endothelial cells were measured."
                ),
            }
        )

    result_root = tmp_path / "results"
    (result_root / "GP-T01").mkdir(parents=True)
    (result_root / "GP-T01" / "final_result.json").write_text(
        json.dumps(
            {
                "paper_id": "GP-T01",
                "experiments": [
                    {"experiment_id": "E1", "payload_name": "eGFP mRNA"}
                ],
                "outcomes": [
                    _outcome("O1", value=41.5, evidence_ids=("E-1",)),
                    _outcome(
                        "O2",
                        qualitative="unrelated spleen accumulation",
                        evidence_ids=("E-1",),
                    ),
                    _outcome(
                        "O3",
                        qualitative="unrelated kidney clearance",
                        evidence_ids=("E-1",),
                    ),
                ],
                "unresolved_items": [{"item": "a"}, {"item": "b"}],
            }
        ),
        encoding="utf-8",
    )
    (result_root / "GP-T01" / "merge_report.json").write_text(
        json.dumps({"unresolved_candidate_ids": ["OC-1"]}), encoding="utf-8"
    )

    packet_root = tmp_path / "packets"
    packet_root.mkdir()
    (packet_root / "GP-T01.json").write_text(
        json.dumps(
            {
                "paper_id": "GP-T01",
                "evidence": [
                    {
                        "evidence_id": "E-1",
                        "text": (
                            "Flow cytometry showed 41.50% of eGFP-positive "
                            "endothelial cells."
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return gold_root, result_root, packet_root


def _run_fixture(tmp_path):
    gold_root, result_root, packet_root = _write_fixture(tmp_path)
    return evaluate(
        gold_root=gold_root,
        output_root=tmp_path / "out",
        result_roots=[result_root],
        packet_root=packet_root,
        task_root=tmp_path / "missing-tasks",
    )


def test_false_additions_are_the_unmatched_result_outcomes(tmp_path):
    summary = _run_fixture(tmp_path)
    assert summary["recovered"] == 1
    assert summary["false_additions"]["count"] == 2
    assert [item["outcome_id"] for item in summary["false_additions"]["items"]] == [
        "O2",
        "O3",
    ]
    assert all(
        item["paper_id"] == "GP-T01" and item["endpoint"] and item["summary"]
        for item in summary["false_additions"]["items"]
    )


def test_precision_and_unresolved_counts_come_from_the_inputs(tmp_path):
    summary = _run_fixture(tmp_path)
    assert summary["precision"] == 1 / 3
    assert summary["unresolved_declared"] == {
        "result_items": 2,
        "merge_candidates": 1,
    }
    assert summary["evidence_accuracy"] == {
        "checked": 1,
        "supported": 1,
        "rate": 1.0,
    }
    assert summary["results"][0]["evidence_supported"] is True


def test_evidence_texts_merge_packet_and_gold_gap_task(tmp_path):
    packet_root = tmp_path / "packets"
    packet_root.mkdir()
    (packet_root / "GP-T01.json").write_text(
        json.dumps({"evidence": [{"evidence_id": "E-1", "text": "packet text"}]}),
        encoding="utf-8",
    )
    task_root = tmp_path / "tasks"
    (task_root / "GP-T01").mkdir(parents=True)
    (task_root / "GP-T01" / "task.json").write_text(
        json.dumps(
            {
                "evidence": [{"evidence_id": "GP-T01-FC-1", "text": "caption text"}],
                "visual_assets": [{"crop_evidence_id": "V-1"}],
            }
        ),
        encoding="utf-8",
    )
    texts = _evidence_texts("GP-T01", packet_root=packet_root, task_root=task_root)
    assert texts == {"E-1": "packet text", "GP-T01-FC-1": "caption text"}


def test_evaluate_emits_the_full_summary_schema(tmp_path):
    summary = evaluate(output_root=tmp_path / "out")
    assert set(summary) == SUMMARY_KEYS
    assert summary["evaluation_version"] == "final-gold-dynamic-1.2.0"
    assert summary["paid_api_requests"] == 0
    assert set(summary["false_additions"]) == {"count", "items"}
    assert set(summary["evidence_accuracy"]) == {"checked", "supported", "rate"}
    assert set(summary["unresolved_declared"]) == {
        "result_items",
        "merge_candidates",
    }
    assert all(
        isinstance(row["evidence_supported"], bool) for row in summary["results"]
    )
    written = json.loads(
        (tmp_path / "out" / "evaluation.json").read_text(encoding="utf-8")
    )
    assert set(written) == SUMMARY_KEYS


def test_recall_regression_is_unchanged_by_the_precision_metrics(tmp_path):
    summary = evaluate(output_root=tmp_path / "out")
    assert summary["recovered"] == 11
    assert summary["total"] == 15
    assert round(summary["rate"], 4) == 0.7333
    assert summary["missing_gold_outcome_ids"] == MISSING_GOLD_OUTCOME_IDS


def test_recall_is_bought_with_precision_and_the_report_says_so(tmp_path):
    """Recall alone would hide what an extra extraction stage costs.

    Matching is one-to-one, so a stage that emits more records can raise
    recall and can only lower precision. Pinning both together stops the
    headline number from moving without its price being visible: 10/15 at
    precision 0.3125 became 11/15 at precision 0.1897.

    The 0.2586 first recorded here was itself inflated. Precision counted
    matches by id-set membership, and the union carries records from several
    runs that each number their own from O1, so one match credited every
    record sharing that id. Fixed alongside; the honest cost of the extra
    stage is larger than it first appeared.
    """
    summary = evaluate(output_root=tmp_path / "out")
    assert round(summary["precision"], 4) == 0.1897
    assert summary["false_additions"]["count"] == 47
    # Precision must equal recovered/records on every root once matches are
    # counted by identity rather than by id.
    records = summary["false_additions"]["count"] + summary["recovered"]
    assert round(summary["recovered"] / records, 4) == round(summary["precision"], 4)


def test_summary_records_which_result_roots_it_scored(tmp_path):
    """The report must be reproducible from itself.

    A default run scores the four baseline roots and a union run scores one
    ensemble root, and the two differ by a whole outcome. Both write the same
    file, so a reader holding it needs it to say which it is; without that,
    re-running the default command silently replaced the union number and the
    result looked like an unreproducible claim rather than a different
    measurement.
    """
    default = evaluate(output_root=tmp_path / "a")
    assert default["result_roots"], "a report that names no source is not reproducible"
    assert all(isinstance(root, str) for root in default["result_roots"])

    union_root = ROOT / "data/staging/extraction/codex_union_v1"
    if not union_root.exists():
        return
    union = evaluate(output_root=tmp_path / "b", result_roots=[union_root])
    assert union["result_roots"] == ["data/staging/extraction/codex_union_v1"]
    assert union["result_roots"] != default["result_roots"]


# --------------------------------------------------------------------------
# What polarity_gate actually is
#
# The gate is named for polarity but is implemented as token presence: if the
# gold qualitative text contains a word from a fixed lexicon, the candidate
# record must contain that same word. Nothing in it looks at negation.
#
# These tests pin that real behaviour, including where it is unsound, because
# a proposal to replace the lexicon with polarity-preserving equivalence
# classes was measured against it and rejected. Anyone revisiting that idea
# needs the failures below to be visible rather than rediscovered: a synonym
# set layered on a negation-blind matcher widens the false-accept surface
# instead of narrowing it.
# --------------------------------------------------------------------------


def _polarity_gate(
    gold_qualitative: str,
    candidate_qualitative: str,
    *,
    relationship: str | None = None,
) -> bool:
    """Run the real scorer and return only its polarity_gate verdict."""
    gold = {
        "endpoint_name": "marker signal",
        "normalization_basis": "",
        "qualitative_outcome": gold_qualitative,
        "outcome_value": "",
        "value_status": "qualitative_reported",
    }
    outcome = _outcome("O1", qualitative=candidate_qualitative)
    outcome["endpoint"] = _field("marker signal", ["E-1"])
    outcome["assay"] = _field("microscopy", ["E-1"])
    if relationship is not None:
        outcome["vision_relationship"] = relationship
    _, detail = _score(gold, {"evidence_text": ""}, outcome, None, {})
    return detail["polarity_gate"]


def test_polarity_gate_passes_when_gold_trips_no_lexicon_term():
    """Most gold rows never reach the gate; it is inert for them."""
    assert _polarity_gate("Luciferase expression was higher.", "anything at all")


@pytest.mark.parametrize(
    "candidate",
    [
        "The signals were distinct and did not overlap.",
        "GFP was absent from LYVE-1-positive cells.",
        "GFP and LYVE-1 signals showed no overlap in any field.",
    ],
)
def test_polarity_gate_rejects_opposite_polarity_that_avoids_the_gold_word(
    candidate,
):
    """Negative controls the gate does reject.

    It rejects them for a shallow reason -- the paraphrase happens not to
    reuse the word "colocalized" -- but the verdict is the right one.
    """
    assert not _polarity_gate("GFP signal colocalized with LYVE-1.", candidate)


def test_polarity_gate_rejects_a_weak_positive_against_a_strong_one():
    """GO-002's shape: "Few ... expressed eGFP" must not match "significant"."""
    assert not _polarity_gate(
        "Few Kupffer cells expressed eGFP.",
        "A significant percentage of CD11b-positive cells expressed eGFP.",
    )


def test_polarity_gate_rejects_a_directional_antonym():
    assert not _polarity_gate(
        "Reduced mitochondrial damage.", "Damage was increased."
    )


def test_a_denial_in_another_word_form_is_rejected_only_by_the_mismatch():
    """Rejected, but not understood.

    "No colocalization was observed" fails because "colocalization" is not
    the token "colocalized", not because anything read the "No". Folding the
    two forms together used to make this a false accept, which is why that
    fold was removed. The rejection is still an accident of vocabulary.
    """
    assert not _polarity_gate(
        "GFP signal colocalized with LYVE-1.", "No colocalization was observed."
    )
    # and the accident is visible: the affirmative paraphrase fails too
    assert not _polarity_gate(
        "GFP signal colocalized with LYVE-1.", "Clear colocalization was observed."
    )


@pytest.mark.parametrize(
    ("gold", "candidate"),
    [
        (
            "GFP signal colocalized with LYVE-1.",
            "GFP was not colocalized with LYVE-1.",
        ),
        ("Reduced mitochondrial damage.", "Damage was not reduced."),
        (
            "Reporter expression localized to macrophages.",
            "Reporter expression was not localized to macrophages.",
        ),
    ],
)
def test_polarity_gate_is_blind_to_negation_of_the_gold_word(gold, candidate):
    """KNOWN UNSOUNDNESS, asserted so it cannot regress silently.

    The gate accepts a flat contradiction whenever the contradiction is
    spelled with the gold's own word. "not colocalized" tokenises to
    {not, colocalized} and the membership test only asks whether
    "colocalized" is present.

    This is the load-bearing fact against replacing the lexicon with
    equivalence classes: the gate does not reject absence claims, it
    rejects unfamiliar vocabulary. Adding synonyms to a matcher with this
    property lets in more contradictions, not fewer.
    """
    assert _polarity_gate(gold, candidate) is True


def test_polarity_gate_rejects_a_true_paraphrase_of_the_gold_claim():
    """The complement of the test above, and the reason the gate looks broken.

    Same measurement, different words, and the gate refuses it. Taken with
    the negation blindness above, the gate's real axis is vocabulary reuse
    rather than polarity agreement.
    """
    assert not _polarity_gate(
        "GFP signal colocalized with LYVE-1.",
        "GFP and LYVE-1 signals were overlapping.",
    )


def test_marker_enum_text_defeats_the_gate_through_underscore_splitting():
    """Carrying a controlled ``relationship`` enum in as free text is unsafe.

    ``selective_vision_contracts.VisualRelationship`` already distinguishes
    "colocalized" from "not_colocalized". Writing that value into a record's
    text fields would let the vision route reach this gate, but the
    tokeniser splits the underscore and the negated enum reads as the
    positive one. A structured comparison is required; a textual one inverts
    the meaning.
    """
    assert _tokens("not_colocalized") == {"not", "colocalized"}
    assert _polarity_gate(
        "GFP signal colocalized with LYVE-1.", "relationship: not_colocalized"
    )


def test_the_depluraliser_leaves_stem_final_s_alone():
    """Regression: a trailing "s" was stripped from any long token.

    Guarding "ss" alone mangled every -us/-is/-as word, which is wrong on its
    own terms and had a second effect: the lexicon entry "obvious" was
    compared against the token "obviou" and could never match.
    """
    assert "obvious" in _tokens("No obvious EGFP-positive cells")
    assert "analysis" in _tokens("the analysis was repeated")
    # real plurals still collapse
    assert "hepatocyte" in _tokens("hepatocytes were labelled")


def test_the_distinctive_lexicon_is_normalised_like_the_text():
    """Every entry must survive the tokeniser, or it is silently dead."""
    for term in _DISTINCTIVE_SURFACE:
        tokens = _tokens(term)
        assert tokens, f"{term!r} tokenises to nothing"
        assert tokens <= DISTINCTIVE_TERMS, (
            f"{term!r} normalises to {tokens}, absent from the lexicon"
        )


def test_eradicated_is_carried_without_being_exercised():
    """Recorded, not trusted: no gold row contains it."""
    gold_terms = set()
    for row in csv.DictReader(
        (ROOT / "data/annotations/gold_v1/outcomes.csv").open(encoding="utf-8")
    ):
        gold_terms |= _tokens(row["qualitative_outcome"])
    assert "eradicated" in DISTINCTIVE_TERMS
    assert "eradicated" not in gold_terms


# --------------------------------------------------------------------------
# The structured relationship route
#
# A record from the vision route declares its relation in a dedicated field
# drawn from a closed vocabulary. Reading that field is the only way the gate
# can see a denial at all, since the prose of a denial contains the affirmed
# word. These tests are the safety property for that route.
# --------------------------------------------------------------------------

COLOCALIZED_GOLD = "GFP signal colocalized with the LSEC marker LYVE-1."


def test_a_denied_relation_cannot_satisfy_the_gold_it_denies():
    """THE safety property of the structured-relationship change.

    Without it this change is a way to inflate the benchmark: the record's
    own prose contains the word "colocalized" inside "not colocalized", so
    the token test accepts it. The declared relation has to override the
    text, or a read that found the opposite of the gold claim scores as a
    match for it.
    """
    text_of_a_denial = "eGFP signal is not colocalized with F4-80-positive cells."
    # the hazard is real: the text alone passes
    assert _polarity_gate(COLOCALIZED_GOLD, text_of_a_denial) is True
    # the declaration overrides it
    assert (
        _polarity_gate(
            COLOCALIZED_GOLD, text_of_a_denial, relationship="not_colocalized"
        )
        is False
    )


def test_a_denied_relation_is_refused_however_the_prose_reads():
    for text in (
        "Signals occupy separate cells.",
        "GFP colocalized with LYVE-1 throughout the section.",
        "",
    ):
        assert not _polarity_gate(
            COLOCALIZED_GOLD, text, relationship="not_colocalized"
        )


def test_an_affirmed_relation_satisfies_the_matching_gold_term():
    """GO-003's shape: the read affirms the relation the gold row asserts."""
    assert _polarity_gate(
        COLOCALIZED_GOLD,
        "GFP signal is present in LYVE-1-positive cells.",
        relationship="colocalized",
    )


def test_an_affirmed_relation_discharges_only_its_own_term():
    """It is not a blanket pass for every distinctive term in the gold row."""
    two_term_gold = "GFP colocalized with LYVE-1 solely in hepatocytes."
    assert not _polarity_gate(
        two_term_gold, "GFP is present in LYVE-1 cells.", relationship="colocalized"
    )
    assert _polarity_gate(
        two_term_gold,
        "GFP is present in LYVE-1 cells, solely there.",
        relationship="colocalized",
    )


def test_a_relation_the_gold_row_does_not_assert_changes_nothing():
    """GO-002's shape: the gold term is "few", so a colocalisation verdict is
    not what is being asked about and must not answer for it."""
    assert not _polarity_gate(
        "Few F4/80-positive Kupffer cells expressed eGFP.",
        "A significant percentage of CD11b-positive cells expressed eGFP.",
        relationship="not_colocalized",
    )
    assert not _polarity_gate(
        "Few F4/80-positive Kupffer cells expressed eGFP.",
        "A significant percentage of CD11b-positive cells expressed eGFP.",
        relationship="colocalized",
    )


def test_records_that_declare_nothing_keep_the_unchanged_behaviour():
    """Every record from the text route declares nothing."""
    assert _polarity_gate(COLOCALIZED_GOLD, "GFP colocalized with LYVE-1.")
    assert not _polarity_gate(COLOCALIZED_GOLD, "Signals were distinct.")


LOCALIZED_GOLD = (
    "Reporter and FAPCAR expression localized to macrophages rather than "
    "Desmin-positive HSCs."
)


def test_the_enum_vocabulary_does_not_cover_localisation():
    """GO-018's term is "localized" and no enum value asserts it.

    The discharge is identity between the declared relation and the gold
    term, so a record declaring "colocalized" does not answer a gold row
    asserting "localized" -- and must not be made to. Adding that edge was
    measured and refused; see the test below for why.
    """
    assert "localized" in DISTINCTIVE_TERMS
    covered = set()
    for value, (base, _) in RELATIONSHIP_POLARITY.items():
        covered |= _tokens(base)
    assert "localized" not in covered, (
        "an enum value now claims to assert localisation; re-run the "
        "argument-structure control before relying on it"
    )
    assert not _polarity_gate(
        LOCALIZED_GOLD,
        "The panels support ZsGreen co-localization with F4/80-positive "
        "macrophages.",
        relationship="colocalized",
    )


def test_a_declared_relation_carries_no_target():
    """Why "colocalized" must not be mapped onto "localized".

    The enum is unary. It says co-presence happened, never with what, while
    a gold row asserting localisation always names a compartment. One real
    observation shows why that gap cannot be closed by a mapping: GP-008's
    GO-018 read asserts co-localisation WITH F4/80 and, in the same
    sentence, non-localisation TO Desmin. Both enum values are supported by
    its own prose, and it can declare only one.

    Under a colocalized -> localized mapping the gold row's fate would flip
    on that arbitrary choice: "colocalized" would admit it and
    "not_colocalized" would veto it, from identical evidence. Worse, a read
    that declared "not_colocalized" because the signal is absent from Desmin
    would veto a gold row asserting localisation to macrophages -- a
    compartment it never examined, and a finding that corroborates the row.

    Identity discharge has no such failure mode: neither value discharges a
    term the vocabulary cannot express.
    """
    both_relations_supported = (
        "FAPCAR co-localizes with F4/80-positive macrophages; these panels "
        "do not support FAPCAR localization to Desmin-positive hepatic "
        "stellate cells."
    )
    for declared in ("colocalized", "not_colocalized"):
        assert not _polarity_gate(
            LOCALIZED_GOLD, both_relations_supported, relationship=declared
        ), f"{declared} must not decide a claim the vocabulary cannot express"


def test_irrelevant_bulk_never_costs_a_record_score():
    """Why a vision record must not be judged on a concatenated field.

    ``lexical`` divides the overlap by ``min(len(expected), len(actual))``,
    so once a candidate is larger than the gold text the denominator pins to
    the gold side. Past that point extra tokens cannot dilute the score;
    they can only add overlap. A record that concatenates several fields
    into one competes on volume rather than on being the right measurement.

    Pinned because it is what makes merged panel reads outscore the text
    record that actually reports the measurement.
    """
    gold = {
        "endpoint_name": "FAPCAR_or_GFP_positive_BMDMs",
        "normalization_basis": "",
        "qualitative_outcome": "Over 80% of BMDMs expressed GFP or FAPCAR.",
        "outcome_value": "80",
        "value_status": "reported_threshold",
    }
    evidence = {"evidence_text": "Over 80% of BMDMs expressed GFP or FAPCAR."}

    def score(endpoint_text):
        outcome = _outcome("X", qualitative="panels support expression")
        outcome["endpoint"] = _field(endpoint_text, ["E-1"])
        outcome["assay"] = _field("microscopy", ["E-1"])
        return _score(gold, evidence, outcome, None, {})[0]

    padding = " ".join(f"zqx{index}" for index in range(60))
    lean = score("BMDMs expressing FAPCAR")
    padded = score(f"BMDMs expressing FAPCAR {padding}")
    assert padded >= lean - 0.1, "padding should not be penalised much"
    # and an overlapping token added to the padded record is pure gain
    assert score(f"BMDMs expressing FAPCAR {padding} GFP") > padded


def test_the_declared_relation_is_matched_in_normalised_space():
    """The enum base must go through the tokeniser, not be compared raw.

    Found by trying a derivational normaliser: it rewrote the lexicon to
    "colocaliz" while the enum still said "colocalized", and the structured
    route stopped firing with nothing failing. Same shape as the "obvious"
    bug. Both sides normalise, so the comparison survives that change.
    """
    base, affirmed = RELATIONSHIP_POLARITY["colocalized"]
    assert _tokens(base) & DISTINCTIVE_TERMS, (
        "the declared relation no longer meets the lexicon it must answer"
    )
    assert affirmed is True


def test_the_structured_route_is_inert_when_the_flag_is_off(monkeypatch):
    monkeypatch.setenv("AI_LNP_FLAG_VISION_RELATIONSHIP_POLARITY", "0")
    # falls back to the token test, which the denial's own prose satisfies
    assert _polarity_gate(
        COLOCALIZED_GOLD,
        "eGFP signal is not colocalized with F4-80-positive cells.",
        relationship="not_colocalized",
    )
    assert not _polarity_gate(
        COLOCALIZED_GOLD,
        "GFP signal is present in LYVE-1-positive cells.",
        relationship="colocalized",
    )

