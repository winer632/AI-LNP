import csv
import json

from src.extraction.evaluate_final_gold_dynamic import (
    ROOT,
    _best_one_to_one_matches,
    _evidence_ids,
    _evidence_supports,
    _evidence_texts,
    _number_in_text,
    _outcome_claim,
    evaluate,
)


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
MISSING_GOLD_OUTCOME_IDS = ["GO-002", "GO-003", "GO-006", "GO-017", "GO-018"]


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
    assert summary["recovered"] == 10
    assert summary["total"] == 15
    assert round(summary["rate"], 4) == 0.6667
    assert summary["missing_gold_outcome_ids"] == MISSING_GOLD_OUTCOME_IDS


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

