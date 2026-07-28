"""Dynamically match frozen gold outcomes to final compact records one-to-one."""

from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path

from src.config_flags import is_enabled


ROOT = Path(__file__).resolve().parents[2]
PRECISION_METRICS_FLAG = "precision_metrics"
GOLD_ROOT = ROOT / "data/annotations/gold_v1"
OUTPUT_ROOT = ROOT / "reports/extraction/final_gold_dynamic_v1"
RESULT_ROOTS = [
    ROOT / "data/staging/extraction/consolidated_gold_gap_merged_v1",
    ROOT / "data/staging/extraction/compact_merged_v1_1",
    ROOT / "data/staging/extraction/compact_merged_v1",
    ROOT / "data/staging/extraction/compact_one_call_v1",
]
PACKET_ROOT = ROOT / "data/staging/rag/compact_api_packets_v1_1"
GOLD_GAP_TASK_ROOT = ROOT / "data/staging/extraction/consolidated_gold_gap_tasks_v1"
STOP = {
    "the", "and", "of", "in", "to", "a", "an", "was", "were", "with",
    "from", "for", "after", "outcome", "cells", "cell", "reported", "result",
    "value", "activity", "expression",
}
# Generic English function words carry no factual claim, so they are dropped
# before the literal evidence check. Polarity words ("no", "few", "not") are
# deliberately absent: they change what an outcome asserts.
FUNCTION_WORDS = {
    "at", "as", "be", "been", "both", "but", "by", "had", "has", "have",
    "into", "is", "it", "its", "on", "or", "over", "than", "that", "their",
    "them", "there", "these", "they", "this", "those", "which", "while",
    "whereas", "during", "between", "per", "via", "when", "where", "within",
}
_NUMBER = re.compile(r"[0-9][0-9,]*(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _value(value):
    return value.get("value") if isinstance(value, dict) else value


def _tokens(text: str) -> set[str]:
    normalized = (
        text.lower()
        .replace("_", " ")
        .replace("exclusively", "solely")
        .replace("colocalization", "colocalized")
    )
    found = set()
    for token in re.findall(r"[a-z0-9]+", normalized):
        if len(token) >= 5 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if len(token) >= 2 and token not in STOP:
            found.add(token)
    return found


def _result_path(paper_id: str, result_roots: list[Path] | None = None) -> Path:
    for root in result_roots if result_roots is not None else RESULT_ROOTS:
        for name in ("final_result.json", "result.json"):
            path = root / paper_id / name
            if path.exists():
                return path
    raise FileNotFoundError(paper_id)


def _evidence_ids(value) -> set[str]:
    """Collect every evidence_id referenced anywhere inside a record."""
    if isinstance(value, dict):
        found = set(value.get("evidence_ids") or [])
        for child in value.values():
            found |= _evidence_ids(child)
        return found
    if isinstance(value, list):
        found = set()
        for child in value:
            found |= _evidence_ids(child)
        return found
    return set()


def _evidence_texts(
    paper_id: str,
    *,
    packet_root: Path = PACKET_ROOT,
    task_root: Path = GOLD_GAP_TASK_ROOT,
) -> dict[str, str]:
    """Load offline evidence text for one paper.

    Two local sources are merged, no network access is performed: the compact
    API packet and, when the paper went through consolidated gold-gap
    recovery, that task's extra evidence block. Visual assets (``V-*`` crop
    ids) carry no text and are therefore absent from the mapping.
    """
    texts: dict[str, str] = {}
    for path in (
        packet_root / f"{paper_id}.json",
        task_root / paper_id / "task.json",
    ):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("evidence", []):
            evidence_id = row.get("evidence_id")
            text = row.get("text")
            if evidence_id and isinstance(text, str):
                texts[evidence_id] = text
    return texts


def _norm(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _numbers_in_text(text: str) -> list[float]:
    found = []
    for match in _NUMBER.finditer(text):
        try:
            found.append(float(match.group(0).replace(",", "")))
        except ValueError:
            continue
    return found


def _number_in_text(value: float, text: str) -> bool:
    """True when ``value`` occurs as a literal number in ``text``.

    Formatting is tolerated so that a reported ``16.5`` still matches the
    source string ``16.50%``; the numeric identity must hold exactly.
    """
    tolerance = max(1e-9, abs(value) * 1e-6)
    return any(
        abs(candidate - value) <= tolerance
        for candidate in _numbers_in_text(text)
    )


def _claim_terms(text: str) -> set[str]:
    return {token for token in _tokens(text) if token not in FUNCTION_WORDS}


def _outcome_claim(outcome: dict) -> tuple[str, object]:
    """Return the value this outcome record asserts, and how to check it.

    ``outcome_value`` is the asserted value whenever the record reports one;
    a record without a numeric or textual ``outcome_value`` asserts its
    ``qualitative_outcome`` instead.
    """
    value = _value(outcome.get("outcome_value"))
    if isinstance(value, bool):
        value = None
    if isinstance(value, (int, float)):
        return "numeric", float(value)
    if isinstance(value, str) and value.strip():
        return "text", value
    qualitative = _value(outcome.get("qualitative_outcome"))
    if isinstance(qualitative, str) and qualitative.strip():
        return "qualitative", qualitative
    return "none", None


def _evidence_supports(
    outcome: dict,
    evidence_texts: dict[str, str],
) -> tuple[bool, bool, dict]:
    """Check the record's asserted value against its own cited evidence.

    Returns ``(checked, supported, detail)``. A record is *checked* when it
    asserts something and at least one of its cited evidence ids resolves to
    local text. Numeric claims must appear as a literal number; textual and
    qualitative claims must have every content term present verbatim in the
    cited text, so a paraphrase that introduces an unsupported term fails.
    """
    cited = sorted(_evidence_ids(outcome))
    resolved = [eid for eid in cited if eid in evidence_texts]
    source = " ".join(evidence_texts[eid] for eid in resolved)
    claim_type, claim = _outcome_claim(outcome)
    detail = {
        "claim_type": claim_type,
        "claim": claim,
        "cited_evidence_ids": cited,
        "resolved_evidence_ids": resolved,
        "unsupported_terms": [],
    }
    if claim_type == "none" or not resolved:
        return False, False, detail
    if claim_type == "numeric":
        return True, _number_in_text(float(claim), source), detail
    if claim_type == "text" and _norm(claim) and _norm(claim) in _norm(source):
        return True, True, detail
    missing = sorted(_claim_terms(str(claim)) - _claim_terms(source))
    detail["unsupported_terms"] = missing
    return True, not missing, detail


def _outcome_summary(paper_id: str, outcome: dict, experiment: dict | None) -> dict:
    return {
        "paper_id": paper_id,
        "outcome_id": outcome.get("outcome_id"),
        "experiment_id": outcome.get("experiment_id"),
        "endpoint": _value(outcome.get("endpoint")),
        "outcome_value": _value(outcome.get("outcome_value")),
        "outcome_unit": _value(outcome.get("outcome_unit")),
        "summary": _result_text(outcome, experiment),
    }


def _merge_candidate_count(result_path: Path) -> int:
    report_path = result_path.parent / "merge_report.json"
    if not report_path.exists():
        return 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return len(report.get("unresolved_candidate_ids") or [])


def _result_text(outcome: dict, experiment: dict | None) -> str:
    values = [
        _value(outcome.get(field))
        for field in (
            "endpoint",
            "assay",
            "comparator",
            "outcome_unit",
            "qualitative_outcome",
        )
    ]
    if experiment:
        values.extend(
            _value(experiment.get(field))
            for field in (
                "payload_name",
                "encoded_product",
                "delivery_recipient_cell",
                "therapeutic_target_cell",
                "tissue_or_organ",
                "disease_model",
            )
        )
    return " ".join(str(value) for value in values if value is not None)


def _score(
    gold: dict[str, str],
    gold_evidence: dict[str, str],
    outcome: dict,
    experiment: dict | None,
    endpoint_frequency: dict[str, int],
) -> tuple[float, dict]:
    expected_text = " ".join(
        [
            gold["endpoint_name"],
            gold["normalization_basis"],
            gold["qualitative_outcome"],
            gold_evidence["evidence_text"],
        ]
    )
    actual_text = _result_text(outcome, experiment)
    outcome_only_text = _result_text(outcome, None)
    endpoint_actual_text = " ".join(
        str(_value(outcome.get(field)) or "")
        for field in ("endpoint", "assay")
    )
    expected = _tokens(expected_text)
    actual = _tokens(actual_text)
    overlap = expected & actual
    endpoint_tokens = _tokens(gold["endpoint_name"])
    outcome_only_tokens = _tokens(outcome_only_text)
    endpoint_overlap = endpoint_tokens & outcome_only_tokens
    endpoint_field_overlap = endpoint_tokens & _tokens(endpoint_actual_text)
    rare_endpoint_tokens = {
        token
        for token in endpoint_tokens
        if endpoint_frequency.get(token, 0) <= 2
    }
    rare_endpoint_overlap = rare_endpoint_tokens & outcome_only_tokens
    qualitative_overlap = (
        _tokens(gold["qualitative_outcome"]) & outcome_only_tokens
    )
    lexical = len(overlap) / max(1, min(len(expected), len(actual)))
    gold_value = float(gold["outcome_value"]) if gold["outcome_value"] else None
    actual_value = _value(outcome.get("outcome_value"))
    exact_numeric = (
        gold_value is not None
        and isinstance(actual_value, (int, float))
        and abs(float(actual_value) - gold_value) <= 1e-9
    )
    text_numeric = bool(
        gold_value is not None
        and re.search(
            rf"(?<!\d){re.escape(format(gold_value, 'g'))}(?!\d)",
            actual_text,
        )
    )
    numeric_bonus = 0.55 if exact_numeric else 0.4 if text_numeric else 0.0
    qualitative_bonus = (
        0.2
        if gold["qualitative_outcome"] and len(overlap) >= 3
        else 0.0
    )
    endpoint_specificity_bonus = (
        0.2 * len(endpoint_field_overlap) / max(1, len(endpoint_tokens))
    )
    score = min(
        1.0,
        lexical + numeric_bonus + qualitative_bonus + endpoint_specificity_bonus,
    )
    endpoint_gate = bool(
        endpoint_overlap
        and (
            exact_numeric
            or text_numeric
            or (
                (rare_endpoint_overlap and len(endpoint_overlap) >= 2)
                or len(qualitative_overlap) >= 3
            )
        )
    )
    distinctive_qualitative = _tokens(gold["qualitative_outcome"]) & {
        "few",
        "no",
        "obvious",
        "solely",
        "eliminated",
        "eradicated",
        "localized",
        "colocalized",
        "reduced",
        "defenestration",
    }
    polarity_gate = (
        not distinctive_qualitative
        or bool(distinctive_qualitative & outcome_only_tokens)
    )
    numeric_gate = (
        gold_value is None
        or gold.get("value_status") != "reported"
        or exact_numeric
        or text_numeric
    )
    if not endpoint_gate or not polarity_gate or not numeric_gate:
        score = 0.0
    return score, {
        "overlap_terms": sorted(overlap),
        "exact_numeric": exact_numeric,
        "numeric_in_text": text_numeric,
        "actual_text": actual_text,
        "endpoint_overlap_terms": sorted(endpoint_overlap),
        "endpoint_field_overlap_terms": sorted(endpoint_field_overlap),
        "rare_endpoint_overlap_terms": sorted(rare_endpoint_overlap),
        "qualitative_overlap_terms": sorted(qualitative_overlap),
        "endpoint_gate": endpoint_gate,
        "polarity_gate": polarity_gate,
        "numeric_gate": numeric_gate,
    }


def _best_one_to_one_matches(
    gold_rows: list[dict[str, str]],
    result_outcomes: list[dict],
    scored: dict[tuple[int, int], tuple[float, dict]],
    *,
    minimum_score: float = 0.42,
) -> dict[str, dict]:
    """Maximize total paper-level score without reusing a result outcome."""

    @lru_cache(maxsize=None)
    def choose(gold_index: int, used_mask: int) -> tuple[float, tuple]:
        if gold_index >= len(gold_rows):
            return 0.0, ()
        best_score, best_pairs = choose(gold_index + 1, used_mask)
        for outcome_index in range(len(result_outcomes)):
            if used_mask & (1 << outcome_index):
                continue
            score, _ = scored[(gold_index, outcome_index)]
            if score < minimum_score:
                continue
            remaining_score, remaining_pairs = choose(
                gold_index + 1,
                used_mask | (1 << outcome_index),
            )
            total_score = score + remaining_score
            if total_score > best_score:
                best_score = total_score
                best_pairs = (
                    (gold_index, outcome_index),
                    *remaining_pairs,
                )
        return best_score, best_pairs

    _, pairs = choose(0, 0)
    matches = {}
    for gold_index, outcome_index in pairs:
        gold = gold_rows[gold_index]
        outcome = result_outcomes[outcome_index]
        score, detail = scored[(gold_index, outcome_index)]
        matches[gold["gold_outcome_id"]] = {
            "outcome_id": outcome.get("outcome_id"),
            "score": round(score, 4),
            **detail,
        }
    return matches


def evaluate(
    *,
    gold_root: Path = GOLD_ROOT,
    output_root: Path = OUTPUT_ROOT,
    result_roots: list[Path] | None = None,
    packet_root: Path = PACKET_ROOT,
    task_root: Path = GOLD_GAP_TASK_ROOT,
) -> dict:
    outcomes = _rows(gold_root / "outcomes.csv")
    evidence = {
        row["evidence_id"]: row for row in _rows(gold_root / "evidence.csv")
    }
    experiments = {
        row["gold_experiment_id"]: row
        for row in _rows(gold_root / "experiments.csv")
    }
    by_paper: dict[str, list[dict[str, str]]] = {}
    endpoint_frequency: dict[str, int] = {}
    for gold in outcomes:
        for token in _tokens(gold["endpoint_name"]):
            endpoint_frequency[token] = endpoint_frequency.get(token, 0) + 1
        paper_id = experiments[gold["gold_experiment_id"]]["gold_paper_id"]
        by_paper.setdefault(paper_id, []).append(gold)
    results = []
    result_outcome_count = 0
    false_additions: list[dict] = []
    unresolved_result_items = 0
    unresolved_merge_candidates = 0
    for paper_id, gold_rows in by_paper.items():
        result_path = _result_path(paper_id, result_roots)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result_experiments = {
            row["experiment_id"]: row for row in result.get("experiments", [])
        }
        result_outcomes = result.get("outcomes", [])
        result_outcome_count += len(result_outcomes)
        unresolved_result_items += len(result.get("unresolved_items") or [])
        unresolved_merge_candidates += _merge_candidate_count(result_path)
        evidence_texts = _evidence_texts(
            paper_id,
            packet_root=packet_root,
            task_root=task_root,
        )
        scored = {}
        for gold_index, gold in enumerate(gold_rows):
            for outcome_index, outcome in enumerate(result_outcomes):
                score, detail = _score(
                    gold,
                    evidence[gold["evidence_id"]],
                    outcome,
                    result_experiments.get(outcome.get("experiment_id")),
                    endpoint_frequency,
                )
                scored[(gold_index, outcome_index)] = (score, detail)
        matches = _best_one_to_one_matches(
            gold_rows,
            result_outcomes,
            scored,
        )
        outcomes_by_id = {
            outcome.get("outcome_id"): outcome for outcome in result_outcomes
        }
        for gold in gold_rows:
            gold_id = gold["gold_outcome_id"]
            match = matches.get(gold_id)
            checked = supported = False
            check_detail = None
            if match is not None:
                matched_outcome = outcomes_by_id.get(match["outcome_id"])
                if matched_outcome is not None:
                    checked, supported, check_detail = _evidence_supports(
                        matched_outcome,
                        evidence_texts,
                    )
            results.append(
                {
                    "gold_outcome_id": gold_id,
                    "paper_id": paper_id,
                    "recovered": gold_id in matches,
                    "match": match,
                    "evidence_checked": checked,
                    "evidence_supported": supported,
                    "evidence_check": check_detail,
                }
            )
        matched_outcome_ids = {match["outcome_id"] for match in matches.values()}
        for outcome in result_outcomes:
            if outcome.get("outcome_id") in matched_outcome_ids:
                continue
            false_additions.append(
                _outcome_summary(
                    paper_id,
                    outcome,
                    result_experiments.get(outcome.get("experiment_id")),
                )
            )
    recovered = sum(row["recovered"] for row in results)
    matched_records = result_outcome_count - len(false_additions)
    evidence_checked = sum(row["evidence_checked"] for row in results)
    evidence_supported = sum(row["evidence_supported"] for row in results)
    summary = {
        "evaluation_version": "final-gold-dynamic-1.1.0",
        "matching": "semantic, numeric, one-to-one; no hard-coded recovered IDs",
        "recovered": recovered,
        "total": len(results),
        "rate": recovered / len(results),
        "missing_gold_outcome_ids": [
            row["gold_outcome_id"] for row in results if not row["recovered"]
        ],
        "results": results,
        "paid_api_requests": 0,
    }
    # P0-b. Recall alone cannot distinguish a run that found more from a run
    # that emitted more, so these travel with it -- but they are additive to a
    # report other tooling already reads, so the flag governs whether they
    # appear rather than whether they are computed.
    if is_enabled(PRECISION_METRICS_FLAG):
        summary.update(
            {
                "precision": matched_records / result_outcome_count
                if result_outcome_count
                else 0.0,
                "false_additions": {
                    "count": len(false_additions),
                    "items": false_additions,
                },
                "evidence_accuracy": {
                    "checked": evidence_checked,
                    "supported": evidence_supported,
                    "rate": evidence_supported / evidence_checked
                    if evidence_checked
                    else 0.0,
                },
                "unresolved_declared": {
                    "result_items": unresolved_result_items,
                    "merge_candidates": unresolved_merge_candidates,
                },
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
