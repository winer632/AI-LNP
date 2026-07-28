"""Dynamically match frozen gold outcomes to final compact records one-to-one."""

from __future__ import annotations

import csv
import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

from src.config_flags import is_enabled
from src.extraction.selective_vision_contracts import RELATIONSHIP_POLARITY


ROOT = Path(__file__).resolve().parents[2]
PRECISION_METRICS_FLAG = "precision_metrics"
VISION_RELATIONSHIP_FLAG = "vision_relationship_polarity"
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

# Terms whose presence in a gold qualitative claim makes the claim specific
# enough that a candidate record has to echo them to count as the same
# measurement.
#
# What this set does NOT do is decide polarity, despite the name of the gate
# built on it. Membership is tested against the candidate's tokens, and
# tokenising a denial leaves the affirmed word in place: "not colocalized"
# yields {"not", "colocalized"} and satisfies a gold row asserting
# "colocalized". Measured on 500 generated opposite-polarity candidates, this
# test accepts 70 of them.
#
# The axis it really separates is vocabulary reuse, not direction, so any
# change that makes matching more generous makes it accept MORE denials:
# equivalence classes over these terms took the same 500 to 150 accepted, and
# collapsing the -ize/-ization family took them to 110. Both were rejected on
# that measurement. `_declared_relationship` is the structural route that can
# see a denial, and the only one that improved the number.
#
# Written in surface form and normalised through `_tokens` below, never
# compared raw. `obvious` used to be listed here as a literal and was
# unreachable for it: the de-pluraliser turned the text "obvious" into
# "obviou", so the entry could never match anything. Normalising the set
# through the same function as the text makes that class of bug impossible
# rather than merely fixed once.
#
# `eradicated` occurs in no gold qualitative claim, so it is never looked up.
# Retained because the set describes claim vocabulary generally, not only the
# rows frozen today.
_DISTINCTIVE_SURFACE = {
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


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _value(value):
    return value.get("value") if isinstance(value, dict) else value


# Biological marker names are written with the separator between the stem and
# its numeric suffix chosen freely: LYVE1 / LYVE-1 / LYVE_1, F4/80 / F4_80.
# The gold set uses both spellings of one marker inside a single row -- GO-003
# writes LYVE1_positive_LSECs in endpoint_name and LYVE-1 in
# qualitative_outcome, GO-002 writes F4_80 and F4/80 -- so by the gold set's
# own usage the spellings denote the same entity.
#
# Without this the underscore becomes a space and the marker splits into
# lyve + 1 while the unhyphenated spelling stays one token, so the name the
# claim is about can never match. Applied to both sides of every comparison,
# so it removes a spelling difference rather than favouring a prediction.
#
# The separator must be punctuation. An earlier version also fused across a
# space and silently ate real tokens: "more than 80% of BMDMs" became
# "than80", which dropped the 80 an evidence check was looking for. A space
# between a word and a number is ordinary prose, not a marker name.
# Lookarounds rather than \b: underscore is a word character to the regex
# engine, so \b never fires inside F4_80_positive and that spelling would not
# fuse while F4/80 did.
_MARKER_SEPARATOR = re.compile(r"(?<![a-z0-9])([a-z][a-z0-9]*)[-_/](\d+)(?![0-9])")


def _fuse_marker_names(text: str) -> str:
    return _MARKER_SEPARATOR.sub(lambda m: f"{m.group(1)}{m.group(2)}", text)


def _tokens(text: str) -> set[str]:
    # "colocalization" was folded into "colocalized" here. Removed on
    # measurement: it bought no recall on any root, and it cost precision on
    # the polarity gate, taking generated opposite-polarity candidates from
    # 70/500 accepted to 80/500. The gate is negation-blind, so every extra
    # surface form that reaches a lexicon entry reaches it just as readily
    # from inside a denial -- "no colocalization was observed" became a match
    # for a gold row asserting colocalisation. The recall it was there for is
    # now served by the declared relationship, which can tell the two apart.
    normalized = (
        _fuse_marker_names(text.lower())
        .replace("_", " ")
        .replace("exclusively", "solely")
    )
    found = set()
    for token in re.findall(r"[a-z0-9]+", normalized):
        # A trailing "s" is a plural marker only when it does not belong to
        # the stem. Guarding "ss" alone mangled every -us/-is/-as word:
        # "obvious" became "obviou" and "analysis" became "analysi". That is
        # wrong on its own terms, and it silently killed the "obvious" entry
        # in the distinctive lexicon, which is compared against these tokens.
        if (
            len(token) >= 5
            and token.endswith("s")
            and not token.endswith(("ss", "us", "is", "as"))
        ):
            token = token[:-1]
        if len(token) >= 2 and token not in STOP:
            found.add(token)
    return found


# Normalised through `_tokens`, so a lexicon entry is compared in the same
# space as the text it is matched against and cannot go stale when the
# tokeniser changes.
DISTINCTIVE_TERMS = {
    token for term in _DISTINCTIVE_SURFACE for token in _tokens(term)
}


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


def _declared_relationship(outcome: dict) -> tuple[str, bool] | None:
    """The relation this record declares structurally, as ``(base, affirmed)``.

    Reads the dedicated ``vision_relationship`` field written by
    :mod:`src.extraction.merge_vision_observations` from the vision contract's
    closed vocabulary. Returns ``None`` for a record that declares nothing,
    which is every record produced by the text route, so those keep the
    unchanged token behaviour.

    Structural on purpose. The same value carried as text would be read by the
    tokeniser as its own opposite, because "not_colocalized" contains
    "colocalized"; consulting the enum is what makes a denial legible.
    """
    if not is_enabled(VISION_RELATIONSHIP_FLAG):
        return None
    value = _value(outcome.get("vision_relationship"))
    if not isinstance(value, str):
        return None
    return RELATIONSHIP_POLARITY.get(value)


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
    distinctive_qualitative = _tokens(gold["qualitative_outcome"]) & DISTINCTIVE_TERMS
    polarity_gate = (
        not distinctive_qualitative
        or bool(distinctive_qualitative & outcome_only_tokens)
    )
    # A record that declares a relation structurally is judged on that
    # declaration rather than on its prose. This is the only way a denial can
    # be seen at all: the prose of a denial contains the affirmed word, so the
    # token test above reads "not colocalized" as a match for "colocalized".
    declared = _declared_relationship(outcome)
    if declared is not None:
        base, affirmed = declared
        # Normalised through `_tokens`, for the same reason the lexicon is:
        # comparing the raw enum value against normalised tokens is how a
        # hard-coded string goes dead without anything failing.
        base_tokens = _tokens(base)
        asserted = base_tokens & distinctive_qualitative
        if asserted:
            if not affirmed:
                # The record denies exactly what the gold row asserts.
                polarity_gate = False
            else:
                # The declaration discharges its own term; any other
                # distinctive term the gold row carries still has to be met.
                remaining = distinctive_qualitative - asserted
                polarity_gate = not remaining or bool(
                    remaining & outcome_only_tokens
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
            # Position in result_outcomes, and the only identifier here that
            # is unique. outcome_id is not: a union root merges records from
            # several runs and each run numbers its own records from O1, so
            # codex_union_v1 holds 51 records under 32 distinct ids. Callers
            # must select the matched record by this index -- selecting by id
            # picks an arbitrary record among the duplicates.
            "outcome_index": outcome_index,
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
        for gold in gold_rows:
            gold_id = gold["gold_outcome_id"]
            match = matches.get(gold_id)
            checked = supported = False
            check_detail = None
            if match is not None:
                # By index, not by id. Looking the record up by outcome_id
                # returned whichever duplicate came last in the list, so on a
                # union root the evidence check could run against a record
                # the matcher never chose.
                matched_outcome = result_outcomes[match["outcome_index"]]
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
        # Exclude the matched records by position. Excluding by outcome_id
        # dropped every record sharing an id with a matched one, so a union
        # root's 19 duplicate ids were subtracted from the false-addition
        # count and precision came out higher than the records justify.
        matched_indices = {match["outcome_index"] for match in matches.values()}
        for outcome_index, outcome in enumerate(result_outcomes):
            if outcome_index in matched_indices:
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
        "evaluation_version": "final-gold-dynamic-1.2.0",
        # Which results were scored. Without this the file cannot be
        # reproduced from itself: the default single-configuration run and the
        # cross-configuration union both write here and differ by a whole
        # outcome, so a reader had no way to tell which number they were
        # holding, and re-running the default command silently replaced one
        # with the other.
        "result_roots": [
            str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root)
            for root in (result_roots if result_roots is not None else RESULT_ROOTS)
        ],
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



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Required: this rewrites tracked report files in place.",
    )
    args = parser.parse_args()
    if not args.confirm_write:
        parser.error("--confirm-write is required; this rewrites tracked files")
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
