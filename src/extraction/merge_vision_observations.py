"""Turn accepted panel reads into outcome records and merge them into a result.

A SelectiveVisionResponse is an observation about an image, not an outcome
record, so nothing downstream can score it. This converts the ones that resolved
into the outcome shape the evaluator reads, and merges them into an existing
extraction result.

Only ``resolved`` observations are converted. ``missing``, ``ambiguous`` and
``human_review`` are carried into the report but never become records: an
observation that abstained must not silently turn into a claim, which is the
whole point of the abstain rule. An abstaining observation therefore never
acquires a ``vision_relationship`` either -- it returns before the field is
written.

The observation's ``relationship`` is carried through as a structured field
rather than dropped. It is a closed seven-value vocabulary
(:data:`~src.extraction.selective_vision_contracts.VisualRelationship`) that
the vision model chose under schema constraint, having never seen the gold
set, so carrying it propagates a structured judgement the pipeline already
made rather than injecting benchmark vocabulary. The evaluator can then
compare a declared relation against a gold claim structurally instead of
guessing at polarity from prose.

It is deliberately NOT spliced into any text field. The negative member is
spelled with the positive one inside it, so a text encoding of
"not_colocalized" tokenises to {"not", "colocalized"} and reads to a
bag-of-words matcher as an affirmation of the very relation it denies. The
field is structured precisely so that no consumer has to survive that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _field(value: Any, evidence_ids: list[str]) -> dict[str, Any]:
    if value in (None, ""):
        return {
            "value": None,
            "status": "missing",
            "evidence_ids": [],
            "missing_reason": "Not printed in the panel.",
        }
    return {
        "value": value,
        "status": "reported",
        "evidence_ids": evidence_ids,
        "missing_reason": None,
    }


def _descriptive_labels(labels: list[str]) -> list[str]:
    """Drop bare axis ticks from the text a match is computed over.

    A panel read transcribes every readable label, which includes the axis
    scale: a killing-efficiency chart prints 0, 20, 40, 60, 80, 100. Those are
    scale markings, not measurements, and putting them in the record's text
    made the evaluator's numeric check succeed for any gold value that happened
    to equal a tick. GO-016's gold value is 20, so a vision record about
    phagocytosis matched it and displaced the evidence-bearing text record that
    actually reported it.

    A tick is a bare number: no unit, no percent sign, no other character.
    "41.5%" and "Phagocytic index(%)" are kept, because a printed value with a
    unit is a real observation. The full list stays on the record under
    printed_labels, so nothing is lost as provenance -- it just no longer feeds
    matching.
    """
    kept = []
    for label in labels:
        text = str(label).strip()
        if not text:
            continue
        if text.replace(".", "", 1).replace("-", "", 1).isdigit():
            continue
        kept.append(text)
    return kept


def observation_to_outcome(
    observation: dict[str, Any], *, outcome_id: str, experiment_id: str
) -> dict[str, Any] | None:
    if observation.get("disposition") != "resolved":
        return None
    fragment = observation.get("corrected_fragment") or {}
    evidence_ids = list(observation.get("supporting_evidence_ids") or [])
    # What this field carries is what the observation was made ON: the panel,
    # and the marker names read off it. Not the claim, and not the description
    # of the picture.
    #
    # It used to carry the claim sentence and visible_support as well. Both are
    # scored -- _result_text reads endpoint alongside qualitative_outcome -- so
    # the claim was counted twice and the provenance sentence was matched as
    # though it were a measurement. That put vision records at 55-62 tokens
    # against 8-19 for text records, and because `lexical` divides by
    # min(len(expected), len(actual)), bulk past the gold length is free while
    # every incidental overlap is gain. A record won by being verbose: on the
    # GP-008-merged root a panel read took GO-015 at 0.9958 on 6 overlap terms
    # from the text record reporting it at 1.0 on 13.
    #
    # figure_or_table is already the assay, and _result_text unions token sets,
    # so naming it here again would add nothing.
    parts = [
        observation.get("panel_or_table_cell") or "",
        " ".join(_descriptive_labels(observation.get("printed_labels") or [])),
    ]
    endpoint = " ".join(part for part in parts if part).strip() or observation.get(
        "figure_or_table"
    )
    return {
        "outcome_id": outcome_id,
        "experiment_id": experiment_id,
        "assay": _field(observation.get("figure_or_table"), evidence_ids),
        "endpoint": _field(endpoint, evidence_ids),
        "comparator": _field(None, evidence_ids),
        "outcome_value": _field(fragment.get("outcome_value"), evidence_ids),
        "outcome_unit": _field(fragment.get("outcome_unit"), evidence_ids),
        "qualitative_outcome": _field(fragment.get("qualitative_outcome"), evidence_ids),
        "source_stage": "vision",
        "vision_panel": observation.get("panel_or_table_cell"),
        "printed_labels": list(observation.get("printed_labels") or []),
        # Structured, never folded into the text above. See the module
        # docstring: the enum's negative member contains its own positive
        # member as a substring, so only a structured comparison is safe.
        "vision_relationship": observation.get("relationship"),
    }


def merge_into_result(
    result_path: Path, observations: list[dict[str, Any]], output_path: Path
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    existing = {str(row.get("outcome_id")) for row in result.get("outcomes", [])}
    experiment_id = (
        str(result["experiments"][0]["experiment_id"])
        if result.get("experiments")
        else "E1"
    )

    added = []
    skipped = []
    index = 1
    for observation in observations:
        outcome_id = f"V{index}"
        while outcome_id in existing:
            index += 1
            outcome_id = f"V{index}"
        record = observation_to_outcome(
            observation, outcome_id=outcome_id, experiment_id=experiment_id
        )
        if record is None:
            skipped.append(
                {
                    "finding_id": observation.get("finding_id"),
                    "disposition": observation.get("disposition"),
                }
            )
            continue
        existing.add(outcome_id)
        index += 1
        result.setdefault("outcomes", []).append(record)
        added.append(outcome_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "result_path": str(result_path),
        "output_path": str(output_path),
        "added_outcome_ids": added,
        "skipped_observations": skipped,
        "total_outcomes": len(result.get("outcomes", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge accepted panel reads into an extraction result."
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--observation", action="append", dest="observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observations = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.observations
    ]
    print(
        json.dumps(
            merge_into_result(args.result, observations, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
