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


def observation_to_outcome(
    observation: dict[str, Any], *, outcome_id: str, experiment_id: str
) -> dict[str, Any] | None:
    if observation.get("disposition") != "resolved":
        return None
    fragment = observation.get("corrected_fragment") or {}
    evidence_ids = list(observation.get("supporting_evidence_ids") or [])
    # The observation was made about a specific panel, and what the model
    # actually read off it is in printed_labels. Dropping those loses the
    # marker names the claim is about, so carry them alongside the sentence.
    parts = [
        fragment.get("qualitative_outcome") or "",
        " ".join(observation.get("printed_labels") or []),
        observation.get("visible_support") or "",
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
