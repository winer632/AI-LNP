"""Blind second opinion on pairs the deterministic matcher rejected.

Why this is a side channel and not a replacement
------------------------------------------------
``evaluate_final_gold_dynamic`` is deterministic: same inputs, same number,
byte-reproducible. That property is why the 10/15 baseline could be verified at
all. Swapping its matcher for a model would trade it away, and a benchmark you
cannot reproduce cannot be used to judge a change.

So this never touches the primary number. It re-examines only the pairs the
deterministic gates rejected, and reports a separate count alongside.

What it is for
--------------
The gates are lexical, and lexical gates have a known failure mode here. Gold
writes ``GFP_expression_in_LYVE1_positive_LSECs``; a vision read writes ``GFP
signal is present in LYVE-1-positive cells``. Under ``[a-z0-9]+``, ``LYVE1``
stays one token and ``LYVE-1`` splits into ``lyve`` and ``1``, so the marker
name the claim is about can never match. The gold set is inconsistent with
itself on this: ``endpoint_name`` writes ``LYVE1`` while its own
``qualitative_outcome`` writes ``LYVE-1``.

Blinding
--------
The judge sees two descriptions labelled A and B, in an order fixed by content
hash rather than by role, and is not told which came from the gold set. It is
asked only whether they describe the same measurement of the same quantity in
the same population. It cannot look anything up: the call runs in an isolated
directory with --sandbox read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from src.extraction.run_codex_one_call import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    HARNESS,
    codex_available,
)

ROOT = Path(__file__).resolve().parents[2]
GOLD_ROOT = ROOT / "data/annotations/gold_v1"
DEFAULT_OUTPUT = ROOT / "reports/extraction/blind_adjudication_v1"

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["same_measurement", "confidence", "reason"],
    "properties": {
        "same_measurement": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    },
}

PROMPT = (
    "Two descriptions of a scientific result are given below, labelled A and B.\n\n"
    "Decide whether they describe the SAME measurement: the same quantity, "
    "measured on the same population, with the same direction or polarity.\n\n"
    "Same measurement means the same endpoint on the same cell type or "
    "population. Different endpoints on the same cells are NOT the same, and "
    "the same endpoint on different cell types is NOT the same. A stated "
    "presence and a stated absence of the same signal are NOT the same. "
    "Differences of wording, punctuation, hyphenation or field naming do not "
    "make them different.\n\n"
    "Answer false if you are unsure. A wrong 'true' silently inflates a "
    "benchmark; a wrong 'false' only leaves a gap visible.\n"
)


def _order(first: str, second: str) -> tuple[str, str, bool]:
    """Order the pair by content hash so position never encodes role."""
    a = hashlib.sha256(first.encode("utf-8")).hexdigest()
    b = hashlib.sha256(second.encode("utf-8")).hexdigest()
    if a <= b:
        return first, second, False
    return second, first, True


def judge_pair(
    gold_text: str,
    candidate_text: str,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout: int = 600,
) -> dict[str, Any]:
    left, right, swapped = _order(gold_text, candidate_text)
    prompt = f"{PROMPT}\nA: {left}\n\nB: {right}\n"

    workdir = Path(tempfile.mkdtemp(prefix="codex_judge_"))
    try:
        schema_path = workdir / "schema.json"
        schema_path.write_text(json.dumps(JUDGE_SCHEMA), encoding="utf-8")
        output_path = workdir / "last_message.txt"
        started = time.time()
        completed = subprocess.run(
            [
                "codex", "exec", "-m", model,
                "-c", f"model_reasoning_effort={reasoning_effort}",
                "--output-schema", str(schema_path),
                "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
                "-o", str(output_path), "-",
            ],
            cwd=workdir, input=prompt, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "OPENAI_API_KEY": "", "SENSENOVA_API_KEY": ""},
        )
        elapsed = time.time() - started
        if completed.returncode != 0 or not output_path.exists():
            return {"error": completed.stderr[-300:], "same_measurement": False}
        verdict = json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    verdict["blinded_order_swapped"] = swapped
    verdict["elapsed_seconds"] = round(elapsed, 3)
    return verdict


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def gold_description(outcome: dict[str, str]) -> str:
    parts = [
        outcome.get("endpoint_name", "").replace("_", " "),
        outcome.get("outcome_value", ""),
        outcome.get("outcome_unit", "").replace("_", " "),
        outcome.get("qualitative_outcome", ""),
        outcome.get("normalization_basis", "").replace("_", " "),
    ]
    return " | ".join(part for part in parts if part)


def run(
    *,
    result_root: Path,
    missing_ids: list[str],
    output_root: Path = DEFAULT_OUTPUT,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    outcomes = {row["gold_outcome_id"]: row for row in _rows(GOLD_ROOT / "outcomes.csv")}
    experiments = {
        row["gold_experiment_id"]: row for row in _rows(GOLD_ROOT / "experiments.csv")
    }

    from src.extraction.evaluate_final_gold_dynamic import _result_text

    report = {
        "report_version": "blind-adjudication-1.0.0",
        "harness": HARNESS,
        "model": model,
        "note": (
            "Second opinion on pairs the deterministic matcher rejected. Never "
            "replaces the primary recall number; reported alongside it."
        ),
        "verdicts": [],
    }
    for gold_id in missing_ids:
        gold = outcomes.get(gold_id)
        if gold is None:
            continue
        paper_id = experiments[gold["gold_experiment_id"]]["gold_paper_id"]
        result_path = result_root / paper_id / "final_result.json"
        if not result_path.exists():
            result_path = result_root / paper_id / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        gold_text = gold_description(gold)

        best = None
        for outcome in result.get("outcomes", []):
            candidate_text = _result_text(outcome, None)
            verdict = judge_pair(gold_text, candidate_text, model=model)
            row = {
                "gold_outcome_id": gold_id,
                "paper_id": paper_id,
                "outcome_id": outcome.get("outcome_id"),
                "same_measurement": bool(verdict.get("same_measurement")),
                "confidence": verdict.get("confidence"),
                "reason": verdict.get("reason"),
            }
            report["verdicts"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if row["same_measurement"] and row["confidence"] == "high":
                best = row
                break
        if best:
            report.setdefault("adjudicated_matches", []).append(gold_id)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "adjudication.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blind second opinion on deterministically rejected matches."
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--missing-id", action="append", dest="missing_ids", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--confirm-codex-quota", action="store_true")
    args = parser.parse_args()

    if not args.confirm_codex_quota:
        parser.error("--confirm-codex-quota is required")
    if not codex_available():
        parser.error("`codex` was not found on PATH")

    report = run(
        result_root=args.result_root,
        missing_ids=args.missing_ids,
        output_root=args.output_root,
        model=args.model,
    )
    print(json.dumps(
        {"adjudicated_matches": report.get("adjudicated_matches", []),
         "verdicts": len(report["verdicts"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
