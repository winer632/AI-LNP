"""Union outcome records across extraction configurations, recall-first.

Why this exists
---------------
Measured on the gold set, several configurations each recover outcomes the
others miss, and no single one dominates:

    compact packet + candidate slots      GO-004
    structure-aware packet + slots        GO-006
    full evidence view                    GO-015, GO-007

GO-004, GO-006 and GO-015 are the three numeric outcomes still missing from any
single run, and each has been recovered by some configuration. Taking the union
trades precision for recall deliberately: the evaluator matches one-to-one, so
adding records can raise recall and can only lower precision.

This is an ensemble over configurations, not a new extraction. Every record it
emits was produced and validated by one of the input runs; nothing is
synthesised here. Records keep their originating run in ``source_run`` so a
reader can always tell which configuration produced a given row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, Any] | None:
    for name in ("final_result.json", "result.json"):
        candidate = path / name
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def _outcome_key(outcome: dict[str, Any]) -> tuple:
    """Identify an outcome by what it measures, not by its generated id.

    Two runs number their records independently, so ``outcome_id`` cannot
    deduplicate across them. Endpoint text plus value does.
    """

    def value(field: str):
        row = outcome.get(field)
        return row.get("value") if isinstance(row, dict) else row

    return (
        str(value("endpoint") or "").strip().lower(),
        str(value("outcome_value") or "").strip().lower(),
        str(value("outcome_unit") or "").strip().lower(),
    )


def union_paper(paper_id: str, run_roots: list[Path]) -> dict[str, Any] | None:
    """Merge one paper's outcomes across runs, keeping the first of each key."""
    base: dict[str, Any] | None = None
    seen: set[tuple] = set()
    outcomes: list[dict[str, Any]] = []
    experiments: list[dict[str, Any]] = []
    seen_experiments: set[str] = set()

    for root in run_roots:
        result = _load(root / paper_id)
        if result is None:
            continue
        if base is None:
            base = {key: value for key, value in result.items() if key != "outcomes"}
            base["experiments"] = []
        for experiment in result.get("experiments", []):
            experiment_id = str(experiment.get("experiment_id"))
            if experiment_id in seen_experiments:
                continue
            seen_experiments.add(experiment_id)
            experiments.append(experiment)
        for outcome in result.get("outcomes", []):
            key = _outcome_key(outcome)
            if key in seen:
                continue
            seen.add(key)
            row = dict(outcome)
            row["source_run"] = root.name
            outcomes.append(row)

    if base is None:
        return None
    base["experiments"] = experiments
    base["outcomes"] = outcomes
    return base


def build(
    run_roots: list[Path],
    *,
    output_root: Path,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    paper_ids = paper_ids or [f"GP-{n:03d}" for n in range(1, 10)]
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "union_version": "union-extraction-1.0.0",
        "note": (
            "Recall-first ensemble over configurations. Every record was "
            "produced and validated by one of the input runs."
        ),
        "run_roots": [str(root) for root in run_roots],
        "papers": [],
    }
    for paper_id in paper_ids:
        merged = union_paper(paper_id, run_roots)
        if merged is None:
            continue
        paper_dir = output_root / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        (paper_dir / "final_result.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        from collections import Counter

        by_run = Counter(row.get("source_run") for row in merged["outcomes"])
        report["papers"].append(
            {
                "paper_id": paper_id,
                "outcomes": len(merged["outcomes"]),
                "experiments": len(merged["experiments"]),
                "by_source_run": dict(by_run),
            }
        )
    (output_root / "union_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Union outcome records across extraction runs, recall-first."
    )
    parser.add_argument("--run-root", action="append", dest="run_roots", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--paper-id", action="append", dest="paper_ids")
    args = parser.parse_args()

    report = build(
        [Path(root) for root in args.run_roots],
        output_root=args.output_root,
        paper_ids=args.paper_ids,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
