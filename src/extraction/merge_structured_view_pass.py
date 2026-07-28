"""Fold a second extraction pass into the pipeline's current result set.

Why this exists
---------------
The result set the benchmark scores is not one call. It is a stack: a base
compact extraction (``compact_one_call_v1``), then merge stages that add
records the base run did not produce (``compact_merged_v1``,
``consolidated_gold_gap_merged_v1``). Every stage keeps what the previous stage
found and adds to it, and the benchmark resolves each paper to the deepest
stage that produced it.

Changing the evidence view changes how a paper is *read*, so it belongs in that
same stack rather than beside it. The structured view
(``structured_evidence_view``) keeps the table rows the compact packet's ranker
drops -- on GP-006 the compact packet contains ``1.01`` zero times and the
structured packet at the same budget contains it four times -- and the records
that come back from reading those rows have to join the result set without
discarding the ones already there.

What this is NOT
----------------
This is not an ensemble over configurations. ``union_extraction_results``
merges runs taken at *different* flag settings (full view on, compact, slot
enforcement off) and is explicitly recall-first. This module merges exactly two
inputs at the *same*, shipped flag settings: the current result for a paper and
one further pass over the same paper. Both are the default configuration; only
the stage differs.

The cost is precision, and it is structural rather than incidental: the
benchmark matches gold to results one-to-one, so a stage that adds records can
raise recall and can only lower precision. ``precision_metrics`` reports it.

Known limitation
----------------
Both stages number their experiments independently and both start at ``EX1``,
so an experiment id from the second pass that collides with an earlier one is
dropped and its outcomes bind to the earlier experiment record. That is
inherited from ``union_paper`` and left as-is on purpose: the two stages read
the same paper, so the colliding ids usually do denote the same experiment, and
re-keying would split one experiment into two. It does mean the experiment
context attached to a second-pass outcome is the earlier stage's. The
evaluation's gates read the outcome's own fields, so this can move a lexical
score but cannot by itself create a match.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.extraction.union_extraction_results import union_paper


ROOT = Path(__file__).resolve().parents[2]
MERGE_VERSION = "structured-view-pass-merge-1.0.0"

# The stack as it stands before this stage, deepest first. Same order and same
# first-match rule the benchmark uses to resolve a paper, so a merged paper is
# built on the record set that was actually being scored.
DEFAULT_BASELINE_ROOTS = [
    ROOT / "data/staging/extraction/consolidated_gold_gap_merged_v1",
    ROOT / "data/staging/extraction/compact_merged_v1_1",
    ROOT / "data/staging/extraction/compact_merged_v1",
    ROOT / "data/staging/extraction/compact_one_call_v1",
]
DEFAULT_PASS_ROOT = ROOT / "data/staging/extraction/structured_compact_one_call_v1"
DEFAULT_OUTPUT_ROOT = ROOT / "data/staging/extraction/structured_view_merged_v1"
RESULT_FILENAMES = ("final_result.json", "result.json")


def resolve_baseline(paper_id: str, baseline_roots: list[Path]) -> Path | None:
    """Return the root holding the paper's current result, or None."""
    for root in baseline_roots:
        for name in RESULT_FILENAMES:
            if (root / paper_id / name).exists():
                return root
    return None


def merge_paper(
    paper_id: str,
    *,
    baseline_roots: list[Path],
    pass_root: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Merge one paper's current result with the second pass over that paper."""
    baseline_root = resolve_baseline(paper_id, baseline_roots)
    has_pass = any(
        (pass_root / paper_id / name).exists() for name in RESULT_FILENAMES
    )
    roots = [root for root in (baseline_root, pass_root if has_pass else None) if root]
    detail: dict[str, Any] = {
        "paper_id": paper_id,
        "baseline_root": baseline_root.name if baseline_root else None,
        "second_pass_present": has_pass,
    }
    if not roots:
        return None, detail
    merged = union_paper(paper_id, roots)
    if merged is None:
        return None, detail
    by_run = Counter(row.get("source_run") for row in merged["outcomes"])
    detail.update(
        {
            "outcomes": len(merged["outcomes"]),
            "experiments": len(merged["experiments"]),
            "outcomes_by_stage": dict(by_run),
            "outcomes_added_by_second_pass": by_run.get(pass_root.name, 0),
        }
    )
    return merged, detail


def build(
    *,
    baseline_roots: list[Path] = DEFAULT_BASELINE_ROOTS,
    pass_root: Path = DEFAULT_PASS_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    paper_ids: list[str] | None = None,
) -> dict[str, Any]:
    paper_ids = paper_ids or [f"GP-{n:03d}" for n in range(1, 10)]
    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "merge_version": MERGE_VERSION,
        "note": (
            "One configuration, two stages: each paper's current result plus a "
            "second pass over the same paper through the structured evidence "
            "view. Records keep the stage that produced them in source_run."
        ),
        "baseline_roots": [_display(root) for root in baseline_roots],
        "second_pass_root": _display(pass_root),
        "papers": [],
    }
    for paper_id in paper_ids:
        merged, detail = merge_paper(
            paper_id, baseline_roots=baseline_roots, pass_root=pass_root
        )
        report["papers"].append(detail)
        if merged is None:
            continue
        paper_dir = output_root / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        (paper_dir / "final_result.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # Carried forward, not regenerated. Anything reading a result directory
        # reads the merge report beside it -- the evaluation counts the
        # candidate ids an earlier stage left unresolved -- and those candidates
        # are not in the second pass's id space, so a stage that dropped the
        # file would silently reset that count to zero rather than report it.
        detail["carried_merge_report"] = _carry_merge_report(
            paper_id, baseline_roots=baseline_roots, paper_dir=paper_dir
        )
    (output_root / "merge_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _carry_merge_report(
    paper_id: str, *, baseline_roots: list[Path], paper_dir: Path
) -> bool:
    baseline_root = resolve_baseline(paper_id, baseline_roots)
    if baseline_root is None:
        return False
    source = baseline_root / paper_id / "merge_report.json"
    if not source.exists():
        return False
    (paper_dir / "merge_report.json").write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return True


def _display(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        action="append",
        dest="baseline_roots",
        type=Path,
        help="Repeatable, deepest stage first; the first match wins per paper.",
    )
    parser.add_argument("--pass-root", type=Path, default=DEFAULT_PASS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--paper-id", action="append", dest="paper_ids")
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Required: this rewrites tracked result files in place.",
    )
    args = parser.parse_args()
    if not args.confirm_write:
        parser.error("--confirm-write is required; this rewrites tracked files")
    report = build(
        baseline_roots=args.baseline_roots or DEFAULT_BASELINE_ROOTS,
        pass_root=args.pass_root,
        output_root=args.output_root,
        paper_ids=args.paper_ids,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
