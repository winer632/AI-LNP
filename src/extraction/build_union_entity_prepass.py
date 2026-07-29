"""Overlay one entity-prepass run onto the 13/15 root, for measurement only.

Why an overlay rather than a full sweep
---------------------------------------
The capability adds a stage before extraction; measuring it means re-extracting.
Re-extracting all nine papers would change every paper at once and make the
delta unattributable, so this follows the same shape the interpretive-admission
measurement used: re-run one paper, merge its records over the base root's, and
copy every other paper byte for byte. Whatever moves, moved because of that one
run.

Merging rather than swapping is deliberate for the same reason it was there: a
swap can lose a gold row the base already recovers, and a loss caused by not
sending a record is not evidence about the capability under test.

The experiment-id rename
------------------------
``union_paper`` keeps the *first* record for a colliding ``experiment_id``, just
as it keeps the first outcome for a colliding key. The base root and the overlay
both number their experiments from E1, and the overlay's experiment records are
where the entity table's answer actually lands -- an experiment's target-cell
field is the one that names the population. Without the rename the base's
experiment would win and the treatment would be silently discarded from the
thing being measured. Renaming keeps both, and each run's outcomes stay attached
to the experiment their own run described.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from src.extraction.union_extraction_results import union_paper

ROOT = Path(__file__).resolve().parents[2]

BASE_ROOT = ROOT / "data/staging/extraction/codex_union_vision_v3"
OVERLAY_ROOT = ROOT / "data/staging/extraction/codex_entity_prepass_gp008_v1"
ENTITY_TABLE_ROOT = ROOT / "data/staging/extraction/entity_tables_v1"
DEFAULT_OUTPUT = ROOT / "data/staging/extraction/codex_union_vision_v3_entity_prepass"
OVERLAY_PAPER = "GP-008"
# Suffix, not a fixed map: the overlay's experiment ids are whatever that run
# chose, and hard-coding E1/E2 would silently no-op on a run that numbered
# differently.
RENAME_SUFFIX = "-e"


def _repo_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _load(run_dir: Path) -> dict[str, Any] | None:
    for name in ("final_result.json", "result.json"):
        candidate = run_dir / name
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def rename_experiments(result: dict[str, Any], suffix: str) -> dict[str, Any]:
    """Suffix every experiment id and repoint the outcomes that reference it."""
    renamed = dict(result)
    mapping = {
        str(row.get("experiment_id")): f"{row.get('experiment_id')}{suffix}"
        for row in result.get("experiments", [])
    }
    renamed["experiments"] = [
        {**row, "experiment_id": mapping[str(row.get("experiment_id"))]}
        for row in result.get("experiments", [])
    ]
    renamed["outcomes"] = [
        {**row, "experiment_id": mapping.get(
            str(row.get("experiment_id")), row.get("experiment_id")
        )}
        for row in result.get("outcomes", [])
    ]
    return renamed


def build(
    *,
    output_root: Path = DEFAULT_OUTPUT,
    base_root: Path = BASE_ROOT,
    overlay_root: Path = OVERLAY_ROOT,
    overlay_paper: str = OVERLAY_PAPER,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    overlay = _load(overlay_root / overlay_paper)
    if overlay is None:
        raise FileNotFoundError(f"no {overlay_paper} result under {overlay_root}")
    base = _load(base_root / overlay_paper)
    if base is None:
        raise FileNotFoundError(f"no {overlay_paper} result under {base_root}")

    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(base_root, output_root)

    scratch = scratch_root or (output_root / ".overlay")
    (scratch / overlay_paper).mkdir(parents=True, exist_ok=True)
    (scratch / overlay_paper / "final_result.json").write_text(
        json.dumps(rename_experiments(overlay, RENAME_SUFFIX), ensure_ascii=False,
                   indent=2) + "\n",
        encoding="utf-8",
    )

    merged = union_paper(overlay_paper, [base_root, scratch])
    if merged is None:
        raise FileNotFoundError(f"nothing to merge for {overlay_paper}")
    (output_root / overlay_paper / "final_result.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if scratch_root is None:
        shutil.rmtree(scratch, ignore_errors=True)

    outcomes = sum(
        len(json.loads(path.read_text(encoding="utf-8")).get("outcomes", []))
        for path in sorted(output_root.glob("GP-*/final_result.json"))
    )
    table_path = ENTITY_TABLE_ROOT / overlay_paper / "entity_table.json"
    manifest = {
        "union_version": "union-extraction-1.2.0",
        "purpose": (
            "Measure what entity_resolution_prepass costs and buys. Not a "
            "shipped result root: the flag it measures ships off."
        ),
        "note": (
            f"{_repo_path(base_root)} with the entity-prepass "
            f"{overlay_paper} run merged over its {overlay_paper}. Every other "
            "paper is copied byte for byte from the base. Rebuild with "
            "`python -m src.extraction.build_union_entity_prepass "
            "--confirm-write`."
        ),
        "base_root": _repo_path(base_root),
        "overlay_root": _repo_path(overlay_root),
        "overlay_paper": overlay_paper,
        "entity_table": _repo_path(table_path) if table_path.exists() else None,
        "experiment_id_suffix": RENAME_SUFFIX,
        "outcomes": {
            "base": len(base.get("outcomes", [])),
            "overlay": len(overlay.get("outcomes", [])),
            "merged": len(merged.get("outcomes", [])),
        },
        "total_outcomes": outcomes,
        "requires_flags": ["vision_relationship_polarity", "record_level_salvage"],
        "scoring_reads_flag": "vision_relationship_polarity",
    }
    (output_root / "union_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-root", type=Path, default=BASE_ROOT)
    parser.add_argument("--overlay-root", type=Path, default=OVERLAY_ROOT)
    parser.add_argument("--overlay-paper", default=OVERLAY_PAPER)
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Required: this rewrites a tracked result root in place.",
    )
    args = parser.parse_args()
    if not args.confirm_write:
        parser.error("--confirm-write is required; this rewrites tracked files")
    print(json.dumps(
        build(
            output_root=args.output_root,
            base_root=args.base_root,
            overlay_root=args.overlay_root,
            overlay_paper=args.overlay_paper,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
