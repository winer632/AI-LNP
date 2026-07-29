"""Overlay a nine-paper run onto the 13/15 root, for measurement only.

Why a sweep rather than one paper
---------------------------------
The two capabilities measured before this one changed what a record *says*
about one paper, so re-running that paper isolated them. This one changes what
a *field* means, on every record of every paper, and a definition that helps one
paper and quietly damages three is exactly the outcome a single-paper overlay
cannot see. So all nine are re-run and all nine are merged.

That makes the confound the opposite one: merging any fresh nine-paper run adds
records and lowers precision, whether or not the definition did anything. This
module therefore builds either arm from the same code path -- the flag-on run,
and a flag-off run that was already committed -- so the two are comparable and
the merge effect can be subtracted rather than argued about.

Merging rather than swapping, and the experiment-id rename, are both inherited
from ``build_union_entity_prepass`` and are there for the same reasons: a swap
can lose a gold row the base already recovers, and ``union_paper`` keeps the
*first* record for a colliding ``experiment_id``, so without the rename the
base's experiments would win and the overlay's outcomes would be attached to
experiments that never described them.
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
OVERLAY_ROOT = ROOT / "data/staging/extraction/codex_endpoint_definition_v1"
DEFAULT_OUTPUT = (
    ROOT / "data/staging/extraction/codex_union_vision_v3_endpoint_definition"
)
PAPER_IDS = [f"GP-{index:03d}" for index in range(1, 10)]
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
        {
            **row,
            "experiment_id": mapping.get(
                str(row.get("experiment_id")), row.get("experiment_id")
            ),
        }
        for row in result.get("outcomes", [])
    ]
    return renamed


def build(
    *,
    output_root: Path = DEFAULT_OUTPUT,
    base_root: Path = BASE_ROOT,
    overlay_root: Path = OVERLAY_ROOT,
    paper_ids: list[str] | None = None,
    purpose: str | None = None,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    paper_ids = paper_ids or PAPER_IDS
    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(base_root, output_root)

    scratch = scratch_root or (output_root / ".overlay")
    papers = []
    for paper_id in paper_ids:
        overlay = _load(overlay_root / paper_id)
        base = _load(base_root / paper_id)
        if overlay is None or base is None:
            # A paper the overlay never produced -- an ineligible verdict, a
            # rejected response -- keeps the base's records untouched, and says
            # so rather than being silently absent from the manifest.
            papers.append(
                {
                    "paper_id": paper_id,
                    "overlay": None if overlay is None else len(
                        overlay.get("outcomes", [])
                    ),
                    "base": None if base is None else len(base.get("outcomes", [])),
                    "merged": None if base is None else len(base.get("outcomes", [])),
                    "note": "overlay absent; base copied unchanged"
                    if overlay is None
                    else "base absent; nothing to merge onto",
                }
            )
            continue
        (scratch / paper_id).mkdir(parents=True, exist_ok=True)
        (scratch / paper_id / "final_result.json").write_text(
            json.dumps(
                rename_experiments(overlay, RENAME_SUFFIX), ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        merged = union_paper(paper_id, [base_root, scratch])
        if merged is None:
            continue
        (output_root / paper_id / "final_result.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        papers.append(
            {
                "paper_id": paper_id,
                "base": len(base.get("outcomes", [])),
                "overlay": len(overlay.get("outcomes", [])),
                "merged": len(merged.get("outcomes", [])),
            }
        )
    if scratch_root is None:
        shutil.rmtree(scratch, ignore_errors=True)

    total = sum(
        len(json.loads(path.read_text(encoding="utf-8")).get("outcomes", []))
        for path in sorted(output_root.glob("GP-*/final_result.json"))
    )
    manifest = {
        "union_version": "union-extraction-1.2.0",
        "purpose": purpose
        or (
            "Measure what endpoint_definition costs and buys. Not a shipped "
            "result root: the flag it measures ships off."
        ),
        "note": (
            f"{_repo_path(base_root)} with every paper of "
            f"{_repo_path(overlay_root)} merged over its own. Rebuild with "
            "`python -m src.extraction.build_union_endpoint_definition "
            f"--overlay-root {_repo_path(overlay_root)} --output-root "
            f"{_repo_path(output_root)} --confirm-write`."
        ),
        "base_root": _repo_path(base_root),
        "overlay_root": _repo_path(overlay_root),
        "experiment_id_suffix": RENAME_SUFFIX,
        "papers": papers,
        "total_outcomes": total,
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
    parser.add_argument("--purpose", default=None)
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Required: this rewrites a tracked result root in place.",
    )
    args = parser.parse_args()
    if not args.confirm_write:
        parser.error("--confirm-write is required; this rewrites tracked files")
    print(
        json.dumps(
            build(
                output_root=args.output_root,
                base_root=args.base_root,
                overlay_root=args.overlay_root,
                purpose=args.purpose,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
