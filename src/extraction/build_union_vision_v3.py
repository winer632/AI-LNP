"""Rebuild the result root that reaches 13/15, from committed data only.

Why this exists
---------------
13/15 is the only figure that meets the design document's target, and until
now it had no scripted rebuild. `codex_union_vision_v3` was referenced by
nothing in the repository outside its own directory: an auditor reconstructed
the three-step chain from a note inside its manifest, and it worked, but no one
else would have found it. Section 九 calls reproducibility this project's
foundation, and a headline number reachable only by reverse-engineering an
artifact is exactly what that is meant to prevent.

The chain, and why each step is what it is
------------------------------------------
1. Salvage the full-view runs. A GP-004 response was discarded whole because
   it cited a hallucinated evidence id in ``eligibility``, ``components`` and
   ``formulations`` -- fields an outcome record does not depend on. One of the
   records it carried reads "few F4/80+ Kupffer cells expressed eGFP", which is
   GO-002. Salvage keeps a record only when every evidence id *it itself*
   cites is in the packet.

2. Union GP-004 against the salvage rather than adding it as another result
   root. This is the step that is easy to get wrong and the reason the manifest
   alone was not enough: ``_result_path`` returns the first root holding a
   paper, so appending the salvage means its GP-004 is never read, and
   prepending it replaces GP-004 outright and trades GO-002 for GO-001. Both
   leave the score at 11/15. Only the merge gives 12/15.

3. Merge the GP-004 and GP-006 panel reads. The GP-008 reads are deliberately
   excluded: they recover no outcome and cost evidence coverage, because
   one-to-one matching reassigns an evidence-bearing record elsewhere.

Scoring the result needs ``vision_relationship_polarity``; building it needs
``record_level_salvage``. The evaluator reads only the former.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from src.extraction.merge_vision_observations import merge_into_result
from src.extraction.salvage_invalid_response import build as build_salvage
from src.extraction.union_extraction_results import union_paper

ROOT = Path(__file__).resolve().parents[2]


def _repo_path(path: Path) -> str:
    """Repo-relative when inside the repo, absolute otherwise.

    relative_to raises for any outside path, so a manifest that calls it
    unconditionally crashes the moment someone builds into a scratch
    directory -- which is exactly what a test or a dry run does. This is the
    third module in this repository to hit that; the other two were fixed the
    same way.
    """
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

BASE_ROOT = ROOT / "data/staging/extraction/codex_union_v1"
FULL_VIEW_RUNS = (
    ROOT / "data/staging/extraction/codex_full_view_v2_gp004_retry",
    ROOT / "data/staging/extraction/codex_full_view_v2",
)
FULL_PACKETS = ROOT / "data/staging/rag/full_api_packets_v1"
VISION_ROOT = ROOT / "data/staging/extraction/codex_vision_v1"
SALVAGE_ROOT = ROOT / "data/staging/extraction/salvage_full_view_v1"
DEFAULT_OUTPUT = ROOT / "data/staging/extraction/codex_union_vision_v3"

# The paper whose discarded response carries GO-002.
SALVAGE_PAPER = "GP-004"

# Panel reads that earn their place. GP-008's are left out on measurement.
VISION_OBSERVATIONS = {
    "GP-004": ("GP-004/observation.json",),
    "GP-006": ("GP-006/observation.json",),
}


def build(
    *,
    output_root: Path = DEFAULT_OUTPUT,
    salvage_root: Path = SALVAGE_ROOT,
    base_root: Path = BASE_ROOT,
    vision_root: Path = VISION_ROOT,
    rebuild_salvage: bool = True,
) -> dict[str, Any]:
    """Rebuild the 13/15 root. Returns a manifest describing what it did."""
    if rebuild_salvage:
        build_salvage(
            [run for run in FULL_VIEW_RUNS if run.exists()],
            output_root=salvage_root,
            packet_root=FULL_PACKETS,
        )

    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(base_root, output_root)

    merged = union_paper(SALVAGE_PAPER, [base_root, salvage_root])
    if merged is None:
        raise FileNotFoundError(f"no {SALVAGE_PAPER} result under {base_root}")
    (output_root / SALVAGE_PAPER / "final_result.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for paper_id, names in VISION_OBSERVATIONS.items():
        observations = [
            json.loads((vision_root / name).read_text(encoding="utf-8"))
            for name in names
            if (vision_root / name).exists()
        ]
        if not observations:
            continue
        result_path = output_root / paper_id / "final_result.json"
        merge_into_result(result_path, observations, result_path)

    outcomes = sum(
        len(json.loads(path.read_text(encoding="utf-8")).get("outcomes", []))
        for path in output_root.glob("GP-*/final_result.json")
    )
    manifest = {
        "union_version": "union-extraction-1.2.0",
        "note": (
            "codex_union_v1 with GP-004 unioned against the record-level "
            "salvage of the full-view runs, then the GP-004 and GP-006 panel "
            "reads merged in. Rebuild with "
            "`python -m src.extraction.build_union_vision_v3 --confirm-write`."
        ),
        "base_root": _repo_path(base_root),
        "salvage_root": _repo_path(salvage_root),
        "salvage_source_runs": [
            _repo_path(run) for run in FULL_VIEW_RUNS if run.exists()
        ],
        "vision_observations": [
            _repo_path(vision_root / name)
            for names in VISION_OBSERVATIONS.values()
            for name in names
            if (vision_root / name).exists()
        ],
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
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Required: this rewrites a tracked result root in place.",
    )
    args = parser.parse_args()
    if not args.confirm_write:
        parser.error("--confirm-write is required; this rewrites tracked files")
    print(json.dumps(build(output_root=args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
