"""Freeze approved v3 experiment maps before detailed extraction."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.output_paths import artifact_path

from .run_abstract_first import ROOT


RUN = ROOT / "data" / "staging" / "extraction" / "g1_v3_boundaries"
REVIEW = ROOT / "data" / "review" / "day5_g1_v3_boundary_review.jsonl"

# Canonical locations of the frozen artifacts. The actual write targets are
# resolved per call through `src.output_paths` so a test run redirects them
# instead of rewriting tracked files. See `freeze`.
FROZEN = ROOT / "data" / "staging" / "extraction" / "g1_v3_frozen_boundaries"
REPORT = ROOT / "reports" / "extraction" / "day5_g1_v3_frozen_boundaries.json"
FROZEN_PARTS = ("data", "staging", "extraction", "g1_v3_frozen_boundaries")
REPORT_PARTS = ("reports", "extraction")


CUSTOM_GP007 = [
    ("ACT in-vivo HIRI experiment", ["S03", "S04", "S05", "S06"], "ACT", "HIRI mice"),
    ("ACT in-vitro LSEC experiment", ["S03", "S04", "S05", "S06"], "ACT", "hypoxia-reoxygenation or lactate-stimulated LSECs"),
    ("LSEC-specific Micu1-overexpression experiment", ["S03", "S07"], "Micu1-overexpression virus", None),
    ("ACT plus siMicu1-LNP experiment", ["S03", "S07"], "ACT + siMicu1 LNP", None),
    ("ACT plus lactate-inhibitor experiment", ["S07"], "ACT + lactate inhibitor", None),
    ("ACT-derivative structural-analysis experiment", ["S03", "S08"], "ACT derivatives", None),
]


def freeze(*, output_root: Path | str | None = None) -> dict:
    """Freeze the human-approved v3 experiment boundaries.

    ``output_root`` overrides where the frozen boundary files and the report are
    written. When omitted the target comes from
    :func:`src.output_paths.output_root`, which resolves to the repository for a
    real run and to a scratch directory under pytest so the tracked artifacts
    are never rewritten by the test suite.
    """
    reviews = [json.loads(line) for line in REVIEW.read_text(encoding="utf-8").splitlines() if line.strip()]
    incomplete = [row["paper_id"] for row in reviews if row.get("boundary_decision") not in {"reader_a", "reader_b", "custom_required"} or not row.get("reviewer_reason")]
    if incomplete:
        raise ValueError(f"Incomplete boundary decisions: {incomplete}")
    frozen_dir = artifact_path(*FROZEN_PARTS, root=output_root, create_parents=False)
    frozen_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_path(*REPORT_PARTS, REPORT.name, root=output_root)
    frozen_papers = []
    for row in reviews:
        paper_id = row["paper_id"]
        if row["boundary_decision"] in {"reader_a", "reader_b"}:
            source_reader = row["boundary_decision"]
            source = json.loads((RUN / paper_id / f"{source_reader}.validated.json").read_text(encoding="utf-8"))
            experiments = source["experiments"]
        elif paper_id == "GP-007":
            source_reader = "human_custom"
            sentence_lookup = {item["sentence_id"]: item["text"] for item in json.loads((RUN / paper_id / "sentences.json").read_text(encoding="utf-8"))}
            experiments = []
            for index, (label, sentence_ids, treatment, model) in enumerate(CUSTOM_GP007, 1):
                experiments.append({
                    "reader_experiment_key": f"CUSTOM-{index:02d}",
                    "experiment_label": label,
                    "evidence_sentence_ids": sentence_ids,
                    "experiment_anchor_quote": sentence_lookup[sentence_ids[-1]],
                    "formulation_or_delivery_system_mention": "siMicu1 lipid nanoparticles" if "siMicu1" in label else None,
                    "payload_or_treatment_mention": treatment,
                    "biological_model_mention": model,
                    "recipient_cell_mention": "LSECs" if "LSEC" in label or "siMicu1" in label else None,
                    "therapeutic_target_mention": None,
                    "distinctness_reason": "Human-approved custom boundary; see boundary review rationale.",
                })
        else:
            raise ValueError(f"No custom-map implementation for {paper_id}")
        frozen = {
            "contract_version": "3.0.0",
            "paper_id": paper_id,
            "boundary_source": source_reader,
            "reviewer": row.get("reviewer"),
            "reviewed_at": row.get("reviewed_at"),
            "reviewer_reason": row.get("reviewer_reason"),
            "experiments": [
                {"experiment_id": f"{paper_id}-E{index:02d}", **experiment}
                for index, experiment in enumerate(experiments, 1)
            ],
        }
        (frozen_dir / f"{paper_id}.json").write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        frozen_papers.append({"paper_id": paper_id, "experiments": len(experiments), "source": source_reader})
    report = {"frozen_at": datetime.now(timezone.utc).isoformat(), "papers": frozen_papers, "status": "ready_for_experiment_scoped_extraction"}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Required: this rewrites tracked files in place.",
    )
    args = parser.parse_args()
    if not args.confirm_write:
        parser.error("--confirm-write is required; this rewrites tracked files")
    print(json.dumps(freeze(), indent=2))


if __name__ == "__main__":
    main()
