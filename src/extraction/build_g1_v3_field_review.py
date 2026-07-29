"""Build a focused, saved-decision review packet from v3 field audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .run_abstract_first import ROOT


FIELDS = ROOT / "data" / "staging" / "extraction" / "g1_v3_fields"
SENTENCES = ROOT / "data" / "staging" / "extraction" / "g1_v3_boundaries"
BOUNDARIES = ROOT / "data" / "staging" / "extraction" / "g1_v3_frozen_boundaries"
PACKET = ROOT / "data" / "review" / "day5_g1_v3_field_review.jsonl"


def field_value(record: dict[str, Any], path: str) -> Any:
    current: Any = record
    for segment in path.replace("]", "").split("."):
        if "[" in segment:
            name, index = segment.split("[")
            current = current.get(name, [])[int(index)]
        elif isinstance(current, dict):
            current = current.get(segment)
        else:
            return None
    return current


def build() -> dict[str, Any]:
    previous = {}
    if PACKET.exists():
        previous = {
            row["review_id"]: row
            for line in PACKET.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in [json.loads(line)]
        }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for paper_dir in sorted(path for path in FIELDS.glob("GP-*") if path.is_dir()):
        paper_id = paper_dir.name
        source_path = SENTENCES / paper_id / "sentences.json"
        source = json.loads(source_path.read_text()) if source_path.exists() else []
        source_lookup = {row["sentence_id"]: row["text"] for row in source}
        boundary_map = json.loads((BOUNDARIES / f"{paper_id}.json").read_text())
        boundary_lookup = {
            row["experiment_id"]: row["evidence_sentence_ids"]
            for row in boundary_map.get("experiments", [])
        }
        experiments = {
            path.stem.replace(".validated", ""): json.loads(path.read_text())
            for path in paper_dir.glob("GP-*-E*.validated.json")
        }
        findings: list[dict[str, Any]] = []
        deterministic_path = paper_dir / "deterministic_audit.json"
        if deterministic_path.exists():
            for issue in json.loads(deterministic_path.read_text()):
                findings.append({
                    **issue,
                    "severity": "blocking",
                    "explanation": {
                        "quote_not_in_cited_sentence": "The claimed evidence quote is not verbatim in its cited approved sentence.",
                        "possibly_merged_outcomes": "The endpoint name may combine multiple distinct outcomes and should be checked.",
                    }.get(issue["issue_type"], issue["issue_type"]),
                })
        verifier_path = paper_dir / "verification.validated.json"
        if verifier_path.exists():
            verification = json.loads(verifier_path.read_text())
            for issue in verification["issues"]:
                if "correctly missing. No issue." in issue["explanation"]:
                    continue
                findings.append(issue)
        for issue in findings:
            if issue["field_name"] == "endpoint_name_reported":
                issue = {**issue, "field_name": "lnp_outcomes[0].endpoint_name_reported"}
            key = (issue["experiment_id"], issue["field_name"], issue["issue_type"])
            if key in seen:
                continue
            seen.add(key)
            experiment = experiments[issue["experiment_id"]]
            cited_ids: set[str] = set(boundary_lookup.get(issue["experiment_id"], []))
            corrective_id = issue.get("supporting_or_corrective_sentence_id")
            if corrective_id:
                cited_ids.add(corrective_id)
            if not cited_ids:
                cited_ids = set(source_lookup)
            approved_context = " ".join(
                f'{sentence_id}: {source_lookup[sentence_id]}'
                for sentence_id in sorted(cited_ids)
                if sentence_id in source_lookup
            )
            review_id = f"G1V3-{len(rows) + 1:04d}"
            prior = previous.get(review_id, {})
            rows.append({
                "review_id": review_id,
                "paper_id": paper_id,
                "entity_type": issue["experiment_id"],
                "field_name": issue["field_name"],
                "value": field_value(experiment, issue["field_name"]),
                "entity_context": experiment,
                "abstract": approved_context,
                "evidence_quotes": [quote for quote in [
                    issue.get("supporting_or_corrective_quote"),
                    field_value(experiment, issue["field_name"] + ".evidence_quote"),
                ] if quote],
                "preliminary_classification": f'{issue.get("severity", "review")} · {issue["issue_type"]}',
                "verifier_explanation": issue["explanation"],
                "human_decision": prior.get("human_decision", "pending"),
                "reviewer_reason": prior.get("reviewer_reason", ""),
                "reviewer": prior.get("reviewer", ""),
                "reviewed_at": prior.get("reviewed_at"),
            })
    PACKET.parent.mkdir(parents=True, exist_ok=True)
    with PACKET.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "packet": str(PACKET),
        "review_items": len(rows),
        "blocking": sum(row["preliminary_classification"].startswith("blocking") for row in rows),
        "review": sum(row["preliminary_classification"].startswith("review") for row in rows),
    }
    report = ROOT / "reports" / "extraction" / "day5_g1_v3_field_review_summary.json"
    report.write_text(json.dumps(summary, indent=2) + "\n")
    return summary



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
    print(json.dumps(build(), indent=2))


if __name__ == "__main__":
    main()
