"""Build conservative evidence audit, G1 metrics, and human review packet."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .run_abstract_first import ROOT, gold_inputs


GOLD = ROOT / "data" / "annotations" / "gold_v1"
BUNDLES = ROOT / "data" / "staging" / "extraction" / "abstract_first_v1"
RETRIES = ROOT / "data" / "staging" / "extraction" / "abstract_entity_retry_v1"
REPORT = ROOT / "reports" / "extraction" / "day5_g1_evidence_audit.json"
PACKET = ROOT / "reports" / "extraction" / "day5_g1_human_review.html"
PACKET_JSONL = ROOT / "data" / "review" / "day5_g1_human_review.jsonl"


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def load_csv(name: str) -> list[dict[str, str]]:
    with (GOLD / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def literal_in_source(value: Any, source: str) -> bool:
    needle = norm(value)
    return bool(needle and len(needle) >= 2 and needle in norm(source))


def extracted_facts(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = {row["evidence_id"]: row for row in bundle["evidence"]}
    facts = []
    for entity_type in ("formulations", "components", "experiments", "outcomes"):
        for record_index, record in enumerate(bundle[entity_type]):
            for field_name, field in record.items():
                if not isinstance(field, dict) or "value_status" not in field:
                    continue
                if field["value_status"] != "reported":
                    continue
                quotes = [evidence[eid]["evidence_text"] for eid in field["evidence_ids"] if eid in evidence]
                facts.append({"entity_type": entity_type[:-1], "record_index": record_index, "field_name": field_name, "value": field["value"], "evidence_quotes": quotes, "confidence": min((evidence[eid]["extraction_confidence"] for eid in field["evidence_ids"] if eid in evidence), default="low")})
    return facts


def retry_facts(paper_id: str) -> list[dict[str, Any]]:
    facts = []
    for entity in ("formulation", "component", "experiment", "outcome"):
        path = RETRIES / f"{paper_id}.{entity}.validated.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record_index, record in enumerate(payload["records"]):
            for field in record:
                if field["value_status"] == "reported":
                    facts.append({"entity_type": entity, "record_index": record_index, "field_name": field["field_name"], "value": field["value"], "evidence_quotes": [field["evidence_quote"]] if field["evidence_quote"] else [], "confidence": field["confidence"]})
    return facts


def build() -> dict[str, Any]:
    inputs = {item["paper_id"]: item for item in gold_inputs()}
    gold_tables = {name: load_csv(name + ".csv") for name in ("formulations", "components", "experiments", "outcomes")}
    gold_evidence = {row["evidence_id"]: row for row in load_csv("evidence.csv")}
    critical_fields = {
        "formulation": {"formulation_name", "composition_raw", "composition_basis", "np_ratio"},
        "component": {"component_name_reported", "component_role", "inchikey", "molar_percentage", "percentage_unit"},
        "experiment": {"delivery_recipient_cell", "therapeutic_target_cell", "cell_source", "species", "in_vitro_in_vivo", "payload_type", "payload_name", "reporter", "dose", "dose_unit", "route", "timepoint", "timepoint_unit", "assay", "comparator_type", "comparator_description"},
        "outcome": {"endpoint_family", "endpoint_name", "outcome_value", "outcome_unit", "normalization_basis", "uncertainty_value", "uncertainty_type", "qualitative_outcome", "value_status"},
    }
    review: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for paper_id, source in inputs.items():
        bundle_path = BUNDLES / f"{paper_id}.json"
        facts = extracted_facts(json.loads(bundle_path.read_text(encoding="utf-8"))) if bundle_path.exists() else retry_facts(paper_id)
        if source["eligible_records_expected"] and not facts:
            review.append({"paper_id": paper_id, "entity_type": "paper", "field_name": "model_response", "preliminary_classification": "invalid_model_response", "human_decision": "pending"})
            counts["invalid_model_response_papers"] += 1
        for fact in facts:
            quote_in_abstract = all(literal_in_source(q, source["abstract"]) for q in fact["evidence_quotes"]) and bool(fact["evidence_quotes"])
            value_in_quote = any(literal_in_source(fact["value"], q) for q in fact["evidence_quotes"])
            if quote_in_abstract and value_in_quote:
                classification = "literal_source_support"
                counts["source_supported"] += 1
            elif quote_in_abstract:
                classification = "semantic_support_requires_human_review"
                counts["needs_human_support_review"] += 1
            else:
                classification = "evidence_quote_not_verbatim_in_abstract"
                counts["unsupported_or_bad_quote"] += 1
            row = {"paper_id": paper_id, **fact, "quote_in_abstract": quote_in_abstract, "value_literal_in_quote": value_in_quote, "preliminary_classification": classification, "human_decision": "pending"}
            audit.append(row)
            if classification != "literal_source_support" or fact["confidence"] != "high":
                review.append(row)

        # Availability audit: conservative literal check; nonliteral cases are not called omissions automatically.
        related_formulations = [r for r in gold_tables["formulations"] if r["gold_paper_id"] == paper_id]
        formulation_ids = {r["gold_formulation_id"] for r in related_formulations}
        related_components = [r for r in gold_tables["components"] if r["gold_formulation_id"] in formulation_ids]
        related_experiments = [r for r in gold_tables["experiments"] if r["gold_paper_id"] == paper_id]
        experiment_ids = {r["gold_experiment_id"] for r in related_experiments}
        related_outcomes = [r for r in gold_tables["outcomes"] if r["gold_experiment_id"] in experiment_ids]
        for entity, rows in (("formulation", related_formulations), ("component", related_components), ("experiment", related_experiments), ("outcome", related_outcomes)):
            for gold_row in rows:
                for field_name, expected in gold_row.items():
                    if field_name not in critical_fields[entity] or expected == "":
                        continue
                    if literal_in_source(expected, source["abstract"]):
                        status = "gold_value_literal_in_abstract"
                    else:
                        location = gold_evidence.get(gold_row.get("evidence_id", ""), {}).get("evidence_location_type")
                        if location and location != "abstract":
                            status = "abstract_omission"
                        else:
                            status = "availability_needs_human_classification"
                            review.append({"paper_id": paper_id, "entity_type": entity, "field_name": field_name, "gold_value": expected, "abstract": source["abstract"], "preliminary_classification": status, "human_decision": "pending"})
                    counts[status] += 1

    # Conservative metrics: only literal source support is provisionally accepted; human rows remain outside numerator.
    extracted_total = len(audit)
    supported = counts["source_supported"]
    metrics = {
        "provisional_literal_precision": supported / extracted_total if extracted_total else None,
        "traceable_abstract_evidence_coverage": sum(r["quote_in_abstract"] for r in audit) / extracted_total if extracted_total else None,
        "human_review_rows": len(review),
        "g1_threshold": 0.90,
        "g1_status": "pending_human_approval",
        "metric_note": "Final precision/recall require human decisions for semantic support and abstract availability.",
    }
    result = {"metrics": metrics, "counts": dict(counts), "extracted_field_audit": audit}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    PACKET_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with PACKET_JSONL.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(review, 1):
            handle.write(json.dumps({"review_id": f"G1-{index:04d}", **row}, ensure_ascii=False) + "\n")
    table_rows = []
    for index, row in enumerate(review, 1):
        evidence = " | ".join(row.get("evidence_quotes", [])) or row.get("abstract", "")
        value = row.get("value", row.get("gold_value", ""))
        table_rows.append(f"<tr><td>G1-{index:04d}</td><td>{html.escape(row['paper_id'])}</td><td>{html.escape(row.get('entity_type',''))}</td><td>{html.escape(row.get('field_name',''))}</td><td>{html.escape(str(value))}</td><td>{html.escape(evidence)}</td><td>{html.escape(row['preliminary_classification'])}</td><td>□ Correct □ Incorrect □ Ambiguous/absent</td></tr>")
    PACKET.write_text("<!doctype html><meta charset='utf-8'><title>Day 5 G1 review</title><style>body{font:14px sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px;vertical-align:top}th{position:sticky;top:0;background:#fff}td:nth-child(6){max-width:520px}</style><h1>Day 5 G1 human review</h1><p>Review flagged semantic-support and source-availability decisions. Accepted values must be explicitly supported and correctly linked.</p><table><tr><th>ID</th><th>Paper</th><th>Entity</th><th>Field</th><th>Value</th><th>Evidence/abstract</th><th>Preliminary class</th><th>Decision</th></tr>" + "".join(table_rows) + "</table>", encoding="utf-8")
    return result



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
    print(json.dumps(build()["metrics"], indent=2))


if __name__ == "__main__":
    main()
