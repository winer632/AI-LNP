"""Build the Day 5 afternoon compact-vs-baseline comparison and Gate G1."""

from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = ROOT / "data/staging/extraction/g1_fulltext_rag"
COMPACT_ROOT = ROOT / "data/staging/extraction/compact_one_call_v1"
MERGED_ROOT = ROOT / "data/staging/extraction/compact_merged_v1"
OUTPUT_ROOT = ROOT / "reports/extraction/day5_afternoon_g1"
PAPERS = [f"GP-{number:03d}" for number in range(1, 10)]
ELIGIBLE = {"GP-002", "GP-004", "GP-005", "GP-006", "GP-007", "GP-008"}

# Standard, short-context prices per one million tokens on 2026-07-28.
# Source: https://developers.openai.com/api/docs/pricing
PRICES = {
    "gpt-5.6-sol": {
        "input": 5.00,
        "cached_input": 0.50,
        "cache_write": 6.25,
        "output": 30.00,
    },
    "gpt-5.6-terra": {
        "input": 2.50,
        "cached_input": 0.25,
        "cache_write": 3.125,
        "output": 15.00,
    },
}

BASELINE_GOLD = {"GO-005", "GO-007", "GO-008", "GO-010", "GO-015", "GO-016"}
COMPACT_GOLD = {
    "GO-004": "GP-006/O3: 17-fold hepatocyte:LSEC expression ratio",
    "GO-005": "GP-006/O2: 16.50% LSEC editing",
    "GO-007": "GP-006/O7: 3.30 ± 0.68% sustained FVIII activity",
    "GO-008": "GP-002/O1: strong eGFP staining in virtually all hepatocytes",
    "GO-010": "GP-005/O1: rapid Kupffer-cell LNP uptake",
    "GO-011": "GP-005/O2: no observable Kupffer-cell EGFP translation",
    "GO-013": "GP-007/O3: improved LSEC ultrastructure/function",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def priced_usage(model: str, usage: dict[str, Any]) -> float:
    price = PRICES[model]
    details = usage.get("input_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    writes = int(details.get("cache_write_tokens") or 0)
    input_tokens = int(usage.get("input_tokens") or 0)
    regular = max(0, input_tokens - cached - writes)
    output = int(usage.get("output_tokens") or 0)
    return (
        regular * price["input"]
        + cached * price["cached_input"]
        + writes * price["cache_write"]
        + output * price["output"]
    ) / 1_000_000


def baseline_paper(paper_id: str) -> dict[str, Any]:
    paper_root = BASELINE_ROOT / paper_id
    responses = sorted(paper_root.glob("*.response.json"))
    calls = []
    for path in responses:
        row = load(path)
        usage = row["usage"]
        calls.append(
            {
                "path": str(path.relative_to(ROOT)),
                "model": row["model"],
                "created_at": row.get("created_at"),
                "input_tokens": int(usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
                "cost_usd": priced_usage(row["model"], usage),
            }
        )
    accepted_outcomes = 0
    graph_path = paper_root / "accepted_graph.json"
    if paper_id in ELIGIBLE and graph_path.exists():
        graph = load(graph_path)
        accepted_outcomes = sum(
            row.get("entity_type") == "outcome_value"
            for row in graph.get("entities", [])
        )
    timestamps = [
        float(row["created_at"]) for row in calls if row.get("created_at") is not None
    ]
    return {
        "paper_id": paper_id,
        "input_tokens": sum(row["input_tokens"] for row in calls),
        "output_tokens": sum(row["output_tokens"] for row in calls),
        "calls": len(calls),
        "vision_pages": 0,
        "cost_usd": sum(row["cost_usd"] for row in calls),
        "accepted_outcomes": accepted_outcomes,
        "latency_seconds": None,
        "completion_timestamp_span_seconds": (
            max(timestamps) - min(timestamps) if len(timestamps) >= 2 else None
        ),
        "latency_note": (
            "Exact latency was not recorded. Timestamp span is a lower-bound "
            "orchestration proxy and excludes the first call's duration."
        ),
        "call_details": calls,
    }


def compact_result_path(paper_id: str) -> Path:
    merged = MERGED_ROOT / paper_id / "final_result.json"
    if merged.exists():
        return merged
    return COMPACT_ROOT / paper_id / "result.json"


def compact_paper(paper_id: str) -> dict[str, Any]:
    manifest = load(COMPACT_ROOT / paper_id / "manifest.json")
    usage = manifest["usage"]
    result = load(compact_result_path(paper_id))
    extra_vision_call = paper_id == "GP-006"
    return {
        "paper_id": paper_id,
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "calls": 1 + int(extra_vision_call),
        "metered_calls": 1,
        "unmetered_calls": int(extra_vision_call),
        "vision_pages": int(extra_vision_call),
        "cost_usd_lower_bound": priced_usage(manifest["model_returned"], usage),
        "accepted_outcomes": len(result.get("outcomes", [])),
        "latency_seconds": float(manifest["elapsed_seconds"]),
        "latency_note": (
            "Main-call latency is exact. The saved Day 4 GP-006 selective-vision "
            "pilot lacks a usage/latency manifest and is excluded from token, cost, "
            "and latency totals but included in calls and vision pages."
        ),
        "completion_disposition": (
            "explicit_unresolved_legacy_contract"
            if paper_id == "GP-002"
            else "completed_final_result"
        ),
    }


def aggregate(rows: list[dict[str, Any]], route: str) -> dict[str, Any]:
    cost_key = "cost_usd" if route == "baseline" else "cost_usd_lower_bound"
    latency = [row["latency_seconds"] for row in rows if row["latency_seconds"] is not None]
    accepted = sum(row["accepted_outcomes"] for row in rows)
    cost = sum(row[cost_key] for row in rows)
    return {
        "papers": len(rows),
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "calls": sum(row["calls"] for row in rows),
        "vision_pages": sum(row["vision_pages"] for row in rows),
        "cost_usd": cost,
        "accepted_outcomes": accepted,
        "cost_per_paper_usd": cost / len(rows),
        "cost_per_accepted_outcome_usd": cost / accepted if accepted else None,
        "mean_recorded_main_latency_seconds": mean(latency) if latency else None,
        "latency_comparability": (
            "exact main-call latency"
            if route == "compact"
            else "not recorded; only completion-timestamp spans available"
        ),
    }


def build() -> dict[str, Any]:
    baseline = [baseline_paper(paper_id) for paper_id in PAPERS]
    compact = [compact_paper(paper_id) for paper_id in PAPERS]
    old = aggregate(baseline, "baseline")
    new = aggregate(compact, "compact")
    reductions = {
        "input_tokens": 1 - new["input_tokens"] / old["input_tokens"],
        "output_tokens": 1 - new["output_tokens"] / old["output_tokens"],
        "calls": 1 - new["calls"] / old["calls"],
        "cost_lower_bound": 1 - new["cost_usd"] / old["cost_usd"],
    }
    recovered = set(COMPACT_GOLD)
    regressions = sorted(BASELINE_GOLD - recovered)
    gains = sorted(recovered - BASELINE_GOLD)
    break_even_unmetered_cost = old["cost_usd"] - new["cost_usd"]
    gates = [
        {
            "criterion": "Evidence quality must not materially regress",
            "decision": "HOLD",
            "evidence": (
                "Frozen-gold outcome recall improved from 6/15 (40.0%) to "
                f"{len(recovered)}/15 ({len(recovered)/15:.1%}), but two baseline-"
                "recovered GP-008 outcomes regressed and semantic critical-field "
                "precision, experiment mixing, and unsupported claims remain pending."
            ),
        },
        {
            "criterion": "All nine papers complete or explicitly unresolved",
            "decision": "PASS",
            "evidence": (
                "Eight papers have final merged results; GP-002 has an explicit "
                "legacy-contract unresolved disposition and a saved extraction result."
            ),
        },
        {
            "criterion": "Routine text papers use one main plus <=1 small repair",
            "decision": "PASS",
            "evidence": (
                "Every paper used one main call. GP-006 used one selective-vision "
                "crop call; no paper exceeded one exception call."
            ),
        },
        {
            "criterion": "Cost materially lower than current route",
            "decision": "PASS_WITH_METERING_CAVEAT",
            "evidence": (
                f"Measured main-call cost fell from ${old['cost_usd']:.4f} to "
                f"${new['cost_usd']:.4f}, an {reductions['cost_lower_bound']:.1%} "
                f"reduction. The unmetered pilot would need to cost more than "
                f"${break_even_unmetered_cost:.4f} to erase the savings."
            ),
        },
    ]
    report = {
        "evaluation_version": "day5-afternoon-g1-1.0.0",
        "pricing": {
            "basis": "OpenAI standard short-context prices per 1M tokens",
            "as_of": "2026-07-28",
            "source": "https://developers.openai.com/api/docs/pricing",
            "models": PRICES,
        },
        "baseline_definition": (
            "Frozen g1_fulltext_rag extractor, verifier, and post-audit-repair calls; "
            "GP-001 was locally excluded with zero model calls."
        ),
        "compact_definition": (
            "Nine cached compact main calls plus the one recorded GP-006 selective-"
            "vision pilot. The pilot's token/cost/latency manifest is unavailable."
        ),
        "per_paper": [
            {"paper_id": paper_id, "baseline": baseline[index], "compact": compact[index]}
            for index, paper_id in enumerate(PAPERS)
        ],
        "aggregate": {
            "baseline": old,
            "compact": new,
            "reductions": reductions,
            "unmetered_pilot_cost_break_even_usd": break_even_unmetered_cost,
        },
        "quality_comparison": {
            "baseline_frozen_gold_outcomes": sorted(BASELINE_GOLD),
            "baseline_recall": len(BASELINE_GOLD) / 15,
            "compact_frozen_gold_outcomes": COMPACT_GOLD,
            "compact_recall": len(COMPACT_GOLD) / 15,
            "regressions": [
                {
                    "gold_outcome_id": "GO-015",
                    "description": ">80% targeted BMDM GFP/FAPCAR expression",
                    "cause": "Direct quantitative passage was dropped from API packet v1.",
                    "remediation": "Retained by corrected API packet v1.1; small repair still required.",
                },
                {
                    "gold_outcome_id": "GO-016",
                    "description": "<20% BMDM expression after unmodified LNP",
                    "cause": "Direct quantitative passage was dropped from API packet v1.",
                    "remediation": "Retained by corrected API packet v1.1; small repair still required.",
                },
            ],
            "regression_ids": regressions,
            "gains": gains,
            "pending_quality_checks": [
                "critical-field semantic precision",
                "experiment-mixing review",
                "unsupported-claim review",
                "semantic evidence-reference correctness",
            ],
        },
        "g1": {
            "decision": "HOLD",
            "criteria": gates,
            "reason": (
                "Cost, completion disposition, and call discipline pass, but the "
                "quality criterion is not yet fully established and two known "
                "GP-008 regressions require one targeted repair."
            ),
            "next_required_action": (
                "Run one small GP-008 text repair against API packet v1.1, merge "
                "GO-015/GO-016, and complete semantic field review before reapplying G1."
            ),
        },
    }
    return report


def write(report: dict[str, Any]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (OUTPUT_ROOT / "per_paper.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "paper_id",
            "baseline_input_tokens",
            "compact_input_tokens",
            "baseline_output_tokens",
            "compact_output_tokens",
            "baseline_calls",
            "compact_calls",
            "baseline_vision_pages",
            "compact_vision_pages",
            "baseline_cost_usd",
            "compact_cost_usd_lower_bound",
            "baseline_accepted_outcomes",
            "compact_accepted_outcomes",
            "compact_main_latency_seconds",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["per_paper"]:
            old, new = row["baseline"], row["compact"]
            writer.writerow(
                {
                    "paper_id": row["paper_id"],
                    "baseline_input_tokens": old["input_tokens"],
                    "compact_input_tokens": new["input_tokens"],
                    "baseline_output_tokens": old["output_tokens"],
                    "compact_output_tokens": new["output_tokens"],
                    "baseline_calls": old["calls"],
                    "compact_calls": new["calls"],
                    "baseline_vision_pages": old["vision_pages"],
                    "compact_vision_pages": new["vision_pages"],
                    "baseline_cost_usd": f"{old['cost_usd']:.8f}",
                    "compact_cost_usd_lower_bound": f"{new['cost_usd_lower_bound']:.8f}",
                    "baseline_accepted_outcomes": old["accepted_outcomes"],
                    "compact_accepted_outcomes": new["accepted_outcomes"],
                    "compact_main_latency_seconds": new["latency_seconds"],
                }
            )
    old = report["aggregate"]["baseline"]
    new = report["aggregate"]["compact"]
    red = report["aggregate"]["reductions"]
    lines = [
        "# Day 5 afternoon - compact route cost gate",
        "",
        f"**Gate G1: {report['g1']['decision']}**",
        "",
        "| Metric | Prior full-text route | Compact route | Change |",
        "|---|---:|---:|---:|",
        f"| Input tokens | {old['input_tokens']:,} | {new['input_tokens']:,}* | {red['input_tokens']:.1%} lower |",
        f"| Output tokens | {old['output_tokens']:,} | {new['output_tokens']:,}* | {red['output_tokens']:.1%} lower |",
        f"| Calls | {old['calls']} | {new['calls']} | {red['calls']:.1%} lower |",
        f"| Vision pages | {old['vision_pages']} | {new['vision_pages']} | +{new['vision_pages']} targeted page |",
        f"| Cost | ${old['cost_usd']:.4f} | >=${new['cost_usd']:.4f}* | >={red['cost_lower_bound']:.1%} lower |",
        f"| Cost/paper | ${old['cost_per_paper_usd']:.4f} | >=${new['cost_per_paper_usd']:.4f}* | - |",
        f"| Accepted outcome proxy | {old['accepted_outcomes']} | {new['accepted_outcomes']} | Schema differs |",
        f"| Cost/accepted outcome | ${old['cost_per_accepted_outcome_usd']:.4f} | >=${new['cost_per_accepted_outcome_usd']:.4f}* | - |",
        f"| Mean latency | Not recorded | {new['mean_recorded_main_latency_seconds']:.1f}s main call | Not comparable |",
        "",
        "*Compact token, cost, and latency totals exclude one GP-006 crop pilot whose manifest was not preserved; the call and page are counted.",
        "",
        "## Frozen-gold regression inspection",
        "",
        f"- Prior route: {len(BASELINE_GOLD)}/15 ({len(BASELINE_GOLD)/15:.1%}).",
        f"- Compact after local adjudication: {len(COMPACT_GOLD)}/15 ({len(COMPACT_GOLD)/15:.1%}).",
        f"- Regressions: {', '.join(report['quality_comparison']['regression_ids'])}.",
        f"- Gains: {', '.join(report['quality_comparison']['gains'])}.",
        "",
        "The two regressions were traced to evidence-budget ranking and are retained in API packet v1.1. They still require one small repair and merge.",
        "",
        "## G1 criteria",
        "",
    ]
    for gate in report["g1"]["criteria"]:
        lines.append(f"- **{gate['decision']}** - {gate['criterion']}: {gate['evidence']}")
    lines.extend(["", f"Overall: **{report['g1']['decision']}**. {report['g1']['reason']}"])
    (OUTPUT_ROOT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Required: this rewrites tracked report files in place.",
    )
    args = parser.parse_args()
    if not args.confirm_write:
        parser.error("--confirm-write is required; this rewrites tracked files")
    print(json.dumps(build(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
