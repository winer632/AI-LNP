import hashlib
import json
from pathlib import Path

import yaml

from src.extraction.compact_prompt_v1 import (
    COMPACT_EXTRACTION_PROMPT,
    PROMPT_VERSION,
    prompt_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "config" / "extraction" / "compact_route_v1.yaml"
BASELINE = (
    ROOT
    / "docs"
    / "extraction"
    / "baselines"
    / "fulltext_rag_v4.json"
)


def test_prompt_is_short_and_contains_critical_scientific_rules():
    assert len(COMPACT_EXTRACTION_PROMPT.split()) <= 160
    required = [
        "Do not infer hepatocytes from liver-level evidence.",
        "Do not mix facts from different experiments.",
        "Do not store payload as an LNP component.",
        "Do not convert a mechanism, hypothesis, or interpretation into a measured outcome.",
        "reported value",
        "valid packet evidence IDs",
        "missing",
        "Set eligibility to eligible only",
        "Ineligible or uncertain papers must return empty extraction lists.",
    ]
    for rule in required:
        assert rule in COMPACT_EXTRACTION_PROMPT


def test_prompt_hash_is_deterministic():
    assert prompt_sha256() == hashlib.sha256(
        COMPACT_EXTRACTION_PROMPT.encode("utf-8")
    ).hexdigest()


def test_route_versions_prompt_contract_packet_and_baseline():
    route = yaml.safe_load(ROUTE.read_text(encoding="utf-8"))
    assert route["route_version"] == "compact-route-1.1.0"
    assert route["status"] == "main_calls_attempted_pending_human_verification"
    assert route["prompt"]["version"] == PROMPT_VERSION
    assert route["prompt"]["sha256"] == prompt_sha256()
    assert route["response_contract"]["version"] == "compact-1.1.0"
    schema_path = ROOT / route["response_contract"]["generated_schema"]
    assert route["response_contract"]["schema_sha256"] == hashlib.sha256(
        schema_path.read_bytes()
    ).hexdigest()
    assert route["evidence_packet"]["version"] == "compact-packet-1.0.0"
    assert route["evidence_packet"]["implementation"] == (
        "src.rag.compact_packet.CompactEvidencePacket"
    )
    assert route["evidence_packet"]["implementation_status"] == (
        "day_2_morning_human_approved"
    )
    api_packet = route["evidence_packet"]["api_serialization"]
    assert api_packet["version"] == "compact-api-packet-1.0.0"
    assert api_packet["implementation"] == (
        "src.rag.compact_api_packet.CompactApiPacket"
    )
    assert api_packet["evidence_budget_estimated_tokens"] == 16000
    assert api_packet["budget_status"] == "human_approved"
    # Opened once the gate's own requirement was met: the route is implemented
    # and gold-evaluated. The evidence that satisfied it is recorded in the
    # gate so this assertion cannot drift away from its justification.
    assert route["activation_gate"]["active"] is True
    assert route["activation_gate"]["requirement_met"]["union_recall"] == "11/15"
    assert route["baseline"]["manifest"] == str(
        BASELINE.relative_to(ROOT)
    )


def test_frozen_baseline_checksums_match_files():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    for key in ("runner", "response_contract"):
        item = baseline[key]
        actual = hashlib.sha256(
            (ROOT / item["path"]).read_bytes()
        ).hexdigest()
        assert actual == item["sha256"]
