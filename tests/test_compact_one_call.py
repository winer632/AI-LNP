import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.extraction.compact_prompt_v1 import PROMPT_VERSION, prompt_sha256
from src.extraction.run_compact_one_call import (
    DEFAULT_MAX_INPUT_TOKENS,
    PACKET_ROOT,
    PACKET_ROOT_BY_VIEW,
    SCHEMA_PATH,
    estimated_input_tokens,
    packet_root_for,
    request_fingerprint,
    run_one,
)
from src.rag.compact_api_packet import ApiEvidence, ApiSource, CompactApiPacket


class FakeResponse:
    id = "resp_test"
    model = "gpt-5.6-terra-test"
    output = []
    output_text = ""
    usage = SimpleNamespace(
        model_dump=lambda mode="json": {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
        }
    )

    def model_dump(self, mode="json"):
        return {"id": self.id, "model": self.model}


class FakeResponses:
    def __init__(self, parsed):
        self.calls = []
        self.parsed = parsed

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = FakeResponse()
        response.output_text = self.parsed.model_dump_json()
        return response


def empty_response(paper_id="GP-TEST"):
    from src.extraction.compact_contracts import CompactExtractionResponse

    return CompactExtractionResponse.model_validate(
        {
            "contract_version": "compact-1.1.0",
            "paper_id": paper_id,
            "eligibility": {
                "decision": "uncertain",
                "reason_codes": ["FULL_TEXT_REQUIRED"],
                "evidence_ids": [],
                "explanation": "No evidence was supplied in this test packet.",
            },
            "formulations": [],
            "components": [],
            "experiments": [],
            "outcomes": [],
            "unresolved_items": [],
        }
    )


def build_packet(paper_id="GP-TEST", evidence=None, sources=None):
    unsigned = {
        "packet_version": "compact-api-packet-1.0.0",
        "paper_id": paper_id,
        "blocked_fields": [],
        "sources": [
            row.model_dump(mode="json", exclude_none=True)
            for row in (sources or [])
        ],
        "evidence": [
            row.model_dump(mode="json", exclude_none=True)
            for row in (evidence or [])
        ],
    }
    checksum = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return CompactApiPacket.model_validate({**unsigned, "packet_checksum": checksum})


def write_packet(root: Path, paper_id="GP-TEST", evidence=None, sources=None):
    packet = build_packet(paper_id, evidence=evidence, sources=sources)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{paper_id}.json").write_text(packet.model_dump_json())
    return packet


def full_view_packet(paper_id="GP-TEST"):
    """A packet shaped like build_full_outcome_inventory.full_corpus_view output."""
    sources = [
        ApiSource(
            source_id="FC-S-0001",
            chunk_id="block-1",
            source_path="data/raw/paper.xml",
            source_kind="pmc_xml",
            block_type="paragraph",
            section="Results",
        )
    ]
    evidence = [
        ApiEvidence(
            evidence_id=f"{paper_id}-FC-000{index}",
            text=(
                f"LNP-mRNA treatment increased hepatocyte editing to {40 + index}% "
                "compared with control."
            ),
            retrieval_field_tags=["outcomes" if index % 2 else "full_corpus"],
            source_ids=["FC-S-0001"],
        )
        for index in range(4)
    ]
    return evidence, sources


def test_run_one_makes_exactly_one_structured_call_and_records_usage(tmp_path):
    packet_root = tmp_path / "packets"
    output_root = tmp_path / "output"
    write_packet(packet_root)
    responses = FakeResponses(empty_response())
    client = SimpleNamespace(responses=responses)

    manifest = run_one(
        "GP-TEST",
        model="gpt-5.6-terra",
        client=client,
        packet_root=packet_root,
        output_root=output_root,
    )

    assert len(responses.calls) == 1
    assert responses.calls[0]["store"] is False
    assert responses.calls[0]["service_tier"] == "default"
    assert len(responses.calls[0]["prompt_cache_key"]) <= 64
    assert responses.calls[0]["text"]["format"]["strict"] is True
    assert manifest["paid_api_requests"] == 1
    assert manifest["usage"]["total_tokens"] == 120
    assert manifest["eligibility"]["decision"] == "uncertain"
    assert (output_root / "GP-TEST" / "result.json").exists()


def test_run_one_refuses_duplicate_paid_call(tmp_path):
    packet_root = tmp_path / "packets"
    output_root = tmp_path / "output"
    write_packet(packet_root)
    responses = FakeResponses(empty_response())
    client = SimpleNamespace(responses=responses)
    run_one(
        "GP-TEST",
        model="gpt-5.6-terra",
        client=client,
        packet_root=packet_root,
        output_root=output_root,
    )

    with pytest.raises(FileExistsError, match="duplicate paid call"):
        run_one(
            "GP-TEST",
            model="gpt-5.6-terra",
            client=client,
            packet_root=packet_root,
            output_root=output_root,
        )
    assert len(responses.calls) == 1


def test_compact_view_is_the_default_and_records_its_view(tmp_path):
    packet_root = tmp_path / "packets"
    output_root = tmp_path / "output"
    write_packet(packet_root)
    responses = FakeResponses(empty_response())
    client = SimpleNamespace(responses=responses)

    manifest = run_one(
        "GP-TEST",
        model="gpt-5.6-terra",
        client=client,
        packet_root=packet_root,
        output_root=output_root,
    )

    request = json.loads(
        (output_root / "GP-TEST" / "request.json").read_text(encoding="utf-8")
    )
    assert manifest["evidence_view"] == "compact"
    assert request["evidence_view"] == "compact"
    assert packet_root_for() == PACKET_ROOT
    assert set(PACKET_ROOT_BY_VIEW) == {"compact", "full"}


def test_compact_request_fingerprint_is_frozen():
    """The compact prompt cache key must not shift when a view field is added."""
    packet = build_packet()
    legacy = hashlib.sha256(
        json.dumps(
            {
                "paper_id": packet.paper_id,
                "packet_checksum": packet.packet_checksum,
                "prompt_version": PROMPT_VERSION,
                "prompt_checksum": prompt_sha256(),
                "schema_checksum": hashlib.sha256(
                    SCHEMA_PATH.read_bytes()
                ).hexdigest(),
                "model": "gpt-5.6-terra",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert request_fingerprint(packet, "gpt-5.6-terra") == legacy
    assert request_fingerprint(packet, "gpt-5.6-terra", "compact") == legacy
    assert request_fingerprint(packet, "gpt-5.6-terra", "full") != legacy


def test_full_view_run_uses_its_own_packet_root_cache_key_and_manifest(tmp_path):
    full_root = tmp_path / "full_packets"
    output_root = tmp_path / "output"
    evidence, sources = full_view_packet()
    packet = write_packet(full_root, evidence=evidence, sources=sources)
    responses = FakeResponses(empty_response())
    client = SimpleNamespace(responses=responses)

    manifest = run_one(
        "GP-TEST",
        model="gpt-5.6-terra",
        client=client,
        packet_root=full_root,
        output_root=output_root,
        evidence_view="full",
    )

    assert len(responses.calls) == 1
    assert manifest["paid_api_requests"] == 1
    assert manifest["evidence_view"] == "full"
    assert manifest["packet_checksum"] == packet.packet_checksum
    assert manifest["estimated_input_tokens"]["total"] > 0
    fingerprint = manifest["request_fingerprint"]
    assert fingerprint == request_fingerprint(packet, "gpt-5.6-terra", "full")
    assert fingerprint != request_fingerprint(packet, "gpt-5.6-terra")
    assert responses.calls[0]["prompt_cache_key"] == fingerprint
    request = json.loads(
        (output_root / "GP-TEST" / "request.json").read_text(encoding="utf-8")
    )
    assert request["evidence_view"] == "full"
    assert request["packet_root"] == str(full_root)
    assert request["packet"]["evidence"][0]["evidence_id"].startswith("GP-TEST-FC-")
    # The full view's evidence is what the response may cite.
    sent_packet = json.loads(responses.calls[0]["input"][1]["content"])
    assert len(sent_packet["evidence"]) == len(evidence)


def test_oversized_input_is_refused_before_any_paid_call(tmp_path):
    full_root = tmp_path / "full_packets"
    output_root = tmp_path / "output"
    evidence, sources = full_view_packet()
    packet = write_packet(full_root, evidence=evidence, sources=sources)
    responses = FakeResponses(empty_response())
    client = SimpleNamespace(responses=responses)
    tokens = estimated_input_tokens(packet.model_dump(mode="json", exclude_none=True))

    with pytest.raises(ValueError, match="above the"):
        run_one(
            "GP-TEST",
            model="gpt-5.6-terra",
            client=client,
            packet_root=full_root,
            output_root=output_root,
            evidence_view="full",
            max_input_tokens=tokens["total"] - 1,
        )

    assert responses.calls == []
    assert not (output_root / "GP-TEST").exists()

    manifest = run_one(
        "GP-TEST",
        model="gpt-5.6-terra",
        client=client,
        packet_root=full_root,
        output_root=output_root,
        evidence_view="full",
        max_input_tokens=tokens["total"],
    )
    assert manifest["max_input_tokens"] == tokens["total"]
    assert manifest["estimated_input_tokens"] == tokens
    assert DEFAULT_MAX_INPUT_TOKENS >= tokens["total"]


def test_unknown_evidence_view_is_rejected(tmp_path):
    packet_root = tmp_path / "packets"
    write_packet(packet_root)
    client = SimpleNamespace(responses=FakeResponses(empty_response()))

    with pytest.raises(ValueError, match="Unknown evidence view"):
        run_one(
            "GP-TEST",
            model="gpt-5.6-terra",
            client=client,
            packet_root=packet_root,
            output_root=tmp_path / "output",
            evidence_view="everything",
        )
