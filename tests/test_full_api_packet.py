"""Full local evidence view as an extraction-ready API packet."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.extraction.assess_outcome_complexity import (
    COMPACT_THRESHOLDS,
    FULL_CORPUS_THRESHOLDS,
    assess,
    detect_evidence_view,
)
from src.extraction.run_compact_one_call import load_packet
from src.rag.compact_api_packet import ApiEvidence, ApiSource, CompactApiPacket
from src.rag.compact_packet import (
    CompactEvidence,
    CompactEvidencePacket,
    DeduplicationSummary,
    SourceLocation,
    normalize_text,
)
from src.rag.full_api_packet import (
    build_all,
    build_full_packet,
    build_manifest,
    resolve_directory,
    write_outputs,
)
from src.rag.models import DocumentBlock


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ID = "GP-TEST"
RETRIEVAL_TEXT = (
    "LNP-mRNA treatment increased hepatocyte editing to 45% compared with control."
)
CORPUS_OUTCOME_TEXT = (
    "Figure 2 shows that LNP-mRNA increased luciferase expression 12-fold "
    "in hepatocytes compared with untreated liver."
)
CORPUS_BACKGROUND_TEXT = (
    "Lipid nanoparticles have been studied by many groups over the past decade."
)


def _review_packet(paper_id: str = PAPER_ID) -> CompactEvidencePacket:
    location = SourceLocation(
        chunk_id="block-retrieval",
        source_path="data/raw/paper.xml",
        source_kind="pmc_xml",
        block_type="paragraph",
        section="Results",
        subsection="Editing",
    )
    evidence = CompactEvidence(
        evidence_id=f"{paper_id}-E-0001",
        clause_ids=["clause-1"],
        chunk_ids=["block-retrieval"],
        text=RETRIEVAL_TEXT,
        normalized_text_sha256=hashlib.sha256(
            normalize_text(RETRIEVAL_TEXT).encode("utf-8")
        ).hexdigest(),
        field_tags=["outcomes", "payload"],
        field_groups=["outcome"],
        query_ids=["q-outcomes"],
        entity_types=["outcome"],
        experiment_candidate_ids=["EC-0001"],
        source_locations=[location],
    )
    return CompactEvidencePacket(
        paper_id=paper_id,
        source_retrieval_packet=f"{paper_id}.json",
        blocked_fields={},
        evidence=[evidence],
        deduplication=DeduplicationSummary(
            input_hits=1,
            unique_chunks=1,
            removed_chunk_duplicates=0,
            removed_normalized_passages=0,
            input_clauses=1,
            unique_evidence_items=1,
        ),
        packet_checksum="0" * 64,
    )


def _block(
    block_id: str,
    section_path: str,
    text: str,
    *,
    paper_id: str = PAPER_ID,
    block_type: str = "paragraph",
) -> DocumentBlock:
    return DocumentBlock(
        block_id=block_id,
        paper_id=paper_id,
        source_path="data/raw/paper.xml",
        source_kind="pmc_xml",
        section_path=section_path,
        block_type=block_type,
        text=text,
        char_start=0,
        char_end=len(text),
        parser="pmc-xml",
        parser_confidence=1.0,
    )


def _fixture_roots(tmp_path: Path, paper_id: str = PAPER_ID) -> tuple[Path, Path]:
    review_root = tmp_path / "review_packets"
    corpus_root = tmp_path / "corpus"
    review_root.mkdir(parents=True, exist_ok=True)
    corpus_root.mkdir(parents=True, exist_ok=True)
    (review_root / f"{paper_id}.json").write_text(
        _review_packet(paper_id).model_dump_json(indent=2),
        encoding="utf-8",
    )
    blocks = [
        _block("block-retrieval", "Results > Editing", RETRIEVAL_TEXT, paper_id=paper_id),
        _block("block-corpus-outcome", "Results > Expression", CORPUS_OUTCOME_TEXT, paper_id=paper_id),
        _block("block-corpus-background", "Introduction", CORPUS_BACKGROUND_TEXT, paper_id=paper_id),
    ]
    (corpus_root / f"{paper_id}.blocks.jsonl").write_text(
        "\n".join(block.model_dump_json() for block in blocks) + "\n",
        encoding="utf-8",
    )
    return review_root, corpus_root


def _built(tmp_path: Path) -> tuple[CompactApiPacket, Path, Path]:
    review_root, corpus_root = _fixture_roots(tmp_path)
    packet = build_full_packet(
        PAPER_ID,
        review_packet_root=review_root,
        corpus_root=corpus_root,
    )
    return packet, review_root, corpus_root


def test_written_packet_passes_load_packet_checksum_validation(tmp_path):
    packet, review_root, corpus_root = _built(tmp_path)
    manifest = build_manifest(
        packet,
        corpus_path=corpus_root / f"{PAPER_ID}.blocks.jsonl",
        compact_packet_root=tmp_path / "missing_compact",
    )
    output_root = tmp_path / "full_packets"
    packet_path, manifest_path = write_outputs(packet, manifest, output_root)

    assert packet_path.exists() and manifest_path.exists()
    reloaded = load_packet(PAPER_ID, output_root)
    assert reloaded.packet_checksum == packet.packet_checksum
    assert reloaded == packet


def test_corrupted_full_packet_is_rejected_by_load_packet(tmp_path):
    packet, _, corpus_root = _built(tmp_path)
    output_root = tmp_path / "full_packets"
    write_outputs(packet, build_manifest(packet), output_root)
    payload = json.loads((output_root / f"{PAPER_ID}.json").read_text(encoding="utf-8"))
    payload["evidence"][0]["text"] += " tampered"
    (output_root / f"{PAPER_ID}.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_packet(PAPER_ID, output_root)


def test_full_view_contains_every_parsed_block_not_only_retrieval_evidence(tmp_path):
    packet, _, _ = _built(tmp_path)
    texts = [row.text for row in packet.evidence]

    assert RETRIEVAL_TEXT in texts
    assert CORPUS_OUTCOME_TEXT in texts
    assert CORPUS_BACKGROUND_TEXT in texts
    corpus_source_chunks = {row.chunk_id for row in packet.sources}
    assert {"block-corpus-outcome", "block-corpus-background"} <= corpus_source_chunks


def test_corpus_evidence_degrades_to_two_field_tags_and_no_experiment_anchor(tmp_path):
    packet, _, _ = _built(tmp_path)
    corpus_rows = [
        row for row in packet.evidence if row.evidence_id.startswith(f"{PAPER_ID}-FC-")
    ]
    retrieval_rows = [
        row
        for row in packet.evidence
        if not row.evidence_id.startswith(f"{PAPER_ID}-FC-")
    ]

    assert corpus_rows and retrieval_rows
    assert {
        tag for row in corpus_rows for tag in row.retrieval_field_tags
    } <= {"outcomes", "full_corpus"}
    assert all(row.experiment_candidate_ids == [] for row in corpus_rows)
    outcome_row = next(row for row in corpus_rows if row.text == CORPUS_OUTCOME_TEXT)
    background_row = next(
        row for row in corpus_rows if row.text == CORPUS_BACKGROUND_TEXT
    )
    assert outcome_row.retrieval_field_tags == ["outcomes"]
    assert background_row.retrieval_field_tags == ["full_corpus"]
    # Retrieval evidence keeps its richer tags and its experiment anchor.
    assert retrieval_rows[0].retrieval_field_tags == ["outcomes", "payload"]
    assert retrieval_rows[0].experiment_candidate_ids == ["EC-0001"]


def test_manifest_reports_counts_tokens_and_compact_delta(tmp_path):
    packet, _, corpus_root = _built(tmp_path)
    compact_root = tmp_path / "compact_packets"
    compact_root.mkdir()
    compact_unsigned = {
        "packet_version": "compact-api-packet-1.0.0",
        "paper_id": PAPER_ID,
        "blocked_fields": [],
        "sources": [],
        "evidence": [
            ApiEvidence(
                evidence_id=f"{PAPER_ID}-E-0001",
                text=RETRIEVAL_TEXT,
                retrieval_field_tags=["outcomes"],
                source_ids=[],
            ).model_dump(mode="json", exclude_none=True)
        ],
    }
    checksum = hashlib.sha256(
        json.dumps(
            compact_unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    (compact_root / f"{PAPER_ID}.json").write_text(
        json.dumps({**compact_unsigned, "packet_checksum": checksum}),
        encoding="utf-8",
    )

    manifest = build_manifest(
        packet,
        review_packet=_review_packet(),
        corpus_path=corpus_root / f"{PAPER_ID}.blocks.jsonl",
        compact_packet_root=compact_root,
    )

    assert manifest["evidence_view"] == "full"
    assert manifest["counts"]["evidence"] == len(packet.evidence)
    assert manifest["counts"]["sources"] == len(packet.sources)
    assert manifest["counts"]["retrieval_evidence"] == 1
    assert manifest["counts"]["corpus_expanded_evidence"] >= 2
    assert manifest["counts"]["full_corpus_only_evidence"] >= 1
    tokens = manifest["estimated_input_tokens"]
    assert tokens["total"] == (
        tokens["prompt"] + tokens["response_schema"] + tokens["evidence_packet"]
    )
    comparison = manifest["compact_packet_comparison"]
    assert comparison["compact_evidence"] == 1
    assert comparison["added_evidence"] == len(packet.evidence) - 1
    assert comparison["added_evidence_packet_tokens"] > 0
    assert manifest["corpus_available"] is True


def test_missing_corpus_is_an_error_unless_explicitly_allowed(tmp_path):
    review_root, corpus_root = _fixture_roots(tmp_path)
    (corpus_root / f"{PAPER_ID}.blocks.jsonl").unlink()

    with pytest.raises(FileNotFoundError, match="parsed corpus blocks"):
        build_full_packet(
            PAPER_ID, review_packet_root=review_root, corpus_root=corpus_root
        )

    degraded = build_full_packet(
        PAPER_ID,
        review_packet_root=review_root,
        corpus_root=corpus_root,
        allow_missing_corpus=True,
    )
    assert len(degraded.evidence) == 1


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.rag.full_api_packet", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(REPO_ROOT),
            "OPENAI_API_KEY": "",
            "SENSENOVA_API_KEY": "",
        },
    )


@pytest.mark.parametrize("relative", [True, False])
def test_cli_accepts_relative_and_absolute_output_dir(tmp_path, relative):
    review_root, corpus_root = _fixture_roots(tmp_path)
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    output_dir = "packets_out" if relative else str(tmp_path / "absolute_out")

    completed = _run_cli(
        [
            "--output-dir",
            output_dir,
            "--review-packet-root",
            str(review_root),
            "--corpus-root",
            str(corpus_root),
            "--compact-packet-root",
            str(tmp_path / "missing_compact"),
        ],
        cwd=workdir,
    )

    assert completed.returncode == 0, completed.stderr
    resolved = (
        (workdir / output_dir) if relative else Path(output_dir)
    )
    assert (resolved / f"{PAPER_ID}.json").exists()
    assert (resolved / f"{PAPER_ID}.manifest.json").exists()
    assert (resolved / "manifest.md").exists()
    run_manifest = json.loads((resolved / "manifest.json").read_text(encoding="utf-8"))
    assert [row["paper_id"] for row in run_manifest["papers"]] == [PAPER_ID]
    assert json.loads(completed.stdout)["papers"][0]["evidence"] >= 3
    assert load_packet(PAPER_ID, resolved).paper_id == PAPER_ID


def test_build_all_writes_packet_and_manifests(tmp_path):
    review_root, corpus_root = _fixture_roots(tmp_path)
    output_root = tmp_path / "out"

    run_manifest = build_all(
        review_packet_root=review_root,
        corpus_root=corpus_root,
        output_root=output_root,
        compact_packet_root=tmp_path / "missing_compact",
    )

    assert run_manifest["evidence_view"] == "full"
    assert len(run_manifest["papers"]) == 1
    assert (output_root / "manifest.json").exists()
    assert (output_root / "manifest.md").exists()
    assert load_packet(PAPER_ID, output_root).paper_id == PAPER_ID


def test_resolve_directory_accepts_relative_and_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_directory("nested/out") == (tmp_path / "nested/out").resolve()
    assert resolve_directory(tmp_path / "abs") == (tmp_path / "abs").resolve()


# --- view-aware complexity thresholds ---------------------------------------


def _api_packet(evidence: list[ApiEvidence]) -> CompactApiPacket:
    return CompactApiPacket(
        paper_id=PAPER_ID,
        blocked_fields=[],
        sources=[
            ApiSource(
                source_id="S-1",
                chunk_id="block-1",
                source_path="data/raw/paper.xml",
                source_kind="pmc_xml",
                block_type="paragraph",
                section="Results",
            )
        ],
        evidence=evidence,
        packet_checksum="0" * 64,
    )


def _filler_evidence(count: int, tags: list[str]) -> list[ApiEvidence]:
    return [
        ApiEvidence(
            evidence_id=f"{PAPER_ID}-FC-{index:04d}",
            text=(
                f"LNP-mRNA treatment increased hepatocyte editing to {40 + index}% "
                "and reduced fibrosis compared with control."
            ),
            retrieval_field_tags=list(tags),
            source_ids=["S-1"],
        )
        for index in range(count)
    ]


def test_full_corpus_view_is_detected_from_evidence_tags(tmp_path):
    packet, _, _ = _built(tmp_path)
    assert detect_evidence_view(packet) == "full_corpus"
    assert detect_evidence_view(_api_packet(_filler_evidence(2, ["outcomes"]))) == "compact"


def test_compact_view_scoring_is_unchanged(tmp_path):
    """The frozen compact behaviour: same route, score, reasons, and counters."""
    packet = _api_packet(_filler_evidence(12, ["outcomes"]))
    assessment = assess(packet)

    assert assessment.route == "complex"
    assert assessment.complexity_score == 7
    assert assessment.reasons == [
        "eight_or_more_direct_outcome_groups",
        "multiple_endpoint_families",
        "four_or_more_intervention_endpoint_passages",
        "multiple_intervention_endpoint_families",
    ]
    assert assessment.candidate_groups == 12
    assert assessment.endpoint_families == 3
    assert assessment.visual_candidate_groups == 0
    # Explicitly requesting the compact profile must be identical to the default.
    assert assess(packet, thresholds=COMPACT_THRESHOLDS) == assessment
    assert assess(packet, evidence_view="compact") == assessment


def test_full_corpus_volume_alone_no_longer_forces_the_complex_route():
    packet = _api_packet(_filler_evidence(12, ["full_corpus"]))

    view_aware = assess(packet)
    with_compact_thresholds = assess(packet, thresholds=COMPACT_THRESHOLDS)

    assert with_compact_thresholds.route == "complex"
    assert view_aware.route == "simple"
    assert view_aware.complexity_score < with_compact_thresholds.complexity_score
    assert "full_corpus_view_thresholds" in view_aware.reasons


def test_full_view_still_reaches_complex_on_real_outcome_density():
    packet = _api_packet(
        _filler_evidence(40, ["outcomes"]) + _filler_evidence(5, ["full_corpus"])
    )
    assert assess(packet).route == "complex"


def test_unknown_evidence_view_is_rejected():
    with pytest.raises(ValueError, match="Unknown evidence view"):
        assess(_api_packet(_filler_evidence(1, ["outcomes"])), evidence_view="nope")


def test_full_corpus_thresholds_scale_only_volume_rules():
    assert FULL_CORPUS_THRESHOLDS.many_outcome_groups > COMPACT_THRESHOLDS.many_outcome_groups
    assert (
        FULL_CORPUS_THRESHOLDS.many_outcome_tagged_passages
        > COMPACT_THRESHOLDS.many_outcome_tagged_passages
    )
    assert FULL_CORPUS_THRESHOLDS.full_corpus_only_weight < 1.0
    assert (
        FULL_CORPUS_THRESHOLDS.many_endpoint_families
        == COMPACT_THRESHOLDS.many_endpoint_families
    )
    assert (
        FULL_CORPUS_THRESHOLDS.multiple_visual_locations
        == COMPACT_THRESHOLDS.multiple_visual_locations
    )
