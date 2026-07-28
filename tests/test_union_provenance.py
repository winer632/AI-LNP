"""The union has to be rebuildable from the repository, not just readable.

`codex_union_v1` is the artifact the 11/15 headline is measured on, and it is an
ensemble: nothing in it was extracted, every record was produced by one of five
input runs. That only means anything if those runs are here. They were not --
three of the five were cited by absolute path into a working directory that no
clone has, and 35 of the 51 records came from them, so the number could be
evaluated but not rebuilt.

These tests pin the repaired state: the manifest cites roots that exist, and
rebuilding from exactly those roots produces the committed artifact. See
docs/extraction/union_provenance.md for the provenance table and for the honest
9/15 the union scored without the three restored roots.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.extraction.evaluate_final_gold_dynamic import evaluate
from src.extraction.union_extraction_results import _cite, build

ROOT = Path(__file__).resolve().parents[1]
UNION_ROOT = ROOT / "data/staging/extraction/codex_union_v1"
MANIFEST = json.loads((UNION_ROOT / "union_manifest.json").read_text(encoding="utf-8"))
RUN_ROOTS = [ROOT / root for root in MANIFEST["run_roots"]]


def _committed(paper_id: str) -> dict:
    return json.loads(
        (UNION_ROOT / paper_id / "final_result.json").read_text(encoding="utf-8")
    )


def _paper_ids() -> list[str]:
    return [row["paper_id"] for row in MANIFEST["papers"]]


def test_every_run_root_the_manifest_cites_exists_in_the_repository():
    """A root outside the clone is a provenance claim nobody can check.

    The manifest used to cite three roots under /private/tmp: a reader could
    confirm the union scored 11/15 and had no way to confirm where 35 of its
    records came from.
    """
    missing = [root for root in MANIFEST["run_roots"] if not (ROOT / root).is_dir()]
    assert missing == [], f"union_manifest.json cites roots not in the repository: {missing}"


def test_the_vision_union_names_inputs_that_exist_too():
    """The same rule for the artifact derived from this one.

    codex_union_vision_v2 is codex_union_v1 plus two panel reads, and it is the
    12/15 figure. It cites its base and its observations by path, so the same
    citation has to hold.
    """
    manifest_path = (
        ROOT / "data/staging/extraction/codex_union_vision_v2/union_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cited = [manifest["base_root"], *manifest["vision_observations"]]
    missing = [path for path in cited if not (ROOT / path).exists()]
    assert missing == [], f"codex_union_vision_v2 cites paths not in the repository: {missing}"


def test_no_run_root_is_cited_by_absolute_path():
    """Relative to the repository root, or it is not a repository path."""
    absolute = [root for root in MANIFEST["run_roots"] if Path(root).is_absolute()]
    assert absolute == []


def test_a_root_outside_the_repository_is_still_recorded_in_full(tmp_path):
    """Relativising is for repository paths; anything else keeps its own path.

    An experimental root under /tmp is a legitimate thing to union over. What
    it must not do is come out looking like a repository path, because that is
    a citation a reader would go looking for and not find.
    """
    outside = tmp_path / "experiment_root"
    outside.mkdir()

    assert _cite(outside) == str(outside)
    assert _cite(ROOT / "data/staging/extraction/codex_union_v1") == (
        "data/staging/extraction/codex_union_v1"
    )


def test_every_source_run_a_record_names_is_one_of_the_cited_roots():
    """`source_run` is the run root's basename, so the two cannot drift apart."""
    cited = {root.name for root in RUN_ROOTS}
    named = {
        outcome["source_run"]
        for paper_id in _paper_ids()
        for outcome in _committed(paper_id)["outcomes"]
    }
    assert named <= cited, f"records name runs the manifest does not cite: {named - cited}"


def test_rebuilding_from_the_cited_roots_reproduces_the_manifest_byte_for_byte(tmp_path):
    build(RUN_ROOTS, output_root=tmp_path / "union")

    rebuilt = (tmp_path / "union" / "union_manifest.json").read_bytes()
    assert rebuilt == (UNION_ROOT / "union_manifest.json").read_bytes()


def test_rebuilding_from_the_cited_roots_reproduces_every_record(tmp_path):
    """Same records, same order, same origin -- modulo one known difference.

    `outcome_id` is not compared directly. The committed artifact was written
    before the builder learned to rename an id that collides with one already
    taken, so a rebuild today emits O1-2 where the artifact holds a second O1
    and records the original under `source_outcome_id`. Everything the
    evaluator reads is compared, and `test_the_rebuilt_union_measures_the_same`
    pins that the difference costs nothing.
    """
    build(RUN_ROOTS, output_root=tmp_path / "union")

    for paper_id in _paper_ids():
        committed = _committed(paper_id)
        rebuilt = json.loads(
            (tmp_path / "union" / paper_id / "final_result.json").read_text(
                encoding="utf-8"
            )
        )
        assert rebuilt["experiments"] == committed["experiments"], paper_id
        assert len(rebuilt["outcomes"]) == len(committed["outcomes"]), paper_id
        for rebuilt_row, committed_row in zip(
            rebuilt["outcomes"], committed["outcomes"]
        ):
            restored = dict(rebuilt_row)
            restored["outcome_id"] = restored.pop(
                "source_outcome_id", restored["outcome_id"]
            )
            assert restored == committed_row, paper_id


def test_the_rebuilt_union_measures_the_same(tmp_path):
    """The number a clone can rebuild is the number the repository reports."""
    build(RUN_ROOTS, output_root=tmp_path / "union")

    committed = evaluate(result_roots=[UNION_ROOT])
    rebuilt = evaluate(result_roots=[tmp_path / "union"])

    assert rebuilt["recovered"] == committed["recovered"] == 11
    assert rebuilt["total"] == committed["total"] == 15
    assert rebuilt["missing_gold_outcome_ids"] == committed["missing_gold_outcome_ids"]
    for key in ("precision", "evidence_accuracy"):
        if key in committed:
            assert rebuilt[key] == committed[key]


def test_the_repair_run_still_names_where_its_fragments_came_from():
    """Every path a merge report records has to resolve inside the clone.

    `repair_v2_merged` is the largest single contributor to the union, and its
    merge reports name the task and model fragment behind each recovered
    record by the absolute path they had when the merge ran. Those files are
    committed under data/staging/extraction/ with the same relative path, so
    the citation is checkable rather than decorative.
    """
    stage = ROOT / "data/staging/extraction"
    scratch_prefix = re.compile(r"^.*/scratchpad/")
    reports = sorted((stage / "repair_v2_merged").glob("GP-*/missing_record_merge_report.json"))
    assert reports, "the repair run is missing its merge reports"

    unresolved = []
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cited = [report["result_path"]]
        cited += [row["task_path"] for row in report["inputs"]]
        cited += [row["fragment_path"] for row in report["inputs"]]
        for path in cited:
            if not (stage / scratch_prefix.sub("", path)).exists():
                unresolved.append(path)
    assert unresolved == []


def test_each_restored_call_names_a_packet_the_repository_holds():
    """A run whose input packet is absent cannot be explained, only believed."""
    packets = {
        json.loads(path.read_text(encoding="utf-8")).get("packet_checksum"): path
        for path in (ROOT / "data/staging/rag").glob("*/GP-006.json")
    }
    for run in ("extraction_sc", "extraction_p3_compact"):
        request = json.loads(
            (
                ROOT / "data/staging/extraction" / run / "GP-006/request.json"
            ).read_text(encoding="utf-8")
        )
        checksum = request["packet_checksum"]
        assert checksum in packets, f"{run} was sent a packet the repository does not hold"
