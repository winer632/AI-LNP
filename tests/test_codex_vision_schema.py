"""The strict response schema the vision panel reader sends to the provider.

Two regressions live here. The provider rejects a request whose schema leaves
any object open, and ``to_strict_json_schema`` emits ``additionalProperties:
true`` for ``corrected_fragment`` because the contract types it as a free-form
mapping -- so the seal has to *force* the flag, not default it. Sealing that
same object shut without giving it a typed shape is the opposite failure: the
vision path can then only return an empty fragment, and an observation that
resolved has nothing to carry its value in.

Nothing here starts a subprocess or reaches the network; the one test that
exercises ``run_panel`` replaces ``subprocess.run`` outright.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from src.extraction.run_codex_vision import (
    _FRAGMENT_SCHEMA,
    _seal,
    _strict_vision_schema,
    run_panel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
# Real observations written by this module's own runs, kept under data/.
COMMITTED_OBSERVATIONS = [
    REPO_ROOT / "data/staging/extraction/codex_vision_v1/GP-004/observation.json",
    REPO_ROOT / "data/staging/extraction/codex_vision_v1/GP-006/observation.json",
]


def _raw_schema() -> dict:
    from openai.lib._pydantic import to_strict_json_schema

    from src.extraction.selective_vision_contracts import SelectiveVisionResponse

    return to_strict_json_schema(SelectiveVisionResponse)


def _objects(node, path="$"):
    """Yield every ``(path, node)`` the provider will read as a JSON object."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield path, node
        for key, value in node.items():
            yield from _objects(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _objects(value, f"{path}[{index}]")


def test_the_contract_really_does_produce_an_open_object():
    """Precondition for the seal: without it, one object comes back open.

    If this ever stops holding, the sealing tests below stop proving anything,
    so it is asserted rather than assumed.
    """
    open_objects = [
        path
        for path, node in _objects(_raw_schema())
        if node.get("additionalProperties") is True
    ]
    assert open_objects, (
        "to_strict_json_schema no longer emits an open object; the seal tests "
        "below are no longer exercising the case they were written for"
    )


def test_seal_forces_additional_properties_false_over_an_explicit_true():
    """A default would leave the provider's invalid_json_schema rejection in place."""
    sealed = _seal(
        {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "fragment": {"type": "object", "additionalProperties": True},
            },
        }
    )
    assert sealed["additionalProperties"] is False
    assert sealed["properties"]["fragment"]["additionalProperties"] is False


def _still_open(schema) -> dict:
    return {
        path: node.get("additionalProperties")
        for path, node in _objects(schema)
        if node.get("additionalProperties") is not False
    }


def test_sealing_alone_closes_every_object_in_the_contract_schema():
    """The seal has to stand on its own, not lean on the fragment being replaced.

    ``_strict_vision_schema`` happens to overwrite the one object the contract
    leaves open today. Any future free-form mapping on the contract would be
    sealed only by ``_seal`` itself, so ``_seal`` is checked directly against
    the real schema the provider would otherwise reject.
    """
    assert _still_open(_seal(_raw_schema())) == {}


def test_strict_vision_schema_leaves_no_object_open():
    assert _still_open(_strict_vision_schema()) == {}


def test_every_sealed_object_requires_all_of_its_properties():
    """Strict mode rejects a schema that declares an optional property."""
    sealed = _seal(
        {"type": "object", "properties": {"beta": {"type": "string"}, "alpha": {"type": "string"}}}
    )
    assert sealed["required"] == ["alpha", "beta"]

    for path, node in _objects(_strict_vision_schema()):
        properties = sorted(node.get("properties", {}))
        assert sorted(node.get("required", [])) == properties, path


@pytest.mark.parametrize(
    "observation_path", COMMITTED_OBSERVATIONS, ids=lambda p: p.parent.name
)
def test_strict_schema_accepts_the_real_committed_observations(observation_path):
    """The sealed schema must still describe the replies this route has produced.

    Both files were written by ``run_panel`` and both resolved, so both carry a
    populated ``corrected_fragment``. Sealing that object without giving it a
    typed shape makes every one of them invalid.
    """
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    jsonschema.validate(observation, _strict_vision_schema())


def test_corrected_fragment_can_express_the_fields_the_merge_step_reads():
    """The merge step reads values out of the fragment; the schema must allow them.

    ``merge_vision_observations.observation_to_outcome`` builds an outcome
    record from ``qualitative_outcome`` / ``outcome_value`` / ``outcome_unit``.
    A fragment schema that cannot carry those turns every resolved observation
    into an outcome whose fields are all missing.
    """
    fragment_schema = _strict_vision_schema()["properties"]["corrected_fragment"]
    typed = next(
        option
        for option in fragment_schema["anyOf"]
        if option.get("type") == "object"
    )
    for field in ("qualitative_outcome", "outcome_value", "outcome_unit"):
        assert field in typed["properties"], field
        assert field in typed["required"], field

    jsonschema.validate(
        {
            "qualitative_outcome": "GFP is present in LYVE-1-positive cells.",
            "outcome_value": 41.5,
            "outcome_unit": "%",
            "relationship": "colocalized",
        },
        _FRAGMENT_SCHEMA,
    )


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _stub_codex(monkeypatch, reply: str):
    """Answer the panel call with ``reply`` instead of running ``codex``."""
    seen: dict = {}

    def fake_run(command, **kwargs):
        seen["command"] = list(command)
        seen["env"] = kwargs.get("env", {})
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text(reply, encoding="utf-8")
        return _FakeCompleted()

    monkeypatch.setattr("src.extraction.run_codex_vision.subprocess.run", fake_run)
    return seen


def _panel_image(tmp_path: Path) -> Path:
    image = tmp_path / "panel.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n fake panel bytes")
    return image


def test_run_panel_records_a_contract_rejection_instead_of_accepting_it(
    tmp_path, monkeypatch
):
    """A reply that breaks the abstain rule must never be written as an observation.

    ``visually_estimated`` with ``disposition: resolved`` is the exact shape the
    abstain rule exists to stop, so it has to land as a rejected contract with
    no ``observation.json`` behind it.
    """
    reply = json.dumps(
        {
            "finding_id": "GO-002-fig2",
            "disposition": "resolved",
            "field_name": "qualitative_outcome",
            "corrected_fragment": {
                "qualitative_outcome": "roughly half the cells look positive",
                "outcome_value": 50.0,
                "outcome_unit": "%",
                "relationship": None,
            },
            "value_status": "visually_estimated",
            "supporting_evidence_ids": [],
            "figure_or_table": "Figure 2",
            "panel_or_table_cell": "a",
            "visible_support": "Green signal covers about half the field.",
            "derivation": None,
            "confidence": "low",
            "requires_human_review": False,
            "printed_labels": [],
            "object_id": None,
            "page_id": None,
            "image_sha256": None,
            "panel_label": None,
            "relationship": None,
        }
    )
    seen = _stub_codex(monkeypatch, reply)

    manifest = run_panel(
        paper_id="GP-004",
        finding_id="GO-002-fig2",
        image_path=_panel_image(tmp_path),
        claim="eGFP colocalises with F4/80.",
        caption="Figure 2. eGFP and F4-80 staining.",
        output_root=tmp_path / "runs",
    )

    assert manifest["contract_accepted"] is False
    assert manifest["contract_error"]
    assert manifest["disposition"] is None
    run_dir = tmp_path / "runs" / "GP-004" / "GO-002-fig2"
    assert (run_dir / "response.json").exists()
    assert not (run_dir / "observation.json").exists()

    # The route must not be able to fall back onto a billed API path.
    assert seen["env"]["OPENAI_API_KEY"] == ""
    assert manifest["openai_api_requests"] == 0


def test_run_panel_refuses_to_overwrite_an_existing_response(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "GP-004" / "GO-002-fig2"
    run_dir.mkdir(parents=True)
    (run_dir / "response.json").write_text("{}", encoding="utf-8")

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("codex must not be invoked for an existing response")

    monkeypatch.setattr("src.extraction.run_codex_vision.subprocess.run", explode)

    with pytest.raises(FileExistsError):
        run_panel(
            paper_id="GP-004",
            finding_id="GO-002-fig2",
            image_path=_panel_image(tmp_path),
            claim="eGFP colocalises with F4/80.",
            caption="",
            output_root=tmp_path / "runs",
        )
