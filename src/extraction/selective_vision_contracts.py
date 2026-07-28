"""Contracts for the Day 4 selective figure/table vision route.

P2 -- "视觉观察", section 7 of
``docs/ai_lnp_recall_improvement_design_zh.html`` -- extends this route in two
ways:

1. an observation must name the printed text it read (``printed_labels``), so
   that "只接受印刷可见或确定性推导的数值" is checkable rather than merely
   asserted in the prompt; and
2. an observation is anchored to the *panel object* that P1's uniparse
   ingestion already located and cropped (``object_id`` / ``page_id`` /
   ``bbox`` / ``image_sha256``), instead of to a crop this route re-derives.

Two invariants keep that additive:

* every P2 field is optional with a default, so task and response artefacts
  written before P2 keep validating and keep loading from ``data/``; and
* every P2 *rule* is gated behind the ``local_vlm_vision`` feature flag
  (alias ``P2``), which is off by default. With the flag off this module
  accepts and rejects exactly what it accepted and rejected before P2, and
  :meth:`SelectiveVisionTask.unsigned_payload` signs exactly the same bytes.

The pre-P2 abstain rule in :meth:`SelectiveVisionResponse.enforce_visual_safety`
is unconditional and is deliberately *not* gated: a visually estimated value
always goes to a human, flag or no flag.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.config_flags import is_enabled
from src.extraction.compact_validation import ValidationFinding


P2_FLAG = "local_vlm_vision"

# Controlled vocabulary for the design document's ``relationship`` field. A
# free-text relationship would let the model smuggle a conclusion into an
# observation, which is the one thing this route must not allow.
VisualRelationship = Literal[
    "colocalized",
    "not_colocalized",
    "increased",
    "decreased",
    "unchanged",
    "present",
    "absent",
]

# Every relationship asserts a base relation together with a sign. This lives
# next to the vocabulary because it is a property of the vocabulary itself, not
# of any one consumer: "not_colocalized" denies exactly the relation
# "colocalized" affirms, and "absent" denies "present".
#
# A consumer must read the sign from here rather than infer it from the string.
# The names are designed to be read by people, so the negative member is
# spelled with the positive one inside it, and any text-level test sees the
# affirmation inside the denial -- tokenising "not_colocalized" yields
# {"not", "colocalized"}, which reads as a match for "colocalized".
RELATIONSHIP_POLARITY: dict[str, tuple[str, bool]] = {
    "colocalized": ("colocalized", True),
    "not_colocalized": ("colocalized", False),
    "present": ("present", True),
    "absent": ("present", False),
    "increased": ("increased", True),
    "decreased": ("increased", False),
    "unchanged": ("unchanged", True),
}


def p2_enabled() -> bool:
    """Whether the P2 visual-observation rules are switched on right now.

    Read per call, never cached at import time, so tests and gradual rollouts
    can flip ``AI_LNP_FLAG_LOCAL_VLM_VISION`` between calls.
    """
    return is_enabled(P2_FLAG)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CropBox(StrictModel):
    """PDF point coordinates, measured from the page's top-left corner."""

    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(gt=0)
    y1: float = Field(gt=0)

    @model_validator(mode="after")
    def positive_area(self) -> "CropBox":
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("crop box must have positive width and height")
        return self


PANEL_BLOCK_TYPES = ("figure", "table", "table_row", "caption")


class VisualPanel(StrictModel):
    """One figure/table object that uniparse already located and cropped.

    Built from a single ``DocumentBlock`` emitted by the P1 ingestion branch
    (``src/rag/ingestion.py::uniparse_blocks``), which is the only producer of
    ``bbox`` / ``image_ref`` / ``image_sha256`` / ``caption_text``. Field names
    follow the observation object in design-document section 7.

    ``bbox`` is uniparse page space and is *not* guaranteed to be PDF points,
    so it is carried as provenance only. It is never handed to the PyMuPDF
    renderer; the panel's own PNG is used instead.
    """

    object_id: str
    page_id: int = Field(ge=1)
    block_type: Literal["figure", "table", "table_row", "caption"]
    figure_or_table: str | None = None
    panel_label: str | None = None
    bbox: CropBox | None = None
    image_ref: str | None = None
    image_sha256: str | None = None
    caption_text: str | None = None
    parser: str | None = None

    @property
    def has_image(self) -> bool:
        """Whether this panel points at a checkable, already-cropped PNG."""
        return bool(self.image_ref and self.image_sha256)

    @classmethod
    def from_block(cls, block: Any) -> "VisualPanel | None":
        """Return the panel for one ``DocumentBlock``, or ``None`` if it is not one.

        Typed as ``Any`` so that this contract module does not have to import
        the RAG models. A block that is not a visual object, that has no page,
        or whose bbox does not survive :class:`CropBox` validation still yields
        a panel (or ``None``) rather than raising: a malformed geometry from a
        VLM-driven parser must degrade to "no crop box", never to a crash.
        """
        block_type = getattr(block, "block_type", None)
        page_number = getattr(block, "page_number", None)
        if block_type not in PANEL_BLOCK_TYPES:
            return None
        if not isinstance(page_number, int) or page_number < 1:
            return None
        raw_bbox = getattr(block, "bbox", None)
        bbox = None
        if raw_bbox is not None:
            try:
                bbox = CropBox(
                    x0=raw_bbox.x0, y0=raw_bbox.y0, x1=raw_bbox.x1, y1=raw_bbox.y1
                )
            except ValidationError:
                bbox = None
        return cls(
            object_id=block.block_id,
            page_id=page_number,
            block_type=block_type,
            figure_or_table=(
                getattr(block, "figure_number", None)
                or getattr(block, "table_number", None)
            ),
            bbox=bbox,
            image_ref=getattr(block, "image_ref", None),
            image_sha256=getattr(block, "image_sha256", None),
            caption_text=getattr(block, "caption_text", None),
            parser=getattr(block, "parser", None),
        )


class VisionReferral(StrictModel):
    """Explicit proof that text extraction left one table/figure unresolved."""

    referral_version: Literal["selective-vision-referral-1.0.0"]
    paper_id: str
    finding_id: str
    trigger: Literal["unresolved_table", "unresolved_figure"]
    reason: str = Field(min_length=1, max_length=500)
    source_id: str
    page_number: int = Field(ge=1)
    figure_or_table: str
    crop_box: CropBox | None = None
    caption_evidence_id: str
    referring_results_evidence_ids: list[str] = Field(min_length=1, max_length=3)
    methods_evidence_ids: list[str] = Field(default_factory=list, max_length=3)
    # P2: the uniparse object this referral points at, when P1 produced one.
    panel: VisualPanel | None = None


class VisionTextEvidence(StrictModel):
    evidence_id: str
    text: str
    source_ids: list[str]


class SelectiveVisionTask(StrictModel):
    task_version: Literal["selective-vision-task-1.0.0"]
    paper_id: str
    finding: ValidationFinding
    trigger: Literal["unresolved_table", "unresolved_figure"]
    trigger_reason: str
    source_pdf: str
    source_pdf_sha256: str
    page_number: int = Field(ge=1)
    figure_or_table: str
    crop_box: CropBox | None
    crop_path: str
    crop_sha256: str
    crop_evidence_id: str
    caption: VisionTextEvidence
    referring_results_passages: list[VisionTextEvidence] = Field(
        min_length=1, max_length=3
    )
    methods_context: list[VisionTextEvidence] = Field(
        default_factory=list, max_length=3
    )
    expected_schema_fragment: dict[str, Any]
    task_checksum: str
    # P2 provenance. Defaults reproduce the pre-P2 task exactly; see
    # `unsigned_payload` for why the defaults are also unsigned.
    crop_source: Literal["rendered_pdf_region", "uniparse_panel"] = (
        "rendered_pdf_region"
    )
    panel: VisualPanel | None = None

    def unsigned_payload(self) -> dict[str, Any]:
        """Return the body covered by ``task_checksum``.

        P2 fields are omitted while they still hold their pre-P2 defaults. That
        is what makes a task built with ``local_vlm_vision`` off checksum -- and
        therefore fingerprint, and therefore cache-hit -- byte-identically to a
        task built before P2 existed, while a task that really does carry a
        uniparse panel signs that panel.
        """
        payload = self.model_dump(mode="json", exclude={"task_checksum"})
        if self.panel is None:
            payload.pop("panel", None)
        if self.crop_source == "rendered_pdf_region":
            payload.pop("crop_source", None)
        return payload

    def text_payload(self) -> dict[str, Any]:
        """Return only the text and location context allowed by the timeline."""
        visual_location: dict[str, Any] = {
            "page_number": self.page_number,
            "figure_or_table": self.figure_or_table,
            "crop_box": (
                self.crop_box.model_dump(mode="json") if self.crop_box else None
            ),
            "crop_evidence_id": self.crop_evidence_id,
        }
        if self.panel is not None:
            # Added only when a panel exists, so a pre-P2 task serializes to the
            # same request bytes it always did.
            visual_location["panel"] = self.panel.model_dump(
                mode="json", exclude_none=True
            )
            visual_location["crop_source"] = self.crop_source
        return {
            "paper_id": self.paper_id,
            "finding": self.finding.model_dump(mode="json"),
            "trigger": self.trigger,
            "trigger_reason": self.trigger_reason,
            "visual_location": visual_location,
            "caption": self.caption.model_dump(mode="json"),
            "referring_results_passages": [
                row.model_dump(mode="json")
                for row in self.referring_results_passages
            ],
            "methods_context": [
                row.model_dump(mode="json") for row in self.methods_context
            ],
            "expected_schema_fragment": self.expected_schema_fragment,
        }


class SelectiveVisionResponse(StrictModel):
    """One structured observation of one figure/table object.

    Mapping to the observation object in design-document section 7: ``value``
    and ``unit`` live inside ``corrected_fragment`` (they are contract-typed
    there, so they cannot be free text), and ``abstained`` is derived rather
    than stored -- see :attr:`abstained`.
    """

    finding_id: str
    disposition: Literal["resolved", "missing", "ambiguous", "human_review"]
    field_name: str
    corrected_fragment: dict[str, Any] | None
    value_status: Literal[
        "exact_reported", "derived", "visually_estimated", "not_resolved"
    ]
    supporting_evidence_ids: list[str]
    figure_or_table: str
    panel_or_table_cell: str | None
    visible_support: str = Field(min_length=1, max_length=500)
    derivation: str | None
    confidence: Literal["high", "medium", "low"]
    requires_human_review: bool
    # ------------------------------------------------------------------ P2 --
    # Printed text read verbatim off the crop, e.g. ["F4/80", "GFP"]. This is
    # the evidence base for "只接受印刷可见或确定性推导的数值": an
    # exact_reported value that cannot name what it read is not a reading.
    printed_labels: list[str] = Field(default_factory=list, max_length=64)
    # Provenance back to the uniparse panel object this observation looked at.
    object_id: str | None = None
    page_id: int | None = Field(default=None, ge=1)
    image_sha256: str | None = None
    panel_label: str | None = None
    relationship: VisualRelationship | None = None

    @property
    def abstained(self) -> bool:
        """Design-document section 7 ``abstained``, derived instead of stored.

        Deriving it removes the failure mode where a response claims a value
        *and* claims to have abstained. A response abstains exactly when it
        commits no value or defers to a human.
        """
        return self.value_status == "not_resolved" or self.requires_human_review

    @model_validator(mode="after")
    def enforce_visual_safety(self) -> "SelectiveVisionResponse":
        if self.value_status == "visually_estimated":
            if self.disposition != "human_review" or not self.requires_human_review:
                raise ValueError("visually estimated values require human review")
            if not self.panel_or_table_cell:
                raise ValueError("visual estimates require panel_or_table_cell")
        if self.disposition == "resolved":
            if self.corrected_fragment is None:
                raise ValueError("resolved requires corrected_fragment")
            if self.value_status not in {"exact_reported", "derived"}:
                raise ValueError("resolved requires exact_reported or derived")
            if not self.panel_or_table_cell:
                raise ValueError("resolved requires panel_or_table_cell")
        elif self.disposition in {"missing", "ambiguous"}:
            if self.corrected_fragment is not None:
                raise ValueError("missing/ambiguous cannot return a correction")
            if self.value_status != "not_resolved":
                raise ValueError("missing/ambiguous requires not_resolved")
        if self.value_status == "exact_reported" and p2_enabled():
            # "Exact" must mean "printed on the page". Without the literal
            # labels there is nothing separating a transcribed value from a
            # confident guess, which is the failure mode the P2 rule exists to
            # prevent. Gated so responses recorded before the flag existed stay
            # loadable.
            if not self.printed_labels:
                raise ValueError(
                    "exact_reported requires printed_labels transcribed from the image"
                )
        if self.value_status == "derived" and not self.derivation:
            raise ValueError("derived values require a derivation")
        if self.disposition == "human_review" and not self.requires_human_review:
            raise ValueError("human_review disposition requires review")
        if p2_enabled():
            self._enforce_p2_rules()
        return self

    def _enforce_p2_rules(self) -> None:
        """Rules that exist only while ``local_vlm_vision`` is on.

        Kept out of :meth:`enforce_visual_safety`'s unconditional body so that
        with the flag off this class accepts and rejects exactly what it did
        before P2, including every response already written under ``data/``.
        """
        labels = [label.strip() for label in self.printed_labels]
        if any(not label for label in labels):
            raise ValueError("printed_labels must not contain blank entries")
        if self.value_status == "exact_reported" and not labels:
            raise ValueError(
                "exact_reported values require printed_labels: an exactly "
                "reported number must be quoted from printed text on the crop"
            )


# --------------------------------------------------------------------------- #
# Request-side projection of the P2 fields
#
# The runner asks for a *strict* JSON schema with additionalProperties=False, so
# a field the schema does not declare can never come back. These two helpers are
# what the runner splices in when the flag is on; keeping them here means the
# schema, the prompt and the validator that enforces them cannot drift apart.
# --------------------------------------------------------------------------- #

P2_RESPONSE_PROPERTIES: dict[str, Any] = {
    "printed_labels": {"type": "array", "items": {"type": "string"}},
    "object_id": {"type": ["string", "null"]},
    "page_id": {"type": ["integer", "null"]},
    "image_sha256": {"type": ["string", "null"]},
    "panel_label": {"type": ["string", "null"]},
    "relationship": {
        "type": ["string", "null"],
        "enum": [*VisualRelationship.__args__, None],
    },
}

P2_VISION_PROMPT_SUFFIX = """
Report printed_labels: every label, axis tick and annotation you actually read
off the crop, verbatim, one string each. An exact_reported value is only
allowed when printed_labels quotes the printed text you read it from; if the
number is not printed, do not report it as exact. Echo object_id, page_id and
image_sha256 from the panel you were given, name the panel in panel_label, and
use relationship only from the listed vocabulary."""


def p2_response_properties() -> dict[str, Any]:
    """Extra strict-schema properties for the current flag state."""
    return dict(P2_RESPONSE_PROPERTIES) if p2_enabled() else {}


def p2_prompt_suffix() -> str:
    """Prompt text for the current flag state; empty when P2 is off."""
    return P2_VISION_PROMPT_SUFFIX if p2_enabled() else ""
