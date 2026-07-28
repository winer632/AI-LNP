from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Reuse the Day 8 multimodal bounding-box contract so PDF geometry has exactly
# one definition in the repository. That module only depends on pydantic, so no
# import cycle is created by referencing it from the RAG core models.
from src.extraction.pdf_multimodal_contracts import BoundingBox


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# "pdf_page" is retained for backwards compatibility with block corpora that
# were produced by the pre-uniparse PyMuPDF whole-page ingestion path.
BlockType = Literal[
    "title",
    "abstract",
    "paragraph",
    "table",
    "figure_caption",
    "pdf_page",
    "figure",
    "caption",
    "heading",
    "table_row",
]
SourceKind = Literal["pmc_xml", "pdf", "grobid_tei", "uniparse"]

# Verdict of the uniparse cross-check layer (design document, section 十).
# uniparse's table HTML is model output, not page truth, so a block that came
# from it records whether an independent read of the same pixels agrees.
#   verified     every number the parse asserts was found by another reader
#   unverified   no reader could establish it -- refer to a human
#   contradicted a confident independent reader did not produce a claimed value,
#                or the region's own text layer does not contain a sequence the
#                parse asserts
#   no_claims    the block asserts no numbers, so there is nothing to cross-check
#   no_geometry  the block asserts numbers but records no page/bbox to re-read
VerificationStatus = Literal[
    "verified", "unverified", "contradicted", "no_claims", "no_geometry"
]


class DocumentBlock(StrictModel):
    block_id: str
    paper_id: str
    source_path: str
    source_kind: SourceKind
    section_path: str
    block_type: BlockType
    text: str = Field(min_length=1)
    page_number: int | None = None
    xml_element_id: str | None = None
    table_number: str | None = None
    figure_number: str | None = None
    char_start: int = 0
    char_end: int
    parser: str
    parser_confidence: float = Field(ge=0, le=1)
    # The design document's section 9 identifier, e.g.
    # "GP-006-mmc1-p01-tabS2-r2". Carried *alongside* block_id, never instead
    # of it: block_id is cited by committed extraction results, so replacing it
    # would invalidate every one of them. None on every corpus built before
    # `structured_evidence_ids` was switched on, which is why it is optional.
    locator_id: str | None = None
    # Structured-ingestion fields. Every one of these must stay optional so that
    # block corpora written before uniparse ingestion keep validating, and so
    # that reverse construction from a retrieval hit (compact_packet.block_from_hit)
    # does not have to supply them.
    bbox: BoundingBox | None = None
    table_html: str | None = None
    caption_text: str | None = None
    image_ref: str | None = None
    image_sha256: str | None = None
    # Set by src/rag/uniparse_verification.py when the cross-check layer runs.
    # None means the check did not run for this block, which is not the same as
    # "it passed" -- that distinction is the whole point of the field.
    verification_status: VerificationStatus | None = None
    verification_method: str | None = None


class EntityCandidate(StrictModel):
    candidate_id: str
    paper_id: str
    block_id: str
    text: str
    entity_type: Literal[
        "lnp", "lipid_or_material", "payload", "cell", "tissue", "disease",
        "species", "route", "dose", "timepoint", "assay", "outcome",
        "gene_or_protein", "composition_signal", "formulation_label",
    ]
    char_start: int
    char_end: int
    detector: str
    confidence: float = Field(ge=0, le=1)


class RetrievalQuery(StrictModel):
    query_id: str
    paper_id: str
    question: str
    field_group: Literal[
        "formulation", "component", "payload", "experiment_boundary",
        "recipient_cell", "therapeutic_target", "model_context", "outcome",
    ]
    required_entity_types: list[str] = Field(default_factory=list)


class RetrievalHit(StrictModel):
    query_id: str
    block_id: str
    paper_id: str
    text: str
    section_path: str
    source_path: str
    source_kind: SourceKind | None = None
    block_type: BlockType | None = None
    page_number: int | None
    xml_element_id: str | None = None
    table_number: str | None = None
    figure_number: str | None = None
    locator_id: str | None = None
    bbox: BoundingBox | None = None
    table_html: str | None = None
    caption_text: str | None = None
    image_ref: str | None = None
    image_sha256: str | None = None
    lexical_score: float | None = None
    vector_score: float | None = None
    fused_score: float
    entity_types: list[str] = Field(default_factory=list)


class RetrievalPacket(StrictModel):
    query: RetrievalQuery
    hits: list[RetrievalHit]
    retrieval_methods: list[str]
    warnings: list[str]


class EvidenceGate(StrictModel):
    """Machine-checkable decision about whether extraction may proceed."""

    passed: bool
    paper_id: str
    query_id: str
    accepted_block_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
