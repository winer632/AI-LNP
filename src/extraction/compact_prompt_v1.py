"""Short scientific instructions for compact extraction route v1.

Four prompt versions live here.

``compact-prompt-1.1.0`` (:data:`COMPACT_EXTRACTION_PROMPT`)
    The frozen baseline. Its text and checksum are part of the compact request
    fingerprint and are pinned in ``config/extraction/compact_route_v1.yaml``,
    so it must not change.

``compact-prompt-1.2.1`` (:data:`CANDIDATE_SLOT_EXTRACTION_PROMPT`)
    The baseline verbatim, plus the candidate-slot rules. Used only when the
    ``candidate_slot_enforcement`` (P3) flag is on. Versioning the new prompt
    separately -- rather than editing the baseline in place -- is what keeps the
    flag-off request byte-identical, including its prompt cache key.

``compact-prompt-1.3.0`` / ``compact-prompt-1.3.1``
    Those two verbatim, plus :data:`CELL_IDENTITY_RULE`. Used only when the
    ``cell_line_identity`` flag is on. Same discipline again: a new version
    rather than an edit, so both prompts above stay byte-identical when the
    flag is off.

``compact-prompt-1.5.0`` / ``compact-prompt-1.5.1``
    Whichever text the flags above selected, plus :data:`ENTITY_TABLE_RULE`.
    Used only when the ``entity_resolution_prepass`` flag is on. Same
    discipline once more; with the flag off every prompt above is
    byte-identical to what shipped.

Call :func:`active_prompt` instead of importing a constant directly, so a call
site automatically picks up the text, version, and checksum that belong
together.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, NamedTuple

from src.config_flags import is_enabled
from src.extraction.compact_contracts import (
    ENDPOINT_DEFINITION,
    ENDPOINT_DEFINITION_FLAG,
)
from src.extraction.outcome_coverage_contracts import OutcomeCandidate


PROMPT_VERSION = "compact-prompt-1.1.0"

COMPACT_EXTRACTION_PROMPT = (
    "Extract only directly reported LNP evidence from the supplied packet. "
    "Use no outside knowledge. For every scientific field, return either a reported value "
    "with valid packet evidence IDs or missing with a short reason and no evidence IDs. "
    "Set eligibility to eligible only for original experimental LNP delivery of mRNA, siRNA, "
    "saRNA, or circRNA with evidence relevant to hepatocytes, Kupffer cells, LSECs, or hepatic "
    "stellate cells and a formulation-experiment-outcome link. Set a clearly failed criterion "
    "to ineligible and insufficient evidence to uncertain. Ineligible or uncertain papers "
    "must return empty extraction lists. "
    "Keep formulations, components, experiments, and outcomes separate and preserve their "
    "links. Do not infer hepatocytes from liver-level evidence. "
    "Do not mix facts from different experiments. "
    "Do not store payload as an LNP component. "
    "Do not convert a mechanism, hypothesis, or interpretation into a measured outcome. "
    "A reported negative result is an outcome, not missing. List unresolved ambiguities. "
    "Return only the structured response required by the supplied schema."
)

CANDIDATE_SLOT_PROMPT_VERSION = "compact-prompt-1.2.1"

CANDIDATE_SLOT_RULES = (
    " The second user message lists candidate_slots: outcome candidates enumerated "
    "locally from this same packet, each with a candidate_id and a short summary of "
    "the evidence that produced it. Return exactly one candidate_dispositions entry "
    "for every candidate_id supplied, and no entry for any other id. "
    "Use extracted only when at least one evidence_id listed for that candidate also appears in the evidence_ids of an outcome record you "
    "return; this is checked mechanically, and a candidate marked extracted whose evidence you never cite is rejected. "
    "not_an_outcome when the cited evidence is a method, hypothesis, or "
    "interpretation rather than a measured or reported result, and unresolved when "
    "the paper does report such an outcome but the packet does not give you a usable "
    "value. Every disposition other than extracted requires a short reason naming what "
    "is missing or why the evidence does not qualify; silence is not a valid "
    "disposition. A candidate slot is not permission to invent an outcome: if the "
    "evidence does not support a record, say so instead of extracting one."
)

CANDIDATE_SLOT_EXTRACTION_PROMPT = COMPACT_EXTRACTION_PROMPT + CANDIDATE_SLOT_RULES

CELL_IDENTITY_FLAG = "cell_line_identity"

# A named cell line and the population it stands for are two different facts,
# and a record that carries only the line name has dropped the biological one.
# This asks for both, and only from the packet: the paper is what says what a
# line is, so a line the packet never characterises stays uncharacterised
# rather than being filled in from what the model happens to know.
CELL_IDENTITY_RULE = (
    " When a record names a cell line, also name the cell type or population "
    "that line represents and the state it was in when measured, taking both "
    "from what this packet says about that line. Keep the line's own name "
    "alongside them rather than replacing it. When the packet does not say "
    "what a line represents, name the line alone: do not supply the cell type "
    "from outside knowledge."
)

CELL_IDENTITY_PROMPT_VERSION = "compact-prompt-1.3.0"
CELL_IDENTITY_EXTRACTION_PROMPT = COMPACT_EXTRACTION_PROMPT + CELL_IDENTITY_RULE

CELL_IDENTITY_SLOT_PROMPT_VERSION = "compact-prompt-1.3.1"
CELL_IDENTITY_SLOT_EXTRACTION_PROMPT = (
    CANDIDATE_SLOT_EXTRACTION_PROMPT + CELL_IDENTITY_RULE
)

# Enough of the grouped evidence to identify the slot, without re-sending the
# packet text the model already has in the first user message.
SLOT_SUMMARY_MAX_CHARS = 400


class PromptSelection(NamedTuple):
    """One prompt with the version and checksum that belong to it."""

    text: str
    version: str
    checksum: str


def prompt_sha256() -> str:
    return hashlib.sha256(COMPACT_EXTRACTION_PROMPT.encode("utf-8")).hexdigest()


def candidate_slot_prompt_sha256() -> str:
    return hashlib.sha256(
        CANDIDATE_SLOT_EXTRACTION_PROMPT.encode("utf-8")
    ).hexdigest()


def cell_identity_prompt_sha256(candidate_slot_enforcement: bool = False) -> str:
    text = (
        CELL_IDENTITY_SLOT_EXTRACTION_PROMPT
        if candidate_slot_enforcement
        else CELL_IDENTITY_EXTRACTION_PROMPT
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


ENTITY_RESOLUTION_FLAG = "entity_resolution_prepass"

# The difference from CELL_IDENTITY_RULE above is where the answer comes from.
# That rule asks the model to work out what a line is while it is extracting;
# this one tells it that the question has already been answered, separately,
# against this same packet, and that the answer is on the table it was handed
# with its evidence ids attached. The rule names no line and no population: it
# describes a table whose contents come from the paper.
ENTITY_TABLE_RULE = (
    " An entity_table may be supplied alongside the packet. It lists cell lines "
    "this packet names and, for each, the population that line represents and "
    "the state it was in, together with the packet evidence ids those were read "
    "from. It was derived from this same packet by a separate pass and adds no "
    "outside knowledge; every entry that could not be shown in its own cited "
    "evidence was removed before you saw it. When a record you return names a "
    "line the table covers, name that population and state alongside the line's "
    "own name rather than instead of it, and cite the table's evidence ids for "
    "that line together with whatever evidence the record already cites. A line "
    "the table does not cover is left as the bare line name. The entity_table "
    "states what a cell is, never what an experiment showed: it is not evidence "
    "for an outcome and never licenses a record the packet does not support."
)

ENTITY_TABLE_PROMPT_VERSION = "compact-prompt-1.5.0"
ENTITY_TABLE_SLOT_PROMPT_VERSION = "compact-prompt-1.5.1"

ENDPOINT_DEFINITION_PROMPT_VERSION = "compact-prompt-1.6.0"
ENDPOINT_DEFINITION_SLOT_PROMPT_VERSION = "compact-prompt-1.6.1"

# The words are the contract's, imported rather than restated, because the
# schema field and the instruction are the same definition and two copies of a
# definition drift. Nothing here names a population, a cell type, a marker or a
# finding: the rule is about what the field is for, and holds for a paper this
# project has never seen.
ENDPOINT_DEFINITION_RULE = (
    " The endpoint field of an outcome record is defined as follows. "
    + ENDPOINT_DEFINITION
)

INTERPRETIVE_OUTCOME_FLAG = "interpretive_outcome_admission"

INTERPRETIVE_PROMPT_VERSION = "compact-prompt-1.4.0"
CANDIDATE_SLOT_INTERPRETIVE_PROMPT_VERSION = "compact-prompt-1.4.1"

# What the two prompts above tell the model an outcome is:
#
#   "Extract only directly reported LNP evidence"
#   "Do not convert a mechanism, hypothesis, or interpretation into a measured
#    outcome."
#   "not_an_outcome when the cited evidence is a method, hypothesis, or
#    interpretation rather than a measured or reported result"
#
# Read together those draw the line at *measured*, and the annotation side does
# not: the frozen gold set records claims a paper states about its own results
# in prose. So the rule below narrows the prohibition to what it is actually
# for -- inventing a value the text does not give -- rather than deleting it.
# Deliberately written about the *kind* of statement and its provenance, with
# no endpoint, marker, cell type or finding named: naming one would make the
# prompt an answer key rather than a rule.
INTERPRETIVE_OUTCOME_RULES = (
    " Read the rule about interpretation as follows. A sentence stating what "
    "this paper's own experiments showed is a reportable outcome even when it "
    "is written as a summary, a conclusion, or an account of how an effect "
    "works rather than as a measurement. Report it as an outcome record whose "
    "qualitative_outcome carries that claim in the paper's own terms, with "
    "outcome_value missing unless the cited evidence itself states a value: "
    "the prohibition on converting an interpretation into a measured outcome "
    "forbids inventing a value or a unit for such a claim, not reporting the "
    "claim. This admits nothing the paper does not assert. A statement "
    "attributed to earlier work, an aim, a hypothesis, an expectation, a "
    "proposal, or a description of a method or an assay is still not an "
    "outcome. Cite the packet evidence that carries the statement, and assert "
    "no more than that evidence says; a record whose claim goes beyond its own "
    "cited evidence is worse than no record."
)

# Only meaningful alongside CANDIDATE_SLOT_RULES, which is the text that
# defines the three disposition codes in the first place.
INTERPRETIVE_DISPOSITION_RULES = (
    " Accordingly a candidate whose cited evidence states something this paper "
    "found is extracted, or unresolved when the packet gives no usable value "
    "for it, rather than not_an_outcome; reserve not_an_outcome for evidence "
    "that reports no finding of this paper at all."
)

INTERPRETIVE_EXTRACTION_PROMPT = (
    COMPACT_EXTRACTION_PROMPT + INTERPRETIVE_OUTCOME_RULES
)

CANDIDATE_SLOT_INTERPRETIVE_EXTRACTION_PROMPT = (
    CANDIDATE_SLOT_EXTRACTION_PROMPT
    + INTERPRETIVE_OUTCOME_RULES
    + INTERPRETIVE_DISPOSITION_RULES
)

# Enough of the grouped evidence to identify the slot, without re-sending the
# packet text the model already has in the first user message.
SLOT_SUMMARY_MAX_CHARS = 400


class PromptSelection(NamedTuple):
    """One prompt with the version and checksum that belong to it."""

    text: str
    version: str
    checksum: str


def prompt_sha256() -> str:
    return hashlib.sha256(COMPACT_EXTRACTION_PROMPT.encode("utf-8")).hexdigest()


def candidate_slot_prompt_sha256() -> str:
    return hashlib.sha256(
        CANDIDATE_SLOT_EXTRACTION_PROMPT.encode("utf-8")
    ).hexdigest()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()



def active_prompt(
    candidate_slots: bool = False,
    *,
    interpretive_outcome_admission: bool | None = None,
    cell_line_identity: bool | None = None,
    entity_resolution_prepass: bool | None = None,
    endpoint_definition: bool | None = None,
) -> PromptSelection:
    """Pick the prompt for this call from the flags that amend it.

    Four independent switches append to the frozen texts, never rewrite them,
    so a request made with all off is byte-identical to what shipped. They
    are checked in a fixed order and may all apply: the interpretive rule
    changes what counts as an outcome, the cell-identity rule changes how a
    record names its population from the packet alone, the entity-table
    rule tells the model that a separate grounded pass has already answered
    that question and handed it the answer, and the endpoint rule says what
    the endpoint field is for -- a question no version of this prompt has ever
    answered.
    """
    if candidate_slots:
        version = CANDIDATE_SLOT_PROMPT_VERSION
        text = CANDIDATE_SLOT_EXTRACTION_PROMPT
    else:
        version = PROMPT_VERSION
        text = COMPACT_EXTRACTION_PROMPT

    # An explicit argument beats the flag in both directions, so a caller can
    # pin a prompt regardless of deployment configuration and a test can
    # exercise both sides without touching the environment.
    interpretive = (
        is_enabled(INTERPRETIVE_OUTCOME_FLAG)
        if interpretive_outcome_admission is None
        else interpretive_outcome_admission
    )
    cell_identity = (
        is_enabled(CELL_IDENTITY_FLAG)
        if cell_line_identity is None
        else cell_line_identity
    )
    entity_table = (
        is_enabled(ENTITY_RESOLUTION_FLAG)
        if entity_resolution_prepass is None
        else entity_resolution_prepass
    )
    endpoint_defined = (
        is_enabled(ENDPOINT_DEFINITION_FLAG)
        if endpoint_definition is None
        else endpoint_definition
    )

    if interpretive:
        # Use the prompts the interpretive work assembled rather than
        # re-appending here: the slotted variant needs its own disposition
        # rules, not the same rules twice.
        if candidate_slots:
            version = CANDIDATE_SLOT_INTERPRETIVE_PROMPT_VERSION
            text = CANDIDATE_SLOT_INTERPRETIVE_EXTRACTION_PROMPT
        else:
            version = INTERPRETIVE_PROMPT_VERSION
            text = INTERPRETIVE_EXTRACTION_PROMPT

    if cell_identity:
        version = (
            CELL_IDENTITY_SLOT_PROMPT_VERSION
            if candidate_slots
            else CELL_IDENTITY_PROMPT_VERSION
        )
        text = text + CELL_IDENTITY_RULE

    if entity_table:
        version = (
            ENTITY_TABLE_SLOT_PROMPT_VERSION
            if candidate_slots
            else ENTITY_TABLE_PROMPT_VERSION
        )
        text = text + ENTITY_TABLE_RULE

    if endpoint_defined:
        version = (
            ENDPOINT_DEFINITION_SLOT_PROMPT_VERSION
            if candidate_slots
            else ENDPOINT_DEFINITION_PROMPT_VERSION
        )
        text = text + ENDPOINT_DEFINITION_RULE

    return PromptSelection(
        text=text,
        version=version,
        checksum=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _summary(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= SLOT_SUMMARY_MAX_CHARS:
        return collapsed
    return collapsed[: SLOT_SUMMARY_MAX_CHARS - 1].rstrip() + "…"


def candidate_slot_payload(
    candidates: Iterable[OutcomeCandidate],
) -> list[dict[str, object]]:
    """Render outcome candidates as the answerable slots sent with the packet.

    Only the identity of each candidate travels: its id, the endpoint family and
    figure/table that produced it, the evidence IDs it groups, and a truncated
    summary. The evidence text itself is already in the packet.
    """

    return [
        {
            "candidate_id": candidate.candidate_id,
            "endpoint_family": candidate.endpoint_family,
            "figure_or_table": candidate.figure_or_table,
            "route_hint": candidate.route_hint,
            "evidence_ids": list(candidate.evidence_ids),
            "summary": _summary(candidate.evidence_text),
        }
        for candidate in candidates
    ]
