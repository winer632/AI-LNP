"""Short scientific instructions for compact extraction route v1.

Four prompt versions live here, over two independent switches.

``compact-prompt-1.1.0`` (:data:`COMPACT_EXTRACTION_PROMPT`)
    The frozen baseline. Its text and checksum are part of the compact request
    fingerprint and are pinned in ``config/extraction/compact_route_v1.yaml``,
    so it must not change.

``compact-prompt-1.2.1`` (:data:`CANDIDATE_SLOT_EXTRACTION_PROMPT`)
    The baseline verbatim, plus the candidate-slot rules. Used only when the
    ``candidate_slot_enforcement`` (P3) flag is on. Versioning the new prompt
    separately -- rather than editing the baseline in place -- is what keeps the
    flag-off request byte-identical, including its prompt cache key.

``compact-prompt-1.3.0`` (:data:`INTERPRETIVE_EXTRACTION_PROMPT`) and
``compact-prompt-1.3.1``
(:data:`CANDIDATE_SLOT_INTERPRETIVE_EXTRACTION_PROMPT`)
    Those two, each with the interpretive-outcome rules appended. Used only
    when the ``interpretive_outcome_admission`` flag is on. Same discipline
    again: the two prompts above are appended to, never rewritten, so every
    request made with the flag off stays byte-identical.

Call :func:`active_prompt` instead of importing a constant directly, so a call
site automatically picks up the text, version, and checksum that belong
together.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, NamedTuple

from src.config_flags import is_enabled
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

INTERPRETIVE_OUTCOME_FLAG = "interpretive_outcome_admission"

INTERPRETIVE_PROMPT_VERSION = "compact-prompt-1.3.0"
CANDIDATE_SLOT_INTERPRETIVE_PROMPT_VERSION = "compact-prompt-1.3.1"

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
    candidate_slot_enforcement: bool = False,
    interpretive_outcome_admission: bool | None = None,
) -> PromptSelection:
    """Return the prompt for this call, with its own version and checksum.

    ``interpretive_outcome_admission`` defaults to whatever the flag resolves
    to, so a caller that knows nothing about it still records the prompt it
    actually sent. An explicit value wins, which is what lets a test pin both
    sides of the switch without touching the environment.
    """

    if interpretive_outcome_admission is None:
        interpretive_outcome_admission = is_enabled(INTERPRETIVE_OUTCOME_FLAG)

    if interpretive_outcome_admission:
        text, version = (
            (
                CANDIDATE_SLOT_INTERPRETIVE_EXTRACTION_PROMPT,
                CANDIDATE_SLOT_INTERPRETIVE_PROMPT_VERSION,
            )
            if candidate_slot_enforcement
            else (INTERPRETIVE_EXTRACTION_PROMPT, INTERPRETIVE_PROMPT_VERSION)
        )
        return PromptSelection(text, version, _sha256(text))

    if candidate_slot_enforcement:
        return PromptSelection(
            CANDIDATE_SLOT_EXTRACTION_PROMPT,
            CANDIDATE_SLOT_PROMPT_VERSION,
            candidate_slot_prompt_sha256(),
        )
    return PromptSelection(COMPACT_EXTRACTION_PROMPT, PROMPT_VERSION, prompt_sha256())


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
