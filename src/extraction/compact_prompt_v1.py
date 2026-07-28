"""Short scientific instructions for compact extraction route v1.

Two prompt versions live here.

``compact-prompt-1.1.0`` (:data:`COMPACT_EXTRACTION_PROMPT`)
    The frozen baseline. Its text and checksum are part of the compact request
    fingerprint and are pinned in ``config/extraction/compact_route_v1.yaml``,
    so it must not change.

``compact-prompt-1.2.1`` (:data:`CANDIDATE_SLOT_EXTRACTION_PROMPT`)
    The baseline verbatim, plus the candidate-slot rules. Used only when the
    ``candidate_slot_enforcement`` (P3) flag is on. Versioning the new prompt
    separately -- rather than editing the baseline in place -- is what keeps the
    flag-off request byte-identical, including its prompt cache key.

Call :func:`active_prompt` instead of importing a constant directly, so a call
site automatically picks up the text, version, and checksum that belong
together.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, NamedTuple

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


def active_prompt(candidate_slot_enforcement: bool = False) -> PromptSelection:
    """Return the prompt for this call, with its own version and checksum."""

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
