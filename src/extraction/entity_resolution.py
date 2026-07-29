"""Resolve what a named cell line *is*, from the paper's own words, before extraction.

Why this exists
---------------
An extraction record that names only a cell line has kept a catalogue fact and
dropped the biological one. A reader who does not already know the line cannot
tell which population was measured, and an annotation that names the population
instead of the line describes the same experiment in words the record never uses.

Asking the extraction prompt for it does not work when the paper never says it in
one sentence. The fact is usually a *join* across two places a single reading pass
does not put side by side -- a reagent/resource table row and a cell-culture
sentence, sharing a supplier and a species. So this runs first, over exactly the
evidence where that join lives, and hands the answer forward.

The discipline
--------------
This module is the reason the capability is defensible rather than dangerous. A
wrong "line -> population" mapping contaminates every downstream record that names
the line and is harder to notice than an omission, because it looks like a fact.
So the rules here are deliberately harsher than "the model said so":

1. **Every mapping cites packet evidence ids, and those ids must exist.** A mapping
   with no citation, or citing an id this packet does not contain, is not written.
2. **Every word of an asserted field must be findable in the text it cites.** The
   population and state are checked term by term against the union of the cited
   evidence texts. A field that asserts a word the cited sentences do not contain
   is dropped and the drop is recorded. This is what stops the model from filling
   in what it happens to know about a well-known line: outside knowledge has no
   sentence to point at.
3. **A mapping left with nothing to say is discarded.** A bare line name is what
   the record already had.

Rule 2 is the one that makes the two-sentence join legal without making guessing
legal. Terms are checked against the *union* of the cited texts, so an answer that
needs a table row and a sentence together is grounded when it cites both -- and an
answer that needs a fact neither sentence contains is not grounded no matter how
many ids it cites.

Nothing in this module names a cell line, a cell type, a supplier or a species.
The selectors below key on document structure (tables, methods sections) and on
provenance/culture vocabulary that any wet-lab paper uses; the answers come from
the paper.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from src.rag.compact_api_packet import ApiEvidence, ApiSource, CompactApiPacket


ENTITY_TABLE_VERSION = "entity-table-1.0.0"


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CellLineIdentity(StrictModel):
    """One line, and what the paper says it is.

    ``population`` and ``state`` are nullable on purpose. "The packet does not
    say" is a correct answer and has to be expressible, or the only way to fill
    the field is to invent it.
    """

    line_name: str
    population: str | None = None
    state: str | None = None
    species: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    derivation: str | None = None


class EntityResolutionResponse(StrictModel):
    """What the pre-pass model is asked to return, before grounding."""

    paper_id: str
    cell_lines: list[CellLineIdentity] = Field(default_factory=list)
    notes: str | None = None


class GroundedEntityTable(StrictModel):
    """What survives grounding, and therefore what travels into extraction."""

    entity_table_version: str = ENTITY_TABLE_VERSION
    paper_id: str
    cell_lines: list[CellLineIdentity] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Text normalisation and term checking
# --------------------------------------------------------------------------- #

# Words that carry no assertion. A field is not made ungrounded by an article.
# Deliberately short and generic: this is grammar, not biology. Nothing that
# names a cell, a state, a tissue or an organism belongs here, because dropping
# such a word would be dropping the very claim being checked.
_GRAMMAR_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
        "from", "with", "by", "as", "is", "are", "was", "were", "be", "been",
        "that", "this", "these", "those", "it", "its", "which", "into", "via",
        "per", "line", "lines",
    }
)

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """NFKC, casefold, collapse whitespace. The same rule the packet ids use."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def terms(text: str | None) -> set[str]:
    """Content terms of an asserted field.

    Single characters are dropped: they are grammar (``a``), or the fragment a
    hyphenated product code leaves behind when it is split on its separator, and
    requiring a bare digit to appear would fail on nothing and pass on
    everything.
    """
    if not text:
        return set()
    return {
        token
        for token in _TOKEN.findall(normalize(text))
        if len(token) > 1 and token not in _GRAMMAR_WORDS
    }


def _haystack_terms(texts: Iterable[str]) -> set[str]:
    """Every word of the cited text, plus its singular/plural counterpart.

    The only morphology allowed. A paper writes "the mouse cell lines were
    obtained from" and an answer about one of them says "cell line"; refusing
    that is refusing English grammar, not refusing a guess. Inflection cannot
    turn one entity into another, so this cannot launder an unsupported claim --
    whereas stemming across parts of speech could, which is why it stops here.
    """
    joined = " ".join(texts)
    found = {token for token in _TOKEN.findall(normalize(joined))}
    variants = set()
    for token in found:
        variants.add(token + "s")
        if token.endswith("s") and len(token) > 2:
            variants.add(token[:-1])
    return found | variants


# --------------------------------------------------------------------------- #
# Selecting the evidence the join lives in
# --------------------------------------------------------------------------- #

# Provenance and culture vocabulary. Every wet-lab paper that uses a cell line
# says where it came from and how it was grown, in words like these. None of
# them names an entity.
_PROVENANCE = re.compile(
    r"\b(?:purchas\w*|procur\w*|obtain\w*|acquir\w*|sourc\w*|suppli\w*|"
    r"provided|donated|gift\w*|deposit\w*|"
    r"cell\s+bank|cell\s+line|cell\s+lines|catalog\w*|catalogue|"
    r"cultur\w*|passag\w*|maintain\w*|grown|seeded|"
    r"authenticat\w*|mycoplasma|str\s+profil\w*|"
    r"derived\s+from|isolated\s+from|immortali\w*|"
    r"differentiat\w*|induc\w*|stimulat\w*|activat\w*|polari\w*|"
    r"transduc\w*|enrich\w*|sorted|selected\s+for)\b",
    re.I,
)

# Section labels a parser produces for the parts of a paper that say what the
# materials were. Structure, not content.
_METHODS_SECTION = re.compile(
    r"\b(?:method|methods|material|materials|reagent|reagents|resource|"
    r"resources|experimental\s+model|experimental\s+models|cell\s+cultur\w*|"
    r"supplement\w*|supporting\s+information|key\s+resources?)\b",
    re.I,
)

# A cell line is named like a product code: a short alphabetic stem, an optional
# separator, and digits. Requiring a nearby "cell"/"cells"/"line"/"clone" is what
# keeps figure numbers and antibody catalogue codes out. This finds *candidates*
# for the model to read; it never decides what one is.
_LINE_SHAPED = re.compile(
    r"(?<![\w-])[A-Z][A-Za-z]{0,7}[-‐‑‒–/]?\d{1,4}[A-Za-z]?(?![\w-])"
)
_CELL_WORD = re.compile(r"\b(?:cell|cells|line|lines|clone|clones|monocytes?)\b", re.I)
_LINE_WINDOW = 60


def has_line_shaped_mention(text: str) -> bool:
    """True when the text names something shaped like a cell line, near a cell word."""
    for match in _LINE_SHAPED.finditer(text):
        window = text[
            max(0, match.start() - _LINE_WINDOW) : match.end() + _LINE_WINDOW
        ]
        if _CELL_WORD.search(window):
            return True
    return False


_STRUCTURED_BLOCK_TYPES = frozenset(
    {"table", "table_row", "table_cell", "caption", "figure_caption"}
)


def identity_evidence_score(
    evidence: ApiEvidence, sources: dict[str, ApiSource]
) -> tuple[int, list[str]]:
    """Rank one evidence item as input to the pre-pass, and say why.

    Additive so that the row a join needs from *both* sides scores highest: a
    reagent-table row naming a line in a resources section collects the table
    bonus, the line-shape bonus and the section bonus at once.
    """
    reasons: list[str] = []
    score = 0
    if has_line_shaped_mention(evidence.text):
        score += 40
        reasons.append("names_line_shaped_entity")
    if _PROVENANCE.search(evidence.text):
        score += 30
        reasons.append("provenance_or_culture_vocabulary")
    rows = [sources[sid] for sid in evidence.source_ids if sid in sources]
    if any(row.block_type in _STRUCTURED_BLOCK_TYPES for row in rows):
        score += 20
        reasons.append("structured_block")
    if any(
        _METHODS_SECTION.search(" ".join(filter(None, (row.section, row.subsection))))
        for row in rows
    ):
        score += 15
        reasons.append("methods_or_resources_section")
    return score, reasons


def select_identity_evidence(
    packet: CompactApiPacket, *, max_items: int = 260
) -> list[ApiEvidence]:
    """The methods and reagent-table slice of a packet, highest signal first.

    Deterministic: ties break on evidence id, so the same packet always produces
    the same pre-pass input and the same request fingerprint.
    """
    sources = {row.source_id: row for row in packet.sources}
    scored: list[tuple[int, str, ApiEvidence]] = []
    for evidence in packet.evidence:
        score, _ = identity_evidence_score(evidence, sources)
        if score <= 0:
            continue
        scored.append((score, evidence.evidence_id, evidence))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in scored[:max_items]]


def evidence_payload(
    evidence: Iterable[ApiEvidence], sources: dict[str, ApiSource]
) -> list[dict[str, Any]]:
    """Render the selected evidence for the pre-pass call.

    Section, block type and page travel with the text because the join this pass
    has to make is often *between* a table row and a sentence, and a reader
    cannot see that they belong together without knowing where each came from.
    """
    rendered = []
    for item in evidence:
        rows = [sources[sid] for sid in item.source_ids if sid in sources]
        first = rows[0] if rows else None
        rendered.append(
            {
                "evidence_id": item.evidence_id,
                "text": item.text,
                "block_type": first.block_type if first else None,
                "section": first.section if first else None,
                "subsection": first.subsection if first else None,
                "page_number": first.page_number if first else None,
            }
        )
    return rendered


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #


class FieldDrop(StrictModel):
    field: str
    value: str
    missing_terms: list[str]


class MappingReport(StrictModel):
    line_name: str
    kept: bool
    reason: str | None = None
    cited_evidence_ids: list[str] = Field(default_factory=list)
    unknown_evidence_ids: list[str] = Field(default_factory=list)
    dropped_fields: list[FieldDrop] = Field(default_factory=list)


class GroundingReport(StrictModel):
    paper_id: str
    proposed: int
    kept: int
    mappings: list[MappingReport] = Field(default_factory=list)


_CHECKED_FIELDS = ("population", "state", "species")


def ground_entity_table(
    response: EntityResolutionResponse, packet: CompactApiPacket
) -> tuple[GroundedEntityTable, GroundingReport]:
    """Keep only what the packet can be shown to say.

    Rejects, in order:

    * a mapping citing nothing;
    * a mapping citing an id this packet does not contain -- the whole mapping,
      not just the id, because the reasoning that produced it ran over text this
      packet cannot show;
    * a line name that does not occur in the text it cites;
    * a field asserting a term the cited text does not contain -- that field
      only, set to null and recorded;
    * a mapping left asserting nothing about the line.
    """
    known = {item.evidence_id: item.text for item in packet.evidence}
    kept: list[CellLineIdentity] = []
    reports: list[MappingReport] = []

    for mapping in response.cell_lines:
        cited = list(dict.fromkeys(mapping.evidence_ids))
        unknown = [eid for eid in cited if eid not in known]
        if not cited:
            reports.append(
                MappingReport(
                    line_name=mapping.line_name, kept=False, reason="no_evidence_cited"
                )
            )
            continue
        if unknown:
            reports.append(
                MappingReport(
                    line_name=mapping.line_name,
                    kept=False,
                    reason="cited_evidence_id_not_in_packet",
                    cited_evidence_ids=cited,
                    unknown_evidence_ids=unknown,
                )
            )
            continue

        texts = [known[eid] for eid in cited]
        haystack = _haystack_terms(texts)
        normalized_join = normalize(" ".join(texts))

        # The line name is checked twice: term by term, and as a contiguous
        # string. A reworded population is legitimate; a reworded line name is a
        # different line.
        name_missing = sorted(terms(mapping.line_name) - haystack)
        if name_missing or normalize(mapping.line_name) not in normalized_join:
            reports.append(
                MappingReport(
                    line_name=mapping.line_name,
                    kept=False,
                    reason="line_name_not_in_cited_evidence",
                    cited_evidence_ids=cited,
                    dropped_fields=[
                        FieldDrop(
                            field="line_name",
                            value=mapping.line_name,
                            missing_terms=name_missing,
                        )
                    ],
                )
            )
            continue

        fields: dict[str, str | None] = {}
        drops: list[FieldDrop] = []
        for field in _CHECKED_FIELDS:
            value = getattr(mapping, field)
            if value is None or not value.strip():
                fields[field] = None
                continue
            missing = sorted(terms(value) - haystack)
            if missing:
                fields[field] = None
                drops.append(
                    FieldDrop(field=field, value=value, missing_terms=missing)
                )
            else:
                fields[field] = value

        if fields["population"] is None and fields["state"] is None:
            reports.append(
                MappingReport(
                    line_name=mapping.line_name,
                    kept=False,
                    reason="nothing_grounded_beyond_the_line_name",
                    cited_evidence_ids=cited,
                    dropped_fields=drops,
                )
            )
            continue

        kept.append(
            CellLineIdentity(
                line_name=mapping.line_name,
                population=fields["population"],
                state=fields["state"],
                species=fields["species"],
                evidence_ids=cited,
                derivation=mapping.derivation,
            )
        )
        reports.append(
            MappingReport(
                line_name=mapping.line_name,
                kept=True,
                cited_evidence_ids=cited,
                dropped_fields=drops,
            )
        )

    table = GroundedEntityTable(paper_id=response.paper_id, cell_lines=kept)
    report = GroundingReport(
        paper_id=response.paper_id,
        proposed=len(response.cell_lines),
        kept=len(kept),
        mappings=reports,
    )
    return table, report


def restrict_to_packet(
    table: GroundedEntityTable, packet: CompactApiPacket
) -> GroundedEntityTable:
    """Re-check a stored table against the packet it is about to travel with.

    The pre-pass and the extraction may run over different evidence views. A
    mapping whose citations are not in *this* packet cannot be checked by the
    model reading it, so it does not travel.
    """
    known = {item.evidence_id for item in packet.evidence}
    return GroundedEntityTable(
        entity_table_version=table.entity_table_version,
        paper_id=table.paper_id,
        cell_lines=[
            mapping
            for mapping in table.cell_lines
            if mapping.evidence_ids and known.issuperset(mapping.evidence_ids)
        ],
    )


def entity_table_payload(table: GroundedEntityTable) -> dict[str, Any]:
    """What is appended to the extraction prompt."""
    return {
        "entity_table_version": table.entity_table_version,
        "cell_lines": [
            {
                "line_name": mapping.line_name,
                "population": mapping.population,
                "state": mapping.state,
                "species": mapping.species,
                "evidence_ids": list(mapping.evidence_ids),
            }
            for mapping in table.cell_lines
        ],
    }
