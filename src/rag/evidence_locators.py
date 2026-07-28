"""Deterministic, human-legible evidence locators.

Why this exists
---------------
Section 9 of the recall-improvement design document ("确定性 ID", subtitle
"可复现性是这个项目的立身之本") records that a compact packet rebuilt from
current code disagrees with the committed baseline for 6 of 9 papers, and
prescribes three fixes. The first is the identifier scheme::

    Evidence ID | GP-001-E-0431f0f16a6f0571 (content hash, drifts the moment
                | the content moves)
                | -> structured deterministic ID, e.g.
                |    GP-006-mmc1-p01-tabS2-r2-c7

A content hash answers "is this the same text?" and nothing else. It cannot be
read, cannot be checked against the PDF by a human, and changes when a parser
fixes a stray space. A *locator* answers "where in the paper is this?", which
is the question a reproducibility audit actually asks.

The grammar
-----------
::

    <paper>-<file>-p<NN>-<anchor>[-r<row>][-c<column>][-s<clause>]

``paper``
    The gold paper id, ``GP-006``. The only segment allowed to contain ``-``.
``file``
    The source file's stem with every non-alphanumeric character removed:
    ``.../PMC11617921/mmc1.pdf`` -> ``mmc1``, matching the design document's
    example. Distinguishes the article XML from each supplement.
``p<NN>``
    One-based page number, zero padded to at least two digits. ``p00`` means
    the source has no pagination, which is every PMC XML block.
``anchor``
    ``<kind><label>`` where kind is ``tab`` (table, table row), ``fig``
    (figure), ``cap`` (caption of either), or a three letter abbreviation of
    the block type otherwise (``par``, ``hed``, ``abs``, ``ttl``, ``pg``).
    ``label`` is the printed table or figure number with its noun and
    punctuation stripped -- "Table S2" -> ``S2`` -- so ``tabS2`` is what a
    reader sees on the page. A block with no printed label gets a zero-padded
    ordinal within its (file, page, kind) scope instead: ``par007``.
``-r<row>`` / ``-c<column>``
    One-based row and column inside a table grid, counting the header as row
    1. A ``table_row`` block carries the row; a single cell carries both.
``-s<clause>``
    One-based clause ordinal inside a block, for the sentence-level evidence
    the compact packet emits. Prose has no cells, so this is the prose
    counterpart of ``-c``: the last segment always names the smallest unit the
    id addresses.

What this scheme trades away
----------------------------
A locator is stable against *content* edits and unstable against *positional*
ones: renumbering the pages of a supplement moves every locator on it. That is
the trade the design document asks for -- it names "内容一动即漂移" (drifts the
moment the content moves) as the defect -- and it is not free, so it is stated
here rather than left to be discovered. Two consequences follow:

* An unlabelled block's ordinal depends on how many same-kind blocks precede
  it on its page. Inserting one shifts the rest of that page.
* Ordinals therefore have to be minted by walking a document in reading order,
  which is what :class:`LocatorMinter` is for. Minting the same document twice
  gives the same answer; minting one block in isolation cannot.

This module is pure: it reads no flags and touches no disk. Whether locators
are minted at all is decided at the ingestion and packet-building boundaries,
which is where ``structured_evidence_ids`` is read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath


__all__ = [
    "Locator",
    "LocatorMinter",
    "cell_locator",
    "clause_locator",
    "file_token",
    "label_token",
    "parse_locator",
    "row_locator",
]


_NON_ALNUM = re.compile(r"[^0-9A-Za-z]+")

# Nouns that precede a printed label. Dropped so that "Table S2", "Supplementary
# Table S2" and a bare "S2" all yield the same token, because the same table is
# referred to in all three ways across the corpus.
_LABEL_NOUNS = frozenset(
    {
        "table",
        "tables",
        "figure",
        "figures",
        "fig",
        "figs",
        "supplementary",
        "supplemental",
        "extended",
        "data",
        "scheme",
        "chart",
    }
)

# Three-letter kinds, one per BlockType in src/rag/models.py. "blk" is the
# fallback for a block type added later; it keeps ids well formed instead of
# raising during ingestion, and the round-trip test pins the mapping so a new
# type is noticed rather than silently pooled into "blk".
KIND_BY_BLOCK_TYPE = {
    "title": "ttl",
    "abstract": "abs",
    "paragraph": "par",
    "heading": "hed",
    "table": "tab",
    "table_row": "tab",
    "figure": "fig",
    "figure_caption": "cap",
    "caption": "cap",
    "pdf_page": "pg",
}
FALLBACK_KIND = "blk"

LOCATOR_RE = re.compile(
    r"^(?P<paper>[A-Za-z]+-\d+)"
    r"-(?P<file>[0-9A-Za-z]+)"
    r"-p(?P<page>\d+)"
    r"-(?P<anchor>[0-9A-Za-z]+)"
    r"(?:-r(?P<row>\d+))?"
    r"(?:-c(?P<column>\d+))?"
    r"(?:-s(?P<clause>\d+))?$"
)


@dataclass(frozen=True)
class Locator:
    """A parsed locator. ``str(locator)`` returns the id it came from."""

    paper_id: str
    file: str
    page: int
    anchor: str
    row: int | None = None
    column: int | None = None
    clause: int | None = None

    def __str__(self) -> str:
        parts = [self.paper_id, self.file, f"p{self.page:02d}", self.anchor]
        if self.row is not None:
            parts.append(f"r{self.row}")
        if self.column is not None:
            parts.append(f"c{self.column}")
        if self.clause is not None:
            parts.append(f"s{self.clause}")
        return "-".join(parts)


def parse_locator(value: str) -> Locator | None:
    """Parse a locator, or return ``None`` when ``value`` is not one.

    Returning ``None`` rather than raising is deliberate: callers hold mixed
    populations of hash ids and locators during the rollout, and asking "is
    this a locator?" must not need a try/except.
    """
    match = LOCATOR_RE.match(str(value or ""))
    if match is None:
        return None
    return Locator(
        paper_id=match.group("paper"),
        file=match.group("file"),
        page=int(match.group("page")),
        anchor=match.group("anchor"),
        row=None if match.group("row") is None else int(match.group("row")),
        column=(
            None if match.group("column") is None else int(match.group("column"))
        ),
        clause=(
            None if match.group("clause") is None else int(match.group("clause"))
        ),
    )


def file_token(source_path: str) -> str:
    """``.../PMC11617921/mmc1.pdf`` -> ``mmc1``.

    Only the final suffix is dropped, so ``pnas.2534673123.sapp.pdf`` keeps its
    internal dots as content and becomes ``pnas2534673123sapp``. Dropping every
    dotted part would collapse it to ``pnas``, which is not a file.
    """
    name = PurePosixPath(str(source_path or "").replace("\\", "/")).name
    stem = name[: -len(PurePosixPath(name).suffix)] if PurePosixPath(name).suffix else name
    token = _NON_ALNUM.sub("", stem)
    return token or "unknown"


def label_token(value: str | None) -> str | None:
    """``"Table S2"`` -> ``"S2"``; ``None`` when there is no printed label."""
    if not value:
        return None
    words = _NON_ALNUM.sub(" ", str(value)).split()
    while len(words) > 1 and words[0].lower() in _LABEL_NOUNS:
        words.pop(0)
    if len(words) == 1 and words[0].lower() in _LABEL_NOUNS:
        return None
    return "".join(words) or None


def row_locator(base: str, row: int) -> str:
    """Address one row of the table ``base`` names."""
    return f"{base}-r{int(row)}"


def cell_locator(base: str, row: int, column: int) -> str:
    """Address one cell. ``cell_locator("GP-006-mmc1-p01-tabS2", 2, 7)``."""
    return f"{base}-r{int(row)}-c{int(column)}"


def clause_locator(base: str, clause: int) -> str:
    """Address one clause of a prose block, the sentence-level evidence unit."""
    return f"{base}-s{int(clause)}"


class LocatorMinter:
    """Assign one locator per block, walking a paper in reading order.

    Ordinals and collision suffixes both depend on what has already been
    minted, so a minter is per paper and must be fed blocks in document order.
    Feeding the same order twice gives the same ids; that is the whole point,
    and :func:`src.rag.compact_packet.mint_corpus_locators` relies on it to
    reproduce ingestion's ids from a corpus file without re-parsing anything.
    """

    def __init__(self, paper_id: str) -> None:
        self.paper_id = paper_id
        self._ordinals: dict[tuple[str, int, str], int] = {}
        self._claimed: set[str] = set()

    # -- internals ---------------------------------------------------------

    def _anchor(
        self,
        *,
        file: str,
        page: int,
        kind: str,
        label: str | None,
    ) -> str:
        if label:
            return f"{kind}{label}"
        key = (file, page, kind)
        self._ordinals[key] = self._ordinals.get(key, 0) + 1
        return f"{kind}{self._ordinals[key]:03d}"

    def _claim(self, file: str, page: int, anchor: str) -> str:
        """Return the first unclaimed id for ``anchor``, disambiguating if needed.

        Two blocks can legitimately want the same anchor -- a table printed
        across a page break is labelled once and parsed twice -- so a claimed
        anchor gets ``x2``, ``x3`` appended rather than silently colliding. An
        id that is not unique is not an id.
        """
        candidate = f"{self.paper_id}-{file}-p{page:02d}-{anchor}"
        suffix = 1
        while candidate in self._claimed:
            suffix += 1
            candidate = f"{self.paper_id}-{file}-p{page:02d}-{anchor}x{suffix}"
        self._claimed.add(candidate)
        return candidate

    # -- public API --------------------------------------------------------

    def block(
        self,
        *,
        source_path: str,
        page_number: int | None,
        block_type: str,
        table_number: str | None = None,
        figure_number: str | None = None,
    ) -> str:
        """Mint the locator for one non-table block."""
        file = file_token(source_path)
        page = int(page_number or 0)
        kind = KIND_BY_BLOCK_TYPE.get(block_type, FALLBACK_KIND)
        label = label_token(
            table_number if kind == "tab" else
            figure_number if kind == "fig" else
            (table_number or figure_number) if kind == "cap" else
            None
        )
        return self._claim(file, page, self._anchor(
            file=file, page=page, kind=kind, label=label
        ))

    def table(
        self,
        *,
        source_path: str,
        page_number: int | None,
        table_number: str | None = None,
    ) -> str:
        """Mint the base locator a table and every one of its rows share.

        Rows are ``row_locator(base, i)`` and cells ``cell_locator(base, i, j)``,
        so the row ordinals come from the grid rather than from a counter --
        a row the flattener skips because every cell is empty must not shift
        the number of the row after it.
        """
        return self.block(
            source_path=source_path,
            page_number=page_number,
            block_type="table",
            table_number=table_number,
        )
