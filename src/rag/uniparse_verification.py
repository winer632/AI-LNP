"""Cross-check uniparse's transcription against the pixels it came from.

Why this module exists (design document, section 十)
----------------------------------------------------
uniparse is a VLM-driven parser, not a deterministic one. The tidy table HTML it
returns is *model output*, not page truth. The design document is blunt about
what follows::

    必须对 uniparse 的表格与图注结果额外加一层校验（建议：与表格裁剪 PNG 交叉核对,
    或对关键数值要求人工确认），否则只是把幻觉从抽取阶段挪到解析阶段 ——
    而且更隐蔽，因为它长得像结构化数据。

    (An extra verification layer over uniparse's table and caption results is
    mandatory -- cross-check against the table crop PNG, or require human
    confirmation of key values -- otherwise you have merely moved hallucination
    from the extraction stage to the parsing stage, and made it *harder* to see,
    because it now looks like structured data.)

Section 七 states the acceptance rule this layer enforces: only values that are
printed-visible or deterministically derived may be accepted; anything estimated
by eye must be abstained and referred to a human. It says explicitly that the
rule applies to uniparse's table HTML and not only to VLM vision output.

What it checks
--------------
Every uniparse block carries its own geometry -- ``source_path``, ``page_number``
and ``bbox``. That is enough to go back to the PDF and look at the same pixels
again, independently of the parse:

1. **Crop** the source PDF at the recorded bbox with PyMuPDF. This is a
   deterministic render; no model is involved.
2. **Read that crop back** with readers that did not produce the parse:

   ``pdf_text_layer``  -- ``page.get_text(clip=bbox)``. Free, model-free and
                          authoritative when the region has a text layer.
   ``ocr``             -- OCR of the rendered crop (:class:`MacVisionOcrEngine`
                          or :class:`TesseractOcrEngine`). Local, no network.
   ``uniparse_reread`` -- re-parse *only* that region as a standalone one-page
                          PDF and compare. A second read with none of the
                          surrounding document as context.

3. **Compare the numbers.** Every numeric token the parse asserts must appear in
   at least one independent read.

Corroboration and contradiction are deliberately asymmetric
-----------------------------------------------------------
Measured on the GP-006 Table S2 crop that supplies GO-006's ``1.01``: macOS
Vision OCR read the LSEC total as ``1.01 ÷ 0.38 %`` (right digits, ``±``
misread as ``÷``) but also read the Hepatocyte ``+1`` cell as ``3.39 $ 0.92 %``
and the ``+3`` cell as ``0.02 ± 0.02 %`` where the page prints ``0.91`` and
``0.01``. uniparse was right and the OCR was wrong -- at reported confidence
0.5 and 0.3 respectively, while every cell it read correctly came back at 1.0.

So: **finding a claimed number in a noisy read is strong evidence; failing to
find it is weak evidence.** A token is corroborated if any reader saw it, at any
confidence. A token is *contradicted* only when a reader that was confident
about every line it produced in that region (``HIGH_CONFIDENCE``) still did not
produce it. A read containing any uncertain line lands on ``unverified`` --
human review -- and never on ``contradicted``. Treating OCR disagreement as
proof of hallucination would have raised two false alarms on a table that is
transcribed correctly; the first cut of this module did exactly that, and marked
the Table S2 blocks ``contradicted`` over uniparse's correct ``0.91``.

What it found on the committed corpus
-------------------------------------
Run over every uniparse block in the nine gold papers with the offline readers
only (254 blocks; GP-001/004/006/007/008 are the papers with supplements):

* **GP-006 Table S2 verifies.** All twelve numbers of the row that supplies
  GO-006's ``1.01 ± 0.38 %`` are recovered from a 600 dpi crop of the recorded
  bbox, and a second uniparse read of that region alone returns byte-identical
  table HTML. The design document's pre-verification claim holds.
* **GP-004 Supplementary Tables 2 and 3 do not.** 42 of the 45 distinct
  sequence-length runs uniparse produced for pages 10-11 of
  ``41467_2021_20903_MOESM1_ESM.pdf`` do not occur in those pages' own text
  layer. On page 11 uniparse emitted a decoder repetition loop: the 23-mer
  ``GCTGCTCCCTGCGCTGCTCCCTG`` occurs 173 times in its output and nowhere in the
  source. Its EGF row scores 0.026 identity against the real EGF sequence, and
  HGF is truncated to 353 of 2184 bases. Those pages are ordinary vector text,
  so this is not a hard parse: a deterministic reader gets it right and the VLM
  did not. Six blocks in the committed corpus carry it.

  CORRECTION. This paragraph previously said uniparse gave the eGFP and HGF
  rows the Luciferase row's sequence. That was wrong, and wrong in the
  direction that exonerated this repository's own code. Each row carries its
  own gene -- the eGFP row matches real eGFP at 0.977 identity and Luciferase
  at 0.113. The apparent swap came from ``table_row_entries`` treating a
  header-less table's first row as a header, so the Luciferase sequence became
  a pseudo column name pasted into every other row, and a truncated view of any
  row read as though it carried Luciferase's sequence. That parser bug also
  dropped the Luciferase row entirely. Both are fixed; the fabrication above is
  what remains after fixing them.

That second result is section 十's warning happening: "前置验证证明它在 GP-006
上是对的，但这不等于它在所有论文上都对" -- verifying it on GP-006 does not mean
it is right on every paper.

Scope and limits, stated rather than implied
--------------------------------------------
* Only **numeric** tokens are checked. That is what section 七's hard rule is
  about, and it is what an outcome record cites. A block that asserts no numbers
  is reported as ``no_claims``: this layer makes no claim about prose, and says
  so instead of implying a check it did not run.
* Numbers are compared unsigned and with trailing zeros normalised, so ``0.10``
  matches ``0.1``. Sign is dropped because a minus rendered as an en-dash would
  otherwise read as a mismatch on every reader.
* A block with no ``bbox`` or no ``page_number`` cannot be cropped at all. That
  is ``no_geometry``, and it requires human review -- an unlocatable numeric
  claim is exactly the thing this layer exists to refuse to wave through.
* Verification never deletes evidence. It stamps a verdict and records a
  warning. Dropping an ``unverified`` block would trade a hallucination risk for
  a silent-omission risk, which is the failure this project is already fighting.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.extraction.pdf_multimodal_contracts import BoundingBox

from .models import DocumentBlock, VerificationStatus
from .uniparse_client import UniparseClient, UniparseDocument


ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = ROOT / "data" / "staging" / "rag" / "gold_v1"

# Rendering resolution for the crop handed to OCR. 600 dpi is what the GP-006
# measurement above was taken at; below ~300 dpi the small table type degrades
# fast and the OCR read stops being worth comparing against.
DEFAULT_DPI = 600

# A reader must be at least this confident before its silence counts as a
# contradiction rather than as "could not tell". See the module docstring: the
# two cells macOS Vision got wrong came back at 0.5 and 0.3.
HIGH_CONFIDENCE = 0.9

# Blocks whose text asserts values a downstream record would cite. `paragraph`
# and `heading` are excluded: they are prose, and this layer checks numbers.
VERIFIABLE_BLOCK_TYPES = frozenset({"table", "table_row", "caption", "figure_caption"})

# An alphanumeric run this long is not a word. It is a nucleotide sequence, an
# accession, a lot number -- exactly the low-information string a VLM degenerates
# on, and exactly the thing a human proofreader's eye slides over. Long enough
# that no ordinary English or ligature difference reaches it.
LONG_TOKEN_CHARS = 24

# Padding added around the bbox when cutting the standalone re-read PDF. A bbox
# that clips a glyph in half makes the second read fail for a reason that has
# nothing to do with the first read being wrong.
REREAD_PADDING_PT = 4.0

OCR_CACHE_DIR_ENV = "AI_LNP_OCR_CACHE_DIR"
VISION_OCR_BINARY_ENV = "AI_LNP_VISION_OCR"

_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_LONG_TOKEN = re.compile(rf"[A-Za-z0-9]{{{LONG_TOKEN_CHARS},}}")


__all__ = [
    "BlockVerification",
    "MacVisionOcrEngine",
    "OcrEngine",
    "OcrLine",
    "TesseractOcrEngine",
    "VerificationReport",
    "VerificationSession",
    "available_ocr_engine",
    "block_body_text",
    "canonical_number",
    "crop_pdf_region",
    "long_token_claims",
    "numeric_claims",
    "render_crop",
    "verify_and_stamp",
    "verify_blocks",
]


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #


def canonical_number(token: str) -> str:
    """Normalise one numeric token so equal magnitudes compare equal.

    ``0.10`` and ``0.1`` are the same measurement written two ways, and two
    readers will disagree about which one the page shows. ``0.00`` and ``0``
    likewise. Sign is dropped -- see the module docstring.
    """
    whole, _, fraction = token.partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def numeric_claims(*texts: str | None) -> tuple[str, ...]:
    """Every distinct number a piece of parser output asserts, in first order.

    Order is kept stable so a report diff is readable; duplicates are dropped so
    ``0.01 ± 0.01`` does not demand two separate corroborations of one value.
    """
    seen: list[str] = []
    for text in texts:
        for match in _NUMBER.finditer(text or ""):
            value = canonical_number(match.group(0))
            if value not in seen:
                seen.append(value)
    return tuple(seen)


def long_token_claims(*texts: str | None) -> tuple[str, ...]:
    """Sequence-like runs the parse asserts, e.g. a stretch of a nucleotide.

    These get their own check because they fail differently from numbers. A VLM
    asked to copy 1,700 bases does not misread one digit, it *degenerates*:
    it repeats a run, drops a homopolymer, or -- as uniparse did on GP-004
    Supplementary Table 2 -- gives three different genes the first gene's
    sequence. No amount of numeric agreement notices that.
    """
    seen: list[str] = []
    for text in texts:
        for match in _LONG_TOKEN.finditer(text or ""):
            token = match.group(0)
            if token not in seen:
                seen.append(token)
    return tuple(seen)


def block_body_text(block: DocumentBlock) -> str:
    """The block's text minus the table/figure label ingestion prepends to it.

    ``ingestion.table_block_text`` and ``table_row_texts`` synthesise a leading
    ``"Table 2 | "`` so a row block can be cited on its own. That label comes
    from the *caption*, which sits outside the table's bbox, so its number can
    never appear in a crop of the table body. Leaving it in made every numbered
    table look contradicted -- it is a claim about the parse's own bookkeeping,
    not about the pixels this layer is auditing.
    """
    for label in (block.table_number, block.figure_number):
        prefix = f"{label} | "
        if label and block.text.startswith(prefix):
            return block.text[len(prefix):]
    return block.text


# --------------------------------------------------------------------------- #
# OCR engines
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float


class OcrEngine(Protocol):
    """A local OCR backend. Implementations must not make network calls."""

    name: str

    def read(self, png: bytes) -> list[OcrLine]: ...


class _TempPng:
    def __init__(self, png: bytes):
        self._png = png

    def __enter__(self) -> str:
        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            handle.write(self._png)
        finally:
            handle.close()
        self._path = handle.name
        return self._path

    def __exit__(self, *_) -> None:
        try:
            os.unlink(self._path)
        except OSError:
            pass


class MacVisionOcrEngine:
    """macOS Vision text recognition, through a helper compiled on demand.

    The helper is built from ``resources/vision_ocr.swift`` into a cache
    directory outside the repository, so a build artefact never shows up in
    ``git status``. Set ``AI_LNP_VISION_OCR`` to a prebuilt binary to skip
    compilation entirely.
    """

    name = "macos_vision"
    SOURCE = Path(__file__).resolve().parent / "resources" / "vision_ocr.swift"

    def __init__(self, binary: str | Path | None = None):
        self._binary = Path(binary) if binary else None

    @staticmethod
    def cache_dir() -> Path:
        raw = (os.environ.get(OCR_CACHE_DIR_ENV) or "").strip()
        base = Path(raw).expanduser() if raw else Path.home() / ".cache" / "ai-lnp"
        return base

    @classmethod
    def supported(cls) -> bool:
        if os.environ.get(VISION_OCR_BINARY_ENV):
            return True
        return sys.platform == "darwin" and shutil.which("xcrun") is not None

    def binary(self) -> Path:
        if self._binary is not None:
            return self._binary
        configured = (os.environ.get(VISION_OCR_BINARY_ENV) or "").strip()
        if configured:
            self._binary = Path(configured).expanduser()
            return self._binary
        target = self.cache_dir() / "vision_ocr"
        if not target.exists() or target.stat().st_mtime < self.SOURCE.stat().st_mtime:
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["xcrun", "swiftc", "-O", str(self.SOURCE), "-o", str(target)],
                check=True,
                capture_output=True,
                timeout=300,
            )
        self._binary = target
        return target

    def read(self, png: bytes) -> list[OcrLine]:
        with _TempPng(png) as path:
            result = subprocess.run(
                [str(self.binary()), path],
                check=True,
                capture_output=True,
                timeout=120,
            )
        payload = json.loads(result.stdout or b"{}")
        return [
            OcrLine(text=str(row.get("text", "")), confidence=float(row.get("confidence", 0.0)))
            for row in payload.get("lines", [])
        ]


class TesseractOcrEngine:
    """Tesseract via its TSV output, which carries a per-word confidence."""

    name = "tesseract"

    def __init__(self, binary: str | None = None):
        self._binary = binary or shutil.which("tesseract")

    @classmethod
    def supported(cls) -> bool:
        return shutil.which("tesseract") is not None

    def read(self, png: bytes) -> list[OcrLine]:
        if not self._binary:
            raise RuntimeError("tesseract is not installed")
        result = subprocess.run(
            [self._binary, "stdin", "stdout", "--psm", "6", "tsv"],
            input=png,
            check=True,
            capture_output=True,
            timeout=180,
        )
        rows = result.stdout.decode("utf-8", "replace").splitlines()
        if not rows:
            return []
        header = rows[0].split("\t")
        try:
            conf_at, text_at = header.index("conf"), header.index("text")
            line_key = [header.index(name) for name in ("block_num", "par_num", "line_num")]
        except ValueError:
            return []
        grouped: dict[tuple[str, ...], list[tuple[str, float]]] = {}
        for row in rows[1:]:
            columns = row.split("\t")
            if len(columns) <= max(conf_at, text_at, *line_key):
                continue
            word = columns[text_at].strip()
            if not word:
                continue
            try:
                confidence = float(columns[conf_at]) / 100.0
            except ValueError:
                continue
            key = tuple(columns[index] for index in line_key)
            grouped.setdefault(key, []).append((word, confidence))
        return [
            OcrLine(
                text=" ".join(word for word, _ in words),
                # A line is only as trustworthy as its least certain word.
                confidence=min(confidence for _, confidence in words),
            )
            for words in grouped.values()
            if words
        ]


def available_ocr_engine(prefer: str | None = None) -> OcrEngine | None:
    """First usable local OCR backend, or ``None`` when there is none.

    Returning ``None`` is a supported outcome: the verifier then reports what it
    could and could not establish rather than pretending the check ran.
    """
    candidates: list[type] = [MacVisionOcrEngine, TesseractOcrEngine]
    if prefer:
        candidates.sort(key=lambda engine: engine.name != prefer)
    for candidate in candidates:
        if candidate.supported():
            return candidate()
    return None


# --------------------------------------------------------------------------- #
# Crops
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CropRender:
    """A deterministic re-render of the region a block says it came from."""

    source_path: str
    page_number: int
    bbox: tuple[float, float, float, float]
    dpi: int
    png: bytes
    sha256: str
    text_layer: str

    @property
    def key(self) -> tuple:
        return (self.source_path, self.page_number, self.bbox, self.dpi)


def _open_pdf(path: Path):
    try:
        import pymupdf
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError("PyMuPDF is required to verify uniparse output") from error
    return pymupdf, pymupdf.open(path)


def render_crop(
    pdf_path: Path,
    page_number: int,
    bbox: Sequence[float],
    *,
    dpi: int = DEFAULT_DPI,
) -> CropRender:
    """Render, and read the text layer of, the recorded region of a PDF page.

    ``page_number`` is one-based, matching ``DocumentBlock.page_number``.
    """
    pymupdf, document = _open_pdf(pdf_path)
    try:
        page = document[page_number - 1]
        rect = pymupdf.Rect(*[float(value) for value in bbox[:4]])
        pixmap = page.get_pixmap(clip=rect, dpi=dpi)
        png = pixmap.tobytes("png")
        text_layer = page.get_text("text", clip=rect) or ""
    finally:
        document.close()
    relative = str(pdf_path.relative_to(ROOT)) if pdf_path.is_relative_to(ROOT) else str(pdf_path)
    return CropRender(
        source_path=relative,
        page_number=page_number,
        bbox=tuple(float(value) for value in bbox[:4]),
        dpi=dpi,
        png=png,
        sha256=hashlib.sha256(png).hexdigest(),
        text_layer=text_layer,
    )


def crop_pdf_region(
    pdf_path: Path,
    page_number: int,
    bbox: Sequence[float],
    destination: Path,
    *,
    padding: float = REREAD_PADDING_PT,
) -> Path:
    """Write a one-page PDF holding only the recorded region.

    This is what the second uniparse read is given: the same vector content the
    first read saw, with every other element of the document removed, so a
    matching transcription cannot come from remembering the rest of the page.
    """
    pymupdf, source = _open_pdf(pdf_path)
    try:
        rect = pymupdf.Rect(*[float(value) for value in bbox[:4]])
        clip = pymupdf.Rect(
            rect.x0 - padding, rect.y0 - padding, rect.x1 + padding, rect.y1 + padding
        )
        out = pymupdf.open()
        try:
            page = out.new_page(width=clip.width, height=clip.height)
            page.show_pdf_page(
                pymupdf.Rect(0, 0, clip.width, clip.height),
                source,
                page_number - 1,
                clip=clip,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            out.save(destination)
        finally:
            out.close()
    finally:
        source.close()
    return destination


# --------------------------------------------------------------------------- #
# Independent reads
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IndependentRead:
    """What one reader recovered from the crop, and how sure it was."""

    method: str
    available: bool
    tokens: frozenset[str] = frozenset()
    # True only when the reader was confident about *everything* it saw in this
    # region, which is what makes its silence about a claimed number mean
    # something. See the module docstring.
    decisive: bool = False
    uncertain_lines: int = 0
    text: str = ""
    detail: str = ""
    # True for a reader with no recognition step at all -- today only the PDF's
    # own text layer. Only such a reader may be used to check long sequence-like
    # tokens, where an exact match is required and OCR noise would guarantee
    # false alarms.
    deterministic: bool = False

    @property
    def dense_text(self) -> str:
        """Text with all whitespace removed, so line wrapping cannot break a match."""
        return "".join(self.text.split())


def _read_from_lines(
    method: str,
    lines: Iterable[OcrLine],
    detail: str = "",
    *,
    deterministic: bool = False,
) -> IndependentRead:
    """Fold a reader's per-line output into one comparable read.

    ``decisive`` is all-or-nothing on purpose. A reader that flags even one line
    as uncertain has told you it could not resolve part of this region, and a
    number missing from such a read is as likely to be the reader's failure as
    the parser's invention. Scoring it per-token instead would need the reader's
    line boxes aligned to the parser's cells, which no reader here provides --
    and getting that alignment wrong is precisely how this layer would start
    accusing a correct table. Measured: on the GP-006 Table S2 crop, macOS
    Vision returned 21 lines of which two were below this bar, and those two
    were the only ones it got wrong.
    """
    lines = list(lines)
    text = "\n".join(line.text for line in lines)
    uncertain = [line for line in lines if line.confidence < HIGH_CONFIDENCE]
    if uncertain:
        note = (
            f"{len(uncertain)} of {len(lines)} lines below {HIGH_CONFIDENCE} "
            "confidence, so this read cannot contradict"
        )
        detail = f"{detail}; {note}" if detail else note
    return IndependentRead(
        method=method,
        available=bool(lines),
        tokens=frozenset(numeric_claims(text)),
        decisive=bool(lines) and not uncertain,
        uncertain_lines=len(uncertain),
        text=text,
        detail=detail,
        deterministic=deterministic,
    )


def read_text_layer(crop: CropRender) -> IndependentRead:
    """The PDF's own text layer inside the bbox. Deterministic, model-free."""
    text = crop.text_layer.strip()
    if not text:
        return IndependentRead(
            method="pdf_text_layer",
            available=False,
            detail="region has no text layer (rendered image)",
        )
    # Content stream text is what the file itself declares, so it is confident by
    # construction -- there is no recognition step that could be unsure.
    return _read_from_lines(
        "pdf_text_layer",
        [OcrLine(text=text, confidence=1.0)],
        deterministic=True,
    )


def read_ocr(crop: CropRender, engine: OcrEngine | None) -> IndependentRead:
    if engine is None:
        return IndependentRead(
            method="ocr", available=False, detail="no local OCR engine installed"
        )
    try:
        lines = engine.read(crop.png)
    except Exception as error:  # noqa: BLE001 - an OCR failure is a non-result
        return IndependentRead(
            method="ocr",
            available=False,
            detail=f"{engine.name} failed: {type(error).__name__}: {error}",
        )
    read = _read_from_lines(f"ocr:{engine.name}", lines, detail=f"{len(lines)} lines")
    return read


def read_uniparse_reread(
    crop_pdf: Path, client: UniparseClient | object | None
) -> IndependentRead:
    """Re-parse the isolated region and report the numbers it comes back with."""
    if client is None:
        return IndependentRead(
            method="uniparse_reread", available=False, detail="no re-read client supplied"
        )
    try:
        document = client.parse_pdf(crop_pdf)
    except Exception as error:  # noqa: BLE001 - an unreachable service is a non-result
        return IndependentRead(
            method="uniparse_reread",
            available=False,
            detail=f"re-read failed: {type(error).__name__}: {error}",
        )
    texts = _reread_texts(document)
    if not texts:
        return IndependentRead(
            method="uniparse_reread", available=False, detail="re-read returned no content"
        )
    # A second read is a second model, not a measurement: it is decisive because
    # a disagreement between two independent parses of the same pixels is a real
    # signal, unlike a low-confidence OCR wobble.
    return _read_from_lines("uniparse_reread", [OcrLine(text="\n".join(texts), confidence=1.0)])


def _reread_texts(document: UniparseDocument) -> list[str]:
    texts: list[str] = []
    for page in document.pages:
        for element in page.content:
            if element.text:
                texts.append(element.text)
            for annotation in list(element.captions) + list(element.footnotes):
                if annotation.text:
                    texts.append(annotation.text)
    if not texts and document.markdown:
        texts.append(document.markdown)
    return texts


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #


class BlockVerification(BaseModel):
    """One block's verdict, with the evidence that produced it."""

    model_config = ConfigDict(extra="forbid")

    block_id: str
    paper_id: str
    block_type: str
    source_path: str
    page_number: int | None = None
    bbox: BoundingBox | None = None
    status: VerificationStatus
    claims: list[str] = Field(default_factory=list)
    corroborated: list[str] = Field(default_factory=list)
    uncorroborated: list[str] = Field(default_factory=list)
    long_tokens: int = 0
    # Sequence-like runs the parse asserts that the region's own text layer does
    # not contain. A non-empty list here is not a "maybe": it is the file itself
    # disagreeing with the parse.
    fabricated_long_tokens: list[str] = Field(default_factory=list)
    methods_attempted: list[str] = Field(default_factory=list)
    methods_available: list[str] = Field(default_factory=list)
    corroborating_methods: list[str] = Field(default_factory=list)
    contradicting_methods: list[str] = Field(default_factory=list)
    crop_sha256: str | None = None
    crop_dpi: int | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def accepted(self) -> bool:
        """May the numbers in this block be cited as printed-visible values?"""
        return self.status in {"verified", "no_claims"}

    @property
    def requires_human_review(self) -> bool:
        """Section 七's hard rule: anything not established goes to a human."""
        return not self.accepted

    @property
    def method(self) -> str:
        """Compact provenance string stamped onto the block."""
        if self.corroborating_methods:
            return "+".join(self.corroborating_methods)
        return "+".join(self.methods_available) or "none"


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_ids: list[str] = Field(default_factory=list)
    dpi: int = DEFAULT_DPI
    ocr_engine: str | None = None
    reread_enabled: bool = False
    counts: dict[str, int] = Field(default_factory=dict)
    blocks: list[BlockVerification] = Field(default_factory=list)

    @property
    def needs_review(self) -> list[BlockVerification]:
        return [row for row in self.blocks if row.requires_human_review]


@dataclass(frozen=True)
class _Verdict:
    status: VerificationStatus
    corroborated: list[str]
    uncorroborated: list[str]
    corroborating: list[str]
    contradicting: list[str]
    fabricated: list[str]


def _verdict(
    claims: Sequence[str],
    long_tokens: Sequence[str],
    reads: Sequence[IndependentRead],
) -> _Verdict:
    available = [read for read in reads if read.available]
    corroborated: list[str] = []
    uncorroborated: list[str] = []
    corroborating: list[str] = []
    contradicting: list[str] = []
    for claim in claims:
        sources = [read.method for read in available if claim in read.tokens]
        if sources:
            corroborated.append(claim)
            for method in sources:
                if method not in corroborating:
                    corroborating.append(method)
            continue
        uncorroborated.append(claim)
        for read in available:
            if read.decisive and read.method not in contradicting:
                contradicting.append(read.method)

    # Long tokens are checked only against a deterministic reader. An exact
    # 24-plus character match against OCR output would fail on correct parses,
    # and against a second VLM read it would prove nothing.
    fabricated: list[str] = []
    deterministic = [read for read in available if read.deterministic]
    for token in long_tokens:
        found = [read for read in deterministic if token in read.dense_text]
        if deterministic and not found:
            fabricated.append(token)
            for read in deterministic:
                if read.method not in contradicting:
                    contradicting.append(read.method)
            continue
        for read in found:
            if read.method not in corroborating:
                corroborating.append(read.method)

    if not available:
        status: VerificationStatus = "unverified"
    elif fabricated:
        status = "contradicted"
    elif not uncorroborated:
        status = "verified"
    elif contradicting:
        status = "contradicted"
    else:
        status = "unverified"
    return _Verdict(
        status=status,
        corroborated=corroborated,
        uncorroborated=uncorroborated,
        corroborating=corroborating,
        contradicting=contradicting,
        fabricated=fabricated,
    )


class VerificationSession:
    """Verifies blocks, reusing one crop/read per distinct page region.

    A table and each of its ``table_row`` children record the same bbox, so
    without caching a nine-row table would be rendered, OCR'd and re-parsed nine
    times over.
    """

    def __init__(
        self,
        *,
        root: Path = ROOT,
        dpi: int = DEFAULT_DPI,
        ocr_engine: OcrEngine | None = None,
        reread_client: UniparseClient | object | None = None,
        work_dir: Path | None = None,
    ):
        self.root = root
        self.dpi = dpi
        self.ocr_engine = ocr_engine
        self.reread_client = reread_client
        self._work_dir = work_dir
        self._crops: dict[tuple, CropRender | str] = {}
        self._reads: dict[tuple, list[IndependentRead]] = {}
        self._temp: tempfile.TemporaryDirectory | None = None

    # -- lifecycle ------------------------------------------------------- #

    def close(self) -> None:
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None

    def __enter__(self) -> "VerificationSession":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _scratch(self) -> Path:
        if self._work_dir is not None:
            self._work_dir.mkdir(parents=True, exist_ok=True)
            return self._work_dir
        if self._temp is None:
            self._temp = tempfile.TemporaryDirectory(prefix="ai-lnp-verify-")
        return Path(self._temp.name)

    # -- reads ------------------------------------------------------------ #

    def _crop(self, source_path: str, page_number: int, bbox: tuple) -> CropRender | str:
        key = (source_path, page_number, bbox, self.dpi)
        if key not in self._crops:
            pdf_path = self.root / source_path
            if not pdf_path.is_file():
                self._crops[key] = f"source PDF not on disk: {source_path}"
            else:
                try:
                    # Keep the block's own source_path on the crop: render_crop
                    # relativises against the module ROOT, which is not this
                    # session's root when a caller points at another checkout.
                    self._crops[key] = replace(
                        render_crop(pdf_path, page_number, bbox, dpi=self.dpi),
                        source_path=source_path,
                    )
                except Exception as error:  # noqa: BLE001
                    self._crops[key] = (
                        f"could not crop {source_path} p{page_number}: "
                        f"{type(error).__name__}: {error}"
                    )
        return self._crops[key]

    def _independent_reads(self, crop: CropRender) -> list[IndependentRead]:
        if crop.key in self._reads:
            return self._reads[crop.key]
        reads = [read_text_layer(crop), read_ocr(crop, self.ocr_engine)]
        if self.reread_client is not None:
            page_tag = f"p{crop.page_number:03d}"
            name = f"{Path(crop.source_path).stem}-{page_tag}-{crop.sha256[:10]}.pdf"
            try:
                crop_pdf = crop_pdf_region(
                    self.root / crop.source_path,
                    crop.page_number,
                    crop.bbox,
                    self._scratch() / name,
                )
            except Exception as error:  # noqa: BLE001
                reads.append(
                    IndependentRead(
                        method="uniparse_reread",
                        available=False,
                        detail=f"could not cut crop PDF: {type(error).__name__}: {error}",
                    )
                )
            else:
                reads.append(read_uniparse_reread(crop_pdf, self.reread_client))
        self._reads[crop.key] = reads
        return reads

    # -- verdict ---------------------------------------------------------- #

    def verify(self, block: DocumentBlock) -> BlockVerification:
        body = block_body_text(block)
        claims = numeric_claims(body, block.table_html)
        long_tokens = long_token_claims(body, block.table_html)
        base = dict(
            block_id=block.block_id,
            paper_id=block.paper_id,
            block_type=block.block_type,
            source_path=block.source_path,
            page_number=block.page_number,
            bbox=block.bbox,
            claims=list(claims),
            long_tokens=len(long_tokens),
        )
        if not claims and not long_tokens:
            return BlockVerification(
                status="no_claims",
                notes=["block asserts no numeric or sequence values to cross-check"],
                **base,
            )
        if block.bbox is None or block.page_number is None:
            return BlockVerification(
                status="no_geometry",
                uncorroborated=list(claims),
                notes=[
                    "block asserts values but records no page/bbox, so the "
                    "source region cannot be re-read"
                ],
                **base,
            )
        bbox = (block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)
        crop = self._crop(block.source_path, block.page_number, bbox)
        if isinstance(crop, str):
            return BlockVerification(
                status="unverified",
                uncorroborated=list(claims),
                notes=[crop],
                **base,
            )
        reads = self._independent_reads(crop)
        verdict = _verdict(claims, long_tokens, reads)
        return BlockVerification(
            status=verdict.status,
            corroborated=verdict.corroborated,
            uncorroborated=verdict.uncorroborated,
            fabricated_long_tokens=verdict.fabricated,
            methods_attempted=[read.method for read in reads],
            methods_available=[read.method for read in reads if read.available],
            corroborating_methods=verdict.corroborating,
            contradicting_methods=verdict.contradicting,
            crop_sha256=crop.sha256,
            crop_dpi=crop.dpi,
            notes=[f"{read.method}: {read.detail}" for read in reads if read.detail],
            **base,
        )


def verify_blocks(
    blocks: Iterable[DocumentBlock],
    *,
    root: Path = ROOT,
    dpi: int = DEFAULT_DPI,
    ocr_engine: OcrEngine | None = None,
    reread_client: UniparseClient | object | None = None,
    only_uniparse: bool = True,
    work_dir: Path | None = None,
) -> VerificationReport:
    """Verify every block this layer is responsible for.

    ``only_uniparse`` keeps the scope to what section 十 is about. PMC XML tables
    are a deterministic parse of marked-up cells and are not at risk of the
    failure mode this guards against.
    """
    rows = [
        block
        for block in blocks
        if block.block_type in VERIFIABLE_BLOCK_TYPES
        and (not only_uniparse or block.source_kind == "uniparse")
    ]
    with VerificationSession(
        root=root,
        dpi=dpi,
        ocr_engine=ocr_engine,
        reread_client=reread_client,
        work_dir=work_dir,
    ) as session:
        verifications = [session.verify(block) for block in rows]
    counts: dict[str, int] = {}
    for row in verifications:
        counts[row.status] = counts.get(row.status, 0) + 1
    return VerificationReport(
        paper_ids=sorted({row.paper_id for row in verifications}),
        dpi=dpi,
        ocr_engine=getattr(ocr_engine, "name", None),
        reread_enabled=reread_client is not None,
        counts=counts,
        blocks=verifications,
    )


def verify_and_stamp(
    blocks: Sequence[DocumentBlock],
    *,
    root: Path = ROOT,
    dpi: int = DEFAULT_DPI,
    ocr_engine: OcrEngine | None = None,
    reread_client: UniparseClient | object | None = None,
    warnings: list[str] | None = None,
) -> list[DocumentBlock]:
    """Return the blocks with a verification verdict recorded on each.

    Blocks are never dropped. A block that could not be verified keeps its text
    and gains ``verification_status`` so that everything downstream -- and every
    human reading the corpus -- can see that its numbers rest on a single VLM
    transcription and nothing else.
    """
    report = verify_blocks(
        blocks,
        root=root,
        dpi=dpi,
        ocr_engine=ocr_engine,
        reread_client=reread_client,
        work_dir=None,
    )
    verdicts = {row.block_id: row for row in report.blocks}
    stamped: list[DocumentBlock] = []
    for block in blocks:
        verdict = verdicts.get(block.block_id)
        if verdict is None:
            stamped.append(block)
            continue
        stamped.append(
            block.model_copy(
                update={
                    "verification_status": verdict.status,
                    "verification_method": verdict.method,
                }
            )
        )
    if warnings is not None:
        for row in report.needs_review:
            warnings.append(
                f"uniparse verification {row.status} for {row.block_id} "
                f"({row.source_path} p{row.page_number}): "
                f"uncorroborated {row.uncorroborated}; "
                f"readers {row.methods_available or ['none']}"
            )
    return stamped


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def load_corpus(paper_id: str, corpus_root: Path = CORPUS_ROOT) -> list[DocumentBlock]:
    path = corpus_root / f"{paper_id}.blocks.jsonl"
    return [
        DocumentBlock.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-check uniparse table/caption output against the PDF."
    )
    parser.add_argument("--paper-id", action="append", dest="paper_ids", required=True)
    parser.add_argument("--block-id", action="append", dest="block_ids")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help=(
            "Repository root the blocks' source_path values are relative to. "
            "Needed when running from a worktree that does not hold the "
            "(gitignored) source PDFs."
        ),
    )
    parser.add_argument("--corpus-root", type=Path, default=None)
    parser.add_argument(
        "--reread",
        action="store_true",
        help="Also re-parse each region alone through uniparse (network call).",
    )
    parser.add_argument("--ocr-engine", default=None)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    corpus_root = args.corpus_root or (args.root / "data" / "staging" / "rag" / "gold_v1")
    blocks: list[DocumentBlock] = []
    for paper_id in args.paper_ids:
        blocks.extend(load_corpus(paper_id, corpus_root))
    if args.block_ids:
        wanted = set(args.block_ids)
        blocks = [block for block in blocks if block.block_id in wanted]

    engine = None if args.no_ocr else available_ocr_engine(args.ocr_engine)
    report = verify_blocks(
        blocks,
        root=args.root,
        dpi=args.dpi,
        ocr_engine=engine,
        reread_client=UniparseClient() if args.reread else None,
    )
    payload = report.model_dump(mode="json", exclude_none=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if any(row.status == "contradicted" for row in report.blocks) else 0


if __name__ == "__main__":
    raise SystemExit(_main())
