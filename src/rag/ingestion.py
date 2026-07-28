from __future__ import annotations

import csv
import hashlib
import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

from src.config_flags import is_enabled
from src.extraction.pdf_multimodal_contracts import BoundingBox

from .models import DocumentBlock
from .uniparse_client import (
    IMAGE_ROOT,
    UniparseAnnotation,
    UniparseClient,
    UniparseContent,
    UniparseDocument,
    save_images,
)


ROOT = Path(__file__).resolve().parents[2]
GOLD_PAPERS = ROOT / "data" / "annotations" / "gold_v1" / "papers.csv"
XML_ROOT = ROOT / "data" / "raw" / "fulltext" / "gold_v1" / "xml"
OA_ROOT = ROOT / "data" / "raw" / "fulltext" / "oa_packages"
CORPUS_ROOT = ROOT / "data" / "staging" / "rag" / "gold_v1"

UNIPARSE_ENABLED_ENV = "UNIPARSE_ENABLED"
UNIPARSE_FLAG = "uniparse_ingestion"
SUPPLEMENT_SECTION_ROOT = "Supplement"

LABEL_PREFIX = r"(?:Supplementary\s+|Supplemental\s+|Extended\s+Data\s+)?"
# The trailing character class stops the number from swallowing the sentence
# period in captions such as "Table S2. Insertion Frequency ...".
LABEL_NUMBER = r"([A-Za-z]?\d(?:[\w.\-]*\w)?)"
TABLE_LABEL = re.compile(rf"\b{LABEL_PREFIX}Table\s+{LABEL_NUMBER}", re.I)
FIGURE_LABEL = re.compile(
    rf"\b{LABEL_PREFIX}(?:Figure|Fig\.?)\s+{LABEL_NUMBER}", re.I
)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def element_text(element: ET.Element) -> str:
    return compact("".join(element.itertext()))


def stable_id(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()[:20]


def relative_source(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def gold_manifest() -> list[dict[str, str]]:
    with GOLD_PAPERS.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_xml(candidate_id: str, pmcid: str) -> Path:
    package_xml = sorted((OA_ROOT / pmcid).glob("*.nxml"))
    if package_xml:
        return package_xml[0]
    matches = sorted(XML_ROOT.glob(f"{candidate_id}_{pmcid}.xml"))
    if not matches:
        raise FileNotFoundError(f"No PMC XML for {candidate_id} {pmcid}")
    return matches[0]


def section_title(sec: ET.Element, fallback: str) -> str:
    title = next((child for child in sec if local_name(child.tag) == "title"), None)
    return element_text(title) if title is not None else fallback


# --------------------------------------------------------------------------
# Table structure helpers, shared by the uniparse and PMC XML paths so that a
# table cell can be cited identically no matter which parser produced it.
# --------------------------------------------------------------------------


class _TableGridParser(HTMLParser):
    """Collect `<tr>`/`<td>` text from a simple HTML table."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.grid: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            if self._row is None:
                self._row = []
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None:
            assert self._row is not None
            self._row.append(compact("".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.grid.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def close(self) -> None:  # tolerate unclosed trailing tags
        super().close()
        if self._cell is not None and self._row is not None:
            self._row.append(compact("".join(self._cell)))
            self._cell = None
        if self._row is not None:
            self.grid.append(self._row)
            self._row = None


def table_grid_from_html(html: str) -> list[list[str]]:
    parser = _TableGridParser()
    parser.feed(html or "")
    parser.close()
    return [row for row in parser.grid if any(cell for cell in row)]


def table_grid_from_xml(element: ET.Element) -> list[list[str]]:
    """Extract a cell grid from a JATS `table-wrap`/`table` element."""
    grid: list[list[str]] = []
    for row in (node for node in element.iter() if local_name(node.tag) == "tr"):
        cells = [
            element_text(cell)
            for cell in row
            if local_name(cell.tag) in {"td", "th"}
        ]
        if any(cells):
            grid.append(cells)
    return grid


def grid_to_html(grid: list[list[str]]) -> str:
    """Serialize a cell grid to the same minimal HTML shape uniparse returns."""
    rows = "".join(
        "<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>"
        for row in grid
    )
    return f"<table>{rows}</table>"


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def table_block_text(grid: list[list[str]], label: str | None) -> str:
    """Flatten a table for lexical retrieval, keeping row and column order."""
    body = " ; ".join(" | ".join(cell for cell in row) for row in grid)
    return f"{label} | {body}" if label else body


def table_row_texts(grid: list[list[str]], label: str | None) -> list[str]:
    """Render one self-contained sentence per data row.

    Each row repeats its table label, its row label and every column header so
    that a single row block answers a question such as
    "(Table S2, LSEC, total insertion frequency)" without any other context.
    Separators avoid ". " because downstream clause splitting breaks there.
    """
    if len(grid) < 2:
        return []
    header = grid[0]
    texts: list[str] = []
    for row in grid[1:]:
        row_label = row[0] if row else ""
        pairs = []
        for index, cell in enumerate(row):
            if index == 0 and row_label:
                continue
            if not cell:
                continue
            column = header[index] if index < len(header) else ""
            pairs.append(f"{column}: {cell}" if column else cell)
        if not pairs:
            continue
        prefix = " | ".join(part for part in (label, row_label) if part)
        texts.append(f"{prefix} | {'; '.join(pairs)}" if prefix else "; ".join(pairs))
    return texts


def label_from_text(text: str, pattern: re.Pattern[str], noun: str) -> str | None:
    match = pattern.search(text or "")
    return f"{noun} {match.group(1)}" if match else None


def table_number_from(*texts: str) -> str | None:
    for text in texts:
        label = label_from_text(text, TABLE_LABEL, "Table")
        if label:
            return label
    return None


def figure_number_from(*texts: str) -> str | None:
    for text in texts:
        label = label_from_text(text, FIGURE_LABEL, "Figure")
        if label:
            return label
    return None


# --------------------------------------------------------------------------
# PMC XML
# --------------------------------------------------------------------------


def xml_blocks(paper_id: str, path: Path) -> list[DocumentBlock]:
    root = ET.fromstring(path.read_bytes())
    source_path = relative_source(path)
    rows: list[DocumentBlock] = []
    seen: set[int] = set()
    emitted: set[str] = set()

    def add_text(text: str, block_type: str, section: str, **metadata) -> None:
        if not text:
            return
        block_id = f"{paper_id}-B-{stable_id(str(path), section, block_type, text)}"
        if block_id in emitted:
            return
        emitted.add(block_id)
        rows.append(DocumentBlock(
            block_id=block_id, paper_id=paper_id, source_path=source_path,
            source_kind="pmc_xml", section_path=section, block_type=block_type,
            text=text, char_start=0, char_end=len(text), parser="pmc_xml",
            parser_confidence=1.0, **metadata,
        ))

    def add(element: ET.Element, block_type: str, section: str, **metadata):
        text = element_text(element)
        if not text or id(element) in seen:
            return
        seen.add(id(element))
        add_text(
            text, block_type, section,
            xml_element_id=element.attrib.get("id"), **metadata,
        )

    for title in (element for element in root.iter() if local_name(element.tag) == "article-title"):
        add(title, "title", "Article")
        break
    for abstract in (element for element in root.iter() if local_name(element.tag) == "abstract"):
        for paragraph in (x for x in abstract.iter() if local_name(x.tag) == "p"):
            add(paragraph, "abstract", "Abstract")
        break

    body = next((element for element in root.iter() if local_name(element.tag) == "body"), None)
    if body is not None:
        def visit(container: ET.Element, parents: list[str]):
            for child in container:
                name = local_name(child.tag)
                section = " > ".join(parents) or "Body"
                if name == "sec":
                    heading = section_title(child, "Untitled section")
                    visit(child, parents + [heading])
                elif name == "p":
                    add(child, "paragraph", section)
                elif name == "table-wrap":
                    add_table_wrap(child, section)
                elif name == "fig":
                    label = next((element_text(x) for x in child if local_name(x.tag) == "label"), "")
                    captions = [x for x in child.iter() if local_name(x.tag) == "caption"]
                    for caption in captions:
                        add(caption, "figure_caption", section, figure_number=label or None)

        def add_table_wrap(element: ET.Element, section: str) -> None:
            label = next(
                (element_text(x) for x in element if local_name(x.tag) == "label"), ""
            )
            caption = next(
                (element_text(x) for x in element if local_name(x.tag) == "caption"), ""
            )
            grid = table_grid_from_xml(element)
            table_number = label or table_number_from(label, caption)
            # The flattened whole-table text is kept as the `table` block so
            # existing corpora and block ids stay comparable; the structure is
            # added alongside it as HTML plus one block per data row.
            add(
                element, "table", section,
                table_number=table_number or None,
                table_html=grid_to_html(grid) if grid else None,
                caption_text=caption or None,
            )
            for row_text in table_row_texts(grid, table_number):
                add_text(
                    row_text, "table_row", section,
                    table_number=table_number or None,
                    xml_element_id=element.attrib.get("id"),
                )

        visit(body, [])
    return rows


# --------------------------------------------------------------------------
# Supplement PDFs
# --------------------------------------------------------------------------


def uniparse_enabled() -> bool:
    """Resolve the P1 flag through the registry.

    UNIPARSE_ENABLED is honoured because deployments set it, but it is now
    declared on the flag as extra_env_vars rather than read here. Reading it
    first meant it beat both override() and the flag's own
    AI_LNP_FLAG_UNIPARSE_INGESTION, so a test or a caller could not turn
    uniparse off in an environment that had it set -- a bypass, not an
    override, whatever the previous docstring called it.
    """
    return is_enabled(UNIPARSE_FLAG)


def uniparse_blocks(
    paper_id: str,
    path: Path,
    document: UniparseDocument,
    *,
    images: dict[str, dict[str, str | int]] | None = None,
    section_root: str = SUPPLEMENT_SECTION_ROOT,
) -> list[DocumentBlock]:
    """Convert a uniparse document into structured, citable evidence blocks.

    Page furniture returned in `page.others` (running headers, footers, page
    numbers) is deliberately dropped: it is never evidence and would otherwise
    dilute retrieval.
    """
    images = images or {}
    source_path = relative_source(path)
    parser_name = document.parser_name
    rows: list[DocumentBlock] = []
    emitted: set[str] = set()
    heading_stack: list[tuple[int, str]] = []

    def current_section() -> str:
        return " > ".join([section_root, *(title for _, title in heading_stack)])

    def emit(block_type: str, text: str, page_number: int, **metadata) -> None:
        text = compact(text)
        if not text:
            return
        section = current_section()
        block_id = (
            f"{paper_id}-B-{stable_id(source_path, str(page_number), block_type, text)}"
        )
        if block_id in emitted:
            return
        emitted.add(block_id)
        rows.append(DocumentBlock(
            block_id=block_id, paper_id=paper_id, source_path=source_path,
            source_kind="uniparse", section_path=section, block_type=block_type,
            text=text, page_number=page_number, char_start=0, char_end=len(text),
            parser=parser_name, parser_confidence=0.9, **metadata,
        ))

    def image_metadata(element: UniparseContent) -> dict[str, str]:
        if not element.img_path:
            return {}
        asset = images.get(element.img_path, {})
        metadata: dict[str, str] = {"image_ref": element.img_path}
        digest = asset.get("sha256")
        if isinstance(digest, str):
            metadata["image_sha256"] = digest
        return metadata

    def annotations_of(element: UniparseContent) -> list[UniparseAnnotation]:
        return [
            item
            for item in list(element.captions) + list(element.footnotes)
            if compact(item.text or "")
        ]

    def emit_captions(
        annotations: list[UniparseAnnotation],
        page_number: int,
        fallback_bbox: BoundingBox | None,
        *,
        table_number: str | None = None,
        figure_number: str | None = None,
    ) -> None:
        for item in annotations:
            # A caption sits outside the element it describes, so prefer its
            # own geometry; downstream vision crops depend on it.
            emit(
                "caption", item.text or "", page_number,
                bbox=bbox_of(item.bbox) or fallback_bbox,
                table_number=table_number, figure_number=figure_number,
            )

    for page in document.pages:
        # uniparse page_id is zero-based; page_number stays one-based so that
        # it matches PyMuPDF, the Day 8 vision tasks and human page references.
        page_number = page.page_id + 1
        for element in page.content:
            bbox = bbox_of(element.bbox)
            kind = (element.type or "").lower()
            annotations = annotations_of(element)
            annotation_texts = [compact(item.text or "") for item in annotations]
            caption_text = " ".join(annotation_texts) or None

            if kind == "heading":
                title = compact(element.text or "")
                if not title:
                    continue
                level = element.heading_level or 1
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                emit("heading", title, page_number, bbox=bbox)
                heading_stack.append((level, title))
                continue

            if kind == "table":
                grid = (
                    table_grid_from_html(element.text or "")
                    if (element.text_format or "").lower() == "html"
                    else []
                )
                table_number = table_number_from(*annotation_texts)
                if grid:
                    emit(
                        "table", table_block_text(grid, table_number), page_number,
                        bbox=bbox, table_number=table_number,
                        table_html=grid_to_html(grid), caption_text=caption_text,
                        **image_metadata(element),
                    )
                    for row_text in table_row_texts(grid, table_number):
                        emit(
                            "table_row", row_text, page_number, bbox=bbox,
                            table_number=table_number, caption_text=caption_text,
                        )
                elif compact(element.text or ""):
                    emit(
                        "table", element.text or "", page_number, bbox=bbox,
                        table_number=table_number, caption_text=caption_text,
                        **image_metadata(element),
                    )
                emit_captions(
                    annotations, page_number, bbox, table_number=table_number
                )
                continue

            if kind == "image":
                figure_number = figure_number_from(*annotation_texts)
                # An image carries no extractable text of its own, so the
                # caption is the figure block's text. A separate caption block
                # would only duplicate it, so annotations are folded in here.
                body = compact(element.text or "") or " ".join(annotation_texts)
                if body:
                    emit(
                        "figure", body, page_number, bbox=bbox,
                        figure_number=figure_number, caption_text=caption_text,
                        **image_metadata(element),
                    )
                continue

            # "text", "equation" and any future scalar element type.
            emit("paragraph", element.text or "", page_number, bbox=bbox)

    return rows


def bbox_of(values: list[float] | None) -> BoundingBox | None:
    if not values or len(values) < 4:
        return None
    x0, y0, x1, y1 = (float(value) for value in values[:4])
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def pymupdf_page_blocks(paper_id: str, path: Path) -> list[DocumentBlock]:
    """Legacy whole-page fallback. Loses all table and figure structure."""
    try:
        import pymupdf
    except ImportError as error:
        raise RuntimeError("PyMuPDF is required in the RAG environment") from error
    rows = []
    with pymupdf.open(path) as document:
        for page_index, page in enumerate(document):
            text = compact(page.get_text("text", sort=True))
            if not text:
                continue
            rows.append(DocumentBlock(
                block_id=f"{paper_id}-B-{stable_id(str(path), str(page_index + 1), text)}",
                paper_id=paper_id, source_path=relative_source(path), source_kind="pdf",
                section_path=f"Supplement page {page_index + 1}", block_type="pdf_page",
                text=text, page_number=page_index + 1, char_start=0, char_end=len(text),
                parser="pymupdf", parser_confidence=0.75,
            ))
    return rows


def pdf_blocks(
    paper_id: str,
    path: Path,
    *,
    client: UniparseClient | None = None,
    image_root: Path = IMAGE_ROOT,
    warnings: list[str] | None = None,
    allow_fallback: bool = True,
) -> list[DocumentBlock]:
    """Parse a supplement PDF, preferring structured uniparse ingestion.

    Falls back to the legacy PyMuPDF page dump when uniparse is unavailable so
    that corpus building degrades instead of failing outright.
    """
    if uniparse_enabled():
        try:
            client = client or UniparseClient()
            document = client.parse_pdf(path)
            images = save_images(document, paper_id, image_root=image_root)
            return uniparse_blocks(paper_id, path, document, images=images)
        except Exception as error:  # noqa: BLE001 - degradation is the point
            message = (
                f"uniparse ingestion failed for {relative_source(path)}: "
                f"{type(error).__name__}: {error}"
            )
            if not allow_fallback:
                raise
            if warnings is not None:
                warnings.append(message)
            print(f"WARNING: {message}; falling back to pymupdf page blocks")
    return pymupdf_page_blocks(paper_id, path)


def supplement_pdfs(pmcid: str) -> Iterable[Path]:
    directory = OA_ROOT / pmcid
    if not directory.exists():
        return []
    return [
        path for path in sorted(directory.glob("*.pdf"))
        if not re.search(r"(Article_|pnas\.202534673\.pdf$|main\.pdf$)", path.name)
    ]


def build_corpus(
    *,
    paper_ids: list[str] | None = None,
    skip_missing_xml: bool = False,
) -> dict:
    """Rebuild the block corpus.

    ``paper_ids`` restricts the run; ``skip_missing_xml`` records a paper whose
    source XML is absent as a warning instead of aborting. Without it one
    missing file ends the run after earlier papers are already on disk, leaving
    a corpus that is neither the old one nor the new one.
    """
    CORPUS_ROOT.mkdir(parents=True, exist_ok=True)
    client = UniparseClient()
    warnings: list[str] = []
    manifest = {
        "papers": [],
        "uniparse_available": client.available(),
        "uniparse_base_url": client.base_url,
        "warnings": warnings,
    }
    for paper in gold_manifest():
        paper_id, candidate_id, pmcid = paper["gold_paper_id"], paper["candidate_id"], paper["pmcid"]
        if paper_ids and paper_id not in paper_ids:
            continue
        try:
            xml_path = find_xml(candidate_id, pmcid)
        except FileNotFoundError as error:
            if not skip_missing_xml:
                raise
            warnings.append(f"{paper_id}: {error}")
            continue
        blocks = xml_blocks(paper_id, xml_path)
        pdf_paths = list(supplement_pdfs(pmcid))
        for pdf_path in pdf_paths:
            blocks.extend(
                pdf_blocks(paper_id, pdf_path, client=client, warnings=warnings)
            )
        output = CORPUS_ROOT / f"{paper_id}.blocks.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for block in blocks:
                handle.write(block.model_dump_json() + "\n")
        manifest["papers"].append({
            "paper_id": paper_id, "pmcid": pmcid, "xml_path": relative_source(xml_path),
            "supplement_pdfs": [relative_source(path) for path in pdf_paths],
            "blocks": len(blocks), "characters": sum(len(block.text) for block in blocks),
            "block_types": {
                block_type: sum(row.block_type == block_type for row in blocks)
                for block_type in sorted({row.block_type for row in blocks})
            },
        })
    # A filtered run describes only the papers it rebuilt, so writing its
    # manifest wholesale would delete the entries for every paper it skipped
    # and leave a manifest that disagrees with the .blocks.jsonl files still on
    # disk. Merge into the existing manifest instead, replacing the rebuilt
    # papers and keeping the rest.
    manifest_path = CORPUS_ROOT / "manifest.json"
    if paper_ids and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        rebuilt = {row["paper_id"] for row in manifest["papers"]}
        kept = [
            row for row in previous.get("papers", [])
            if row.get("paper_id") not in rebuilt
        ]
        manifest["papers"] = sorted(
            kept + manifest["papers"], key=lambda row: row.get("paper_id", "")
        )
        manifest["partial_rebuild"] = sorted(rebuilt)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the block corpus.")
    parser.add_argument("--paper-id", action="append", dest="paper_ids")
    parser.add_argument(
        "--skip-missing-xml",
        action="store_true",
        help="Record a missing source XML as a warning instead of aborting.",
    )
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Required: this rewrites the corpus on disk.",
    )
    args = parser.parse_args()
    if not args.confirm_write:
        parser.error("--confirm-write is required; this rewrites the corpus")
    print(json.dumps(
        build_corpus(
            paper_ids=args.paper_ids, skip_missing_xml=args.skip_missing_xml
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
