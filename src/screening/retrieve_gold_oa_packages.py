"""Retrieve complete lawful PMC Open Access packages for the gold set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import ssl
import tarfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import certifi

from src.rag.ingestion import GOLD_PAPERS, OA_ROOT, ROOT


OA_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
USER_AGENT = "AI-LNP evidence project (lawful PMC OA package retrieval)"
PMC_BIN = "https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/bin/{name}"
EUROPE_PMC_PDF = "https://europepmc.org/articles/{pmcid}?pdf=render"
EUROPE_PMC_SUPPLEMENTS = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/"
    "{pmcid}/supplementaryFiles"
)
XML_ROOT = ROOT / "data/raw/fulltext/gold_v1/xml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_bytes(url: str, timeout: float = 120.0, attempts: int = 4) -> bytes:
    context = ssl.create_default_context(cafile=certifi.where())
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                return response.read()
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def package_url(pmcid: str) -> str:
    root = ET.fromstring(request_bytes(OA_API.format(pmcid=pmcid)))
    link = next(
        (node for node in root.iter("link") if node.attrib.get("format") == "tgz"),
        None,
    )
    if link is None or not link.attrib.get("href"):
        error = next((node.text for node in root.iter("error")), "no tgz OA link")
        raise RuntimeError(f"{pmcid}: {error}")
    return link.attrib["href"].replace("ftp://", "https://")


def local_xml(candidate_id: str, pmcid: str) -> Path:
    matches = sorted(XML_ROOT.glob(f"{candidate_id}_{pmcid}.xml"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one XML for {candidate_id} {pmcid}")
    return matches[0]


def linked_assets(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    names: set[str] = set()
    for node in root.iter():
        href = next(
            (value for key, value in node.attrib.items() if key.endswith("}href") or key == "href"),
            None,
        )
        if not href:
            continue
        href = href.removeprefix("file:")
        if href.lower().endswith((".pdf", ".xlsx", ".xls", ".csv", ".zip")):
            names.add(Path(href).name)
    return sorted(names)


def retrieve_linked_package(
    paper: dict[str, str], destination: Path
) -> tuple[str, list[Path]]:
    pmcid, candidate_id = paper["pmcid"], paper["candidate_id"]
    xml_path = local_xml(candidate_id, pmcid)
    destination.mkdir(parents=True, exist_ok=True)
    copied_xml = destination / f"{pmcid}.nxml"
    if not copied_xml.exists():
        shutil.copy2(xml_path, copied_xml)
    files = [copied_xml]
    for name in linked_assets(xml_path):
        path = destination / name
        if not path.exists():
            path.write_bytes(request_bytes(PMC_BIN.format(pmcid=pmcid, name=name)))
        files.append(path)
    return "europe_pmc_linked_assets", files


def safe_extract_zip(archive: Path, destination: Path) -> None:
    resolved = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            target = (destination / name).resolve()
            if target != resolved and resolved not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {name}")
        handle.extractall(destination)


def retrieve_europe_pmc_package(
    paper: dict[str, str],
    destination: Path,
    *,
    warnings: list[str] | None = None,
) -> tuple[str, list[Path]]:
    """Primary retrieval path: Europe PMC rendered PDF plus supplementary files.

    This is the only route that reliably yields the supplement PDFs the
    extraction pipeline depends on; the NCBI OA service refuses tgz packages
    for most of the gold set.
    """
    pmcid, candidate_id = paper["pmcid"], paper["candidate_id"]
    xml_path = local_xml(candidate_id, pmcid)
    destination.mkdir(parents=True, exist_ok=True)
    copied_xml = destination / f"{pmcid}.nxml"
    if not copied_xml.exists():
        shutil.copy2(xml_path, copied_xml)
    pdf_names = [name for name in linked_assets(xml_path) if name.lower().endswith(".pdf")]
    main_name = pdf_names[0] if pdf_names else f"{pmcid}.pdf"
    main_pdf = destination / main_name
    if not main_pdf.exists():
        main_pdf.write_bytes(request_bytes(EUROPE_PMC_PDF.format(pmcid=pmcid)))

    def note(message: str) -> None:
        if warnings is not None:
            warnings.append(message)
        print(f"WARNING: {message}")

    # A missing supplement archive must not fail the whole paper, but it must
    # never be silent either: a supplement that quietly vanishes is exactly how
    # a table like Table S2 disappears from the evidence pool.
    supplement_zip = destination / f"{pmcid}_supplementary.zip"
    if not supplement_zip.exists():
        try:
            supplement_zip.write_bytes(request_bytes(
                EUROPE_PMC_SUPPLEMENTS.format(pmcid=pmcid), attempts=2
            ))
        except urllib.error.HTTPError as error:
            supplement_zip.unlink(missing_ok=True)
            if error.code != 404:
                raise
            note(f"{pmcid}: Europe PMC reports no supplementary files (HTTP 404)")
        except Exception as error:
            supplement_zip.unlink(missing_ok=True)
            note(
                f"{pmcid}: supplementary file download failed: "
                f"{type(error).__name__}: {error}"
            )
    if supplement_zip.exists():
        if not supplement_zip.stat().st_size:
            supplement_zip.unlink()
            note(f"{pmcid}: Europe PMC returned an empty supplementary archive")
        elif not zipfile.is_zipfile(supplement_zip):
            note(f"{pmcid}: supplementary download is not a ZIP archive; kept as-is")
        else:
            # Deliberately unguarded: a corrupt or unsafe archive is a real
            # failure and must abort this strategy rather than be swallowed.
            safe_extract_zip(supplement_zip, destination)
    return "europe_pmc_pdf_and_supplements", [
        path for path in destination.rglob("*") if path.is_file()
    ]


def retrieve_ncbi_oa_package(
    paper: dict[str, str], destination: Path, archive_root: Path
) -> tuple[str, str, str | None, str | None]:
    """Fallback retrieval path: the NCBI OA tgz package."""
    pmcid = paper["pmcid"]
    url = package_url(pmcid)
    archive = archive_root / f"{pmcid}.tar.gz"
    if not archive.exists():
        archive.write_bytes(request_bytes(url))
    safe_extract(archive, destination)
    flatten_single_directory(destination)
    return "ncbi_oa_tgz", url, str(archive.relative_to(ROOT)), sha256(archive)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if target != resolved and resolved not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.name}")
        handle.extractall(destination, filter="data")


def flatten_single_directory(destination: Path) -> None:
    children = [path for path in destination.iterdir() if path.name != ".package.json"]
    if len(children) != 1 or not children[0].is_dir():
        return
    nested = children[0]
    for path in list(nested.iterdir()):
        path.rename(destination / path.name)
    nested.rmdir()


def run() -> dict:
    with GOLD_PAPERS.open(newline="", encoding="utf-8") as handle:
        papers = list(csv.DictReader(handle))
    archive_root = ROOT / "data/raw/fulltext/oa_archives"
    archive_root.mkdir(parents=True, exist_ok=True)
    results = []
    for paper in papers:
        paper_id, pmcid = paper["gold_paper_id"], paper["pmcid"]
        destination = OA_ROOT / pmcid
        manifest_path = destination / ".package.json"
        if manifest_path.exists():
            results.append(json.loads(manifest_path.read_text()))
            continue
        result = {
            "paper_id": paper_id,
            "pmcid": pmcid,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
        }
        warnings: list[str] = []
        attempts: list[dict[str, str]] = []

        def europe_pmc() -> tuple[str, str, str | None, str | None]:
            method, _ = retrieve_europe_pmc_package(
                paper, destination, warnings=warnings
            )
            return method, EUROPE_PMC_PDF.format(pmcid=pmcid), None, None

        def linked_assets_strategy() -> tuple[str, str, str | None, str | None]:
            method, _ = retrieve_linked_package(paper, destination)
            return method, PMC_BIN.format(pmcid=pmcid, name="{linked_asset}"), None, None

        def ncbi_oa() -> tuple[str, str, str | None, str | None]:
            return retrieve_ncbi_oa_package(paper, destination, archive_root)

        # Europe PMC is the primary route because it is the one that actually
        # delivers supplementary files; the NCBI OA tgz endpoint denies most of
        # the gold set and is kept only as a last resort.
        strategies = (
            ("europe_pmc_pdf_and_supplements", europe_pmc),
            ("europe_pmc_linked_assets", linked_assets_strategy),
            ("ncbi_oa_tgz", ncbi_oa),
        )
        try:
            retrieval_method = url = None
            archive_path = archive_digest = None
            last_error: Exception | None = None
            for name, strategy in strategies:
                try:
                    retrieval_method, url, archive_path, archive_digest = strategy()
                    break
                except Exception as error:
                    last_error = error
                    attempts.append({
                        "strategy": name,
                        "error": f"{type(error).__name__}: {error}",
                    })
            if retrieval_method is None:
                assert last_error is not None
                raise last_error
            files = sorted(
                path for path in destination.rglob("*")
                if path.is_file() and path.name != ".package.json"
            )
            result.update({
                "status": "retrieved",
                "retrieval_method": retrieval_method,
                "package_url": url,
                "archive_path": archive_path,
                "archive_sha256": archive_digest,
                "files": [{
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                } for path in files],
                "pdf_count": sum(path.suffix.lower() == ".pdf" for path in files),
                "failed_strategies": attempts,
                "warnings": warnings,
            })
            destination.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n"
            )
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            result["failed_strategies"] = attempts
            result["warnings"] = warnings
        results.append(result)
    output = ROOT / "data/staging/extraction/day8_final_gate/retrieval_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    return {
        "papers": len(results),
        "retrieved": sum(row["status"] == "retrieved" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "pdfs": sum(row.get("pdf_count", 0) for row in results),
    }



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-write",
        action="store_true",
        help="Required: this rewrites tracked files in place.",
    )
    args = parser.parse_args()
    if not args.confirm_write:
        parser.error("--confirm-write is required; this rewrites tracked files")
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
