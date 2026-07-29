"""Retrieval strategy order and the OA API host, pinned without a network.

Design document sections 2.6 and 6.1 record two measured facts about
``src/screening/retrieve_gold_oa_packages.py``:

* ``https://pmc.ncbi.nlm.nih.gov/utils/oa/oa.fcgi`` answers **404**, while
  ``https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi`` answers **200**; and
* even with the host fixed, the tgz the OA API points at is unobtainable
  (FTP 550 / HTTPS 404), whereas one Europe PMC ``supplementaryFiles`` request
  returns the supplement PDFs the extraction pipeline depends on.

So Europe PMC was promoted to the primary route and the NCBI OA tgz demoted to
last resort. Until now that ordering was a claim the code made to nobody.

``request_bytes`` is the module's only network boundary. Every test here
replaces it with a recorder, so the whole of ``run()`` executes with no socket
and no ``data/`` write: ``ROOT``, ``OA_ROOT``, ``XML_ROOT`` and ``GOLD_PAPERS``
are redirected into ``tmp_path``.
"""

from __future__ import annotations

import csv
import io
import json
import tarfile
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest

from src.screening import retrieve_gold_oa_packages as retrieval


# A real row of data/annotations/gold_v1/papers.csv. GP-006 is the paper the
# design document actually measured the Europe PMC supplement endpoint against.
PAPER = {
    "gold_paper_id": "GP-006",
    "candidate_id": "candidate_00168",
    "pmcid": "PMC11617921",
}

# Enough JATS for linked_assets() to find one supplement PDF, which is what
# makes the second strategy reach the network and therefore able to fail.
NXML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<article xmlns:xlink="http://www.w3.org/1999/xlink">'
    "<body><sec><title>Results</title></sec></body>"
    "<supplementary-material>"
    '<media xlink:href="mmc1.pdf" mimetype="application"/>'
    "</supplementary-material>"
    "</article>"
)

OA_API_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    "<OA><records><record id=\"PMC11617921\">"
    '<link format="tgz" '
    'href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/ab/cd/PMC11617921.tar.gz"/>'
    "</record></records></OA>"
)

PDF_BYTES = b"%PDF-1.7\n% rendered article\n%%EOF\n"


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        for name, payload in members.items():
            handle.writestr(name, payload)
    return buffer.getvalue()


def _tar_gz_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as handle:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class _Boundary:
    """Stand-in for ``request_bytes``: records every URL, answers by route."""

    def __init__(self, handler):
        self.handler = handler
        self.urls: list[str] = []
        self.unexpected: list[str] = []

    def __call__(self, url: str, timeout: float = 120.0, attempts: int = 4) -> bytes:
        self.urls.append(url)
        try:
            return self.handler(url)
        except _Unexpected:
            self.unexpected.append(url)
            raise

    @property
    def hosts(self) -> list[str]:
        return [urlsplit(url).hostname for url in self.urls]


class _Unexpected(RuntimeError):
    """Raised by a handler for a URL the test did not plan for."""


def _install(monkeypatch, tmp_path: Path, handler) -> tuple[Path, _Boundary]:
    root = tmp_path / "checkout"
    xml_root = root / "data/raw/fulltext/gold_v1/xml"
    xml_root.mkdir(parents=True)
    (xml_root / f"{PAPER['candidate_id']}_{PAPER['pmcid']}.xml").write_text(
        NXML, encoding="utf-8"
    )

    papers = root / "data/annotations/gold_v1/papers.csv"
    papers.parent.mkdir(parents=True)
    with papers.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PAPER))
        writer.writeheader()
        writer.writerow(PAPER)

    boundary = _Boundary(handler)
    monkeypatch.setattr(retrieval, "ROOT", root)
    monkeypatch.setattr(retrieval, "OA_ROOT", root / "data/raw/fulltext/oa_packages")
    monkeypatch.setattr(retrieval, "XML_ROOT", xml_root)
    monkeypatch.setattr(retrieval, "GOLD_PAPERS", papers)
    monkeypatch.setattr(retrieval, "request_bytes", boundary)
    return root, boundary


def _manifest(root: Path) -> dict:
    path = root / "data/staging/extraction/day8_final_gate/retrieval_manifest.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert len(rows) == 1
    return rows[0]


# ---------------------------------------------------------------------------
# Strategy order
# ---------------------------------------------------------------------------


def test_europe_pmc_is_tried_first_and_the_ncbi_oa_api_is_never_reached(
    tmp_path, monkeypatch
):
    """The promotion, stated as behaviour: when Europe PMC works, NCBI is not called."""

    def handler(url: str) -> bytes:
        if "europepmc.org" in url:
            return PDF_BYTES
        if url.endswith("/supplementaryFiles"):
            return _zip_bytes({"mmc2.pdf": PDF_BYTES, "gr1.jpg": b"\xff\xd8\xff"})
        raise _Unexpected(url)

    root, boundary = _install(monkeypatch, tmp_path, handler)

    summary = retrieval.run()
    row = _manifest(root)

    assert boundary.unexpected == []
    assert summary == {"papers": 1, "retrieved": 1, "failed": 0, "pdfs": 2}
    assert row["retrieval_method"] == "europe_pmc_pdf_and_supplements"
    # Nothing was attempted before it, and nothing after it was needed.
    assert row["failed_strategies"] == []
    assert boundary.hosts[0] == "europepmc.org"
    assert not [url for url in boundary.urls if "oa.fcgi" in url]


def test_the_supplement_pdfs_actually_land_next_to_the_article(tmp_path, monkeypatch):
    """The whole reason Europe PMC was promoted: supplements arrive, unpacked."""

    def handler(url: str) -> bytes:
        if "europepmc.org" in url:
            return PDF_BYTES
        if url.endswith("/supplementaryFiles"):
            return _zip_bytes({"mmc2.pdf": PDF_BYTES, "gr1.jpg": b"\xff\xd8\xff"})
        raise _Unexpected(url)

    root, boundary = _install(monkeypatch, tmp_path, handler)

    retrieval.run()

    package = root / "data/raw/fulltext/oa_packages" / PAPER["pmcid"]
    assert boundary.unexpected == []
    assert (package / "mmc1.pdf").read_bytes() == PDF_BYTES  # rendered article
    assert (package / "mmc2.pdf").read_bytes() == PDF_BYTES  # from the supplement zip
    assert (package / "gr1.jpg").exists()
    assert (package / f"{PAPER['pmcid']}.nxml").exists()


def test_ncbi_oa_is_the_last_resort_and_its_host_is_www_ncbi_nlm_nih_gov(
    tmp_path, monkeypatch
):
    """Both Europe PMC routes must be exhausted before the OA tgz is touched.

    The host is asserted from the URL the code actually requested, not from the
    constant, so reverting section 2.6's correction to ``pmc.ncbi.nlm.nih.gov``
    fails here.
    """

    def handler(url: str) -> bytes:
        if "europepmc.org" in url:
            raise HTTPError(url, 500, "Internal Server Error", {}, None)
        if url.endswith("/supplementaryFiles"):
            raise HTTPError(url, 404, "Not Found", {}, None)
        if "/bin/" in url:  # the linked-asset route
            raise HTTPError(url, 403, "Forbidden", {}, None)
        if "oa.fcgi" in url:
            return OA_API_XML.encode()
        if url.endswith(".tar.gz"):
            return _tar_gz_bytes({
                f"{PAPER['pmcid']}/{PAPER['pmcid']}.nxml": NXML.encode(),
                f"{PAPER['pmcid']}/mmc1.pdf": PDF_BYTES,
            })
        raise _Unexpected(url)

    root, boundary = _install(monkeypatch, tmp_path, handler)

    summary = retrieval.run()
    row = _manifest(root)

    assert boundary.unexpected == []
    assert summary["retrieved"] == 1
    assert row["retrieval_method"] == "ncbi_oa_tgz"
    # Order, recorded by the code itself: the two Europe PMC routes were tried
    # and failed before the tgz route ran at all.
    assert [attempt["strategy"] for attempt in row["failed_strategies"]] == [
        "europe_pmc_pdf_and_supplements",
        "europe_pmc_linked_assets",
    ]

    oa_calls = [url for url in boundary.urls if "oa.fcgi" in url]
    assert len(oa_calls) == 1
    assert urlsplit(oa_calls[0]).hostname == "www.ncbi.nlm.nih.gov"
    assert urlsplit(oa_calls[0]).path == "/pmc/utils/oa/oa.fcgi"
    # ... and it came after both Europe PMC hosts, not before them.
    assert boundary.hosts[0] == "europepmc.org"
    assert boundary.hosts.index("www.ncbi.nlm.nih.gov") > boundary.hosts.index(
        "pmc.ncbi.nlm.nih.gov"
    )
    # The ftp:// href the OA API hands back is rewritten before it is fetched.
    assert row["package_url"].startswith("https://")


def test_a_missing_supplement_archive_is_warned_about_rather_than_swallowed(
    tmp_path, monkeypatch
):
    """A supplement that quietly vanishes is how a table leaves the evidence pool."""

    def handler(url: str) -> bytes:
        if "europepmc.org" in url:
            return PDF_BYTES
        if url.endswith("/supplementaryFiles"):
            raise HTTPError(url, 404, "Not Found", {}, None)
        raise _Unexpected(url)

    root, boundary = _install(monkeypatch, tmp_path, handler)

    summary = retrieval.run()
    row = _manifest(root)

    assert boundary.unexpected == []
    # A missing supplement must not fail the paper ...
    assert summary["retrieved"] == 1
    assert row["retrieval_method"] == "europe_pmc_pdf_and_supplements"
    # ... and must not be silent about it either.
    assert row["warnings"] == [
        f"{PAPER['pmcid']}: Europe PMC reports no supplementary files (HTTP 404)"
    ]
    package = root / "data/raw/fulltext/oa_packages" / PAPER["pmcid"]
    assert not (package / f"{PAPER['pmcid']}_supplementary.zip").exists()


def test_every_strategy_failing_is_reported_as_a_failed_paper(tmp_path, monkeypatch):
    """The fallback chain ends; it does not loop or silently report success."""

    def handler(url: str) -> bytes:
        raise HTTPError(url, 503, "Service Unavailable", {}, None)

    root, boundary = _install(monkeypatch, tmp_path, handler)

    summary = retrieval.run()
    row = _manifest(root)

    assert summary == {"papers": 1, "retrieved": 0, "failed": 1, "pdfs": 0}
    assert row["status"] == "failed"
    assert [attempt["strategy"] for attempt in row["failed_strategies"]] == [
        "europe_pmc_pdf_and_supplements",
        "europe_pmc_linked_assets",
        "ncbi_oa_tgz",
    ]


# ---------------------------------------------------------------------------
# Archive safety, on the last-resort route
# ---------------------------------------------------------------------------


def test_a_tgz_that_escapes_the_destination_is_refused(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    archive.write_bytes(_tar_gz_bytes({"../escaped.txt": b"nope"}))

    with pytest.raises(ValueError, match="Unsafe archive member"):
        retrieval.safe_extract(archive, tmp_path / "destination")

    assert not (tmp_path / "escaped.txt").exists()


def test_a_zip_that_escapes_the_destination_is_refused(tmp_path):
    archive = tmp_path / "evil.zip"
    archive.write_bytes(_zip_bytes({"../escaped.txt": b"nope"}))
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        retrieval.safe_extract_zip(archive, destination)

    assert not (tmp_path / "escaped.txt").exists()
