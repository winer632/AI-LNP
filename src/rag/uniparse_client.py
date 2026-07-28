"""Client for the uniparse structured document-parsing service.

uniparse turns a PDF into an ordered, typed element stream (headings, text,
tables with HTML, images with captions) plus page furniture (`others`) such as
running headers and page numbers. That is what lets supplement tables be
ingested as real tables instead of one flattened page of text.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = ROOT / "data" / "raw" / "fulltext" / "uniparse"

DEFAULT_BASE_URL = "http://100.67.181.118:8020"
BASE_URL_ENV = "UNIPARSE_BASE_URL"
PARSE_PATH = "/api/v1/parse"
USER_AGENT = "AI-LNP evidence project (uniparse structured ingestion)"

# DO NOT ADD heading_level_enabled OR heading_level_method TO THE REQUEST.
#
# The deployed uniparse container resolves heading levels through an LLM whose
# base_url in config.yaml is still the unsubstituted placeholder
# "http://<llm-server-openai>:<port>/v1/". Sending either field switches that
# code path on, the client tries to parse "<port>" as a TCP port, and the whole
# request fails with HTTP 500 "Invalid port: '<port>'". The service already
# returns usable heading_level values without them.
FORBIDDEN_FORM_FIELDS = frozenset({"heading_level_enabled", "heading_level_method"})


def base_url_from_env(base_url: str | None = None) -> str:
    return (base_url or os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")


class UniparseModel(BaseModel):
    """Lenient by design: a service upgrade must not break ingestion."""

    model_config = ConfigDict(extra="allow")


class UniparseAnnotation(UniparseModel):
    """A caption or footnote attached to a table or image element."""

    type: str | None = None
    text: str | None = None
    bbox: list[float] | None = None


class UniparseContent(UniparseModel):
    type: str
    bbox: list[float] | None = None
    text: str | None = None
    text_format: str | None = None
    heading_level: int | None = None
    img_path: str | None = None
    captions: list[UniparseAnnotation] = Field(default_factory=list)
    footnotes: list[UniparseAnnotation] = Field(default_factory=list)


class UniparseOther(UniparseModel):
    """Page furniture (running header/footer, page number). Never evidence."""

    type: str | None = None
    text: str | None = None
    bbox: list[float] | None = None


class UniparsePage(UniparseModel):
    page_id: int
    page_width: float | None = None
    page_height: float | None = None
    content: list[UniparseContent] = Field(default_factory=list)
    others: list[UniparseOther] = Field(default_factory=list)


class UniparseMeta(UniparseModel):
    filesize: int | None = None
    version: str | None = None


class UniparseDocument(UniparseModel):
    meta: UniparseMeta = Field(default_factory=UniparseMeta)
    pages: list[UniparsePage] = Field(default_factory=list)
    images: dict[str, str] = Field(default_factory=dict)
    markdown: str = ""

    @property
    def parser_name(self) -> str:
        """Value written to DocumentBlock.parser, e.g. "uniparse-1.1.0"."""
        return f"uniparse-{self.meta.version or 'unknown'}"


class UniparseError(RuntimeError):
    """Raised when the service cannot be reached or returns an unusable body."""


def form_fields(skip_image_elements: bool = False) -> dict[str, str]:
    """Return the multipart text fields sent alongside the PDF.

    Kept separate from transport so the forbidden heading fields can be
    asserted against directly in tests.
    """
    fields = {"skip_image_elements": "true" if skip_image_elements else "false"}
    assert not FORBIDDEN_FORM_FIELDS & set(fields)
    return fields


def encode_multipart(
    fields: dict[str, str],
    *,
    file_field: str,
    filename: str,
    payload: bytes,
    content_type: str = "application/pdf",
) -> tuple[str, bytes]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode()
    )
    parts.append(payload)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


class UniparseClient:
    """Thin, retrying HTTP client for POST {base_url}/api/v1/parse."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 900.0,
        attempts: int = 3,
    ):
        self.base_url = base_url_from_env(base_url)
        self.endpoint = self.base_url + PARSE_PATH
        self.timeout = timeout
        self.attempts = attempts
        # uniparse is reached over the tailnet. A configured HTTP(S) proxy has
        # no route to it and answers with its own error page, so open the
        # connection directly.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def available(self, timeout: float = 5.0) -> bool:
        """Cheap reachability probe used to decide whether to fall back."""
        request = urllib.request.Request(
            self.base_url + "/docs", headers={"User-Agent": USER_AGENT}
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                return 200 <= response.status < 500
        except urllib.error.HTTPError as error:
            # Any HTTP answer proves the service is listening.
            return error.code < 500
        except Exception:
            return False

    def _post(self, body: bytes, content_type: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            request = urllib.request.Request(
                self.endpoint,
                data=body,
                headers={"Content-Type": content_type, "User-Agent": USER_AGENT},
            )
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    return response.read()
            except Exception as error:  # noqa: BLE001 - retried and re-raised below
                last_error = error
                if attempt + 1 < self.attempts:
                    time.sleep(2**attempt)
        assert last_error is not None
        raise UniparseError(
            f"uniparse request to {self.endpoint} failed: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def parse_pdf(
        self, path: Path, *, skip_image_elements: bool = False
    ) -> UniparseDocument:
        content_type, body = encode_multipart(
            form_fields(skip_image_elements=skip_image_elements),
            file_field="file",
            filename=path.name,
            payload=path.read_bytes(),
        )
        raw = self._post(body, content_type)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise UniparseError(
                f"uniparse returned a non-JSON body for {path.name}"
            ) from error
        return UniparseDocument.model_validate(payload)


def save_images(
    document: UniparseDocument,
    paper_id: str,
    *,
    image_root: Path = IMAGE_ROOT,
    source_name: str | None = None,
) -> dict[str, dict[str, str | int]]:
    """Write the base64 PNGs returned by uniparse to disk under the paper id.

    Returns a mapping of the uniparse `img_path` to the stored asset's
    repository-relative path, byte size and sha256 digest.
    """
    if not document.images:
        return {}
    destination = image_root / paper_id
    if source_name:
        destination = destination / source_name
    destination.mkdir(parents=True, exist_ok=True)

    saved: dict[str, dict[str, str | int]] = {}
    for img_path, encoded in document.images.items():
        try:
            content = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            continue
        # img_path looks like "images/page_001_block_004_table_<hash>.png";
        # flatten it so nothing can escape the destination directory.
        target = destination / Path(img_path).name
        target.write_bytes(content)
        saved[img_path] = {
            "path": str(target.relative_to(ROOT))
            if target.is_relative_to(ROOT)
            else str(target),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    return saved
