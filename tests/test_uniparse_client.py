import base64
import hashlib
import json

import pytest

from src.rag.uniparse_client import (
    DEFAULT_BASE_URL,
    FORBIDDEN_FORM_FIELDS,
    PARSE_PATH,
    UniparseClient,
    UniparseDocument,
    UniparseError,
    base_url_from_env,
    encode_multipart,
    form_fields,
    save_images,
)


PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake-image-bytes").decode()


def _document() -> dict:
    return {
        "meta": {"filesize": 42, "version": "1.1.0"},
        "pages": [
            {
                "page_id": 0,
                "page_width": 612.0,
                "page_height": 792.0,
                "content": [
                    {
                        "type": "heading",
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                        "text": "Supplemental Material",
                        "text_format": "markdown",
                        "heading_level": 1,
                    }
                ],
                "others": [{"type": "page_number", "text": "1"}],
            }
        ],
        "images": {"images/page_001_block_004_table_abc.png": PNG},
        "markdown": "# Supplemental Material",
    }


def test_form_fields_never_send_the_heading_level_options():
    """The deployed container 500s on these: its heading LLM base_url is an
    unsubstituted "http://<llm-server-openai>:<port>/v1/" placeholder."""
    fields = form_fields()
    assert set(fields) & FORBIDDEN_FORM_FIELDS == set()
    assert "heading_level_enabled" not in fields
    assert "heading_level_method" not in fields


def test_skip_image_elements_is_sent_as_false_by_default():
    assert form_fields()["skip_image_elements"] == "false"
    assert form_fields(skip_image_elements=True)["skip_image_elements"] == "true"


def test_encoded_request_body_omits_the_heading_level_options():
    content_type, body = encode_multipart(
        form_fields(), file_field="file", filename="mmc1.pdf", payload=b"%PDF-1.4"
    )
    assert content_type.startswith("multipart/form-data; boundary=")
    for field in FORBIDDEN_FORM_FIELDS:
        assert field.encode() not in body
    assert b'name="skip_image_elements"' in body
    assert b'filename="mmc1.pdf"' in body
    assert b"%PDF-1.4" in body


def test_base_url_prefers_the_environment_variable(monkeypatch):
    monkeypatch.delenv("UNIPARSE_BASE_URL", raising=False)
    assert base_url_from_env() == DEFAULT_BASE_URL
    monkeypatch.setenv("UNIPARSE_BASE_URL", "http://uniparse.internal:9000/")
    assert base_url_from_env() == "http://uniparse.internal:9000"
    assert base_url_from_env("http://explicit:1/") == "http://explicit:1"


def test_client_endpoint_is_the_documented_parse_route(monkeypatch):
    monkeypatch.delenv("UNIPARSE_BASE_URL", raising=False)
    assert UniparseClient().endpoint == DEFAULT_BASE_URL + PARSE_PATH


def test_document_model_parses_the_recorded_response_shape():
    document = UniparseDocument.model_validate(_document())
    assert document.parser_name == "uniparse-1.1.0"
    page = document.pages[0]
    assert page.page_id == 0
    assert page.content[0].type == "heading"
    assert page.content[0].heading_level == 1
    assert page.others[0].type == "page_number"


def test_document_model_tolerates_unknown_server_fields():
    payload = _document()
    payload["pages"][0]["content"][0]["future_field"] = "value"
    payload["unexpected_top_level"] = 1
    document = UniparseDocument.model_validate(payload)
    assert document.pages[0].content[0].type == "heading"


def test_parse_pdf_posts_multipart_and_returns_a_document(tmp_path, monkeypatch):
    pdf = tmp_path / "mmc1.pdf"
    pdf.write_bytes(b"%PDF-1.4 body")
    client = UniparseClient("http://uniparse.test:8020")
    captured: dict[str, object] = {}

    def fake_post(body: bytes, content_type: str) -> bytes:
        captured["body"] = body
        captured["content_type"] = content_type
        return json.dumps(_document()).encode()

    monkeypatch.setattr(client, "_post", fake_post)
    document = client.parse_pdf(pdf)

    assert document.parser_name == "uniparse-1.1.0"
    assert b"%PDF-1.4 body" in captured["body"]
    for field in FORBIDDEN_FORM_FIELDS:
        assert field.encode() not in captured["body"]


def test_parse_pdf_raises_uniparse_error_on_a_non_json_body(tmp_path, monkeypatch):
    pdf = tmp_path / "mmc1.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    client = UniparseClient("http://uniparse.test:8020")
    monkeypatch.setattr(client, "_post", lambda body, ct: b"<html>500</html>")
    with pytest.raises(UniparseError):
        client.parse_pdf(pdf)


def test_post_retries_then_raises_uniparse_error(tmp_path, monkeypatch):
    client = UniparseClient("http://uniparse.test:8020", attempts=3)
    calls = []

    class Boom(Exception):
        pass

    def fake_open(request, timeout):
        calls.append(request)
        raise Boom("connection refused")

    monkeypatch.setattr(client._opener, "open", fake_open)
    monkeypatch.setattr("src.rag.uniparse_client.time.sleep", lambda seconds: None)
    with pytest.raises(UniparseError):
        client._post(b"body", "multipart/form-data; boundary=x")
    assert len(calls) == 3


def test_save_images_writes_png_files_with_digests(tmp_path):
    document = UniparseDocument.model_validate(_document())
    saved = save_images(document, "GP-006", image_root=tmp_path)
    key = "images/page_001_block_004_table_abc.png"
    assert set(saved) == {key}
    written = tmp_path / "GP-006" / "page_001_block_004_table_abc.png"
    assert written.exists()
    content = written.read_bytes()
    assert saved[key]["bytes"] == len(content)
    assert saved[key]["sha256"] == hashlib.sha256(content).hexdigest()


def test_save_images_flattens_paths_and_skips_undecodable_entries(tmp_path):
    payload = _document()
    payload["images"] = {
        "images/../../escape.png": PNG,
        "images/broken.png": "not-base64!!",
    }
    saved = save_images(UniparseDocument.model_validate(payload), "GP-006", image_root=tmp_path)
    assert set(saved) == {"images/../../escape.png"}
    assert (tmp_path / "GP-006" / "escape.png").exists()
    assert not (tmp_path / "escape.png").exists()
