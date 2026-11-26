"""
Automated test suite for Phase 2 – PDF Processing.
Tests PDF validation, page-by-page extraction, database persistence,
error handling (empty, corrupted, encrypted, scanned), and Flask endpoints.
Uses isolated temporary test directories and a temporary SQLite test database.
"""

import io
import shutil
import tempfile
from pathlib import Path

import pymupdf
import pytest

from app import create_app
from database.db import (
    delete_document,
    get_all_documents,
    get_document,
    get_document_pages,
    init_db,
    insert_document,
    insert_document_pages,
)
from utils.pdf_processor import (
    extract_text_from_pdf,
    get_pdf_metadata,
    validate_pdf,
)


@pytest.fixture
def test_env():
    """Sets up an isolated temporary workspace and database for testing."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    upload_folder = temp_path / "uploads"
    upload_folder.mkdir(parents=True, exist_ok=True)
    test_db_path = temp_path / "test_assistant.db"

    app = create_app("testing")
    app.config["UPLOAD_FOLDER"] = str(upload_folder)
    app.config["DATABASE_PATH"] = test_db_path
    app.config["TESTING"] = True

    # Initialize isolated test database
    init_db(test_db_path)

    client = app.test_client()

    yield {
        "app": app,
        "client": client,
        "temp_path": temp_path,
        "upload_folder": upload_folder,
        "db_path": test_db_path,
    }

    # Teardown
    shutil.rmtree(temp_dir, ignore_errors=True)


# ------------------------------------------------------------------------------
# Helper functions for generating test PDFs
# ------------------------------------------------------------------------------

def create_valid_pdf(pages_text: list[str]) -> bytes:
    """Creates a valid PDF document with the specified text on each page."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_scanned_pdf() -> bytes:
    """Creates a PDF with a page containing visual shapes but zero extractable text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(50, 50, 300, 300), fill=(0.7, 0.7, 0.7))
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_encrypted_pdf(password: str = "secure123") -> bytes:
    """Creates a password-protected PDF document."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Confidential Exam Revision Notes")
    pdf_bytes = doc.tobytes(
        encryption=getattr(pymupdf, "PDF_ENCRYPT_AES_256", 4),
        user_pw=password,
    )
    doc.close()
    return pdf_bytes


# ------------------------------------------------------------------------------
# Unit Tests: utils/pdf_processor.py
# ------------------------------------------------------------------------------

def test_validate_pdf_valid_file(test_env):
    """Test validation of a well-formed PDF document."""
    pdf_data = create_valid_pdf(["Hello world! This is valid text."])
    file_path = test_env["upload_folder"] / "valid.pdf"
    file_path.write_bytes(pdf_data)

    res = validate_pdf(file_path)
    assert res["valid"] is True
    assert res["error"] is None
    assert res["page_count"] == 1
    assert res["is_encrypted"] is False


def test_validate_pdf_empty_file(test_env):
    """Test rejection of a 0-byte file."""
    empty_file = test_env["upload_folder"] / "empty.pdf"
    empty_file.write_bytes(b"")

    res = validate_pdf(empty_file)
    assert res["valid"] is False
    assert "empty" in res["error"].lower()


def test_validate_pdf_corrupted_file(test_env):
    """Test rejection of corrupted PDF binary stream."""
    corrupted_file = test_env["upload_folder"] / "corrupt.pdf"
    corrupted_file.write_bytes(b"%PDF-1.4\n<<<invalid stream corrupted data>>>")

    res = validate_pdf(corrupted_file)
    assert res["valid"] is False
    assert "corrupted" in res["error"].lower() or "invalid" in res["error"].lower()


def test_validate_pdf_invalid_header(test_env):
    """Test rejection when binary header does not start with %PDF-."""
    fake_file = test_env["upload_folder"] / "fake.pdf"
    fake_file.write_bytes(b"Plain text masquerading as PDF")

    res = validate_pdf(fake_file)
    assert res["valid"] is False
    assert "%PDF-" in res["error"]


def test_validate_pdf_password_protected(test_env):
    """Test detection and rejection of password-protected PDFs."""
    enc_data = create_encrypted_pdf("mypassword")
    enc_file = test_env["upload_folder"] / "encrypted.pdf"
    enc_file.write_bytes(enc_data)

    res = validate_pdf(enc_file)
    assert res["valid"] is False
    assert "password" in res["error"].lower()
    assert res["is_encrypted"] is True


def test_multi_page_pdf_text_extraction(test_env):
    """Test accurate page count, word count, and text preservation for multi-page documents."""
    page_texts = [
        "First chapter covers basic algorithmic complexity and big-O notation.",
        "Second chapter investigates balanced binary search trees and heap structures.",
        "Third chapter explores graph traversal algorithms including Dijkstra and Floyd-Warshall.",
    ]
    pdf_data = create_valid_pdf(page_texts)
    pdf_file = test_env["upload_folder"] / "multi_page.pdf"
    pdf_file.write_bytes(pdf_data)

    result = extract_text_from_pdf(pdf_file)
    assert result["success"] is True
    assert result["total_pages"] == 3
    assert result["has_text"] is True
    assert result["is_scanned"] is False

    # Verify per-page extraction
    pages = result["pages"]
    assert len(pages) == 3
    for idx, expected_text in enumerate(page_texts):
        page_num = idx + 1
        assert pages[idx]["page_number"] == page_num
        assert expected_text in pages[idx]["text"]
        assert pages[idx]["word_count"] == len(expected_text.split())

    # Verify total word count calculation
    expected_total_words = sum(len(t.split()) for t in page_texts)
    assert result["total_words"] == expected_total_words


def test_scanned_pdf_detection(test_env):
    """Test that image-only / non-extractable PDFs are detected as scanned with zero words."""
    pdf_data = create_scanned_pdf()
    pdf_file = test_env["upload_folder"] / "scanned.pdf"
    pdf_file.write_bytes(pdf_data)

    result = extract_text_from_pdf(pdf_file)
    assert result["success"] is True
    assert result["total_pages"] == 1
    assert result["total_words"] == 0
    assert result["has_text"] is False
    assert result["is_scanned"] is True


def test_get_pdf_metadata(test_env):
    """Test metadata extraction helper."""
    pdf_data = create_valid_pdf(["Some sample content."])
    pdf_file = test_env["upload_folder"] / "meta_test.pdf"
    pdf_file.write_bytes(pdf_data)

    meta = get_pdf_metadata(pdf_file)
    assert meta["page_count"] == 1
    assert meta["file_size"] > 0


# ------------------------------------------------------------------------------
# Unit Tests: database/db.py
# ------------------------------------------------------------------------------

def test_database_crud_operations(test_env):
    """Test document and pages database insertion, retrieval, and cascade deletion."""
    db_path = test_env["db_path"]

    # 1. Insert document
    doc_id = insert_document(
        original_filename="calculus_notes.pdf",
        stored_filename="calc_hash_01.pdf",
        file_path=str(test_env["upload_folder"] / "calc_hash_01.pdf"),
        total_pages=2,
        total_words=150,
        processing_status="processed",
        db_path=db_path,
    )
    assert doc_id > 0

    # 2. Insert pages
    pages_to_insert = [
        {"page_number": 1, "text": "Page one derivatives."},
        {"page_number": 2, "text": "Page two integrals and areas."},
    ]
    insert_document_pages(doc_id, pages_to_insert, db_path=db_path)

    # 3. Retrieve document
    doc = get_document(doc_id, db_path=db_path)
    assert doc is not None
    assert doc["original_filename"] == "calculus_notes.pdf"
    assert doc["total_pages"] == 2
    assert doc["total_words"] == 150
    assert doc["processing_status"] == "processed"

    # 4. Retrieve pages
    pages = get_document_pages(doc_id, db_path=db_path)
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert pages[0]["extracted_text"] == "Page one derivatives."
    assert pages[1]["page_number"] == 2
    assert pages[1]["extracted_text"] == "Page two integrals and areas."

    # 5. Retrieve all documents
    all_docs = get_all_documents(db_path=db_path)
    assert len(all_docs) == 1

    # 6. Delete document
    del_path = delete_document(doc_id, db_path=db_path)
    assert del_path == str(test_env["upload_folder"] / "calc_hash_01.pdf")

    # Verify document and cascade pages are removed
    assert get_document(doc_id, db_path=db_path) is None
    assert get_document_pages(doc_id, db_path=db_path) == []


# ------------------------------------------------------------------------------
# Integration Tests: Flask Endpoints
# ------------------------------------------------------------------------------

def test_upload_valid_pdf_route_json(test_env):
    """Test POST /upload endpoint with valid PDF via AJAX/JSON request."""
    client = test_env["client"]
    pdf_bytes = create_valid_pdf(["Flask endpoint test page with several words."])

    data = {
        "file": (io.BytesIO(pdf_bytes), "lecture_one.pdf"),
    }
    response = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )

    assert response.status_code == 201
    json_data = response.get_json()
    assert json_data["success"] is True
    assert json_data["total_pages"] == 1
    assert json_data["total_words"] == 7
    assert json_data["is_scanned"] is False
    assert "redirect_url" in json_data

    # Verify document appears in database
    doc_id = json_data["document_id"]
    doc = get_document(doc_id, db_path=test_env["db_path"])
    assert doc is not None
    assert doc["original_filename"] == "lecture_one.pdf"


def test_upload_valid_pdf_route_redirect(test_env):
    """Test POST /upload standard form submission redirects to document details."""
    client = test_env["client"]
    pdf_bytes = create_valid_pdf(["Standard HTML form submission."])

    data = {
        "file": (io.BytesIO(pdf_bytes), "sample.pdf"),
    }
    response = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert "/document/" in response.headers["Location"]


def test_upload_scanned_pdf_route(test_env):
    """Test POST /upload with scanned image PDF sets status to scanned_no_text and provides OCR notice."""
    client = test_env["client"]
    pdf_bytes = create_scanned_pdf()

    data = {
        "file": (io.BytesIO(pdf_bytes), "scanned_doc.pdf"),
    }
    response = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )

    assert response.status_code == 201
    json_data = response.get_json()
    assert json_data["success"] is True
    assert json_data["is_scanned"] is True
    assert json_data["total_words"] == 0
    assert "OCR" in json_data["notice"]

    # Verify stored status in DB
    doc = get_document(json_data["document_id"], db_path=test_env["db_path"])
    assert doc is not None
    assert doc["processing_status"] == "scanned_no_text"


def test_upload_no_file(test_env):
    """Test POST /upload with missing file payload."""
    client = test_env["client"]
    response = client.post(
        "/upload",
        data={},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "no file" in json_data["error"].lower()


def test_upload_empty_filename(test_env):
    """Test POST /upload with empty filename."""
    client = test_env["client"]
    data = {
        "file": (io.BytesIO(b""), ""),
    }
    response = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False


def test_upload_invalid_extension(test_env):
    """Test POST /upload rejects non-PDF file formats."""
    client = test_env["client"]
    data = {
        "file": (io.BytesIO(b"Just plain text."), "notes.txt"),
    }
    response = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "unsupported" in json_data["error"].lower()


def test_upload_empty_pdf(test_env):
    """Test POST /upload rejects 0-byte PDF files."""
    client = test_env["client"]
    data = {
        "file": (io.BytesIO(b""), "empty.pdf"),
    }
    response = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "empty" in json_data["error"].lower()


def test_upload_corrupted_pdf(test_env):
    """Test POST /upload rejects corrupted PDF content."""
    client = test_env["client"]
    data = {
        "file": (io.BytesIO(b"%PDF-1.5 broken binary body"), "broken.pdf"),
    }
    response = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False


def test_upload_password_protected_pdf(test_env):
    """Test POST /upload rejects password-protected PDFs."""
    client = test_env["client"]
    pdf_bytes = create_encrypted_pdf("secret123")
    data = {
        "file": (io.BytesIO(pdf_bytes), "locked.pdf"),
    }
    response = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "password" in json_data["error"].lower()


def test_documents_list_page(test_env):
    """Test GET /documents renders table and stats."""
    client = test_env["client"]
    # Upload one doc first
    pdf_bytes = create_valid_pdf(["Document for listing test."])
    client.post(
        "/upload",
        data={"file": (io.BytesIO(pdf_bytes), "list_doc.pdf")},
        content_type="multipart/form-data",
    )

    response = client.get("/documents")
    assert response.status_code == 200
    assert b"Uploaded Study Materials" in response.data
    assert b"list_doc.pdf" in response.data


def test_document_details_page(test_env):
    """Test GET /document/<id> renders metadata and extracted page content."""
    client = test_env["client"]
    pdf_bytes = create_valid_pdf(["Specific test content for details verification."])
    up_res = client.post(
        "/upload",
        data={"file": (io.BytesIO(pdf_bytes), "details_target.pdf")},
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    doc_id = up_res.get_json()["document_id"]

    response = client.get(f"/document/{doc_id}")
    assert response.status_code == 200
    assert b"details_target.pdf" in response.data
    assert b"Specific test content for details verification." in response.data


def test_document_text_endpoint(test_env):
    """Test GET /document/<id>/text returns JSON and text viewer."""
    client = test_env["client"]
    pdf_bytes = create_valid_pdf(["Text endpoint inspection line."])
    up_res = client.post(
        "/upload",
        data={"file": (io.BytesIO(pdf_bytes), "text_endpoint.pdf")},
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    doc_id = up_res.get_json()["document_id"]

    # Test JSON output
    json_resp = client.get(f"/document/{doc_id}/text?format=json")
    assert json_resp.status_code == 200
    data = json_resp.get_json()
    assert data["success"] is True
    assert "Text endpoint inspection line." in data["full_text"]

    # Test HTML view
    html_resp = client.get(f"/document/{doc_id}/text")
    assert html_resp.status_code == 200
    assert b"Text endpoint inspection line." in html_resp.data


def test_document_deletion_route(test_env):
    """Test POST /document/<id>/delete removes document from database and file system."""
    client = test_env["client"]
    pdf_bytes = create_valid_pdf(["Content that will be deleted."])
    up_res = client.post(
        "/upload",
        data={"file": (io.BytesIO(pdf_bytes), "delete_me.pdf")},
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    doc_id = up_res.get_json()["document_id"]

    # Verify file exists on disk
    doc_record = get_document(doc_id, db_path=test_env["db_path"])
    assert doc_record is not None
    stored_path = Path(doc_record["file_path"])
    assert stored_path.exists()

    # Trigger deletion
    del_res = client.post(
        f"/document/{doc_id}/delete",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert del_res.status_code == 200
    assert del_res.get_json()["success"] is True

    # Verify removed from database
    assert get_document(doc_id, db_path=test_env["db_path"]) is None

    # Verify physical file is deleted from disk
    assert not stored_path.exists()


def test_document_details_not_found(test_env):
    """Test GET /document/<id> returns 404 for non-existent document ID."""
    client = test_env["client"]
    response = client.get("/document/999999", headers={"X-Requested-With": "XMLHttpRequest"})
    assert response.status_code == 404
    json_data = response.get_json()
    assert json_data["success"] is False


def test_upload_file_exceeding_max_size(test_env):
    """Test rejection when file upload exceeds configured MAX_CONTENT_LENGTH limit."""
    # Temporarily set MAX_CONTENT_LENGTH to 500 KB to test without allocating 17 MB in memory
    test_env["app"].config["MAX_CONTENT_LENGTH"] = 500 * 1024
    client = test_env["client"]

    oversized_data = b"%PDF-1.4\n" + (b"A" * (600 * 1024))
    data = {
        "file": (io.BytesIO(oversized_data), "too_large.pdf"),
    }
    response = client.post(
        "/upload",
        data=data,
        content_type="multipart/form-data",
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    assert response.status_code == 413
    json_data = response.get_json()
    assert json_data["success"] is False
    assert "exceeds" in json_data["error"].lower()

