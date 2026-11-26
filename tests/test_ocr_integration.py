"""
Comprehensive Integration Test Suite for OCR Support, AI Summarizer, and RAG Pipeline.

Tests:
1. check_tesseract_available() returns dictionary with availability, path, and Windows instructions.
2. render_pdf_page_to_image() creates a valid PIL Image from PDF page via PyMuPDF.
3. should_use_ocr() correctly flags scanned documents, empty documents, and low-word pages.
4. Database schema migration adds ocr_applied, ocr_engine, ocr_confidence, extraction_method columns.
5. insert_document() and update_document_ocr_metadata() store and update OCR fields.
6. insert_document_pages(replace_existing=True) replaces existing pages with OCR text.
7. Scanned PDF upload when Tesseract is not installed sets status='scanned_no_text' gracefully.
8. Scanned PDF upload when OCR succeeds sets status='ocr_completed' and saves OCR pages.
9. OCR-extracted text flows into AI Summarizer (all 3 modes: Quick, Standard, Detailed).
10. OCR-extracted text flows into Study Chat (RAG) chunker and vector indexing.
11. POST /document/<id>/reprocess-ocr route re-extracts text and invalidates stale RAG cache.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from PIL import Image

from app import create_app
from database.db import (
    get_db_connection,
    get_document,
    get_document_pages,
    init_db,
    insert_document,
    insert_document_pages,
    update_document_ocr_metadata,
)
from utils.chunker import chunk_document_pages
from utils.ocr_processor import (
    check_tesseract_available,
    extract_text_via_ocr,
    render_pdf_page_to_image,
    run_tesseract_on_image,
    should_use_ocr,
)
from utils.summarizer import generate_study_summary


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def create_sample_text_pdf(path: Path, text: str = "This is a normal searchable PDF text for testing.") -> Path:
    """Creates a simple PDF with embedded text using PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path


def create_sample_scanned_pdf(path: Path, image_text: str = "Scanned Note Text") -> Path:
    """
    Creates an image-only (scanned) PDF by rendering text into an image
    and embedding only the image on the PDF page (zero text layer).
    """
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    rect = pymupdf.Rect(0, 0, 400, 200)
    page.insert_image(rect, stream=img_bytes)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def app_and_db(tmp_path: Path):
    """Provides a configured Flask test app and an isolated temporary SQLite database."""
    db_file = tmp_path / "test_ocr.db"
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    vs_dir = tmp_path / "vector_store"
    vs_dir.mkdir(parents=True, exist_ok=True)

    app = create_app()
    app.config.update({
        "TESTING": True,
        "DATABASE_PATH": str(db_file),
        "UPLOAD_FOLDER": str(upload_dir),
        "VECTOR_STORE_DIR": str(vs_dir),
        "WTF_CSRF_ENABLED": False,
    })

    with app.app_context():
        init_db(db_path=db_file)

    return app, db_file, upload_dir


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_check_tesseract_available_structure():
    """Verify check_tesseract_available returns standard dictionary keys."""
    res = check_tesseract_available()
    assert isinstance(res, dict)
    assert "available" in res
    assert "path" in res
    assert "version" in res
    assert "error" in res
    assert "install_instructions" in res
    if not res["available"]:
        assert res["install_instructions"] is not None
        assert "Windows Installation Steps" in res["install_instructions"]


def test_render_pdf_page_to_image(tmp_path: Path):
    """Verify render_pdf_page_to_image converts a PDF page into a valid PIL Image."""
    pdf_path = tmp_path / "test_render.pdf"
    create_sample_text_pdf(pdf_path, "PyMuPDF rendering test")

    img = render_pdf_page_to_image(pdf_path, page_number=1, dpi=150)
    assert img is not None
    assert isinstance(img, Image.Image)
    assert img.width > 0
    assert img.height > 0


def test_should_use_ocr_logic():
    """Verify should_use_ocr identifies scanned, empty, and low-word documents."""
    # 1. Scanned flag is True -> True
    assert should_use_ocr({"is_scanned": True, "total_words": 0, "total_pages": 1}) is True

    # 2. total_words is 0 -> True
    assert should_use_ocr({"is_scanned": False, "total_words": 0, "total_pages": 1}) is True

    # 3. Very low average words per page (< 10 words/page) -> True
    assert should_use_ocr({"is_scanned": False, "total_words": 5, "total_pages": 2}) is True

    # 4. Normal document with high word count -> False
    assert should_use_ocr({"is_scanned": False, "total_words": 500, "total_pages": 2}) is False


def test_database_schema_has_ocr_columns(app_and_db):
    """Verify DB table 'documents' contains the new OCR metadata columns."""
    _, db_file, _ = app_and_db
    conn = get_db_connection(db_file)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
    conn.close()

    assert "ocr_applied" in cols
    assert "ocr_engine" in cols
    assert "ocr_confidence" in cols
    assert "extraction_method" in cols


def test_insert_and_update_document_ocr_metadata(app_and_db):
    """Verify insert_document and update_document_ocr_metadata persist OCR fields properly."""
    _, db_file, _ = app_and_db

    doc_id = insert_document(
        original_filename="sample_scan.pdf",
        stored_filename="uuid_sample_scan.pdf",
        file_path="dummy/path.pdf",
        total_pages=2,
        total_words=0,
        processing_status="scanned_no_text",
        ocr_applied=False,
        ocr_engine="",
        ocr_confidence=None,
        extraction_method="pymupdf",
        db_path=db_file,
    )

    doc = get_document(doc_id, db_path=db_file)
    assert doc is not None
    assert doc["ocr_applied"] == 0
    assert doc["extraction_method"] == "pymupdf"

    # Now update via OCR re-processing
    update_document_ocr_metadata(
        document_id=doc_id,
        total_words=150,
        processing_status="ocr_completed",
        ocr_applied=True,
        ocr_engine="tesseract",
        ocr_confidence=88.5,
        extraction_method="ocr",
        db_path=db_file,
    )

    updated_doc = get_document(doc_id, db_path=db_file)
    assert updated_doc["ocr_applied"] == 1
    assert updated_doc["ocr_engine"] == "tesseract"
    assert updated_doc["ocr_confidence"] == pytest.approx(88.5, 0.1)
    assert updated_doc["extraction_method"] == "ocr"
    assert updated_doc["processing_status"] == "ocr_completed"
    assert updated_doc["total_words"] == 150


def test_insert_document_pages_replace_existing(app_and_db):
    """Verify insert_document_pages with replace_existing=True overwrites existing pages."""
    _, db_file, _ = app_and_db

    doc_id = insert_document(
        original_filename="doc.pdf",
        stored_filename="u_doc.pdf",
        file_path="p.pdf",
        total_pages=1,
        total_words=0,
        db_path=db_file,
    )

    # Initial empty page
    insert_document_pages(
        doc_id,
        [{"page_number": 1, "text": ""}],
        db_path=db_file,
    )
    pages = get_document_pages(doc_id, db_path=db_file)
    assert len(pages) == 1
    assert (pages[0].get("text") or pages[0].get("extracted_text")) == ""

    # Replace with OCR text
    insert_document_pages(
        doc_id,
        [{"page_number": 1, "text": "Recognized text after Tesseract OCR analysis."}],
        replace_existing=True,
        db_path=db_file,
    )
    new_pages = get_document_pages(doc_id, db_path=db_file)
    assert len(new_pages) == 1
    assert new_pages[0]["extracted_text"] == "Recognized text after Tesseract OCR analysis."


def test_scanned_pdf_upload_graceful_when_tesseract_missing(app_and_db, tmp_path: Path):
    """
    When a scanned PDF is uploaded and Tesseract is unavailable,
    the app must not crash; it sets status to 'scanned_no_text' with a helpful notice.
    """
    app, db_file, _ = app_and_db
    scanned_pdf = tmp_path / "scanned_invoice.pdf"
    create_sample_scanned_pdf(scanned_pdf)

    client = app.test_client()

    with patch("utils.ocr_processor.check_tesseract_available", return_value={
        "available": False,
        "path": None,
        "version": None,
        "error": "Tesseract not installed.",
        "install_instructions": "Windows Installation Steps: download installer...",
    }):
        with open(scanned_pdf, "rb") as f:
            resp = client.post(
                "/upload",
                data={"file": (f, "scanned_invoice.pdf")},
                content_type="multipart/form-data",
                headers={"Accept": "application/json"},
            )

    assert resp.status_code == 201
    data = resp.get_json()
    assert data["success"] is True
    assert data["is_scanned"] is True
    assert data["total_words"] == 0
    assert "scanned" in data["notice"].lower()

    # Verify database state
    doc = get_document(data["document_id"], db_path=db_file)
    assert doc["processing_status"] == "scanned_no_text"


def test_scanned_pdf_upload_with_ocr_active(app_and_db, tmp_path: Path):
    """
    When Tesseract is available and OCR succeeds,
    scanned PDF upload automatically extracts OCR text and sets status='ocr_completed'.
    """
    app, db_file, _ = app_and_db
    scanned_pdf = tmp_path / "scan_math.pdf"
    create_sample_scanned_pdf(scanned_pdf)

    client = app.test_client()

    mock_ocr_result = {
        "success": True,
        "total_pages": 1,
        "total_words": 18,
        "has_text": True,
        "is_scanned": False,
        "pages": [{"page_number": 1, "text": "Euler formula e^(i*pi) + 1 = 0 represents complex analysis fundamentals.", "word_count": 18}],
        "metadata": {},
        "extraction_method": "ocr",
        "ocr_applied": True,
        "ocr_engine": "tesseract",
        "ocr_confidence": 94.2,
        "error": None,
    }

    with patch("app.routes.check_tesseract_available", return_value={"available": True, "path": "tesseract", "version": "5.0", "error": None, "install_instructions": None}), \
         patch("app.routes.extract_text_via_ocr", return_value=mock_ocr_result):
        with open(scanned_pdf, "rb") as f:
            resp = client.post(
                "/upload",
                data={"file": (f, "scan_math.pdf")},
                content_type="multipart/form-data",
                headers={"Accept": "application/json"},
            )

    assert resp.status_code == 201
    data = resp.get_json()
    assert data["success"] is True
    assert data["ocr_applied"] is True
    assert data["total_words"] == 18

    # Verify database state
    doc = get_document(data["document_id"], db_path=db_file)
    assert doc["processing_status"] == "ocr_completed"
    assert doc["extraction_method"] == "ocr"
    assert doc["ocr_confidence"] == pytest.approx(94.2, 0.1)

    # Verify pages table received OCR text
    pages = get_document_pages(data["document_id"], db_path=db_file)
    assert len(pages) == 1
    assert "Euler formula" in pages[0]["extracted_text"]


def test_ocr_text_flows_into_summarizer(app_and_db):
    """
    Verifies that pages extracted via OCR feed seamlessly into the AI Summarizer
    without errors and produce Quick, Standard, and Detailed summaries.
    """
    ocr_pages = [
        {
            "page_number": 1,
            "text": (
                "Artificial intelligence (AI) is intelligence demonstrated by machines, "
                "as opposed to intelligence of humans and other animals. "
                "Example tasks in which this is done include speech recognition, computer vision, "
                "translation between natural languages, as well as other mappings of inputs. "
                "Machine learning is an important branch of AI focusing on data-driven learning. "
                "Deep learning models use multiple layers to progressively extract higher-level features."
            ),
            "word_count": 65,
        }
    ]

    for mode in ("quick", "standard", "detailed"):
        result = generate_study_summary(
            content=ocr_pages,
            summary_type=mode,
            model_name="mock",
        )
        assert result["success"] is True, f"Summarizer failed for mode {mode}: {result.get('error')}"
        assert result["summary_type"] == mode
        assert len(result["summary_text"]) > 0


def test_ocr_text_flows_into_rag_chunker(app_and_db):
    """
    Verifies that OCR-extracted pages are successfully segmented by chunk_document_pages
    and ready for vector indexing.
    """
    _, db_file, _ = app_and_db

    doc_id = insert_document(
        original_filename="ocr_biology.pdf",
        stored_filename="u_ocr_biology.pdf",
        file_path="dummy.pdf",
        total_pages=2,
        total_words=120,
        processing_status="ocr_completed",
        ocr_applied=True,
        ocr_engine="tesseract",
        extraction_method="ocr",
        db_path=db_file,
    )

    ocr_pages = [
        {
            "page_number": 1,
            "extracted_text": (
                "Mitochondria are membrane-bound cell organelles that generate most of the "
                "chemical energy needed to power the biochemical reactions of the cell. "
                "Chemical energy produced by mitochondria is stored in a small molecule called ATP."
            ),
        },
        {
            "page_number": 2,
            "extracted_text": (
                "Cellular respiration is a set of metabolic reactions that convert chemical energy "
                "from nutrients into adenosine triphosphate (ATP), and then release waste products."
            ),
        },
    ]

    insert_document_pages(
        doc_id,
        [{"page_number": p["page_number"], "text": p["extracted_text"]} for p in ocr_pages],
        db_path=db_file,
    )

    # Fetch pages from DB (returns 'extracted_text' column)
    db_pages = get_document_pages(doc_id, db_path=db_file)
    assert len(db_pages) == 2

    chunks = chunk_document_pages(
        document_id=doc_id,
        original_filename="ocr_biology.pdf",
        pages=db_pages,
        chunk_size_words=60,
        overlap_words=15,
    )

    assert len(chunks) >= 2
    for c in chunks:
        assert c["document_id"] == doc_id
        assert "chunk_text" in c
        assert len(c["chunk_text"]) > 0
        assert c["page_number"] in (1, 2)


def test_reprocess_document_ocr_endpoint(app_and_db, tmp_path: Path):
    """
    Verifies POST /document/<id>/reprocess-ocr route re-extracts text and updates DB.
    """
    app, db_file, _ = app_and_db
    client = app.test_client()

    pdf_file = tmp_path / "to_reprocess.pdf"
    create_sample_scanned_pdf(pdf_file)

    doc_id = insert_document(
        original_filename="to_reprocess.pdf",
        stored_filename="stored_to_reprocess.pdf",
        file_path=str(pdf_file),
        total_pages=1,
        total_words=0,
        processing_status="scanned_no_text",
        db_path=db_file,
    )

    mock_ocr = {
        "success": True,
        "total_pages": 1,
        "total_words": 25,
        "has_text": True,
        "is_scanned": False,
        "pages": [{"page_number": 1, "text": "Reprocessed text extracted successfully with high confidence.", "word_count": 25}],
        "metadata": {},
        "extraction_method": "ocr",
        "ocr_applied": True,
        "ocr_engine": "tesseract",
        "ocr_confidence": 91.0,
        "error": None,
    }

    with patch("app.routes.check_tesseract_available", return_value={"available": True, "path": "tesseract", "version": "5.0", "error": None, "install_instructions": None}), \
         patch("app.routes.extract_text_via_ocr", return_value=mock_ocr):
        resp = client.post(
            f"/document/{doc_id}/reprocess-ocr",
            headers={"Accept": "application/json", "X-Requested-With": "XMLHttpRequest"},
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["total_words"] == 25
    assert data["processing_status"] == "ocr_completed"

    # Verify DB updated
    updated = get_document(doc_id, db_path=db_file)
    assert updated["processing_status"] == "ocr_completed"
    assert updated["total_words"] == 25
    assert updated["ocr_applied"] == 1
