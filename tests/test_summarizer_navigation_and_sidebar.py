"""
Test Suite: AI Summarizer Access & Navigation and Study Chat Navigation Removal.

Tests:
1. test_sidebar_visible_navigation_items:
   - Verifies the 8 required sidebar items:
     MAIN WORKSPACE: Study Desk, My Documents, AI Summarizer, Quiz Generator, Learning Map
     ACADEMIC PROGRESS: Progress & Analytics, Study Goals, Settings
   - Verifies 'Study Chat (RAG)' is NOT present in the visible sidebar.
2. test_dashboard_no_visible_study_chat_links:
   - Verifies that 'Open Chat ->' and 'Study Chat' pills are removed from visible dashboard links.
3. test_chat_backend_routes_preserved:
   - Verifies that /chat and chatbot routes remain functional in backend (not deleted).
4. test_get_summarizer_hub_no_doc_id_required:
   - Verifies GET /summarizer returns 200 OK without requiring doc_id in URL.
5. test_get_summarizer_hub_with_doc_id_preselect:
   - Verifies GET /summarizer?doc_id={id} pre-selects the requested document.
6. test_post_summarizer_direct:
   - Verifies POST /summarizer with document_id and summary_type generates summary.
7. test_post_document_summarize_existing_endpoint:
   - Verifies POST /document/{doc_id}/summarize continues to work identically.
8. test_summarizer_scanned_pdf_zero_words_ocr_notice:
   - Verifies scanned PDF with 0 words displays/returns OCR requirement error, no fake summary.
9. test_summarizer_modes_quick_standard_detailed:
   - Verifies all 3 summary modes (Quick, Standard, Detailed) produce structured output.
10. test_all_sidebar_routes_accessible_no_build_errors:
   - Iterates through all sidebar links, asserting status_code 200 and no BuildError.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app import create_app
from database.db import (
    get_summary,
    init_db,
    insert_document,
    insert_document_pages,
)
from utils.summarizer import SummarizerEngine


@pytest.fixture
def app_and_client(tmp_path: Path):
    """Sets up an isolated test application and database."""
    db_file = tmp_path / "test_summarizer_nav.db"
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    app = create_app()
    app.config.update({
        "TESTING": True,
        "DATABASE_PATH": str(db_file),
        "UPLOAD_FOLDER": str(upload_dir),
        "SUMMARIZATION_MODEL": "mock",
        "WTF_CSRF_ENABLED": False,
    })

    with app.app_context():
        init_db(db_path=db_file)

    client = app.test_client()
    return app, client, db_file


def _seed_sample_doc(db_file: Path, title: str = "Operating Systems Principles.pdf", words: int = 250) -> int:
    """Helper to insert a sample readable document."""
    doc_id = insert_document(
        original_filename=title,
        stored_filename=f"uuid_{title}",
        file_path="dummy/path.pdf",
        total_pages=2,
        total_words=words,
        processing_status="processed",
        db_path=db_file,
    )
    insert_document_pages(
        doc_id,
        [
            {
                "page_number": 1,
                "text": (
                    "Operating systems manage computer hardware resources and provide common services for programs. "
                    "Process management is a core responsibility that involves process scheduling and context switching. "
                    "Virtual memory provides an idealized abstraction of the storage resources that are actually available. "
                    "Paging is a memory management scheme by which a computer stores and retrieves data from secondary storage."
                ),
            },
            {
                "page_number": 2,
                "text": (
                    "A deadlock is a state in which each member of a group is waiting for another member to take action. "
                    "Mutual exclusion, hold and wait, no preemption, and circular wait are four necessary conditions for deadlock. "
                    "File systems control how data is stored and retrieved, separating information into individual files."
                ),
            },
        ],
        db_path=db_file,
    )
    return doc_id


def _seed_scanned_doc(db_file: Path, title: str = "empty_scan.pdf") -> int:
    """Helper to insert an image-only / zero-words scanned document."""
    doc_id = insert_document(
        original_filename=title,
        stored_filename=f"uuid_{title}",
        file_path="dummy/scanned.pdf",
        total_pages=1,
        total_words=0,
        processing_status="scanned_no_text",
        db_path=db_file,
    )
    insert_document_pages(
        doc_id,
        [{"page_number": 1, "text": ""}],
        db_path=db_file,
    )
    return doc_id


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_sidebar_visible_navigation_items(app_and_client):
    """
    Task 4 & Task 3:
    Verify the final visible sidebar navigation order and items:
    - Study Desk
    - My Documents
    - AI Summarizer
    - Quiz Generator
    - Learning Map
    - Progress & Analytics
    - Study Goals
    - Settings
    And verify Study Chat (RAG) is completely removed from the visible sidebar.
    """
    _, client, _ = app_and_client
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    required_items = [
        "Study Desk",
        "My Documents",
        "AI Summarizer",
        "Quiz Generator",
        "Learning Map",
        "Progress & Analytics",
        "Study Goals",
        "Settings",
    ]

    for item in required_items:
        assert f"<span>{item}</span>" in html, f"Missing visible sidebar item: {item}"

    # Verify Study Chat is removed from the visible sidebar
    assert "Study Chat (RAG)" not in html
    assert "main.chat_hub_redirect" not in html


def test_dashboard_no_visible_study_chat_links(app_and_client):
    """
    Task 3:
    Verify Study Chat navigation is removed from dashboard cards/links,
    while AI Summarizer, Quiz Generator, and Learning Map are present.
    """
    _, client, _ = app_and_client
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # Verify no visible "Open Chat ->" or "💬 Study Chat" navigation buttons
    assert "Open Chat →" not in html
    assert "💬 Study Chat" not in html

    # Verify AI Study tools navigation exists
    assert "Open Summarizer →" in html or "AI Summarizer" in html
    assert "Quiz Generator" in html


def test_chat_backend_routes_preserved(app_and_client, tmp_path):
    """
    Task 3 constraint:
    Verify that removing Study Chat from visible navigation does NOT delete
    the backend chatbot route (/chat) or cause errors.
    """
    _, client, db_file = app_and_client
    doc_id = _seed_sample_doc(db_file)

    # /chat redirects to document_chat or documents list as designed
    resp = client.get("/chat", follow_redirects=False)
    assert resp.status_code in (302, 200)

    # /document/<id>/chat responds
    resp_doc_chat = client.get(f"/document/{doc_id}/chat")
    assert resp_doc_chat.status_code == 200


def test_get_summarizer_hub_no_doc_id_required(app_and_client):
    """
    Task 1:
    Clicking AI Summarizer in the sidebar opens /summarizer without requiring a document ID in URL.
    Does not produce a 404 error or Flask BuildError.
    """
    _, client, db_file = app_and_client
    _seed_sample_doc(db_file, "Algorithms.pdf")

    resp = client.get("/summarizer")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # Check page elements
    assert "AI Study Summarizer" in html
    assert "Algorithms.pdf" in html
    assert "Summarization Workspace" in html
    assert 'id="summarization-section"' in html
    assert 'name="summary_type_select"' in html


def test_get_summarizer_hub_empty_library(app_and_client):
    """
    Task 1:
    When no documents exist, /summarizer renders cleanly without crashing
    and shows an upload prompt.
    """
    _, client, _ = app_and_client
    resp = client.get("/summarizer")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    assert "No Documents in Library" in html
    assert "Upload Your First PDF" in html


def test_get_summarizer_hub_with_doc_id_preselect(app_and_client):
    """
    Task 1 & Task 2:
    GET /summarizer?doc_id={id} pre-selects the specified document.
    """
    _, client, db_file = app_and_client
    doc1 = _seed_sample_doc(db_file, "Document_One.pdf")
    doc2 = _seed_sample_doc(db_file, "Document_Two.pdf")

    resp = client.get(f"/summarizer?doc_id={doc2}")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    assert f'data-doc-id="{doc2}"' in html
    assert "Document_Two.pdf" in html


def test_post_summarizer_direct(app_and_client):
    """
    Task 1:
    POST /summarizer generates summary for the specified document and returns JSON.
    """
    _, client, db_file = app_and_client
    doc_id = _seed_sample_doc(db_file)

    resp = client.post(
        "/summarizer",
        json={"document_id": doc_id, "summary_type": "standard"},
        headers={"Accept": "application/json", "X-Requested-With": "XMLHttpRequest"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["summary_type"] == "standard"
    assert len(data["summary_text"]) > 0
    assert "stats" in data

    # Verify summary was saved in SQLite
    saved = get_summary(doc_id, summary_type="standard", db_path=db_file)
    assert saved is not None
    assert saved["summary_type"] == "standard"


def test_post_document_summarize_existing_endpoint(app_and_client):
    """
    Task 2:
    POST /document/<id>/summarize continues to work identically for individual document page.
    """
    _, client, db_file = app_and_client
    doc_id = _seed_sample_doc(db_file)

    resp = client.post(
        f"/document/{doc_id}/summarize",
        json={"summary_type": "quick"},
        headers={"Accept": "application/json", "X-Requested-With": "XMLHttpRequest"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["summary_type"] == "quick"


def test_summarizer_scanned_pdf_zero_words_ocr_notice(app_and_client):
    """
    Task 2:
    If a scanned PDF has zero extracted words, return an OCR requirement message
    instead of attempting to generate a fake summary.
    """
    _, client, db_file = app_and_client
    scanned_id = _seed_scanned_doc(db_file, "empty_scan.pdf")

    # Via direct /summarizer endpoint
    resp = client.post(
        "/summarizer",
        json={"document_id": scanned_id, "summary_type": "standard"},
        headers={"Accept": "application/json", "X-Requested-With": "XMLHttpRequest"},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "OCR" in data["error"]
    assert data.get("requires_ocr") is True

    # Via /document/<id>/summarize endpoint
    resp2 = client.post(
        f"/document/{scanned_id}/summarize",
        json={"summary_type": "detailed"},
        headers={"Accept": "application/json", "X-Requested-With": "XMLHttpRequest"},
    )
    assert resp2.status_code == 400
    data2 = resp2.get_json()
    assert data2["success"] is False
    assert "OCR" in data2["error"]

    # Also verify that on GET /summarizer?doc_id={scanned_id}, the page shows OCR warning
    resp_page = client.get(f"/summarizer?doc_id={scanned_id}")
    assert resp_page.status_code == 200
    page_html = resp_page.get_data(as_text=True)
    assert "OCR Required" in page_html


def test_summarizer_modes_quick_standard_detailed(app_and_client):
    """
    Task 1:
    Verify that Quick, Standard, and Detailed summary modes are supported
    and produce valid responses.
    """
    _, client, db_file = app_and_client
    doc_id = _seed_sample_doc(db_file)

    for mode in ("quick", "standard", "detailed"):
        resp = client.post(
            f"/document/{doc_id}/summarize",
            json={"summary_type": mode},
            headers={"Accept": "application/json", "X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200, f"Mode {mode} failed: {resp.data}"
        data = resp.get_json()
        assert data["success"] is True
        assert data["summary_type"] == mode
        assert len(data["summary_text"]) > 0


def test_all_sidebar_routes_accessible_no_build_errors(app_and_client):
    """
    Task 5:
    Test all sidebar navigation links across the application:
    1. Study Desk (/)
    2. My Documents (/documents)
    3. AI Summarizer (/summarizer)
    4. Quiz Generator (/quizzes)
    5. Learning Map (/learning-map)
    6. Progress & Analytics (/analytics)
    7. Study Goals (/goals)
    8. Settings (/settings)
    Assert all return HTTP 200 with zero Flask BuildErrors or 404s.
    """
    _, client, db_file = app_and_client
    _seed_sample_doc(db_file)

    routes = [
        ("/", "Study Desk"),
        ("/dashboard", "Study Desk (Dashboard)"),
        ("/documents", "My Documents"),
        ("/summarizer", "AI Summarizer"),
        ("/quizzes", "Quiz Generator"),
        ("/learning-map", "Learning Map"),
        ("/analytics", "Progress & Analytics"),
        ("/goals", "Study Goals"),
        ("/settings", "Settings"),
    ]

    for path, label in routes:
        resp = client.get(path)
        assert resp.status_code == 200, f"Route {path} ({label}) returned status {resp.status_code}"
        html = resp.get_data(as_text=True)
        # Ensure template rendered successfully
        assert "<html" in html
        assert "SmartStudy.ai" in html
